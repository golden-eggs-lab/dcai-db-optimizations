import collections
import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
try:
    from sklearn.neighbors import DistanceMetric
except ImportError:
    # Fallback for very old sklearn versions
    from sklearn.neighbors.dist_metrics import DistanceMetric
from torch import nn
from torch.nn.functional import normalize
from tqdm import tqdm

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available. Install it with: pip install faiss-cpu or pip install faiss-gpu")
    print("Falling back to exact KNN search.")

sys.path.append("../../")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from utilities.data_loader import get_glue_tensor_dataset
from utilities.preprocessors import processors
from utilities.trainers import my_evaluate

# from acquisition.bertscorer import calculate_bertscore


logger = logging.getLogger(__name__)

# Global cache for FAISS index to enable incremental updates across iterations
_faiss_index_cache = {
    'index': None,
    'labeled_inds': None,
    'embeddings': None,
    'dimension': None,
    'index_type': None,
    'index_to_labeled_map': None  # Maps index position to labeled_inds position
}


def contrastive_acquisition(args, annotations_per_iteration, X_original, y_original,
                            labeled_inds, candidate_inds, discarded_inds, original_inds,
                            tokenizer,
                            train_results,
                            results_dpool, logits_dpool, bert_representations=None,
                            train_dataset=None,
                            model=None,
                            tfidf_dtrain_reprs=None, tfidf_dpool_reprs=None,
                            iteration=1):
    """

    :param args: arguments (such as flags, device, etc)
    :param annotations_per_iteration: acquisition size
    :param X_original: list of all data
    :param y_original: list of all labels
    :param labeled_inds: indices of current labeled/training examples
    :param candidate_inds: indices of current unlabeled examples (pool)
    :param discarded_inds: indices of examples that should not be considered for acquisition/annotation
    :param original_inds: indices of all data (this is a list of indices of the X_original list)
    :param tokenizer: tokenizer
    :param train_results: dictionary with results from training/validation phase (for logits) of training set
    :param results_dpool: dictionary with results from training/validation phase (for logits) of unlabeled set (pool)
    :param logits_dpool: logits for all examples in the pool
    :param bert_representations: representations of pretrained bert (ablation)
    :param train_dataset: the training set in the tensor format
    :param model: the fine-tuned model of the iteration
    :param tfidf_dtrain_reprs: tf-idf representations of training set (ablation)
    :param tfidf_dpool_reprs: tf-idf representations of unlabeled set (ablation)
    :return:
    """
    """
    CAL (Contrastive Active Learning)
    Acquire data by choosing those with the largest KL divergence in the predictions between a candidate dpool input
     and its nearest neighbours in the training set.
     Our proposed approach includes:
     args.cls = True
     args.operator = "mean"
     the rest are False. We use them (True) in some experiments for ablation/analysis
     args.mean_emb = False
     args.mean_out = False
     args.bert_score = False 
     args.tfidf = False 
     args.reverse = False
     args.knn_lab = False
     args.ce = False
    :return:
    """
    processor = processors[args.task_name]()
    if model is None and train_results is not None:
        model = train_results['model']

    if args.bert_score:  # BERT score representations (ablation)
        train_dataset = get_glue_tensor_dataset(labeled_inds, args, args.task_name, tokenizer, train=True)
        _train_results, train_logits = my_evaluate(train_dataset, args, model, prefix="",
                                                   al_test=False, mc_samples=None,
                                                   return_mean_embs=args.mean_embs,
                                                   return_mean_output=args.mean_out,
                                                   return_cls=args.cls
                                                   )
        criterion = nn.KLDivLoss(reduction='none') if not args.ce else nn.CrossEntropyLoss()
        bert_score_matrix, bs_calc_time = calculate_bertscore(args, X_original, original_inds)
        assert bert_score_matrix.shape[0] == len(original_inds), 'bs {}, ori'.format(bert_score_matrix.shape[0],
                                                                                     len(original_inds))
        kl_scores = []
        distances = []
        pairs = []
        dist = DistanceMetric.get_metric('euclidean')
        num_adv = None
        for unlab_i, zipped in enumerate(zip(candidate_inds, logits_dpool)):
            candidate, unlab_logit = zipped
            all_similarities = bert_score_matrix[candidate]
            labeled_data_similarities = all_similarities[labeled_inds]
            labeled_neighborhood_inds = np.argpartition(labeled_data_similarities, -args.num_nei)[-args.num_nei:]
            distances_ = labeled_data_similarities[labeled_neighborhood_inds]
            distances.append(distances_)
            labeled_neighbours_labels = train_dataset.tensors[3][labeled_neighborhood_inds]
            neigh_prob = F.softmax(train_logits[labeled_neighborhood_inds], dim=-1)

            if args.ce:
                kl = np.array([criterion(unlab_logit.view(-1, args.num_classes), label.view(-1)) for label in
                               labeled_neighbours_labels])
            else:
                uda_softmax_temp = 1
                candidate_log_prob = F.log_softmax(unlab_logit / uda_softmax_temp, dim=-1)
                kl = np.array([torch.sum(criterion(candidate_log_prob, n), dim=-1).numpy() for n in neigh_prob])
            # confidence masking
            if args.conf_mask:
                conf_mask = torch.max(neigh_prob, dim=-1)[0] > args.conf_thresh
                conf_mask = conf_mask.type(torch.float32)
                kl = kl * conf_mask.numpy()
            if args.operator == "mean":
                kl_scores.append(kl.mean())
            elif args.operator == "max":
                kl_scores.append(kl.max())
            elif args.operator == "median":
                kl_scores.append(np.median(kl))

        distances = np.array([np.array(xi) for xi in distances])

        if args.reverse:
            selected_inds = np.argpartition(kl_scores, annotations_per_iteration)[:annotations_per_iteration]
        else:
            selected_inds = np.argpartition(kl_scores, -annotations_per_iteration)[-annotations_per_iteration:]

    elif args.tfidf and args.cls:  # Half neighbourhood with tfidf - half with cls embs (ablation)
        if train_dataset is None:
            train_dataset = get_glue_tensor_dataset(labeled_inds, args, args.task_name, tokenizer, train=True)
        _train_results, train_logits = my_evaluate(train_dataset, args, model, prefix="",
                                                   al_test=False, mc_samples=None,
                                                   return_mean_embs=args.mean_embs,
                                                   return_mean_output=args.mean_out,
                                                   return_cls=args.cls
                                                   )
        dtrain_tfidf = tfidf_dtrain_reprs
        dpool_tfidf = tfidf_dpool_reprs

        embs = 'bert_cls'
        dtrain_cls = normalize(_train_results[embs]).detach().cpu()
        dpool_cls = normalize(results_dpool[embs]).detach().cpu()

        distances = None
        nei_stats_list = []
        num_adv = None
        if not args.knn_lab:
            # centroids: UNLABELED data points
            
            # Use ANN (Approximate Nearest Neighbors) or exact KNN
            use_ann = getattr(args, 'use_ann', False) and FAISS_AVAILABLE
            
            if use_ann:
                logger.info("Using FAISS for TF-IDF and CLS neighborhoods")
                
                # Build FAISS index for TF-IDF
                d_tfidf = dtrain_tfidf.shape[1]
                xb_tfidf = dtrain_tfidf.numpy().astype('float32')
                xq_tfidf = dpool_tfidf.numpy().astype('float32')
                index_tfidf = faiss.IndexFlatL2(d_tfidf)
                index_tfidf.add(xb_tfidf)
                
                # Build FAISS index for CLS
                d_cls = dtrain_cls.shape[1]
                xb_cls = dtrain_cls.numpy().astype('float32')
                xq_cls = dpool_cls.numpy().astype('float32')
                index_cls = faiss.IndexFlatL2(d_cls)
                index_cls.add(xb_cls)
                
                # Batch search
                k = args.num_nei
                distances_tfidf_all, neighbours_tfidf_all = index_tfidf.search(xq_tfidf, k)
                distances_cls_all, neighbours_cls_all = index_cls.search(xq_cls, k)
                
                logger.info(f"FAISS indexes built for TF-IDF and CLS with {index_tfidf.ntotal} vectors")
            else:
                logger.info("Using sklearn KNeighborsClassifier for TF-IDF and CLS neighborhoods")
                # Original sklearn KNN implementation
                # tfidf neighbourhood
                neigh_tfidf = KNeighborsClassifier(n_neighbors=args.num_nei)
                neigh_tfidf.fit(X=dtrain_tfidf, y=np.array(y_original)[labeled_inds])

                # cls neighbourhood
                neigh_cls = KNeighborsClassifier(n_neighbors=args.num_nei)
                neigh_cls.fit(X=dtrain_cls, y=np.array(y_original)[labeled_inds])

            criterion = nn.KLDivLoss(reduction='none') if not args.ce else nn.CrossEntropyLoss()
            dist = DistanceMetric.get_metric('euclidean')

            kl_scores = []
            num_adv = 0
            distances = []
            pairs = []
            label_list = processor.get_labels()
            label_map = {label: i for i, label in enumerate(label_list)}
            for unlab_i, unlab_logit in enumerate(
                    tqdm(logits_dpool, desc="Finding neighbours for every unlabeled data point")):
                # unlab candidate data point
                unlab_true_label = label_map[y_original[candidate_inds[unlab_i]]]
                unlab_pred_label = int(np.argmax(unlab_logit))
                correct_prediction = True if unlab_true_label == unlab_pred_label else False

                if use_ann:
                    # Use pre-computed FAISS results
                    distances_tfidf = distances_tfidf_all[unlab_i:unlab_i+1]
                    neighbours_tfidf = neighbours_tfidf_all[unlab_i:unlab_i+1]
                    distances_cls = distances_cls_all[unlab_i:unlab_i+1]
                    neighbours_cls = neighbours_cls_all[unlab_i:unlab_i+1]
                else:
                    # Use sklearn KNN
                    # tfidf neighbourhood
                    unlab_tfidf = dpool_tfidf[unlab_i]
                    distances_tfidf, neighbours_tfidf = neigh_tfidf.kneighbors(X=[unlab_tfidf.numpy()],
                                                                               return_distance=True)
                    # cls neighbourhood
                    unlab_cls = dpool_cls[unlab_i]
                    distances_cls, neighbours_cls = neigh_cls.kneighbors(X=[unlab_cls.numpy()], return_distance=True)
                
                labeled_neighbours_tfidf_inds = np.array(labeled_inds)[neighbours_tfidf[0]]  # orig inds
                labeled_neighbours_tfidf_labels = train_dataset.tensors[3][neighbours_tfidf[0]]
                logits_neigh_tfidf = [train_logits[n] for n in neighbours_tfidf]
                preds_neigh_tfidf = [np.argmax(train_logits[n], axis=1) for n in neighbours_tfidf]
                neigh_prob_tfidf = F.softmax(train_logits[neighbours_tfidf], dim=-1)

                labeled_neighbours_cls_inds = np.array(labeled_inds)[neighbours_cls[0]]  # orig inds
                labeled_neighbours_cls_labels = train_dataset.tensors[3][neighbours_cls[0]]
                logits_neigh_cls = [train_logits[n] for n in neighbours_cls]
                preds_neigh_cls = [np.argmax(train_logits[n], axis=1) for n in neighbours_cls]
                neigh_prob_cls = F.softmax(train_logits[neighbours_cls], dim=-1)

                distances.append((distances_tfidf[0].mean(), distances_cls[0].mean()))
                common_neighbours_inds = [x for x in neighbours_tfidf[0] if x in neighbours_cls[0]]
                common_neighbours_labels = train_dataset.tensors[3][common_neighbours_inds]

                common_neighbours_inds_orig = list(np.array(labeled_inds)[common_neighbours_inds])
                common_neighbours_labels_orig = [label_map[y_original[x]] for x in common_neighbours_inds_orig]
                assert sorted(common_neighbours_labels_orig) == sorted(common_neighbours_labels)

                # same predicted label with neighbourhood (percentage)
                pred_label_tfif_per = len([x for x in labeled_neighbours_tfidf_labels if x == unlab_pred_label]) / len(
                    labeled_neighbours_tfidf_labels)
                pred_label_cls_per = len([x for x in labeled_neighbours_cls_labels if x == unlab_pred_label]) / len(
                    labeled_neighbours_cls_labels)

                # same true label with neighbourhood (percentage)
                true_label_tfif_per = len([x for x in labeled_neighbours_tfidf_labels if x == unlab_true_label]) / len(
                    labeled_neighbours_tfidf_labels)
                true_label_cls_per = len([x for x in labeled_neighbours_cls_labels if x == unlab_true_label]) / len(
                    labeled_neighbours_cls_labels)

                nei_stats = {'pred_label_tfif_per': pred_label_tfif_per,
                             'pred_label_cls_per': pred_label_cls_per,
                             'true_label_tfif_per': true_label_tfif_per,
                             'true_label_cls_per': true_label_cls_per,
                             'common_neighbours': len(common_neighbours_inds_orig)}
                nei_stats_list.append(nei_stats)

                # calculate score
                if args.ce:
                    kl = np.array([criterion(unlab_logit.view(-1, args.num_classes), label.view(-1)) for label in
                                   labeled_neighbours_tfidf_labels]
                                  + [criterion(unlab_logit.view(-1, args.num_classes), label.view(-1)) for label in
                                     labeled_neighbours_cls_labels])
                else:
                    uda_softmax_temp = 1
                    candidate_log_prob = F.log_softmax(unlab_logit / uda_softmax_temp, dim=-1)
                    kl = np.array(
                        [torch.sum(criterion(candidate_log_prob, n), dim=-1).numpy() for n in neigh_prob_tfidf]
                        + [torch.sum(criterion(candidate_log_prob, n), dim=-1).numpy() for n in neigh_prob_cls])

                if args.operator == "mean":
                    kl_scores.append(kl.mean())
                elif args.operator == "max":
                    kl_scores.append(kl.max())
                elif args.operator == "median":
                    kl_scores.append(np.median(kl))

            if args.reverse:
                selected_inds = np.argpartition(kl_scores, annotations_per_iteration)[:annotations_per_iteration]
            else:
                selected_inds = np.argpartition(kl_scores, -annotations_per_iteration)[-annotations_per_iteration:]

    else:  # standard method
        if train_dataset is None:
            train_dataset = get_glue_tensor_dataset(labeled_inds, args, args.task_name, tokenizer, train=True)
        _train_results, train_logits = my_evaluate(train_dataset, args, model, prefix="",
                                                   al_test=False, mc_samples=None,
                                                   return_mean_embs=args.mean_embs,
                                                   return_mean_output=args.mean_out,
                                                   return_cls=args.cls
                                                   )
        if args.bert_rep and bert_representations is not None:
            # Use representations of pretrained model
            dtrain_reprs = bert_representations[labeled_inds]
            dpool_reprs = bert_representations[candidate_inds]
        elif tfidf_dtrain_reprs is not None:
            # Use tfidf representations
            dtrain_reprs = tfidf_dtrain_reprs
            dpool_reprs = tfidf_dpool_reprs
        else:
            # Use representations of current fine-tuned model *CAL*
            if args.mean_embs and args.cls:
                dtrain_reprs = torch.cat((_train_results['bert_mean_inputs'], _train_results['bert_cls']), dim=1)
                dpool_reprs = torch.cat((results_dpool['bert_mean_inputs'], results_dpool['bert_cls']), dim=1)
            elif args.mean_embs:
                embs = 'bert_mean_inputs'
                dtrain_reprs = _train_results[embs]
                dpool_reprs = results_dpool[embs]
            elif args.mean_out:
                embs = 'bert_mean_output'
                dtrain_reprs = _train_results[embs]
                dpool_reprs = results_dpool[embs]
            elif args.cls:
                embs = 'bert_cls'
                dtrain_reprs = _train_results[embs]
                dpool_reprs = results_dpool[embs]
            else:
                NotImplementedError

        if tfidf_dtrain_reprs is None:
            train_bert_emb = normalize(dtrain_reprs).detach().cpu()
            dpool_bert_emb = normalize(dpool_reprs).detach().cpu()
        else:
            train_bert_emb = dtrain_reprs
            dpool_bert_emb = dpool_reprs

        distances = None

        num_adv = None
        if not args.knn_lab:  # centroids: UNLABELED data points
            #####################################################
            # Contrastive Active Learning (CAL)
            #####################################################
            import time
            
            # Use ANN (Approximate Nearest Neighbors) or exact KNN
            use_ann = getattr(args, 'use_ann', False) and FAISS_AVAILABLE
            # Use sklearn ANN (ball_tree/kd_tree with approximate search)
            use_sklearn_ann = getattr(args, 'use_sklearn_ann', False)
            # Use Milvus Lite for vector database KNN search
            use_milvus = getattr(args, 'use_milvus', False)
            # Use Apache Spark for distributed KNN search
            use_spark = getattr(args, 'use_spark', False)
            
            # Timing statistics
            knn_build_time = 0
            knn_search_time = 0
            knn_start_time = time.time()
            
            if use_milvus:
                logger.info("Using Milvus Lite (local database file) for vector database KNN search")
                build_start = time.time()
                
                # Milvus Lite implementation using MilvusClient (pymilvus 2.3+)
                try:
                    from pymilvus import MilvusClient
                except ImportError:
                    raise ImportError("pymilvus is not installed or version < 2.3. Install it with: pip install pymilvus>=2.3")
                
                d = train_bert_emb.shape[1]  # dimension
                k = args.num_nei  # number of neighbors
                
                # Prepare data
                xb = train_bert_emb.numpy().astype('float32')  # database vectors (labeled data)
                xq = dpool_bert_emb.numpy().astype('float32')  # query vectors (unlabeled data)
                
                # Use indicator-specific database file to avoid conflicts when running multiple experiments in parallel
                # Milvus Lite has a 36-char limit on db filename, so use hash for long indicators
                indicator = getattr(args, 'indicator', 'default')
                if len(f'milvus_{indicator}.db') > 35:
                    import hashlib
                    short_hash = hashlib.md5(indicator.encode()).hexdigest()[:8]
                    milvus_lite_db = f'./milvus_{short_hash}.db'
                else:
                    milvus_lite_db = f'./milvus_{indicator}.db'
                
                # Initialize Milvus collection name
                collection_name = f"cal_collection_iter{iteration}"
                
                # Check if we can reuse previous collection
                # Use indicator-specific cache key to avoid conflicts between parallel experiments
                cache_key = f'_milvus_client_cache_{indicator}'
                if cache_key not in globals():
                    globals()[cache_key] = {
                        'client': None,
                        'collection_name': None,
                        'labeled_inds': [],
                        'dimension': None
                    }
                
                use_incremental = getattr(args, 'use_incremental_index', True)
                can_reuse_collection = False
                _milvus_client_cache = globals()[cache_key]
                
                # Initialize client if needed
                if _milvus_client_cache['client'] is None:
                    _milvus_client_cache['client'] = MilvusClient(uri=milvus_lite_db)
                    logger.info(f"Created MilvusClient with database file: {milvus_lite_db}")
                
                client = _milvus_client_cache['client']
                
                # Check if we can reuse previous collection
                if use_incremental and _milvus_client_cache['collection_name'] is not None:
                    prev_labeled = set(_milvus_client_cache['labeled_inds'])
                    curr_labeled = set(labeled_inds)
                    
                    if (prev_labeled.issubset(curr_labeled) and 
                        _milvus_client_cache['dimension'] == d):
                        new_inds = list(curr_labeled - prev_labeled)
                        new_data_ratio = len(new_inds) / len(labeled_inds) if len(labeled_inds) > 0 else 1.0
                        
                        # Allow incremental if new data < 30%
                        if new_data_ratio < 0.3:
                            can_reuse_collection = True
                            collection_name = _milvus_client_cache['collection_name']
                            logger.info(f"Reusing Milvus collection with {len(prev_labeled)} vectors, adding {len(new_inds)} new vectors ({new_data_ratio*100:.1f}%)")
                
                if can_reuse_collection:
                    # Reuse existing collection and add new vectors
                    prev_labeled = set(_milvus_client_cache['labeled_inds'])
                    new_inds = [i for i in labeled_inds if i not in prev_labeled]
                    
                    if len(new_inds) > 0:
                        new_ind_positions = [labeled_inds.index(i) for i in new_inds]
                        new_vectors = xb[new_ind_positions]
                        
                        # Prepare data for insertion
                        data = [{"id": int(idx), "vector": vec.tolist()} for idx, vec in zip(new_inds, new_vectors)]
                        
                        client.insert(collection_name=collection_name, data=data)
                        logger.info(f"Added {len(new_inds)} new vectors to collection")
                        
                        globals()[cache_key]['labeled_inds'] = labeled_inds.copy()
                    
                    knn_build_time = time.time() - build_start
                    build_method = "incremental"
                else:
                    # Build new collection from scratch
                    # Drop old collection if exists
                    if _milvus_client_cache['collection_name'] is not None:
                        try:
                            if client.has_collection(_milvus_client_cache['collection_name']):
                                client.drop_collection(_milvus_client_cache['collection_name'])
                                logger.info(f"Dropped old collection: {_milvus_client_cache['collection_name']}")
                        except:
                            pass
                    
                    # Drop collection if exists (in case of name collision)
                    if client.has_collection(collection_name):
                        client.drop_collection(collection_name)
                        logger.info(f"Dropped existing collection: {collection_name}")
                    
                    # Get index parameters
                    index_type = getattr(args, 'milvus_index_type', 'FLAT')
                    metric_type = getattr(args, 'milvus_metric_type', 'L2')
                    
                    # Create collection with index
                    # MilvusClient.create_collection automatically creates AUTOINDEX by default
                    # For explicit index control, we need to pass index_params
                    if index_type == "IVF_FLAT":
                        nlist = getattr(args, 'milvus_nlist', 100)
                        # Adjust nlist if data is too small
                        if nlist > len(xb):
                            nlist = max(min(len(xb) // 2, 100), 8)
                            logger.warning(f"Adjusted nlist from {getattr(args, 'milvus_nlist', 100)} to {nlist} (data size={len(xb)})")
                        
                        index_params = client.prepare_index_params()
                        index_params.add_index(
                            field_name="vector",
                            index_type="IVF_FLAT",
                            metric_type=metric_type,
                            params={"nlist": nlist}
                        )
                        logger.info(f"Creating Milvus collection with IVF_FLAT index (nlist={nlist})")
                    elif index_type == "HNSW":
                        index_params = client.prepare_index_params()
                        index_params.add_index(
                            field_name="vector",
                            index_type="HNSW",
                            metric_type=metric_type,
                            params={"M": 16, "efConstruction": 256}
                        )
                        logger.info(f"Creating Milvus collection with HNSW index")
                    else:  # FLAT
                        index_params = client.prepare_index_params()
                        index_params.add_index(
                            field_name="vector",
                            index_type="FLAT",
                            metric_type=metric_type
                        )
                        logger.info(f"Creating Milvus collection with FLAT index (exact search)")
                    
                    # Create collection
                    client.create_collection(
                        collection_name=collection_name,
                        dimension=d,
                        index_params=index_params,
                        metric_type=metric_type
                    )
                    logger.info(f"Created Milvus collection: {collection_name} (dim={d})")
                    
                    # Prepare and insert data
                    data = [{"id": int(idx), "vector": vec.tolist()} for idx, vec in zip(labeled_inds, xb)]
                    client.insert(collection_name=collection_name, data=data)
                    logger.info(f"Inserted {len(xb)} vectors into Milvus collection")
                    
                    # Update cache
                    globals()[cache_key]['collection_name'] = collection_name
                    globals()[cache_key]['labeled_inds'] = labeled_inds.copy()
                    globals()[cache_key]['dimension'] = d
                    
                    knn_build_time = time.time() - build_start
                    build_method = "from_scratch"
                    logger.info(f"Milvus collection built from scratch with {len(xb)} vectors in {knn_build_time:.4f}s")
                
                # Batch search for all unlabeled data points
                search_start = time.time()
                
                index_type = getattr(args, 'milvus_index_type', 'FLAT')
                
                # Prepare search parameters
                search_params = {}
                if index_type == "IVF_FLAT":
                    nprobe = getattr(args, 'milvus_nprobe', 10)
                    search_params = {"nprobe": nprobe}
                elif index_type == "HNSW":
                    ef = getattr(args, 'milvus_ef', 64)
                    search_params = {"ef": ef}
                
                # Search in batches to avoid memory issues
                batch_size = 1000
                distances_all = []
                neighbours_all = []
                
                for i in range(0, len(xq), batch_size):
                    batch_queries = xq[i:i+batch_size].tolist()
                    
                    results = client.search(
                        collection_name=collection_name,
                        data=batch_queries,
                        limit=k,
                        search_params=search_params,
                        output_fields=["id"]
                    )
                    
                    # results is a list of lists, one per query
                    for query_result in results:
                        query_distances = []
                        query_neighbours = []
                        
                        for hit in query_result:
                            # hit is a dict with 'id', 'distance', etc.
                            query_distances.append(float(hit['distance']))
                            query_neighbours.append(int(hit['id']))
                        
                        # Pad if necessary (when database has fewer than k vectors)
                        if len(query_distances) < k:
                            query_distances.extend([float('inf')] * (k - len(query_distances)))
                            query_neighbours.extend([-1] * (k - len(query_neighbours)))
                        
                        distances_all.append(query_distances)
                        neighbours_all.append(query_neighbours)
                
                distances_all = np.array(distances_all, dtype='float32')
                neighbours_all = np.array(neighbours_all, dtype='int64')
                
                knn_search_time = time.time() - search_start
                logger.info(f"Milvus search completed for {len(xq)} queries in {knn_search_time:.4f}s")
                
                # Convert Milvus IDs (actual sample indices) back to positions in labeled_inds array
                # Create mapping: sample_id -> position in labeled_inds
                id_to_position = {sample_id: pos for pos, sample_id in enumerate(labeled_inds)}
                
                # Convert neighbor IDs to positions
                neighbours_all_positions = np.zeros_like(neighbours_all)
                for i in range(len(neighbours_all)):
                    for j in range(len(neighbours_all[i])):
                        neighbor_id = neighbours_all[i][j]
                        if neighbor_id == -1:  # Padding value
                            neighbours_all_positions[i][j] = -1
                        else:
                            neighbours_all_positions[i][j] = id_to_position[neighbor_id]
                
                neighbours_all = neighbours_all_positions
                logger.info(f"Converted Milvus sample IDs to labeled_inds positions")
                
            elif use_spark:
                logger.info(f"Using Apache Spark for distributed KNN search")
                build_start = time.time()
                
                # Spark KNN implementation
                try:
                    from pyspark.sql import SparkSession
                    # Note: numpy is already imported at module level as np
                except ImportError:
                    raise ImportError("pyspark is not installed. Install it with: pip install pyspark")
                
                d = train_bert_emb.shape[1]  # dimension
                k = args.num_nei  # number of neighbors
                xb = train_bert_emb.numpy().astype('float32')  # database vectors
                xq = dpool_bert_emb.numpy().astype('float32')  # query vectors
                
                # Initialize Spark session
                spark_method = getattr(args, 'spark_knn_method', 'exact')
                num_partitions = getattr(args, 'spark_partitions', None)
                
                spark = SparkSession.builder \
                    .appName("CAL_KNN_Search") \
                    .config("spark.driver.memory", "8g") \
                    .config("spark.executor.memory", "8g") \
                    .config("spark.sql.shuffle.partitions", "200") \
                    .getOrCreate()
                
                logger.info(f"Spark session started with method={spark_method}")
                
                if spark_method == "lsh":
                    # Custom LSH implementation for batch processing (without Spark MLlib)
                    num_tables = getattr(args, 'spark_lsh_num_tables', 5)
                    bucket_width = getattr(args, 'spark_lsh_bucket_length', 2.0)
                    
                    logger.info(f"Building custom LSH index with {num_tables} hash tables, bucket_width={bucket_width}")
                    
                    # Generate random projection vectors for LSH
                    # Each hash table has its own random projection matrix
                    np.random.seed(42)
                    hash_matrices = []
                    for _ in range(num_tables):
                        # Random Gaussian projection matrix: (num_projections, d)
                        # More projections = more precise bucketing
                        num_projections = max(1, int(d / 16))  # Adaptive: ~48 projections for 768-dim
                        random_matrix = np.random.randn(num_projections, d).astype('float32')
                        hash_matrices.append(random_matrix)
                    
                    logger.info(f"Generated {num_tables} hash tables with {num_projections} projections each")
                    
                    # Broadcast hash matrices and training data
                    broadcast_hash_matrices = spark.sparkContext.broadcast(hash_matrices)
                    broadcast_xb = spark.sparkContext.broadcast(xb)
                    broadcast_k = spark.sparkContext.broadcast(k)
                    broadcast_bucket_width = spark.sparkContext.broadcast(bucket_width)
                    
                    knn_build_time = time.time() - build_start
                    search_start = time.time()
                    
                    # Create RDD of query vectors
                    query_rdd = spark.sparkContext.parallelize(
                        [(int(i), xq[i].tolist()) for i in range(len(xq))],
                        numSlices=num_partitions or 64
                    )
                    
                    logger.info(f"Created query RDD with {len(xq)} vectors, partitions={query_rdd.getNumPartitions()}")
                    
                    # Define LSH search function for a partition of queries
                    def lsh_search_partition(iterator):
                        import numpy as _np  # Use different name to avoid shadowing outer scope
                        from collections import defaultdict
                        
                        train_data = broadcast_xb.value
                        hash_mats = broadcast_hash_matrices.value
                        num_neighbors = broadcast_k.value
                        bw = broadcast_bucket_width.value
                        
                        # Pre-compute hash codes for all training data
                        # This is done once per partition (amortized cost)
                        train_hash_codes = []
                        for hash_mat in hash_mats:
                            # Project training data: (N, d) @ (d, num_proj) = (N, num_proj)
                            projections = train_data @ hash_mat.T
                            # Quantize into buckets
                            hash_codes = _np.floor(projections / bw).astype('int32')
                            train_hash_codes.append(hash_codes)
                        
                        results = []
                        for query_id, query_vec in iterator:
                            query = _np.array(query_vec, dtype='float32')
                            
                            # Hash the query with all hash tables
                            candidate_set = set()
                            for table_idx, hash_mat in enumerate(hash_mats):
                                # Project query
                                query_proj = query @ hash_mat.T
                                query_hash = _np.floor(query_proj / bw).astype('int32')
                                
                                # Find training points in same buckets
                                train_hashes = train_hash_codes[table_idx]
                                # Check which training points match the query hash
                                matches = _np.all(train_hashes == query_hash, axis=1)
                                candidate_indices = _np.where(matches)[0]
                                candidate_set.update(candidate_indices.tolist())
                            
                            # If no candidates found (rare), fall back to random sampling
                            if len(candidate_set) == 0:
                                candidate_set = set(_np.random.choice(len(train_data), 
                                                                     min(num_neighbors * 10, len(train_data)), 
                                                                     replace=False))
                            
                            # Compute exact distances only for candidates
                            candidates = list(candidate_set)
                            candidate_vectors = train_data[candidates]
                            distances = _np.linalg.norm(candidate_vectors - query, axis=1)
                            
                            # Get top k from candidates
                            if len(distances) <= num_neighbors:
                                top_k_idx = _np.arange(len(distances))
                            else:
                                top_k_idx = _np.argpartition(distances, num_neighbors)[:num_neighbors]
                            
                            # Sort the top k by distance
                            top_k_idx = top_k_idx[_np.argsort(distances[top_k_idx])]
                            
                            # Map back to original indices
                            top_k_indices = [candidates[i] for i in top_k_idx]
                            top_k_distances = distances[top_k_idx]
                            
                            # Pad if necessary
                            if len(top_k_indices) < num_neighbors:
                                top_k_indices = top_k_indices + [-1] * (num_neighbors - len(top_k_indices))
                                top_k_distances = _np.concatenate([top_k_distances, 
                                                                  _np.full(num_neighbors - len(top_k_distances), _np.inf)])
                            
                            results.append((query_id, top_k_indices[:num_neighbors], top_k_distances[:num_neighbors].tolist()))
                        
                        return iter(results)
                    
                    # Execute distributed LSH search
                    logger.info("Starting distributed LSH search...")
                    result_rdd = query_rdd.mapPartitions(lsh_search_partition)
                    
                    # Collect results
                    results = result_rdd.collect()
                    
                    # Sort by query_id and extract neighbors/distances
                    results.sort(key=lambda x: x[0])
                    
                    all_neighbors = []
                    all_distances = []
                    
                    for query_id, neighbors, distances in results:
                        all_neighbors.append(neighbors[:k])
                        all_distances.append(distances[:k])
                    
                    neighbours_all = np.array(all_neighbors, dtype='int64')
                    distances_all = np.array(all_distances, dtype='float32')
                    
                    # Clean up broadcast variables
                    broadcast_hash_matrices.unpersist()
                    broadcast_xb.unpersist()
                    broadcast_k.unpersist()
                    broadcast_bucket_width.unpersist()
                    
                else:
                    # Exact KNN using Spark RDD mapPartitions (optimized)
                    logger.info("Using Spark exact KNN (RDD mapPartitions with broadcast)")
                    
                    # Broadcast training data for efficient distributed access
                    broadcast_xb = spark.sparkContext.broadcast(xb)
                    broadcast_k = spark.sparkContext.broadcast(k)
                    
                    knn_build_time = time.time() - build_start
                    search_start = time.time()
                    
                    # Create RDD of query vectors with indices
                    query_rdd = spark.sparkContext.parallelize(
                        [(int(i), xq[i].tolist()) for i in range(len(xq))],
                        numSlices=num_partitions or 64
                    )
                    
                    logger.info(f"Created query RDD with {len(xq)} vectors, partitions={query_rdd.getNumPartitions()}")
                    
                    # Define function to compute KNN for a partition of queries
                    def compute_knn_partition(iterator):
                        import numpy as _np  # Use different name to avoid shadowing outer scope
                        train_data = broadcast_xb.value
                        num_neighbors = broadcast_k.value
                        
                        results = []
                        for query_id, query_vec in iterator:
                            query = _np.array(query_vec, dtype='float32')
                            
                            # Compute L2 distances to all training points
                            distances = _np.linalg.norm(train_data - query, axis=1)
                            
                            # Get top k nearest neighbors
                            if len(distances) <= num_neighbors:
                                top_k_indices = _np.arange(len(distances))
                                top_k_distances = distances
                            else:
                                top_k_indices = _np.argpartition(distances, num_neighbors)[:num_neighbors]
                                top_k_indices = top_k_indices[_np.argsort(distances[top_k_indices])]
                                top_k_distances = distances[top_k_indices]
                            
                            results.append((query_id, top_k_indices.tolist(), top_k_distances.tolist()))
                        
                        return iter(results)
                    
                    # Execute distributed KNN computation
                    result_rdd = query_rdd.mapPartitions(compute_knn_partition)
                    
                    # Collect results
                    results = result_rdd.collect()
                    
                    # Sort by query_id and extract neighbors/distances
                    results.sort(key=lambda x: x[0])
                    
                    all_neighbors = []
                    all_distances = []
                    
                    for query_id, neighbors, distances in results:
                        # Pad if necessary
                        if len(neighbors) < k:
                            neighbors = neighbors + [-1] * (k - len(neighbors))
                            distances = distances + [float('inf')] * (k - len(distances))
                        all_neighbors.append(neighbors[:k])
                        all_distances.append(distances[:k])
                    
                    neighbours_all = np.array(all_neighbors, dtype='int64')
                    distances_all = np.array(all_distances, dtype='float32')
                    
                    # Clean up broadcast variables
                    broadcast_xb.unpersist()
                    broadcast_k.unpersist()
                
                knn_search_time = time.time() - search_start
                logger.info(f"Spark {spark_method} search completed for {len(xq)} queries in {knn_search_time:.4f}s")
                
                # Clean up Spark session
                spark.stop()
                logger.info("Spark session stopped")
                
            elif use_ann and not use_sklearn_ann:
                logger.info("Using FAISS for Approximate Nearest Neighbors (ANN) search")
                build_start = time.time()
                
                # FAISS ANN implementation
                d = train_bert_emb.shape[1]  # dimension
                k = args.num_nei  # number of neighbors
                
                # Build FAISS index
                xb = train_bert_emb.numpy().astype('float32')  # database vectors (labeled data)
                xq = dpool_bert_emb.numpy().astype('float32')  # query vectors (unlabeled data)
                
                # Check if we can use incremental update
                global _faiss_index_cache
                use_incremental = getattr(args, 'use_incremental_index', True)  # Enable by default
                can_reuse_index = False
                
                if use_incremental and _faiss_index_cache['index'] is not None:
                    # Check if we can reuse the previous index
                    prev_labeled = set(_faiss_index_cache['labeled_inds'])
                    curr_labeled = set(labeled_inds)
                    
                    # Check conditions for incremental update:
                    # 1. Previous labeled data is a subset of current (only added, no removal)
                    # 2. Same dimension
                    # 3. For IVF: new data is <30% of total (avoid retraining clusters)
                    if (prev_labeled.issubset(curr_labeled) and 
                        _faiss_index_cache['dimension'] == d):
                        
                        new_inds = list(curr_labeled - prev_labeled)
                        new_data_ratio = len(new_inds) / len(labeled_inds)
                        
                        # For IndexFlatL2: always allow incremental
                        # For IndexIVFFlat: only if new data < 30%
                        if (_faiss_index_cache['index_type'] == 'IndexFlatL2' or 
                            (_faiss_index_cache['index_type'] == 'IndexIVFFlat' and new_data_ratio < 0.3)):
                            can_reuse_index = True
                            logger.info(f"Reusing previous index with {len(prev_labeled)} vectors, adding {len(new_inds)} new vectors ({new_data_ratio*100:.1f}%)")
                
                rebuild_from_scratch = False
                
                if can_reuse_index:
                    # Incremental update
                    index = _faiss_index_cache['index']
                    index_to_labeled_map = _faiss_index_cache['index_to_labeled_map'].copy()
                    
                    # Find new indices to add
                    prev_labeled = set(_faiss_index_cache['labeled_inds'])
                    new_inds = [i for i in labeled_inds if i not in prev_labeled]
                    
                    if len(new_inds) > 0:
                        # Get new vectors
                        new_ind_positions = [labeled_inds.index(i) for i in new_inds]
                        new_vectors = xb[new_ind_positions].astype('float32')
                        
                        # Add new vectors to existing index
                        index.add(new_vectors)
                        
                        # Update the index-to-labeled mapping for the new vectors
                        for pos in new_ind_positions:
                            index_to_labeled_map.append(pos)
                        
                        # Update cache with new labeled_inds and mapping
                        _faiss_index_cache['labeled_inds'] = labeled_inds.copy()
                        _faiss_index_cache['index_to_labeled_map'] = index_to_labeled_map
                        
                        logger.info(f"Added {len(new_inds)} new vectors to existing index")
                    
                    knn_build_time = time.time() - build_start
                    build_method = "incremental"
                else:
                    # Build from scratch
                    rebuild_from_scratch = True
                    
                    # IndexFlatL2 for exact search (can be replaced with IndexIVFFlat for faster approximate search)
                    if getattr(args, 'ann_nprobe', None) is not None:
                        # Use IVF (Inverted File Index) for approximate nearest neighbor search
                        nlist = max(1, min(100, len(labeled_inds) // 10))  # number of clusters (at least 1)
                        quantizer = faiss.IndexFlatL2(d)
                        index = faiss.IndexIVFFlat(quantizer, d, nlist)
                        index.train(xb)
                        index.nprobe = getattr(args, 'ann_nprobe', 10)  # number of clusters to search
                        index_type = 'IndexIVFFlat'
                        logger.info(f"Building FAISS IndexIVFFlat (approximate search) with nlist={nlist}, nprobe={index.nprobe}")
                    else:
                        # Use exact search with FAISS (still faster than sklearn for batch queries)
                        index = faiss.IndexFlatL2(d)
                        index_type = 'IndexFlatL2'
                        logger.info(f"Building FAISS IndexFlatL2 (exact search)")
                    
                    index.add(xb)
                    
                    # Create index-to-labeled mapping (initially just 0, 1, 2, ..., len(labeled_inds)-1)
                    index_to_labeled_map = list(range(len(labeled_inds)))
                    
                    # Update cache
                    _faiss_index_cache['index'] = index
                    _faiss_index_cache['labeled_inds'] = labeled_inds.copy()
                    _faiss_index_cache['dimension'] = d
                    _faiss_index_cache['index_type'] = index_type
                    _faiss_index_cache['index_to_labeled_map'] = index_to_labeled_map
                    
                    knn_build_time = time.time() - build_start
                    build_method = "from_scratch"
                
                if rebuild_from_scratch:
                    logger.info(f"FAISS index built from scratch with {index.ntotal} vectors in {knn_build_time:.4f}s")
                else:
                    logger.info(f"FAISS index updated incrementally, total {index.ntotal} vectors in {knn_build_time:.4f}s")
                
                # Batch search for all unlabeled data points
                search_start = time.time()
                distances_all, neighbours_all = index.search(xq, k)
                knn_search_time = time.time() - search_start
                logger.info(f"FAISS search completed for {len(xq)} queries in {knn_search_time:.4f}s")
            elif use_sklearn_ann:
                # sklearn ANN using ball_tree algorithm with approximate search
                from sklearn.neighbors import NearestNeighbors
                
                logger.info("Using sklearn NearestNeighbors with ball_tree algorithm (approximate search)")
                build_start = time.time()
                
                # Use ball_tree algorithm which can do approximate search with smaller leaf_size
                # Smaller leaf_size = faster but less accurate
                # Default leaf_size=30, we use smaller for more approximate results
                ann_leaf_size = getattr(args, 'ann_leaf_size', 10)  # Smaller = more approximate
                
                neigh = NearestNeighbors(
                    n_neighbors=args.num_nei,
                    algorithm='ball_tree',  # ball_tree is good for approximate search
                    leaf_size=ann_leaf_size,  # Smaller leaf_size for approximate results
                    metric='euclidean'
                )
                neigh.fit(X=train_bert_emb)
                knn_build_time = time.time() - build_start
                logger.info(f"sklearn ball_tree ANN index built (leaf_size={ann_leaf_size}) in {knn_build_time:.4f}s")
            else:
                logger.info("Using sklearn KNeighborsClassifier for exact KNN search")
                build_start = time.time()
                
                # Original sklearn KNN implementation
                neigh = KNeighborsClassifier(n_neighbors=args.num_nei)
                neigh.fit(X=train_bert_emb, y=np.array(y_original)[labeled_inds])
                knn_build_time = time.time() - build_start
                logger.info(f"sklearn KNN index built in {knn_build_time:.4f}s")
            
            # criterion = nn.KLDivLoss(reduction='none') if not args.ce else nn.BCEWithLogitsLoss()
            criterion = nn.KLDivLoss(reduction='none') if not args.ce else nn.CrossEntropyLoss()
            dist = DistanceMetric.get_metric('euclidean')

            kl_scores = []
            num_adv = 0
            distances = []
            pairs = []
            
            # Probability caching optimization (Algorithm 1 line 3)
            cache_probs = getattr(args, 'cache_probabilities', True)
            if cache_probs:
                cache_start = time.time()
                
                # Lazy cache for labeled samples (only compute when first accessed as neighbor)
                train_probs_cache = {}  # Dictionary: index -> probability
                cache_access_count = {}  # Track how many times each cached sample is accessed
                total_cache_hits = 0  # Total number of cache hits (reuses)
                total_cache_misses = 0  # Total number of cache misses (first access)
                
                # Pre-compute log probabilities for all unlabeled data (candidates)
                # This is always beneficial since we iterate through all candidates
                uda_softmax_temp = 1 if not args.ce else None
                if uda_softmax_temp is not None:
                    pool_log_probs = F.log_softmax(logits_dpool / uda_softmax_temp, dim=-1)  # [N_pool, num_classes]
                else:
                    pool_log_probs = None
                
                cache_time = time.time() - cache_start
                logger.info(f"Pre-computed log probabilities for {len(logits_dpool)} unlabeled samples in {cache_time:.4f}s")
                logger.info(f"Using lazy caching for labeled samples (compute on first access)")
            else:
                train_probs_cache = None
                cache_access_count = None
                pool_log_probs = None
                cache_time = 0
                total_cache_hits = 0
                total_cache_misses = 0
            
            # Track sklearn search time if not using Milvus/Spark/FAISS ANN
            # Only reset knn_search_time if we're using sklearn (not Milvus/Spark which already set it)
            if not use_ann and not use_milvus and not use_spark:
                knn_search_time = 0
            
            # Detailed timing for loop components - FIXED VERSION
            time_mapping = 0              # FAISS index to labeled_inds mapping
            time_data_prep = 0            # Data extraction and indexing (neighbors, labels, logits)
            time_list_comprehension = 0   # List comprehensions for logits and preds
            time_prob_cache = 0           # Probability cache lookup and computation
            time_kl_compute = 0           # KL divergence computation
            time_kl_aggregate = 0         # KL aggregation (mean/max/median)
            time_pred_stats = 0           # Prediction comparison statistics
            time_tqdm = 0                 # tqdm progress bar overhead
            time_other = 0                # Miscellaneous overhead
            
            loop_start_total = time.time()
            
            for unlab_i, candidate in enumerate(
                    tqdm(zip(dpool_bert_emb, logits_dpool), desc="Finding neighbours for every unlabeled data point", mininterval=0.5)):
                iter_start = time.time()
                
                # Step 1: Unpack candidate data
                step_start = time.time()
                unlab_representation, unlab_logit = candidate
                time_other += time.time() - step_start
                
                # Step 2: Index mapping (FAISS/Milvus/Spark) or KNN search (sklearn)
                if use_spark or use_milvus or (use_ann and not use_sklearn_ann):
                    step_start = time.time()
                    distances_ = distances_all[unlab_i:unlab_i+1]
                    neighbours_raw = neighbours_all[unlab_i:unlab_i+1]
                    if use_milvus:
                        # Milvus returns label_idx directly, no need for mapping
                        neighbours = np.array([neighbours_raw[0]])
                    elif use_spark:
                        # Spark returns label_idx directly (position in labeled_inds)
                        neighbours = np.array([neighbours_raw[0]])
                    else:
                        # FAISS needs index-to-labeled mapping
                        neighbours = np.array([[index_to_labeled_map[n] for n in neighbours_raw[0]]])
                    time_mapping += time.time() - step_start
                else:
                    # Both sklearn exact KNN and sklearn ANN use kneighbors
                    step_start = time.time()
                    distances_, neighbours = neigh.kneighbors(X=[candidate[0].numpy()], return_distance=True)
                    knn_search_time += time.time() - step_start
                
                # Step 3: Data preparation - extract neighbor information
                step_start = time.time()
                distances.append(distances_[0])
                labeled_neighbours_inds = np.array(labeled_inds)[neighbours[0]]
                labeled_neighbours_labels = train_dataset.tensors[3][neighbours[0]]
                time_data_prep += time.time() - step_start
                
                # Step 4: List comprehensions for logits and predictions
                step_start = time.time()
                logits_neigh = [train_logits[n] for n in neighbours]
                logits_candidate = candidate[1]
                preds_neigh = [np.argmax(train_logits[n], axis=1) for n in neighbours]
                time_list_comprehension += time.time() - step_start
                
                # Step 5: Probability computation with caching
                step_start = time.time()
                if cache_probs and train_probs_cache is not None:
                    neigh_prob_list = []
                    for n_idx in neighbours[0]:
                        if n_idx not in train_probs_cache:
                            train_probs_cache[n_idx] = F.softmax(train_logits[n_idx:n_idx+1], dim=-1)
                            cache_access_count[n_idx] = 1
                            total_cache_misses += 1
                        else:
                            cache_access_count[n_idx] += 1
                            total_cache_hits += 1
                        neigh_prob_list.append(train_probs_cache[n_idx])
                    neigh_prob = torch.cat(neigh_prob_list, dim=0)
                else:
                    neigh_prob = F.softmax(train_logits[neighbours], dim=-1)
                time_prob_cache += time.time() - step_start
                
                # Step 6: Prediction statistics
                step_start = time.time()
                pred_candidate = [np.argmax(candidate[1])]
                num_diff_pred = len(list(set(preds_neigh).intersection(pred_candidate)))
                if num_diff_pred > 0: num_adv += 1
                time_pred_stats += time.time() - step_start
                
                # Step 7: KL divergence computation
                step_start = time.time()
                if args.ce:
                    kl = np.array([criterion(unlab_logit.view(-1, args.num_classes), label.view(-1)) for label in
                                   labeled_neighbours_labels])
                else:
                    if cache_probs and pool_log_probs is not None:
                        candidate_log_prob = pool_log_probs[unlab_i]
                    else:
                        uda_softmax_temp = 1
                        candidate_log_prob = F.log_softmax(candidate[1] / uda_softmax_temp, dim=-1)
                    kl = np.array([torch.sum(criterion(candidate_log_prob, n), dim=-1).numpy() for n in neigh_prob])
                
                # Confidence masking
                if args.conf_mask:
                    conf_mask = torch.max(neigh_prob, dim=-1)[0] > args.conf_thresh
                    conf_mask = conf_mask.type(torch.float32)
                    kl = kl * conf_mask.numpy()
                time_kl_compute += time.time() - step_start
                
                # Step 8: KL aggregation
                step_start = time.time()
                if args.operator == "mean":
                    kl_scores.append(kl.mean())
                elif args.operator == "max":
                    kl_scores.append(kl.max())
                elif args.operator == "median":
                    kl_scores.append(np.median(kl))
                time_kl_aggregate += time.time() - step_start
                
                # Calculate tqdm overhead (difference between iteration time and measured steps)
                iter_total = time.time() - iter_start
                measured_time = (time_mapping if use_ann else 0) + time_data_prep + time_list_comprehension + \
                               time_prob_cache + time_pred_stats + time_kl_compute + time_kl_aggregate + time_other
                time_tqdm += max(0, iter_total - measured_time)  # Remaining time is tqdm overhead

            distances = np.array([np.array(xi) for xi in distances])
            
            # Calculate total KNN time
            knn_total_time = time.time() - knn_start_time
            
            # Calculate cache statistics
            if cache_probs and train_probs_cache is not None:
                cache_size = len(train_probs_cache)
                cache_hit_rate = cache_size / len(labeled_inds) * 100 if len(labeled_inds) > 0 else 0
                total_accesses = total_cache_hits + total_cache_misses
                computation_savings = total_cache_hits / total_accesses * 100 if total_accesses > 0 else 0
                avg_reuse = total_accesses / cache_size if cache_size > 0 else 0
                
                logger.info(f"Probability cache statistics:")
                logger.info(f"  - Unique samples cached: {cache_size}/{len(labeled_inds)} ({cache_hit_rate:.1f}%)")
                logger.info(f"  - Total accesses: {total_accesses} (Hits: {total_cache_hits}, Misses: {total_cache_misses})")
                logger.info(f"  - Computation savings: {total_cache_hits}/{total_accesses} softmax calls avoided ({computation_savings:.1f}%)")
                logger.info(f"  - Average reuse per cached sample: {avg_reuse:.2f}x")
            
            # Calculate loop total and verify accounting
            loop_total_time = time.time() - loop_start_total
            accounted_time = time_mapping + time_data_prep + time_list_comprehension + \
                           time_prob_cache + time_pred_stats + time_kl_compute + \
                           time_kl_aggregate + time_tqdm + time_other
            unaccounted_time = loop_total_time - accounted_time
            
            # Log detailed timing breakdown (NEW DETAILED VERSION)
            logger.info(f"=" * 80)
            logger.info(f"DETAILED LOOP TIMING BREAKDOWN (Total: {loop_total_time:.4f}s)")
            logger.info(f"=" * 80)
            logger.info(f"  1. Index mapping (FAISS→labeled_inds):    {time_mapping:.4f}s ({time_mapping/loop_total_time*100:5.2f}%)")
            logger.info(f"  2. Data preparation (indexing):           {time_data_prep:.4f}s ({time_data_prep/loop_total_time*100:5.2f}%)")
            logger.info(f"  3. List comprehensions (logits/preds):    {time_list_comprehension:.4f}s ({time_list_comprehension/loop_total_time*100:5.2f}%)")
            logger.info(f"  4. Probability cache/compute:             {time_prob_cache:.4f}s ({time_prob_cache/loop_total_time*100:5.2f}%)")
            logger.info(f"  5. Prediction statistics:                 {time_pred_stats:.4f}s ({time_pred_stats/loop_total_time*100:5.2f}%)")
            logger.info(f"  6. KL divergence computation:             {time_kl_compute:.4f}s ({time_kl_compute/loop_total_time*100:5.2f}%)")
            logger.info(f"  7. KL aggregation (mean/max):             {time_kl_aggregate:.4f}s ({time_kl_aggregate/loop_total_time*100:5.2f}%)")
            logger.info(f"  8. tqdm progress bar overhead:            {time_tqdm:.4f}s ({time_tqdm/loop_total_time*100:5.2f}%)")
            logger.info(f"  9. Other Python overhead:                 {time_other:.4f}s ({time_other/loop_total_time*100:5.2f}%)")
            logger.info(f"  -  Unaccounted (measurement error):       {unaccounted_time:.4f}s ({unaccounted_time/loop_total_time*100:5.2f}%)")
            logger.info(f"=" * 80)
            logger.info(f"  Accounted time:   {accounted_time:.4f}s ({accounted_time/loop_total_time*100:.1f}%)")
            logger.info(f"  Loop total:       {loop_total_time:.4f}s (100.0%)")
            logger.info(f"=" * 80)
            
            # Log KNN timing statistics
            if use_milvus:
                index_type = getattr(args, 'milvus_index_type', 'FLAT')
                metric_type = getattr(args, 'milvus_metric_type', 'L2')
                logger.info(f"Milvus {index_type} Timing - Build: {knn_build_time:.4f}s, Search: {knn_search_time:.4f}s, Total: {knn_total_time:.4f}s")
                logger.info(f"Milvus config: index={index_type}, metric={metric_type}, build_method={build_method}")
                if cache_probs:
                    logger.info(f"Probability cache initialization time: {cache_time:.4f}s")
                logger.info(f"Average search time per query: {knn_search_time/len(candidate_inds)*1000:.2f}ms")
            elif use_spark:
                spark_method = getattr(args, 'spark_knn_method', 'exact')
                logger.info(f"Spark {spark_method} Timing - Build: {knn_build_time:.4f}s, Search: {knn_search_time:.4f}s, Total: {knn_total_time:.4f}s")
                if spark_method == 'lsh':
                    num_tables = getattr(args, 'spark_lsh_num_tables', 5)
                    bucket_width = getattr(args, 'spark_lsh_bucket_length', 2.0)
                    logger.info(f"Spark LSH config: num_tables={num_tables}, bucket_width={bucket_width}")
                if cache_probs:
                    logger.info(f"Probability cache initialization time: {cache_time:.4f}s")
                logger.info(f"Average search time per query: {knn_search_time/len(candidate_inds)*1000:.2f}ms")
            elif use_ann and not use_sklearn_ann:
                logger.info(f"FAISS ANN Timing - Build: {knn_build_time:.4f}s, Search: {knn_search_time:.4f}s, Total: {knn_total_time:.4f}s")
                if cache_probs:
                    logger.info(f"Probability cache initialization time: {cache_time:.4f}s")
                logger.info(f"Average search time per query: {knn_search_time/len(candidate_inds)*1000:.2f}ms")
            elif use_sklearn_ann:
                logger.info(f"sklearn ball_tree ANN search completed in {knn_search_time:.4f}s")
                logger.info(f"KNN Timing - Build: {knn_build_time:.4f}s, Search: {knn_search_time:.4f}s, Total: {knn_total_time:.4f}s")
                if cache_probs:
                    logger.info(f"Probability cache initialization time: {cache_time:.4f}s")
                logger.info(f"Average search time per query: {knn_search_time/len(candidate_inds)*1000:.2f}ms")
            else:
                logger.info(f"sklearn exact KNN search completed in {knn_search_time:.4f}s")
                logger.info(f"KNN Timing - Build: {knn_build_time:.4f}s, Search: {knn_search_time:.4f}s, Total: {knn_total_time:.4f}s")
                if cache_probs:
                    logger.info(f"Probability cache initialization time: {cache_time:.4f}s")
                logger.info(f"Average search time per query: {knn_search_time/len(candidate_inds)*1000:.2f}ms")

            logger.info('Total Different predictions for similar inputs: {}'.format(num_adv))

            # select argmax
            if args.reverse:  # if True select opposite (ablation)
                selected_inds = np.argpartition(kl_scores, annotations_per_iteration)[:annotations_per_iteration]
            else:
                selected_inds = np.argpartition(kl_scores, -annotations_per_iteration)[-annotations_per_iteration:]
            #############################################################################################################################

        else:  # centroids: LABELED data points (ablation)
            criterion = nn.KLDivLoss(reduction='sum') if not args.ce else nn.CrossEntropyLoss()
            # step 1: find neighbours for each *LABELED* data point
            N = dpool_bert_emb.shape[0]
            d = dpool_bert_emb.shape[1]
            k = 5

            xb = dpool_bert_emb.numpy()  # pool
            xq = train_bert_emb.numpy()  # candidates

            index = faiss.IndexFlatL2(d)  # build the index, d=size of vectors
            # here we assume xb contains a n-by-d numpy matrix of type float32
            index.add(xb)  # add vectors to the index
            print(index.ntotal)
            k = args.num_nei  # we want 4 similar vectors
            distances, neighbours = index.search(xq, k)
            kl_scores_per_unlab = np.array([0.] * len(candidate_inds))
            for i, pair in enumerate(neighbours):
                # labeled_logit = train_logits[i]
                labeled_prob = F.softmax(train_logits[i], dim=-1)
                # labeled_log_prob =  F.log_softmax(labeled_logit)
                # unlabeled_logits = logits_dpool[pair]
                # unlabeled_probs = [F.softmax(logits_dpool[i], dim=-1) for i in pair]
                # unlabeled_log_prob = [F.log_softmax(logits_dpool[i] / 1, dim=-1) for i in pair]
                # kl = np.array([torch.sum(criterion(F.log_softmax(logits_dpool[n] / 1, dim=-1),
                #                                    labeled_prob), dim=-1).numpy() for n in pair])
                labeled_neighbours_label = train_dataset.tensors[3][i]
                # KL divergence for each labeled data point (candidate) and labeled (query)
                if args.ce:
                    # kl = np.array([criterion(F.log_softmax(logits_dpool[n] / 1, dim=-1), labeled_prob).numpy()
                    #                for n in pair])
                    kl = np.array(
                        [criterion(logits_dpool[n].view(-1, args.num_classes), labeled_neighbours_label.view(-1)) for n
                         in
                         pair])
                else:
                    kl = np.array([criterion(F.log_softmax(logits_dpool[n] / 1, dim=-1), labeled_prob).numpy()
                                   for n in pair])
                # if kl socre calucalte before update with mean
                scores = np.array([np.append(kl_scores_per_unlab[pair][i], kl[i]) for i in range(0, len(pair))]).mean(
                    axis=1)
                # replace old scores for these unlabeled data
                kl_scores_per_unlab[pair] = scores

            # select argmax
            selected_inds = np.argpartition(kl_scores_per_unlab, -annotations_per_iteration)[
                            -annotations_per_iteration:]
            print()

    # map from dpool inds to original inds
    sampled_ind = list(np.array(candidate_inds)[selected_inds])  # in terms of original inds

    if num_adv is not None:
        num_adv_per = round(num_adv / len(candidate_inds), 2)

    y_lab = np.asarray(y_original, dtype='object')[labeled_inds]
    X_unlab = np.asarray(X_original, dtype='object')[candidate_inds]
    y_unlab = np.asarray(y_original, dtype='object')[candidate_inds]

    labels_list_previous = list(y_lab)
    c = collections.Counter(labels_list_previous)
    stats_list_previous = [(i, c[i] / len(labels_list_previous) * 100.0) for i in c]

    new_samples = np.asarray(X_original, dtype='object')[sampled_ind]
    new_labels = np.asarray(y_original, dtype='object')[sampled_ind]

    # Mean and std of length of selected sequences
    if args.task_name in ['sst-2', 'ag_news', 'dbpedia', 'trec-6', 'imdb', 'pubmed', 'sentiment']: # single sequence
        l = [len(x.split()) for x in new_samples]
    elif args.dataset_name in ['mrpc', 'mnli', 'qnli', 'cola', 'rte', 'qqp', 'nli']:
        l = [len(sentence[0].split()) + len(sentence[1].split()) for sentence in new_samples]  # pairs of sequences
    assert type(l) is list, "type l: {}, l: {}".format(type(l), l)
    length_mean = np.mean(l)
    length_std = np.std(l)
    length_min = np.min(l)
    length_max = np.max(l)

    # Percentages of each class
    labels_list_selected = list(np.array(y_original)[sampled_ind])
    c = collections.Counter(labels_list_selected)
    stats_list = [(i, c[i] / len(labels_list_selected) * 100.0) for i in c]

    labels_list_after = list(new_labels) + list(y_lab)
    c = collections.Counter(labels_list_after)
    stats_list_all = [(i, c[i] / len(labels_list_after) * 100.0) for i in c]

    assert len(set(sampled_ind)) == len(sampled_ind)
    assert bool(not set(sampled_ind) & set(labeled_inds))
    assert len(new_samples) == annotations_per_iteration, 'len(new_samples)={}, annotatations_per_it={}'.format(
        len(new_samples),
        annotations_per_iteration)
    assert len(labeled_inds) + len(candidate_inds) + len(discarded_inds) == len(original_inds), "labeled {}, " \
                                                                                                "candidate {}, " \
                                                                                                "disgarded {}, " \
                                                                                                "original {}".format(
        len(labeled_inds),
        len(candidate_inds),
        len(discarded_inds),
        len(original_inds))

    stats = {'length': {'mean': float(length_mean),
                        'std': float(length_std),
                        'min': float(length_min),
                        'max': float(length_max)},
             'class_selected_samples': stats_list,
             'class_samples_after': stats_list_all,
             'class_samples_before': stats_list_previous,
             }
    
    # Add KNN/ANN timing statistics
    if 'knn_build_time' in locals():
        stats['knn_build_time'] = float(knn_build_time)
        stats['knn_search_time'] = float(knn_search_time)
        stats['knn_total_time'] = float(knn_total_time)
        if use_milvus:
            index_type = getattr(args, 'milvus_index_type', 'FLAT')
            stats['knn_method'] = f'Milvus-{index_type}'
            stats['milvus_index_type'] = index_type
            stats['milvus_metric_type'] = getattr(args, 'milvus_metric_type', 'L2')
            if index_type == 'IVF_FLAT':
                stats['milvus_nlist'] = getattr(args, 'milvus_nlist', 100)
                stats['milvus_nprobe'] = getattr(args, 'milvus_nprobe', 10)
            elif index_type == 'HNSW':
                stats['milvus_ef'] = getattr(args, 'milvus_ef', 64)
        elif use_spark:
            spark_method = getattr(args, 'spark_knn_method', 'exact')
            stats['knn_method'] = f'Spark-{spark_method}'
            stats['spark_knn_method'] = spark_method
            if spark_method == 'lsh':
                stats['spark_lsh_num_tables'] = getattr(args, 'spark_lsh_num_tables', 5)
                stats['spark_lsh_bucket_length'] = getattr(args, 'spark_lsh_bucket_length', 2.0)
            stats['spark_partitions'] = getattr(args, 'spark_partitions', None)
        elif use_sklearn_ann:
            stats['knn_method'] = 'sklearn-ANN-balltree'
        elif use_ann:
            stats['knn_method'] = 'FAISS-ANN'
        else:
            stats['knn_method'] = 'sklearn-KNN-exact'
        stats['probability_cache_enabled'] = cache_probs
        if cache_probs and 'cache_time' in locals():
            stats['probability_cache_time'] = float(cache_time)
            stats['probability_cache_size'] = len(train_probs_cache) if train_probs_cache is not None else 0
            stats['probability_total_accesses'] = total_cache_hits + total_cache_misses
            stats['probability_cache_hits'] = total_cache_hits
            stats['probability_cache_misses'] = total_cache_misses
            stats['probability_computation_savings_pct'] = (total_cache_hits / (total_cache_hits + total_cache_misses) * 100) if (total_cache_hits + total_cache_misses) > 0 else 0
            stats['probability_avg_reuse'] = (total_cache_hits + total_cache_misses) / len(train_probs_cache) if len(train_probs_cache) > 0 else 0
        if cache_probs and 'train_probs_cache' in locals() and train_probs_cache is not None:
            stats['probability_cache_size'] = len(train_probs_cache)
            stats['probability_cache_hit_rate'] = len(train_probs_cache) / len(labeled_inds) * 100 if len(labeled_inds) > 0 else 0
        if use_ann and 'index' in locals():
            # Determine index type
            if hasattr(index, 'nprobe'):
                stats['knn_index_type'] = f'IndexIVFFlat (nprobe={index.nprobe})'
            else:
                stats['knn_index_type'] = 'IndexFlatL2 (exact)'
            # Add build method info
            if 'build_method' in locals():
                stats['knn_build_method'] = build_method
                stats['knn_index_size'] = index.ntotal
    
    if distances is not None:
        if type(distances) is list:
            stats['distances'] = distances
        else:
            stats['distances'] = [float(x) for x in distances.mean(axis=1)]

    return sampled_ind, stats
