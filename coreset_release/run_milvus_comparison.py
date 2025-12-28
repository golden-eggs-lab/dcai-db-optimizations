"""
Milvus vector database backend comparison.

Demonstrates coreset selection using Milvus instead of FAISS:
1. Milvus Exact: FLAT index (exact search) + Recompute cosine
2. Milvus ANN: IVF_FLAT index (ANN search) + Cache L2

Usage:
    python run_milvus_comparison.py
    
Requirements:
    pip install pymilvus
"""

import json
import time
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_distances

try:
    from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
except ImportError:
    print("Please install pymilvus: pip install pymilvus")
    exit(1)

from config import CACHE_DIR, ARTIFACTS_DIR, SELECTION_CONFIG


class MilvusManager:
    """Manage Milvus collections for coreset selection."""
    
    def __init__(self, db_path: str = "milvus_coreset.db"):
        self.client = MilvusClient(uri=db_path)
    
    def create_collection(self, name: str, dim: int, use_ann: bool = False, nlist: int = 16):
        """Create a collection with specified index type."""
        if self.client.has_collection(name):
            self.client.drop_collection(name)
        
        schema = CollectionSchema([
            FieldSchema("id", DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema("vec", DataType.FLOAT_VECTOR, dim=dim),
        ])
        self.client.create_collection(collection_name=name, schema=schema)
        
        # Create index
        index_params = self.client.prepare_index_params()
        if use_ann:
            index_params.add_index(
                field_name="vec",
                index_type="IVF_FLAT",
                metric_type="L2",
                params={"nlist": nlist}
            )
        else:
            index_params.add_index(
                field_name="vec",
                index_type="FLAT",
                metric_type="L2",
                params={}
            )
        self.client.create_index(collection_name=name, index_params=index_params)
    
    def update_centroids(self, name: str, centroids: np.ndarray):
        """Update centroids in collection."""
        self.client.delete(collection_name=name, filter="id >= 0")
        payload = [{"id": i, "vec": c.tolist()} for i, c in enumerate(centroids)]
        self.client.insert(name, payload)
    
    def search(self, name: str, vectors: np.ndarray, nprobe: int = 8, batch_size: int = 2048):
        """Batched search to avoid message size limits."""
        N = vectors.shape[0]
        labels = np.empty(N, dtype=np.int32)
        dists = np.empty(N, dtype=np.float32)
        
        for i in range(0, N, batch_size):
            chunk = vectors[i:i+batch_size]
            res = self.client.search(
                collection_name=name,
                data=chunk.tolist(),
                limit=1,
                output_fields=["id"],
                search_params={"metric_type": "L2", "params": {"nprobe": nprobe}},
            )
            labels[i:i+len(res)] = [r[0]["id"] for r in res]
            dists[i:i+len(res)] = [r[0]["distance"] for r in res]
        
        return labels, dists
    
    def close(self):
        try:
            self.client.close()
        except:
            pass


def milvus_coreset_selection(
    vectors: np.ndarray,
    K: int,
    A: int,
    use_ann: bool = False,
    reuse_l2: bool = False,
    seed: int = 42,
    nlist: int = 16,
    nprobe: int = 8,
    verbose: bool = True,
):
    """
    Coreset selection using Milvus vector database.
    
    Args:
        vectors: (N, D) embeddings
        K: Number of clusters
        A: Samples per cluster
        use_ann: Use IVF_FLAT (ANN) vs FLAT (exact)
        reuse_l2: Reuse L2 distances for cosine
        seed: Random seed
    """
    N, dim = vectors.shape
    rng = np.random.default_rng(seed)
    
    mode = f"Milvus {'ANN' if use_ann else 'Exact'} + {'Cache L2' if reuse_l2 else 'Recompute'}"
    if verbose:
        print(f"\n{'='*50}")
        print(f"{mode}")
        print(f"N={N}, K={K}, A={A}")
    
    start_time = time.time()
    
    # Initialize Milvus
    db_path = str(ARTIFACTS_DIR / "milvus_temp.db")
    milvus = MilvusManager(db_path)
    collection_name = "centroids"
    milvus.create_collection(collection_name, dim, use_ann=use_ann, nlist=nlist)
    
    # Initialize centroids
    init_idx = rng.choice(N, size=K, replace=False)
    centroids = vectors[init_idx].copy()
    
    tol = 1e-4
    l2_all = None
    iterator = tqdm(range(50), desc=f"KMeans ({mode})") if verbose else range(50)
    
    for _ in iterator:
        milvus.update_centroids(collection_name, centroids)
        labels, D = milvus.search(collection_name, vectors.astype(np.float32), nprobe=nprobe)
        l2_all = np.sqrt(D)
        
        new_centroids = np.zeros_like(centroids)
        counts = np.bincount(labels, minlength=K).astype(np.float32)
        np.add.at(new_centroids, labels, vectors)
        
        non_empty = counts > 0
        if np.any(non_empty):
            new_centroids[non_empty] /= counts[non_empty][:, None]
        
        empty = np.where(~non_empty)[0]
        if len(empty):
            new_centroids[empty] = vectors[rng.choice(N, len(empty))]
        
        diff = np.linalg.norm(new_centroids - centroids)
        if diff / max(np.linalg.norm(centroids), 1e-8) < tol:
            centroids = new_centroids
            milvus.update_centroids(collection_name, centroids)
            labels, D = milvus.search(collection_name, vectors.astype(np.float32), nprobe=nprobe)
            l2_all = np.sqrt(D)
            break
        centroids = new_centroids
    
    kmeans_time = time.time() - start_time
    
    # Coreset selection
    start_time = time.time()
    
    if reuse_l2:
        centroid_norms_sq = np.sum(centroids ** 2, axis=1)
        vector_norms_sq = np.sum(vectors ** 2, axis=1)
    
    Dc_indices = []
    iterator = tqdm(range(K), desc="Coreset Selection") if verbose else range(K)
    
    for cluster_id in iterator:
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue
        
        if reuse_l2:
            # Convert cached L2 to cosine
            l2_dists = l2_all[cluster_indices]
            v_norms = np.sqrt(vector_norms_sq[cluster_indices])
            c_norm = np.sqrt(centroid_norms_sq[cluster_id])
            dot = (vector_norms_sq[cluster_indices] + centroid_norms_sq[cluster_id] - l2_dists**2) / 2
            cosine_dists = 1 - dot / (v_norms * c_norm + 1e-8)
        else:
            # Recompute cosine
            cluster_vectors = vectors[cluster_indices]
            centroid = centroids[cluster_id].reshape(1, -1)
            cosine_dists = cosine_distances(cluster_vectors, centroid).reshape(-1)
        
        num_easy = min(int(0.5 * A), len(cosine_dists))
        num_hard = min(int(0.5 * A), len(cosine_dists) - num_easy)
        
        if num_easy + num_hard > 0:
            sorted_idx = np.argsort(cosine_dists)
            selected = list(sorted_idx[:num_easy]) + list(sorted_idx[-num_hard:])
            Dc_indices.extend(cluster_indices[selected])
    
    selection_time = time.time() - start_time
    
    milvus.close()
    
    selected_indices = np.unique(np.array(Dc_indices))
    total_time = kmeans_time + selection_time
    
    if verbose:
        print(f"KMeans time:    {kmeans_time:.2f}s")
        print(f"Selection time: {selection_time:.2f}s")
        print(f"Total time:     {total_time:.2f}s")
        print(f"Selected:       {len(selected_indices)} samples")
    
    return selected_indices, {
        "kmeans_time": kmeans_time,
        "selection_time": selection_time,
        "total_time": total_time,
        "n_selected": len(selected_indices),
    }


def main():
    print("="*60)
    print("MILVUS VECTOR DATABASE COMPARISON")
    print("="*60)
    
    # Load embeddings
    vectors = np.load(CACHE_DIR / "coedit_embeddings.npy").astype(np.float32)
    print(f"Loaded embeddings: {vectors.shape}")
    
    K = SELECTION_CONFIG["K"]
    A = SELECTION_CONFIG["A"]
    seed = SELECTION_CONFIG["seed"]
    
    # Run all 4 configurations
    configs = [
        ("Milvus Exact + Recompute", False, False),
        ("Milvus Exact + Cache L2", False, True),
        ("Milvus ANN + Recompute", True, False),
        ("Milvus ANN + Cache L2", True, True),
    ]
    
    results = {}
    baseline_indices = None
    
    for name, use_ann, reuse_l2 in configs:
        indices, timing = milvus_coreset_selection(
            vectors, K, A,
            use_ann=use_ann,
            reuse_l2=reuse_l2,
            seed=seed,
        )
        
        results[name] = timing
        
        if baseline_indices is None:
            baseline_indices = set(indices)
        else:
            recall = len(set(indices) & baseline_indices) / len(baseline_indices)
            results[name]["recall"] = recall
    
    # Summary
    print("\n" + "="*70)
    print("MILVUS COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Config':<35} {'Total':<10} {'Speedup':<10}")
    print("-"*70)
    
    baseline_time = results["Milvus Exact + Recompute"]["total_time"]
    for name, timing in results.items():
        speedup = baseline_time / timing["total_time"]
        print(f"{name:<35} {timing['total_time']:<10.2f} {speedup:<10.2f}x")
    
    # Save results
    output_dir = ARTIFACTS_DIR / "milvus_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
