#!/usr/bin/env python3
"""
CIFAR-10 Milvus Experiment: SemDeDup vs FairDeDup with Milvus Lite
- Compare Milvus Exact (FLAT) vs Milvus ANN (IVF_FLAT)
- Compare with/without cache (term reuse)
- Downstream classification evaluation
- ANN uses IVF_FLAT to align with FAISS experiment

Usage:
    python run_cifar10_milvus_experiment.py
    python run_cifar10_milvus_experiment.py --skip-downstream  # Skip downstream evaluation
"""

import os
import sys
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Milvus Lite imports
try:
    from pymilvus import MilvusClient
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    print("Error: pymilvus not installed. Run: pip install 'pymilvus>=2.4.0'")

# ============================================================================
# Configuration
# ============================================================================

# Get base directory (parent of experiments folder)
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG = {
    'dataset_name': 'CIFAR-10',
    'dataset_size': 50000,
    'emb_size': 512,
    'num_clusters': 10,
    'eps_list': [0.05, 0.10, 0.15],
    'embs_memory_loc': os.path.join(_BASE_DIR, 'embeddings', 'cifar10_embeddings.npy'),
    'save_folder': os.path.join(_BASE_DIR, 'output', 'cifar10_milvus'),
    
    # Milvus configs
    'db_path': os.path.join(_BASE_DIR, 'output', 'milvus_cifar10_experiment.db'),
    'k_neighbors': 100,  # for ANN search
    
    # Training config
    'batch_size': 128,
    'num_epochs': 20,
    'learning_rate': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# Utility Functions
# ============================================================================

def init_memmap_embs(path, size, dim):
    """Load embeddings as memory-mapped array."""
    return np.memmap(path, dtype='float32', mode='r', shape=(size, dim))


def run_clustering():
    """Run K-Means clustering on embeddings using FAISS."""
    import faiss
    
    print("\n" + "=" * 70)
    print("Step 1: K-Means Clustering (FAISS)")
    print("=" * 70)
    
    embs = init_memmap_embs(
        CONFIG['embs_memory_loc'],
        CONFIG['dataset_size'],
        CONFIG['emb_size']
    )
    
    embs_array = np.array(embs, dtype=np.float32)
    faiss.normalize_L2(embs_array)
    
    print(f"Dataset size: {CONFIG['dataset_size']:,}")
    print(f"Embedding dim: {CONFIG['emb_size']}")
    print(f"Num clusters: {CONFIG['num_clusters']}")
    
    start_time = time.time()
    kmeans = faiss.Kmeans(
        CONFIG['emb_size'],
        CONFIG['num_clusters'],
        niter=20,
        verbose=True,
        spherical=True,
        gpu=False
    )
    kmeans.train(embs_array)
    
    _, assignments = kmeans.index.search(embs_array, 1)
    assignments = assignments.flatten()
    
    clustering_time = time.time() - start_time
    print(f"\n✓ Clustering done in {clustering_time:.2f}s")
    
    sorted_clusters_path = os.path.join(CONFIG['save_folder'], 'sorted_clusters')
    os.makedirs(sorted_clusters_path, exist_ok=True)
    
    centroids = kmeans.centroids
    
    for cluster_id in range(CONFIG['num_clusters']):
        cluster_mask = assignments == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        cluster_embs = embs_array[cluster_indices]
        centroid = centroids[cluster_id:cluster_id+1]
        distances = np.dot(cluster_embs, centroid.T).flatten()
        
        sorted_order = np.argsort(-distances)
        sorted_indices = cluster_indices[sorted_order]
        sorted_distances = distances[sorted_order]
        
        cluster_data = np.column_stack([sorted_distances, sorted_indices])
        np.save(os.path.join(sorted_clusters_path, f"cluster_{cluster_id}.npy"), cluster_data)
    
    print(f"✓ Saved clusters to {sorted_clusters_path}")
    
    cluster_sizes = [np.sum(assignments == i) for i in range(CONFIG['num_clusters'])]
    print(f"\nCluster sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, avg={np.mean(cluster_sizes):.0f}")
    
    return clustering_time


# ============================================================================
# SemDeDup Milvus Processor
# ============================================================================

class MilvusSemDeDupProcessor:
    """SemDeDup using Milvus Lite."""
    
    def __init__(self, use_ann=False, use_cache=False, db_path=None):
        self.use_ann = use_ann
        self.use_cache = use_cache
        self.eps_list = CONFIG['eps_list']
        self.sorted_clusters_path = os.path.join(CONFIG['save_folder'], 'sorted_clusters')
        self.num_clusters = CONFIG['num_clusters']
        self.embs = init_memmap_embs(
            CONFIG['embs_memory_loc'],
            CONFIG['dataset_size'],
            CONFIG['emb_size']
        )
        self.k_neighbors = CONFIG['k_neighbors']
        
        # Use separate db for each configuration to avoid conflicts
        db_suffix = f"{'ann' if use_ann else 'exact'}_{'cache' if use_cache else 'nocache'}"
        self.db_path = db_path or f"./milvus_semdedup_{db_suffix}.db"
        
        # Clean up old db
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        
        self.client = MilvusClient(self.db_path)
        
        # Stats
        self.stats = {
            'index_build_calls': 0,
            'search_calls': 0,
            'total_clusters': 0,
        }
        
        # Debug flag
        self.verbose = True
    
    def get_term_reuse_stats(self):
        """Get term reuse statistics."""
        num_eps = len(self.eps_list)
        return {
            'actual': self.stats.copy(),
            'without_reuse': {
                'index_build_calls': self.stats['total_clusters'] * num_eps,
                'search_calls': self.stats['total_clusters'] * num_eps,
            },
            'saved': {
                'index_build_calls': self.stats['total_clusters'] * num_eps - self.stats['index_build_calls'],
                'search_calls': self.stats['total_clusters'] * num_eps - self.stats['search_calls'],
            }
        }
    
    def _get_index_params(self):
        """Get index parameters based on mode. Uses IVF_FLAT to align with FAISS."""
        if self.use_ann:
            return {
                "metric_type": "IP",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 100}  # Same as FAISS experiment
            }
        else:
            return {
                "metric_type": "IP",
                "index_type": "FLAT",
                "params": {}
            }
    
    def process_cluster(self, cluster_id: int, dataframes_dir: str) -> float:
        """Process a single cluster."""
        df_file_loc = os.path.join(dataframes_dir, f"cluster_{cluster_id}.pkl")
        
        cluster_i = np.load(
            os.path.join(self.sorted_clusters_path, f"cluster_{cluster_id}.npy"),
            allow_pickle=True
        )
        cluster_size = cluster_i.shape[0]
        
        self.stats['total_clusters'] += 1
        
        if cluster_size == 1:
            points_to_remove_df = pd.DataFrame()
            points_to_remove_df["indices"] = [0]
            for eps in self.eps_list:
                points_to_remove_df[f"eps={eps}"] = [False]
            with open(df_file_loc, "wb") as f:
                pickle.dump(points_to_remove_df, f)
            return 0.0
        
        cluster_ids = cluster_i[:, 1].astype("int32")
        cluster_reps = np.array(self.embs[cluster_ids], dtype=np.float32)
        
        # Normalize
        norms = np.linalg.norm(cluster_reps, axis=1, keepdims=True)
        cluster_reps_norm = cluster_reps / (norms + 1e-8)
        
        start_time = time.time()
        
        collection_name = f"semdedup_c{cluster_id}"
        
        if self.client.has_collection(collection_name):
            self.client.drop_collection(collection_name)
        
        if self.verbose:
            print(f"      [cluster {cluster_id}] size={cluster_size}, creating collection...", flush=True)
        
        # Create collection
        self.client.create_collection(
            collection_name=collection_name,
            dimension=CONFIG['emb_size'],
            metric_type="IP"
        )
        
        # Insert data
        data = [
            {"id": i, "vector": cluster_reps_norm[i].tolist()}
            for i in range(cluster_size)
        ]
        self.client.insert(collection_name=collection_name, data=data)
        
        if self.verbose:
            t1 = time.time() - start_time
            print(f"      [cluster {cluster_id}] insert done in {t1:.2f}s, searching...", flush=True)
        
        self.stats['index_build_calls'] += 1
        
        points_to_remove_df = pd.DataFrame()
        points_to_remove_df["indices"] = list(range(cluster_size))
        
        if self.use_cache:
            # CACHE MODE: Search once with k neighbors, apply all eps thresholds
            k = cluster_size if not self.use_ann else min(cluster_size, self.k_neighbors)
            
            results = self.client.search(
                collection_name=collection_name,
                data=cluster_reps_norm.tolist(),
                limit=k,
                output_fields=["id"]
            )
            self.stats['search_calls'] += 1
            
            # Compute max similarity for each point (excluding self)
            M = np.zeros(cluster_size, dtype=np.float32)
            for i, hits in enumerate(results):
                for hit in hits:
                    if hit['id'] != i:
                        M[i] = max(M[i], hit['distance'])
                        break  # Only need the max (first non-self)
            
            # Apply all eps thresholds (term reuse!)
            for eps in self.eps_list:
                threshold = 1 - eps
                to_remove = M > threshold
                points_to_remove_df[f"eps={eps}"] = to_remove
        else:
            # NO CACHE MODE: Separate search for each eps
            for eps in self.eps_list:
                k = cluster_size if not self.use_ann else min(cluster_size, self.k_neighbors)
                
                results = self.client.search(
                    collection_name=collection_name,
                    data=cluster_reps_norm.tolist(),
                    limit=k,
                    output_fields=["id"]
                )
                self.stats['search_calls'] += 1
                
                threshold = 1 - eps
                to_remove = np.zeros(cluster_size, dtype=bool)
                
                for i, hits in enumerate(results):
                    for hit in hits:
                        if hit['id'] != i and hit['distance'] > threshold:
                            to_remove[i] = True
                            break
                
                points_to_remove_df[f"eps={eps}"] = to_remove
        
        elapsed = time.time() - start_time
        
        self.client.drop_collection(collection_name)
        
        with open(df_file_loc, "wb") as f:
            pickle.dump(points_to_remove_df, f)
        
        return elapsed
    
    def run(self, output_subdir: str) -> Tuple[float, Dict]:
        """Run SemDeDup."""
        output_dir = os.path.join(CONFIG['save_folder'], output_subdir)
        dataframes_dir = os.path.join(output_dir, "dataframes")
        os.makedirs(dataframes_dir, exist_ok=True)
        
        total_time = 0
        for cluster_id in range(self.num_clusters):
            elapsed = self.process_cluster(cluster_id, dataframes_dir)
            total_time += elapsed
            print(f"    Cluster {cluster_id+1}/{self.num_clusters}: {elapsed:.2f}s (total: {total_time:.2f}s)", flush=True)
        
        # Count duplicates
        dup_counts = {}
        for eps in self.eps_list:
            total_dups = 0
            for cluster_id in range(self.num_clusters):
                df_file = os.path.join(dataframes_dir, f"cluster_{cluster_id}.pkl")
                with open(df_file, "rb") as f:
                    df = pickle.load(f)
                total_dups += df[f"eps={eps}"].sum()
            dup_counts[eps] = total_dups
        
        # Cleanup db
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        
        return total_time, dup_counts
    
    def cleanup(self):
        """Clean up resources."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)


# ============================================================================
# FairDeDup Milvus Processor
# ============================================================================

class MilvusFairDeDupProcessor:
    """FairDeDup using Milvus Lite."""
    
    def __init__(self, use_ann=False, use_cache=False, db_path=None):
        self.use_ann = use_ann
        self.use_cache = use_cache
        self.eps_list = CONFIG['eps_list']
        self.sorted_clusters_path = os.path.join(CONFIG['save_folder'], 'sorted_clusters')
        self.num_clusters = CONFIG['num_clusters']
        self.embs = init_memmap_embs(
            CONFIG['embs_memory_loc'],
            CONFIG['dataset_size'],
            CONFIG['emb_size']
        )
        self.k_neighbors = CONFIG['k_neighbors']
        
        db_suffix = f"{'ann' if use_ann else 'exact'}_{'cache' if use_cache else 'nocache'}"
        self.db_path = db_path or f"./milvus_fairdedup_{db_suffix}.db"
        
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        
        self.client = MilvusClient(self.db_path)
        
        self.stats = {
            'index_build_calls': 0,
            'search_calls': 0,
            'total_clusters': 0,
        }
    
    def get_term_reuse_stats(self):
        """Get term reuse statistics."""
        num_eps = len(self.eps_list)
        return {
            'actual': self.stats.copy(),
            'without_reuse': {
                'index_build_calls': self.stats['total_clusters'] * num_eps,
                'search_calls': self.stats['total_clusters'] * num_eps,
            },
            'saved': {
                'index_build_calls': self.stats['total_clusters'] * num_eps - self.stats['index_build_calls'],
                'search_calls': self.stats['total_clusters'] * num_eps - self.stats['search_calls'],
            }
        }
    
    def select_prototype(self, group_indices: List[int], cluster_reps: np.ndarray) -> int:
        """Select prototype (closest to group centroid)."""
        group_embs = cluster_reps[group_indices]
        centroid = np.mean(group_embs, axis=0, keepdims=True)
        
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
        group_embs_norm = group_embs / (np.linalg.norm(group_embs, axis=1, keepdims=True) + 1e-8)
        
        similarities = group_embs_norm @ centroid_norm.T
        best_idx = np.argmax(similarities)
        
        return group_indices[best_idx]
    
    def union_find_groups(self, n: int, pairs: List[Tuple[int, int]]) -> Dict[int, List[int]]:
        """Run Union-Find to get connected components."""
        parent = list(range(n))
        rank = [0] * n
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                if rank[px] < rank[py]:
                    px, py = py, px
                parent[py] = px
                if rank[px] == rank[py]:
                    rank[px] += 1
        
        for i, j in pairs:
            union(i, j)
        
        groups = {}
        for i in range(n):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)
        
        return groups
    
    def process_cluster(self, cluster_id: int, dataframes_dir: str) -> float:
        """Process a single cluster."""
        df_file_loc = os.path.join(dataframes_dir, f"cluster_{cluster_id}.pkl")
        
        cluster_i = np.load(
            os.path.join(self.sorted_clusters_path, f"cluster_{cluster_id}.npy"),
            allow_pickle=True
        )
        cluster_size = cluster_i.shape[0]
        
        self.stats['total_clusters'] += 1
        
        if cluster_size == 1:
            points_to_remove_df = pd.DataFrame()
            points_to_remove_df["indices"] = [0]
            for eps in self.eps_list:
                points_to_remove_df[f"eps={eps}"] = [False]
            with open(df_file_loc, "wb") as f:
                pickle.dump(points_to_remove_df, f)
            return 0.0
        
        cluster_ids = cluster_i[:, 1].astype("int32")
        cluster_reps = np.array(self.embs[cluster_ids], dtype=np.float32)
        
        norms = np.linalg.norm(cluster_reps, axis=1, keepdims=True)
        cluster_reps_norm = cluster_reps / (norms + 1e-8)
        
        start_time = time.time()
        
        collection_name = f"fairdedup_c{cluster_id}"
        
        if self.client.has_collection(collection_name):
            self.client.drop_collection(collection_name)
        
        self.client.create_collection(
            collection_name=collection_name,
            dimension=CONFIG['emb_size'],
            metric_type="IP"
        )
        
        data = [
            {"id": i, "vector": cluster_reps_norm[i].tolist()}
            for i in range(cluster_size)
        ]
        self.client.insert(collection_name=collection_name, data=data)
        
        self.stats['index_build_calls'] += 1
        
        points_to_remove_df = pd.DataFrame()
        points_to_remove_df["indices"] = list(range(cluster_size))
        
        if self.use_cache:
            # CACHE MODE: Search once, apply all eps thresholds
            k = cluster_size if not self.use_ann else min(cluster_size, self.k_neighbors)
            
            results = self.client.search(
                collection_name=collection_name,
                data=cluster_reps_norm.tolist(),
                limit=k,
                output_fields=["id"]
            )
            self.stats['search_calls'] += 1
            
            # Collect ALL pairs with distances
            all_pairs_with_dist = []
            for i, hits in enumerate(results):
                for hit in hits:
                    if hit['id'] != i:
                        pair = (min(i, hit['id']), max(i, hit['id']), hit['distance'])
                        all_pairs_with_dist.append(pair)
            
            # Remove duplicate pairs
            all_pairs_with_dist = list(set(all_pairs_with_dist))
            
            # Apply each eps threshold (term reuse!)
            for eps in self.eps_list:
                threshold = 1 - eps
                valid_pairs = [(i, j) for i, j, d in all_pairs_with_dist if d > threshold]
                
                groups = self.union_find_groups(cluster_size, valid_pairs)
                
                to_remove = np.zeros(cluster_size, dtype=bool)
                for group in groups.values():
                    if len(group) > 1:
                        prototype_idx = self.select_prototype(group, cluster_reps)
                        for idx in group:
                            if idx != prototype_idx:
                                to_remove[idx] = True
                
                points_to_remove_df[f"eps={eps}"] = to_remove
        else:
            # NO CACHE MODE: Separate search for each eps
            for eps in self.eps_list:
                k = cluster_size if not self.use_ann else min(cluster_size, self.k_neighbors)
                
                results = self.client.search(
                    collection_name=collection_name,
                    data=cluster_reps_norm.tolist(),
                    limit=k,
                    output_fields=["id"]
                )
                self.stats['search_calls'] += 1
                
                threshold = 1 - eps
                
                valid_pairs = set()
                for i, hits in enumerate(results):
                    for hit in hits:
                        if hit['id'] != i and hit['distance'] > threshold:
                            valid_pairs.add((min(i, hit['id']), max(i, hit['id'])))
                
                groups = self.union_find_groups(cluster_size, list(valid_pairs))
                
                to_remove = np.zeros(cluster_size, dtype=bool)
                for group in groups.values():
                    if len(group) > 1:
                        prototype_idx = self.select_prototype(group, cluster_reps)
                        for idx in group:
                            if idx != prototype_idx:
                                to_remove[idx] = True
                
                points_to_remove_df[f"eps={eps}"] = to_remove
        
        elapsed = time.time() - start_time
        
        self.client.drop_collection(collection_name)
        
        with open(df_file_loc, "wb") as f:
            pickle.dump(points_to_remove_df, f)
        
        return elapsed
    
    def run(self, output_subdir: str) -> Tuple[float, Dict]:
        """Run FairDeDup."""
        output_dir = os.path.join(CONFIG['save_folder'], output_subdir)
        dataframes_dir = os.path.join(output_dir, "dataframes")
        os.makedirs(dataframes_dir, exist_ok=True)
        
        total_time = 0
        for cluster_id in range(self.num_clusters):
            elapsed = self.process_cluster(cluster_id, dataframes_dir)
            total_time += elapsed
            print(f"    Cluster {cluster_id+1}/{self.num_clusters}: {elapsed:.2f}s (total: {total_time:.2f}s)", flush=True)
        
        dup_counts = {}
        for eps in self.eps_list:
            total_dups = 0
            for cluster_id in range(self.num_clusters):
                df_file = os.path.join(dataframes_dir, f"cluster_{cluster_id}.pkl")
                with open(df_file, "rb") as f:
                    df = pickle.load(f)
                total_dups += df[f"eps={eps}"].sum()
            dup_counts[eps] = total_dups
        
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        
        return total_time, dup_counts
    
    def cleanup(self):
        """Clean up resources."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)


# ============================================================================
# Downstream Evaluation
# ============================================================================

def extract_kept_indices(output_dir, eps):
    """Extract indices of samples to keep."""
    kept_indices = []
    sorted_clusters_path = os.path.join(CONFIG['save_folder'], 'sorted_clusters')
    
    for cluster_id in range(CONFIG['num_clusters']):
        cluster_data = np.load(
            os.path.join(sorted_clusters_path, f"cluster_{cluster_id}.npy"),
            allow_pickle=True
        )
        cluster_indices = cluster_data[:, 1].astype(int)
        
        df_file = os.path.join(output_dir, f"cluster_{cluster_id}.pkl")
        with open(df_file, 'rb') as f:
            df = pickle.load(f)
        
        remove_mask = df[f'eps={eps}'].values
        keep_mask = ~remove_mask
        
        kept_cluster_indices = cluster_indices[keep_mask]
        kept_indices.extend(kept_cluster_indices.tolist())
    
    return sorted(kept_indices)


class SimpleResNet(nn.Module):
    """Simple ResNet for CIFAR-10."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def train_and_evaluate(train_indices, name):
    """Train model on subset and evaluate on test set."""
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    train_subset = Subset(train_dataset, train_indices)
    
    train_loader = DataLoader(train_subset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)
    
    model = SimpleResNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=CONFIG['learning_rate'], 
                          momentum=CONFIG['momentum'], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['num_epochs'])
    
    start_time = time.time()
    for epoch in range(CONFIG['num_epochs']):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
    
    train_time = time.time() - start_time
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = correct / total
    
    return accuracy, train_time


def run_downstream_evaluation(results):
    """Run downstream classification evaluation."""
    print("\n" + "=" * 80)
    print("Downstream Evaluation (CIFAR-10 Classification)")
    print("=" * 80)
    print(f"Training for {CONFIG['num_epochs']} epochs each")
    print(f"Device: {device}")
    
    eval_results = {}
    
    # Baseline: Full dataset
    print("\n[Baseline] Full training set", flush=True)
    full_indices = list(range(CONFIG['dataset_size']))
    acc, train_time = train_and_evaluate(full_indices, "Full")
    eval_results['Full'] = {'samples': len(full_indices), 'accuracy': acc, 'train_time': train_time}
    print(f"  Samples: {len(full_indices):,}, Accuracy: {acc*100:.2f}%, Time: {train_time:.1f}s", flush=True)
    
    # Evaluate all configurations (or filter to unique dedup results)
    # Note: Exact and ANN produce same duplicates, so we evaluate all for completeness
    methods_to_eval = [k for k in results.keys() if k != 'downstream']
    
    for name in methods_to_eval:
        if name not in results:
            continue
        
        result = results[name]
        output_dir = os.path.join(CONFIG['save_folder'], name.lower(), "dataframes")
        
        if not os.path.exists(output_dir):
            print(f"\n[{name}] Output directory not found, skipping", flush=True)
            continue
        
        print(f"\n[{name}]", flush=True)
        
        for eps in CONFIG['eps_list']:
            kept_indices = extract_kept_indices(output_dir, eps)
            
            if len(kept_indices) < 100:
                print(f"  eps={eps}: Too few samples ({len(kept_indices)}), skipping", flush=True)
                continue
            
            acc, train_time = train_and_evaluate(kept_indices, f"{name}_eps{eps}")
            key = f"{name}_eps{eps}"
            eval_results[key] = {'samples': len(kept_indices), 'accuracy': acc, 'train_time': train_time}
            
            print(f"  eps={eps}: {len(kept_indices):,} samples, Accuracy: {acc*100:.2f}%, Time: {train_time:.1f}s", flush=True)
    
    return eval_results


# ============================================================================
# Main Experiment
# ============================================================================

def run_milvus_experiment(skip_downstream=False, only_downstream=False):
    """Run full Milvus experiment comparing all configurations.
    
    Args:
        skip_downstream: Skip downstream evaluation after dedup
        only_downstream: Skip dedup, only run downstream using saved results
    """
    
    if not MILVUS_AVAILABLE:
        print("Error: pymilvus not installed. Run: pip install 'pymilvus>=2.4.0'")
        sys.exit(1)
    
    print("=" * 80)
    print("CIFAR-10 Milvus Experiment: SemDeDup vs FairDeDup")
    print("=" * 80)
    print(f"Dataset: {CONFIG['dataset_name']}")
    print(f"Dataset size: {CONFIG['dataset_size']:,}")
    print(f"Embedding dim: {CONFIG['emb_size']}")
    print(f"Num clusters: {CONFIG['num_clusters']}")
    print(f"Epsilon values: {CONFIG['eps_list']}")
    print(f"K neighbors (ANN): {CONFIG['k_neighbors']}")
    print("=" * 80)
    
    os.makedirs(CONFIG['save_folder'], exist_ok=True)
    
    # Check if clustering is done
    sorted_clusters_path = os.path.join(CONFIG['save_folder'], 'sorted_clusters')
    if not os.path.exists(sorted_clusters_path):
        print("\nClustering not found, running clustering first...")
        run_clustering()
    else:
        print(f"\n✓ Using existing clusters from {sorted_clusters_path}")
    
    results = {}
    
    # If only_downstream, load existing results and skip to downstream
    if only_downstream:
        results_file = os.path.join(CONFIG['save_folder'], 'milvus_results.pkl')
        if not os.path.exists(results_file):
            print(f"Error: No existing results found at {results_file}")
            print("Run the full experiment first without --only-downstream")
            sys.exit(1)
        
        print(f"\n✓ Loading existing results from {results_file}")
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
        print(f"  Loaded {len([k for k in results.keys() if not k.startswith('downstream')])} configurations")
        
        # Jump directly to downstream evaluation
        eval_results = run_downstream_evaluation(results)
        
        # Print downstream summary
        print("\n" + "=" * 80)
        print("Downstream Accuracy Summary")
        print("=" * 80)
        print(f"{'Configuration':<40} {'Samples':>10} {'Accuracy':>10}")
        print("-" * 60)
        for key, val in eval_results.items():
            print(f"{key:<40} {val['samples']:>10,} {val['accuracy']*100:>9.2f}%")
        print("-" * 60)
        
        # Save with eval results
        results['downstream'] = eval_results
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"\n✓ Results with downstream evaluation saved to {results_file}")
        
        return results
    # ========== SemDeDup Experiments ==========
    print("\n" + "=" * 80)
    print("SemDeDup with Milvus Lite")
    print("=" * 80)
    
    configurations_semdedup = [
        ("SemDeDup_Milvus_Exact_NoCache", False, False),  # Baseline
        ("SemDeDup_Milvus_ANN_Cache", True, True),        # Optimized
    ]
    
    for name, use_ann, use_cache in configurations_semdedup:
        print(f"\n[{name}]")
        print(f"  Index: {'IVF_FLAT (ANN)' if use_ann else 'FLAT (Exact)'}")
        print(f"  Cache: {'Enabled' if use_cache else 'Disabled'}")
        
        processor = MilvusSemDeDupProcessor(use_ann=use_ann, use_cache=use_cache)
        
        start = time.time()
        total_time, dup_counts = processor.run(name.lower())
        wall_time = time.time() - start
        
        stats = processor.get_term_reuse_stats()
        
        results[name] = {
            'time': total_time,
            'wall_time': wall_time,
            'duplicates': dup_counts,
            'stats': stats,
        }
        
        print(f"  Time: {total_time:.2f}s (wall: {wall_time:.2f}s)")
        print(f"  Search calls: {stats['actual']['search_calls']}")
        for eps, count in dup_counts.items():
            pct = count / CONFIG['dataset_size'] * 100
            print(f"    eps={eps}: {count:,} duplicates ({pct:.1f}%)")
        
        processor.cleanup()
    
    # ========== FairDeDup Experiments ==========
    print("\n" + "=" * 80)
    print("FairDeDup with Milvus Lite")
    print("=" * 80)
    
    configurations_fairdedup = [
        ("FairDeDup_Milvus_Exact_NoCache", False, False),  # Baseline
        ("FairDeDup_Milvus_ANN_Cache", True, True),        # Optimized
    ]
    
    for name, use_ann, use_cache in configurations_fairdedup:
        print(f"\n[{name}]")
        print(f"  Index: {'IVF_FLAT (ANN)' if use_ann else 'FLAT (Exact)'}")
        print(f"  Cache: {'Enabled' if use_cache else 'Disabled'}")
        
        processor = MilvusFairDeDupProcessor(use_ann=use_ann, use_cache=use_cache)
        
        start = time.time()
        total_time, dup_counts = processor.run(name.lower())
        wall_time = time.time() - start
        
        stats = processor.get_term_reuse_stats()
        
        results[name] = {
            'time': total_time,
            'wall_time': wall_time,
            'duplicates': dup_counts,
            'stats': stats,
        }
        
        print(f"  Time: {total_time:.2f}s (wall: {wall_time:.2f}s)")
        print(f"  Search calls: {stats['actual']['search_calls']}")
        for eps, count in dup_counts.items():
            pct = count / CONFIG['dataset_size'] * 100
            print(f"    eps={eps}: {count:,} duplicates ({pct:.1f}%)")
        
        processor.cleanup()
    
    # ========== Summary ==========
    print("\n" + "=" * 80)
    print("SUMMARY: Milvus Experiment Results")
    print("=" * 80)
    
    # Time comparison table
    print("\n### Timing Comparison (seconds)")
    print("-" * 70)
    print(f"{'Configuration':<40} {'Time':>10} {'Speedup':>10}")
    print("-" * 70)
    
    baseline_time = results.get('SemDeDup_Milvus_Exact_NoCache', {}).get('time', 1)
    
    for name in list(results.keys()):
        t = results[name]['time']
        speedup = baseline_time / t if t > 0 else float('inf')
        print(f"{name:<40} {t:>10.2f} {speedup:>10.1f}x")
    
    print("-" * 70)
    
    # Term reuse comparison
    print("\n### Term Reuse Statistics (Search Calls)")
    print("-" * 70)
    print(f"{'Configuration':<40} {'Actual':>10} {'Without':>10} {'Saved':>10}")
    print("-" * 70)
    
    for name in results.keys():
        stats = results[name]['stats']
        actual = stats['actual']['search_calls']
        without = stats['without_reuse']['search_calls']
        saved = stats['saved']['search_calls']
        print(f"{name:<40} {actual:>10} {without:>10} {saved:>10}")
    
    print("-" * 70)
    
    # Duplicate comparison
    print("\n### Duplicate Detection Comparison")
    print("-" * 70)
    
    for eps in CONFIG['eps_list']:
        print(f"\neps={eps}:")
        for name in results.keys():
            count = results[name]['duplicates'][eps]
            pct = count / CONFIG['dataset_size'] * 100
            print(f"  {name:<40} {count:>8,} ({pct:.1f}%)")
    
    print("\n" + "=" * 80)
    print("✓ Milvus Experiment Complete!")
    print("=" * 80)
    
    # Save results
    results_file = os.path.join(CONFIG['save_folder'], 'milvus_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n✓ Results saved to {results_file}")
    
    # Run downstream evaluation
    if not skip_downstream:
        eval_results = run_downstream_evaluation(results)
        
        # Print downstream summary
        print("\n" + "=" * 80)
        print("Downstream Accuracy Summary")
        print("=" * 80)
        print(f"{'Configuration':<40} {'Samples':>10} {'Accuracy':>10}")
        print("-" * 60)
        for key, val in eval_results.items():
            print(f"{key:<40} {val['samples']:>10,} {val['accuracy']*100:>9.2f}%")
        print("-" * 60)
        
        # Save with eval results
        results['downstream'] = eval_results
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"\n✓ Results with downstream evaluation saved to {results_file}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 Milvus Experiment")
    parser.add_argument('--skip-downstream', action='store_true',
                        help='Skip downstream classification evaluation')
    parser.add_argument('--only-downstream', action='store_true',
                        help='Skip dedup, only run downstream using saved results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--run-id', type=str, default='',
                        help='Unique identifier for this run (e.g., milvus_run1)')
    args = parser.parse_args()
    
    # Set random seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        print(f"Random seed set to: {args.seed}")
    
    # Update config with run-id
    if args.run_id:
        CONFIG['save_folder'] = os.path.join(_BASE_DIR, 'output', f'milvus_run_{args.run_id}')
        CONFIG['db_path'] = os.path.join(_BASE_DIR, 'output', f'milvus_{args.run_id}.db')
        print(f"Run ID: {args.run_id}")
        print(f"Output folder: {CONFIG['save_folder']}")
    
    run_milvus_experiment(skip_downstream=args.skip_downstream, 
                          only_downstream=args.only_downstream)
