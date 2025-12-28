#!/usr/bin/env python3
"""
Complete CIFAR-10 Experiment: SemDeDup vs FairDeDup
- Compare FAISS Exact vs FAISS ANN (fair comparison, same platform)
- Compare with/without cache (term reuse)
- Downstream classification evaluation

Based on run_semdedup_optimized.py --compare design:
- FAISS Exact: IndexFlatIP (O(n²))
- FAISS ANN: IndexIVFFlat (O(n log n))
"""

import os
import sys
import time
import pickle
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import faiss

# ============================================================================
# Configuration
# ============================================================================

# Get base directory (parent of experiments folder)
import os as _os
_BASE_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

CONFIG = {
    'dataset_name': 'CIFAR-10',
    'dataset_size': 50000,
    'emb_size': 512,
    'num_clusters': 10,
    'eps_list': [0.05, 0.10, 0.15],
    'embs_memory_loc': _os.path.join(_BASE_DIR, 'embeddings', 'cifar10_embeddings.npy'),
    'save_folder': _os.path.join(_BASE_DIR, 'output', 'cifar10_full'),
    
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


# ============================================================================
# Step 1: Clustering
# ============================================================================

def run_clustering():
    """Run K-Means clustering on embeddings."""
    print("\n" + "=" * 70)
    print("Step 1: K-Means Clustering")
    print("=" * 70)
    
    embs = init_memmap_embs(
        CONFIG['embs_memory_loc'],
        CONFIG['dataset_size'],
        CONFIG['emb_size']
    )
    
    # Load all embeddings to memory
    embs_array = np.array(embs, dtype=np.float32)
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embs_array)
    
    print(f"Dataset size: {CONFIG['dataset_size']:,}")
    print(f"Embedding dim: {CONFIG['emb_size']}")
    print(f"Num clusters: {CONFIG['num_clusters']}")
    
    # K-Means clustering
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
    
    # Assign points to clusters
    _, assignments = kmeans.index.search(embs_array, 1)
    assignments = assignments.flatten()
    
    clustering_time = time.time() - start_time
    print(f"\n✓ Clustering done in {clustering_time:.2f}s")
    
    # Save clusters
    clusters_path = os.path.join(CONFIG['save_folder'], 'clusters')
    sorted_clusters_path = os.path.join(CONFIG['save_folder'], 'sorted_clusters')
    os.makedirs(clusters_path, exist_ok=True)
    os.makedirs(sorted_clusters_path, exist_ok=True)
    
    # Compute distances to centroids for sorting
    centroids = kmeans.centroids
    
    for cluster_id in range(CONFIG['num_clusters']):
        cluster_mask = assignments == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        # Compute distance to centroid
        cluster_embs = embs_array[cluster_indices]
        centroid = centroids[cluster_id:cluster_id+1]
        distances = np.dot(cluster_embs, centroid.T).flatten()
        
        # Sort by distance (descending - closest first)
        sorted_order = np.argsort(-distances)
        sorted_indices = cluster_indices[sorted_order]
        sorted_distances = distances[sorted_order]
        
        # Save as [distance, original_index]
        cluster_data = np.column_stack([sorted_distances, sorted_indices])
        np.save(os.path.join(sorted_clusters_path, f"cluster_{cluster_id}.npy"), cluster_data)
    
    print(f"✓ Saved clusters to {sorted_clusters_path}")
    
    # Print cluster sizes
    cluster_sizes = [np.sum(assignments == i) for i in range(CONFIG['num_clusters'])]
    print(f"\nCluster sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, avg={np.mean(cluster_sizes):.0f}")
    
    return clustering_time


# ============================================================================
# Step 2: Deduplication Processors
# ============================================================================

class SemDeDupProcessor:
    """SemDeDup using FAISS (same as run_semdedup_optimized.py --compare)."""
    
    def __init__(self, use_ann=False, use_cache=False):
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
        self.ann_k = 100      # top-k for ANN mode
        self.ann_nprobe = 10  # nprobe for IVF
        
        # Check if GPU FAISS is available
        self.use_gpu_faiss = torch.cuda.is_available() and hasattr(faiss, 'index_cpu_to_gpu')
        
        # Term reuse statistics
        self.stats = {
            'normalize_calls': 0,
            'index_build_calls': 0,
            'search_calls': 0,
            'total_clusters': 0,
        }
        
        # Timing breakdown (cumulative across all clusters)
        self.timing = {
            'load': 0.0,
            'normalize': 0.0,
            'index_build': 0.0,
            'search': 0.0,
            'apply_threshold': 0.0,
            'total': 0.0,
        }
    
    def get_timing_breakdown(self):
        """Get timing breakdown for each component."""
        total = self.timing['total'] if self.timing['total'] > 0 else 1e-9
        search_calls = self.stats['search_calls']
        
        return {
            'load': {'time': self.timing['load'], 'pct': self.timing['load']/total*100},
            'normalize': {'time': self.timing['normalize'], 'pct': self.timing['normalize']/total*100},
            'index_build': {
                'time': self.timing['index_build'],
                'pct': self.timing['index_build']/total*100,
                'calls': self.stats['index_build_calls'],
                'per_call': self.timing['index_build']/self.stats['index_build_calls'] if self.stats['index_build_calls'] > 0 else 0,
            },
            'search': {
                'time': self.timing['search'],
                'pct': self.timing['search']/total*100,
                'calls': search_calls,
                'per_call': self.timing['search']/search_calls if search_calls > 0 else 0,
            },
            'apply_threshold': {'time': self.timing['apply_threshold'], 'pct': self.timing['apply_threshold']/total*100},
            'total': total,
        }
    
    def get_term_reuse_stats(self):
        """Get term reuse statistics."""
        num_eps = len(self.eps_list)
        return {
            'actual': self.stats.copy(),
            'without_reuse': {
                'normalize_calls': self.stats['total_clusters'] * num_eps,
                'index_build_calls': self.stats['total_clusters'] * num_eps,
                'search_calls': self.stats['total_clusters'] * num_eps,
            },
            'saved': {
                'normalize_calls': self.stats['total_clusters'] * num_eps - self.stats['normalize_calls'],
                'index_build_calls': self.stats['total_clusters'] * num_eps - self.stats['index_build_calls'],
                'search_calls': self.stats['total_clusters'] * num_eps - self.stats['search_calls'],
            }
        }
    
    def semdedup_exact_faiss(self, cluster_embs):
        """
        FAISS Exact: IndexFlatIP for exact nearest neighbor search.
        Complexity: O(n²)
        Returns: (M, index_build_time, search_time)
        """
        n, d = cluster_embs.shape
        
        # Build exact search index
        t0 = time.time()
        index = faiss.IndexFlatIP(d)
        
        # Move to GPU if available
        if self.use_gpu_faiss:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        
        index.add(cluster_embs)
        self.stats['index_build_calls'] += 1
        index_build_time = time.time() - t0
        
        # Search all neighbors (k=n for exact)
        t0 = time.time()
        k = n
        similarities, indices = index.search(cluster_embs, k)
        self.stats['search_calls'] += 1
        search_time = time.time() - t0
        
        # Vectorized: Get max similarity excluding self
        # Mask out self-similarity
        self_mask = (indices == np.arange(n)[:, None])
        similarities_masked = np.where(self_mask, -np.inf, similarities)
        M = np.max(similarities_masked, axis=1)
        
        return M, index_build_time, search_time
    
    def semdedup_ann_faiss(self, cluster_embs):
        """
        FAISS ANN: IndexIVFFlat for approximate nearest neighbor search.
        Complexity: O(n * nprobe * n/nlist) ≈ O(n log n)
        Returns: (M, index_build_time, search_time)
        """
        n, d = cluster_embs.shape
        
        # For small clusters, fall back to exact
        if n <= self.ann_k:
            return self.semdedup_exact_faiss(cluster_embs)
        
        # Build IVF index
        t0 = time.time()
        nlist = max(1, min(n // 10, 100))
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.nprobe = min(self.ann_nprobe, nlist)
        
        # Move to GPU if available
        if self.use_gpu_faiss:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        
        # Train and add
        index.train(cluster_embs)
        index.add(cluster_embs)
        self.stats['index_build_calls'] += 1
        index_build_time = time.time() - t0
        
        # Search top-k neighbors
        t0 = time.time()
        k = min(self.ann_k, n)
        similarities, indices = index.search(cluster_embs, k)
        self.stats['search_calls'] += 1
        search_time = time.time() - t0
        
        # Vectorized: Get max similarity excluding self
        self_mask = (indices == np.arange(n)[:, None])
        similarities_masked = np.where(self_mask, -np.inf, similarities)
        M = np.max(similarities_masked, axis=1)
        
        return M, index_build_time, search_time
    
    def process_cluster(self, cluster_id, output_dir):
        """Process a single cluster using FAISS."""
        df_file = os.path.join(output_dir, f"cluster_{cluster_id}.pkl")
        
        # Load cluster data
        t0 = time.time()
        cluster_data = np.load(
            os.path.join(self.sorted_clusters_path, f"cluster_{cluster_id}.npy"),
            allow_pickle=True
        )
        cluster_size = cluster_data.shape[0]
        
        self.stats['total_clusters'] += 1
        
        if cluster_size == 1:
            df = pd.DataFrame({'indices': [0]})
            for eps in self.eps_list:
                df[f'eps={eps}'] = [False]
            with open(df_file, 'wb') as f:
                pickle.dump(df, f)
            return 0.0
        
        cluster_indices = cluster_data[:, 1].astype(int)
        cluster_embs = np.array(self.embs[cluster_indices], dtype=np.float32)
        load_time = time.time() - t0
        self.timing['load'] += load_time
        
        start_time = time.time()
        
        # Normalize for cosine similarity (FAISS IP on normalized = cosine)
        t0 = time.time()
        faiss.normalize_L2(cluster_embs)
        self.stats['normalize_calls'] += 1
        normalize_time = time.time() - t0
        self.timing['normalize'] += normalize_time
        
        df = pd.DataFrame({'indices': list(range(cluster_size))})
        
        if self.use_cache:
            # Cache optimization: compute M once, reuse for all eps
            if self.use_ann:
                M, idx_time, search_time = self.semdedup_ann_faiss(cluster_embs)
            else:
                M, idx_time, search_time = self.semdedup_exact_faiss(cluster_embs)
            
            self.timing['index_build'] += idx_time
            self.timing['search'] += search_time
            
            # Apply thresholds (reuse M for all eps)
            t0 = time.time()
            for eps in self.eps_list:
                df[f'eps={eps}'] = M > (1 - eps)
            self.timing['apply_threshold'] += time.time() - t0
        else:
            # No cache: compute separately for each eps
            for eps in self.eps_list:
                if self.use_ann:
                    M, idx_time, search_time = self.semdedup_ann_faiss(cluster_embs.copy())
                else:
                    M, idx_time, search_time = self.semdedup_exact_faiss(cluster_embs.copy())
                
                self.timing['index_build'] += idx_time
                self.timing['search'] += search_time
                
                t0 = time.time()
                df[f'eps={eps}'] = M > (1 - eps)
                self.timing['apply_threshold'] += time.time() - t0
        
        elapsed = time.time() - start_time
        self.timing['total'] += load_time + elapsed
        
        with open(df_file, 'wb') as f:
            pickle.dump(df, f)
        
        return elapsed


class FairDeDupProcessor:
    """FairDeDup using FAISS (same platform as SemDeDup for fair comparison)."""
    
    def __init__(self, use_ann=False, use_cache=False):
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
        self.ann_k = 100      # top-k for ANN mode
        self.ann_nprobe = 10  # nprobe for IVF
        
        # Check if GPU FAISS is available
        self.use_gpu_faiss = torch.cuda.is_available() and hasattr(faiss, 'index_cpu_to_gpu')
        
        # Term reuse statistics
        self.stats = {
            'normalize_calls': 0,
            'index_build_calls': 0,
            'search_calls': 0,
            'total_clusters': 0,
        }
        
        # Timing breakdown (cumulative across all clusters)
        self.timing = {
            'load': 0.0,
            'normalize': 0.0,
            'index_build': 0.0,
            'search': 0.0,
            'apply_threshold': 0.0,  # includes union-find + prototype selection
            'total': 0.0,
        }
    
    def get_timing_breakdown(self):
        """Get timing breakdown for each component."""
        total = self.timing['total'] if self.timing['total'] > 0 else 1e-9
        search_calls = self.stats['search_calls']
        
        return {
            'load': {'time': self.timing['load'], 'pct': self.timing['load']/total*100},
            'normalize': {'time': self.timing['normalize'], 'pct': self.timing['normalize']/total*100},
            'index_build': {
                'time': self.timing['index_build'],
                'pct': self.timing['index_build']/total*100,
                'calls': self.stats['index_build_calls'],
                'per_call': self.timing['index_build']/self.stats['index_build_calls'] if self.stats['index_build_calls'] > 0 else 0,
            },
            'search': {
                'time': self.timing['search'],
                'pct': self.timing['search']/total*100,
                'calls': search_calls,
                'per_call': self.timing['search']/search_calls if search_calls > 0 else 0,
            },
            'apply_threshold': {'time': self.timing['apply_threshold'], 'pct': self.timing['apply_threshold']/total*100},
            'total': total,
        }
    
    def get_term_reuse_stats(self):
        """Get term reuse statistics."""
        num_eps = len(self.eps_list)
        return {
            'actual': self.stats.copy(),
            'without_reuse': {
                'normalize_calls': self.stats['total_clusters'] * num_eps,
                'index_build_calls': self.stats['total_clusters'] * num_eps,
                'search_calls': self.stats['total_clusters'] * num_eps,
            },
            'saved': {
                'normalize_calls': self.stats['total_clusters'] * num_eps - self.stats['normalize_calls'],
                'index_build_calls': self.stats['total_clusters'] * num_eps - self.stats['index_build_calls'],
                'search_calls': self.stats['total_clusters'] * num_eps - self.stats['search_calls'],
            }
        }
    
    def select_prototype(self, group, cluster_embs):
        """Select prototype (closest to group centroid)."""
        group_embs = cluster_embs[group]
        centroid = np.mean(group_embs, axis=0, keepdims=True)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        
        sims = np.dot(group_embs, centroid.T).flatten()
        return group[np.argmax(sims)]
    
    def find_groups(self, cluster_size, pairs):
        """Union-Find to get connected components."""
        parent = list(range(cluster_size))
        rank = [0] * cluster_size
        
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
        for i in range(cluster_size):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)
        
        return list(groups.values())
    
    def get_all_pairs_exact(self, cluster_embs):
        """Get all pairs with similarities using FAISS Exact.
        Returns: (all_pairs, index_build_time, search_time)
        """
        n, d = cluster_embs.shape
        
        # Build exact search index
        t0 = time.time()
        index = faiss.IndexFlatIP(d)
        if self.use_gpu_faiss:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        
        index.add(cluster_embs)
        self.stats['index_build_calls'] += 1
        index_build_time = time.time() - t0
        
        # Search all neighbors
        t0 = time.time()
        similarities, indices = index.search(cluster_embs, n)
        self.stats['search_calls'] += 1
        search_time = time.time() - t0
        
        # Vectorized: Collect all pairs (i < j) with their similarities
        row_idx = np.repeat(np.arange(n), n)
        col_idx = indices.flatten()
        sims = similarities.flatten()
        
        # Keep only i < j pairs
        mask = row_idx < col_idx
        all_pairs = list(zip(row_idx[mask], col_idx[mask], sims[mask]))
        
        return all_pairs, index_build_time, search_time
    
    def get_all_pairs_ann(self, cluster_embs):
        """Get approximate pairs with similarities using FAISS ANN.
        Returns: (all_pairs, index_build_time, search_time)
        """
        n, d = cluster_embs.shape
        
        # For small clusters, fall back to exact
        if n <= self.ann_k:
            return self.get_all_pairs_exact(cluster_embs)
        
        # Build IVF index
        t0 = time.time()
        nlist = max(1, min(n // 10, 100))
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.nprobe = min(self.ann_nprobe, nlist)
        
        if self.use_gpu_faiss:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        
        index.train(cluster_embs)
        index.add(cluster_embs)
        self.stats['index_build_calls'] += 1
        index_build_time = time.time() - t0
        
        # Search top-k neighbors
        t0 = time.time()
        k = min(self.ann_k, n)
        similarities, indices = index.search(cluster_embs, k)
        self.stats['search_calls'] += 1
        search_time = time.time() - t0
        
        # Vectorized: Collect pairs (i, j) with their similarities
        row_idx = np.repeat(np.arange(n), k)
        col_idx = indices.flatten()
        sims = similarities.flatten()
        
        # Filter out self-pairs
        valid_mask = row_idx != col_idx
        row_idx = row_idx[valid_mask]
        col_idx = col_idx[valid_mask]
        sims = sims[valid_mask]
        
        # Ensure i < j for unique pairs
        min_idx = np.minimum(row_idx, col_idx)
        max_idx = np.maximum(row_idx, col_idx)
        
        # Use pandas to get max similarity per pair (faster than dict)
        import pandas as pd_temp
        df_pairs = pd_temp.DataFrame({'i': min_idx, 'j': max_idx, 'sim': sims})
        df_pairs = df_pairs.groupby(['i', 'j'])['sim'].max().reset_index()
        
        return list(zip(df_pairs['i'].values, df_pairs['j'].values, df_pairs['sim'].values)), index_build_time, search_time
    
    def process_cluster(self, cluster_id, output_dir):
        """Process a single cluster using FAISS."""
        df_file = os.path.join(output_dir, f"cluster_{cluster_id}.pkl")
        
        # Load cluster data
        t0 = time.time()
        cluster_data = np.load(
            os.path.join(self.sorted_clusters_path, f"cluster_{cluster_id}.npy"),
            allow_pickle=True
        )
        cluster_size = cluster_data.shape[0]
        
        self.stats['total_clusters'] += 1
        
        if cluster_size == 1:
            df = pd.DataFrame({'indices': [0]})
            for eps in self.eps_list:
                df[f'eps={eps}'] = [False]
            with open(df_file, 'wb') as f:
                pickle.dump(df, f)
            return 0.0
        
        cluster_indices = cluster_data[:, 1].astype(int)
        cluster_embs = np.array(self.embs[cluster_indices], dtype=np.float32)
        cluster_embs_orig = cluster_embs.copy()
        load_time = time.time() - t0
        self.timing['load'] += load_time
        
        start_time = time.time()
        
        # Normalize for cosine similarity
        t0 = time.time()
        faiss.normalize_L2(cluster_embs)
        self.stats['normalize_calls'] += 1
        normalize_time = time.time() - t0
        self.timing['normalize'] += normalize_time
        
        df = pd.DataFrame({'indices': list(range(cluster_size))})
        
        if self.use_cache:
            # Cache: get all pairs once, reuse for all eps
            if self.use_ann:
                all_pairs, idx_time, search_time = self.get_all_pairs_ann(cluster_embs)
            else:
                all_pairs, idx_time, search_time = self.get_all_pairs_exact(cluster_embs)
            
            self.timing['index_build'] += idx_time
            self.timing['search'] += search_time
            
            t0 = time.time()
            for eps in self.eps_list:
                threshold = 1 - eps
                valid_pairs = [(i, j) for i, j, sim in all_pairs if sim > threshold]
                groups = self.find_groups(cluster_size, valid_pairs)
                
                to_remove = np.zeros(cluster_size, dtype=bool)
                for group in groups:
                    if len(group) > 1:
                        prototype = self.select_prototype(group, cluster_embs_orig)
                        for idx in group:
                            if idx != prototype:
                                to_remove[idx] = True
                
                df[f'eps={eps}'] = to_remove
            self.timing['apply_threshold'] += time.time() - t0
        else:
            # No cache: search separately for each eps
            for eps in self.eps_list:
                threshold = 1 - eps
                
                if self.use_ann:
                    all_pairs, idx_time, search_time = self.get_all_pairs_ann(cluster_embs.copy())
                else:
                    all_pairs, idx_time, search_time = self.get_all_pairs_exact(cluster_embs.copy())
                
                self.timing['index_build'] += idx_time
                self.timing['search'] += search_time
                
                t0 = time.time()
                valid_pairs = [(i, j) for i, j, sim in all_pairs if sim > threshold]
                groups = self.find_groups(cluster_size, valid_pairs)
                
                to_remove = np.zeros(cluster_size, dtype=bool)
                for group in groups:
                    if len(group) > 1:
                        prototype = self.select_prototype(group, cluster_embs_orig)
                        for idx in group:
                            if idx != prototype:
                                to_remove[idx] = True
                
                df[f'eps={eps}'] = to_remove
                self.timing['apply_threshold'] += time.time() - t0
        
        elapsed = time.time() - start_time
        self.timing['total'] += load_time + elapsed
        
        with open(df_file, 'wb') as f:
            pickle.dump(df, f)
        
        return elapsed


# ============================================================================
# Step 2: Run Deduplication
# ============================================================================

def run_deduplication():
    """Run all deduplication experiments."""
    print("\n" + "=" * 70)
    print("Step 2: Deduplication Experiments")
    print("=" * 70)
    
    results = {}
    
    experiments = [
        # (name, algorithm, use_ann, use_cache)
        ("SemDeDup_Exact_NoCache", "semdedup", False, False),
        ("SemDeDup_Exact_Cache", "semdedup", False, True),
        ("SemDeDup_ANN_NoCache", "semdedup", True, False),
        ("SemDeDup_ANN_Cache", "semdedup", True, True),
        ("FairDeDup_Exact_NoCache", "fairdedup", False, False),
        ("FairDeDup_Exact_Cache", "fairdedup", False, True),
        ("FairDeDup_ANN_NoCache", "fairdedup", True, False),
        ("FairDeDup_ANN_Cache", "fairdedup", True, True),
    ]
    
    for name, algo, use_ann, use_cache in experiments:
        print(f"\n{'='*50}")
        print(f"Running: {name}")
        print(f"  Algorithm: {algo}, ANN: {use_ann}, Cache: {use_cache}")
        print(f"{'='*50}")
        
        output_dir = os.path.join(CONFIG['save_folder'], name, 'dataframes')
        os.makedirs(output_dir, exist_ok=True)
        
        if algo == "semdedup":
            processor = SemDeDupProcessor(use_ann=use_ann, use_cache=use_cache)
        else:
            processor = FairDeDupProcessor(use_ann=use_ann, use_cache=use_cache)
        
        total_time = 0
        for cluster_id in tqdm(range(CONFIG['num_clusters']), desc=name):
            elapsed = processor.process_cluster(cluster_id, output_dir)
            total_time += elapsed
        
        # Get term reuse statistics
        term_stats = processor.get_term_reuse_stats()
        
        # Get timing breakdown
        timing_breakdown = processor.get_timing_breakdown()
        
        # Count duplicates
        dup_counts = {eps: 0 for eps in CONFIG['eps_list']}
        for cluster_id in range(CONFIG['num_clusters']):
            df_file = os.path.join(output_dir, f"cluster_{cluster_id}.pkl")
            with open(df_file, 'rb') as f:
                df = pickle.load(f)
            for eps in CONFIG['eps_list']:
                dup_counts[eps] += df[f'eps={eps}'].sum()
        
        results[name] = {
            'time': total_time,
            'duplicates': dup_counts,
            'output_dir': output_dir,
            'term_reuse_stats': term_stats,
            'timing_breakdown': timing_breakdown,
        }
        
        print(f"\n  Time: {total_time:.2f}s")
        for eps in CONFIG['eps_list']:
            pct = dup_counts[eps] / CONFIG['dataset_size'] * 100
            kept = CONFIG['dataset_size'] - dup_counts[eps]
            print(f"  eps={eps}: {dup_counts[eps]:,} removed, {kept:,} kept ({100-pct:.1f}%)")
        
        # Print term reuse statistics
        saved = term_stats['saved']
        actual = term_stats['actual']
        print(f"  Operations: normalize={actual['normalize_calls']}, index_build={actual['index_build_calls']}, search={actual['search_calls']}")
        print(f"  Saved: normalize={saved['normalize_calls']}, index_build={saved['index_build_calls']}, search={saved['search_calls']}")
        
        # Print timing breakdown
        tb = timing_breakdown
        print(f"\n  Timing Breakdown:")
        print(f"    Load:            {tb['load']['time']:>7.3f}s ({tb['load']['pct']:>5.1f}%)")
        print(f"    Normalize:       {tb['normalize']['time']:>7.3f}s ({tb['normalize']['pct']:>5.1f}%)")
        print(f"    Index Build:     {tb['index_build']['time']:>7.3f}s ({tb['index_build']['pct']:>5.1f}%)  [{tb['index_build']['calls']} calls, {tb['index_build']['per_call']:.4f}s/call]")
        print(f"    Search:          {tb['search']['time']:>7.3f}s ({tb['search']['pct']:>5.1f}%)  [{tb['search']['calls']} calls, {tb['search']['per_call']:.4f}s/call]")
        print(f"    Apply Threshold: {tb['apply_threshold']['time']:>7.3f}s ({tb['apply_threshold']['pct']:>5.1f}%)")
        print(f"    Total:           {tb['total']:>7.3f}s")
    
    return results


# ============================================================================
# Step 3: Extract Kept Indices
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


# ============================================================================
# Step 4: Downstream Evaluation
# ============================================================================

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
    
    # Data transforms
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
    
    # Load datasets
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    # Create subset
    train_subset = Subset(train_dataset, train_indices)
    
    train_loader = DataLoader(train_subset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)
    
    # Model
    model = SimpleResNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=CONFIG['learning_rate'], 
                          momentum=CONFIG['momentum'], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['num_epochs'])
    
    # Training
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
    
    # Evaluation
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


def run_downstream_evaluation(dedup_results):
    """Run downstream classification evaluation."""
    print("\n" + "=" * 70)
    print("Step 3: Downstream Evaluation (CIFAR-10 Classification)")
    print("=" * 70)
    print(f"Training for {CONFIG['num_epochs']} epochs each")
    print(f"Device: {device}")
    
    eval_results = {}
    
    # Baseline: Full dataset
    print("\n[Baseline] Full training set")
    full_indices = list(range(CONFIG['dataset_size']))
    acc, train_time = train_and_evaluate(full_indices, "Full")
    eval_results['Full'] = {'samples': len(full_indices), 'accuracy': acc, 'train_time': train_time}
    print(f"  Samples: {len(full_indices):,}, Accuracy: {acc*100:.2f}%, Time: {train_time:.1f}s")
    
    # For each dedup method
    for name, result in dedup_results.items():
        print(f"\n[{name}]")
        
        for eps in CONFIG['eps_list']:
            kept_indices = extract_kept_indices(result['output_dir'], eps)
            
            if len(kept_indices) < 100:
                print(f"  eps={eps}: Too few samples ({len(kept_indices)}), skipping")
                continue
            
            acc, train_time = train_and_evaluate(kept_indices, f"{name}_eps{eps}")
            key = f"{name}_eps{eps}"
            eval_results[key] = {'samples': len(kept_indices), 'accuracy': acc, 'train_time': train_time}
            
            print(f"  eps={eps}: {len(kept_indices):,} samples, Accuracy: {acc*100:.2f}%, Time: {train_time:.1f}s")
    
    return eval_results


# ============================================================================
# Summary
# ============================================================================

def print_summary(dedup_results, eval_results):
    """Print final summary."""
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    
    # Deduplication time comparison
    print("\n### Deduplication Time Comparison ###")
    print(f"{'Method':<30} {'Time (s)':<12} {'Speedup':<10}")
    print("-" * 52)
    
    baseline_time = dedup_results.get('SemDeDup_Exact_NoCache', {}).get('time', 1)
    for name, result in dedup_results.items():
        speedup = baseline_time / result['time'] if result['time'] > 0 else 0
        print(f"{name:<30} {result['time']:<12.3f} {speedup:<10.2f}x")
    
    # Duplicates per eps
    print("\n### Duplicates Removed per Epsilon ###")
    header = f"{'Method':<30}"
    for eps in CONFIG['eps_list']:
        header += f" {'eps='+str(eps):>12}"
    print(header)
    print("-" * (30 + 13 * len(CONFIG['eps_list'])))
    
    for name, result in dedup_results.items():
        row = f"{name:<30}"
        for eps in CONFIG['eps_list']:
            dups = result['duplicates'].get(eps, 0)
            row += f" {dups:>12,}"
        print(row)
    
    # Term reuse savings
    print("\n### Term Reuse Savings ###")
    print(f"{'Method':<30} {'Normalize':<12} {'Index':<12} {'Search':<12}")
    print("-" * 66)
    
    for name, result in dedup_results.items():
        stats = result.get('term_reuse_stats', {}).get('saved', {})
        print(f"{name:<30} {stats.get('normalize_calls', 0):<12} {stats.get('index_build_calls', 0):<12} {stats.get('search_calls', 0):<12}")
    
    # Downstream accuracy
    print("\n### Downstream Accuracy Comparison ###")
    print(f"{'Method':<35} {'Samples':<10} {'Accuracy':<10} {'Train Time':<12}")
    print("-" * 67)
    
    for name, result in eval_results.items():
        print(f"{name:<35} {result['samples']:<10,} {result['accuracy']*100:<10.2f}% {result['train_time']:<12.1f}s")
    
    # Analysis
    print("\n### Analysis ###")
    full_acc = eval_results.get('Full', {}).get('accuracy', 0)
    full_time = eval_results.get('Full', {}).get('train_time', 0)
    
    # Analyze key methods for each eps
    for eps in CONFIG['eps_list']:
        print(f"\n--- eps={eps} ---")
        for method in ['SemDeDup_Exact_NoCache', 'SemDeDup_ANN_Cache', 
                       'FairDeDup_Exact_NoCache', 'FairDeDup_ANN_Cache']:
            key = f"{method}_eps{eps}"
            if key in eval_results:
                r = eval_results[key]
                data_reduction = (1 - r['samples'] / CONFIG['dataset_size']) * 100
                acc_diff = (r['accuracy'] - full_acc) * 100
                time_saved = (1 - r['train_time'] / full_time) * 100 if full_time > 0 else 0
                print(f"{method}:")
                print(f"  Data reduction: {data_reduction:.1f}%, Acc change: {acc_diff:+.2f}%, Time saved: {time_saved:.1f}%")
    
    # ============================================================================
    # Table Data: KNN to ANN & Terms Caching Breakdown
    # ============================================================================
    print("\n" + "=" * 80)
    print("TABLE DATA: Optimization Breakdown for SemDeDup")
    print("=" * 80)
    
    # Get baseline (Exact_NoCache) data
    exact_nocache = dedup_results.get('SemDeDup_Exact_NoCache', {})
    exact_cache = dedup_results.get('SemDeDup_Exact_Cache', {})
    ann_nocache = dedup_results.get('SemDeDup_ANN_NoCache', {})
    ann_cache = dedup_results.get('SemDeDup_ANN_Cache', {})
    
    if exact_nocache and ann_cache:
        tb_exact_nocache = exact_nocache.get('timing_breakdown', {})
        tb_exact_cache = exact_cache.get('timing_breakdown', {})
        tb_ann_nocache = ann_nocache.get('timing_breakdown', {})
        tb_ann_cache = ann_cache.get('timing_breakdown', {})
        
        print("\n### KNN to ANN (Search Time per Call) ###")
        print(f"{'Configuration':<30} {'Search/call (s)':<18} {'Search calls':<15} {'Total Search (s)':<18}")
        print("-" * 81)
        for name, tb in [('Exact_NoCache', tb_exact_nocache), ('Exact_Cache', tb_exact_cache),
                         ('ANN_NoCache', tb_ann_nocache), ('ANN_Cache', tb_ann_cache)]:
            if 'search' in tb:
                s = tb['search']
                print(f"SemDeDup_{name:<20} {s.get('per_call', 0):>15.4f}   {s.get('calls', 0):>12}      {s.get('time', 0):>15.4f}")
        
        # Calculate speedup for table
        exact_search_per_call = tb_exact_nocache.get('search', {}).get('per_call', 1)
        ann_search_per_call = tb_ann_nocache.get('search', {}).get('per_call', 1)
        ann_speedup = exact_search_per_call / ann_search_per_call if ann_search_per_call > 0 else 0
        
        # Calculate recall (comparing duplicates found)
        exact_dups = exact_nocache.get('duplicates', {}).get(0.1, 0)  # use eps=0.1
        ann_dups = ann_nocache.get('duplicates', {}).get(0.1, 0)
        recall = ann_dups / exact_dups * 100 if exact_dups > 0 else 100
        
        print(f"\n  → ANN Speedup (per call): {ann_speedup:.1f}x")
        print(f"  → ANN Recall (eps=0.1): {recall:.1f}%")
        
        print("\n### Terms Caching (Search Calls Reduction) ###")
        print(f"{'Configuration':<30} {'Search calls':<15} {'Total Search Time (s)':<22}")
        print("-" * 67)
        nocache_calls = tb_exact_nocache.get('search', {}).get('calls', 0)
        cache_calls = tb_exact_cache.get('search', {}).get('calls', 0)
        nocache_time = tb_exact_nocache.get('search', {}).get('time', 0)
        cache_time = tb_exact_cache.get('search', {}).get('time', 0)
        
        print(f"{'SemDeDup_Exact_NoCache':<30} {nocache_calls:<15} {nocache_time:<22.4f}")
        print(f"{'SemDeDup_Exact_Cache':<30} {cache_calls:<15} {cache_time:<22.4f}")
        
        cache_speedup = nocache_time / cache_time if cache_time > 0 else 0
        computation_reduced = (1 - cache_calls / nocache_calls) * 100 if nocache_calls > 0 else 0
        
        print(f"\n  → Cache Speedup (search time): {cache_speedup:.1f}x")
        print(f"  → Computation Reduced: {computation_reduced:.0f}% ({nocache_calls} → {cache_calls} calls)")
        
        print("\n### Summary Table Data ###")
        print("=" * 80)
        print(f"{'Optimization':<20} {'Time (baseline)':<18} {'Time (optimized)':<18} {'Metric':<25}")
        print("-" * 80)
        print(f"{'KNN → ANN':<20} {exact_search_per_call:<18.4f} {ann_search_per_call:<18.4f} Recall={recall:.1f}%")
        print(f"{'Terms Caching':<20} {nocache_time:<18.4f} {cache_time:<18.4f} Reduced={computation_reduced:.0f}%")
        print("=" * 80)
    
    print("\n" + "=" * 70)
    print("✓ Experiment complete!")
    print("=" * 70)


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("CIFAR-10 Full Experiment: SemDeDup vs FairDeDup")
    print("=" * 70)
    print(f"Dataset: {CONFIG['dataset_name']} ({CONFIG['dataset_size']:,} samples)")
    print(f"Embedding: CLIP ViT-B/32 ({CONFIG['emb_size']}-dim)")
    print(f"Clusters: {CONFIG['num_clusters']}")
    print(f"Epsilon values: {CONFIG['eps_list']}")
    print(f"Device: {device}")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(CONFIG['save_folder'], exist_ok=True)
    
    # Step 1: Clustering
    clustering_time = run_clustering()
    
    # Step 2: Deduplication
    dedup_results = run_deduplication()
    
    # Step 3: Downstream evaluation
    eval_results = run_downstream_evaluation(dedup_results)
    
    # Summary
    print_summary(dedup_results, eval_results)
    
    # Save results
    results = {
        'config': CONFIG,
        'clustering_time': clustering_time,
        'dedup_results': dedup_results,
        'eval_results': eval_results,
    }
    
    results_file = os.path.join(CONFIG['save_folder'], 'experiment_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 FAISS Experiment")
    parser.add_argument('--run-id', type=str, default=None,
                        help='Run identifier (default: timestamp). Used to distinguish multiple runs.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Set run_id (use timestamp if not provided)
    if args.run_id is None:
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    else:
        run_id = args.run_id
    
    # Update save folder with run_id
    CONFIG['save_folder'] = _os.path.join(_BASE_DIR, 'output', f'run_{run_id}')
    CONFIG['run_id'] = run_id
    
    # Set seed if provided
    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        CONFIG['seed'] = args.seed
        print(f"Random seed set to: {args.seed}")
    
    print(f"Run ID: {run_id}")
    print(f"Output folder: {CONFIG['save_folder']}")
    
    main()
