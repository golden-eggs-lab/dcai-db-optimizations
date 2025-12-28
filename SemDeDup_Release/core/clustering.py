"""
Clustering utilities for semantic deduplication.

Uses FAISS K-Means for efficient clustering of embeddings.
"""

import os
import time
import numpy as np
import faiss
from typing import Tuple, Optional


def run_kmeans_clustering(
    embeddings: np.ndarray,
    num_clusters: int,
    output_dir: str,
    niter: int = 20,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run K-Means clustering on embeddings.
    
    Args:
        embeddings: (N, D) array of embeddings
        num_clusters: Number of clusters
        output_dir: Directory to save cluster assignments
        niter: Number of K-Means iterations
        seed: Random seed
        verbose: Print progress
    
    Returns:
        labels: (N,) cluster assignments
        centroids: (K, D) cluster centroids
        elapsed_time: Clustering time in seconds
    """
    n, d = embeddings.shape
    
    if verbose:
        print(f"Clustering {n:,} points in {d}D to {num_clusters} clusters")
    
    # Normalize embeddings for cosine similarity
    embs = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(embs)
    
    # Set random seed
    np.random.seed(seed)
    
    start_time = time.time()
    
    # Run K-Means
    kmeans = faiss.Kmeans(
        d,
        num_clusters,
        niter=niter,
        verbose=verbose,
        spherical=True,
        gpu=False,
        seed=seed,
    )
    kmeans.train(embs)
    
    # Assign points to clusters
    _, labels = kmeans.index.search(embs, 1)
    labels = labels.flatten()
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"Clustering completed in {elapsed:.2f}s")
    
    # Save clusters
    sorted_clusters_dir = os.path.join(output_dir, 'sorted_clusters')
    os.makedirs(sorted_clusters_dir, exist_ok=True)
    
    for cluster_id in range(num_clusters):
        cluster_mask = labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        # Sort by distance to centroid
        cluster_embs = embs[cluster_indices]
        centroid = kmeans.centroids[cluster_id:cluster_id+1]
        distances = np.dot(cluster_embs, centroid.T).flatten()
        sorted_order = np.argsort(-distances)  # Descending (closest first)
        
        sorted_indices = cluster_indices[sorted_order]
        
        # Save as (rank, global_index) pairs
        cluster_data = np.column_stack([
            np.arange(len(sorted_indices)),
            sorted_indices
        ])
        
        cluster_path = os.path.join(sorted_clusters_dir, f'cluster_{cluster_id}.npy')
        np.save(cluster_path, cluster_data)
    
    if verbose:
        sizes = [np.sum(labels == i) for i in range(num_clusters)]
        print(f"Cluster sizes: min={min(sizes)}, max={max(sizes)}, avg={np.mean(sizes):.0f}")
    
    return labels, kmeans.centroids, elapsed


def load_clusters(
    clusters_dir: str,
    num_clusters: int,
) -> list:
    """
    Load cluster assignments from disk.
    
    Args:
        clusters_dir: Path to sorted_clusters directory
        num_clusters: Number of clusters
    
    Returns:
        List of (N_i, 2) arrays with (rank, global_index) for each cluster
    """
    clusters = []
    for i in range(num_clusters):
        path = os.path.join(clusters_dir, f'cluster_{i}.npy')
        cluster = np.load(path, allow_pickle=True)
        clusters.append(cluster)
    return clusters
