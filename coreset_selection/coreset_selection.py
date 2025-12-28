"""
Core coreset selection algorithms.

Implements both baseline and optimized selection methods:
- Baseline: Exact KMeans (NumPy broadcast) + Recompute cosine distances
- Optimized: ANN KMeans (FAISS IVF) + Cache L2 distances (avoid redundant computation)

Based on experiments/run_comparison.py
"""

import time
import math
import numpy as np
import faiss
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_distances


def baseline_selection(vectors: np.ndarray, K: int, A: int, seed: int = 42, verbose: bool = True):
    """
    Baseline coreset selection: Exact KMeans (NumPy) + Recompute cosine distances.
    
    This is the un-optimized version that:
    - Uses NumPy broadcast for exact distance computation in KMeans
    - Recomputes cosine distances for easy/hard sample selection
    
    Args:
        vectors: (N, D) embedding matrix
        K: Number of clusters
        A: Target samples per cluster
        seed: Random seed
        verbose: Whether to show progress
        
    Returns:
        selected_indices: Array of selected sample indices
        timing_info: Dict with timing breakdown
    """
    N, dim = vectors.shape
    rng = np.random.default_rng(seed)
    
    if verbose:
        print(f"Baseline Selection: N={N}, K={K}, A={A}")
    
    timing = {"kmeans": 0, "selection": 0}
    
    # ============ Phase 1: Exact KMeans (NumPy broadcast) ============
    start_time = time.time()
    
    # Initialize centroids randomly
    selected_init = rng.choice(N, size=K, replace=False)
    centroids = vectors[selected_init].copy()
    
    tol = 1e-4
    max_iter = 50
    labels = None
    l2_all = None
    
    iterator = tqdm(range(max_iter), desc="KMeans (Exact)") if verbose else range(max_iter)
    
    for iter_idx in iterator:
        # Exact distance computation using NumPy broadcast (N x K matrix)
        dists = np.sum((vectors[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1).astype(np.int32)
        l2_all = dists[np.arange(N), labels]  # Squared L2 distances
        
        # Update centroids
        new_centroids = np.zeros_like(centroids)
        counts = np.bincount(labels, minlength=K).astype(np.float32)
        np.add.at(new_centroids, labels, vectors)
        
        non_empty = counts > 0
        if np.any(non_empty):
            new_centroids[non_empty] /= counts[non_empty][:, None]
        
        # Handle empty clusters
        empty = np.where(~non_empty)[0]
        if len(empty):
            new_centroids[empty] = vectors[rng.choice(N, len(empty))]
        
        # Check convergence
        diff = np.linalg.norm(new_centroids - centroids)
        if diff / max(np.linalg.norm(centroids), 1e-8) < tol:
            centroids = new_centroids
            # Final assignment
            dists = np.sum((vectors[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            labels = np.argmin(dists, axis=1).astype(np.int32)
            l2_all = dists[np.arange(N), labels]
            break
        centroids = new_centroids
    
    timing["kmeans"] = time.time() - start_time
    
    # ============ Phase 2: Coreset Selection (Recompute Cosine) ============
    start_time = time.time()
    
    Dc_indices = []
    iterator = tqdm(range(K), desc="Coreset (Recompute)") if verbose else range(K)
    
    for cluster_id in iterator:
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue
        
        # RECOMPUTE cosine distances (baseline behavior)
        cluster_vectors = vectors[cluster_indices]
        centroid = centroids[cluster_id].reshape(1, -1)
        dists = cosine_distances(cluster_vectors, centroid).reshape(-1)
        
        # Select easy (closest) and hard (farthest) samples
        num_easy = min(int(0.5 * A), len(dists))
        num_hard = min(int(0.5 * A), len(dists) - num_easy)
        
        if num_easy + num_hard > 0:
            sorted_indices = np.argsort(dists)
            selected = list(sorted_indices[:num_easy]) + list(sorted_indices[-num_hard:])
            Dc_indices.extend(cluster_indices[selected])
    
    timing["selection"] = time.time() - start_time
    timing["total"] = timing["kmeans"] + timing["selection"]
    
    selected_indices = np.unique(np.array(Dc_indices))
    
    if verbose:
        print(f"  KMeans time: {timing['kmeans']:.2f}s")
        print(f"  Selection time: {timing['selection']:.2f}s")
        print(f"  Total time: {timing['total']:.2f}s")
        print(f"  Selected: {len(selected_indices)} samples")
    
    return selected_indices, timing


def optimized_selection(vectors: np.ndarray, K: int, A: int, seed: int = 42, 
                        nlist: int = 256, nprobe: int = 128, verbose: bool = True):
    """
    Optimized coreset selection: ANN KMeans (FAISS IVF) + Cache L2 distances.
    
    Key optimizations:
    1. Use FAISS IVF index for approximate nearest neighbor in KMeans
    2. Cache L2 distances computed during KMeans
    3. Convert cached L2 to cosine via algebraic identity (no recomputation)
    
    Args:
        vectors: (N, D) embedding matrix
        K: Number of clusters
        A: Target samples per cluster
        seed: Random seed
        nlist: Number of cells for IVF index (will be capped based on K)
        nprobe: Number of cells to probe for search
        verbose: Whether to show progress
        
    Returns:
        selected_indices: Array of selected sample indices
        timing_info: Dict with timing breakdown
    """
    N, dim = vectors.shape
    rng = np.random.default_rng(seed)
    
    if verbose:
        print(f"Optimized Selection: N={N}, K={K}, A={A}")
    
    timing = {"kmeans": 0, "selection": 0}
    
    # ============ Phase 1: ANN KMeans with FAISS IVF ============
    start_time = time.time()
    
    # Initialize centroids
    selected_init = rng.choice(N, size=K, replace=False)
    centroids = vectors[selected_init].copy()
    
    # Build IVF index
    sqrtK = max(1, int(round(math.sqrt(K))))
    effective_nlist = min(nlist, sqrtK)
    
    quantizer = faiss.IndexFlatL2(dim)
    index_ivf = faiss.IndexIVFFlat(quantizer, dim, effective_nlist)
    
    # Train with random sample
    min_train = 39 * effective_nlist
    train_size = min(N, max(min_train, 10000))
    train_idx = rng.choice(N, size=train_size, replace=False)
    train_sample = vectors[train_idx].astype(np.float32)
    index_ivf.train(train_sample)
    
    # Set nprobe
    target_probe = max(1, int(0.5 * effective_nlist))
    index_ivf.nprobe = min(nprobe, effective_nlist, target_probe)
    
    if verbose:
        print(f"  FAISS config: nlist={effective_nlist}, nprobe={index_ivf.nprobe}")
    
    tol = 1e-4
    max_iter = 50
    labels = None
    l2_all = None
    
    iterator = tqdm(range(max_iter), desc="KMeans (ANN)") if verbose else range(max_iter)
    
    for iter_idx in iterator:
        # ANN search
        index_ivf.reset()
        index_ivf.add(centroids.astype(np.float32))
        D, I = index_ivf.search(vectors.astype(np.float32), 1)
        labels = I.ravel().astype(np.int32)
        l2_all = D.ravel()  # Squared L2 distances (FAISS returns squared)
        
        # Update centroids
        new_centroids = np.zeros_like(centroids)
        counts = np.bincount(labels, minlength=K).astype(np.float32)
        np.add.at(new_centroids, labels, vectors)
        
        non_empty = counts > 0
        if np.any(non_empty):
            new_centroids[non_empty] /= counts[non_empty][:, None]
        
        # Handle empty clusters
        empty = np.where(~non_empty)[0]
        if len(empty):
            new_centroids[empty] = vectors[rng.choice(N, len(empty))]
        
        # Check convergence
        diff = np.linalg.norm(new_centroids - centroids)
        if diff / max(np.linalg.norm(centroids), 1e-8) < tol:
            centroids = new_centroids
            # Final assignment with cached distances
            index_ivf.reset()
            index_ivf.add(centroids.astype(np.float32))
            D, I = index_ivf.search(vectors.astype(np.float32), 1)
            labels = I.ravel().astype(np.int32)
            l2_all = D.ravel()
            break
        centroids = new_centroids
    
    timing["kmeans"] = time.time() - start_time
    
    # ============ Phase 2: Coreset Selection (Reuse L2) ============
    start_time = time.time()
    
    # Precompute norms for cosine conversion
    centroid_norms_sq = np.sum(centroids ** 2, axis=1)
    vector_norms_sq = np.sum(vectors ** 2, axis=1)
    
    Dc_indices = []
    iterator = tqdm(range(K), desc="Coreset (Cache L2)") if verbose else range(K)
    
    for cluster_id in iterator:
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue
        
        # Reuse L2 distances to compute cosine (no recomputation!)
        # cos(a,b) = dot(a,b) / (||a|| * ||b||)
        # dot(a,b) = (||a||² + ||b||² - ||a-b||²) / 2
        l2_dist_sq = l2_all[cluster_indices]
        v_norms_sq = vector_norms_sq[cluster_indices]
        c_norm_sq = centroid_norms_sq[cluster_id]
        
        dot_products = 0.5 * (c_norm_sq + v_norms_sq - l2_dist_sq)
        v_norms = np.sqrt(v_norms_sq)
        c_norm = np.sqrt(c_norm_sq)
        
        denominator = v_norms * c_norm
        safe_denominator = np.maximum(denominator, 1e-12)
        cosine_sim = dot_products / safe_denominator
        cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
        cosine_dists = 1 - cosine_sim
        
        # Select easy (closest) and hard (farthest) samples
        num_easy = min(int(0.5 * A), len(cosine_dists))
        num_hard = min(int(0.5 * A), len(cosine_dists) - num_easy)
        
        if num_easy + num_hard > 0:
            sorted_indices = np.argsort(cosine_dists)
            selected = list(sorted_indices[:num_easy]) + list(sorted_indices[-num_hard:])
            Dc_indices.extend(cluster_indices[selected])
    
    timing["selection"] = time.time() - start_time
    timing["total"] = timing["kmeans"] + timing["selection"]
    
    selected_indices = np.unique(np.array(Dc_indices))
    
    if verbose:
        print(f"  KMeans time: {timing['kmeans']:.2f}s")
        print(f"  Selection time: {timing['selection']:.2f}s")
        print(f"  Total time: {timing['total']:.2f}s")
        print(f"  Selected: {len(selected_indices)} samples")
    
    return selected_indices, timing


def compute_recall(baseline_indices: np.ndarray, optimized_indices: np.ndarray) -> float:
    """Compute recall of optimized selection vs baseline."""
    baseline_set = set(baseline_indices)
    optimized_set = set(optimized_indices)
    
    if len(baseline_set) == 0:
        return 0.0
    
    overlap = len(baseline_set & optimized_set)
    return overlap / len(baseline_set)
