"""
Ablation study with timing breakdown.

Tests 4 configurations to isolate optimization contributions:
1. Baseline: Exact KMeans + Recompute cosine
2. ANN Only: ANN KMeans + Recompute cosine
3. Cache Only: Exact KMeans + Cache L2
4. ANN + Cache: ANN KMeans + Cache L2 (full optimized)

Usage:
    python run_ablation.py
"""

import json
import time
import math
import numpy as np
import faiss
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_distances

from config import CACHE_DIR, ARTIFACTS_DIR, SELECTION_CONFIG, FAISS_CONFIG


def exact_kmeans(vectors, K, seed=42, verbose=True):
    """Exact KMeans clustering."""
    N, dim = vectors.shape
    rng = np.random.default_rng(seed)
    
    selected_init = rng.choice(N, size=K, replace=False)
    centroids = vectors[selected_init].copy()
    
    tol = 1e-4
    iterator = tqdm(range(50), desc="KMeans (Exact)") if verbose else range(50)
    
    for _ in iterator:
        dists = np.linalg.norm(vectors[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        
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
            dists = np.linalg.norm(vectors[:, None, :] - centroids[None, :, :], axis=2)
            labels = np.argmin(dists, axis=1)
            break
        centroids = new_centroids
    
    # Compute L2 distances for potential caching
    l2_dists = np.min(dists, axis=1)
    
    return labels, centroids, l2_dists


def ann_kmeans(vectors, K, seed=42, nlist=256, nprobe=128, verbose=True):
    """ANN KMeans using FAISS IVF."""
    N, dim = vectors.shape
    rng = np.random.default_rng(seed)
    
    sqrtK = max(1, int(round(math.sqrt(K))))
    effective_nlist = min(nlist, sqrtK)
    
    quantizer = faiss.IndexFlatL2(dim)
    index_ivf = faiss.IndexIVFFlat(quantizer, dim, effective_nlist)
    
    min_train = 39 * effective_nlist
    train_size = min(N, max(min_train, 10000))
    train_idx = rng.choice(N, size=train_size, replace=False)
    train_sample = vectors[train_idx].astype(np.float32)
    index_ivf.train(train_sample)
    index_ivf.nprobe = max(1, int(0.5 * effective_nlist))
    
    selected_init = rng.choice(N, size=K, replace=False)
    centroids = vectors[selected_init].copy()
    
    tol = 1e-4
    iterator = tqdm(range(50), desc="KMeans (ANN)") if verbose else range(50)
    l2_all = None
    
    for _ in iterator:
        index_ivf.reset()
        index_ivf.add(centroids.astype(np.float32))
        D, I = index_ivf.search(vectors.astype(np.float32), 1)
        labels = I.ravel().astype(np.int32)
        l2_all = np.sqrt(D.ravel())
        
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
            index_ivf.reset()
            index_ivf.add(centroids.astype(np.float32))
            D, I = index_ivf.search(vectors.astype(np.float32), 1)
            labels = I.ravel().astype(np.int32)
            l2_all = np.sqrt(D.ravel())
            break
        centroids = new_centroids
    
    return labels, centroids, l2_all


def recompute_cosine_selection(vectors, labels, centroids, K, A, verbose=True):
    """Coreset selection by recomputing cosine distances."""
    Dc_indices = []
    iterator = tqdm(range(K), desc="Coreset (Recompute)") if verbose else range(K)
    
    for cluster_id in iterator:
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue
        
        cluster_vectors = vectors[cluster_indices]
        centroid = centroids[cluster_id].reshape(1, -1)
        dists = cosine_distances(cluster_vectors, centroid).reshape(-1)
        
        num_easy = min(int(0.5 * A), len(dists))
        num_hard = min(int(0.5 * A), len(dists) - num_easy)
        
        if num_easy + num_hard > 0:
            sorted_indices = np.argsort(dists)
            selected = list(sorted_indices[:num_easy]) + list(sorted_indices[-num_hard:])
            Dc_indices.extend(cluster_indices[selected])
    
    return np.unique(np.array(Dc_indices))


def cache_l2_selection(vectors, labels, centroids, l2_all, K, A, verbose=True):
    """Coreset selection using cached L2 distances."""
    centroid_norms_sq = np.sum(centroids ** 2, axis=1)
    vector_norms_sq = np.sum(vectors ** 2, axis=1)
    
    Dc_indices = []
    iterator = tqdm(range(K), desc="Coreset (Cache L2)") if verbose else range(K)
    
    for cluster_id in iterator:
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue
        
        l2_dists = l2_all[cluster_indices]
        v_norms = np.sqrt(vector_norms_sq[cluster_indices])
        c_norm = np.sqrt(centroid_norms_sq[cluster_id])
        
        dot_products = (vector_norms_sq[cluster_indices] + centroid_norms_sq[cluster_id] - l2_dists ** 2) / 2
        cosine_sim = dot_products / (v_norms * c_norm + 1e-8)
        cosine_dists = 1 - cosine_sim
        
        num_easy = min(int(0.5 * A), len(cosine_dists))
        num_hard = min(int(0.5 * A), len(cosine_dists) - num_easy)
        
        if num_easy + num_hard > 0:
            sorted_indices = np.argsort(cosine_dists)
            selected = list(sorted_indices[:num_easy]) + list(sorted_indices[-num_hard:])
            Dc_indices.extend(cluster_indices[selected])
    
    return np.unique(np.array(Dc_indices))


def run_config(name, kmeans_fn, selection_fn, vectors, K, A, verbose=True):
    """Run a single ablation configuration."""
    print(f"\n{'='*50}")
    print(f"Config: {name}")
    print(f"{'='*50}")
    
    # KMeans
    start = time.time()
    labels, centroids, l2_all = kmeans_fn(vectors, K, verbose=verbose)
    kmeans_time = time.time() - start
    
    # Selection
    start = time.time()
    selected_indices = selection_fn(vectors, labels, centroids, l2_all, K, A, verbose=verbose)
    selection_time = time.time() - start
    
    total_time = kmeans_time + selection_time
    
    print(f"KMeans time:    {kmeans_time:.2f}s")
    print(f"Selection time: {selection_time:.2f}s")
    print(f"Total time:     {total_time:.2f}s")
    print(f"Selected:       {len(selected_indices)} samples")
    
    return {
        "kmeans_time": kmeans_time,
        "selection_time": selection_time,
        "total_time": total_time,
        "n_selected": len(selected_indices),
    }, selected_indices


def main():
    print("="*60)
    print("ABLATION STUDY WITH TIMING BREAKDOWN")
    print("="*60)
    
    # Load embeddings
    vectors = np.load(CACHE_DIR / "coedit_embeddings.npy").astype(np.float32)
    print(f"Loaded embeddings: {vectors.shape}")
    
    N = vectors.shape[0]
    K = SELECTION_CONFIG["K"]
    A = SELECTION_CONFIG["A"]
    
    print(f"\nConfig: N={N}, K={K}, A={A}")
    
    # Define configurations
    configs = [
        ("Baseline (Exact + Recompute)", 
         lambda v, k, verbose: exact_kmeans(v, k, verbose=verbose),
         lambda v, l, c, d, k, a, verbose: recompute_cosine_selection(v, l, c, k, a, verbose)),
        
        ("ANN Only (ANN + Recompute)", 
         lambda v, k, verbose: ann_kmeans(v, k, verbose=verbose),
         lambda v, l, c, d, k, a, verbose: recompute_cosine_selection(v, l, c, k, a, verbose)),
        
        ("Cache Only (Exact + Cache)", 
         lambda v, k, verbose: exact_kmeans(v, k, verbose=verbose),
         lambda v, l, c, d, k, a, verbose: cache_l2_selection(v, l, c, d, k, a, verbose)),
        
        ("ANN + Cache (Full Optimized)", 
         lambda v, k, verbose: ann_kmeans(v, k, verbose=verbose),
         lambda v, l, c, d, k, a, verbose: cache_l2_selection(v, l, c, d, k, a, verbose)),
    ]
    
    results = {}
    baseline_indices = None
    
    for name, kmeans_fn, selection_fn in configs:
        timing, indices = run_config(name, kmeans_fn, selection_fn, vectors, K, A)
        results[name] = timing
        
        if baseline_indices is None:
            baseline_indices = set(indices)
        else:
            recall = len(set(indices) & baseline_indices) / len(baseline_indices)
            results[name]["recall"] = recall
            print(f"Recall vs baseline: {recall*100:.1f}%")
    
    # Summary table
    print("\n" + "="*70)
    print("ABLATION SUMMARY")
    print("="*70)
    print(f"{'Config':<30} {'KMeans':<10} {'Select':<10} {'Total':<10} {'Speedup':<10}")
    print("-"*70)
    
    baseline_time = results["Baseline (Exact + Recompute)"]["total_time"]
    
    for name in results:
        r = results[name]
        speedup = baseline_time / r["total_time"]
        print(f"{name:<30} {r['kmeans_time']:<10.2f} {r['selection_time']:<10.2f} {r['total_time']:<10.2f} {speedup:<10.2f}x")
    
    # Save results
    output_dir = ARTIFACTS_DIR / "ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
