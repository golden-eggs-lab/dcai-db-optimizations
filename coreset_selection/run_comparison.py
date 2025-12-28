"""
Selection-only timing comparison.

Compares baseline vs optimized coreset selection:
- Baseline: Exact KMeans + Recompute cosine distances
- Optimized: ANN KMeans + Cache L2 distances

Usage:
    python run_comparison.py
"""

import json
import numpy as np
from pathlib import Path

from config import CACHE_DIR, ARTIFACTS_DIR, SELECTION_CONFIG, FAISS_CONFIG
from coreset_selection import baseline_selection, optimized_selection, compute_recall


def main():
    print("="*60)
    print("CORESET SELECTION TIMING COMPARISON")
    print("="*60)
    
    # Load embeddings
    embeddings_path = CACHE_DIR / "coedit_embeddings.npy"
    if not embeddings_path.exists():
        print(f"Error: Embeddings not found at {embeddings_path}")
        print("Please run prepare_data.py first.")
        return
    
    vectors = np.load(embeddings_path).astype(np.float32)
    print(f"Loaded embeddings: {vectors.shape}")
    
    # Selection parameters
    N = vectors.shape[0]
    K = SELECTION_CONFIG["K"]
    A = SELECTION_CONFIG["A"]
    seed = SELECTION_CONFIG["seed"]
    
    print(f"\nSelection Config:")
    print(f"  N = {N}")
    print(f"  K = {K} clusters")
    print(f"  A = {A} samples/cluster")
    print(f"  Ratio = {K * A / N * 100:.1f}%")
    print(f"  Seed = {seed}")
    
    # Run baseline
    print("\n" + "-"*60)
    baseline_indices, baseline_timing = baseline_selection(
        vectors, K, A, seed=seed, verbose=True
    )
    
    # Run optimized
    print("\n" + "-"*60)
    optimized_indices, optimized_timing = optimized_selection(
        vectors, K, A, seed=seed,
        nlist=FAISS_CONFIG["nlist"],
        nprobe=FAISS_CONFIG["nprobe"],
        verbose=True
    )
    
    # Compute recall
    recall = compute_recall(baseline_indices, optimized_indices)
    
    # Results
    speedup = baseline_timing["total"] / optimized_timing["total"]
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Baseline time:   {baseline_timing['total']:.2f}s")
    print(f"Optimized time:  {optimized_timing['total']:.2f}s")
    print(f"Speedup:         {speedup:.2f}x")
    print(f"Recall:          {recall*100:.1f}%")
    print(f"Baseline samples:   {len(baseline_indices)}")
    print(f"Optimized samples:  {len(optimized_indices)}")
    
    # Save results
    output_dir = ARTIFACTS_DIR / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "baseline": {
            "total_time": baseline_timing["total"],
            "kmeans_time": baseline_timing["kmeans"],
            "selection_time": baseline_timing["selection"],
            "n_selected": len(baseline_indices),
        },
        "optimized": {
            "total_time": optimized_timing["total"],
            "kmeans_time": optimized_timing["kmeans"],
            "selection_time": optimized_timing["selection"],
            "n_selected": len(optimized_indices),
        },
        "speedup": speedup,
        "recall": recall,
        "config": {
            "N": N,
            "K": K,
            "A": A,
            "seed": seed,
            "nlist": FAISS_CONFIG["nlist"],
            "nprobe": FAISS_CONFIG["nprobe"],
        }
    }
    
    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save selected indices
    np.save(output_dir / "baseline_indices.npy", baseline_indices)
    np.save(output_dir / "optimized_indices.npy", optimized_indices)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
