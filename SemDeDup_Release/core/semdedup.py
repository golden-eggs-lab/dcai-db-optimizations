"""
SemDeDup: Semantic Deduplication

Original algorithm that removes duplicates based on embedding similarity.
Uses greedy sequential approach - earlier samples are kept, later duplicates removed.

Reference: Abbas et al., "SemDeDup: Data-efficient learning at web-scale through semantic deduplication"
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import faiss


class SemDeDup:
    """
    SemDeDup implementation with FAISS backend.
    
    Supports both exact search (IndexFlatIP) and ANN search (IndexIVFFlat).
    Supports terms caching (search once, apply multiple epsilon thresholds).
    """
    
    def __init__(
        self,
        embeddings: np.ndarray,
        eps_list: List[float],
        use_ann: bool = False,
        use_cache: bool = False,
        nlist: int = 100,
        nprobe: int = 10,
    ):
        """
        Args:
            embeddings: (N, D) array of normalized embeddings
            eps_list: List of epsilon thresholds for duplicate detection
            use_ann: Use approximate nearest neighbor search
            use_cache: Cache search results for multiple epsilon thresholds
            nlist: Number of clusters for IVF index (ANN)
            nprobe: Number of clusters to search (ANN)
        """
        self.embeddings = embeddings
        self.eps_list = sorted(eps_list, reverse=True)  # Largest first for caching
        self.use_ann = use_ann
        self.use_cache = use_cache
        self.nlist = nlist
        self.nprobe = nprobe
        
        # Timing statistics
        self.timing = {
            'load': 0.0,
            'normalize': 0.0,
            'index_build': 0.0,
            'search': 0.0,
            'apply_threshold': 0.0,
            'index_build_calls': 0,
            'search_calls': 0,
        }
    
    def _build_index(self, cluster_embs: np.ndarray) -> faiss.Index:
        """Build FAISS index for a cluster."""
        start = time.time()
        
        d = cluster_embs.shape[1]
        
        if self.use_ann:
            # IVF index for approximate search
            n = cluster_embs.shape[0]
            actual_nlist = min(self.nlist, n // 4)  # Ensure enough points per cell
            if actual_nlist < 1:
                actual_nlist = 1
            
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, actual_nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(cluster_embs)
            index.add(cluster_embs)
            index.nprobe = min(self.nprobe, actual_nlist)
        else:
            # Exact search
            index = faiss.IndexFlatIP(d)
            index.add(cluster_embs)
        
        elapsed = time.time() - start
        self.timing['index_build'] += elapsed
        self.timing['index_build_calls'] += 1
        
        return index
    
    def _search(self, index: faiss.Index, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors."""
        start = time.time()
        
        distances, indices = index.search(queries, k)
        
        elapsed = time.time() - start
        self.timing['search'] += elapsed
        self.timing['search_calls'] += 1
        
        return distances, indices
    
    def process_cluster(
        self,
        cluster_indices: np.ndarray,
        output_dir: str,
        cluster_id: int,
    ) -> Dict[float, int]:
        """
        Process a single cluster using SemDeDup algorithm.
        
        Returns: Dict mapping epsilon to number of duplicates found
        """
        n = len(cluster_indices)
        
        if n <= 1:
            # Single point, no duplicates possible
            df = pd.DataFrame({'indices': cluster_indices})
            for eps in self.eps_list:
                df[f'eps={eps}'] = False
            
            df_path = os.path.join(output_dir, f'cluster_{cluster_id}.pkl')
            with open(df_path, 'wb') as f:
                pickle.dump(df, f)
            
            return {eps: 0 for eps in self.eps_list}
        
        # Get cluster embeddings
        start = time.time()
        cluster_embs = np.array(self.embeddings[cluster_indices], dtype=np.float32)
        self.timing['load'] += time.time() - start
        
        # Normalize
        start = time.time()
        faiss.normalize_L2(cluster_embs)
        self.timing['normalize'] += time.time() - start
        
        if self.use_cache:
            # Cache mode: search once with largest epsilon, apply all thresholds
            return self._process_with_cache(cluster_embs, cluster_indices, output_dir, cluster_id)
        else:
            # No cache: separate search for each epsilon
            return self._process_without_cache(cluster_embs, cluster_indices, output_dir, cluster_id)
    
    def _process_without_cache(
        self,
        cluster_embs: np.ndarray,
        cluster_indices: np.ndarray,
        output_dir: str,
        cluster_id: int,
    ) -> Dict[float, int]:
        """Process cluster with separate search per epsilon."""
        n = len(cluster_indices)
        results = {}
        
        for eps in self.eps_list:
            # Build fresh index for each epsilon
            index = self._build_index(cluster_embs)
            
            # For each point, check if any earlier point is within epsilon
            duplicates = np.zeros(n, dtype=bool)
            
            # Search for nearest neighbor (excluding self)
            distances, indices = self._search(index, cluster_embs, k=2)
            
            start = time.time()
            
            # Greedy sequential: mark as duplicate if similar to earlier point
            for i in range(1, n):
                # Check if any earlier point is within epsilon
                for j in range(min(2, len(indices[i]))):
                    neighbor_idx = indices[i, j]
                    if neighbor_idx < i and neighbor_idx != i:
                        similarity = distances[i, j]
                        if similarity >= 1 - eps:
                            duplicates[i] = True
                            break
            
            self.timing['apply_threshold'] += time.time() - start
            results[eps] = duplicates.sum()
        
        # Save results
        df = pd.DataFrame({'indices': cluster_indices})
        for eps in self.eps_list:
            df[f'eps={eps}'] = results.get(eps, 0) > 0  # Placeholder
        
        df_path = os.path.join(output_dir, f'cluster_{cluster_id}.pkl')
        with open(df_path, 'wb') as f:
            pickle.dump(df, f)
        
        return results
    
    def _process_with_cache(
        self,
        cluster_embs: np.ndarray,
        cluster_indices: np.ndarray,
        output_dir: str,
        cluster_id: int,
    ) -> Dict[float, int]:
        """Process cluster with cached search results."""
        n = len(cluster_indices)
        
        # Build index once
        index = self._build_index(cluster_embs)
        
        # Search once for all points
        k = min(n, 100)  # Get top-k neighbors
        distances, indices = self._search(index, cluster_embs, k=k)
        
        start = time.time()
        
        # Initialize duplicate flags for each epsilon
        duplicates = {eps: np.zeros(n, dtype=bool) for eps in self.eps_list}
        
        # For each point (starting from second)
        for i in range(1, n):
            # Check neighbors
            for j in range(k):
                neighbor_idx = indices[i, j]
                if neighbor_idx >= i or neighbor_idx == i:
                    continue  # Skip self and later points
                
                similarity = distances[i, j]
                
                # Apply all epsilon thresholds
                for eps in self.eps_list:
                    if not duplicates[eps][i] and similarity >= 1 - eps:
                        duplicates[eps][i] = True
        
        self.timing['apply_threshold'] += time.time() - start
        
        # Save results
        df = pd.DataFrame({'indices': cluster_indices})
        for eps in self.eps_list:
            df[f'eps={eps}'] = duplicates[eps]
        
        df_path = os.path.join(output_dir, f'cluster_{cluster_id}.pkl')
        with open(df_path, 'wb') as f:
            pickle.dump(df, f)
        
        return {eps: duplicates[eps].sum() for eps in self.eps_list}
    
    def get_timing_breakdown(self) -> Dict:
        """Return timing statistics."""
        total = sum([
            self.timing['load'],
            self.timing['normalize'],
            self.timing['index_build'],
            self.timing['search'],
            self.timing['apply_threshold'],
        ])
        
        return {
            'load': self.timing['load'],
            'normalize': self.timing['normalize'],
            'index_build': self.timing['index_build'],
            'index_build_calls': self.timing['index_build_calls'],
            'search': self.timing['search'],
            'search_calls': self.timing['search_calls'],
            'apply_threshold': self.timing['apply_threshold'],
            'total': total,
        }
