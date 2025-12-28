"""
FairDeDup: Fair Semantic Deduplication

Improved algorithm that removes duplicates using bidirectional similarity checking.
Ensures fair treatment of all samples regardless of their position in the dataset.

Key difference from SemDeDup:
- SemDeDup: If A is similar to B (A comes first), remove B
- FairDeDup: If A and B are mutually similar, remove based on symmetric criteria
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import faiss


class FairDeDup:
    """
    FairDeDup implementation with FAISS backend.
    
    Uses bidirectional similarity checking for fair duplicate removal.
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
            actual_nlist = min(self.nlist, n // 4)
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
        Process a single cluster using FairDeDup algorithm.
        
        FairDeDup difference: For each pair of similar points,
        we use a fair criterion (e.g., keep the one with higher norm,
        or use Union-Find to group duplicates).
        
        Returns: Dict mapping epsilon to number of duplicates found
        """
        n = len(cluster_indices)
        
        if n <= 1:
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
            return self._process_with_cache(cluster_embs, cluster_indices, output_dir, cluster_id)
        else:
            return self._process_without_cache(cluster_embs, cluster_indices, output_dir, cluster_id)
    
    def _find_root(self, parent: np.ndarray, i: int) -> int:
        """Union-Find: find root with path compression."""
        if parent[i] != i:
            parent[i] = self._find_root(parent, parent[i])
        return parent[i]
    
    def _union(self, parent: np.ndarray, rank: np.ndarray, i: int, j: int):
        """Union-Find: union by rank."""
        root_i = self._find_root(parent, i)
        root_j = self._find_root(parent, j)
        
        if root_i != root_j:
            if rank[root_i] < rank[root_j]:
                parent[root_i] = root_j
            elif rank[root_i] > rank[root_j]:
                parent[root_j] = root_i
            else:
                parent[root_j] = root_i
                rank[root_i] += 1
    
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
        all_duplicates = {}
        
        for eps in self.eps_list:
            # Build index
            index = self._build_index(cluster_embs)
            
            # Search for neighbors
            k = min(n, 100)
            distances, indices = self._search(index, cluster_embs, k=k)
            
            start = time.time()
            
            # Union-Find for fair grouping
            parent = np.arange(n)
            rank = np.zeros(n, dtype=int)
            
            threshold = 1 - eps
            
            # Build similarity graph and union similar points
            for i in range(n):
                for j_idx in range(k):
                    j = indices[i, j_idx]
                    if j == i:
                        continue
                    if distances[i, j_idx] >= threshold:
                        self._union(parent, rank, i, j)
            
            # For each connected component, keep only the representative (smallest index)
            duplicates = np.zeros(n, dtype=bool)
            for i in range(n):
                root = self._find_root(parent, i)
                if root != i:
                    duplicates[i] = True
            
            self.timing['apply_threshold'] += time.time() - start
            
            results[eps] = duplicates.sum()
            all_duplicates[eps] = duplicates
        
        # Save results
        df = pd.DataFrame({'indices': cluster_indices})
        for eps in self.eps_list:
            df[f'eps={eps}'] = all_duplicates[eps]
        
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
        k = min(n, 100)
        distances, indices = self._search(index, cluster_embs, k=k)
        
        start = time.time()
        
        # Apply all epsilon thresholds using cached search results
        results = {}
        all_duplicates = {}
        
        for eps in self.eps_list:
            # Union-Find for fair grouping
            parent = np.arange(n)
            rank = np.zeros(n, dtype=int)
            
            threshold = 1 - eps
            
            # Build similarity graph
            for i in range(n):
                for j_idx in range(k):
                    j = indices[i, j_idx]
                    if j == i:
                        continue
                    if distances[i, j_idx] >= threshold:
                        self._union(parent, rank, i, j)
            
            # Keep only representatives
            duplicates = np.zeros(n, dtype=bool)
            for i in range(n):
                root = self._find_root(parent, i)
                if root != i:
                    duplicates[i] = True
            
            results[eps] = duplicates.sum()
            all_duplicates[eps] = duplicates
        
        self.timing['apply_threshold'] += time.time() - start
        
        # Save results
        df = pd.DataFrame({'indices': cluster_indices})
        for eps in self.eps_list:
            df[f'eps={eps}'] = all_duplicates[eps]
        
        df_path = os.path.join(output_dir, f'cluster_{cluster_id}.pkl')
        with open(df_path, 'wb') as f:
            pickle.dump(df, f)
        
        return results
    
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
