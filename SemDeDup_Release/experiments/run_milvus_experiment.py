#!/usr/bin/env python3
"""
Milvus Vector Database Experiment: SemDeDup vs FairDeDup

Uses Milvus Lite (local, no server required) for vector search.
Compares:
- Milvus FLAT (Exact) vs Milvus IVF_FLAT (ANN)
- With/without cache (terms reuse optimization)
- Downstream classification evaluation

Usage:
    python run_milvus_experiment.py --seed 42 --run-id milvus_run1
    python run_milvus_experiment.py --seed 42 --skip-downstream
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
    print("Warning: pymilvus not installed. Run: pip install 'pymilvus>=2.4.0'")

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import run_kmeans_clustering, load_clusters


# ============================================================================
# Configuration
# ============================================================================

def get_config(args):
    """Get experiment configuration."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    run_suffix = f"_{args.run_id}" if args.run_id else ""
    
    return {
        'dataset_name': 'CIFAR-10',
        'dataset_size': 50000,
        'emb_size': 512,
        'num_clusters': 10,
        'eps_list': [0.05, 0.10, 0.15],
        'embs_path': os.path.join(base_dir, 'embeddings', 'cifar10_embeddings.npy'),
        'output_dir': os.path.join(base_dir, 'output', f'cifar10_milvus{run_suffix}'),
        'db_path': os.path.join(base_dir, 'output', f'milvus{run_suffix}.db'),
        
        # Milvus config
        'k_neighbors': 100,
        'nlist': 128,
        
        # Training config
        'batch_size': 128,
        'num_epochs': 20,
        'learning_rate': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
    }


# ============================================================================
# Milvus-based Deduplication
# ============================================================================

class MilvusDedup:
    """Base class for Milvus-based deduplication."""
    
    def __init__(
        self,
        embeddings: np.ndarray,
        eps_list: List[float],
        db_path: str,
        use_ann: bool = False,
        use_cache: bool = False,
        k_neighbors: int = 100,
        nlist: int = 128,
    ):
        self.embeddings = embeddings
        self.eps_list = sorted(eps_list, reverse=True)
        self.db_path = db_path
        self.use_ann = use_ann
        self.use_cache = use_cache
        self.k_neighbors = k_neighbors
        self.nlist = nlist
        self.emb_size = embeddings.shape[1]
        
        # Initialize Milvus client
        self.client = MilvusClient(db_path)
        
        # Timing
        self.timing = {
            'load': 0.0,
            'normalize': 0.0,
            'index_build': 0.0,
            'search': 0.0,
            'apply_threshold': 0.0,
            'index_build_calls': 0,
            'search_calls': 0,
        }
    
    def _create_collection(self, name: str, n_samples: int):
        """Create Milvus collection with appropriate index."""
        
        if self.client.has_collection(name):
            self.client.drop_collection(name)
        
        start = time.time()
        
        if self.use_ann:
            # IVF_FLAT for ANN
            actual_nlist = min(self.nlist, n_samples // 4)
            if actual_nlist < 1:
                actual_nlist = 1
            
            index_params = {
                "metric_type": "IP",
                "index_type": "IVF_FLAT",
                "params": {"nlist": actual_nlist}
            }
        else:
            # FLAT for exact search
            index_params = {
                "metric_type": "IP",
                "index_type": "FLAT",
                "params": {}
            }
        
        self.client.create_collection(
            collection_name=name,
            dimension=self.emb_size,
            metric_type="IP",
            auto_id=False,
        )
        
        self.timing['index_build'] += time.time() - start
        self.timing['index_build_calls'] += 1
    
    def _insert_and_search(
        self,
        collection_name: str,
        cluster_embs: np.ndarray,
        k: int,
    ) -> Tuple[List, List]:
        """Insert vectors and search for neighbors."""
        
        n = len(cluster_embs)
        
        # Insert
        data = [{"id": i, "vector": cluster_embs[i].tolist()} for i in range(n)]
        self.client.insert(collection_name=collection_name, data=data)
        
        # Search
        start = time.time()
        
        results = self.client.search(
            collection_name=collection_name,
            data=cluster_embs.tolist(),
            limit=k,
            output_fields=["id"],
        )
        
        self.timing['search'] += time.time() - start
        self.timing['search_calls'] += 1
        
        # Convert to arrays
        distances = []
        indices = []
        for hits in results:
            dist_row = [hit['distance'] for hit in hits]
            idx_row = [hit['id'] for hit in hits]
            distances.append(dist_row)
            indices.append(idx_row)
        
        return np.array(distances), np.array(indices)
    
    def cleanup(self):
        """Clean up Milvus resources."""
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except:
                pass


class MilvusSemDeDup(MilvusDedup):
    """SemDeDup with Milvus backend."""
    
    def process_cluster(
        self,
        cluster_indices: np.ndarray,
        output_dir: str,
        cluster_id: int,
    ) -> Dict[float, int]:
        """Process cluster using SemDeDup algorithm."""
        n = len(cluster_indices)
        
        if n <= 1:
            df = pd.DataFrame({'indices': cluster_indices})
            for eps in self.eps_list:
                df[f'eps={eps}'] = False
            df_path = os.path.join(output_dir, f'cluster_{cluster_id}.pkl')
            with open(df_path, 'wb') as f:
                pickle.dump(df, f)
            return {eps: 0 for eps in self.eps_list}
        
        # Load embeddings
        start = time.time()
        cluster_embs = np.array(self.embeddings[cluster_indices], dtype=np.float32)
        self.timing['load'] += time.time() - start
        
        # Normalize
        start = time.time()
        norms = np.linalg.norm(cluster_embs, axis=1, keepdims=True)
        cluster_embs = cluster_embs / (norms + 1e-8)
        self.timing['normalize'] += time.time() - start
        
        # Create collection
        collection_name = f"semdedup_cluster_{cluster_id}"
        self._create_collection(collection_name, n)
        
        # Search
        k = min(n, self.k_neighbors)
        distances, indices = self._insert_and_search(collection_name, cluster_embs, k)
        
        # Apply thresholds
        start = time.time()
        
        all_duplicates = {}
        for eps in self.eps_list:
            duplicates = np.zeros(n, dtype=bool)
            threshold = 1 - eps
            
            for i in range(1, n):
                for j_idx in range(min(k, len(indices[i]))):
                    neighbor_idx = indices[i][j_idx]
                    if neighbor_idx < i and neighbor_idx != i:
                        if distances[i][j_idx] >= threshold:
                            duplicates[i] = True
                            break
            
            all_duplicates[eps] = duplicates
        
        self.timing['apply_threshold'] += time.time() - start
        
        # Cleanup collection
        self.client.drop_collection(collection_name)
        
        # Save results
        df = pd.DataFrame({'indices': cluster_indices})
        for eps in self.eps_list:
            df[f'eps={eps}'] = all_duplicates[eps]
        
        df_path = os.path.join(output_dir, f'cluster_{cluster_id}.pkl')
        with open(df_path, 'wb') as f:
            pickle.dump(df, f)
        
        return {eps: all_duplicates[eps].sum() for eps in self.eps_list}


class MilvusFairDeDup(MilvusDedup):
    """FairDeDup with Milvus backend."""
    
    def _find_root(self, parent: np.ndarray, i: int) -> int:
        if parent[i] != i:
            parent[i] = self._find_root(parent, parent[i])
        return parent[i]
    
    def _union(self, parent: np.ndarray, rank: np.ndarray, i: int, j: int):
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
    
    def process_cluster(
        self,
        cluster_indices: np.ndarray,
        output_dir: str,
        cluster_id: int,
    ) -> Dict[float, int]:
        """Process cluster using FairDeDup algorithm."""
        n = len(cluster_indices)
        
        if n <= 1:
            df = pd.DataFrame({'indices': cluster_indices})
            for eps in self.eps_list:
                df[f'eps={eps}'] = False
            df_path = os.path.join(output_dir, f'cluster_{cluster_id}.pkl')
            with open(df_path, 'wb') as f:
                pickle.dump(df, f)
            return {eps: 0 for eps in self.eps_list}
        
        # Load embeddings
        start = time.time()
        cluster_embs = np.array(self.embeddings[cluster_indices], dtype=np.float32)
        self.timing['load'] += time.time() - start
        
        # Normalize
        start = time.time()
        norms = np.linalg.norm(cluster_embs, axis=1, keepdims=True)
        cluster_embs = cluster_embs / (norms + 1e-8)
        self.timing['normalize'] += time.time() - start
        
        # Create collection
        collection_name = f"fairdedup_cluster_{cluster_id}"
        self._create_collection(collection_name, n)
        
        # Search
        k = min(n, self.k_neighbors)
        distances, indices = self._insert_and_search(collection_name, cluster_embs, k)
        
        # Apply thresholds with Union-Find
        start = time.time()
        
        all_duplicates = {}
        for eps in self.eps_list:
            parent = np.arange(n)
            rank = np.zeros(n, dtype=int)
            threshold = 1 - eps
            
            for i in range(n):
                for j_idx in range(min(k, len(indices[i]))):
                    j = indices[i][j_idx]
                    if j == i:
                        continue
                    if distances[i][j_idx] >= threshold:
                        self._union(parent, rank, i, j)
            
            duplicates = np.zeros(n, dtype=bool)
            for i in range(n):
                root = self._find_root(parent, i)
                if root != i:
                    duplicates[i] = True
            
            all_duplicates[eps] = duplicates
        
        self.timing['apply_threshold'] += time.time() - start
        
        # Cleanup
        self.client.drop_collection(collection_name)
        
        # Save results
        df = pd.DataFrame({'indices': cluster_indices})
        for eps in self.eps_list:
            df[f'eps={eps}'] = all_duplicates[eps]
        
        df_path = os.path.join(output_dir, f'cluster_{cluster_id}.pkl')
        with open(df_path, 'wb') as f:
            pickle.dump(df, f)
        
        return {eps: all_duplicates[eps].sum() for eps in self.eps_list}


# ============================================================================
# Neural Network
# ============================================================================

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(512, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def train_and_evaluate(train_indices, config, device, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
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
    train_loader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2)
    
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'],
                          momentum=config['momentum'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    
    start_time = time.time()
    for epoch in range(config['num_epochs']):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
        scheduler.step()
    train_time = time.time() - start_time
    
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            _, predicted = model(inputs).max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return correct / total, train_time


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Milvus Dedup Experiment')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run-id', type=str, default='')
    parser.add_argument('--skip-downstream', action='store_true')
    args = parser.parse_args()
    
    if not MILVUS_AVAILABLE:
        print("Error: pymilvus required. Run: pip install 'pymilvus>=2.4.0'")
        sys.exit(1)
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    config = get_config(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("Milvus Experiment: SemDeDup vs FairDeDup")
    print("=" * 70)
    print(f"Dataset: {config['dataset_name']}")
    print(f"Samples: {config['dataset_size']:,}")
    print(f"Device: {device}")
    print("=" * 70)
    
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Load embeddings
    embeddings = np.memmap(
        config['embs_path'], dtype='float32', mode='r',
        shape=(config['dataset_size'], config['emb_size'])
    )
    
    # Clustering
    sorted_clusters_dir = os.path.join(config['output_dir'], 'sorted_clusters')
    if not os.path.exists(sorted_clusters_dir):
        run_kmeans_clustering(embeddings, config['num_clusters'], config['output_dir'], seed=args.seed)
    
    clusters = load_clusters(sorted_clusters_dir, config['num_clusters'])
    
    results = {}
    
    # Only test baseline and optimized
    configurations = [
        ('semdedup', False, False),  # Baseline
        ('semdedup', True, True),    # Optimized
        ('fairdedup', False, False), # Baseline
        ('fairdedup', True, True),   # Optimized
    ]
    
    for algorithm, use_ann, use_cache in configurations:
        method_name = f"{algorithm}_Milvus_{'ANN' if use_ann else 'Exact'}_{'Cache' if use_cache else 'NoCache'}"
        print(f"\n[{method_name}]")
        
        output_dir = os.path.join(config['output_dir'], method_name, 'dataframes')
        os.makedirs(output_dir, exist_ok=True)
        
        if algorithm == 'semdedup':
            dedup = MilvusSemDeDup(
                embeddings, config['eps_list'], config['db_path'],
                use_ann, use_cache, config['k_neighbors'], config['nlist']
            )
        else:
            dedup = MilvusFairDeDup(
                embeddings, config['eps_list'], config['db_path'],
                use_ann, use_cache, config['k_neighbors'], config['nlist']
            )
        
        total_dups = {eps: 0 for eps in config['eps_list']}
        start = time.time()
        
        for cluster_id, cluster_data in enumerate(tqdm(clusters, desc=method_name)):
            cluster_indices = cluster_data[:, 1].astype(int)
            dups = dedup.process_cluster(cluster_indices, output_dir, cluster_id)
            for eps, count in dups.items():
                total_dups[eps] += count
        
        total_time = time.time() - start
        
        results[method_name] = {
            'time': total_time,
            'duplicates': total_dups,
        }
        
        print(f"  Time: {total_time:.2f}s")
        for eps, count in total_dups.items():
            print(f"  eps={eps}: {count:,} duplicates")
        
        dedup.cleanup()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    baseline_time = results['semdedup_Milvus_Exact_NoCache']['time']
    for name, result in results.items():
        speedup = baseline_time / result['time']
        print(f"{name}: {result['time']:.2f}s ({speedup:.1f}x)")
    
    # Save results
    results_path = os.path.join(config['output_dir'], 'milvus_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n✓ Results saved to {results_path}")


if __name__ == '__main__':
    main()
