#!/usr/bin/env python3
"""
FAISS In-Memory Experiment: SemDeDup vs FairDeDup

Compares:
- FAISS Exact (IndexFlatIP) vs FAISS ANN (IndexIVFFlat)
- With/without cache (terms reuse optimization)
- Downstream classification evaluation

Usage:
    python run_experiment.py --seed 42 --run-id run1
    python run_experiment.py --seed 42 --skip-downstream
"""

import os
import sys
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import faiss

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import SemDeDup, FairDeDup, run_kmeans_clustering, load_clusters


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
        'output_dir': os.path.join(base_dir, 'output', f'cifar10{run_suffix}'),
        
        # Training config
        'batch_size': 128,
        'num_epochs': 20,
        'learning_rate': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        
        # ANN config
        'nlist': 100,
        'nprobe': 10,
    }


# ============================================================================
# Neural Network for Classification
# ============================================================================

class SimpleCNN(nn.Module):
    """Simple CNN for CIFAR-10 classification."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def train_and_evaluate(
    train_indices: np.ndarray,
    config: dict,
    device: torch.device,
    seed: int = 42,
) -> tuple:
    """Train model and evaluate accuracy."""
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
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
    
    train_loader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2)
    
    # Model
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['learning_rate'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    
    # Training
    start_time = time.time()
    for epoch in range(config['num_epochs']):
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


# ============================================================================
# Main Experiment
# ============================================================================

def run_dedup_experiment(
    algorithm: str,
    embeddings: np.ndarray,
    clusters: list,
    config: dict,
    use_ann: bool,
    use_cache: bool,
) -> dict:
    """Run a single deduplication experiment."""
    
    # Select algorithm
    if algorithm == 'semdedup':
        dedup = SemDeDup(
            embeddings=embeddings,
            eps_list=config['eps_list'],
            use_ann=use_ann,
            use_cache=use_cache,
            nlist=config['nlist'],
            nprobe=config['nprobe'],
        )
    else:
        dedup = FairDeDup(
            embeddings=embeddings,
            eps_list=config['eps_list'],
            use_ann=use_ann,
            use_cache=use_cache,
            nlist=config['nlist'],
            nprobe=config['nprobe'],
        )
    
    # Create output directory
    method_name = f"{algorithm}_{'ANN' if use_ann else 'Exact'}_{'Cache' if use_cache else 'NoCache'}"
    output_dir = os.path.join(config['output_dir'], method_name, 'dataframes')
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all clusters
    total_duplicates = {eps: 0 for eps in config['eps_list']}
    
    start_time = time.time()
    for cluster_id, cluster_data in enumerate(tqdm(clusters, desc=method_name)):
        cluster_indices = cluster_data[:, 1].astype(int)
        
        dups = dedup.process_cluster(cluster_indices, output_dir, cluster_id)
        
        for eps, count in dups.items():
            total_duplicates[eps] += count
    
    total_time = time.time() - start_time
    
    # Get timing breakdown
    timing = dedup.get_timing_breakdown()
    
    return {
        'method': method_name,
        'time': total_time,
        'timing': timing,
        'duplicates': total_duplicates,
    }


def main():
    parser = argparse.ArgumentParser(description='FAISS In-Memory Dedup Experiment')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--run-id', type=str, default='', help='Run identifier')
    parser.add_argument('--skip-downstream', action='store_true', help='Skip downstream evaluation')
    args = parser.parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"Random seed: {args.seed}")
    if args.run_id:
        print(f"Run ID: {args.run_id}")
    
    config = get_config(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("FAISS In-Memory Experiment: SemDeDup vs FairDeDup")
    print("=" * 70)
    print(f"Dataset: {config['dataset_name']}")
    print(f"Samples: {config['dataset_size']:,}")
    print(f"Clusters: {config['num_clusters']}")
    print(f"Epsilon values: {config['eps_list']}")
    print(f"Device: {device}")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Load embeddings
    print("\nLoading embeddings...")
    embeddings = np.memmap(
        config['embs_path'],
        dtype='float32',
        mode='r',
        shape=(config['dataset_size'], config['emb_size'])
    )
    
    # Clustering
    sorted_clusters_dir = os.path.join(config['output_dir'], 'sorted_clusters')
    if not os.path.exists(sorted_clusters_dir):
        print("\nRunning clustering...")
        run_kmeans_clustering(
            embeddings=embeddings,
            num_clusters=config['num_clusters'],
            output_dir=config['output_dir'],
            seed=args.seed,
        )
    else:
        print(f"\n✓ Using existing clusters from {sorted_clusters_dir}")
    
    clusters = load_clusters(sorted_clusters_dir, config['num_clusters'])
    
    # Run experiments
    results = {}
    
    configurations = [
        ('semdedup', False, False),  # Exact + NoCache (baseline)
        ('semdedup', False, True),   # Exact + Cache
        ('semdedup', True, False),   # ANN + NoCache
        ('semdedup', True, True),    # ANN + Cache (optimized)
        ('fairdedup', False, False), # Baseline
        ('fairdedup', False, True),  # Exact + Cache
        ('fairdedup', True, False),  # ANN + NoCache
        ('fairdedup', True, True),   # ANN + Cache (optimized)
    ]
    
    print("\n" + "=" * 70)
    print("Running Deduplication Experiments")
    print("=" * 70)
    
    for algorithm, use_ann, use_cache in configurations:
        result = run_dedup_experiment(
            algorithm=algorithm,
            embeddings=embeddings,
            clusters=clusters,
            config=config,
            use_ann=use_ann,
            use_cache=use_cache,
        )
        results[result['method']] = result
        
        print(f"\n{result['method']}:")
        print(f"  Time: {result['time']:.2f}s")
        for eps, count in result['duplicates'].items():
            pct = count / config['dataset_size'] * 100
            print(f"  eps={eps}: {count:,} duplicates ({pct:.1f}%)")
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    baseline_time = results['semdedup_Exact_NoCache']['time']
    print(f"\n{'Method':<35} {'Time':>10} {'Speedup':>10}")
    print("-" * 55)
    for name, result in results.items():
        speedup = baseline_time / result['time']
        print(f"{name:<35} {result['time']:>9.2f}s {speedup:>9.1f}x")
    
    # Downstream evaluation
    if not args.skip_downstream:
        print("\n" + "=" * 70)
        print("Downstream Evaluation (CIFAR-10 Classification)")
        print("=" * 70)
        
        eval_results = {}
        
        # Baseline: full dataset
        print("\n[Baseline] Full training set")
        all_indices = np.arange(config['dataset_size'])
        acc, train_time = train_and_evaluate(all_indices, config, device, args.seed)
        print(f"  Samples: {len(all_indices):,}, Accuracy: {acc*100:.2f}%, Time: {train_time:.1f}s")
        eval_results['Full'] = {'samples': len(all_indices), 'accuracy': acc, 'time': train_time}
        
        # Deduplicated datasets
        for method_name, result in results.items():
            print(f"\n[{method_name}]")
            
            output_dir = os.path.join(config['output_dir'], method_name, 'dataframes')
            
            for eps in config['eps_list']:
                # Collect kept indices
                kept_indices = []
                for cluster_id in range(config['num_clusters']):
                    df_path = os.path.join(output_dir, f'cluster_{cluster_id}.pkl')
                    with open(df_path, 'rb') as f:
                        df = pickle.load(f)
                    
                    mask = ~df[f'eps={eps}']
                    kept_indices.extend(df.loc[mask, 'indices'].tolist())
                
                kept_indices = np.array(kept_indices)
                
                acc, train_time = train_and_evaluate(kept_indices, config, device, args.seed)
                print(f"  eps={eps}: {len(kept_indices):,} samples, Accuracy: {acc*100:.2f}%, Time: {train_time:.1f}s")
                
                key = f"{method_name}_eps{eps}"
                eval_results[key] = {'samples': len(kept_indices), 'accuracy': acc, 'time': train_time}
        
        results['downstream'] = eval_results
    
    # Save results
    results_path = os.path.join(config['output_dir'], 'experiment_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n✓ Results saved to {results_path}")
    
    print("\n" + "=" * 70)
    print("✓ Experiment complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
