#!/usr/bin/env python3
"""
Compute CLIP embeddings for image datasets.

Usage:
    python compute_embeddings.py --dataset cifar10 --output embeddings/cifar10_embeddings.npy
    python compute_embeddings.py --dataset stl10 --output embeddings/stl10_embeddings.npy
"""

import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torchvision import datasets, transforms

try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False
    print("Warning: open_clip not installed. Run: pip install open_clip_torch")


def compute_embeddings(dataset_name: str, output_path: str, batch_size: int = 256):
    """Compute CLIP embeddings for a dataset."""
    
    if not OPEN_CLIP_AVAILABLE:
        print("Error: open_clip required. Run: pip install open_clip_torch")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load CLIP model
    print("Loading CLIP ViT-B/32...")
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    model = model.to(device)
    model.eval()
    
    # Load dataset
    print(f"Loading {dataset_name}...")
    if dataset_name.lower() == 'cifar10':
        dataset = datasets.CIFAR10(
            root='./data', train=True, download=True,
            transform=preprocess
        )
    elif dataset_name.lower() == 'stl10':
        dataset = datasets.STL10(
            root='./data', split='train', download=True,
            transform=preprocess
        )
    elif dataset_name.lower() == 'cifar100':
        dataset = datasets.CIFAR100(
            root='./data', train=True, download=True,
            transform=preprocess
        )
    else:
        print(f"Unknown dataset: {dataset_name}")
        return
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    # Compute embeddings
    print(f"Computing embeddings for {len(dataset)} images...")
    all_embeddings = []
    
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Embedding"):
            images = images.to(device)
            embeddings = model.encode_image(images)
            embeddings = embeddings.cpu().numpy()
            all_embeddings.append(embeddings)
    
    all_embeddings = np.vstack(all_embeddings).astype(np.float32)
    print(f"Embeddings shape: {all_embeddings.shape}")
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, all_embeddings)
    print(f"✓ Saved to {output_path}")
    
    # Also save as memmap for large datasets
    memmap_path = output_path.replace('.npy', '_memmap.npy')
    memmap = np.memmap(memmap_path, dtype='float32', mode='w+', shape=all_embeddings.shape)
    memmap[:] = all_embeddings
    memmap.flush()
    print(f"✓ Saved memmap to {memmap_path}")


def main():
    parser = argparse.ArgumentParser(description='Compute CLIP embeddings')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['cifar10', 'cifar100', 'stl10'],
                        help='Dataset name')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for embeddings')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size for embedding computation')
    args = parser.parse_args()
    
    compute_embeddings(args.dataset, args.output, args.batch_size)


if __name__ == '__main__':
    main()
