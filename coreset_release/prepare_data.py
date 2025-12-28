"""
Prepare CoEDIT dataset and compute Sentence-T5 embeddings.

Usage:
    python prepare_data.py
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from config import CACHE_DIR, DATASET_NAME, EMBEDDING_MODEL


def download_coedit():
    """Download and preprocess CoEDIT dataset."""
    from datasets import load_dataset
    
    print(f"Loading dataset: {DATASET_NAME}")
    ds = load_dataset(DATASET_NAME)
    
    samples = []
    for split in ["train", "validation"]:
        if split in ds:
            for item in tqdm(ds[split], desc=f"Processing {split}"):
                samples.append({
                    "src": item["src"],
                    "tgt": item["tgt"],
                    "task": item.get("task", "unknown"),
                })
    
    print(f"Total samples: {len(samples)}")
    
    # Save dataset
    output_path = CACHE_DIR / "coedit_dataset.json"
    with open(output_path, 'w') as f:
        json.dump(samples, f)
    
    print(f"Dataset saved to {output_path}")
    return samples


def compute_embeddings(samples):
    """Compute Sentence-T5 embeddings for all samples."""
    from sentence_transformers import SentenceTransformer
    
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    # Encode source texts
    texts = [s["src"] for s in samples]
    
    print(f"Computing embeddings for {len(texts)} samples...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    
    # Save embeddings
    output_path = CACHE_DIR / "coedit_embeddings.npy"
    np.save(output_path, embeddings.astype(np.float32))
    
    print(f"Embeddings saved to {output_path}")
    print(f"Shape: {embeddings.shape}")
    
    return embeddings


def main():
    # Download dataset
    dataset_path = CACHE_DIR / "coedit_dataset.json"
    if dataset_path.exists():
        print(f"Dataset already exists at {dataset_path}")
        with open(dataset_path, 'r') as f:
            samples = json.load(f)
    else:
        samples = download_coedit()
    
    # Compute embeddings
    embeddings_path = CACHE_DIR / "coedit_embeddings.npy"
    if embeddings_path.exists():
        print(f"Embeddings already exist at {embeddings_path}")
    else:
        compute_embeddings(samples)
    
    print("\nData preparation complete!")


if __name__ == "__main__":
    main()
