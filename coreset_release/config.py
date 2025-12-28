"""
Configuration for DEFT-UCS Coreset Selection experiments.

Optimizing DEFT-UCS algorithm through:
1. ANN KMeans: Use FAISS IVF for approximate nearest neighbor in clustering
2. Cache L2: Reuse L2 distances from KMeans for cosine computation
"""

from pathlib import Path

# ============== Paths ==============
BASE_DIR = Path(__file__).parent
CACHE_DIR = BASE_DIR / "cache"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Create directories
CACHE_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ============== Dataset ==============
DATASET_NAME = "grammarly/coedit"
EMBEDDING_MODEL = "sentence-transformers/sentence-t5-base"
MODEL_NAME = "google/flan-t5-base"

# ============== Selection Parameters ==============
# Following DEFT-UCS paper setup
N_SAMPLES = 69071  # Total CoEDIT samples
K = 7              # Number of clusters (from paper)
SELECTION_RATIO = 0.325  # 32.5% selection ratio

SELECTION_CONFIG = {
    "K": K,
    "A": max(1, int(N_SAMPLES * SELECTION_RATIO / K)),  # ~3206 samples per cluster
    "selection_ratio": SELECTION_RATIO,
    "alpha": 0.5,  # Easy sample ratio
    "beta": 0.5,   # Hard sample ratio
    "seed": 42,
}

# ============== FAISS Configuration ==============
FAISS_CONFIG = {
    "nlist": 256,   # Number of IVF cells
    "nprobe": 128,  # Number of cells to probe (50% of nlist)
}

# ============== Training Configuration ==============
TRAINING_CONFIG = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "learning_rate": 3e-5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_input_length": 256,
    "max_target_length": 256,
    "fp16": True,
    "logging_steps": 50,
    "save_strategy": "epoch",
    "eval_strategy": "epoch",
}
