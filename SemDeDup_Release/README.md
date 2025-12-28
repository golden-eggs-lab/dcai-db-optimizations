# Semantic Deduplication: SemDeDup and FairDeDup

This repository contains implementations of two semantic deduplication methods with optimization techniques for efficient large-scale data deduplication.

---

## 🧠 Abstract

Semantic deduplication removes near-duplicate samples from training data by comparing embedding similarities. We implement and compare two deduplication strategies:

1. **SemDeDup** (Abbas et al.): Greedy sequential deduplication that keeps earlier samples and removes later duplicates
2. **FairDeDup**: Fairness-aware deduplication using Union-Find that treats all samples symmetrically

Both methods support two optimization techniques:
- **ANN Search**: Approximate Nearest Neighbor search reduces complexity from O(n²) to O(n log n)
- **Terms Caching**: Reusing search results across multiple epsilon thresholds reduces redundant computation by 67%

---

## 📊 Method Comparison

| Method | Description | Fairness |
|--------|-------------|----------|
| **SemDeDup** | Greedy sequential: earlier samples kept, later duplicates removed | Position-biased |
| **FairDeDup** | Union-Find grouping: symmetric treatment of all samples | Fair |

### Optimization Variants

| Configuration | Search | Caching | Complexity |
|---------------|--------|---------|------------|
| `Exact_NoCache` | Brute-force | ❌ | O(n² × |ε|) |
| `Exact_Cache` | Brute-force | ✅ | O(n²) |
| `ANN_NoCache` | IVF Index | ❌ | O(n log n × |ε|) |
| `ANN_Cache` | IVF Index | ✅ | O(n log n) |

---

## 📦 Setup

### 🔧 1. Install Environment

```bash
conda env create -f environment.yml
conda activate semdedup
```

Or install dependencies manually:

```bash
pip install numpy pandas torch torchvision faiss-cpu tqdm
pip install "pymilvus>=2.4.0"  # For Milvus backend (optional)
```

### 🔑 2. Prepare Embeddings

Generate CLIP embeddings for your dataset:

```bash
python experiments/compute_embeddings.py --dataset cifar10 --output embeddings/cifar10_embeddings.npy
```

---

## 🚀 Quick Start

### Run Full Experiment (FAISS In-Memory)

```bash
# Run experiment comparing SemDeDup vs FairDeDup (all 8 configurations)
python experiments/run_cifar10_experiment.py --seed 42 --run-id run1
```

### Run with Milvus Vector Database

```bash
# Using Milvus Lite (local, no server needed)
python experiments/run_cifar10_milvus_experiment.py --seed 42 --run-id milvus_run1

# Skip downstream evaluation (faster)
python experiments/run_cifar10_milvus_experiment.py --seed 42 --skip-downstream
```

---

## 📊 Experiment Configurations

We test 8 configurations total (2 methods × 4 optimization levels):

| Method | Configuration | Search | Cache | Description |
|--------|---------------|--------|-------|-------------|
| SemDeDup | `Exact_NoCache` | FAISS Flat | ❌ | Baseline |
| SemDeDup | `Exact_Cache` | FAISS Flat | ✅ | + Caching |
| SemDeDup | `ANN_NoCache` | FAISS IVF | ❌ | + ANN |
| SemDeDup | `ANN_Cache` | FAISS IVF | ✅ | Optimized |
| FairDeDup | `Exact_NoCache` | FAISS Flat | ❌ | Baseline |
| FairDeDup | `Exact_Cache` | FAISS Flat | ✅ | + Caching |
| FairDeDup | `ANN_NoCache` | FAISS IVF | ❌ | + ANN |
| FairDeDup | `ANN_Cache` | FAISS IVF | ✅ | Optimized |

---

## 🧱 Project Structure

```
SemDeDup_Release/
├── README.md                    # This file
├── environment.yml              # Conda environment
│
├── experiments/                 # Experiment scripts
│   ├── run_cifar10_experiment.py       # FAISS in-memory experiment
│   ├── run_cifar10_milvus_experiment.py # Milvus vector DB experiment
│   └── compute_embeddings.py           # Embedding generation
│
├── embeddings/                  # Pre-computed embeddings (generated)
│   └── README.md
│
├── configs/                     # Configuration files
│   └── default_config.yaml      # Default experiment settings
│
└── scripts/                     # Utility scripts
    └── run_ablation.sh          # Run ablation study
```

## 🔧 Configuration Options

### Command Line Arguments

```bash
python experiments/run_cifar10_experiment.py \
    --seed 42 \                    # Random seed
    --run-id experiment1           # Unique run identifier
```

```bash
python experiments/run_cifar10_milvus_experiment.py \
    --seed 42 \                    # Random seed
    --run-id milvus1 \             # Unique run identifier
    --skip-downstream              # Skip classification eval
```

### Environment Variables

```bash
export CUDA_VISIBLE_DEVICES=0      # GPU selection
export FAISS_NO_GPU=1              # Force CPU mode
```

## 📄 License

This project is licensed under the Apache 2.0 License. See [LICENSE](LICENSE) for details.
