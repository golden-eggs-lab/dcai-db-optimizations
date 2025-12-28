# CAL-Opt: Optimizing Contrastive Active Learning

This repository contains the implementation of efficient optimizations for Contrastive Active Learning (CAL), including Approximate Nearest Neighbor (ANN) search and probability caching.

---

## 🧠 Overview

Contrastive Active Learning (CAL) selects samples based on KL divergence between predictions of similar labeled and unlabeled samples. The original implementation suffers from computational bottlenecks. We propose two optimizations:

1. **ANN Search**: Replace exact KNN with ball-tree based approximate nearest neighbor search
2. **Probability Caching**: Cache softmax probabilities for labeled samples to avoid redundant computations
3. **Milvus Backend** (optional): Use Milvus vector database for scalable KNN search with IVF/HNSW indexes

---

## 📦 Setup

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download Data

```bash
bash get_data.sh
```

---

## 🚀 Quick Start

### Run Optimization Comparison

```bash
bash run_optimization_comparison.sh
```

### Run Ablation Study

```bash
bash run_ablation.sh
```

### Run Milvus Backend Comparison

```bash
bash run_milvus_comparison.sh
```

### View Results

```bash
python show_acc.py
python show_acc.py --ablation
```

## 🧱 Project Structure

```
├── run_al.py                    # Main AL experiment script
├── acquisition/
│   └── cal.py                   # CAL with ANN + caching + Milvus
├── utilities/                   # Data loading, training, metrics
├── run_optimization_comparison.sh  # In-memory optimization comparison
├── run_ablation.sh              # Ablation study
├── run_milvus_comparison.sh     # Milvus backend comparison
└── show_acc.py                  # Results visualization
```

---

## ⚙️ Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--use_sklearn_ann` | Use ball-tree ANN | False |
| `--cache_probabilities` | Enable probability caching | False |
| `--use_milvus` | Use Milvus vector database | False |
| `--milvus_index_type` | Milvus index type (FLAT/IVF_FLAT/HNSW) | FLAT |
| `--init_train_data` | Initial labeled data | 1% |
| `--acquisition_size` | Samples per iteration | 2% |
| `--budget` | Total annotation budget | 15% |

### Example

```bash
# Baseline
python run_al.py --dataset_name sst-2 --acquisition cal \
    --use_sklearn_ann False --cache_probabilities False

# Optimized (in-memory ANN + cache)
python run_al.py --dataset_name sst-2 --acquisition cal \
    --use_sklearn_ann True --cache_probabilities True

# Milvus backend (IVF + cache)
python run_al.py --dataset_name sst-2 --acquisition cal \
    --use_milvus True --milvus_index_type IVF_FLAT --cache_probabilities True
```

---