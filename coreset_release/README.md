# DEFT-UCS-Opt: Optimizing DEFT Uncertainty-based Coreset Selection

This repository contains the implementation of efficient optimizations for DEFT-UCS, including Approximate Nearest Neighbor (ANN) search and L2 distance caching.

---

## 🧠 Overview

DEFT-UCS selects representative samples for efficient fine-tuning using KMeans clustering and distance-based importance sampling. The original implementation suffers from computational bottlenecks. We propose two optimizations:

1. **ANN KMeans**: Replace exact KMeans with FAISS IVF-based approximate nearest neighbor search
2. **L2 Distance Caching**: Cache L2 distances from KMeans to avoid redundant cosine distance computations
3. **Milvus Backend** (optional): Use Milvus vector database for scalable KMeans with IVF/FLAT indexes

---

## 📦 Setup

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Prepare Data

```bash
python prepare_data.py
```

This downloads the CoEDIT dataset and computes Sentence-T5 embeddings.

---

## 🚀 Quick Start

### Run End-to-End Comparison (Selection + Fine-tune + Eval)

```bash
# Run both baseline and optimized
python run_finetune_comparison.py --setting both

# Run baseline only
python run_finetune_comparison.py --setting baseline

# Run optimized only
python run_finetune_comparison.py --setting optimized

# Evaluate existing models only (skip training)
python run_finetune_comparison.py --eval-only
```

### Run Selection-Only Comparison (Time Benchmark)

```bash
python run_comparison.py
```

### Run Ablation Study

```bash
python run_ablation.py
```

### Run Milvus Backend Comparison

```bash
python run_milvus_comparison.py
```

---

## 🧱 Project Structure

```
├── config.py                    # Configuration (paths, hyperparameters)
├── prepare_data.py              # Data preparation (CoEDIT + embeddings)
├── coreset_selection.py         # Core selection algorithms (baseline/optimized)
├── run_comparison.py            # Selection-only time comparison
├── run_finetune_comparison.py   # End-to-end: selection → fine-tune → evaluation
├── run_ablation.py              # Ablation study (ANN-only, Cache-only, ANN+Cache)
├── run_milvus_comparison.py     # Milvus vector database backend comparison
├── evaluate.py                  # Evaluation metrics (SARI, BLEU, ROUGE-L, FKGL)
└── requirements.txt             # Python dependencies
```

---

## ⚙️ Key Arguments

### run_finetune_comparison.py

| Argument | Description | Default |
|----------|-------------|---------|
| `--setting` | Which setting to run: `baseline`, `optimized`, or `both` | `both` |
| `--eval-only` | Only run evaluation (skip training) | False |
| `--model` | Override model name (e.g., `google/flan-t5-large`) | `google/flan-t5-base` |
| `--experiment-dir` | Use existing experiment directory | None |

### run_comparison.py

| Argument | Description | Default |
|----------|-------------|---------|
| `--selection-ratio` | Target selection ratio | 0.325 |
| `--seed` | Random seed | 42 |

---

## 📊 Experiment Configurations

### Coreset Selection

| Parameter | Value |
|-----------|-------|
| Dataset | CoEDIT (69,071 samples) |
| Embedding | Sentence-T5-base (768-dim) |
| K (clusters) | 7 |
| Selection ratio | 32.5% |
| α (easy samples) | 0.5 |
| β (hard samples) | 0.5 |

### Fine-tuning

| Parameter | Value |
|-----------|-------|
| Model | Flan-T5-base |
| Epochs | 3 |
| Learning rate | 3e-5 |
| Batch size | 8 |
| Gradient accumulation | 4 |

### Evaluation

| Metric | Description |
|--------|-------------|
| SARI | Text simplification quality (keep/add/delete F1, 1-4 grams) |
| BLEU | N-gram overlap with references (sacrebleu) |
| ROUGE-L | Longest common subsequence F1 |
| FKGL | Flesch-Kincaid Grade Level (readability) |

---
