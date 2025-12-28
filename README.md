# Database-Inspired Optimizations for Data-Centric AI

This repository contains efficient optimization implementations for three core Data-Centric AI (DCAI) tasks using database-inspired techniques including Approximate Nearest Neighbor (ANN) search, intermediate results reuse, and Milvus integration.

---

## 🧠 Overview

Data-Centric AI focuses on improving data quality and efficiency rather than model architecture. Many DCAI algorithms involve computationally expensive operations like nearest neighbor search and distance computations. We propose unified optimization strategies across three fundamental tasks:

| Task | Original Bottleneck | Our Optimizations |
|------|---------------------|-------------------|
| **Coreset Selection** | Exact KMeans clustering | ANN KMeans + L2 Distance Reuse |
| **Active Learning** | Exact KNN search | Ball-tree ANN + Probability Reuse |
| **Semantic Deduplication** | O(n²) pairwise comparison | IVF ANN + Search Results Reuse |

### 🔑 Key Techniques

1. **Approximate Nearest Neighbor (ANN) Search**: Replace exact search with efficient index structures (IVF, Ball-tree, HNSW)
2. **Intermediate Results Reuse**: Reuse intermediate results (distances, probabilities, search terms) to avoid redundant computations
3. **Milvus Integration**: Optional vector database backend for scalable, production-ready deployment

---

## 📁 Repository Structure

```
├── README.md                      # This file (overview)
├── coreset_selection/             # Coreset Selection optimizations
├── active_learning/               # Active Learning optimizations
└── semantic_dedup/                # Semantic Deduplication optimizations
```

---

## 📂 Task-Specific Implementations

### 1️⃣ Coreset Selection (`coreset_selection/`)

**Task**: Select representative samples for efficient fine-tuning using unsupervised coreset selection.

**Algorithm**: DEFT-UCS (Unsupervised Coreset Selection)

**Optimizations**:
- ANN KMeans via FAISS IVF index
- L2 distance reuse to avoid redundant cosine computations
- Milvus backend for scalable deployment

👉 [**View Coreset Selection README**](coreset_selection/README.md)

---

### 2️⃣ Active Learning (`active_learning/`)

**Task**: Iteratively select the most informative samples for annotation to maximize model performance with minimal labeling cost.

**Algorithm**: CAL (Contrastive Active Learning)

**Optimizations**:
- Ball-tree based ANN search
- Softmax probability reuse for labeled samples
- Milvus backend with IVF/HNSW indexes

👉 [**View Active Learning README**](active_learning/README.md)

---

### 3️⃣ Semantic Deduplication (`semantic_dedup/`)

**Task**: Remove near-duplicate samples from training data by comparing embedding similarities.

**Algorithms**:
- **SemDeDup**: Greedy sequential deduplication (position-biased)
- **FairDeDup**: Union-Find based symmetric deduplication (fair)

**Optimizations**:
- FAISS IVF-based ANN search
- Search results reuse across multiple epsilon thresholds
- Milvus Lite for local vector database operations

👉 [**View Semantic Deduplication README**](semantic_dedup/README.md)

---

## 🚀 Quick Start

Each task has its own setup and execution instructions. Navigate to the corresponding folder:

```bash
# Coreset Selection
cd coreset_selection && pip install -r requirements.txt

# Active Learning
cd active_learning && pip install -r requirements.txt

# Semantic Deduplication
cd semantic_dedup && conda env create -f environment.yml
```

---