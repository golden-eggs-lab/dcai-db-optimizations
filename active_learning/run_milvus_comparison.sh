#!/bin/bash

# Milvus Backend Comparison Script
# Same configuration as in-memory experiments (baseline_exact vs optimized_ann_cache)
# Config: 1% init, 2% acquisition, 15% budget
#
# Comparison:
#   1. milvus_flat_baseline - Milvus FLAT (exact) + no prob cache
#   2. milvus_ivf_optimized - Milvus IVF_FLAT (approx) + prob cache
#
# Usage: bash run_milvus_comparison.sh [gpu_id]

GPU=${1:-0}

SEEDS=(42)
DATASET="sst-2"

# Common arguments
COMMON_ARGS="--dataset_name $DATASET \
  --acquisition cal \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 256 \
  --num_train_epochs 3 \
  --learning_rate 2e-05 \
  --init_train_data 1% \
  --acquisition_size 2% \
  --budget 15% \
  --init random \
  --use_milvus True"

echo "=========================================="
echo "Milvus Backend Comparison"
echo "Dataset: $DATASET"
echo "Seeds: ${SEEDS[@]}"
echo "GPU: $GPU"
echo "Config: 1% init, 2% acquisition, 15% budget"
echo ""
echo "Baseline: Milvus FLAT (exact) + no prob cache"
echo "Optimized: Milvus IVF_FLAT (approx) + prob cache"
echo "=========================================="

LOGFILE="milvus_comparison.log"

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Seed: $SEED"
    echo "=========================================="
    
    # 1. Milvus FLAT (exact) + no prob cache (baseline)
    echo ""
    echo "[1/2] Running Milvus FLAT (exact) + no cache - seed $SEED"
    CUDA_VISIBLE_DEVICES=$GPU python -u run_al.py \
      $COMMON_ARGS \
      --seed $SEED \
      --milvus_index_type FLAT \
      --cache_probabilities False \
      --indicator milvus_flat_baseline \
      2>&1 | tee -a $LOGFILE
    
    # 2. Milvus IVF_FLAT (approx) + prob cache (optimized)
    echo ""
    echo "[2/2] Running Milvus IVF_FLAT (approx) + cache - seed $SEED"
    CUDA_VISIBLE_DEVICES=$GPU python -u run_al.py \
      $COMMON_ARGS \
      --seed $SEED \
      --milvus_index_type IVF_FLAT \
      --milvus_nlist 100 \
      --milvus_nprobe 10 \
      --cache_probabilities True \
      --indicator milvus_ivf_optimized \
      2>&1 | tee -a $LOGFILE
done

echo ""
echo "=========================================="
echo "All Milvus experiments completed!"
echo "Results saved in:"
echo "  experiments/al_sst-2_bert_cal_milvus_flat_baseline/"
echo "  experiments/al_sst-2_bert_cal_milvus_ivf_optimized/"
echo "=========================================="
