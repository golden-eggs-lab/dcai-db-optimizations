#!/bin/bash
# Ablation study: ANN-only vs Cache-only vs Full optimization
# Dataset: SST-2, Seed: 42

set -e

LOG_FILE="ablation_comparison.log"
echo "========================================" | tee $LOG_FILE
echo "Ablation Study: ANN-only vs Cache-only" | tee -a $LOG_FILE
echo "Started at: $(date)" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE

# Common parameters
DATASET="sst-2"
MODEL="bert-base-cased"
INIT="1%"
ACQ="2%"
BUDGET="15%"
EPOCHS=3
BATCH=16
SEED=42

# Function to run experiment
run_experiment() {
    local indicator=$1
    local use_ann=$2
    local cache_probs=$3
    
    echo "" | tee -a $LOG_FILE
    echo "========================================" | tee -a $LOG_FILE
    echo "Running: $indicator" | tee -a $LOG_FILE
    echo "  use_sklearn_ann=$use_ann, cache_probabilities=$cache_probs" | tee -a $LOG_FILE
    echo "  Started at: $(date)" | tee -a $LOG_FILE
    echo "========================================" | tee -a $LOG_FILE
    
    local start_time=$(date +%s)
    
    CUDA_VISIBLE_DEVICES=0 python -u run_al.py \
        --dataset_name $DATASET \
        --acquisition cal \
        --model_type bert \
        --model_name_or_path $MODEL \
        --per_gpu_train_batch_size $BATCH \
        --per_gpu_eval_batch_size 256 \
        --num_train_epochs $EPOCHS \
        --seed $SEED \
        --init_train_data $INIT \
        --acquisition_size $ACQ \
        --budget $BUDGET \
        --init random \
        --indicator $indicator \
        --use_sklearn_ann $use_ann \
        --cache_probabilities $cache_probs \
        2>&1 | tee -a $LOG_FILE
    
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    echo "SUCCESS: $indicator (seed $SEED) completed in ${elapsed}s" | tee -a $LOG_FILE
}

# 1. ANN-only (ANN + no cache)
run_experiment "ann_only" "True" "False"

# 2. Cache-only (Exact + cache)  
run_experiment "cache_only" "False" "True"

echo "" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
echo "All ablation experiments completed!" | tee -a $LOG_FILE
echo "Finished at: $(date)" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
