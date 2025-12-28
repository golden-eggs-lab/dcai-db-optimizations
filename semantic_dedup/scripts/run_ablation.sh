#!/bin/bash
# Run ablation study on CIFAR-10

set -e

echo "=============================================="
echo "FairDeDup Ablation Study"
echo "=============================================="

SEED=42

# Run 3 times with different seeds for statistical significance
for run_id in run1 run2 run3; do
    echo ""
    echo "Running FAISS experiment: $run_id"
    python experiments/run_experiment.py \
        --seed $SEED \
        --run-id $run_id
done

echo ""
echo "Running Milvus experiment..."
python experiments/run_milvus_experiment.py \
    --seed $SEED \
    --run-id milvus

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "=============================================="
