#!/bin/bash
# =============================================================================
# Optimization Comparison Experiment Script
# Compare: Baseline (sklearn exact KNN) vs Optimized (sklearn ANN + Reuse Prob)
# 
# Datasets: SST-2, AG News
# Settings (from original CAL paper):
#   - Initial labeled data: 1%
#   - Acquisition per iteration: 2%
#   - Total budget: 15%
#   - Model: BERT-BASE-cased (fine-tuned)
#   - 5 random seeds from [1, 9999]
# =============================================================================

set -e

# Experiment settings (from original paper)
INIT_PERCENT="1%"           # 1% initial labeled data
ACQUISITION_PERCENT="2%"    # 2% per iteration
TOTAL_BUDGET="15%"          # 15% total budget

# Random seeds for experiments (5 seeds randomly selected from [1, 9999])
SEEDS=(42 1337 2048 5678 9123)

# GPU to use
GPU=0

# Base output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_OUTPUT_DIR="experiments/optimization_comparison_${TIMESTAMP}"
mkdir -p "${BASE_OUTPUT_DIR}"

# Log file
LOG_FILE="${BASE_OUTPUT_DIR}/experiment_log.txt"
echo "Starting Optimization Comparison Experiment at $(date)" | tee -a "${LOG_FILE}"
echo "Settings: init=${INIT_PERCENT}%, acq=${ACQUISITION_PERCENT}%, budget=${TOTAL_BUDGET}%" | tee -a "${LOG_FILE}"
echo "Seeds: ${SEEDS[*]}" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"

# Datasets to run
DATASETS=("sst-2" "ag_news")

# Function to run single experiment
run_experiment() {
    local dataset=$1
    local seed=$2
    local use_ann=$3
    local cache_prob=$4
    local exp_name=$5
    
    echo "" | tee -a "${LOG_FILE}"
    echo "========================================" | tee -a "${LOG_FILE}"
    echo "Running: ${exp_name}" | tee -a "${LOG_FILE}"
    echo "  Dataset: ${dataset}, Seed: ${seed}" | tee -a "${LOG_FILE}"
    echo "  ANN: ${use_ann}, Cache Prob: ${cache_prob}" | tee -a "${LOG_FILE}"
    echo "  Started at: $(date)" | tee -a "${LOG_FILE}"
    echo "========================================" | tee -a "${LOG_FILE}"
    
    # Output directory for this experiment
    local output_dir="${BASE_OUTPUT_DIR}/${dataset}/${exp_name}/seed_${seed}"
    mkdir -p "${output_dir}"
    
    # Build command (following the same format as run_milvus_experiments.sh)
    local cmd="CUDA_VISIBLE_DEVICES=${GPU} python -u run_al.py \
        --dataset_name ${dataset} \
        --acquisition cal \
        --model_type bert \
        --model_name_or_path bert-base-cased \
        --per_gpu_train_batch_size 16 \
        --per_gpu_eval_batch_size 256 \
        --num_train_epochs 3 \
        --seed ${seed} \
        --init_train_data ${INIT_PERCENT} \
        --acquisition_size ${ACQUISITION_PERCENT} \
        --budget ${TOTAL_BUDGET} \
        --init random \
        --indicator ${exp_name} \
        --use_sklearn_ann ${use_ann} \
        --cache_probabilities ${cache_prob}"
    
    # Log command
    echo "Command: ${cmd}" | tee -a "${LOG_FILE}"
    
    # Run experiment and capture output
    local exp_log="${output_dir}/run.log"
    
    start_time=$(date +%s)
    
    if eval ${cmd} 2>&1 | tee "${exp_log}"; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "SUCCESS: ${exp_name} (seed ${seed}) completed in ${duration}s" | tee -a "${LOG_FILE}"
    else
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "FAILED: ${exp_name} (seed ${seed}) after ${duration}s" | tee -a "${LOG_FILE}"
    fi
    
    # Copy results to output directory
    # Path format: experiments/al_{dataset}_bert_cal_{indicator}/{seed}_{indicator}_cls/
    local exp_result_dir="experiments/al_${dataset}_bert_cal_${exp_name}/${seed}_${exp_name}_cls"
    if [ -d "${exp_result_dir}" ]; then
        echo "Copying results from ${exp_result_dir}" | tee -a "${LOG_FILE}"
        cp -r "${exp_result_dir}"/* "${output_dir}/" 2>/dev/null || true
    else
        # Also check without _cls suffix
        exp_result_dir="experiments/al_${dataset}_bert_cal_${exp_name}/${seed}_${exp_name}"
        if [ -d "${exp_result_dir}" ]; then
            echo "Copying results from ${exp_result_dir}" | tee -a "${LOG_FILE}"
            cp -r "${exp_result_dir}"/* "${output_dir}/" 2>/dev/null || true
        else
            echo "Warning: Result directory not found: ${exp_result_dir}" | tee -a "${LOG_FILE}"
        fi
    fi
}

# Main experiment loop
for dataset in "${DATASETS[@]}"; do
    echo "" | tee -a "${LOG_FILE}"
    echo "###############################################" | tee -a "${LOG_FILE}"
    echo "Starting experiments on ${dataset}" | tee -a "${LOG_FILE}"
    echo "###############################################" | tee -a "${LOG_FILE}"
    
    for seed in "${SEEDS[@]}"; do
        # =====================================================
        # Experiment 1: Baseline (original CAL implementation)
        # - sklearn KNeighborsClassifier (exact KNN)
        # - No probability caching
        # =====================================================
        run_experiment "${dataset}" "${seed}" "False" "False" "baseline_exact"
        
        # =====================================================
        # Experiment 2: Optimized (sklearn ANN + Reuse Prob)
        # - sklearn NearestNeighbors with ball_tree (ANN)
        # - Probability caching enabled
        # =====================================================
        run_experiment "${dataset}" "${seed}" "True" "True" "optimized_ann_cache"
    done
done

# =============================================================================
# Collect and summarize results
# =============================================================================
echo "" | tee -a "${LOG_FILE}"
echo "###############################################" | tee -a "${LOG_FILE}"
echo "Generating Results Summary" | tee -a "${LOG_FILE}"
echo "###############################################" | tee -a "${LOG_FILE}"

python3 << 'PYTHON_SCRIPT'
import os
import json
import glob
from collections import defaultdict

base_dir = os.environ.get('BASE_OUTPUT_DIR', 'experiments/optimization_comparison')
results_summary = defaultdict(lambda: defaultdict(list))

# Find all results files
for result_file in glob.glob(f"{base_dir}/**/results_of_iteration.json", recursive=True):
    parts = result_file.split('/')
    
    # Parse path to get dataset, experiment type, seed
    try:
        # Expected: base_dir/dataset/exp_name/seed_X/results_of_iteration.json
        rel_path = result_file.replace(base_dir + '/', '')
        path_parts = rel_path.split('/')
        if len(path_parts) >= 3:
            dataset = path_parts[0]
            exp_name = path_parts[1]
            seed_str = path_parts[2]
            seed = int(seed_str.replace('seed_', ''))
        else:
            continue
    except:
        continue
    
    # Load results
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
    except:
        continue
    
    # Find last iteration
    last_iter = data.get('last_iteration', None)
    if last_iter is None:
        # Find max iteration key
        iter_keys = [k for k in data.keys() if k.isdigit()]
        if iter_keys:
            last_iter = max(int(k) for k in iter_keys)
    
    if last_iter is None:
        continue
    
    last_iter_data = data.get(str(last_iter), {})
    
    # Extract metrics
    test_results = last_iter_data.get('test_results', {})
    final_acc = test_results.get('acc', None)
    
    # Get timing info
    total_selection_time = 0
    total_knn_time = 0
    for k in data.keys():
        if k.isdigit():
            iter_data = data[k]
            if isinstance(iter_data, dict):
                total_selection_time += iter_data.get('selection_time', 0)
                total_knn_time += iter_data.get('knn_build_time', 0) + iter_data.get('knn_search_time', 0)
    
    if final_acc is not None:
        key = f"{dataset}_{exp_name}"
        results_summary[key]['final_acc'].append(final_acc)
        results_summary[key]['total_selection_time'].append(total_selection_time)
        results_summary[key]['seeds'].append(seed)

# Print summary
print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

for key in sorted(results_summary.keys()):
    data = results_summary[key]
    accs = data['final_acc']
    times = data['total_selection_time']
    seeds = data['seeds']
    
    if accs:
        import statistics
        mean_acc = statistics.mean(accs)
        std_acc = statistics.stdev(accs) if len(accs) > 1 else 0
        mean_time = statistics.mean(times)
        
        print(f"\n{key}:")
        print(f"  Seeds: {seeds}")
        print(f"  Final Test Accuracy: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
        print(f"  Total Selection Time: {mean_time:.2f}s")
        print(f"  Individual results:")
        for i, (s, a, t) in enumerate(zip(seeds, accs, times)):
            print(f"    Seed {s}: acc={a*100:.2f}%, sel_time={t:.2f}s")

# Save summary to JSON
summary_file = f"{base_dir}/results_summary.json"
with open(summary_file, 'w') as f:
    json.dump(dict(results_summary), f, indent=2)
print(f"\nResults saved to: {summary_file}")
PYTHON_SCRIPT

echo "" | tee -a "${LOG_FILE}"
echo "###############################################" | tee -a "${LOG_FILE}"
echo "All experiments completed at $(date)" | tee -a "${LOG_FILE}"
echo "Results saved in: ${BASE_OUTPUT_DIR}" | tee -a "${LOG_FILE}"
echo "###############################################" | tee -a "${LOG_FILE}"
