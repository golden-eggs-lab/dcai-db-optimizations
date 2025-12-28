"""
Show experiment results: test accuracy and total time.
Usage: python show_acc.py
       python show_acc.py --ablation
"""
import json
import glob
import sys

def show_results(patterns):
    """Display results for given experiment patterns."""
    for name, pattern in patterns:
        files = glob.glob(pattern)
        if not files:
            print(f'{name}: NOT FOUND')
            continue
        with open(files[0]) as f:
            data = json.load(f)
        # Only take numeric keys (iteration numbers)
        iter_keys = [k for k in data.keys() if k.isdigit()]
        last_iter = str(max(int(k) for k in iter_keys))
        test_acc = data[last_iter]['test_results']['acc']
        total_time = sum(data[k]['iteration_total_time'] for k in iter_keys)
        print(f"{name}: test_acc={test_acc:.4f}, total_time={total_time:.1f}s")

if __name__ == '__main__':
    # Default: compare baseline vs optimized
    default_patterns = [
        ('baseline_exact', 'experiments/al_sst-2_bert_cal_baseline_exact/*/results_of_iteration.json'),
        ('optimized_ann_cache', 'experiments/al_sst-2_bert_cal_optimized_ann_cache/*/results_of_iteration.json'),
    ]
    
    # Ablation patterns
    ablation_patterns = [
        ('baseline', 'experiments/al_sst-2_bert_cal_baseline_exact/*/results_of_iteration.json'),
        ('ann_only', 'experiments/al_sst-2_bert_cal_ann_only/*/results_of_iteration.json'),
        ('cache_only', 'experiments/al_sst-2_bert_cal_cache_only/*/results_of_iteration.json'),
        ('optimized', 'experiments/al_sst-2_bert_cal_optimized_ann_cache/*/results_of_iteration.json'),
    ]
    
    if len(sys.argv) > 1 and sys.argv[1] == '--ablation':
        print("=== Ablation Study Results ===")
        show_results(ablation_patterns)
    else:
        print("=== Optimization Comparison Results ===")
        show_results(default_patterns)
