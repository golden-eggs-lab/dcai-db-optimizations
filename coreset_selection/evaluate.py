"""
Evaluation metrics for text simplification.

Implements:
- SARI: System output Against References and Input
- BLEU: Bilingual Evaluation Understudy
- ROUGE-L: Longest Common Subsequence based metric
- FKGL: Flesch-Kincaid Grade Level (readability)
"""

import numpy as np
from collections import Counter


def compute_sari(sources, predictions, references):
    """
    Compute SARI score (1-4 grams).
    
    SARI = (keep_score + add_score + delete_score) / 3
    
    - Keep: ngrams kept from source that should be kept (in reference)
    - Add: ngrams added that should be added (in reference, not in source)
    - Delete: ngrams deleted that should be deleted (not in reference)
    """
    def get_ngrams(text, n):
        words = text.lower().split()
        if len(words) < n:
            return Counter()
        return Counter([tuple(words[i:i+n]) for i in range(len(words)-n+1)])
    
    def precision(sys_ngrams, ref_ngrams):
        if len(sys_ngrams) == 0:
            return 0
        overlap = sum((sys_ngrams & ref_ngrams).values())
        return overlap / sum(sys_ngrams.values())
    
    def recall(sys_ngrams, ref_ngrams):
        if len(ref_ngrams) == 0:
            return 0
        overlap = sum((sys_ngrams & ref_ngrams).values())
        return overlap / sum(ref_ngrams.values())
    
    def f1(p, r):
        if p + r == 0:
            return 0
        return 2 * p * r / (p + r)
    
    sari_scores = []
    
    for src, pred, refs in zip(sources, predictions, references):
        if isinstance(refs, str):
            refs = [refs]
        
        keep_scores = []
        add_scores = []
        del_scores = []
        
        for n in range(1, 5):  # 1-4 grams
            src_ngrams = get_ngrams(src, n)
            pred_ngrams = get_ngrams(pred, n)
            
            # Average over all references
            ref_keep_scores = []
            ref_add_scores = []
            ref_del_scores = []
            
            for ref in refs:
                ref_ngrams = get_ngrams(ref, n)
                
                # Keep: ngrams in both src and pred that are also in ref
                src_pred_common = src_ngrams & pred_ngrams
                keep_p = precision(src_pred_common, ref_ngrams)
                keep_r = recall(src_pred_common, src_ngrams & ref_ngrams)
                ref_keep_scores.append(f1(keep_p, keep_r))
                
                # Add: ngrams in pred but not in src, that are in ref
                pred_only = pred_ngrams - src_ngrams
                ref_only = ref_ngrams - src_ngrams
                add_p = precision(pred_only, ref_only)
                ref_add_scores.append(add_p)
                
                # Delete: ngrams in src but not in pred, that are not in ref
                src_only = src_ngrams - pred_ngrams
                not_in_ref = src_ngrams - ref_ngrams
                del_p = precision(src_only, not_in_ref)
                ref_del_scores.append(del_p)
            
            keep_scores.append(np.mean(ref_keep_scores))
            add_scores.append(np.mean(ref_add_scores))
            del_scores.append(np.mean(ref_del_scores))
        
        # Average over n-grams
        keep = np.mean(keep_scores)
        add = np.mean(add_scores)
        delete = np.mean(del_scores)
        
        sari = (keep + add + delete) / 3 * 100
        sari_scores.append(sari)
    
    return np.mean(sari_scores)


def compute_bleu(predictions, references):
    """
    Compute corpus BLEU score using sacrebleu.
    
    Args:
        predictions: List of predicted texts
        references: List of reference lists (each sample can have multiple refs)
    """
    import sacrebleu
    
    # Transpose references for sacrebleu format
    max_refs = max(len(refs) for refs in references)
    refs_transposed = []
    for ref_idx in range(max_refs):
        ref_list = []
        for refs in references:
            if ref_idx < len(refs):
                ref_list.append(refs[ref_idx])
            else:
                ref_list.append(refs[0])  # Fallback to first ref
        refs_transposed.append(ref_list)
    
    bleu = sacrebleu.corpus_bleu(predictions, refs_transposed, force=True)
    return bleu.score


def compute_rouge_l(predictions, references):
    """
    Compute ROUGE-L score (LCS-based).
    
    For multiple references, takes the max score across all refs.
    """
    def lcs_length(x, y):
        """Compute length of longest common subsequence."""
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]
    
    def rouge_l_sentence(pred, ref):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return 0
        
        lcs = lcs_length(pred_tokens, ref_tokens)
        precision = lcs / len(pred_tokens)
        recall = lcs / len(ref_tokens)
        
        if precision + recall == 0:
            return 0
        return 2 * precision * recall / (precision + recall)
    
    scores = []
    for pred, refs in zip(predictions, references):
        if isinstance(refs, str):
            refs = [refs]
        # Take max ROUGE-L across references
        ref_scores = [rouge_l_sentence(pred, ref) for ref in refs]
        scores.append(max(ref_scores))
    
    return np.mean(scores) * 100


def compute_fkgl(texts):
    """
    Compute Flesch-Kincaid Grade Level.
    
    Lower score = easier to read.
    """
    import textstat
    return np.mean([textstat.flesch_kincaid_grade(t) for t in texts])


def evaluate_simplification(sources, predictions, references):
    """
    Compute all simplification metrics.
    
    Args:
        sources: List of original (complex) texts
        predictions: List of model predictions
        references: List of reference simplifications (each can be a list)
    
    Returns:
        Dict with BLEU, SARI, ROUGE-L, and FKGL scores
    """
    results = {}
    
    # BLEU
    try:
        results["bleu"] = compute_bleu(predictions, references)
    except Exception as e:
        print(f"BLEU computation failed: {e}")
        results["bleu"] = None
    
    # SARI
    try:
        results["sari"] = compute_sari(sources, predictions, references)
    except Exception as e:
        print(f"SARI computation failed: {e}")
        results["sari"] = None
    
    # ROUGE-L
    try:
        results["rouge_l"] = compute_rouge_l(predictions, references)
    except Exception as e:
        print(f"ROUGE-L computation failed: {e}")
        results["rouge_l"] = None
    
    # FKGL
    try:
        results["fkgl_prediction"] = compute_fkgl(predictions)
        results["fkgl_source"] = compute_fkgl(sources)
        results["fkgl_reduction"] = results["fkgl_source"] - results["fkgl_prediction"]
    except Exception as e:
        print(f"FKGL computation failed: {e}")
        results["fkgl_prediction"] = None
        results["fkgl_source"] = None
        results["fkgl_reduction"] = None
    
    return results


def print_results(results: dict):
    """Pretty print evaluation results."""
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    if results.get("bleu") is not None:
        print(f"BLEU:    {results['bleu']:.2f}")
    if results.get("sari") is not None:
        print(f"SARI:    {results['sari']:.2f}")
    if results.get("rouge_l") is not None:
        print(f"ROUGE-L: {results['rouge_l']:.2f}")
    if results.get("fkgl_prediction") is not None:
        print(f"FKGL:    {results['fkgl_prediction']:.2f}")
    
    print("="*50)
