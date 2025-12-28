"""
End-to-end comparison: Selection → Fine-tune → Evaluation

Runs complete pipeline for both baseline and optimized selection.

Usage:
    python run_finetune_comparison.py --setting baseline
    python run_finetune_comparison.py --setting optimized
    python run_finetune_comparison.py --setting both
    python run_finetune_comparison.py --eval-only  # Re-evaluate saved models
"""

import os
import json
import argparse
import time
from pathlib import Path

import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset, load_dataset
from tqdm import tqdm

from config import (
    CACHE_DIR, ARTIFACTS_DIR, 
    SELECTION_CONFIG, TRAINING_CONFIG,
    MODEL_NAME,
)
from coreset_selection import baseline_selection, optimized_selection
from evaluate import evaluate_simplification, print_results


def run_selection(setting: str, vectors: np.ndarray, all_samples: list):
    """Run coreset selection based on setting."""
    K = SELECTION_CONFIG["K"]
    A = SELECTION_CONFIG["A"]
    seed = SELECTION_CONFIG["seed"]
    
    if setting == "baseline":
        selected_indices, timing = baseline_selection(vectors, K, A, seed=seed)
    else:
        selected_indices, timing = optimized_selection(vectors, K, A, seed=seed)
    
    samples = [all_samples[i] for i in selected_indices]
    return samples, timing


def prepare_dataset(samples, tokenizer, max_input_length, max_target_length):
    """Prepare HuggingFace dataset for training."""
    inputs = [s["src"] for s in samples]
    targets = [s["tgt"] for s in samples]
    
    dataset = Dataset.from_dict({
        "input_text": inputs,
        "target_text": targets,
    })
    
    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["input_text"],
            max_length=max_input_length,
            truncation=True,
            padding="max_length",
        )
        
        labels = tokenizer(
            examples["target_text"],
            max_length=max_target_length,
            truncation=True,
            padding="max_length",
        )
        
        # Replace padding token id with -100
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    
    return tokenized


def finetune(setting: str, samples: list):
    """Fine-tune model on selected samples."""
    print(f"\n{'='*60}")
    print(f"FINE-TUNING: {setting.upper()}")
    print(f"{'='*60}")
    
    # Split into train/val (90/10)
    np.random.seed(SELECTION_CONFIG["seed"])
    indices = np.random.permutation(len(samples))
    split = int(0.9 * len(samples))
    train_samples = [samples[i] for i in indices[:split]]
    val_samples = [samples[i] for i in indices[split:]]
    
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    
    # Load model and tokenizer
    print(f"\nLoading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    # Prepare datasets
    train_dataset = prepare_dataset(
        train_samples, tokenizer,
        TRAINING_CONFIG["max_input_length"],
        TRAINING_CONFIG["max_target_length"],
    )
    val_dataset = prepare_dataset(
        val_samples, tokenizer,
        TRAINING_CONFIG["max_input_length"],
        TRAINING_CONFIG["max_target_length"],
    )
    
    # Output directory
    output_dir = ARTIFACTS_DIR / "finetune" / setting
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=TRAINING_CONFIG["per_device_eval_batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        warmup_ratio=TRAINING_CONFIG["warmup_ratio"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        fp16=TRAINING_CONFIG["fp16"] and torch.cuda.is_available(),
        logging_steps=TRAINING_CONFIG["logging_steps"],
        save_strategy=TRAINING_CONFIG["save_strategy"],
        eval_strategy=TRAINING_CONFIG["eval_strategy"],
        save_total_limit=1,
        load_best_model_at_end=True,
        predict_with_generate=True,
        generation_max_length=TRAINING_CONFIG["max_target_length"],
        seed=SELECTION_CONFIG["seed"],
    )
    
    # Trainer
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train
    print("\nStarting training...")
    start_time = time.time()
    trainer.train()
    train_time = time.time() - start_time
    
    # Save model
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    
    print(f"\nTraining complete in {train_time/60:.1f} minutes")
    print(f"Model saved to: {final_path}")
    
    return model, tokenizer, train_time


def load_test_data():
    """Load TurkCorpus test set."""
    print("Loading TurkCorpus test set...")
    turk = load_dataset("turk", split="test")
    
    sources = []
    references = []
    for item in turk:
        sources.append(item["original"])
        references.append(item["simplifications"])
    
    print(f"Loaded {len(sources)} test samples")
    return sources, references


def evaluate_model(model, tokenizer, setting: str, sources, references):
    """Evaluate model on text simplification."""
    print(f"\n--- Evaluating {setting} ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    predictions = []
    batch_size = 8
    
    for i in tqdm(range(0, len(sources), batch_size), desc=f"Generating ({setting})"):
        batch = sources[i:i+batch_size]
        
        # IMPORTANT: Add "Simplify: " prefix (CoEDIT format)
        inputs = [f"Simplify: {src}" for src in batch]
        
        tokenized = tokenizer(
            inputs,
            max_length=256,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **tokenized,
                max_new_tokens=256,
                num_beams=4,
                do_sample=False,
                early_stopping=True,
            )
        
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(decoded)
    
    # Compute metrics
    results = evaluate_simplification(sources, predictions, references)
    results["setting"] = setting
    
    return predictions, results


def run_end_to_end(setting: str):
    """Run complete pipeline for a setting."""
    print(f"\n{'='*60}")
    print(f"END-TO-END PIPELINE: {setting.upper()}")
    print(f"{'='*60}")
    
    # Load data
    embeddings = np.load(CACHE_DIR / "coedit_embeddings.npy").astype(np.float32)
    with open(CACHE_DIR / "coedit_dataset.json", 'r') as f:
        all_samples = json.load(f)
    
    # Selection
    print("\n[Step 1/3] Coreset Selection")
    samples, selection_timing = run_selection(setting, embeddings, all_samples)
    
    # Save samples
    output_dir = ARTIFACTS_DIR / "finetune" / setting
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "selected_samples.json", 'w') as f:
        json.dump(samples, f)
    
    # Fine-tune
    print("\n[Step 2/3] Fine-tuning")
    model, tokenizer, train_time = finetune(setting, samples)
    
    # Evaluate
    print("\n[Step 3/3] Evaluation")
    sources, references = load_test_data()
    predictions, results = evaluate_model(model, tokenizer, setting, sources, references)
    
    # Add timing info
    results["selection_time"] = selection_timing["total"]
    results["train_time"] = train_time
    results["n_samples"] = len(samples)
    
    # Save results
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(output_dir / "predictions.json", 'w') as f:
        json.dump(predictions, f, indent=2)
    
    return results


def eval_only(setting: str):
    """Re-evaluate a saved model."""
    model_path = ARTIFACTS_DIR / "finetune" / setting / "final"
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return None
    
    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    sources, references = load_test_data()
    predictions, results = evaluate_model(model, tokenizer, setting, sources, references)
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=["baseline", "optimized", "both"], default="both")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate saved models")
    args = parser.parse_args()
    
    if args.eval_only:
        print("Re-evaluating saved models...")
        for setting in ["baseline", "optimized"]:
            results = eval_only(setting)
            if results:
                print_results(results)
    else:
        if args.setting == "both":
            settings = ["baseline", "optimized"]
        else:
            settings = [args.setting]
        
        all_results = {}
        for setting in settings:
            results = run_end_to_end(setting)
            all_results[setting] = results
            print_results(results)
        
        # Compare if both ran
        if len(all_results) == 2:
            print("\n" + "="*60)
            print("COMPARISON SUMMARY")
            print("="*60)
            
            baseline = all_results["baseline"]
            optimized = all_results["optimized"]
            
            print(f"Selection Speedup: {baseline['selection_time']/optimized['selection_time']:.2f}x")
            print(f"\nMetric Comparison:")
            print(f"  BLEU:    {baseline.get('bleu', 'N/A')} vs {optimized.get('bleu', 'N/A')}")
            print(f"  SARI:    {baseline.get('sari', 'N/A')} vs {optimized.get('sari', 'N/A')}")
            print(f"  ROUGE-L: {baseline.get('rouge_l', 'N/A')} vs {optimized.get('rouge_l', 'N/A')}")
            print(f"  FKGL:    {baseline.get('fkgl_prediction', 'N/A')} vs {optimized.get('fkgl_prediction', 'N/A')}")


if __name__ == "__main__":
    main()
