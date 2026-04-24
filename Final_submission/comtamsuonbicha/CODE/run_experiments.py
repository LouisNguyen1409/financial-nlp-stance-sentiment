import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR, DEVICE, SENTIMENT_LABELS, STANCE_LABELS

os.makedirs(RESULTS_DIR, exist_ok=True)


def step1_load_data():
    """Step 1: Load and inspect both datasets."""
    from src.data_loader import load_financial_phrasebank, load_fomc_dataset
    print("\n" + "="*70)
    print("  STEP 1: LOADING DATASETS")
    print("="*70)
    fpb = load_financial_phrasebank()
    fomc = load_fomc_dataset()
    return fomc, fpb


def step2_baseline(fomc, fpb):
    """Step 2: TF-IDF + Logistic Regression baseline on both tasks."""
    from src.baseline import train_and_evaluate_baseline, run_alternative_baselines
    print("\n" + "="*70)
    print("  STEP 2: BASELINE (TF-IDF + Logistic Regression)")
    print("="*70)

    stance_metrics, _ = train_and_evaluate_baseline(
        fomc["train"], fomc["test"], STANCE_LABELS, "stance"
    )
    sentiment_metrics, _ = train_and_evaluate_baseline(
        fpb["train"], fpb["test"], SENTIMENT_LABELS, "sentiment"
    )

    print("\n" + "="*70)
    print("  STEP 2: ALTERNATIVE BASELINES")
    print("="*70)
    alt_results = run_alternative_baselines(fomc, fpb)

    return {
        "baseline_stance": stance_metrics,
        "baseline_sentiment": sentiment_metrics,
        **alt_results,
    }


def step2b_lexicon(fomc, fpb):
    """Step 2b: Loughran-McDonald lexicon-based classification."""
    from src.lexicon import run_lexicon_experiments
    print("\n" + "="*70)
    print("  STEP 2b: LOUGHRAN-MCDONALD LEXICON")
    print("="*70)
    return run_lexicon_experiments(fomc, fpb)


def step3_pretrained(fomc, fpb):
    """Step 3: Zero-shot and few-shot evaluation of pre-trained models."""
    from src.pretrained_eval import run_all_pretrained_evaluations
    print("\n" + "="*70)
    print("  STEP 3: PRE-TRAINED MODEL EVALUATION")
    print("="*70)
    return run_all_pretrained_evaluations(fomc, fpb)


def step4_finetune(fomc, fpb):
    """Step 4: Fine-tune FinBERT on each dataset separately."""
    from src.finetune_fineBert import finetune_finbert
    print("\n" + "="*70)
    print("  STEP 4: SINGLE-TASK FINE-TUNING")
    print("="*70)

    stance_metrics, _, _ = finetune_finbert(
        fomc["train"], fomc["val"], fomc["test"],
        STANCE_LABELS, "stance", use_weighted_loss=True,
    )
    sentiment_metrics, _, _ = finetune_finbert(
        fpb["train"], fpb["val"], fpb["test"],
        SENTIMENT_LABELS, "sentiment", use_weighted_loss=False,
    )
    return {"finetune_stance": stance_metrics, "finetune_sentiment": sentiment_metrics}


def step5_multitask(fomc, fpb):
    """Step 5: Multi-task training on both datasets."""
    from src.multitask import train_multitask
    print("\n" + "="*70)
    print("  STEP 5: MULTI-TASK TRAINING")
    print("="*70)
    results, _, _ = train_multitask(fomc, fpb)
    return results


def step6_finetune_bert_llrd(fomc, fpb):
    """Step 6: BERT-base-uncased with Layer-wise LR Decay + Gradual Unfreezing."""
    from src.finetune_bert import finetune_bert_llrd
    print("\n" + "="*70)
    print("  STEP 6: BERT-base LLRD + GRADUAL UNFREEZING")
    print("="*70)

    stance_metrics, _, _ = finetune_bert_llrd(
        fomc["train"], fomc["val"], fomc["test"],
        STANCE_LABELS, "stance",
    )
    sentiment_metrics, _, _ = finetune_bert_llrd(
        fpb["train"], fpb["val"], fpb["test"],
        SENTIMENT_LABELS, "sentiment",
    )
    return {
        "bert_llrd_stance": stance_metrics,
        "bert_llrd_sentiment": sentiment_metrics,
    }


def print_summary(all_results):
    """Print a summary comparison table of all experiment results."""
    print("\n" + "="*70)
    print("  SUMMARY OF ALL RESULTS")
    print("="*70)
    print(f"\n  {'Model':<35} {'Task':<12} {'Acc':>6} {'Macro-F1':>9}")
    print(f"  {'─'*35} {'─'*12} {'─'*6} {'─'*9}")

    for key, metrics in sorted(all_results.items()):
        if isinstance(metrics, dict) and "accuracy" in metrics:
            # Infer task from key name
            task = "stance" if "stance" in key else "sentiment"
            model = key.replace(f"_{task}", "").replace("_", " ").title()
            print(f"  {model:<35} {task:<12} {metrics['accuracy']:>6.4f} {metrics['macro_f1']:>9.4f}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Run Financial NLP experiments")
    parser.add_argument(
        "--step", type=int, default=0,
        help="Run a specific step (1-6). 0 = run all steps.",
    )
    args = parser.parse_args()

    print(f"\n  Device: {DEVICE}")
    print(f"  Results directory: {RESULTS_DIR}\n")

    # Step 1 is always needed
    fomc, fpb = step1_load_data()
    all_results = {}

    if args.step == 0 or args.step == 2:
        results = step2_baseline(fomc, fpb)
        all_results.update(results)
        results = step2b_lexicon(fomc, fpb)
        all_results.update(results)

    if args.step == 0 or args.step == 3:
        results = step3_pretrained(fomc, fpb)
        all_results.update(results)

    if args.step == 0 or args.step == 4:
        results = step4_finetune(fomc, fpb)
        all_results.update(results)

    if args.step == 0 or args.step == 5:
        results = step5_multitask(fomc, fpb)
        all_results.update(results)

    if args.step == 0 or args.step == 6:
        results = step6_finetune_bert_llrd(fomc, fpb)
        all_results.update(results)

    if all_results:
        print_summary(all_results)
        # Save combined results
        from src.evaluate import save_results
        save_results(all_results, "all_results_summary.json")


if __name__ == "__main__":
    main()