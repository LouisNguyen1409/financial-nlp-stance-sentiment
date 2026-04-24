import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR


def compute_metrics(y_true, y_pred, label_names):
    """
    Compute per-class F1, macro-F1, and accuracy.

    Args:
        y_true: list/array of ground-truth integer labels
        y_pred: list/array of predicted integer labels
        label_names: list of human-readable label strings

    Returns:
        dict with 'accuracy', 'macro_f1', 'per_class_f1', and 'report' (str)
    """
    report_str = classification_report(
        y_true, y_pred, target_names=label_names, digits=4, zero_division=0
    )
    per_class_f1 = f1_score(
        y_true, y_pred, average=None, labels=range(len(label_names)), zero_division=0
    )
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class_f1": {
            label_names[i]: round(f, 4) for i, f in enumerate(per_class_f1)
        },
        "report": report_str,
    }


def print_classification_report(metrics, model_name, task_name):
    """Pretty-print evaluation results to the console."""
    print(f"\n{'─'*60}")
    print(f"  {model_name} — {task_name}")
    print(f"{'─'*60}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Macro-F1 : {metrics['macro_f1']:.4f}")
    for label, f1 in metrics["per_class_f1"].items():
        print(f"    {label:>10}: F1 = {f1:.4f}")
    print()
    print(metrics["report"])


def plot_confusion_matrix(y_true, y_pred, label_names, model_name, task_name,
                          save_dir=None):
    """
    Generate and save a confusion matrix heatmap.

    The figure is saved to results/<model_name>_<task_name>_cm.png.
    """
    save_dir = save_dir or RESULTS_DIR
    os.makedirs(save_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=range(len(label_names)))
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{model_name} — {task_name}")
    plt.tight_layout()

    fname = f"{model_name.replace(' ', '_')}_{task_name}_cm.png"
    path = os.path.join(save_dir, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [EVAL] Confusion matrix saved to {path}")
    return path


def error_analysis(texts, y_true, y_pred, label_names, top_n=20):
    """
    Analyse misclassified examples.

    Returns:
        - A DataFrame of misclassified examples (sorted by frequency of error type)
        - A summary dict of error-type counts (e.g. "neutral → hawkish": 12)
    """
    errors = []
    for text, true, pred in zip(texts, y_true, y_pred):
        if true != pred:
            errors.append({
                "text": text,
                "true_label": label_names[true],
                "pred_label": label_names[pred],
                "error_type": f"{label_names[true]} → {label_names[pred]}",
            })

    if not errors:
        print("  [EVAL] No misclassifications found!")
        return pd.DataFrame(), {}

    error_df = pd.DataFrame(errors)
    error_counts = error_df["error_type"].value_counts().to_dict()

    print(f"\n  Misclassification breakdown ({len(errors)} total errors):")
    for etype, count in error_counts.items():
        print(f"    {etype}: {count}")

    # Return the top-N most "interesting" errors (longest texts first, as they
    # tend to be the ambiguous ones)
    error_df["text_len"] = error_df["text"].str.len()
    top_errors = error_df.sort_values("text_len", ascending=False).head(top_n)
    return top_errors.drop(columns=["text_len"]), error_counts


def save_results(results_dict, filename):
    """Save a results dictionary to JSON in the results/ folder."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(results_dict, f, indent=2, default=str)
    print(f"  [EVAL] Results saved to {path}")
