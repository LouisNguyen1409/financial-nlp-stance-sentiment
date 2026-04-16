"""
Comprehensive data analysis for the Financial NLP project.
Generates visualizations and statistics for the report and presentation.

Usage:
    python data_analysis.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR, SEED, SENTIMENT_LABELS, STANCE_LABELS

ANALYSIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis")
os.makedirs(ANALYSIS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = sns.color_palette("Set2")
BLUE = "#2E86C1"
DARK_BLUE = "#1B3A5C"
GOLD = "#F39C12"
GREEN = "#27AE60"
RED = "#E74C3C"


# ──────────────────────────────────────────────────────────────────────────────
# 1. Load datasets and compute statistics
# ──────────────────────────────────────────────────────────────────────────────

def load_datasets():
    from src.data_loader import load_financial_phrasebank, load_fomc_dataset
    fpb = load_financial_phrasebank()
    fomc = load_fomc_dataset()
    return fpb, fomc


def dataset_statistics(fpb, fomc):
    """Print and save dataset statistics."""
    print("\n" + "=" * 70)
    print("  DATASET STATISTICS")
    print("=" * 70)

    stats = {}
    for name, ds, labels in [("FPB (Sentiment)", fpb, SENTIMENT_LABELS),
                              ("FOMC (Stance)", fomc, STANCE_LABELS)]:
        print(f"\n  {name}:")
        for split in ["train", "val", "test"]:
            data = ds[split]
            texts = data["text"]
            lengths = [len(t.split()) for t in texts]
            label_counts = Counter(data["label"])
            print(f"    {split}: {len(data)} samples, "
                  f"avg_len={np.mean(lengths):.1f} words, "
                  f"min={min(lengths)}, max={max(lengths)}, "
                  f"median={np.median(lengths):.0f}")
            for i, l in enumerate(labels):
                print(f"      {l}: {label_counts.get(i, 0)} ({label_counts.get(i,0)/len(data)*100:.1f}%)")

        # Combined stats
        all_texts = list(ds["train"]["text"]) + list(ds["val"]["text"]) + list(ds["test"]["text"])
        all_labels = list(ds["train"]["label"]) + list(ds["val"]["label"]) + list(ds["test"]["label"])
        all_lengths = [len(t.split()) for t in all_texts]
        stats[name] = {
            "total": len(all_texts),
            "labels": labels,
            "label_counts": Counter(all_labels),
            "lengths": all_lengths,
            "texts": all_texts,
            "all_labels": all_labels,
        }

    return stats


# ──────────────────────────────────────────────────────────────────────────────
# 2. Class distribution plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_class_distributions(stats):
    """Bar charts showing class distribution for both datasets."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (name, s) in zip(axes, stats.items()):
        labels = s["labels"]
        counts = [s["label_counts"][i] for i in range(len(labels))]
        total = sum(counts)
        pcts = [c / total * 100 for c in counts]

        colors = [BLUE, GOLD, GREEN] if "Sentiment" in name else [GREEN, RED, BLUE]
        bars = ax.bar(labels, counts, color=colors, edgecolor="white", linewidth=1.5)

        for bar, pct, count in zip(bars, pcts, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                    f"{count}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=11)

        ax.set_title(name, fontsize=14, fontweight="bold")
        ax.set_ylabel("Number of Samples")
        ax.set_ylim(0, max(counts) * 1.25)

    plt.tight_layout()
    path = os.path.join(ANALYSIS_DIR, "class_distribution.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Text length distribution
# ──────────────────────────────────────────────────────────────────────────────

def plot_text_lengths(stats):
    """Histogram of text lengths per class for both datasets."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (name, s) in zip(axes, stats.items()):
        labels = s["labels"]
        colors = [BLUE, GOLD, GREEN] if "Sentiment" in name else [GREEN, RED, BLUE]

        for i, (label, color) in enumerate(zip(labels, colors)):
            lengths = [len(t.split()) for t, l in zip(s["texts"], s["all_labels"]) if l == i]
            ax.hist(lengths, bins=30, alpha=0.5, label=f"{label} (n={len(lengths)}, avg={np.mean(lengths):.0f})",
                    color=color, edgecolor="white")

        ax.set_title(f"{name} — Text Length Distribution", fontsize=13, fontweight="bold")
        ax.set_xlabel("Number of Words")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(ANALYSIS_DIR, "text_length_distribution.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 4. Top words per class (TF-IDF feature importance)
# ──────────────────────────────────────────────────────────────────────────────

def plot_top_words(stats):
    """Top discriminative words per class using TF-IDF."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for row, (name, s) in enumerate(stats.items()):
        labels = s["labels"]
        texts = s["texts"]
        all_labels = s["all_labels"]

        tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2),
                                 stop_words="english", sublinear_tf=True)
        X = tfidf.fit_transform(texts)
        feature_names = tfidf.get_feature_names_out()

        for col, (i, label) in enumerate(zip(range(len(labels)), labels)):
            ax = axes[row][col]
            mask = np.array(all_labels) == i
            mean_tfidf = np.asarray(X[mask].mean(axis=0)).flatten()
            top_idx = mean_tfidf.argsort()[-15:][::-1]
            top_words = [feature_names[j] for j in top_idx]
            top_scores = [mean_tfidf[j] for j in top_idx]

            colors_bar = [BLUE, GOLD, GREEN][col]
            ax.barh(range(len(top_words)), top_scores, color=colors_bar, alpha=0.8)
            ax.set_yticks(range(len(top_words)))
            ax.set_yticklabels(top_words, fontsize=9)
            ax.invert_yaxis()
            ax.set_title(f"{name}\n{label}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Mean TF-IDF Score")

    plt.tight_layout()
    path = os.path.join(ANALYSIS_DIR, "top_words_per_class.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 5. Model comparison charts
# ──────────────────────────────────────────────────────────────────────────────

def load_all_results():
    """Load all result JSON files."""
    results = {}
    for f in os.listdir(RESULTS_DIR):
        if f.endswith(".json") and f not in ("summary.json", "all_results_summary.json"):
            with open(os.path.join(RESULTS_DIR, f)) as fh:
                data = json.load(fh)
                results[f.replace(".json", "")] = data
    return results


def plot_model_comparison():
    """Grouped bar chart comparing all models on both tasks."""
    results = load_all_results()

    # Define model order (progression)
    model_order = [
        ("LM Lexicon", "lexicon_rules"),
        ("TF-IDF+LR", "baseline"),
        ("TF-IDF+SVM", "tfidf_svm"),
        ("TF-IDF(tri)+LR", "tfidf_trigram_lr"),
        ("TF-IDF+Lexicon", "lexicon_tfidf"),
        ("GloVe+LR", "glove_lr"),
        ("GloVe+SVM", "glove_svm"),
        ("FinBERT 0-shot", "zeroshot_finbert_native"),
        ("FinBERT few-shot", "fewshot_finbert"),
        ("BERT few-shot", "fewshot_bert-base-uncased"),
        ("RoBERTa few-shot", "fewshot_roberta-base"),
        ("FinBERT fine-tune", "finetune_finbert"),
        ("BERT LLRD", "finetune_bert_llrd"),
        ("Multi-task", "multitask"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    for ax, (task, task_labels) in zip(axes, [("sentiment", SENTIMENT_LABELS),
                                                ("stance", STANCE_LABELS)]):
        names = []
        f1_scores = []
        acc_scores = []
        for display_name, key_prefix in model_order:
            key = f"{key_prefix}_{task}"
            if key in results:
                names.append(display_name)
                f1_scores.append(results[key]["macro_f1"])
                acc_scores.append(results[key]["accuracy"])

        x = np.arange(len(names))
        width = 0.35

        bars1 = ax.bar(x - width/2, acc_scores, width, label="Accuracy",
                       color=BLUE, alpha=0.85, edgecolor="white")
        bars2 = ax.bar(x + width/2, f1_scores, width, label="Macro-F1",
                       color=GOLD, alpha=0.85, edgecolor="white")

        # Highlight best
        best_acc_idx = np.argmax(acc_scores)
        best_f1_idx = np.argmax(f1_scores)
        bars1[best_acc_idx].set_edgecolor(RED)
        bars1[best_acc_idx].set_linewidth(2.5)
        bars2[best_f1_idx].set_edgecolor(RED)
        bars2[best_f1_idx].set_linewidth(2.5)

        ax.set_ylabel("Score")
        ax.set_title(f"{'Sentiment' if task == 'sentiment' else 'Stance'} Classification — All Models",
                     fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
        ax.legend(loc="lower right")
        ax.set_ylim(0, 1.1)

        # Add value labels on best bars
        for idx in [best_acc_idx, best_f1_idx]:
            ax.annotate(f"{acc_scores[idx]:.3f}", (x[idx] - width/2, acc_scores[idx]),
                       ha="center", va="bottom", fontsize=8, fontweight="bold", color=RED)

    plt.tight_layout()
    path = os.path.join(ANALYSIS_DIR, "model_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. Per-class F1 heatmap across models
# ──────────────────────────────────────────────────────────────────────────────

def plot_per_class_f1_heatmap():
    """Heatmap of per-class F1 scores across all models."""
    results = load_all_results()

    model_order = [
        ("LM Lexicon", "lexicon_rules"),
        ("TF-IDF+LR", "baseline"),
        ("TF-IDF+SVM", "tfidf_svm"),
        ("FinBERT 0-shot", "zeroshot_finbert_native"),
        ("FinBERT few-shot", "fewshot_finbert"),
        ("BERT few-shot", "fewshot_bert-base-uncased"),
        ("FinBERT fine-tune", "finetune_finbert"),
        ("BERT LLRD", "finetune_bert_llrd"),
        ("Multi-task", "multitask"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, (task, labels) in zip(axes, [("sentiment", SENTIMENT_LABELS),
                                          ("stance", STANCE_LABELS)]):
        data = []
        model_names = []
        for display_name, key_prefix in model_order:
            key = f"{key_prefix}_{task}"
            if key in results and "per_class_f1" in results[key]:
                pcf1 = results[key]["per_class_f1"]
                row = [pcf1.get(l, 0) for l in labels]
                data.append(row)
                model_names.append(display_name)

        df = pd.DataFrame(data, index=model_names, columns=labels)
        sns.heatmap(df, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax,
                    vmin=0, vmax=1, linewidths=0.5, cbar_kws={"shrink": 0.8})
        ax.set_title(f"Per-Class F1 — {'Sentiment' if task == 'sentiment' else 'Stance'}",
                     fontsize=13, fontweight="bold")
        ax.set_ylabel("")

    plt.tight_layout()
    path = os.path.join(ANALYSIS_DIR, "per_class_f1_heatmap.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 7. Progression chart (baseline → pretrained → finetune → multitask)
# ──────────────────────────────────────────────────────────────────────────────

def plot_progression():
    """Line chart showing performance progression across modelling stages."""
    results = load_all_results()

    stages = [
        ("Lexicon\n(rule)", "lexicon_rules"),
        ("TF-IDF\n+LR", "baseline"),
        ("TF-IDF\n+SVM", "tfidf_svm"),
        ("FinBERT\n0-shot", "zeroshot_finbert_native"),
        ("FinBERT\nfew-shot", "fewshot_finbert"),
        ("FinBERT\nfine-tune", "finetune_finbert"),
        ("BERT\nLLRD", "finetune_bert_llrd"),
        ("Multi-task\nFinBERT", "multitask"),
    ]

    fig, ax = plt.subplots(figsize=(14, 6))

    for task, color, marker, label_name in [
        ("sentiment", BLUE, "o", "Sentiment (Macro-F1)"),
        ("stance", RED, "s", "Stance (Macro-F1)"),
    ]:
        x_labels = []
        y_values = []
        for stage_name, key_prefix in stages:
            key = f"{key_prefix}_{task}"
            if key in results:
                x_labels.append(stage_name)
                y_values.append(results[key]["macro_f1"])

        x = range(len(x_labels))
        ax.plot(x, y_values, marker=marker, markersize=10, linewidth=2.5,
                color=color, label=label_name, alpha=0.9)

        # Annotate each point
        for xi, yi in zip(x, y_values):
            ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                       xytext=(0, 12), ha="center", fontsize=8, color=color)

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel("Macro-F1 Score", fontsize=12)
    ax.set_title("Performance Progression: Baselines → Pre-trained → Fine-tuned → Multi-task",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0.2, 1.05)

    # Add stage separators
    for xpos, stage_label in [(2.5, "Baselines"), (4.5, "Pre-trained"), (6.5, "Fine-tuned")]:
        ax.axvline(x=xpos, color="gray", linestyle="--", alpha=0.4)

    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(ANALYSIS_DIR, "performance_progression.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 8. Domain pre-training gap analysis
# ──────────────────────────────────────────────────────────────────────────────

def plot_domain_pretraining_gap():
    """Bar chart showing the gap between FinBERT and general models."""
    results = load_all_results()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, task in zip(axes, ["sentiment", "stance"]):
        models = {
            "FinBERT": results.get(f"fewshot_finbert_{task}", {}).get("macro_f1", 0),
            "BERT-base": results.get(f"fewshot_bert-base-uncased_{task}", {}).get("macro_f1", 0),
            "RoBERTa": results.get(f"fewshot_roberta-base_{task}", {}).get("macro_f1", 0),
        }

        names = list(models.keys())
        scores = list(models.values())
        colors = [GREEN, BLUE, GOLD]

        bars = ax.bar(names, scores, color=colors, edgecolor="white", linewidth=1.5, width=0.5)

        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{score:.4f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

        # Draw gap arrow
        gap = scores[0] - max(scores[1], scores[2])
        if gap > 0:
            ax.annotate(f"Gap: +{gap:.3f}",
                       xy=(0, scores[0]), xytext=(1.5, scores[0] - 0.05),
                       fontsize=11, color=RED, fontweight="bold",
                       arrowprops=dict(arrowstyle="->", color=RED, lw=2))

        ax.set_title(f"Few-Shot (k=16) — {'Sentiment' if task == 'sentiment' else 'Stance'}",
                     fontsize=13, fontweight="bold")
        ax.set_ylabel("Macro-F1")
        ax.set_ylim(0, max(scores) * 1.2)

    plt.suptitle("Domain-Specific Pre-training Advantage", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(ANALYSIS_DIR, "domain_pretraining_gap.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 9. Multi-task vs Single-task improvement
# ──────────────────────────────────────────────────────────────────────────────

def plot_multitask_improvement():
    """Show improvement from single-task to multi-task."""
    results = load_all_results()

    fig, ax = plt.subplots(figsize=(10, 6))

    comparisons = []
    for task in ["sentiment", "stance"]:
        for model_name, key in [("FinBERT fine-tune", f"finetune_finbert_{task}"),
                                 ("BERT LLRD", f"finetune_bert_llrd_{task}")]:
            single = results.get(key, {}).get("macro_f1", 0)
            multi = results.get(f"multitask_{task}", {}).get("macro_f1", 0)
            improvement = multi - single
            comparisons.append({
                "model": f"{model_name}\n({task})",
                "single_task": single,
                "multi_task": multi,
                "improvement": improvement,
                "task": task,
            })

    df = pd.DataFrame(comparisons)
    x = np.arange(len(df))
    width = 0.3

    bars1 = ax.bar(x - width/2, df["single_task"], width, label="Single-task",
                   color=BLUE, alpha=0.8, edgecolor="white")
    bars2 = ax.bar(x + width/2, df["multi_task"], width, label="Multi-task FinBERT",
                   color=GREEN, alpha=0.8, edgecolor="white")

    for i, (_, row) in enumerate(df.iterrows()):
        sign = "+" if row["improvement"] >= 0 else ""
        color = GREEN if row["improvement"] >= 0 else RED
        ax.annotate(f"{sign}{row['improvement']:.4f}",
                   xy=(x[i] + width/2, row["multi_task"]),
                   xytext=(0, 8), textcoords="offset points",
                   ha="center", fontsize=10, fontweight="bold", color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], fontsize=10)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Multi-task vs Single-task: Macro-F1 Improvement", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(ANALYSIS_DIR, "multitask_improvement.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 10. Task difficulty radar chart
# ──────────────────────────────────────────────────────────────────────────────

def plot_task_difficulty():
    """Radar chart comparing sentiment vs stance difficulty across models."""
    results = load_all_results()

    models = [
        ("TF-IDF+SVM", "tfidf_svm"),
        ("FinBERT 0-shot", "zeroshot_finbert_native"),
        ("FinBERT fine-tune", "finetune_finbert"),
        ("BERT LLRD", "finetune_bert_llrd"),
        ("Multi-task", "multitask"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    model_names = [m[0] for m in models]
    sentiment_f1 = [results.get(f"{m[1]}_sentiment", {}).get("macro_f1", 0) for m in models]
    stance_f1 = [results.get(f"{m[1]}_stance", {}).get("macro_f1", 0) for m in models]
    gap = [s - t for s, t in zip(sentiment_f1, stance_f1)]

    x = np.arange(len(model_names))
    width = 0.25

    ax.bar(x - width, sentiment_f1, width, label="Sentiment F1", color=BLUE, alpha=0.85)
    ax.bar(x, stance_f1, width, label="Stance F1", color=RED, alpha=0.85)
    ax.bar(x + width, gap, width, label="Gap (Sent - Stance)", color=GOLD, alpha=0.85)

    for i in range(len(model_names)):
        ax.text(x[i] + width, gap[i] + 0.01, f"{gap[i]:.3f}",
                ha="center", va="bottom", fontsize=9, color=GOLD, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=10)
    ax.set_ylabel("Score")
    ax.set_title("Task Difficulty: Sentiment vs Stance Gap Across Models",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(ANALYSIS_DIR, "task_difficulty_gap.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 11. Lexicon coverage analysis
# ──────────────────────────────────────────────────────────────────────────────

def plot_lexicon_coverage(stats):
    """Analyze how many words in each class match the LM lexicon."""
    from src.lexicon import (LM_POSITIVE, LM_NEGATIVE, LM_UNCERTAINTY,
                              HAWKISH_WORDS, DOVISH_WORDS, _tokenize)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (name, s) in zip(axes, stats.items()):
        labels = s["labels"]
        lexicon_data = {l: {"positive": 0, "negative": 0, "uncertainty": 0,
                            "hawkish": 0, "dovish": 0, "total_words": 0}
                        for l in labels}

        for text, label_idx in zip(s["texts"], s["all_labels"]):
            label = labels[label_idx]
            tokens = _tokenize(text)
            lexicon_data[label]["total_words"] += len(tokens)
            lexicon_data[label]["positive"] += sum(1 for t in tokens if t in LM_POSITIVE)
            lexicon_data[label]["negative"] += sum(1 for t in tokens if t in LM_NEGATIVE)
            lexicon_data[label]["uncertainty"] += sum(1 for t in tokens if t in LM_UNCERTAINTY)
            lexicon_data[label]["hawkish"] += sum(1 for t in tokens if t in HAWKISH_WORDS)
            lexicon_data[label]["dovish"] += sum(1 for t in tokens if t in DOVISH_WORDS)

        # Normalize by total words
        categories = ["positive", "negative", "uncertainty", "hawkish", "dovish"]
        x = np.arange(len(categories))
        width = 0.25

        colors = [BLUE, GOLD, GREEN]
        for i, (label, color) in enumerate(zip(labels, colors)):
            total = max(lexicon_data[label]["total_words"], 1)
            values = [lexicon_data[label][c] / total * 100 for c in categories]
            ax.bar(x + i * width, values, width, label=label, color=color, alpha=0.8)

        ax.set_xticks(x + width)
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylabel("% of Words Matching Lexicon")
        ax.set_title(f"{name} — Lexicon Word Coverage", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(ANALYSIS_DIR, "lexicon_coverage.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  COMPREHENSIVE DATA ANALYSIS")
    print("=" * 70)

    # Load data
    fpb, fomc = load_datasets()
    stats = dataset_statistics(fpb, fomc)

    # Generate all plots
    print("\n  Generating visualizations...")
    plot_class_distributions(stats)
    plot_text_lengths(stats)
    plot_top_words(stats)
    plot_model_comparison()
    plot_per_class_f1_heatmap()
    plot_progression()
    plot_domain_pretraining_gap()
    plot_multitask_improvement()
    plot_task_difficulty()
    plot_lexicon_coverage(stats)

    print(f"\n  All plots saved to {ANALYSIS_DIR}/")
    print(f"  Total files: {len(os.listdir(ANALYSIS_DIR))}")
    print("=" * 70)


if __name__ == "__main__":
    main()
