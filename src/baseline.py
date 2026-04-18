"""
Baseline models for Financial NLP tasks.

Implements three non-neural baselines:
  1. TF-IDF (bigrams) + Logistic Regression  (original baseline)
  2. TF-IDF (bigrams) + Linear SVM           (best alternative for both tasks)
  3. TF-IDF (1–3 grams, 80k vocab, min_df=2) + Logistic Regression

Baselines #2 and #3 are produced by run_alternative_baselines(...) during
step 2 of run_experiments.py. The lexicon-based baselines (rule-based and
TF-IDF + LM lexicon features) live in src/lexicon.py.

All models are trained and evaluated on both tasks:
  - Stance classification  (hawkish / dovish / neutral)
  - Sentiment classification (positive / neutral / negative)
"""

import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SEED
from src.evaluate import (
    compute_metrics,
    print_classification_report,
    plot_confusion_matrix,
    save_results,
)


# ──────────────────────────────────────────────────────────────────────────────
# Original baseline: TF-IDF + Logistic Regression
# ──────────────────────────────────────────────────────────────────────────────


def build_baseline_pipeline():
    """TF-IDF (bigrams) + Logistic Regression — original baseline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=SEED,
            solver="lbfgs",
        )),
    ])


def train_and_evaluate_baseline(train_split, test_split, label_names, task_name):
    """
    Train the TF-IDF + LR baseline and evaluate on the test set.

    Args:
        train_split: HuggingFace Dataset with 'text' and 'label' columns
        test_split:  HuggingFace Dataset with 'text' and 'label' columns
        label_names: list of human-readable label names
        task_name:   'stance' or 'sentiment' (for logging / file names)

    Returns:
        (metrics_dict, fitted_pipeline)
    """
    print(f"\n{'='*60}")
    print(f"  BASELINE (TF-IDF + Logistic Regression) — {task_name}")
    print(f"{'='*60}")

    train_texts = train_split["text"]
    train_labels = train_split["label"]
    test_texts = test_split["text"]
    test_labels = test_split["label"]

    pipeline = build_baseline_pipeline()
    pipeline.fit(train_texts, train_labels)
    predictions = pipeline.predict(test_texts)

    metrics = compute_metrics(test_labels, predictions, label_names)
    print_classification_report(metrics, "TF-IDF + LR", task_name)
    plot_confusion_matrix(
        test_labels, predictions, label_names, "TF-IDF_LR", task_name
    )
    save_results(
        {"model": "TF-IDF + Logistic Regression", "task": task_name, **metrics},
        f"baseline_{task_name}.json",
    )
    return metrics, pipeline


# ──────────────────────────────────────────────────────────────────────────────
# Best alternative: TF-IDF + SVM (LinearSVC)
# ──────────────────────────────────────────────────────────────────────────────


def build_tfidf_svm_pipeline():
    """TF-IDF (bigrams) + Linear SVM — best alternative for stance."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
        )),
        ("clf", LinearSVC(
            max_iter=2000,
            class_weight="balanced",
            random_state=SEED,
            C=1.0,
        )),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# Best alternative for sentiment: TF-IDF (trigrams) + Logistic Regression
# ──────────────────────────────────────────────────────────────────────────────


def build_tfidf_trigram_lr_pipeline():
    """TF-IDF with uni/bi/trigrams + Logistic Regression — best alternative for sentiment."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=80_000,
            ngram_range=(1, 3),
            sublinear_tf=True,
            strip_accents="unicode",
            min_df=2,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=SEED,
            solver="lbfgs",
            C=1.0,
        )),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# Run alternative baseline experiments
# ──────────────────────────────────────────────────────────────────────────────


def _run_and_eval(pipeline, splits, label_names, task_name, model_name, file_prefix):
    """Train a single pipeline on one task, evaluate, and return metrics."""
    print(f"\n{'='*60}")
    print(f"  {model_name} — {task_name}")
    print(f"{'='*60}")

    train_texts = splits["train"]["text"]
    train_labels = splits["train"]["label"]
    test_texts = splits["test"]["text"]
    test_labels = splits["test"]["label"]

    pipeline.fit(train_texts, train_labels)
    predictions = pipeline.predict(test_texts)

    metrics = compute_metrics(test_labels, predictions, label_names)
    print_classification_report(metrics, model_name, task_name)
    plot_confusion_matrix(
        test_labels, predictions, label_names,
        model_name.replace(" ", "_").replace("+", ""), task_name,
    )
    save_results(
        {"model": model_name, "task": task_name, **metrics},
        f"{file_prefix}_{task_name}.json",
    )
    return metrics


def run_alternative_baselines(fomc_splits, fpb_splits):
    """
    Run the two best alternative baselines on both datasets:
      - TF-IDF + SVM        (best for stance)
      - TF-IDF (trigrams) + LR  (best for sentiment)

    Each model is evaluated on both tasks for comparison.
    Returns a dict of {experiment_key: metrics_dict}.
    """
    from config import STANCE_LABELS, SENTIMENT_LABELS

    results = {}

    experiments = [
        ("TF-IDF + SVM", "tfidf_svm", build_tfidf_svm_pipeline),
        ("TF-IDF (trigrams) + LR", "tfidf_trigram_lr", build_tfidf_trigram_lr_pipeline),
    ]

    for model_name, file_prefix, build_fn in experiments:
        for splits, label_names, task_name in [
            (fomc_splits, STANCE_LABELS, "stance"),
            (fpb_splits, SENTIMENT_LABELS, "sentiment"),
        ]:
            key = f"{file_prefix}_{task_name}"
            results[key] = _run_and_eval(
                build_fn(), splits, label_names, task_name, model_name, file_prefix,
            )

    return results
