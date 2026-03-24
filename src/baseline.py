"""
Baseline model: TF-IDF + Logistic Regression.

This is a simple, non-neural baseline applied to both tasks:
  1. Stance classification  (hawkish / dovish / neutral)
  2. Sentiment classification (positive / neutral / negative)

The pipeline uses TF-IDF to convert text into numeric feature vectors,
then trains a Logistic Regression classifier.
"""

import os
import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SEED
from src.evaluate import (
    compute_metrics,
    print_classification_report,
    plot_confusion_matrix,
    save_results,
)


def build_baseline_pipeline():
    """
    Build a TF-IDF + Logistic Regression pipeline.

    TF-IDF settings:
      - Up to bigrams (ngram_range=(1,2)) to capture short phrases
      - Max 50 000 features to keep memory manageable
      - Sub-linear TF scaling (log normalisation)

    Logistic Regression:
      - max_iter=1000 to ensure convergence
      - class_weight='balanced' to handle class imbalance
    """
    pipeline = Pipeline([
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
    return pipeline


def train_and_evaluate_baseline(train_split, test_split, label_names, task_name):
    """
    Train the TF-IDF + LR baseline and evaluate on the test set.

    Args:
        train_split: HuggingFace Dataset with 'text' and 'label' columns
        test_split:  HuggingFace Dataset with 'text' and 'label' columns
        label_names: list of human-readable label names
        task_name:   'stance' or 'sentiment' (for logging / file names)

    Returns:
        dict of evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"  BASELINE (TF-IDF + Logistic Regression) — {task_name}")
    print(f"{'='*60}")

    # Extract text and labels
    train_texts = train_split["text"]
    train_labels = train_split["label"]
    test_texts = test_split["text"]
    test_labels = test_split["label"]

    # Build, train, predict
    pipeline = build_baseline_pipeline()
    pipeline.fit(train_texts, train_labels)
    predictions = pipeline.predict(test_texts)

    # Evaluate
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
