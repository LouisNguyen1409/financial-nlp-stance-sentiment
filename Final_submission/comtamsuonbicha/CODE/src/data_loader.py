"""
Dataset loading and preprocessing for:
  1. Financial PhraseBank  – sentiment classification (positive / neutral / negative)
  2. FOMC Hawkish-Dovish   – stance classification (hawkish / dovish / neutral)

Both datasets are downloaded from HuggingFace Hub and split into
train / validation / test partitions with stratification.
"""

import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset, DatasetDict

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    SEED, TEST_SIZE, VAL_SIZE,
    FPB_DATASET_NAME, FPB_SUBSET,
    FOMC_DATASET_NAME,
    SENTIMENT_LABELS,
    STANCE_LABELS,
    DATA_DIR, FEW_SHOT_K,
)


# ──────────────────────────────────────────────────────────────────────────────
# Financial PhraseBank
# ──────────────────────────────────────────────────────────────────────────────

def load_financial_phrasebank():
    """
    Load Financial PhraseBank (sentences_allagree) from HuggingFace.

    The dataset has columns: 'sentence', 'label'
    Label mapping: 0 = negative, 1 = neutral, 2 = positive

    Returns a DatasetDict with 'train', 'val', 'test' splits.
    Each example has keys: 'text', 'label' (int), 'label_name' (str).
    """
    print("[DATA] Loading Financial PhraseBank …")
    ds = load_dataset(FPB_DATASET_NAME, FPB_SUBSET)

    # Combine all available splits into one DataFrame, then re-split with
    # a validation set (the original only has train/test)
    frames = []
    for split_name in ds:
        frames.append(ds[split_name].to_pandas())
    df = pd.concat(frames, ignore_index=True)

    # Standardise column names
    df = df.rename(columns={"sentence": "text"})
    df = df[["text", "label"]].copy()
    df["label"] = df["label"].astype(int)
    df["label_name"] = df["label"].map(
        {i: SENTIMENT_LABELS[i] for i in range(len(SENTIMENT_LABELS))}
    )

    # Stratified split: train (70%) / val (10%) / test (20%)
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["label"], random_state=SEED
    )
    train_df, val_df = train_test_split(
        train_df,
        test_size=VAL_SIZE / (1 - TEST_SIZE),
        stratify=train_df["label"],
        random_state=SEED,
    )

    splits = DatasetDict({
        "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
        "val":   Dataset.from_pandas(val_df.reset_index(drop=True)),
        "test":  Dataset.from_pandas(test_df.reset_index(drop=True)),
    })
    _print_split_stats("Financial PhraseBank", splits, SENTIMENT_LABELS)
    return splits


# ──────────────────────────────────────────────────────────────────────────────
# FOMC Hawkish-Dovish
# ──────────────────────────────────────────────────────────────────────────────

def load_fomc_dataset():
    """
    Load FOMC Hawkish-Dovish dataset from HuggingFace (gtfintechlab).

    The dataset has columns: 'sentence', 'label', 'year', etc.
    Label mapping: 0 = dovish, 1 = hawkish, 2 = neutral

    Returns a DatasetDict with 'train', 'val', 'test' splits.
    Each example has keys: 'text', 'label' (int), 'label_name' (str).
    """
    print("[DATA] Loading FOMC Hawkish-Dovish dataset …")

    try:
        ds = load_dataset(FOMC_DATASET_NAME)
    except Exception as e:
        print(f"[DATA] Could not load from HuggingFace ({e}).")
        print("[DATA] Attempting to load from local data/ directory …")
        return _load_fomc_local()

    # Combine all available splits into one DataFrame for re-splitting
    frames = []
    for split_name in ds:
        frames.append(ds[split_name].to_pandas())
    df = pd.concat(frames, ignore_index=True)

    return _process_fomc_df(df)


def _load_fomc_local():
    """Fallback: load FOMC data from CSV files in data/ directory."""
    csv_path = os.path.join(DATA_DIR, "fomc_hawkish_dovish.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"FOMC dataset not found at {csv_path}. "
            "Please download from https://github.com/gtfintechlab/fomc-hawkish-dovish "
            "and place the CSV in the data/ folder."
        )
    df = pd.read_csv(csv_path)
    return _process_fomc_df(df)


def _process_fomc_df(df):
    """Standardise FOMC DataFrame columns and create train/val/test splits."""
    # Rename 'sentence' → 'text' if present
    if "sentence" in df.columns:
        df = df.rename(columns={"sentence": "text"})

    # Keep only the columns we need
    df = df[["text", "label"]].copy()
    df = df.dropna(subset=["text", "label"])

    # Convert string labels to integers if needed
    if df["label"].dtype == object:
        label_map = {v: k for k, v in enumerate(STANCE_LABELS)}
        df["label"] = df["label"].str.strip().str.lower().map(label_map)
        df = df.dropna(subset=["label"])

    df["label"] = df["label"].astype(int)
    df["label_name"] = df["label"].map(
        {i: STANCE_LABELS[i] for i in range(len(STANCE_LABELS))}
    )
    df = df.reset_index(drop=True)

    # Stratified split: train (70%) / val (10%) / test (20%)
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["label"], random_state=SEED
    )
    train_df, val_df = train_test_split(
        train_df,
        test_size=VAL_SIZE / (1 - TEST_SIZE),
        stratify=train_df["label"],
        random_state=SEED,
    )

    splits = DatasetDict({
        "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
        "val":   Dataset.from_pandas(val_df.reset_index(drop=True)),
        "test":  Dataset.from_pandas(test_df.reset_index(drop=True)),
    })
    _print_split_stats("FOMC Hawkish-Dovish", splits, STANCE_LABELS)
    return splits


# ──────────────────────────────────────────────────────────────────────────────
# Few-shot sampling
# ──────────────────────────────────────────────────────────────────────────────

def get_few_shot_subset(dataset_split, k=FEW_SHOT_K):
    """
    Sample k examples per class from a dataset split for few-shot learning.
    Returns a HuggingFace Dataset.
    """
    df = dataset_split.to_pandas()
    pieces = []
    for label_val in sorted(df["label"].unique()):
        subset = df[df["label"] == label_val]
        pieces.append(subset.sample(n=min(k, len(subset)), random_state=SEED))
    sampled = pd.concat(pieces, ignore_index=True)
    return Dataset.from_pandas(sampled)


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def compute_class_weights(dataset_split, num_classes=3):
    """
    Compute inverse-frequency class weights for weighted cross-entropy loss.
    Returns a list of floats, one per class.
    """
    labels = dataset_split["label"]
    counts = Counter(labels)
    total = sum(counts.values())
    weights = [total / (num_classes * counts.get(i, 1)) for i in range(num_classes)]
    return weights


def _print_split_stats(name, splits, label_names):
    """Print dataset split sizes and class distributions."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    for split_name, split_data in splits.items():
        labels = split_data["label"]
        dist = Counter(labels)
        dist_str = ", ".join(
            f"{label_names[k]}: {dist.get(k, 0)}" for k in sorted(dist.keys())
        )
        print(f"  {split_name:>5}: {len(split_data):>5} samples  ({dist_str})")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fpb = load_financial_phrasebank()
    fomc = load_fomc_dataset()
