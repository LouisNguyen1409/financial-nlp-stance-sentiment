"""
Zero-shot and few-shot evaluation of pre-trained models.

Models evaluated:
  - FinBERT  (ProsusAI/finbert)      — domain-specific for financial text
  - BERT-base-uncased                 — general-purpose
  - RoBERTa-base                      — general-purpose

Zero-shot approach:
  - FinBERT has a native financial-sentiment head → used directly on both datasets.
    For sentiment it outputs positive/negative/neutral (direct match).
    For stance it provides a domain-informed prior (we map sentiment → stance proxy).

Few-shot approach:
  - Train a small classification head on top of frozen model embeddings
    using k examples per class (default k=16).
  - This fairly compares the quality of learned representations across models.
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModel,
)
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DEVICE, SEED, BATCH_SIZE, MAX_SEQ_LENGTH,
    FINBERT_MODEL, BERT_BASE_MODEL, ROBERTA_BASE_MODEL,
    SENTIMENT_LABELS, STANCE_LABELS, FEW_SHOT_K,
)
from src.evaluate import (
    compute_metrics,
    print_classification_report,
    plot_confusion_matrix,
    save_results,
)
from src.data_loader import get_few_shot_subset

torch.manual_seed(SEED)


# ──────────────────────────────────────────────────────────────────────────────
# Zero-shot evaluation: FinBERT native head
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_finbert_native(test_split, task_name="sentiment"):
    """
    Evaluate FinBERT using its native financial-sentiment classification head.
    FinBERT outputs: positive / negative / neutral — matches our sentiment labels.
    For stance, we evaluate to show that a sentiment head alone is insufficient.
    """
    print(f"\n[ZERO-SHOT] FinBERT (native head) — {task_name}")
    label_names = SENTIMENT_LABELS if task_name == "sentiment" else STANCE_LABELS

    # FinBERT's pipeline works on CPU to avoid MPS issues with pipeline
    clf = pipeline(
        "text-classification",
        model=FINBERT_MODEL,
        tokenizer=FINBERT_MODEL,
        device=-1,  # CPU for pipeline compatibility
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        top_k=None,  # get all scores
    )

    texts = test_split["text"]
    y_true = test_split["label"]
    y_pred = []

    # FinBERT outputs: positive, negative, neutral
    # For sentiment task: direct mapping (neg=0, neu=1, pos=2)
    # For stance task: map as proxy (negative→dovish, neutral→neutral, positive→hawkish)
    if task_name == "sentiment":
        finbert_to_idx = {"negative": 0, "neutral": 1, "positive": 2}
    else:
        # Proxy mapping: sentiment → stance
        finbert_to_idx = {"negative": 0, "positive": 1, "neutral": 2}  # neg→dovish, pos→hawkish, neu→neutral

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="FinBERT native"):
        batch = texts[i : i + BATCH_SIZE]
        results = clf(batch)
        for r in results:
            # r is a list of dicts with 'label' and 'score'
            best = max(r, key=lambda x: x["score"])
            pred_label = best["label"].lower()
            y_pred.append(finbert_to_idx.get(pred_label, 1))

    metrics = compute_metrics(y_true, y_pred, label_names)
    print_classification_report(metrics, "FinBERT (native)", task_name)
    plot_confusion_matrix(y_true, y_pred, label_names, "FinBERT_native", task_name)
    save_results(
        {"model": "FinBERT (native)", "task": task_name, **metrics},
        f"zeroshot_finbert_native_{task_name}.json",
    )
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Few-shot evaluation
# ──────────────────────────────────────────────────────────────────────────────

class FewShotClassifier(nn.Module):
    """A simple linear classifier on top of frozen transformer embeddings."""

    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, x):
        return self.classifier(x)


def _encode_texts(tokenizer, model, texts, device):
    """Encode a list of texts into [CLS] embeddings using a frozen model."""
    model.eval()
    model.to(device)
    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        inputs = tokenizer(
            batch, padding=True, truncation=True,
            max_length=MAX_SEQ_LENGTH, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # Use [CLS] token embedding (first token)
            cls_emb = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_emb.cpu())

    return torch.cat(all_embeddings, dim=0)


def evaluate_few_shot(model_name, train_split, test_split, label_names,
                      task_name, k=FEW_SHOT_K):
    """
    Few-shot evaluation: train a linear classifier on frozen embeddings
    from k examples per class, evaluate on the full test set.

    This comparison is the fairest way to assess representation quality
    across FinBERT vs BERT-base vs RoBERTa-base.
    """
    print(f"\n[FEW-SHOT k={k}] {model_name} — {task_name}")
    device = DEVICE

    # Load tokenizer and base model (no classification head)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)
    base_model.eval()

    # Sample k examples per class
    few_shot_data = get_few_shot_subset(train_split, k=k)
    print(f"  Training on {len(few_shot_data)} examples ({k} per class)")

    # Encode few-shot training examples
    train_embs = _encode_texts(
        tokenizer, base_model, few_shot_data["text"], device
    )
    train_labels = torch.tensor(few_shot_data["label"])

    # Encode test examples
    test_embs = _encode_texts(
        tokenizer, base_model, test_split["text"], device
    )
    test_labels_true = test_split["label"]

    # Free GPU memory
    base_model.cpu()
    del base_model
    if str(device) == "mps":
        torch.mps.empty_cache()

    # Train a simple linear classifier on the few-shot embeddings
    hidden_size = train_embs.shape[1]
    classifier = FewShotClassifier(hidden_size, len(label_names)).to("cpu")
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Train for 200 epochs (dataset is tiny — ~48 examples — so this is fast)
    classifier.train()
    for epoch in range(200):
        logits = classifier(train_embs)
        loss = criterion(logits, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Predict on test set
    classifier.eval()
    with torch.no_grad():
        test_logits = classifier(test_embs)
        y_pred = test_logits.argmax(dim=1).numpy().tolist()

    metrics = compute_metrics(test_labels_true, y_pred, label_names)
    short_name = model_name.split("/")[-1]
    print_classification_report(metrics, f"{short_name} (few-shot k={k})", task_name)
    plot_confusion_matrix(
        test_labels_true, y_pred, label_names,
        f"{short_name}_fewshot_{k}", task_name,
    )
    save_results(
        {"model": f"{short_name} (few-shot k={k})", "task": task_name, **metrics},
        f"fewshot_{short_name}_{task_name}.json",
    )
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Run all pre-trained evaluations
# ──────────────────────────────────────────────────────────────────────────────

def run_all_pretrained_evaluations(fomc_splits, fpb_splits):
    """
    Run zero-shot and few-shot evaluations for all three models
    on both datasets. This demonstrates the impact of domain-specific
    pre-training (FinBERT) versus general-purpose models (BERT, RoBERTa).
    """
    results = {}

    # --- Zero-shot: FinBERT native sentiment head ---
    # On Financial PhraseBank (direct match for sentiment)
    results["finbert_native_sentiment"] = evaluate_finbert_native(
        fpb_splits["test"], task_name="sentiment"
    )
    # On FOMC (proxy: sentiment → stance mapping)
    results["finbert_native_stance"] = evaluate_finbert_native(
        fomc_splits["test"], task_name="stance"
    )

    # --- Few-shot evaluations for all 3 models ---
    for model_name in [FINBERT_MODEL, BERT_BASE_MODEL, ROBERTA_BASE_MODEL]:
        short = model_name.split("/")[-1]
        for task_name, splits, labels in [
            ("stance", fomc_splits, STANCE_LABELS),
            ("sentiment", fpb_splits, SENTIMENT_LABELS),
        ]:
            key = f"{short}_fewshot_{task_name}"
            results[key] = evaluate_few_shot(
                model_name, splits["train"], splits["test"],
                labels, task_name,
            )

    return results
