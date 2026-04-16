"""
Single-task fine-tuning of FinBERT.

Fine-tunes FinBERT (ProsusAI/finbert) separately on each dataset:
  1. FOMC -> stance classification  (hawkish / dovish / neutral)
  2. FPB -> sentiment classification (positive / neutral / negative)

Each model gets a fresh task-specific classification head on top of
the pre-trained FinBERT encoder. Training uses AdamW with linear
warm-up and optional weighted cross-entropy for class imbalance.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DEVICE, SEED, BATCH_SIZE, MAX_SEQ_LENGTH, LEARNING_RATE,
    WEIGHT_DECAY, FINETUNE_EPOCHS, WARMUP_RATIO,
    FINBERT_MODEL, MODELS_DIR,
    SENTIMENT_LABELS, STANCE_LABELS,
)
from src.evaluate import (
    compute_metrics,
    print_classification_report,
    plot_confusion_matrix,
    save_results,
)
from src.data_loader import compute_class_weights

torch.manual_seed(SEED)


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset wrapper
# ──────────────────────────────────────────────────────────────────────────────

class TextClassificationDataset(TorchDataset):
    """Tokenises texts on-the-fly and returns tensors for the model."""

    def __init__(self, texts, labels, tokenizer, max_length=MAX_SEQ_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Fine-tuning
# ──────────────────────────────────────────────────────────────────────────────

def finetune_finbert(train_split, val_split, test_split, label_names,
                     task_name, use_weighted_loss=True):
    """
    Fine-tune FinBERT on a single task.

    Args:
        train_split, val_split, test_split: HuggingFace Dataset objects
        label_names: list of label strings
        task_name: 'stance' or 'sentiment'
        use_weighted_loss: whether to use inverse-frequency class weights

    Returns:
        dict of test-set metrics, and the fine-tuned model + tokenizer
    """
    print(f"\n{'='*60}")
    print(f"  FINE-TUNING FinBERT — {task_name}")
    print(f"{'='*60}")

    device = DEVICE
    num_labels = len(label_names)

    # Load tokenizer and model with a fresh classification head
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        FINBERT_MODEL,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,  # replace the existing head
    )
    model.to(device)

    # Build DataLoaders
    train_ds = TextClassificationDataset(
        train_split["text"], train_split["label"], tokenizer
    )
    val_ds = TextClassificationDataset(
        val_split["text"], val_split["label"], tokenizer
    )
    test_ds = TextClassificationDataset(
        test_split["text"], test_split["label"], tokenizer
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # Loss function (optionally weighted)
    if use_weighted_loss:
        weights = compute_class_weights(train_split, num_classes=num_labels)
        print(f"  Class weights: {dict(zip(label_names, [f'{w:.3f}' for w in weights]))}")
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float).to(device)
        )
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimiser and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    total_steps = len(train_loader) * FINETUNE_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )

    # Training loop
    best_val_f1 = 0.0
    best_model_state = None

    for epoch in range(FINETUNE_EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{FINETUNE_EPOCHS}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        val_metrics = _evaluate_model(model, val_loader, label_names, device)
        print(
            f"  Epoch {epoch+1}: loss={avg_loss:.4f}  "
            f"val_acc={val_metrics['accuracy']:.4f}  "
            f"val_f1={val_metrics['macro_f1']:.4f}"
        )

        # Save best model
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best model and evaluate on test set
    model.load_state_dict(best_model_state)
    model.to(device)
    test_metrics = _evaluate_model(model, test_loader, label_names, device)

    print_classification_report(test_metrics, "FinBERT (fine-tuned)", task_name)

    # Get predictions for confusion matrix
    y_true, y_pred = _get_predictions(model, test_loader, device)
    plot_confusion_matrix(y_true, y_pred, label_names, "FinBERT_finetuned", task_name)
    save_results(
        {"model": "FinBERT (fine-tuned)", "task": task_name, **test_metrics},
        f"finetune_finbert_{task_name}.json",
    )

    # Save model
    save_path = os.path.join(MODELS_DIR, f"finbert_{task_name}")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"  Model saved to {save_path}")

    return test_metrics, model, tokenizer


def _evaluate_model(model, dataloader, label_names, device):
    """Run model on a DataLoader and return metrics."""
    y_true, y_pred = _get_predictions(model, dataloader, device)
    return compute_metrics(y_true, y_pred, label_names)


def _get_predictions(model, dataloader, device):
    """Get true labels and predictions from a DataLoader."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1).cpu().tolist()

            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    return all_labels, all_preds