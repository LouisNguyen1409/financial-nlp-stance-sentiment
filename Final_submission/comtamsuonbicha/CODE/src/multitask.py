"""
Multi-task learning extension.

Trains a unified model on BOTH datasets simultaneously:
  - Shared FinBERT encoder layers
  - Task-specific classification heads:
      • Stance head  → 3 classes (hawkish / dovish / neutral)
      • Sentiment head → 3 classes (positive / neutral / negative)

Extensions beyond standard fine-tuning:
  1. Joint multi-task training with alternating batches
  2. Weighted cross-entropy loss to address class imbalance in FOMC
  3. Analysis of whether joint training improves generalisation
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DEVICE, SEED, BATCH_SIZE, MAX_SEQ_LENGTH, LEARNING_RATE,
    WEIGHT_DECAY, MULTITASK_EPOCHS, WARMUP_RATIO,
    FINBERT_MODEL, MODELS_DIR,
    SENTIMENT_LABELS, STANCE_LABELS,
)
from src.evaluate import (
    compute_metrics,
    print_classification_report,
    plot_confusion_matrix,
    save_results,
    error_analysis,
)
from src.data_loader import compute_class_weights
from src.finetune_fineBert import TextClassificationDataset

torch.manual_seed(SEED)


# ──────────────────────────────────────────────────────────────────────────────
# Multi-task model
# ──────────────────────────────────────────────────────────────────────────────

class MultiTaskFinBERT(nn.Module):
    """
    FinBERT with two task-specific classification heads.

    Architecture:
      Input → FinBERT encoder (shared) → [CLS] pooling → Dropout
        ├─→ Stance head   (Linear → 3 classes)
        └─→ Sentiment head (Linear → 3 classes)

    During forward pass, the 'task' argument selects which head to use.
    """

    def __init__(self, num_stance_labels=3, num_sentiment_labels=3, dropout=0.1):
        super().__init__()

        # Shared encoder (FinBERT)
        self.encoder = AutoModel.from_pretrained(FINBERT_MODEL)
        hidden_size = self.encoder.config.hidden_size  # 768 for BERT-base

        # Dropout applied after pooling
        self.dropout = nn.Dropout(dropout)

        # Task-specific classification heads
        self.stance_head = nn.Linear(hidden_size, num_stance_labels)
        self.sentiment_head = nn.Linear(hidden_size, num_sentiment_labels)

    def forward(self, input_ids, attention_mask, task="stance"):
        """
        Forward pass through shared encoder and task-specific head.

        Args:
            input_ids:      (batch_size, seq_len)
            attention_mask:  (batch_size, seq_len)
            task:           'stance' or 'sentiment'

        Returns:
            logits: (batch_size, num_labels)
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)

        if task == "stance":
            return self.stance_head(pooled)
        else:
            return self.sentiment_head(pooled)


# ──────────────────────────────────────────────────────────────────────────────
# Multi-task training
# ──────────────────────────────────────────────────────────────────────────────

def train_multitask(fomc_splits, fpb_splits):
    """
    Train the multi-task model on both datasets simultaneously.

    Training procedure:
      1. Create DataLoaders for both tasks
      2. Alternate batches: one stance batch, one sentiment batch
      3. Use weighted cross-entropy for the stance (FOMC) task
      4. Evaluate on both validation sets after each epoch
      5. Save the model with the best average macro-F1

    Returns:
        dict of test metrics for both tasks, and the model + tokenizer
    """
    print(f"\n{'='*60}")
    print(f"  MULTI-TASK TRAINING (Shared FinBERT + Dual Heads)")
    print(f"{'='*60}")

    device = DEVICE
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model = MultiTaskFinBERT().to(device)

    # Build DataLoaders for both tasks
    stance_train = TextClassificationDataset(
        fomc_splits["train"]["text"], fomc_splits["train"]["label"], tokenizer
    )
    stance_val = TextClassificationDataset(
        fomc_splits["val"]["text"], fomc_splits["val"]["label"], tokenizer
    )
    stance_test = TextClassificationDataset(
        fomc_splits["test"]["text"], fomc_splits["test"]["label"], tokenizer
    )
    sentiment_train = TextClassificationDataset(
        fpb_splits["train"]["text"], fpb_splits["train"]["label"], tokenizer
    )
    sentiment_val = TextClassificationDataset(
        fpb_splits["val"]["text"], fpb_splits["val"]["label"], tokenizer
    )
    sentiment_test = TextClassificationDataset(
        fpb_splits["test"]["text"], fpb_splits["test"]["label"], tokenizer
    )

    stance_train_loader = DataLoader(stance_train, batch_size=BATCH_SIZE, shuffle=True)
    stance_val_loader = DataLoader(stance_val, batch_size=BATCH_SIZE)
    stance_test_loader = DataLoader(stance_test, batch_size=BATCH_SIZE)
    sentiment_train_loader = DataLoader(sentiment_train, batch_size=BATCH_SIZE, shuffle=True)
    sentiment_val_loader = DataLoader(sentiment_val, batch_size=BATCH_SIZE)
    sentiment_test_loader = DataLoader(sentiment_test, batch_size=BATCH_SIZE)

    # Weighted cross-entropy for stance (addresses FOMC class imbalance)
    stance_weights = compute_class_weights(fomc_splits["train"], num_classes=3)
    print(f"  Stance class weights: {dict(zip(STANCE_LABELS, [f'{w:.3f}' for w in stance_weights]))}")
    stance_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(stance_weights, dtype=torch.float).to(device)
    )
    sentiment_criterion = nn.CrossEntropyLoss()

    # Optimiser and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    # Total steps = sum of both loaders per epoch × num epochs
    steps_per_epoch = len(stance_train_loader) + len(sentiment_train_loader)
    total_steps = steps_per_epoch * MULTITASK_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )

    # Training loop
    best_avg_f1 = 0.0
    best_model_state = None

    for epoch in range(MULTITASK_EPOCHS):
        model.train()
        total_stance_loss = 0.0
        total_sentiment_loss = 0.0

        # Create iterators for both tasks
        stance_iter = iter(stance_train_loader)
        sentiment_iter = iter(sentiment_train_loader)
        stance_batches = len(stance_train_loader)
        sentiment_batches = len(sentiment_train_loader)
        total_batches = stance_batches + sentiment_batches

        pbar = tqdm(total=total_batches, desc=f"Epoch {epoch+1}/{MULTITASK_EPOCHS}")

        # Alternate between tasks
        stance_done = False
        sentiment_done = False
        step = 0

        while not (stance_done and sentiment_done):
            # Stance batch
            if not stance_done:
                try:
                    batch = next(stance_iter)
                    loss = _train_step(model, batch, stance_criterion, optimizer,
                                       scheduler, device, task="stance")
                    total_stance_loss += loss
                    pbar.update(1)
                    step += 1
                except StopIteration:
                    stance_done = True

            # Sentiment batch
            if not sentiment_done:
                try:
                    batch = next(sentiment_iter)
                    loss = _train_step(model, batch, sentiment_criterion, optimizer,
                                       scheduler, device, task="sentiment")
                    total_sentiment_loss += loss
                    pbar.update(1)
                    step += 1
                except StopIteration:
                    sentiment_done = True

        pbar.close()

        avg_stance_loss = total_stance_loss / max(stance_batches, 1)
        avg_sentiment_loss = total_sentiment_loss / max(sentiment_batches, 1)

        # Validation
        val_stance = _evaluate_multitask(model, stance_val_loader, STANCE_LABELS, device, "stance")
        val_sentiment = _evaluate_multitask(model, sentiment_val_loader, SENTIMENT_LABELS, device, "sentiment")
        avg_f1 = (val_stance["macro_f1"] + val_sentiment["macro_f1"]) / 2

        print(
            f"  Epoch {epoch+1}: "
            f"stance_loss={avg_stance_loss:.4f} sentiment_loss={avg_sentiment_loss:.4f}  "
            f"val_stance_f1={val_stance['macro_f1']:.4f}  "
            f"val_sentiment_f1={val_sentiment['macro_f1']:.4f}  "
            f"avg_f1={avg_f1:.4f}"
        )

        if avg_f1 > best_avg_f1:
            best_avg_f1 = avg_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best model
    model.load_state_dict(best_model_state)
    model.to(device)

    # Test evaluation
    print(f"\n{'─'*60}")
    print("  TEST SET EVALUATION")
    print(f"{'─'*60}")

    test_stance = _evaluate_multitask(
        model, stance_test_loader, STANCE_LABELS, device, "stance"
    )
    test_sentiment = _evaluate_multitask(
        model, sentiment_test_loader, SENTIMENT_LABELS, device, "sentiment"
    )

    print_classification_report(test_stance, "Multi-task FinBERT", "stance")
    print_classification_report(test_sentiment, "Multi-task FinBERT", "sentiment")

    # Confusion matrices
    y_true_s, y_pred_s = _get_multitask_predictions(model, stance_test_loader, device, "stance")
    y_true_t, y_pred_t = _get_multitask_predictions(model, sentiment_test_loader, device, "sentiment")

    plot_confusion_matrix(y_true_s, y_pred_s, STANCE_LABELS, "MultiTask_FinBERT", "stance")
    plot_confusion_matrix(y_true_t, y_pred_t, SENTIMENT_LABELS, "MultiTask_FinBERT", "sentiment")

    # Error analysis
    print("\n[ERROR ANALYSIS] Stance (FOMC):")
    stance_errors, stance_error_counts = error_analysis(
        fomc_splits["test"]["text"], y_true_s, y_pred_s, STANCE_LABELS
    )
    print("\n[ERROR ANALYSIS] Sentiment (Financial PhraseBank):")
    sentiment_errors, sentiment_error_counts = error_analysis(
        fpb_splits["test"]["text"], y_true_t, y_pred_t, SENTIMENT_LABELS
    )

    # Save results
    save_results(
        {"model": "Multi-task FinBERT", "task": "stance", **test_stance},
        "multitask_stance.json",
    )
    save_results(
        {"model": "Multi-task FinBERT", "task": "sentiment", **test_sentiment},
        "multitask_sentiment.json",
    )

    # Save model
    save_path = os.path.join(MODELS_DIR, "multitask_finbert")
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
    tokenizer.save_pretrained(save_path)
    print(f"  Multi-task model saved to {save_path}")

    return {
        "stance": test_stance,
        "sentiment": test_sentiment,
        "stance_errors": stance_errors,
        "sentiment_errors": sentiment_errors,
    }, model, tokenizer


def _train_step(model, batch, criterion, optimizer, scheduler, device, task):
    """Execute one training step on a single batch."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    logits = model(input_ids=input_ids, attention_mask=attention_mask, task=task)
    loss = criterion(logits, labels)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    return loss.item()


def _evaluate_multitask(model, dataloader, label_names, device, task):
    """Evaluate the multi-task model on a single task."""
    y_true, y_pred = _get_multitask_predictions(model, dataloader, device, task)
    return compute_metrics(y_true, y_pred, label_names)


def _get_multitask_predictions(model, dataloader, device, task):
    """Get predictions from the multi-task model for a specific task."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            logits = model(input_ids=input_ids, attention_mask=attention_mask, task=task)
            preds = logits.argmax(dim=-1).cpu().tolist()

            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    return all_labels, all_preds
