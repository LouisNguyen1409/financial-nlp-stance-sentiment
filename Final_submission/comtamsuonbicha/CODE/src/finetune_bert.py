import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DEVICE, SEED, BATCH_SIZE, MAX_SEQ_LENGTH,
    WEIGHT_DECAY, MODELS_DIR,
    BERT_BASE_MODEL,
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

torch.manual_seed(SEED)

# ─── LLRD-specific hyperparameters ────────────────────────────────────────────
LLRD_BASE_LR    = 2e-5   # LR assigned to the top-most (head) layer group
LLRD_DECAY      = 0.9    # Multiplicative per-layer LR decay going downwards
LABEL_SMOOTHING = 0.1    # Label smoothing ε for CrossEntropyLoss
BERT_EPOCHS     = 10     # Training epochs (more than standard to allow full unfreezing)
NUM_BERT_LAYERS = 12     # bert-base-uncased has 12 transformer layers


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset wrapper
# ──────────────────────────────────────────────────────────────────────────────

class TextClassificationDataset(TorchDataset):
    """Tokenises texts on-the-fly and returns model-ready tensors."""

    def __init__(self, texts, labels, tokenizer, max_length=MAX_SEQ_LENGTH):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Layer grouping for LLRD
# ──────────────────────────────────────────────────────────────────────────────

def _build_layer_groups(model):
    """
    Partition model parameters into ordered layer groups for LLRD.

    Groups are returned from top (closest to output) to bottom (embeddings):
      index 0  → classifier head + pooler
      index 1  → transformer layer 11
      index 2  → transformer layer 10
      ...
      index 12 → transformer layer 0
      index 13 → embeddings

    Returns:
        List of dicts: [{"name": str, "params": [Tensor, ...], "lr": float}, ...]
    """
    groups = []

    # Head: classifier linear layer + BERT pooler
    groups.append({
        "name":   "head",
        "params": (list(model.classifier.parameters()) +
                   list(model.bert.pooler.parameters())),
        "lr":     LLRD_BASE_LR,
    })

    # Transformer encoder layers (top → bottom)
    for layer_idx in range(NUM_BERT_LAYERS - 1, -1, -1):
        depth = (NUM_BERT_LAYERS - 1) - layer_idx  # 0 at top, 11 at bottom
        lr    = LLRD_BASE_LR * (LLRD_DECAY ** (depth + 1))
        groups.append({
            "name":   f"encoder_layer_{layer_idx}",
            "params": list(model.bert.encoder.layer[layer_idx].parameters()),
            "lr":     lr,
        })

    # Token / position / type embeddings (lowest LR)
    emb_lr = LLRD_BASE_LR * (LLRD_DECAY ** (NUM_BERT_LAYERS + 1))
    groups.append({
        "name":   "embeddings",
        "params": list(model.bert.embeddings.parameters()),
        "lr":     emb_lr,
    })

    return groups


def _build_optimizer(model, epoch):
    """
    Build an AdamW optimizer for the current epoch under the gradual-unfreezing
    and LLRD schedule.

    Unfreezing schedule (epoch is 1-indexed):
      epoch  1 → head only                       (1 group active)
      epoch  2 → head + encoder_layer_11         (2 groups)
      epoch  3 → head + layers 11-10             (3 groups)
      ...
      epoch 14 → all 14 groups active

    Parameters in frozen groups have requires_grad=False and are excluded from
    the optimizer, saving computation and memory.

    Returns:
        (optimizer, list_of_active_group_names)
    """
    all_groups = _build_layer_groups(model)
    n_active   = min(epoch, len(all_groups))

    # Freeze all parameters first
    for p in model.parameters():
        p.requires_grad_(False)

    # Unfreeze and configure active groups
    param_groups = []
    active_names = []
    for group in all_groups[:n_active]:
        for p in group["params"]:
            p.requires_grad_(True)
        # No weight decay on the head (bias-heavy) to avoid shrinking logits
        wd = 0.0 if group["name"] == "head" else WEIGHT_DECAY
        param_groups.append({
            "params":       group["params"],
            "lr":           group["lr"],
            "weight_decay": wd,
        })
        active_names.append(group["name"])

    optimizer = torch.optim.AdamW(param_groups)
    return optimizer, active_names


# ──────────────────────────────────────────────────────────────────────────────
# Training / evaluation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch; return average loss."""
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="    train", leave=False):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss    = criterion(outputs.logits, labels)

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping prevents exploding gradients when many layers are active
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def _get_predictions(model, loader, device):
    """Run inference; return (y_true, y_pred) lists."""
    model.eval()
    all_true, all_pred = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds   = outputs.logits.argmax(dim=-1).cpu().tolist()

            all_pred.extend(preds)
            all_true.extend(labels.tolist())

    return all_true, all_pred


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def finetune_bert_llrd(train_split, val_split, test_split, label_names, task_name):
    """
    Fine-tune BERT-base-uncased with LLRD + gradual unfreezing.

    Each epoch:
      1. Rebuild the optimizer to unfreeze one additional layer group.
      2. Assign layer-decayed learning rates to all active groups.
      3. Train for one epoch with label-smoothed (+ optionally weighted) CE loss.
      4. Evaluate on the validation set; save best checkpoint by macro-F1.

    Args:
        train_split : HuggingFace Dataset with 'text' and 'label' columns
        val_split   : HuggingFace Dataset with 'text' and 'label' columns
        test_split  : HuggingFace Dataset with 'text' and 'label' columns
        label_names : list of label strings (3 classes)
        task_name   : 'stance' or 'sentiment'

    Returns:
        (metrics_dict, model, tokenizer)
    """
    print(f"\n{'='*60}")
    print(f"  BERT-base LLRD + Gradual Unfreezing — {task_name.upper()}")
    print(f"{'='*60}")
    print(f"  Model       : {BERT_BASE_MODEL}")
    print(f"  Base LR     : {LLRD_BASE_LR}   Layer decay : {LLRD_DECAY}")
    print(f"  Label smooth: {LABEL_SMOOTHING}  Epochs      : {BERT_EPOCHS}")

    device     = DEVICE
    num_labels = len(label_names)

    tokenizer = AutoTokenizer.from_pretrained(BERT_BASE_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(
        BERT_BASE_MODEL, num_labels=num_labels,
    )
    model.to(device)

    # ── DataLoaders ────────────────────────────────────────────────────────
    train_ds = TextClassificationDataset(
        train_split["text"], train_split["label"], tokenizer
    )
    val_ds   = TextClassificationDataset(
        val_split["text"],   val_split["label"],   tokenizer
    )
    test_ds  = TextClassificationDataset(
        test_split["text"],  test_split["label"],  tokenizer
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    # ── Loss function ──────────────────────────────────────────────────────
    # For the stance task the FOMC dataset is class-imbalanced, so combine
    # inverse-frequency class weights with label smoothing.
    # For sentiment the FPB is more balanced, so plain label smoothing suffices.
    if task_name == "stance":
        class_weights = compute_class_weights(train_split, num_classes=num_labels)
        print(
            f"  Class weights: "
            f"{dict(zip(label_names, [f'{w:.3f}' for w in class_weights]))}"
        )
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float).to(device),
            label_smoothing=LABEL_SMOOTHING,
        )
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # ── Training loop ──────────────────────────────────────────────────────
    best_val_f1      = 0.0
    best_model_state = None

    print(f"\n  {'Epoch':>5}  {'Deepest active group':>22}  "
          f"{'Trainable':>10}  {'Train loss':>11}  {'Val acc':>8}  {'Val F1':>7}")
    print(f"  {'─'*5}  {'─'*22}  {'─'*10}  {'─'*11}  {'─'*8}  {'─'*7}")

    for epoch in range(1, BERT_EPOCHS + 1):
        optimizer, active_names = _build_optimizer(model, epoch)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        deepest   = active_names[-1] if active_names else "none"

        avg_loss    = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        y_true_v, y_pred_v = _get_predictions(model, val_loader, device)
        val_metrics = compute_metrics(y_true_v, y_pred_v, label_names)

        print(
            f"  {epoch:>5}  {deepest:>22}  {trainable:>10,}  "
            f"{avg_loss:>11.4f}  {val_metrics['accuracy']:>8.4f}  "
            f"{val_metrics['macro_f1']:>7.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1      = val_metrics["macro_f1"]
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }

    # ── Test evaluation ────────────────────────────────────────────────────
    model.load_state_dict(best_model_state)
    model.to(device)

    y_true_t, y_pred_t = _get_predictions(model, test_loader, device)
    test_metrics       = compute_metrics(y_true_t, y_pred_t, label_names)

    print_classification_report(test_metrics, "BERT-base (LLRD)", task_name)
    plot_confusion_matrix(
        y_true_t, y_pred_t, label_names, "BERT_base_LLRD", task_name
    )
    save_results(
        {
            "model": "BERT-base LLRD + Gradual Unfreezing",
            "task":  task_name,
            **test_metrics,
        },
        f"finetune_bert_llrd_{task_name}.json",
    )

    # Error analysis: examine the most common misclassification patterns
    print(f"\n[ERROR ANALYSIS] {task_name.upper()}:")
    error_analysis(test_split["text"], y_true_t, y_pred_t, label_names)

    # ── Save model ─────────────────────────────────────────────────────────
    save_path = os.path.join(MODELS_DIR, f"bert_llrd_{task_name}")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\n  Model saved → {save_path}")

    return test_metrics, model, tokenizer
