"""
Command-line interface for the Financial NLP Stance & Sentiment classifier.

Accepts raw financial text input and returns:
  - Stance prediction (hawkish / dovish / neutral) + confidence score
  - Sentiment prediction (positive / neutral / negative) + confidence score

Usage:
    # Interactive mode
    python cli.py

    # Single sentence
    python cli.py --text "The Fed signaled further rate hikes ahead"

    # From a file (one sentence per line)
    python cli.py --file input.txt

    # Specify model (default: multitask)
    python cli.py --model finetune --text "Markets rallied on dovish comments"
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DEVICE, MODELS_DIR, MAX_SEQ_LENGTH,
    FINBERT_MODEL, SENTIMENT_LABELS, STANCE_LABELS,
)


def load_multitask_model():
    """Load the multi-task FinBERT model."""
    from src.multitask import MultiTaskFinBERT

    model_path = os.path.join(MODELS_DIR, "multitask_finbert")
    if not os.path.exists(os.path.join(model_path, "model.pt")):
        print("ERROR: Multi-task model not found. Run training first:")
        print("  python run_experiments.py --step 5")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = MultiTaskFinBERT()
    model.load_state_dict(torch.load(
        os.path.join(model_path, "model.pt"),
        map_location=DEVICE, weights_only=True,
    ))
    model.to(DEVICE)
    model.eval()
    return model, tokenizer, "multitask"


def load_finetune_models():
    """Load separately fine-tuned FinBERT models for each task."""
    models = {}
    for task in ["stance", "sentiment"]:
        model_path = os.path.join(MODELS_DIR, f"finbert_{task}")
        if not os.path.exists(model_path):
            print(f"ERROR: Fine-tuned {task} model not found. Run training first:")
            print(f"  python run_experiments.py --step 4")
            sys.exit(1)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(DEVICE)
        model.eval()
        models[task] = (model, tokenizer)
    return models


def predict_multitask(text, model, tokenizer):
    """Get predictions from the multi-task model for both tasks."""
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True,
        max_length=MAX_SEQ_LENGTH, padding=True,
    ).to(DEVICE)

    results = {}
    with torch.no_grad():
        for task, labels in [("stance", STANCE_LABELS), ("sentiment", SENTIMENT_LABELS)]:
            logits = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                task=task,
            )
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu()
            pred_idx = probs.argmax().item()
            results[task] = {
                "label": labels[pred_idx],
                "confidence": probs[pred_idx].item(),
                "all_scores": {labels[i]: probs[i].item() for i in range(len(labels))},
            }
    return results


def predict_finetune(text, models):
    """Get predictions from separate fine-tuned models."""
    results = {}
    for task, labels in [("stance", STANCE_LABELS), ("sentiment", SENTIMENT_LABELS)]:
        model, tokenizer = models[task]
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=MAX_SEQ_LENGTH, padding=True,
        ).to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu()
            pred_idx = probs.argmax().item()
            results[task] = {
                "label": labels[pred_idx],
                "confidence": probs[pred_idx].item(),
                "all_scores": {labels[i]: probs[i].item() for i in range(len(labels))},
            }
    return results


def format_prediction(results):
    """Format prediction results for console output."""
    lines = []
    for task in ["stance", "sentiment"]:
        r = results[task]
        lines.append(f"\n  {task.upper()}:")
        lines.append(f"    Prediction : {r['label']}")
        lines.append(f"    Confidence : {r['confidence']:.4f}")
        lines.append(f"    All scores :")
        for label, score in sorted(r["all_scores"].items(), key=lambda x: -x[1]):
            bar = "█" * int(score * 30)
            lines.append(f"      {label:>10}: {score:.4f} {bar}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Financial NLP Stance & Sentiment Classifier"
    )
    parser.add_argument("--text", type=str, help="A single sentence to classify")
    parser.add_argument("--file", type=str, help="Path to a text file (one sentence per line)")
    parser.add_argument(
        "--model", type=str, default="multitask",
        choices=["multitask", "finetune"],
        help="Which model to use (default: multitask)",
    )
    args = parser.parse_args()

    # Load model
    print(f"\n  Loading model ({args.model}) …")
    if args.model == "multitask":
        model, tokenizer, _ = load_multitask_model()
        predict_fn = lambda text: predict_multitask(text, model, tokenizer)
    else:
        models = load_finetune_models()
        predict_fn = lambda text: predict_finetune(text, models)

    print("  Model loaded successfully.\n")

    # Process input
    if args.text:
        # Single sentence from command line
        print(f"  Input: {args.text}")
        results = predict_fn(args.text)
        print(format_prediction(results))

    elif args.file:
        # Read sentences from file
        with open(args.file, encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]
        for i, sent in enumerate(sentences):
            print(f"\n{'─'*60}")
            print(f"  [{i+1}/{len(sentences)}] {sent}")
            results = predict_fn(sent)
            print(format_prediction(results))

    else:
        # Interactive mode
        print("  Interactive mode. Type a financial sentence and press Enter.")
        print("  Type 'quit' to exit.\n")
        while True:
            try:
                text = input("  >>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Goodbye!")
                break
            if text.lower() in ("quit", "exit", "q"):
                print("  Goodbye!")
                break
            if not text:
                continue
            results = predict_fn(text)
            print(format_prediction(results))
            print()


if __name__ == "__main__":
    main()
