"""
Gradio web demo for the Financial NLP Stance & Sentiment classifier.

Provides a simple web interface where users can:
  - Paste any financial sentence
  - Receive stance (hawkish / dovish / neutral) predictions with confidence
  - Receive sentiment (positive / neutral / negative) predictions with confidence

Usage:
    python demo.py
    # Opens at http://localhost:7860
"""

import os
import sys
import torch
import torch.nn.functional as F
import gradio as gr
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DEVICE, MODELS_DIR, MAX_SEQ_LENGTH,
    FINBERT_MODEL, SENTIMENT_LABELS, STANCE_LABELS,
)
from src.multitask import MultiTaskFinBERT


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_model():
    """Load the multi-task model. Falls back to fine-tuned models if not available."""
    multitask_path = os.path.join(MODELS_DIR, "multitask_finbert")

    if os.path.exists(os.path.join(multitask_path, "model.pt")):
        print("[DEMO] Loading multi-task model …")
        tokenizer = AutoTokenizer.from_pretrained(multitask_path)
        model = MultiTaskFinBERT()
        model.load_state_dict(torch.load(
            os.path.join(multitask_path, "model.pt"),
            map_location=DEVICE, weights_only=True,
        ))
        model.to(DEVICE)
        model.eval()
        return model, tokenizer, "Multi-task FinBERT"
    else:
        raise FileNotFoundError(
            "No trained model found. Please run training first:\n"
            "  python run_experiments.py --step 5"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Prediction function
# ──────────────────────────────────────────────────────────────────────────────

def predict(text, model, tokenizer):
    """
    Predict stance and sentiment for the input text.
    Returns two dicts of {label: confidence} for Gradio's Label component.
    """
    if not text or not text.strip():
        return {}, {}

    inputs = tokenizer(
        text.strip(),
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=True,
    ).to(DEVICE)

    stance_scores = {}
    sentiment_scores = {}

    with torch.no_grad():
        # Stance prediction
        stance_logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            task="stance",
        )
        stance_probs = F.softmax(stance_logits, dim=-1).squeeze(0).cpu()
        for i, label in enumerate(STANCE_LABELS):
            stance_scores[label] = float(stance_probs[i])

        # Sentiment prediction
        sentiment_logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            task="sentiment",
        )
        sentiment_probs = F.softmax(sentiment_logits, dim=-1).squeeze(0).cpu()
        for i, label in enumerate(SENTIMENT_LABELS):
            sentiment_scores[label] = float(sentiment_probs[i])

    return stance_scores, sentiment_scores


# ──────────────────────────────────────────────────────────────────────────────
# Gradio interface
# ──────────────────────────────────────────────────────────────────────────────

def create_demo():
    """Build and return the Gradio demo interface."""
    model, tokenizer, model_name = load_model()

    def classify(text):
        stance_scores, sentiment_scores = predict(text, model, tokenizer)
        return stance_scores, sentiment_scores

    # Example financial sentences for quick testing
    examples = [
        "The Federal Reserve raised interest rates by 75 basis points, signaling more hikes ahead.",
        "The committee decided to maintain the current target range for the federal funds rate.",
        "GDP growth exceeded expectations, driven by strong consumer spending.",
        "Markets tumbled as inflation data came in higher than expected.",
        "The central bank signaled it may begin cutting rates in the coming months.",
        "Corporate earnings beat analyst expectations across most sectors.",
        "Unemployment claims rose sharply, raising concerns about economic slowdown.",
        "The Fed Chair emphasized the need for continued patience on rate decisions.",
    ]

    with gr.Blocks(
        title="Financial NLP: Stance & Sentiment Classifier",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            f"""
            # Financial NLP: Stance & Sentiment Classifier
            **Model:** {model_name}

            Enter a financial sentence to classify its **monetary policy stance**
            (hawkish / dovish / neutral) and **market sentiment**
            (positive / neutral / negative).
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Financial Text",
                    placeholder="Enter a financial sentence here …",
                    lines=3,
                )
                submit_btn = gr.Button("Classify", variant="primary")

            with gr.Column(scale=1):
                stance_output = gr.Label(label="Stance (Monetary Policy)", num_top_classes=3)
                sentiment_output = gr.Label(label="Sentiment (Market)", num_top_classes=3)

        gr.Examples(
            examples=examples,
            inputs=text_input,
            outputs=[stance_output, sentiment_output],
            fn=classify,
            cache_examples=False,
        )

        submit_btn.click(
            fn=classify,
            inputs=text_input,
            outputs=[stance_output, sentiment_output],
        )
        text_input.submit(
            fn=classify,
            inputs=text_input,
            outputs=[stance_output, sentiment_output],
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
