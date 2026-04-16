# Financial NLP: Stance & Sentiment Classification

COMP6713 2026 T1 — Group Project

## Overview

This project addresses two financial NLP classification tasks:

1. **Stance Classification**: Classifying FOMC (Federal Reserve) communications as *hawkish*, *dovish*, or *neutral* based on implied monetary policy position.
2. **Sentiment Classification**: Classifying financial news sentences as *positive*, *negative*, or *neutral* in terms of market sentiment.

## Setup

```bash
# Create virtual environment (requires Python 3.12)
/opt/homebrew/bin/python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
.
├── config.py                 # Central configuration (hyperparameters, paths, labels)
├── src/
│   ├── data_loader.py        # Dataset loading (FOMC + Financial PhraseBank)
│   ├── baseline.py           # TF-IDF + Logistic Regression baseline
│   ├── lexicon.py            # Loughran-McDonald lexicon-based classification
│   ├── pretrained_eval.py    # Zero-shot and few-shot evaluation
│   ├── finetune.py           # Single-task FinBERT fine-tuning
│   ├── multitask.py          # Multi-task model (shared encoder + dual heads)
│   └── evaluate.py           # Metrics, confusion matrices, error analysis
├── run_experiments.py        # Main experiment runner
├── cli.py                    # Command-line inference interface
├── demo.py                   # Gradio web demo
├── requirements.txt          # Python dependencies
├── models/                   # Saved trained models
└── results/                  # Evaluation results and plots
```

## Running Experiments

```bash
# Run all experiments end-to-end
python run_experiments.py

# Or run individual steps:
python run_experiments.py --step 2   # Baseline (TF-IDF + LR)
python run_experiments.py --step 3   # Pre-trained model evaluation
python run_experiments.py --step 4   # FinBERT fine-tuning
python run_experiments.py --step 5   # Multi-task training
```

## Datasets & Lexicon

| Dataset | Source | Labels | Size |
|---------|--------|--------|------|
| FOMC Hawkish-Dovish | [gtfintechlab/fomc_communication](https://huggingface.co/datasets/gtfintechlab/fomc_communication) | hawkish, dovish, neutral | 2,480 |
| Financial PhraseBank | [gtfintechlab/financial_phrasebank_sentences_allagree](https://huggingface.co/datasets/gtfintechlab/financial_phrasebank_sentences_allagree) | positive, negative, neutral | 2,264 |
| LM Lexicon | Loughran & McDonald (2011) Financial Sentiment Dictionary | positive, negative, uncertainty + hawkish/dovish | ~2,700 words |

## Models

| Model | Description |
|-------|-------------|
| TF-IDF + LR | Non-neural baseline using TF-IDF features + Logistic Regression |
| LM Lexicon (rule-based) | Rule-based classification using Loughran-McDonald word counts |
| TF-IDF + LM Lexicon | TF-IDF augmented with Loughran-McDonald lexicon features |
| FinBERT (native) | Zero-shot using ProsusAI/finbert's pre-trained sentiment head |
| FinBERT / BERT / RoBERTa (few-shot) | Linear probe on frozen embeddings with k=16 examples per class |
| FinBERT (fine-tuned) | Full fine-tuning on each dataset separately |
| Multi-task FinBERT | Shared encoder + task-specific heads, trained on both datasets jointly |

## Results Summary

| Model | Task | Accuracy | Macro-F1 |
|-------|------|----------|----------|
| LM Lexicon (rule-based) | Stance | 0.4153 | 0.3885 |
| LM Lexicon (rule-based) | Sentiment | 0.6932 | 0.5315 |
| TF-IDF + LR | Stance | 0.6089 | 0.5873 |
| TF-IDF + LR | Sentiment | 0.8720 | 0.8232 |
| TF-IDF + LM Lexicon | Stance | 0.6109 | 0.5863 |
| TF-IDF + LM Lexicon | Sentiment | 0.8543 | 0.8050 |
| FinBERT (zero-shot) | Sentiment | 0.9735 | 0.9650 |
| FinBERT few-shot (k=16) | Sentiment | 0.9779 | 0.9670 |
| BERT-base few-shot (k=16) | Sentiment | 0.7417 | 0.6500 |
| RoBERTa-base few-shot (k=16) | Sentiment | 0.7682 | 0.6722 |
| FinBERT fine-tuned | Stance | 0.6371 | 0.6194 |
| FinBERT fine-tuned | Sentiment | 0.9669 | 0.9459 |
| **Multi-task FinBERT** | **Stance** | **0.6593** | **0.6478** |
| **Multi-task FinBERT** | **Sentiment** | **0.9801** | **0.9666** |

## CLI Usage

```bash
# Interactive mode
python cli.py

# Single sentence
python cli.py --text "The Fed signaled further rate hikes ahead"

# From file
python cli.py --file input.txt

# Use fine-tuned (single-task) models instead of multi-task
python cli.py --model finetune --text "Markets rallied on dovish comments"
```

## Gradio Demo

```bash
python demo.py
# Opens at http://localhost:7860
```

## Trained Models on HuggingFace

Pre-trained model weights are hosted on HuggingFace Hub:

**[Louisnguyen/financial-nlp-stance-sentiment](https://huggingface.co/Louisnguyen/financial-nlp-stance-sentiment)**

Contains:
- `multitask_finbert/` — Multi-task model (best performance)
- `finbert_stance/` — FinBERT fine-tuned for stance classification
- `finbert_sentiment/` — FinBERT fine-tuned for sentiment classification

## Hardware

- Training was performed on Apple M3 Max (MPS backend)
- All models (FinBERT ~110M params) train comfortably on Apple Silicon
- GPU cluster (H200) available for larger-scale experiments if needed
