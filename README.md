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
│   ├── baseline.py           # TF-IDF + LR + alternative baselines (SVM, trigrams)
│   ├── lexicon.py            # Loughran-McDonald lexicon-based classification
│   ├── pretrained_eval.py    # Zero-shot and few-shot evaluation
│   ├── finetune_fineBert.py  # Single-task FinBERT fine-tuning
│   ├── finetune_bert.py      # BERT-base with LLRD + Gradual Unfreezing
│   ├── multitask.py          # Multi-task model (shared encoder + dual heads)
│   └── evaluate.py           # Metrics, confusion matrices, error analysis
├── run_experiments.py        # Main experiment runner (steps 1–6)
├── cli.py                    # Command-line inference interface
├── demo.py                   # Gradio web demo
├── data_analysis.py          # Generates 10 analysis plots + dataset stats
├── push_to_hf.py             # Uploads trained models to HuggingFace Hub
├── create_presentation.py    # Builds slide-deck / report artefacts
├── requirements.txt          # Python dependencies
├── analysis/                 # 10 PNG plots produced by data_analysis.py
├── models/                   # Saved trained models
└── results/                  # Evaluation results (JSON) + confusion matrices (PNG)
```

## Running Experiments

```bash
# Run all experiments end-to-end
python run_experiments.py

# Or run individual steps:
python run_experiments.py --step 2   # Baselines (TF-IDF+LR, +SVM, +trigrams) + LM lexicon
python run_experiments.py --step 3   # Pre-trained model evaluation (zero/few-shot)
python run_experiments.py --step 4   # Single-task FinBERT fine-tuning
python run_experiments.py --step 5   # Multi-task training
python run_experiments.py --step 6   # BERT-base LLRD + Gradual Unfreezing
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
| TF-IDF + LR | Non-neural baseline: TF-IDF (1-2 grams) + Logistic Regression |
| TF-IDF + SVM | Alternative baseline: TF-IDF (1-2 grams) + LinearSVC (C=1.0) |
| TF-IDF (trigrams) + LR | Alternative baseline: TF-IDF (1-3 grams, max_features=80k) + LR |
| LM Lexicon (rule-based) | Rule-based classification using Loughran-McDonald word counts |
| TF-IDF + LM Lexicon | TF-IDF augmented with Loughran-McDonald lexicon features |
| FinBERT (native) | Zero-shot using ProsusAI/finbert's pre-trained sentiment head |
| FinBERT / BERT / RoBERTa (few-shot) | Linear probe on frozen embeddings with k=16 examples per class |
| FinBERT (fine-tuned) | Full fine-tuning of FinBERT on each dataset separately |
| BERT-base LLRD + Gradual UF | BERT-base-uncased with layer-wise LR decay + gradual unfreezing + label smoothing |
| Multi-task FinBERT | Shared encoder + task-specific heads, trained on both datasets jointly |

## Results Summary

| Model | Task | Accuracy | Macro-F1 |
|-------|------|----------|----------|
| LM Lexicon (rule-based) | Stance    | 0.4153 | 0.3885 |
| LM Lexicon (rule-based) | Sentiment | 0.6932 | 0.5315 |
| TF-IDF + LR             | Stance    | 0.6089 | 0.5873 |
| TF-IDF + LR             | Sentiment | 0.8720 | 0.8232 |
| TF-IDF + SVM            | Stance    | 0.6331 | 0.6061 |
| TF-IDF + SVM            | Sentiment | 0.8940 | 0.8534 |
| TF-IDF (trigrams) + LR  | Stance    | 0.6109 | 0.5914 |
| TF-IDF (trigrams) + LR  | Sentiment | 0.8786 | 0.8310 |
| TF-IDF + LM Lexicon     | Stance    | 0.6109 | 0.5863 |
| TF-IDF + LM Lexicon     | Sentiment | 0.8543 | 0.8050 |
| FinBERT (zero-shot)     | Stance    | 0.4980 | 0.4874 |
| FinBERT (zero-shot)     | Sentiment | 0.9735 | 0.9650 |
| FinBERT few-shot (k=16) | Stance    | 0.4859 | 0.4552 |
| FinBERT few-shot (k=16) | Sentiment | 0.9801 | 0.9690 |
| BERT-base few-shot (k=16)    | Stance    | 0.3790 | 0.3694 |
| BERT-base few-shot (k=16)    | Sentiment | 0.7461 | 0.6599 |
| RoBERTa-base few-shot (k=16) | Stance    | 0.3589 | 0.3489 |
| RoBERTa-base few-shot (k=16) | Sentiment | 0.7572 | 0.6439 |
| FinBERT fine-tuned      | Stance    | 0.6129 | 0.5988 |
| FinBERT fine-tuned      | Sentiment | 0.9669 | 0.9467 |
| BERT-base LLRD + UF     | Stance    | 0.6512 | 0.6371 |
| BERT-base LLRD + UF     | Sentiment | 0.9691 | 0.9533 |
| **Multi-task FinBERT**  | **Stance**    | **0.6774** | **0.6684** |
| **Multi-task FinBERT**  | **Sentiment** | **0.9845** | **0.9772** |

Numbers above come directly from `results/all_results_summary.json`.

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

## Data Analysis Plots

```bash
python data_analysis.py
# Writes 10 PNGs to analysis/ and a dataset-stats JSON
```

## Trained Models on HuggingFace

Pre-trained model weights are hosted on HuggingFace Hub under the
[`Louisnguyen/*`](https://huggingface.co/Louisnguyen) namespace:

- `Louisnguyen/multitask-finbert-financial` — Multi-task model (best)
- `Louisnguyen/finbert-financial-stance` — FinBERT fine-tuned for stance
- `Louisnguyen/finbert-financial-sentiment` — FinBERT fine-tuned for sentiment
- `Louisnguyen/bert-llrd-financial-stance` — BERT-base + LLRD for stance
- `Louisnguyen/bert-llrd-financial-sentiment` — BERT-base + LLRD for sentiment

Upload your own runs via `python push_to_hf.py` (requires `HF_TOKEN`).

## Hardware

- Training was performed on Apple M3 Max (MPS backend)
- All models (FinBERT / BERT-base ~110M params) train comfortably on Apple Silicon
- GPU cluster (H200) available for larger-scale experiments if needed

## Full Documentation

- `DOC.md` — complete technical documentation (English)
- `DOC_VI.md` — complete technical documentation (Vietnamese)
