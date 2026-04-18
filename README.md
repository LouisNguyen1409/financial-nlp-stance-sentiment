# Financial NLP: Stance and Sentiment Classification

UNSW COMP6713 Natural Language Processing — 2026 T1 Group Project.

Team: ComTamSuonBiCha.

## Overview

This project builds and compares classifiers for two financial NLP tasks:

- **Stance classification** on the FOMC Hawkish-Dovish dataset. Each FOMC
  communication sentence is labelled `hawkish`, `dovish`, or `neutral` based on
  the implied monetary-policy stance.
- **Sentiment classification** on the Financial PhraseBank (all-annotator
  agreement subset). Each sentence is labelled `positive`, `negative`, or
  `neutral` from an investor's perspective.

We benchmark rule-based lexicons, TF-IDF linear models, pre-trained transformers
(zero-shot, few-shot linear probe, full fine-tune, LLRD + gradual unfreezing)
and a multi-task FinBERT with a shared encoder and two task heads.

## Setup

Requires Python 3.12.

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Project Structure

```
.
├── config.py
├── src/
│   ├── data_loader.py        FOMC + FPB loading
│   ├── baseline.py           TF-IDF + LR/SVM/trigrams
│   ├── lexicon.py            LM lexicon rule-based + hybrid
│   ├── pretrained_eval.py    Zero-shot + 16-shot linear probe
│   ├── finetune_fineBert.py  Single-task FinBERT fine-tune
│   ├── finetune_bert.py      BERT-base LLRD + Gradual Unfreezing
│   ├── multitask.py          MultiTaskFinBERT shared encoder + 2 heads
│   └── evaluate.py           metrics + confusion matrices + error analysis
├── run_experiments.py        orchestrator (steps 1-6)
├── cli.py                    CLI inference
├── demo.py                   Gradio demo
├── data_analysis.py          10 analysis plots + dataset stats
├── push_to_hf.py             HuggingFace Hub uploader
├── report/                   report.tex (to be regenerated) + report.pdf
├── presentation/             main.tex Beamer slides (to be regenerated) + .sty files
├── analysis/                 10 PNG plots
├── models/                   (gitignored) finbert_* / bert_llrd_* / multitask_finbert
└── results/                  JSON metrics + confusion matrix PNGs
```

## Running Experiments

Run the full pipeline:

```bash
python run_experiments.py
```

Or run individual steps:

```bash
python run_experiments.py --step 2   # TF-IDF baselines (LR / SVM / trigrams)
python run_experiments.py --step 3   # LM lexicon rule-based + hybrid
python run_experiments.py --step 4   # FinBERT zero-shot, few-shot, and fine-tune
python run_experiments.py --step 5   # BERT-base LLRD + gradual unfreezing
python run_experiments.py --step 6   # Multi-task FinBERT (shared encoder, two heads)
```

All metrics and confusion matrices are written to `results/`.

## Datasets and Lexicon

| Dataset                         | Total | Train                            | Val                    | Test                   |
|---------------------------------|-------|----------------------------------|------------------------|------------------------|
| FOMC Hawkish-Dovish             | 2,480 | 1,736 (455 / 424 / 857)          | 248 (65 / 61 / 122)    | 496 (130 / 121 / 245)  |
| Financial PhraseBank (allagree) | 2,264 | 1,584 (212 / 973 / 399)          | 227 (30 / 140 / 57)    | 453 (61 / 278 / 114)   |

Splits are 70 / 10 / 20 stratified by label. FOMC counts are
`dovish / hawkish / neutral`; FPB counts are `negative / neutral / positive`.
The Loughran-McDonald sentiment lexicon is used by the rule-based and hybrid
baselines.

## Models

| Model                           | One-liner                                                                 |
|---------------------------------|---------------------------------------------------------------------------|
| LM Lexicon (rule-based)         | Loughran-McDonald word-count sign rule; no training data used.            |
| TF-IDF + LR                     | Unigram TF-IDF features fed into logistic regression.                     |
| TF-IDF + SVM (LinearSVC)        | Unigram TF-IDF features with a LinearSVC classifier.                      |
| TF-IDF (trigrams) + LR          | TF-IDF over 1-3 grams with logistic regression.                           |
| TF-IDF + LM Lexicon             | Hybrid: TF-IDF features concatenated with LM lexicon counts, then LR.     |
| FinBERT (zero-shot)             | `ProsusAI/finbert` used directly for inference with no training.          |
| FinBERT (few-shot k=16)         | Frozen FinBERT encoder with a linear probe trained on 16 examples/class.  |
| BERT-base (few-shot k=16)       | Frozen `bert-base-uncased` with a 16-shot linear probe.                   |
| RoBERTa-base (few-shot k=16)    | Frozen `roberta-base` with a 16-shot linear probe.                        |
| FinBERT (fine-tuned)            | Single-task full fine-tune of FinBERT per task.                           |
| BERT-base LLRD + Gradual UF     | BERT-base with layer-wise LR decay and gradual unfreezing.                |
| Multi-task FinBERT              | Shared FinBERT encoder with two task heads trained jointly.               |

## Results Summary

Test-set accuracy and macro-F1 for every model, from
`results/all_results_summary.json`.

| Model                           | Sent. Acc | Sent. F1 | Stance Acc | Stance F1 |
|---------------------------------|-----------|----------|------------|-----------|
| LM Lexicon (rule-based)         | 0.6932    | 0.5315   | 0.4153     | 0.3885    |
| TF-IDF + LR                     | 0.8720    | 0.8232   | 0.6089     | 0.5873    |
| TF-IDF + SVM (LinearSVC)        | 0.8940    | 0.8534   | 0.6331     | 0.6061    |
| TF-IDF (trigrams) + LR          | 0.8786    | 0.8310   | 0.6109     | 0.5914    |
| TF-IDF + LM Lexicon             | 0.8543    | 0.8050   | 0.6109     | 0.5863    |
| FinBERT (zero-shot)             | 0.9735    | 0.9650   | 0.4980     | 0.4874    |
| FinBERT (few-shot k=16)         | 0.9779    | 0.9670   | 0.4859     | 0.4534    |
| BERT-base (few-shot k=16)       | 0.7417    | 0.6500   | 0.3851     | 0.3744    |
| RoBERTa-base (few-shot k=16)    | 0.7682    | 0.6722   | 0.3730     | 0.3600    |
| FinBERT (fine-tuned)            | 0.9669    | 0.9459   | 0.6371     | 0.6194    |
| BERT-base LLRD + Gradual UF     | 0.9757    | 0.9670   | 0.6512     | 0.6383    |
| Multi-task FinBERT              | 0.9779    | 0.9666   | 0.6492     | 0.6384    |

BERT-base with LLRD + gradual unfreezing and the Multi-task FinBERT are
essentially tied for the top spot: LLRD is slightly ahead on sentiment while
the multi-task model is slightly ahead on stance. The multi-task model beats
single-task fine-tuned FinBERT by +2.07 pp sentiment macro-F1 and +1.90 pp
stance macro-F1, and careful fine-tuning of a general-purpose BERT with LLRD +
gradual unfreezing is enough to match domain-pretrained FinBERT on both tasks.

## CLI Usage

The CLI is defined in `cli.py` and takes the following flags:

- `--text TEXT` — classify a single sentence.
- `--file FILE` — read a text file with one sentence per line.
- `--model {multitask, finetune}` — which model to use; default `multitask`.

Examples:

```bash
# Interactive mode
python cli.py

# Single sentence
python cli.py --text "The Fed signaled further rate hikes ahead"

# Batch from file
python cli.py --file input.txt

# Use the per-task fine-tuned FinBERT models instead
python cli.py --model finetune --text "Markets rallied on dovish comments"
```

Each prediction returns a stance label and a sentiment label, together with
their confidence scores and per-class probabilities.

## Gradio Demo

```bash
python demo.py
```

Opens a browser demo at <http://localhost:7860> with a text box and
side-by-side stance and sentiment outputs.

## Data Analysis Plots

```bash
python data_analysis.py
```

Writes ten PNG plots (label distributions, length histograms, vocabulary
overlap, top-n-grams, etc.) and dataset statistics to the `analysis/`
directory.

## Trained Models on HuggingFace

Uploaded via `push_to_hf.py` once training is committed:

- `Louisnguyen/multitask-finbert-financial`
- `Louisnguyen/finbert-financial-stance`
- `Louisnguyen/finbert-financial-sentiment`
- `Louisnguyen/bert-llrd-financial-stance`
- `Louisnguyen/bert-llrd-financial-sentiment`

## Hardware

All models were trained on an Apple M3 Max with the PyTorch MPS backend.
`config.py` automatically selects MPS, falling back to CUDA and then CPU.

## Full Documentation

For a full write-up of the methodology, ablations, and per-class analysis see
`DOC.md` (English) and `DOC_VI.md` (Vietnamese).
