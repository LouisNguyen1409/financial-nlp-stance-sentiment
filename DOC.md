# Financial NLP Project — Complete Technical Documentation

This document is a comprehensive guide to every aspect of this project.
It is written for someone completely new to NLP and machine learning.
By the end, you should understand **why** every decision was made,
**how** every algorithm works, and **what** every line of code does.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Background Theory](#2-background-theory)
3. [Environment & Libraries](#3-environment--libraries)
4. [Project Architecture](#4-project-architecture)
5. [Datasets In Depth](#5-datasets-in-depth)
6. [The Loughran-McDonald Lexicon](#6-the-loughran-mcdonald-lexicon)
7. [Baseline Model: TF-IDF + Logistic Regression](#7-baseline-model-tf-idf--logistic-regression)
8. [Pre-trained Transformer Models](#8-pre-trained-transformer-models)
9. [Fine-tuning FinBERT](#9-fine-tuning-finbert)
10. [Multi-task Learning](#10-multi-task-learning)
11. [Evaluation Methodology](#11-evaluation-methodology)
12. [CLI and Demo](#12-cli-and-demo)
13. [Bugs Encountered and How They Were Fixed](#13-bugs-encountered-and-how-they-were-fixed)
14. [Results Analysis](#14-results-analysis)
15. [Key Takeaways](#15-key-takeaways)

---

## 1. What This Project Does

### 1.1 The Two Tasks

This project solves two **text classification** problems in the financial domain:

**Task 1 — Stance Classification (FOMC Dataset)**

Given a sentence from a U.S. Federal Reserve (FOMC) meeting, classify it as:
- **Hawkish**: The text implies tightening monetary policy (raising interest rates,
  reducing money supply). Example: *"Inflation remains elevated and the committee
  believes further rate increases are warranted."*
- **Dovish**: The text implies loosening monetary policy (cutting rates, stimulus).
  Example: *"Economic weakness suggests the need for continued accommodative policy."*
- **Neutral**: The text does not clearly lean either way.
  Example: *"Broad equity price indexes fell sharply over the intermeeting period."*

**Why does this matter?** Central bank communications move financial markets. If a
trader can automatically classify Fed statements as hawkish or dovish, they can
react faster. This is an active area of research in financial NLP.

**Task 2 — Sentiment Classification (Financial PhraseBank)**

Given a sentence from a financial news article, classify it as:
- **Positive**: Good news for the company/market. Example: *"Revenue grew 15% year-over-year."*
- **Negative**: Bad news. Example: *"The company reported a net loss of $50 million."*
- **Neutral**: Factual, no clear sentiment. Example: *"The company is headquartered in Helsinki."*

### 1.2 Why Two Tasks?

We deliberately chose two tasks to:
1. Compare how models handle different types of financial language
2. Enable **multi-task learning** — training one model on both tasks simultaneously
3. Show that domain-specific pre-training (FinBERT) helps across financial tasks

### 1.3 Credit System

The course requires a minimum of 80 credits across four parts:

| Part | Minimum | What We Did | Credits |
|------|---------|-------------|---------|
| A: Problem Definition | 10 | 2 NLP problems × 5 + 2 text domains × 5 | 20 |
| B: Dataset Selection | 20 | 2 existing datasets (10) + 1 lexicon (10) | 20 |
| C: Modelling | 30 | Baseline + 3 pretrained + fine-tune + multi-task | 65+ |
| D: Evaluation | 20 | Quantitative + qualitative + CLI + demo | 30 |
| **Total** | **80** | | **135+** |

---

## 2. Background Theory

### 2.1 Text Classification

Text classification is the task of assigning a **category label** to a piece of text.
It is one of the most fundamental problems in NLP.

The pipeline is always:
```
Raw Text → Feature Extraction → Classification Model → Predicted Label
```

The key question is: **how do we turn text into numbers?** Different approaches:

| Approach | How it works | Era |
|----------|-------------|-----|
| Bag of Words | Count word frequencies | 1990s |
| TF-IDF | Weighted word frequencies | 2000s |
| Word Embeddings | Dense vectors per word (Word2Vec, GloVe) | 2013+ |
| Transformers | Contextual embeddings (BERT, GPT) | 2018+ |

### 2.2 TF-IDF (Term Frequency–Inverse Document Frequency)

TF-IDF is a numerical statistic that reflects how important a word is to a
document within a collection (corpus).

**Term Frequency (TF)**: How often a word appears in a document.
```
TF(word, doc) = count(word in doc) / total_words(doc)
```

**Inverse Document Frequency (IDF)**: How rare a word is across all documents.
```
IDF(word) = log(total_documents / documents_containing(word))
```

**TF-IDF = TF × IDF**

Intuition: A word that appears frequently in one document but rarely in others
is probably important for that document. Common words like "the" get low scores
(high TF but low IDF), while distinctive words get high scores.

**In our code** (`src/baseline.py`):
```python
TfidfVectorizer(
    max_features=50_000,      # keep top 50K features to limit memory
    ngram_range=(1, 2),       # use unigrams AND bigrams
    sublinear_tf=True,        # use log(1 + TF) instead of raw TF
    strip_accents="unicode",  # normalize accented characters
)
```

- `ngram_range=(1, 2)` means we capture both single words ("inflation") and
  two-word phrases ("rate hike"). Bigrams often carry more meaning than unigrams.
- `sublinear_tf=True` applies logarithmic scaling: `1 + log(TF)`. This prevents
  very frequent words from dominating. Without this, a word appearing 100 times
  would be weighted 100× more than one appearing once; with log scaling, it's
  only ~5.6× more.

### 2.3 Logistic Regression

Logistic Regression is a linear classifier that models the probability of each class.

For binary classification:
```
P(class=1 | x) = sigmoid(w·x + b) = 1 / (1 + exp(-(w·x + b)))
```

For multi-class (our case with 3 classes), it uses the **softmax** function:
```
P(class=k | x) = exp(w_k·x + b_k) / Σ_j exp(w_j·x + b_j)
```

The model learns weights `w` that tell it which features (TF-IDF values) are
most predictive of each class.

**In our code**:
```python
LogisticRegression(
    max_iter=1000,             # allow up to 1000 optimization iterations
    class_weight="balanced",   # upweight minority classes
    random_state=SEED,         # reproducibility
    solver="lbfgs",            # L-BFGS optimization algorithm
)
```

- `class_weight="balanced"` is crucial. Without it, the model would be biased
  toward the majority class (neutral). With balanced weights, each class is
  weighted inversely proportional to its frequency:
  `weight_k = n_samples / (n_classes × n_samples_in_class_k)`

### 2.4 Transformers and BERT

**The Transformer Architecture** (Vaswani et al., 2017) is the foundation of
modern NLP. Its key innovation is the **self-attention mechanism**.

**Self-Attention**: For each word in a sentence, the model computes how much
it should "attend to" every other word. This captures long-range dependencies
that previous architectures (RNNs, LSTMs) struggled with.

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

Where Q (query), K (key), V (value) are linear projections of the input.
The `√d_k` scaling prevents the dot products from growing too large.

**BERT** (Bidirectional Encoder Representations from Transformers, Devlin et al., 2019)
is a Transformer trained on two objectives:
1. **Masked Language Modeling (MLM)**: Randomly mask 15% of tokens, predict them.
   E.g., "The cat [MASK] on the mat" → predict "sat".
2. **Next Sentence Prediction (NSP)**: Given two sentences, predict if the second
   follows the first in the original text.

BERT is **bidirectional** — it reads text in both directions simultaneously,
unlike GPT which reads left-to-right only.

**The [CLS] Token**: BERT prepends a special `[CLS]` token to every input.
After processing through all layers, the hidden state of `[CLS]` serves as a
**sentence-level representation** — a single 768-dimensional vector that
captures the meaning of the entire input. This is what we use for classification.

### 2.5 FinBERT

**FinBERT** (ProsusAI/finbert) is BERT further pre-trained on financial text:
- Started from the standard BERT-base model (110M parameters)
- Further pre-trained on a large corpus of financial news, earnings calls,
  and analyst reports
- Then fine-tuned for financial sentiment analysis (positive/negative/neutral)

**Why FinBERT matters**: General-purpose BERT doesn't understand financial jargon
well. For example, "the stock was volatile" is negative in finance but neutral
in general English. FinBERT learns these domain-specific meanings.

### 2.6 Zero-Shot vs Few-Shot vs Fine-Tuning

These are different ways to use a pre-trained model:

**Zero-Shot**: Use the model as-is, with no training on the target task.
FinBERT was already trained for financial sentiment, so we can use its
existing classification head directly on Financial PhraseBank.

**Few-Shot (k=16)**: Give the model a tiny amount of labeled data (16 examples
per class = 48 total). We freeze the model's weights and train only a small
linear classifier on top of its embeddings. This tests the quality of the
model's learned representations.

**Fine-Tuning**: Update ALL model weights on the full training set. This is
the most powerful approach but requires more data and compute. The model adapts
its internal representations specifically for the target task.

### 2.7 Multi-Task Learning

Multi-task learning trains one model on multiple tasks simultaneously.

**Architecture**:
```
Input Text → Shared FinBERT Encoder → [CLS] embedding
                                         ├─→ Stance Head   → hawkish/dovish/neutral
                                         └─→ Sentiment Head → positive/negative/neutral
```

**Why it helps**:
1. **Shared representations**: Both tasks involve understanding financial language.
   Training on sentiment helps the model understand stance, and vice versa.
2. **Regularization**: Learning multiple tasks acts as a form of regularization,
   preventing overfitting to any single task.
3. **Data efficiency**: The stance dataset is small (~1700 train). By also training
   on sentiment data, the shared encoder sees more financial text.

**Training procedure**: We alternate batches from the two datasets. One batch
trains on stance (updating the shared encoder + stance head), then one batch
trains on sentiment (updating the shared encoder + sentiment head).

### 2.8 Weighted Cross-Entropy Loss

Standard cross-entropy loss treats all classes equally:
```
L = -Σ y_true × log(y_pred)
```

When classes are imbalanced (e.g., FOMC has ~2× more neutral than hawkish),
the model learns to predict the majority class. **Weighted cross-entropy**
assigns higher loss to minority classes:
```
L = -Σ w_k × y_true × log(y_pred)
```

Where `w_k = N / (C × n_k)`:
- N = total samples
- C = number of classes
- n_k = samples in class k

For FOMC: dovish weight ≈ 1.27, hawkish weight ≈ 1.37, neutral weight ≈ 0.68.
This means misclassifying a hawkish sentence costs ~2× more than misclassifying
a neutral one, forcing the model to pay more attention to minority classes.

---

## 3. Environment & Libraries

### 3.1 Python Version

We use **Python 3.12** (not the system default 3.14) because PyTorch does not
yet support Python 3.14. The virtual environment is created with:
```bash
/opt/homebrew/bin/python3.12 -m venv venv
```

### 3.2 Hardware

**Apple M3 Max** with MPS (Metal Performance Shaders) backend. MPS is Apple's
GPU compute framework for PyTorch, similar to NVIDIA's CUDA but for Apple Silicon.

```python
# In config.py:
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
```

MPS provides significant speedup over CPU for matrix operations (the core of
neural network training). FinBERT fine-tuning runs at ~1.5-2 batches/second
on MPS vs ~0.3 on CPU.

### 3.3 Library Breakdown

**`torch` (PyTorch 2.10+)**
The deep learning framework. Provides:
- Tensor operations (like NumPy but with GPU support)
- Automatic differentiation (computes gradients for backpropagation)
- Neural network modules (`nn.Module`, `nn.Linear`, etc.)
- Optimizers (`AdamW`, `SGD`)

**`transformers` (HuggingFace Transformers 5.3+)**
The standard library for pre-trained NLP models. Provides:
- `AutoTokenizer`: Converts text to token IDs that models understand
- `AutoModel`: Loads pre-trained model weights
- `AutoModelForSequenceClassification`: Model with a classification head
- `pipeline()`: High-level inference API
- `get_linear_schedule_with_warmup()`: Learning rate scheduler

**`datasets` (HuggingFace Datasets 4.7+)**
Efficient dataset loading and processing:
- `load_dataset()`: Downloads datasets from HuggingFace Hub
- `Dataset`, `DatasetDict`: Efficient columnar storage (Apache Arrow backend)
- Handles train/test splits, shuffling, batching

**`scikit-learn` (1.8+)**
Classical machine learning library:
- `TfidfVectorizer`: Computes TF-IDF features
- `LogisticRegression`: Linear classification
- `train_test_split()`: Stratified data splitting
- `classification_report()`, `confusion_matrix()`: Evaluation metrics

**`pandas` (3.0+)**
Data manipulation library. We use it for:
- Converting HuggingFace datasets to DataFrames for easier manipulation
- Groupby operations for sampling (few-shot)
- Error analysis (sorting misclassified examples)

**`numpy` (2.4+)**
Numerical computing. Used for array operations, especially in lexicon
feature extraction.

**`matplotlib` + `seaborn`**
Visualization libraries. We use them for confusion matrix heatmaps.
`seaborn.heatmap()` produces the annotated confusion matrices saved in `results/`.

**`gradio` (4.19+)**
Web UI framework for ML demos. Creates interactive web interfaces with
minimal code. Our demo has a text input, two label outputs (stance + sentiment),
and example sentences.

**`accelerate` (0.27+)**
HuggingFace library for distributed training. Required by `transformers` for
model loading, even if we only use a single device.

**`tqdm` (4.66+)**
Progress bar library. Wraps iterators to show training progress:
```
Epoch 1/5: 45%|████▌     | 25/55 [00:13<00:16, 1.83it/s]
```

### 3.4 Key Configuration Parameters (config.py)

```python
MAX_SEQ_LENGTH = 128    # Max tokens per input (BERT max is 512, but our
                        # sentences are short — 128 saves memory and time)
BATCH_SIZE = 32         # Number of examples processed together on GPU
LEARNING_RATE = 2e-5    # Standard for BERT fine-tuning (from original paper)
WEIGHT_DECAY = 0.01     # L2 regularization to prevent overfitting
FINETUNE_EPOCHS = 5     # Training passes over the full dataset
MULTITASK_EPOCHS = 8    # More epochs for multi-task (two datasets)
FEW_SHOT_K = 16         # Examples per class for few-shot learning
WARMUP_RATIO = 0.1      # Warm up learning rate for first 10% of training
SEED = 42               # Random seed for reproducibility
TEST_SIZE = 0.2         # 20% of data for testing
VAL_SIZE = 0.1          # 10% of data for validation
```

**Why these values?**
- `LEARNING_RATE = 2e-5`: This is the standard from the original BERT paper
  (Devlin et al., 2019). Too high (e.g., 1e-3) destroys pre-trained weights;
  too low (e.g., 1e-6) learns too slowly.
- `WARMUP_RATIO = 0.1`: Gradually increases the learning rate from 0 to 2e-5
  over the first 10% of training steps. This prevents the model from making
  large, destructive updates early in training when gradients are noisy.
- `MAX_SEQ_LENGTH = 128`: Our sentences are typically 10-50 words (≈15-70 tokens).
  128 gives headroom without wasting memory on padding.

---

## 4. Project Architecture

### 4.1 File-by-File Walkthrough

```
Project/
├── config.py                  # ALL settings in one place
├── src/
│   ├── __init__.py            # Makes src/ a Python package
│   ├── data_loader.py         # Downloads and splits datasets
│   ├── baseline.py            # TF-IDF + Logistic Regression
│   ├── lexicon.py             # Loughran-McDonald lexicon approach
│   ├── pretrained_eval.py     # Zero-shot + few-shot experiments
│   ├── finetune.py            # Single-task FinBERT fine-tuning
│   ├── multitask.py           # Multi-task model architecture + training
│   └── evaluate.py            # Metrics, plots, error analysis
├── run_experiments.py         # Orchestrates all experiments (Steps 1-5)
├── cli.py                     # Command-line prediction tool
├── demo.py                    # Gradio web interface
├── requirements.txt           # Python dependencies
├── models/                    # Saved model weights after training
│   ├── finbert_stance/        # Fine-tuned stance model
│   ├── finbert_sentiment/     # Fine-tuned sentiment model
│   └── multitask_finbert/     # Multi-task model
└── results/                   # JSON metrics + PNG confusion matrices
```

### 4.2 Data Flow

```
run_experiments.py
    │
    ├── Step 1: data_loader.py → Downloads FOMC + FPB datasets
    │                            Splits into train/val/test
    │
    ├── Step 2: baseline.py    → TF-IDF + LR on both tasks
    │           lexicon.py     → LM lexicon rule-based + TF-IDF+lexicon
    │
    ├── Step 3: pretrained_eval.py → FinBERT zero-shot
    │                                Few-shot (FinBERT, BERT, RoBERTa)
    │
    ├── Step 4: finetune.py    → FinBERT fine-tuned on FOMC
    │                            FinBERT fine-tuned on FPB
    │
    └── Step 5: multitask.py   → Joint training on both datasets
```

Each step uses `evaluate.py` to compute metrics and save results.

### 4.3 Why This Structure?

- **Separation of concerns**: Each model type is in its own file. You can
  understand `baseline.py` without reading `multitask.py`.
- **Central config**: All hyperparameters in `config.py` means you change
  settings in one place, not scattered across files.
- **Step-based runner**: `run_experiments.py --step N` lets you re-run
  individual experiments without re-running everything.

---

## 5. Datasets In Depth

### 5.1 FOMC Hawkish-Dovish Dataset

**Source**: Georgia Tech Financial Technology Lab (gtfintechlab)
**Paper**: "Trillion Dollar Words" (Shah et al., ACL 2023)
**HuggingFace ID**: `gtfintechlab/fomc_communication`

**What it contains**: Sentences extracted from FOMC meeting minutes and
statements (2000-2023). Each sentence is labeled by financial experts.

**Label distribution** (after our split):
```
Train: 1736 samples (dovish: 455, hawkish: 424, neutral: 857)
Val:    248 samples (dovish:  65, hawkish:  61, neutral: 122)
Test:   496 samples (dovish: 130, hawkish: 121, neutral: 245)
```

Note the **class imbalance**: neutral (~49%) is nearly twice as common as
hawkish (~24%) or dovish (~26%). This is why we use weighted cross-entropy loss.

**Example sentences**:
- Dovish: *"Low readings on overall and core consumer price inflation in recent
  months, as well as the weakened economic outlook..."*
- Hawkish: *"Our new statement explicitly acknowledges the challenges posed by
  the proximity of interest rates to..."*
- Neutral: *"Broad equity price indexes fell sharply over the intermeeting period on net."*

### 5.2 Financial PhraseBank

**Source**: Malo et al. (2014), "Good Debt or Bad Debt"
**HuggingFace ID**: `gtfintechlab/financial_phrasebank_sentences_allagree`

**What it contains**: 2,264 sentences from English-language financial news,
annotated by 16 finance professionals. The `sentences_allagree` subset means
ALL annotators agreed on the label — highest quality annotations.

**Why "allagree"?** The full dataset has 4,846 sentences with varying agreement
levels (50%, 66%, 75%, 100%). Using 100% agreement gives us the cleanest,
least ambiguous labels. The trade-off is fewer samples.

**Label distribution** (after our split):
```
Train: 1584 samples (negative: 212, neutral: 973, positive: 399)
Val:    227 samples (negative:  30, neutral: 140, positive:  57)
Test:   453 samples (negative:  61, neutral: 278, positive: 114)
```

Again, **class imbalance**: neutral dominates (~61%), negative is the minority (~13%).

### 5.3 Data Splitting Strategy

We use **stratified splitting** to ensure each split has the same class
proportions as the full dataset:

```python
train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
```

The splits are: **70% train / 10% validation / 20% test**.

- **Train**: Model learns from this data
- **Validation**: Used to pick the best model (early stopping)
- **Test**: Final evaluation — NEVER used during training

The `random_state=42` ensures the same split every time we run the code.

### 5.4 HuggingFace Datasets Library

Originally, these datasets used Python loading scripts on HuggingFace. But
the `datasets` library (v4.0+) dropped support for loading scripts in favor
of Parquet files. This is why we use the `gtfintechlab/` mirror versions
instead of the original `takala/financial_phrasebank` — they have been
converted to Parquet format. The data is identical; only the storage format changed.

```python
# This NO LONGER works (old loading script):
load_dataset("takala/financial_phrasebank", "sentences_allagree")

# This WORKS (Parquet version):
load_dataset("gtfintechlab/financial_phrasebank_sentences_allagree", "5768")
```

The `"5768"` is a config name required by this particular dataset version
(it refers to a specific data split configuration).

---

## 6. The Loughran-McDonald Lexicon

### 6.1 What Is a Lexicon?

A **lexicon** (or dictionary) in NLP is a predefined list of words associated
with specific categories. Unlike machine learning approaches that learn from
data, lexicon-based methods use human-curated knowledge.

### 6.2 Why Loughran-McDonald?

Standard sentiment lexicons (like Harvard General Inquirer or VADER) perform
poorly on financial text because many words have **different meanings** in finance:

| Word | General Sentiment | Financial Sentiment |
|------|-------------------|---------------------|
| "liability" | Negative | Neutral (accounting term) |
| "tax" | Negative | Neutral (standard business) |
| "capital" | Neutral | Positive (sign of strength) |
| "crude" | Negative | Neutral (crude oil) |

Loughran and McDonald (2011) created a lexicon specifically for financial text
by analyzing 50,000+ 10-K filings. Their word lists are the gold standard
in financial NLP research.

### 6.3 Our Word Categories

We use six word lists in `src/lexicon.py`:

1. **LM_POSITIVE** (~110 words): Words indicating good outcomes —
   "achieve", "benefit", "profit", "recovery", "strength"

2. **LM_NEGATIVE** (~230 words): Words indicating bad outcomes —
   "bankruptcy", "decline", "default", "loss", "recession"

3. **LM_UNCERTAINTY** (~90 words): Words indicating vagueness —
   "approximate", "contingent", "maybe", "uncertain", "variable"

4. **HAWKISH_WORDS** (~40 words): Monetary policy tightening —
   "hike", "tighten", "inflation", "restrictive", "tapering"

5. **DOVISH_WORDS** (~50 words): Monetary policy easing —
   "cut", "ease", "accommodate", "stimulus", "patient"

The hawkish/dovish lists are custom additions specific to our stance
classification task, curated from central banking literature.

### 6.4 Feature Extraction

For each text, we compute 8 numerical features:

```python
[positive_count, negative_count, uncertainty_count,
 hawkish_count, dovish_count,
 net_sentiment,   # (pos - neg) / total_words
 net_stance,      # (hawkish - dovish) / total_words
 total_words]
```

**Normalization** by total words is important. A 50-word sentence with 3 negative
words is more negative than a 200-word sentence with 3 negative words.

### 6.5 Two Uses of the Lexicon

**Rule-based classifier** (no training needed):
- For sentiment: if net_sentiment > 0.02 → positive; < -0.02 → negative; else neutral
- For stance: if hawkish_count > dovish_count → hawkish; etc.

**TF-IDF + Lexicon features** (trained):
- Concatenate TF-IDF features with the 8 lexicon features
- Train Logistic Regression on the combined feature set
- This lets the model use both word-level patterns (TF-IDF) and
  domain knowledge (lexicon)

### 6.6 Results

The pure rule-based lexicon performed worst (41.5% stance, 69.3% sentiment),
which is expected — it has no learning capability. But it demonstrates that
the lexicon contains **meaningful signal**.

The TF-IDF + Lexicon combination showed marginal improvement over pure TF-IDF
for stance (61.1% vs 60.9%) but slightly lower for sentiment (85.4% vs 87.2%).
This suggests the TF-IDF already captures most of the lexicon's signal.

---

## 7. Baseline Model: TF-IDF + Logistic Regression

### 7.1 Purpose

Every ML project needs a **baseline** — a simple model that sets the floor.
If a complex model doesn't beat the baseline, it's not worth the complexity.

TF-IDF + Logistic Regression is the standard non-neural baseline for text
classification. It's fast, interpretable, and often surprisingly competitive.

### 7.2 Pipeline Details (`src/baseline.py`)

```python
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(...)),
    ("clf", LogisticRegression(...)),
])
```

Scikit-learn's `Pipeline` chains the vectorizer and classifier, ensuring
the same transformations are applied consistently to train and test data.

**Step 1: TF-IDF Vectorization**
```
"The Fed raised rates" → [0, 0, 0.42, 0, 0.71, ..., 0, 0.38]
                          (50,000-dimensional sparse vector)
```

**Step 2: Logistic Regression**
```
[0, 0, 0.42, ..., 0.38] → softmax(W·x + b) → [0.1, 0.7, 0.2]
                                                 (dovish, hawkish, neutral)
```

### 7.3 Results

| Task | Accuracy | Macro-F1 |
|------|----------|----------|
| Stance | 0.6089 | 0.5873 |
| Sentiment | 0.8720 | 0.8232 |

The baseline is surprisingly strong for sentiment (87.2%!) because financial
sentiment often correlates with specific words ("profit" → positive, "loss"
→ negative). Stance is harder because the same words can appear in both
hawkish and dovish contexts — it depends on the **full sentence meaning**.

---

## 8. Pre-trained Transformer Models

### 8.1 Models Compared

| Model | Parameters | Pre-training Data | Financial? |
|-------|------------|-------------------|------------|
| FinBERT | 110M | Financial news + filings | Yes |
| BERT-base-uncased | 110M | Wikipedia + BookCorpus | No |
| RoBERTa-base | 125M | Web text (80GB) | No |

**BERT-base-uncased**: "uncased" means it lowercases all text before tokenizing.
"The" and "the" become the same token. This helps with consistency but loses
information (e.g., proper nouns).

**RoBERTa-base** (Robustly Optimized BERT): Same architecture as BERT but
trained longer, on more data, with better hyperparameters. Removed NSP objective.
Generally performs better than BERT-base.

### 8.2 Zero-Shot: FinBERT Native Head

FinBERT was fine-tuned for sentiment classification (positive/negative/neutral).
We can use it directly — no training needed.

```python
clf = pipeline("text-classification", model="ProsusAI/finbert")
result = clf("Revenue grew 15%")
# → [{"label": "positive", "score": 0.97}]
```

**On sentiment (Financial PhraseBank)**: 97.4% accuracy — essentially a perfect
match because FinBERT was trained for exactly this type of task.

**On stance (FOMC)**: 49.8% accuracy — poor, because sentiment ≠ stance.
A sentence can be negative in sentiment but hawkish in stance
(e.g., "inflation remains dangerously high" — negative news, but hawkish
because it suggests rate hikes).

### 8.3 Few-Shot: Linear Probe

For few-shot evaluation, we:

1. **Freeze** the pre-trained model (no weight updates)
2. Pass all texts through the model to get [CLS] embeddings (768-dim vectors)
3. Train a tiny linear classifier on just 48 examples (16 per class)

```python
class FewShotClassifier(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),                        # prevent overfitting
            nn.Linear(hidden_size, num_labels),     # 768 → 3
        )
```

This tests the **quality of the model's representations**. A model with good
financial understanding should produce embeddings where same-class texts
cluster together, making classification easy even with very few examples.

### 8.4 Encoding Process

```python
def _encode_texts(tokenizer, model, texts, device):
    for batch in batches:
        inputs = tokenizer(batch, padding=True, truncation=True, ...)
        outputs = model(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS] token
```

- `tokenizer(...)` converts text to token IDs + attention masks
- `model(**inputs)` runs the forward pass through all 12 Transformer layers
- `outputs.last_hidden_state[:, 0, :]` extracts the [CLS] embedding
  (first token, all 768 dimensions)

### 8.5 Results Comparison

| Model | Stance F1 | Sentiment F1 |
|-------|-----------|--------------|
| FinBERT (few-shot) | 0.4534 | **0.9670** |
| BERT-base (few-shot) | 0.3744 | 0.6500 |
| RoBERTa-base (few-shot) | 0.3600 | 0.6722 |

**Key insight**: FinBERT destroys the general-purpose models on financial tasks,
even with just 16 examples per class. This proves that **domain-specific
pre-training is highly valuable**. BERT and RoBERTa produce generic embeddings
that don't capture financial meaning well.

---

## 9. Fine-tuning FinBERT

### 9.1 What Changes During Fine-tuning

Unlike few-shot (frozen model + small classifier), fine-tuning updates
**all 110 million parameters** of FinBERT to optimize for the specific task.

The model starts from pre-trained weights and gradually adapts:
- **Early layers** (embeddings, first few Transformer blocks) learn general
  language features and change little
- **Later layers** increasingly specialize for the target task
- **Classification head** (new, randomly initialized) learns the task mapping

### 9.2 Training Loop (`src/finetune.py`)

```python
for epoch in range(FINETUNE_EPOCHS):
    model.train()
    for batch in train_loader:
        # Forward pass
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs.logits, labels)

        # Backward pass
        optimizer.zero_grad()       # clear previous gradients
        loss.backward()             # compute new gradients
        clip_grad_norm_(model.parameters(), 1.0)  # prevent exploding gradients
        optimizer.step()            # update weights
        scheduler.step()            # update learning rate
```

**Gradient clipping** (`clip_grad_norm_(..., 1.0)`) prevents the gradients
from becoming too large, which can destabilize training. It rescales gradients
so their total norm doesn't exceed 1.0.

### 9.3 Learning Rate Schedule

We use **linear warmup + linear decay**:

```
Learning Rate
    ^
2e-5|           /‾‾‾‾‾‾‾‾‾‾‾‾‾‾\
    |          /                  \
    |         /                    \
    |        /                      \
  0 |_______/________________________\___→ Training Steps
    0    10%                         100%
         ↑ warmup ends
```

**Why warmup?** At the start of training, the classification head has random
weights. If the learning rate is immediately high, the gradients from the random
head propagate back and corrupt the carefully pre-trained encoder weights.
Warmup lets the head stabilize before the encoder starts changing significantly.

### 9.4 Best Model Selection

We track validation F1 after each epoch and save the best model:

```python
if val_metrics["macro_f1"] > best_val_f1:
    best_val_f1 = val_metrics["macro_f1"]
    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
```

This prevents **overfitting**: the model might achieve great training performance
in later epochs but worse test performance. By saving the best validation
checkpoint, we get the most generalizable model.

### 9.5 The `TextClassificationDataset` Class

```python
class TextClassificationDataset(TorchDataset):
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
```

This wraps our data for PyTorch's `DataLoader`. For each example:
- **input_ids**: Token IDs (integers). E.g., "The Fed" → [101, 1996, 5765, 102]
- **attention_mask**: Binary mask. 1 = real token, 0 = padding.
  E.g., [1, 1, 1, 1, 0, 0, ..., 0] for a short sentence with padding
- **labels**: Integer class label (0, 1, or 2)

`padding="max_length"` pads all sequences to 128 tokens so they can be batched
together as a single tensor. `truncation=True` cuts sequences longer than 128.

### 9.6 Results

| Task | Accuracy | Macro-F1 |
|------|----------|----------|
| Stance | 0.6371 | 0.6194 |
| Sentiment | 0.9669 | 0.9459 |

Fine-tuning improves over the baseline by:
- Stance: +2.8% accuracy, +3.2% F1 (modest — stance is hard)
- Sentiment: +9.5% accuracy, +12.3% F1 (dramatic improvement)

---

## 10. Multi-task Learning

### 10.1 Model Architecture (`src/multitask.py`)

```python
class MultiTaskFinBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("ProsusAI/finbert")  # shared
        self.dropout = nn.Dropout(0.1)
        self.stance_head = nn.Linear(768, 3)     # task-specific
        self.sentiment_head = nn.Linear(768, 3)  # task-specific

    def forward(self, input_ids, attention_mask, task="stance"):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # [CLS]
        pooled = self.dropout(pooled)

        if task == "stance":
            return self.stance_head(pooled)
        else:
            return self.sentiment_head(pooled)
```

The `task` parameter acts as a switch: same encoder, different output heads.

### 10.2 Alternating Batch Training

```python
while not (stance_done and sentiment_done):
    # Stance batch
    if not stance_done:
        batch = next(stance_iter)
        loss = _train_step(model, batch, stance_criterion, ..., task="stance")

    # Sentiment batch
    if not sentiment_done:
        batch = next(sentiment_iter)
        loss = _train_step(model, batch, sentiment_criterion, ..., task="sentiment")
```

We interleave batches: stance, sentiment, stance, sentiment, ...

This ensures the shared encoder continuously learns from both types of
financial text. If we trained all stance batches first, then all sentiment,
the model might "forget" stance patterns while learning sentiment
(this is called **catastrophic forgetting**).

### 10.3 Separate Loss Functions

```python
# Weighted loss for stance (addresses class imbalance)
stance_criterion = nn.CrossEntropyLoss(
    weight=torch.tensor([1.272, 1.365, 0.675])
)
# Standard loss for sentiment (more balanced classes)
sentiment_criterion = nn.CrossEntropyLoss()
```

We only use weighted loss for stance because the FOMC dataset has significant
class imbalance. Financial PhraseBank is more balanced.

### 10.4 Model Selection

For multi-task, we save the model with the best **average** F1 across both tasks:

```python
avg_f1 = (val_stance_f1 + val_sentiment_f1) / 2
if avg_f1 > best_avg_f1:
    best_model_state = ...
```

This prevents optimizing for one task at the expense of the other.

### 10.5 Results

| Task | Single-task F1 | Multi-task F1 | Improvement |
|------|---------------|---------------|-------------|
| Stance | 0.6194 | **0.6478** | +0.0284 |
| Sentiment | 0.9459 | **0.9666** | +0.0207 |

**Multi-task improves both tasks.** The shared encoder benefits from seeing
more diverse financial text. The stance task especially benefits because it
has fewer training samples — the sentiment data provides helpful regularization.

### 10.6 Error Analysis

The multi-task model makes 169 errors on stance:
```
neutral → hawkish: 49    (most common — neutral sentences with slight hawkish hints)
neutral → dovish:  39    (neutral sentences with slight dovish hints)
dovish → hawkish:  26    (confused dovish/hawkish — subtle language differences)
```

These errors are understandable. Neutral FOMC sentences often contain
**mixed signals** — both hawkish and dovish elements — making classification
genuinely ambiguous even for human experts.

On sentiment, only 9 errors total — 4 of which are "positive → negative"
(e.g., a positive sentence about one company mentioning a competitor's loss).

---

## 11. Evaluation Methodology

### 11.1 Why Macro-F1?

We use **Macro-F1** as our primary metric rather than accuracy because
our datasets have class imbalance.

**Accuracy** can be misleading: if 60% of FOMC sentences are neutral,
a model that predicts "neutral" for everything gets 60% accuracy but is useless.

**F1 Score** balances precision and recall:
```
Precision = True Positives / (True Positives + False Positives)
Recall    = True Positives / (True Positives + False Negatives)
F1        = 2 × Precision × Recall / (Precision + Recall)
```

**Macro-F1** averages F1 across all classes equally:
```
Macro-F1 = (F1_dovish + F1_hawkish + F1_neutral) / 3
```

This gives equal weight to all classes, regardless of their frequency.
A model must perform well on ALL classes to get a high Macro-F1.

### 11.2 Confusion Matrix

A confusion matrix shows where the model makes mistakes. Each cell (i, j)
contains the number of examples with true label i predicted as label j.

```
                 Predicted
              Dov  Haw  Neu
True  Dov  [ 85    6   39 ]    ← 85 correct, 6 confused with hawkish
      Haw  [  5   85   31 ]    ← 31 predicted as neutral
      Neu  [ 55   70  120 ]    ← most errors: neutral → dovish/hawkish
```

The diagonal shows correct predictions. Off-diagonal cells are errors.

We generate these as heatmaps using `seaborn.heatmap()` and save them as
PNG files in `results/`.

### 11.3 Error Analysis (`evaluate.py`)

```python
def error_analysis(texts, y_true, y_pred, label_names, top_n=20):
    errors = []
    for text, true, pred in zip(texts, y_true, y_pred):
        if true != pred:
            errors.append({
                "text": text,
                "true_label": label_names[true],
                "pred_label": label_names[pred],
                "error_type": f"{label_names[true]} → {label_names[pred]}",
            })
```

This function collects all misclassified examples and categorizes them by
error type (e.g., "dovish → hawkish"). Examining these errors reveals:
- **Ambiguous sentences** where even humans might disagree
- **Domain mismatch** where the model lacks specific knowledge
- **Systematic biases** (e.g., always predicting neutral for long sentences)

---

## 12. CLI and Demo

### 12.1 CLI (`cli.py`)

The CLI provides three modes:

**Interactive mode** (`python cli.py`):
```
>>> The Fed raised rates by 75 basis points
  STANCE: hawkish (0.9000)
  SENTIMENT: positive (0.9245)
```

**Single sentence** (`python cli.py --text "..."`)

**File mode** (`python cli.py --file input.txt`): Processes each line.

Internally, it:
1. Loads the multi-task model from `models/multitask_finbert/`
2. Tokenizes the input text
3. Runs it through the shared encoder
4. Gets predictions from both task heads
5. Applies softmax to get confidence scores

```python
logits = model(input_ids, attention_mask, task="stance")
probs = F.softmax(logits, dim=-1)  # convert logits to probabilities
pred_idx = probs.argmax().item()   # pick the highest probability class
```

### 12.2 Gradio Demo (`demo.py`)

Gradio creates a web interface with minimal code:

```python
with gr.Blocks() as demo:
    text_input = gr.Textbox(label="Financial Text")
    stance_output = gr.Label(label="Stance", num_top_classes=3)
    sentiment_output = gr.Label(label="Sentiment", num_top_classes=3)

    submit_btn.click(fn=classify, inputs=text_input,
                     outputs=[stance_output, sentiment_output])
```

When the user types text and clicks "Classify", Gradio calls our `classify()`
function and displays the results as labeled probability bars.

The demo includes 8 example sentences that users can click to try.

---

## 13. Bugs Encountered and How They Were Fixed

### Bug 1: `trust_remote_code` No Longer Supported

**Error**:
```
RuntimeError: Dataset scripts are no longer supported, but found financial_phrasebank.py
```

**Cause**: HuggingFace `datasets` v4.0+ removed support for Python loading
scripts. The original `takala/financial_phrasebank` dataset used a custom
loading script.

**Fix**: Switched to Parquet-based mirrors:
```python
# Before (broken):
load_dataset("takala/financial_phrasebank", "sentences_allagree", trust_remote_code=True)

# After (working):
load_dataset("gtfintechlab/financial_phrasebank_sentences_allagree", "5768")
```

**Lesson**: Always use the latest dataset format. HuggingFace is moving
everything to Parquet for security (loading scripts can execute arbitrary code).

### Bug 2: `multi_class` Parameter Removed from LogisticRegression

**Error**:
```
TypeError: LogisticRegression.__init__() got an unexpected keyword argument 'multi_class'
```

**Cause**: scikit-learn 1.8+ removed the `multi_class` parameter. The solver
now automatically determines the multi-class strategy.

**Fix**: Removed the parameter:
```python
# Before:
LogisticRegression(multi_class="multinomial", solver="lbfgs")

# After:
LogisticRegression(solver="lbfgs")
```

**Lesson**: Newer library versions can deprecate and remove parameters.
Always check release notes when using latest versions.

### Bug 3: Few-Shot Sampling Drops the `label` Column

**Error**:
```
ValueError: Column 'label' doesn't exist.
```

**Cause**: The `groupby("label").apply(lambda x: x.sample(...))` operation
in pandas 3.0+ consumes the groupby column, removing it from the result.
The `include_groups=False` parameter (tried first) also drops it.

**Fix**: Replaced groupby with explicit loop:
```python
# Before (broken in pandas 3.0):
df.groupby("label").apply(lambda x: x.sample(n=k, random_state=SEED))

# After (works in all pandas versions):
pieces = []
for label_val in sorted(df["label"].unique()):
    subset = df[df["label"] == label_val]
    pieces.append(subset.sample(n=min(k, len(subset)), random_state=SEED))
sampled = pd.concat(pieces, ignore_index=True)
```

**Lesson**: `groupby().apply()` behavior varies across pandas versions.
Simple loops are more reliable and easier to debug.

### Bug 4: MPS Device Issues with HuggingFace Pipeline

**Problem**: The HuggingFace `pipeline()` function sometimes fails on MPS
(Apple Silicon GPU) due to unsupported operations.

**Fix**: Use `device=-1` (CPU) for pipeline-based inference, and MPS only
for direct PyTorch training:
```python
clf = pipeline("text-classification", model=FINBERT_MODEL, device=-1)
```

This is a minor performance trade-off — pipeline inference on CPU is fast
enough for small test sets.

### Bug 5: Python 3.14 Incompatibility with PyTorch

**Problem**: The system Python was 3.14.2, but PyTorch doesn't support Python 3.14.

**Fix**: Created a venv with Python 3.12:
```bash
/opt/homebrew/bin/python3.12 -m venv venv
```

**Lesson**: Always check framework compatibility before starting. Use `venv`
to isolate the project from the system Python.

---

## 14. Results Analysis

### 14.1 Complete Results Table

| Model | Stance Acc | Stance F1 | Sent. Acc | Sent. F1 |
|-------|-----------|-----------|-----------|----------|
| LM Lexicon (rules) | 0.4153 | 0.3885 | 0.6932 | 0.5315 |
| TF-IDF + LR | 0.6089 | 0.5873 | 0.8720 | 0.8232 |
| TF-IDF + LM Lexicon | 0.6109 | 0.5863 | 0.8543 | 0.8050 |
| FinBERT (zero-shot) | 0.4980 | 0.4874 | 0.9735 | 0.9650 |
| FinBERT (few-shot k=16) | 0.4859 | 0.4534 | 0.9779 | 0.9670 |
| BERT-base (few-shot k=16) | 0.3851 | 0.3744 | 0.7417 | 0.6500 |
| RoBERTa-base (few-shot k=16) | 0.3730 | 0.3600 | 0.7682 | 0.6722 |
| FinBERT (fine-tuned) | 0.6371 | 0.6194 | 0.9669 | 0.9459 |
| **Multi-task FinBERT** | **0.6593** | **0.6478** | **0.9801** | **0.9666** |

### 14.2 Key Observations

**1. Domain-specific pre-training is the single most important factor.**
FinBERT (few-shot, only 48 examples) achieves 96.7% F1 on sentiment,
beating BERT-base (65.0%) and RoBERTa-base (67.2%) with the same data.
This is a 30+ percentage point gap from domain pre-training alone.

**2. Stance is fundamentally harder than sentiment.**
Even our best model achieves only 64.8% F1 on stance vs 96.7% on sentiment.
Stance requires understanding **implied monetary policy positions** — subtle
reasoning that goes beyond surface-level word meaning.

**3. Multi-task learning helps both tasks.**
The multi-task model beats single-task fine-tuning on both tasks:
- Stance: +2.8% F1 (0.6194 → 0.6478)
- Sentiment: +2.1% F1 (0.9459 → 0.9666)

This confirms that financial stance and sentiment are related tasks that
benefit from shared representations.

**4. The TF-IDF baseline is surprisingly strong for sentiment.**
87.2% accuracy with a bag-of-words model suggests financial sentiment often
boils down to keyword presence. Neural models add value mainly on ambiguous
cases.

**5. Zero-shot FinBERT excels at sentiment but fails at stance.**
FinBERT was trained for sentiment, not stance. Using sentiment labels as
stance proxies (positive→hawkish) achieves only 49.8% — barely above
random (33.3%). This proves stance and sentiment are distinct tasks.

**6. Few-shot BERT/RoBERTa perform worse than the TF-IDF baseline.**
With only 48 training examples, the frozen transformer embeddings don't
capture enough task-specific signal. The TF-IDF baseline, trained on 1700+
examples, beats them handily. This shows that more training data can
compensate for a simpler model.

### 14.3 Model Progression Story

The results tell a clear progression story:

```
Rules only (lexicon)        → ~40% F1 (no learning)
Statistical baseline (LR)   → ~59% F1 (learns from data)
Domain model, few data       → ~45% F1 (right model, too little data)
Domain model, full data      → ~62% F1 (right model, right data)
Multi-task domain model      → ~65% F1 (right model, more data, shared learning)
```

Each step adds something: learning capability → domain knowledge → more data
→ shared multi-task signal.

---

## 15. Key Takeaways

### For NLP Practitioners

1. **Always start with a baseline.** TF-IDF + LR takes 5 seconds to train
   and tells you how hard the problem is.

2. **Domain pre-training > model size.** FinBERT (110M params) on financial
   tasks beats general BERT/RoBERTa (110-125M params). Use domain-specific
   models when they exist.

3. **Multi-task learning is free performance.** If you have related tasks,
   training them jointly costs nothing extra and improves both.

4. **Class imbalance must be addressed.** Without weighted loss, the stance
   model predicts "neutral" for most inputs (easy to get 49% accuracy,
   hard to get high F1).

5. **Evaluation metrics matter.** Accuracy can be misleading with imbalanced
   classes. Always report per-class F1 and macro-F1.

### For This Course

- The code is organized to be runnable in steps: `python run_experiments.py --step N`
- All results are saved as JSON in `results/` for easy analysis
- The CLI and Gradio demo provide quick ways to test the trained models
- Confusion matrices visualize where each model struggles

### For Future Work

- **More data**: The FOMC dataset is small (2,480 sentences). Scraping more
  Fed communications could improve stance classification significantly.
- **Larger models**: GPT-4 or Claude could be evaluated in zero-shot mode
  for comparison with fine-tuned FinBERT.
- **Cross-domain transfer**: Does training on FOMC stance help classify
  ECB (European Central Bank) or BOJ (Bank of Japan) communications?
- **Temporal analysis**: Do hawkish/dovish patterns change over time?
  Could train on older FOMC data and test on newer data.

---

## 16. Complete Code Reference — Every Function Explained

This section documents **every function and class** in the project, file by file.
For each function, we explain: purpose, parameters, return value, and internal logic.

---

### 16.1 `config.py` — Central Configuration

This file contains no functions — only constants. It is imported by every
other file. Everything is centralised here so when you want to change
a setting (e.g., increase batch size), you only edit ONE place.

```python
# Detect compute device in priority order
if torch.backends.mps.is_available():    # Apple Silicon GPU
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():          # NVIDIA GPU
    DEVICE = torch.device("cuda")
else:                                    # CPU if no GPU available
    DEVICE = torch.device("cpu")
```

**Why check MPS before CUDA?** Because we run on MacBook M3 Max.
If running on the H200 cluster, it would automatically select CUDA.

```python
SENTIMENT_LABELS = ["negative", "neutral", "positive"]
SENTIMENT_ID2LABEL = {i: l for i, l in enumerate(SENTIMENT_LABELS)}
# → {0: "negative", 1: "neutral", 2: "positive"}
SENTIMENT_LABEL2ID = {l: i for i, l in enumerate(SENTIMENT_LABELS)}
# → {"negative": 0, "neutral": 1, "positive": 2}
```

The two dicts `ID2LABEL` and `LABEL2ID` help convert back and forth between
integers (what models use) and strings (what humans read).

---

### 16.2 `src/data_loader.py` — Data Loading & Processing

#### `load_financial_phrasebank()`

**Purpose**: Load the Financial PhraseBank dataset from HuggingFace,
standardise columns, and split into train/val/test.

**Parameters**: None (uses constants from config).

**Returns**: `DatasetDict` with 3 keys: `"train"`, `"val"`, `"test"`.
Each split has columns: `text` (str), `label` (int), `label_name` (str).

**Internal logic**:
```python
# 1. Load from HuggingFace
ds = load_dataset(FPB_DATASET_NAME, FPB_SUBSET)

# 2. Combine all available splits into one DataFrame
#    (the original dataset only has train/test, no val)
frames = []
for split_name in ds:
    frames.append(ds[split_name].to_pandas())
df = pd.concat(frames, ignore_index=True)

# 3. Rename column "sentence" → "text" for consistency
df = df.rename(columns={"sentence": "text"})

# 4. Add label_name column for readability
df["label_name"] = df["label"].map({0: "negative", 1: "neutral", 2: "positive"})

# 5. Stratified split: ensures same class proportions in each split
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"])
train_df, val_df = train_test_split(train_df, test_size=0.125, stratify=...)
# 0.125 = 0.1 / 0.8 → since train is already 80%, taking 12.5% of 80% = 10% overall
```

**Why combine then re-split?** Because the original dataset has no validation
set. We need val to select the best model during training.

---

#### `load_fomc_dataset()`

**Purpose**: Load the FOMC Hawkish-Dovish dataset. Has a fallback mechanism:
tries HuggingFace first, falls back to local CSV if it fails.

**Logic**:
```python
try:
    ds = load_dataset(FOMC_DATASET_NAME)    # try HuggingFace
except Exception as e:
    return _load_fomc_local()               # fallback: local file
```

Why have a fallback? Because the network might fail or HuggingFace might
rename the dataset in the future.

---

#### `_load_fomc_local()`

**Purpose**: Fallback function — loads FOMC from CSV file in the `data/` directory.
The `_` prefix is a Python convention meaning "internal function, should not
be called directly from outside the module".

---

#### `_process_fomc_df(df)`

**Purpose**: Standardise the FOMC DataFrame — find the correct text and label
columns, convert string labels to integers, split the dataset.

**Notable logic**:
```python
# Find the text column — may have different names across dataset versions
if "sentence" in df.columns:
    df = df.rename(columns={"sentence": "text"})

# Convert string labels to integers if needed
if df["label"].dtype == object:           # object = string type in pandas
    label_map = {"dovish": 0, "hawkish": 1, "neutral": 2}
    df["label"] = df["label"].str.strip().str.lower().map(label_map)
    # .strip() removes extra whitespace
    # .lower() converts to lowercase
    # .map() applies the dict mapping
```

---

#### `get_few_shot_subset(dataset_split, k=16)`

**Purpose**: Sample k examples from EACH class for few-shot learning.
With k=16 and 3 classes → returns 48 examples.

**Parameters**:
- `dataset_split`: A HuggingFace Dataset split (e.g., `fomc["train"]`)
- `k`: Number of examples per class (default 16)

**Logic**:
```python
pieces = []
for label_val in sorted(df["label"].unique()):  # iterate through: 0, 1, 2
    subset = df[df["label"] == label_val]        # filter examples for this class
    pieces.append(subset.sample(n=min(k, len(subset)), random_state=SEED))
    # .sample(n=k) randomly selects k examples
    # min(k, len(subset)) handles case where class has fewer than k examples
sampled = pd.concat(pieces, ignore_index=True)  # merge into one DataFrame
```

**Why use a loop instead of groupby?** Because `groupby().apply()` in
pandas 3.0 has a bug that drops the label column (see Section 13, Bug 3).

---

#### `compute_class_weights(dataset_split, num_classes=3)`

**Purpose**: Compute class weights for weighted cross-entropy loss.
Fewer samples → higher weight → model pays more attention.

**Formula**: `weight_k = N / (C × n_k)`
- N = total samples, C = number of classes, n_k = samples in class k

**Example** with FOMC (1736 training samples):
```
dovish (455 samples):  1736 / (3 × 455) = 1.272
hawkish (424 samples): 1736 / (3 × 424) = 1.365
neutral (857 samples): 1736 / (3 × 857) = 0.675
```

Neutral has the lowest weight because it has the most samples.

---

#### `_print_split_stats(name, splits, label_names)`

**Purpose**: Print dataset statistics to console — each split's size and
class distribution. Utility function for quick data inspection.

---

### 16.3 `src/evaluate.py` — Evaluation Metrics

#### `compute_metrics(y_true, y_pred, label_names)`

**Purpose**: Compute all evaluation metrics at once.

**Parameters**:
- `y_true`: List of true labels (integers)
- `y_pred`: List of predicted labels (integers)
- `label_names`: List of label name strings

**Returns**: Dict containing:
```python
{
    "accuracy": 0.6593,
    "macro_f1": 0.6478,
    "per_class_f1": {"dovish": 0.6182, "hawkish": 0.6050, "neutral": 0.7202},
    "report": "... full classification_report table ..."
}
```

**Internal logic**:
```python
# f1_score with average=None returns F1 SEPARATELY for each class
per_class_f1 = f1_score(y_true, y_pred, average=None, ...)
# → [0.6182, 0.6050, 0.7202]

# f1_score with average="macro" takes equal-weighted average
macro_f1 = f1_score(y_true, y_pred, average="macro", ...)
# → (0.6182 + 0.6050 + 0.7202) / 3 = 0.6478

# zero_division=0: if a class has no predictions, F1 = 0 instead of error
```

---

#### `print_classification_report(metrics, model_name, task_name)`

**Purpose**: Pretty-print evaluation results to console. Pure display function,
does no computation.

---

#### `plot_confusion_matrix(y_true, y_pred, label_names, model_name, task_name, save_dir=None)`

**Purpose**: Create and save a confusion matrix heatmap.

**Logic**:
```python
# 1. Compute confusion matrix using scikit-learn
cm = confusion_matrix(y_true, y_pred, labels=range(len(label_names)))
# cm is a 2D array, e.g.:
# [[85,  6, 39],     ← dovish: 85 correct, 6 confused with hawk, 39 with neutral
#  [ 5, 85, 31],     ← hawkish: 85 correct
#  [55, 70, 120]]    ← neutral: 120 correct

# 2. Plot using seaborn
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ...)
# annot=True: display numbers in each cell
# fmt="d": integer format (no decimal places)
# cmap="Blues": blue colour palette (darker = higher value)

# 3. Save file
fig.savefig(path, dpi=150)     # dpi=150 for good resolution
plt.close(fig)                  # close figure to free memory
```

**Why `matplotlib.use("Agg")`?** At the top of the file, we set the backend
to "Agg" (Anti-Grain Geometry) — a non-interactive backend that only saves
files. Without this, matplotlib may try to open a GUI window, causing errors
on servers or when running via SSH.

---

#### `error_analysis(texts, y_true, y_pred, label_names, top_n=20)`

**Purpose**: Collect and analyse misclassified examples.

**Returns**:
- `error_df`: DataFrame of misclassified examples, sorted by text length
  (longer sentences tend to be more ambiguous → more interesting to analyse)
- `error_counts`: Dict counting each error type, e.g., `{"neutral → hawkish": 49}`

---

#### `save_results(results_dict, filename)`

**Purpose**: Save results dict as JSON file in the `results/` directory.
`default=str` in `json.dump` handles non-serialisable types
(e.g., numpy float → string).

---

### 16.4 `src/baseline.py` — Baseline Model

#### `build_baseline_pipeline()`

**Purpose**: Create a scikit-learn pipeline of TF-IDF vectoriser + Logistic
Regression. Returns an untrained pipeline.

**Why use Pipeline?** Pipeline ensures:
1. `fit()` is only called on training data (prevents data leakage)
2. `transform()` applies the same transformation to test data
3. A single object manages the entire workflow

---

#### `train_and_evaluate_baseline(train_split, test_split, label_names, task_name)`

**Purpose**: Train the baseline and evaluate on the test set. This is a
"one-button" function — call it and you're done.

**Processing flow**:
```python
# 1. Extract text and labels from HuggingFace Dataset
train_texts = train_split["text"]     # → list of strings
train_labels = train_split["label"]   # → list of integers

# 2. Build, train, predict
pipeline = build_baseline_pipeline()
pipeline.fit(train_texts, train_labels)      # TF-IDF fit + LR fit
predictions = pipeline.predict(test_texts)   # TF-IDF transform + LR predict

# 3. Evaluate
metrics = compute_metrics(test_labels, predictions, label_names)
plot_confusion_matrix(...)
save_results(...)
```

**Returns**: `(metrics_dict, pipeline)` — both the metrics and the trained
pipeline (for reuse if needed).

---

### 16.5 `src/lexicon.py` — Lexicon-Based Classification

#### `_tokenize(text)`

**Purpose**: Split text into simple tokens (words).

```python
def _tokenize(text):
    return re.findall(r'\b[a-z]+\b', text.lower())
```

- `text.lower()`: convert to lowercase
- `re.findall(r'\b[a-z]+\b', ...)`: find all words containing only letters
- `\b` = word boundary
- `[a-z]+` = one or more lowercase letters
- Result: `"The Fed raised rates!"` → `["the", "fed", "raised", "rates"]`

**Why not use a more complex tokeniser?** Because lexicon matching only needs
single-word matching. A simple tokeniser is faster and accurate enough.

---

#### `extract_lexicon_features(texts)`

**Purpose**: Extract 8 numerical features from each text based on the lexicon.

**Parameters**: `texts` — list of text strings.

**Returns**: Numpy array of shape `(n_texts, 8)`.

**Logic for each text**:
```python
tokens = _tokenize(text)          # split into words
total = max(len(tokens), 1)       # avoid division by zero

# Count words in each group using set lookup (O(1) per lookup)
pos_count = sum(1 for t in tokens if t in LM_POSITIVE)
neg_count = sum(1 for t in tokens if t in LM_NEGATIVE)
# ... same for uncertainty, hawkish, dovish

# Compute normalised indices
net_sentiment = (pos_count - neg_count) / total
net_stance = (hawk_count - dove_count) / total
```

**Why use `set` for word lists?** Lookup `t in set` is O(1) (constant time),
while `t in list` is O(n) (linear time). With ~230 negative words and
thousands of tokens, sets are much faster.

---

#### `lexicon_rule_based(test_split, label_names, task_name)`

**Purpose**: Pure rule-based classification — NO training at all.

**Classification logic**:
```python
if task_name == "sentiment":
    # Use net_sentiment (positive - negative, normalised)
    if ns > 0.02:   y_pred = 2   # positive
    elif ns < -0.02: y_pred = 0   # negative
    else:           y_pred = 1   # neutral
else:
    # Use hawkish vs dovish word counts
    if hawk > dove:  y_pred = 1   # hawkish
    elif dove > hawk: y_pred = 0   # dovish
    else:            y_pred = 2   # neutral
```

**Threshold 0.02** for sentiment was chosen empirically. Too low → too
sensitive, too high → everything becomes neutral.

---

#### `lexicon_plus_tfidf(train_split, test_split, label_names, task_name)`

**Purpose**: Combine TF-IDF + 8 lexicon features → train LR.

**Key step — Feature concatenation**:
```python
# TF-IDF gives a sparse 50,000-dimensional vector
train_tfidf = tfidf.fit_transform(train_texts)  # (1736, 50000) sparse

# Lexicon gives a dense 8-dimensional array
train_lex = extract_lexicon_features(train_texts)  # (1736, 8) dense

# Normalise lexicon features to be on the same scale as TF-IDF
scaler = StandardScaler()
train_lex_scaled = scaler.fit_transform(train_lex)
# StandardScaler: (x - mean) / std → mean 0, standard deviation 1

# Concatenate the two matrices: 50,000 + 8 = 50,008 dimensions
train_combined = hstack([train_tfidf, csr_matrix(train_lex_scaled)])
# hstack = horizontal stack (concatenate horizontally)
# csr_matrix() converts dense array to sparse matrix
```

**Why do we need StandardScaler?** TF-IDF values range [0, 1].
Lexicon values (e.g., word counts) can be 0-20. Without normalisation,
LR would ignore the lexicon features because they're on a different scale.

---

#### `run_lexicon_experiments(fomc_splits, fpb_splits)`

**Purpose**: Run all lexicon experiments — calls the 4 functions above.
"Orchestrator" function — no logic of its own, just calls other functions.

---

### 16.6 `src/pretrained_eval.py` — Pre-trained Model Evaluation

#### `evaluate_finbert_native(test_split, task_name="sentiment")`

**Purpose**: Evaluate FinBERT using its built-in classification head (zero-shot).

**Notable logic**:
```python
clf = pipeline(
    "text-classification",
    model=FINBERT_MODEL,
    device=-1,           # CPU — pipeline is unstable on MPS
    top_k=None,          # return scores for ALL classes, not just top-1
)

# Process in batches for speed (faster than one sentence at a time)
for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i : i + BATCH_SIZE]
    results = clf(batch)         # → list of list of dicts
    for r in results:
        best = max(r, key=lambda x: x["score"])  # pick class with highest score
        pred_label = best["label"].lower()       # "Positive" → "positive"
```

**Label mapping for stance**:
When using FinBERT (which classifies sentiment) for stance, we map:
- "negative" → dovish (negative news often relates to easing)
- "positive" → hawkish (positive news often relates to tightening)
- "neutral" → neutral

This is only a crude proxy — and the poor results (49.8%) prove
sentiment ≠ stance.

---

#### `class FewShotClassifier(nn.Module)`

**Purpose**: Simple linear classifier for few-shot learning.

```python
class FewShotClassifier(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),                    # randomly zero 10% of neurons
            nn.Linear(hidden_size, num_labels), # 768 → 3
        )

    def forward(self, x):
        return self.classifier(x)   # x: (batch, 768) → logits: (batch, 3)
```

**nn.Sequential**: Container that chains layers sequentially. Output of
one layer becomes input to the next.

**nn.Dropout(0.1)**: During training, randomly sets 10% of values to 0.
Prevents the model from over-relying on any single feature. Automatically
disabled when `model.eval()` is called.

**nn.Linear(768, 3)**: Linear transformation `y = Wx + b` with
W: (3, 768), b: (3,). Learns to map from 768-dim embedding space to 3 classes.

---

#### `_encode_texts(tokenizer, model, texts, device)`

**Purpose**: Convert a list of texts to [CLS] embeddings using a frozen model.

**Step-by-step logic**:
```python
model.eval()    # disable dropout, set batch norm to inference mode
model.to(device)

for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i : i + BATCH_SIZE]

    # Tokenise: text → token IDs + attention mask
    inputs = tokenizer(
        batch,
        padding=True,         # pad shorter sentences to match longest in batch
        truncation=True,      # truncate sentences longer than max_length
        max_length=128,
        return_tensors="pt",  # return PyTorch tensors (not lists)
    ).to(device)

    # Forward pass without computing gradients (saves memory, runs faster)
    with torch.no_grad():
        outputs = model(**inputs)
        # outputs.last_hidden_state: (batch, seq_len, 768)
        # Take the first token ([CLS]) from each sentence:
        cls_emb = outputs.last_hidden_state[:, 0, :]  # (batch, 768)
        all_embeddings.append(cls_emb.cpu())  # move to CPU for concatenation

return torch.cat(all_embeddings, dim=0)  # concatenate all batches: (n_texts, 768)
```

**`torch.no_grad()`**: Disables the autograd system. Normally PyTorch records
every computation for gradient calculation (backprop). When only doing
inference, we don't need gradients → saves ~50% GPU memory and is ~20% faster.

---

#### `evaluate_few_shot(model_name, train_split, test_split, label_names, task_name, k=16)`

**Purpose**: Few-shot evaluation — train a linear probe on frozen embeddings.

**Full flow**:
```python
# 1. Load base model (without classification head)
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModel.from_pretrained(model_name)

# 2. Get 48 examples (16 × 3 classes)
few_shot_data = get_few_shot_subset(train_split, k=k)

# 3. Encode into embeddings
train_embs = _encode_texts(tokenizer, base_model, few_shot_data["text"], device)
# train_embs: (48, 768)
test_embs = _encode_texts(tokenizer, base_model, test_split["text"], device)
# test_embs: (496, 768) for FOMC

# 4. Free GPU memory (model is ~440MB)
base_model.cpu()
del base_model
torch.mps.empty_cache()

# 5. Train linear probe for 200 epochs (on CPU, fast since only 48 samples)
classifier = FewShotClassifier(768, 3)
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
# Adam: adaptive learning rate, lr=1e-3 is the common default

for epoch in range(200):
    logits = classifier(train_embs)          # (48, 3)
    loss = criterion(logits, train_labels)   # cross-entropy loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 6. Predict
test_logits = classifier(test_embs)           # (496, 3)
y_pred = test_logits.argmax(dim=1).numpy()    # pick class with highest logit
```

**Why 200 epochs?** With only 48 samples and 1 linear layer, training is
extremely fast (~0.1 seconds for 200 epochs). Many epochs ensure convergence.

**Why lr=1e-3 for probe but 2e-5 for fine-tuning?** The probe has only
~2,300 parameters (768×3 + 3) → needs a high learning rate to converge quickly.
Fine-tuning has 110 million pre-trained parameters → needs a low learning rate
to avoid destroying them.

---

#### `run_all_pretrained_evaluations(fomc_splits, fpb_splits)`

**Purpose**: Orchestrate all pretrained evaluations — FinBERT zero-shot on
both datasets, and few-shot for 3 models × 2 tasks = 8 experiments.

---

### 16.7 `src/finetune.py` — FinBERT Fine-Tuning

#### `class TextClassificationDataset(TorchDataset)`

**Purpose**: Wrap text data so PyTorch DataLoader can create batches.

```python
def __len__(self):
    return len(self.texts)     # DataLoader needs to know total sample count

def __getitem__(self, idx):    # DataLoader calls this when it needs sample idx
    encoding = self.tokenizer(
        self.texts[idx],
        padding="max_length",   # ALWAYS pad to 128 tokens
        truncation=True,
        max_length=self.max_length,
        return_tensors="pt",
    )
    return {
        "input_ids": encoding["input_ids"].squeeze(0),
        # squeeze(0): remove extra batch dimension
        # tokeniser returns (1, seq_len), squeeze → (seq_len,)
        # DataLoader will add batch dimension automatically when collating
        "attention_mask": encoding["attention_mask"].squeeze(0),
        "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        # torch.long = int64, required for cross-entropy loss
    }
```

**Why `padding="max_length"` instead of `padding=True`?**
- `padding=True` pads to the longest sentence in the batch → each batch
  has different lengths → cannot cache/optimise
- `padding="max_length"` always pads to 128 → all samples same size →
  more efficient DataLoader

---

#### `finetune_finbert(train_split, val_split, test_split, label_names, task_name, use_weighted_loss=True)`

**Purpose**: Fine-tune all of FinBERT on a single task.

**This is the largest function — explained section by section**:

```python
# --- Load model ---
model = AutoModelForSequenceClassification.from_pretrained(
    FINBERT_MODEL,
    num_labels=num_labels,
    ignore_mismatched_sizes=True,  # ← IMPORTANT
)
```

`ignore_mismatched_sizes=True`: The original FinBERT has a classification
layer for 3 sentiment labels. We also need 3 labels but for a different task.
This flag tells the model loader to create a NEW classification layer with
random weights, while keeping the encoder intact.

```python
# --- Create DataLoaders ---
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
# shuffle=True: shuffle data each epoch → model doesn't memorise order

val_loader = DataLoader(val_ds, batch_size=32)
# NO shuffle for val/test → consistent results across runs
```

```python
# --- Optimizer ---
optimizer = torch.optim.AdamW(
    model.parameters(), lr=2e-5, weight_decay=0.01
)
```

**AdamW** (Adam with Weight Decay): Improvement over Adam optimizer.
- Adam: adaptive learning rate per parameter, uses momentum
- W (weight decay): adds L2 regularisation, prevents weights from growing too large
- `weight_decay=0.01`: each step, weights are multiplied by (1 - 0.01 × lr)

```python
# --- Learning Rate Scheduler ---
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * 0.1),   # first 10% of steps
    num_training_steps=total_steps,
)
```

The scheduler changes learning rate EVERY STEP (not every epoch):
- Steps 1→10%: linearly increase from 0 → 2e-5 (warmup)
- Steps 10%→100%: linearly decrease from 2e-5 → 0 (decay)

```python
# --- Training loop ---
for epoch in range(5):
    model.train()      # enable dropout, batch norm in training mode

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)       # move to GPU
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # outputs.logits: (batch_size, 3) — raw scores before softmax

        loss = criterion(outputs.logits, labels)
        # Cross-entropy applies softmax internally

        optimizer.zero_grad()    # clear old gradients (PyTorch accumulates!)
        loss.backward()          # compute gradients via backpropagation

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # Gradient clipping: if ‖g‖ > 1.0, scale g ← g × 1.0/‖g‖
        # Prevents "gradient explosion" when loss spikes suddenly

        optimizer.step()         # update weights: w ← w - lr × gradient
        scheduler.step()         # adjust learning rate
```

```python
# --- Save best model ---
if val_metrics["macro_f1"] > best_val_f1:
    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    # .cpu(): move tensor to CPU (so it doesn't occupy GPU memory)
    # .clone(): create a copy (not a reference)
    # state_dict(): dict containing all parameters, e.g.:
    # {"bert.encoder.layer.0.attention.self.query.weight": tensor(...), ...}
```

---

#### `_evaluate_model(model, dataloader, label_names, device)`

**Purpose**: Wrapper — calls `_get_predictions` then `compute_metrics`.

#### `_get_predictions(model, dataloader, device)`

**Purpose**: Run model on a DataLoader, return `(y_true, y_pred)`.

```python
model.eval()       # disable dropout
with torch.no_grad():   # disable gradient computation
    for batch in dataloader:
        outputs = model(input_ids=..., attention_mask=...)
        preds = outputs.logits.argmax(dim=-1).cpu().tolist()
        # argmax(dim=-1): pick the index of the class with the highest logit
        # dim=-1 = last dimension = the 3-class dimension
        # .cpu(): move to CPU
        # .tolist(): tensor → Python list
```

---

### 16.8 `src/multitask.py` — Multi-Task Learning

#### `class MultiTaskFinBERT(nn.Module)`

Already explained in detail in [Section 10.1](#101-model-architecture-srcmultitaskpy).
Additional detail:

```python
self.encoder = AutoModel.from_pretrained(FINBERT_MODEL)
hidden_size = self.encoder.config.hidden_size  # 768
```

`self.encoder.config` is a configuration object containing architecture info:
hidden_size=768, num_attention_heads=12, num_hidden_layers=12, etc.

---

#### `train_multitask(fomc_splits, fpb_splits)`

**Purpose**: Train the multi-task model. The largest function in the project.

**Batch alternation part** (detailed explanation):
```python
stance_iter = iter(stance_train_loader)      # create iterator
sentiment_iter = iter(sentiment_train_loader)
stance_done = False
sentiment_done = False

while not (stance_done and sentiment_done):
    # Try to get a stance batch
    if not stance_done:
        try:
            batch = next(stance_iter)   # get next batch
            loss = _train_step(..., task="stance")
        except StopIteration:           # stance data exhausted
            stance_done = True

    # Try to get a sentiment batch
    if not sentiment_done:
        try:
            batch = next(sentiment_iter)
            loss = _train_step(..., task="sentiment")
        except StopIteration:
            sentiment_done = True
```

**StopIteration**: Python raises this exception when an iterator has no more
elements. `next()` tries to get the next element; if exhausted →
StopIteration → mark as done.

The two datasets have different sizes (FOMC: 55 batches, FPB: 50 batches).
When one finishes first, the other continues to completion.

---

#### `_train_step(model, batch, criterion, optimizer, scheduler, device, task)`

**Purpose**: Execute ONE training step on ONE batch.
Separated into its own function for reuse across both tasks.

```python
logits = model(input_ids=..., attention_mask=..., task=task)
# task="stance" → uses stance_head
# task="sentiment" → uses sentiment_head
loss = criterion(logits, labels)
optimizer.zero_grad()
loss.backward()
clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
scheduler.step()
return loss.item()  # .item() converts 0-dim tensor → Python float
```

---

#### `_evaluate_multitask(model, dataloader, label_names, device, task)`

Same as `_evaluate_model` but with additional `task` parameter to select
the correct head.

#### `_get_multitask_predictions(model, dataloader, device, task)`

Same as `_get_predictions` but passes `task` to `model(...)`.

---

### 16.9 `run_experiments.py` — Experiment Orchestration

#### `step1_load_data()` through `step5_multitask()`

Each step function calls the correct module. No complex logic.

#### `print_summary(all_results)`

Prints a summary results table. Infers task name from the dict key:
```python
task = "stance" if "stance" in key else "sentiment"
model = key.replace(f"_{task}", "").replace("_", " ").title()
# "baseline_stance" → task="stance", model="Baseline"
```

#### `main()`

```python
parser = argparse.ArgumentParser(...)
parser.add_argument("--step", type=int, default=0)
# --step 0 (default) = run everything
# --step 2 = only run baseline + lexicon
```

---

### 16.10 `cli.py` — Command-Line Interface

#### `load_multitask_model()`

```python
model = MultiTaskFinBERT()    # create model with correct architecture
model.load_state_dict(torch.load(
    os.path.join(model_path, "model.pt"),
    map_location=DEVICE,       # load onto the correct device (MPS/CPU/CUDA)
    weights_only=True,         # security: only load tensors, not code
))
model.to(DEVICE)
model.eval()                   # inference mode
```

**`weights_only=True`**: When `torch.load()` loads a pickle file, it can
execute arbitrary code (security risk). `weights_only=True` only allows
loading tensor data.

---

#### `load_finetune_models()`

Loads TWO separate models (stance + sentiment). Different from multitask
which is a single model for both.

```python
model = AutoModelForSequenceClassification.from_pretrained(model_path)
# from_pretrained loads both architecture + weights from saved directory
```

---

#### `predict_multitask(text, model, tokenizer)` and `predict_finetune(text, models)`

**Common logic**:
```python
inputs = tokenizer(text, return_tensors="pt", truncation=True, ...)
inputs = inputs.to(DEVICE)

with torch.no_grad():
    logits = model(...)
    probs = F.softmax(logits, dim=-1)    # logits → probabilities (sum = 1.0)
    pred_idx = probs.argmax().item()     # index of highest probability class
    confidence = probs[pred_idx].item()  # probability of that class
```

**`F.softmax`**: Converts logits (can be negative, unbounded) into
probabilities (0-1, sum to 1):
```
softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
```
Example: logits [2.1, -0.5, 0.3] → softmax → [0.72, 0.05, 0.12]

---

#### `format_prediction(results)`

Creates a display string with visual bars:
```python
bar = "█" * int(score * 30)
# score=0.9 → 27 █ characters
# score=0.05 → 1 █ character
```

---

### 16.11 `demo.py` — Gradio Demo

#### `load_model()`

Same as `load_multitask_model()` in cli.py, but also returns the model
name for display on the web interface.

#### `predict(text, model, tokenizer)`

Returns TWO dicts (stance_scores, sentiment_scores) instead of a nested dict.
Gradio needs each output to be a separate dict `{label: score}`.

```python
stance_scores = {}
for i, label in enumerate(STANCE_LABELS):
    stance_scores[label] = float(stance_probs[i])
# → {"dovish": 0.036, "hawkish": 0.340, "neutral": 0.624}
```

`float()` converts a PyTorch tensor → Python float (Gradio can't
understand tensors).

#### `create_demo()`

```python
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # gr.Blocks: flexible container (vs simpler gr.Interface)
    # theme=Soft(): clean, professional look

    with gr.Row():        # horizontal row layout
        with gr.Column(scale=2):  # left column, 2× wider
            text_input = gr.Textbox(...)
        with gr.Column(scale=1):  # right column, narrower
            stance_output = gr.Label(num_top_classes=3)
            # gr.Label: displays labels with probability bars

    gr.Examples(
        examples=examples,
        inputs=text_input,
        fn=classify,
        cache_examples=False,  # don't cache — re-run each time clicked
    )

    # Two ways to trigger: button click or Enter key
    submit_btn.click(fn=classify, inputs=text_input, outputs=[...])
    text_input.submit(fn=classify, inputs=text_input, outputs=[...])
```

**`demo.launch(share=False, server_name="0.0.0.0", server_port=7860)`**:
- `share=False`: don't create a public URL (local access only)
- `server_name="0.0.0.0"`: listen on all network interfaces
- `server_port=7860`: Gradio's default port

---

*Document generated for COMP6713 2026 T1. All code, experiments, and analysis
are reproducible using `python run_experiments.py`.*
