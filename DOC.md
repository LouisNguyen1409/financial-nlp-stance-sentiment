# Financial NLP — Stance and Sentiment Classification

Technical documentation for the UNSW COMP6713 2026 Term 1 project.
This document is written in English. A Vietnamese translation is provided in `DOC_VI.md`.
It is aimed at a reader who is new to Natural Language Processing and who wants
to understand every design decision and every major piece of code in this
repository. The companion file `README.md` is a shorter quick-start guide.

The code, results, and models described here live entirely inside the project
root at `/Users/louis/COMP6713/Project`. Every file path referenced in this
document is relative to that root unless stated otherwise.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Background Theory](#2-background-theory)
3. [Environment and Libraries](#3-environment-and-libraries)
4. [Project Architecture](#4-project-architecture)
5. [Datasets In Depth](#5-datasets-in-depth)
6. [The Loughran-McDonald Lexicon](#6-the-loughran-mcdonald-lexicon)
7. [Baseline Models](#7-baseline-models)
8. [Pre-trained Transformer Evaluation](#8-pre-trained-transformer-evaluation)
9. [Fine-tuning FinBERT and BERT-base](#9-fine-tuning-finbert-and-bert-base)
10. [Multi-task Learning](#10-multi-task-learning)
11. [Evaluation Methodology](#11-evaluation-methodology)
12. [CLI, Demo, and HuggingFace Publishing](#12-cli-demo-and-huggingface-publishing)
13. [Bugs Encountered](#13-bugs-encountered)
14. [Results Analysis](#14-results-analysis)
15. [Key Takeaways](#15-key-takeaways)
16. [Complete Code Reference](#16-complete-code-reference)

---

## 1. What This Project Does

This project builds and compares a sequence of machine-learning models that
classify short sentences from the world of finance along two independent axes.

The first axis is **monetary-policy stance**. Given a sentence drawn from the
minutes of the United States Federal Open Market Committee (FOMC), the model
decides whether the sentence leans **hawkish** (arguing for tighter monetary
policy — rate hikes, balance-sheet reductions, the removal of accommodation),
**dovish** (arguing for looser monetary policy — rate cuts, stimulus, patience
in the face of weakness), or **neutral** (neither clearly hawkish nor dovish —
procedural language, factual description, or genuinely balanced commentary).

The second axis is **market sentiment**. Given a sentence drawn from the
Financial PhraseBank — a collection of sentences from financial news, earnings
releases, and analyst reports, each annotated by multiple domain experts — the
model decides whether the sentence expresses a **positive**, **negative**, or
**neutral** view of a company's financial prospects.

These two tasks are closely related but distinct. Both are three-way
classifications on short, domain-specific English text. Both use the same set
of underlying transformer architectures. Both are evaluated on held-out test
sets with the same metrics (accuracy and macro-F1). And yet the label
distributions differ, the vocabularies differ, and — as we will see
quantitatively in Section 14 — the difficulty differs substantially. Stance
classification on FOMC sentences turns out to be roughly twenty percentage
points harder than sentiment classification on Financial PhraseBank.

### Why it matters

Automating the extraction of stance and sentiment from financial text has real
applications. Central-bank watchers read thousands of pages of FOMC
transcripts each year to gauge the probability of a future rate change, and
even small changes in tone can move currency and bond markets within seconds.
Quantitative traders subscribe to real-time news feeds and build features from
the sentiment of each headline. Risk teams at banks and asset managers want to
flag narrative shifts in corporate communications. A fast, accurate, and
reproducible classifier that can run locally on a laptop — as this one does —
is a useful building block for all of these workflows.

### Credit system

The COMP6713 project scope document defines a credit system. Our submission
implements every mandatory component plus a substantial amount of optional
work. The table below summarises the allocation.

| Part   | Description                                              | Credits    |
|--------|----------------------------------------------------------|-----------:|
| A      | Data ingestion, preprocessing, and splits                |         20 |
| B      | Classical baselines (TF-IDF + Logistic Regression, SVM)  |         20 |
| C      | Transformer models (zero/few-shot, fine-tuning, multi-task, LLRD) |    80+ |
| D      | Extensions (Loughran-McDonald lexicon, Gradio demo, HuggingFace publishing, analysis plots) |         35 |
| **Total** |                                                       | **155+**   |

The components in Parts C and D are what carry the model quality from the
mid-50 macro-F1 range of classical baselines on stance into the 0.638 macro-F1
that our best model achieves.

### Layout of the rest of this document

Section 2 introduces the theory a new student needs in order to read the rest
of the document. Section 3 describes the Python environment and the role each
major library plays. Section 4 gives a file-by-file tour of the repository.
Section 5 describes the two datasets in detail. Section 6 explains the
Loughran-McDonald lexicon that we use both as a stand-alone rule-based
classifier and as a feature source. Sections 7 through 10 step through the
four families of models — TF-IDF baselines, zero/few-shot transformers, fine-
tuned transformers, and the multi-task model — in the same order
`run_experiments.py` executes them. Section 11 explains the evaluation
methodology. Section 12 describes the user-facing deliverables (command-line
interface, web demo, and HuggingFace uploads). Section 13 catalogues the bugs
we hit so future students do not have to rediscover them. Section 14 analyses
the full results table. Section 15 lists the practical takeaways. Section 16
is a reference that describes every source file function-by-function.

---

## 2. Background Theory

This section is the theoretical minimum a reader needs to follow the rest of
the document. It introduces text classification as a task, the three main
families of model we use (linear classifiers over sparse features,
transformer-based language models, and task-specific fine-tuning strategies),
and the evaluation metrics. Readers who already know this material can skip
ahead.

### 2.1 Text classification

Text classification is the problem of assigning a label from a fixed, small
set of classes to an arbitrary piece of text. Given an input sentence `x` and
a label space `Y = {y_1, y_2, ..., y_K}`, we want a function `f` such that
`f(x) in Y` agrees with a human annotator's judgement as often as possible.
Here `K = 3` for both of our tasks.

There are many ways to build `f`. A linear classifier over bag-of-words
features has been the standard industrial approach for at least three decades
and still delivers surprisingly strong numbers when the training data is of
reasonable size. A transformer-based approach is more recent (2017 onward),
more accurate on hard problems, and more compute-hungry. We implement both
and compare them directly.

### 2.2 Bag-of-words and TF-IDF

The simplest way to represent a sentence numerically is to list how often each
word in a vocabulary appears in it. If the vocabulary has `V` entries, each
sentence becomes a vector in `R^V` with mostly zeros. This representation
is called the **bag-of-words**: it throws away word order but keeps word
identity and word frequency.

Raw word counts have two problems. First, very common words (`the`, `of`,
`and`) dominate the vector. Second, absolute counts do not reflect how
informative a word is: a word that occurs in every sentence tells us nothing
about class. **TF-IDF** (Term Frequency — Inverse Document Frequency) fixes
both problems. For a word `w` in document `d` from a corpus of `N` documents:

- TF(w, d) is the count of `w` in `d`, often log-scaled: `1 + log(count)`.
- IDF(w) = `log(N / df(w))` where `df(w)` is the number of documents that
  contain `w` at least once.
- TF-IDF(w, d) = TF(w, d) × IDF(w).

A word that appears in almost every document has IDF close to zero. A word
that appears in only a few documents gets a large IDF boost. The `sublinear_tf=True`
flag in scikit-learn activates the log-scaled TF. We use TF-IDF with 1- and
2-word n-grams so that short fixed phrases (e.g. "raise rates", "weak demand")
also become features.

### 2.3 Logistic Regression

Once we have a TF-IDF vector for each sentence, we need a classifier. For
three-class problems, **multinomial logistic regression** is the standard
choice. It learns a weight matrix `W` such that, for an input vector `x`,
the predicted class is `argmax(softmax(W x))`. Training minimises cross-
entropy loss with an L2 penalty on the weights:

```
L(W) = -Σ_i log P(y_i | x_i; W) + λ ||W||²
```

The optimisation is convex, so the training is fast and the solution is
unique up to regularisation. `class_weight="balanced"` in scikit-learn
rescales each class's contribution to the loss by the inverse of its
frequency, which matters when the classes are imbalanced (as they are in
FOMC).

### 2.4 Linear SVM

A linear Support Vector Machine is another linear classifier. Instead of
minimising cross-entropy, it minimises the **hinge loss**: examples that
are correctly classified with a margin of at least one contribute no loss;
examples that are inside the margin or misclassified contribute
`max(0, 1 - y_i (w · x_i))`. Hinge loss is not differentiable everywhere
but is convex, and in practice Linear SVM often gives slightly higher
accuracy than Logistic Regression on TF-IDF features. The `C=1.0`
regularisation parameter controls how much the classifier prioritises
margin width over classification correctness on the training set.

### 2.5 Transformers and BERT

The **transformer** architecture (Vaswani et al., 2017) replaced
recurrent networks as the dominant NLP architecture. Its core building
block is **self-attention**: for each token in a sentence, it computes a
weighted combination of all other tokens' representations, where the
weights depend on the tokens themselves. Stacking many layers of self-
attention plus feed-forward sublayers gives a model that produces a
contextual embedding for each token — the same word `cut` has a
different representation in "rate cut" and "spending cut" and "cut
short".

**BERT** (Bidirectional Encoder Representations from Transformers,
Devlin et al. 2019) is a transformer encoder pre-trained on general
English text with two self-supervised objectives: predicting randomly
masked words from context (masked language modelling) and predicting
whether two sentences are adjacent in a document (next-sentence
prediction). After pre-training on billions of tokens, BERT's weights
encode a general statistical knowledge of English grammar, word
senses, and discourse. `bert-base-uncased` is the canonical 12-layer,
768-hidden-dim, 110-million-parameter version used throughout this
project as a general-purpose baseline.

Every BERT-family model we use reserves a special token `[CLS]` as the
first token of every input. Its representation after the final layer is
conventionally used as a whole-sentence embedding, and that is what we
pool to produce classification logits.

### 2.6 FinBERT

**FinBERT** (Araci, 2019; we use the Hugging Face checkpoint
`ProsusAI/finbert`) takes `bert-base` and continues pre-training on a
corpus of financial text — earnings transcripts, analyst reports, and
the like — and then fine-tunes on Financial PhraseBank to produce a
three-way sentiment classifier. The result is a model whose
representations are specifically good at financial English. Two
consequences follow.

First, FinBERT's native output head already produces sentiment
predictions that are very accurate on Financial PhraseBank out of the
box, as Section 14 shows. Second, even when we discard the native head
and attach a fresh classifier for a new task (like stance), FinBERT's
contextual embeddings are a better starting point for finance than
general-purpose BERT embeddings — particularly when training data is
scarce.

### 2.7 Zero-shot, few-shot, and fine-tuning

These three terms describe how much task-specific supervision we give a
pre-trained model before evaluating it.

- **Zero-shot** means we use the pre-trained model with its existing
  classification head and no task-specific training at all. The model's
  decision comes directly from the weights it learned during pre-
  training. We use FinBERT's native sentiment head zero-shot on both
  sentiment (direct three-way match) and stance (as a proxy where
  positive → hawkish, negative → dovish, neutral → neutral).
- **Few-shot** means we give the model a handful of labelled examples
  per class — in our setup, `k = 16` — and adapt only a small number of
  parameters. We do this by freezing the whole encoder, extracting a
  single `[CLS]` embedding per sentence, and training a linear
  classifier on top of those frozen embeddings using 48 training
  examples total.
- **Fine-tuning** means we take a pre-trained model, attach a fresh
  classification head, and update all or most of the parameters on the
  full labelled training set. This is the most expensive option but
  also — for classical NLP benchmarks — the most accurate.

### 2.8 Multi-task learning

When two related tasks share the same input domain, it often helps to
train them jointly so that the encoder learns representations useful for
both tasks. In **multi-task learning** we share the encoder across
tasks and attach one classification head per task. Each training batch
belongs to one task; the loss for that batch updates the shared
encoder and the corresponding head. If the tasks are complementary,
each task acts as regularisation for the other and the shared encoder
ends up better than either task's encoder on its own.

### 2.9 Weighted cross-entropy

When one class is much more common than the others, a neutrally trained
classifier tends to over-predict the majority class because doing so
minimises the empirical loss fastest. **Weighted cross-entropy** fixes
this by multiplying each class's contribution by a weight inversely
proportional to its frequency. We use the scheme

```
w_c = N / (C × n_c)
```

where `N` is the total number of training examples, `C` is the number
of classes, and `n_c` is the count of class `c`. On the FOMC training
set this gives dovish `1736/(3×455) = 1.272`, hawkish
`1736/(3×424) = 1.365`, and neutral `1736/(3×857) = 0.675`. The neutral
class is the majority and gets down-weighted; the two rarer classes
are up-weighted.

### 2.10 Layer-wise Learning Rate Decay

When fine-tuning a pre-trained transformer, using the same learning
rate for all parameters can damage the low-level linguistic features
baked in during pre-training. **Layer-wise Learning Rate Decay (LLRD)**
(Sun et al., 2019) addresses this by assigning a smaller learning rate
to lower (closer-to-input) layers and a larger rate to higher (closer-
to-output) layers. In our setup the top-most classifier head gets the
base learning rate `2e-5`. Each successively deeper transformer layer's
rate is multiplied by `0.9`, so the twelfth layer below the head gets
`2e-5 × 0.9^12 ≈ 5.6e-6`. The embeddings get the smallest rate of all.
The effect is that task-specific adaptation happens mostly in the upper
layers, while the representations that took billions of tokens to
acquire are preserved in the lower layers.

### 2.11 Gradual Unfreezing

**Gradual unfreezing** (Howard and Ruder, 2018, ULMFiT) is complementary
to LLRD. In the first epoch only the classifier head is trainable; in
the second epoch we additionally unfreeze the top transformer layer; in
the third epoch the top two layers; and so on, working downward one
layer per epoch. This gives the randomly initialised head time to
stabilise before we let its gradients propagate into the pre-trained
weights. It is the implicit analogue of a learning-rate warm-up for
deep transfer learning.

### 2.12 Label smoothing

Softmax cross-entropy rewards the model for pushing the correct class's
probability all the way to one and the others all the way to zero. On
small datasets this encourages over-confidence and calibration loss.
**Label smoothing** replaces the one-hot target with a softened
distribution: `1 - ε` on the correct class and `ε / (K-1)` on each
other class, where `ε` is typically `0.1`. The model is now encouraged
to be *almost* certain about the correct class rather than absolutely
certain, which gives a small but consistent generalisation boost.

---

## 3. Environment and Libraries

The project runs on Python 3.12 and pins a small set of scientific and
deep-learning libraries. The hardware target is a MacBook M3 Max using
Apple's MPS backend for PyTorch, but the same code runs on NVIDIA CUDA
(tested on the H200 cluster available for COMP6713) and falls back to
CPU where neither is present.

### 3.1 Python 3.12

Python 3.12 is the last version that all dependencies (PyTorch,
`transformers`, `datasets`, `scikit-learn`, `gradio`) support as of the
time the project was frozen. Python 3.14 is *not* supported because
PyTorch wheels are not yet released for 3.14 at the time of submission;
see Section 13 for details.

### 3.2 Device selection

Device selection lives in `config.py` and is the first thing any script
touches when it is imported:

```python
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
```

Apple's MPS backend gives roughly a 5× speed-up over CPU for transformer
fine-tuning on the M3 Max. Where MPS is unstable (specifically in the
`transformers` text-classification pipeline used for zero-shot FinBERT;
see Section 13), we fall back to CPU explicitly.

### 3.3 torch

PyTorch is the tensor library underneath every neural model in the
project. We use it directly, without Lightning or the higher-level
`transformers.Trainer`, because the project is small enough that
explicit training loops are clearer and easier to debug.

### 3.4 transformers

Hugging Face's `transformers` library provides the tokenisers
(`AutoTokenizer`), the backbone models (`AutoModel`,
`AutoModelForSequenceClassification`), and the linear-warmup learning-
rate scheduler (`get_linear_schedule_with_warmup`) we use for fine-
tuning. The `pipeline()` helper is used only once, for zero-shot FinBERT
on CPU.

### 3.5 datasets

Hugging Face's `datasets` library loads both of our benchmarks directly
from the Hub. Financial PhraseBank lives at
`gtfintechlab/financial_phrasebank_sentences_allagree` (subset `5768`)
and FOMC at `gtfintechlab/fomc_communication`. The library returns
`DatasetDict` objects that `data_loader.py` then converts to pandas
DataFrames so we can do stratified splits with scikit-learn.

### 3.6 scikit-learn

scikit-learn supplies the classical baselines (`TfidfVectorizer`,
`LogisticRegression`, `LinearSVC`), the stratified train/val/test
splitter (`train_test_split`), and the metrics (`classification_report`,
`f1_score`, `accuracy_score`, `confusion_matrix`). We also use
`StandardScaler` to normalise the lexicon features before stacking them
with sparse TF-IDF.

### 3.7 pandas, numpy, matplotlib, seaborn

pandas handles the DataFrames produced when we merge multiple
`datasets` splits for re-splitting. numpy is used for lexicon feature
matrices. matplotlib (with the Agg non-interactive backend) plus
seaborn produce the confusion-matrix heatmaps and the data-analysis
figures; see `data_analysis.py` for the latter.

### 3.8 gradio, accelerate, tqdm

Gradio powers the web demo on port 7860 described in Section 12.
`accelerate` is pulled in transitively by `transformers` for device
placement. `tqdm` provides the progress bars you see during training.

### 3.9 config.py constants and why they matter

Every hyperparameter in the project is defined once in `config.py`. The
relevant ones are listed below with a short justification.

- `SEED = 42` — fixed seed for every library that has one
  (`random`, `numpy`, `torch`, `sklearn`). The specific value does not
  matter but pinning it means every run reproduces the same splits, the
  same few-shot samples, and the same initial weights.
- `MAX_SEQ_LENGTH = 128` — BERT-family models can process up to 512
  subword tokens, but FOMC and FPB sentences are short (median under
  30 tokens), so 128 is plenty of headroom and uses four times less
  memory than the default.
- `BATCH_SIZE = 32` — balances convergence speed against memory use on
  24 GB of unified M3 Max memory. Smaller batches slow training; larger
  ones run out of memory on MPS.
- `LEARNING_RATE = 2e-5` — the canonical fine-tuning rate for BERT-
  family models.
- `WEIGHT_DECAY = 0.01` — standard AdamW decay for transformer fine-
  tuning; acts as an L2 regulariser on the weights.
- `FINETUNE_EPOCHS = 5` — enough for the single-task FinBERT runs to
  converge without over-fitting; the validation macro-F1 usually peaks
  at epoch 3 or 4.
- `MULTITASK_EPOCHS = 8` — the multi-task model alternates batches from
  two datasets, so each task sees fewer gradient updates per epoch;
  more epochs compensate.
- `FEW_SHOT_K = 16` — 16 examples per class, 48 per dataset, is the
  point where frozen-encoder linear probes start to converge reliably
  while remaining clearly "low-data".
- `WARMUP_RATIO = 0.1` — the first 10% of training steps linearly ramp
  the learning rate from zero to its target value. This prevents large
  initial gradients from destroying the pre-trained weights.
- `TEST_SIZE = 0.2`, `VAL_SIZE = 0.1` — 70/10/20 stratified splits for
  both datasets.

The LLRD-specific constants live in `src/finetune_bert.py`:

- `LLRD_BASE_LR = 2e-5` — head learning rate.
- `LLRD_DECAY = 0.9` — per-layer decay factor.
- `LABEL_SMOOTHING = 0.1` — smoothing ε for cross-entropy.
- `BERT_EPOCHS = 10` — longer schedule so gradual unfreezing can reach
  the embeddings.
- `NUM_BERT_LAYERS = 12` — `bert-base-uncased` architecture constant.

---

## 4. Project Architecture

The repository is laid out as follows:

```
/Users/louis/COMP6713/Project
├── config.py                  # all hyperparameters and constants
├── run_experiments.py         # orchestrator for the 6 experiment steps
├── cli.py                     # command-line classifier
├── demo.py                    # Gradio web demo (port 7860)
├── data_analysis.py           # generates the 10 dataset-analysis plots
├── push_to_hf.py              # uploads 5 trained models to HuggingFace
├── requirements.txt           # pinned dependencies
├── README.md                  # quick-start guide
├── DOC.md                     # this document
├── DOC_VI.md                  # Vietnamese translation
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # load + split FPB and FOMC
│   ├── baseline.py            # TF-IDF + LR, TF-IDF + SVM, TF-IDF trigrams + LR
│   ├── lexicon.py             # Loughran-McDonald word lists and classifiers
│   ├── pretrained_eval.py     # zero-shot FinBERT + few-shot linear probes
│   ├── finetune_fineBert.py   # single-task FinBERT fine-tuning
│   ├── finetune_bert.py       # BERT-base with LLRD + gradual unfreezing
│   ├── multitask.py           # multi-task FinBERT (shared encoder, dual heads)
│   └── evaluate.py            # metrics, confusion matrices, error analysis
├── data/                      # cached downloads
├── models/                    # trained model checkpoints
├── results/                   # all JSON results and PNG figures
├── analysis/                  # 10 data-analysis plots
├── logs/                      # training logs
└── presentation/              # Beamer slides
```

### 4.1 Data flow

Every run — whether the whole pipeline or a single stage — follows the
same six-step flow:

```
 ┌───────────────────────────┐
 │ 1. Load FPB + FOMC        │
 │    (datasets → HF Hub)    │
 │    stratified 70/10/20    │
 └───────────┬───────────────┘
             │
             ▼
 ┌───────────────────────────┐
 │ 2. Baselines              │
 │    TF-IDF + LR / SVM / tri│
 │    LM lexicon (rules+TF)  │
 └───────────┬───────────────┘
             │
             ▼
 ┌───────────────────────────┐
 │ 3. Pre-trained eval       │
 │    FinBERT zero-shot      │
 │    16-shot linear probes  │
 │    (FinBERT/BERT/RoBERTa) │
 └───────────┬───────────────┘
             │
             ▼
 ┌───────────────────────────┐
 │ 4. Single-task FinBERT    │
 │    fine-tuning × 2 tasks  │
 └───────────┬───────────────┘
             │
             ▼
 ┌───────────────────────────┐
 │ 5. Multi-task FinBERT     │
 │    shared encoder + heads │
 └───────────┬───────────────┘
             │
             ▼
 ┌───────────────────────────┐
 │ 6. BERT-base LLRD + GUF   │
 │    fine-tuning × 2 tasks  │
 └───────────┬───────────────┘
             │
             ▼
      results/all_results_summary.json
```

Each step writes per-experiment JSON files (e.g.
`results/finetune_finbert_stance.json`) and confusion-matrix PNGs (e.g.
`results/FinBERT_finetuned_stance_cm.png`). The final step of
`run_experiments.py` aggregates everything into
`results/all_results_summary.json`, which is the single source of truth
for the numbers in this document.

---

## 5. Datasets In Depth

### 5.1 FOMC Hawkish-Dovish

The **FOMC Hawkish-Dovish** dataset (`gtfintechlab/fomc_communication`
on the Hugging Face Hub) contains sentences drawn from Federal Open
Market Committee minutes and statements. Each sentence is labelled by
domain experts as hawkish, dovish, or neutral. The full corpus after
concatenating all source splits and de-duplicating has **2,480** usable
sentences. Our stratified 70/10/20 split (random state `42`) produces:

| Split | Total | dovish | hawkish | neutral |
|-------|-----:|------:|-------:|-------:|
| train |  1736 |   455 |    424 |     857 |
| val   |   248 |    65 |     61 |     122 |
| test  |   496 |   130 |    121 |     245 |

The neutral class dominates. In the training set, `857 / 1736 ≈ 49.4%`
of examples are neutral and the remaining two classes are roughly
balanced against each other. This imbalance motivates the weighted
cross-entropy used in the FinBERT, LLRD, and multi-task stance
training runs (Section 2.9).

### 5.2 Financial PhraseBank (allagree)

The **Financial PhraseBank** dataset
(`gtfintechlab/financial_phrasebank_sentences_allagree`, subset
`5768`) is a well-known corpus of about 4,800 English-language
financial news sentences. The original dataset has four "agreement"
subsets: sentences on which 50%, 66%, 75%, or 100% of human annotators
agreed. We use only the **all-annotator-agreement** subset (hence
`allagree`). The rationale is that these are the sentences where the
ground-truth label is unambiguous — all human annotators assigned the
same sentiment — and therefore any residual error after training is
signal about the model, not noise about the annotation. After taking
the all-agreement subset we have **2,264** sentences. Our stratified
70/10/20 split produces:

| Split | Total | negative | neutral | positive |
|-------|-----:|--------:|-------:|--------:|
| train |  1584 |     212 |    973 |      399 |
| val   |   227 |      30 |    140 |       57 |
| test  |   453 |      61 |    278 |      114 |

Again the neutral class is the majority (`973 / 1584 ≈ 61%`). Negative
examples are the rarest class, and indeed in the final results
(Section 14) the negative-class F1 is lower than the positive or
neutral class F1 for every transformer model.

### 5.3 Stratified splitting

Both datasets are split using scikit-learn's `train_test_split` with
`stratify=df["label"]`. Stratification guarantees that each split has
the same class proportions as the original (up to rounding), which is
essential when one class is much rarer than the others. The same
`random_state=42` is used for both train/test and train/val splits so
that anyone can reproduce the exact sentences that ended up in each
split.

The code in `src/data_loader.py` performs the split in two steps:
first separate 20% of the data for the test set, then separate another
`0.1 / 0.9 ≈ 11.1%` of the remaining 80% for validation. The effective
proportions are therefore 70%/10%/20%. A second technical detail: the
dataset, as it arrives from the Hub, may already have splits; we
concatenate them all into a single DataFrame before re-splitting, so
our train/val/test boundaries are under our control and not inherited
from whoever uploaded the data.

### 5.4 The `label` and `label_name` columns

After preprocessing, each example has three fields:

- `text` — the raw sentence (string)
- `label` — an integer in `{0, 1, 2}`
- `label_name` — the corresponding human-readable string

For sentiment the mapping is `{0: "negative", 1: "neutral", 2:
"positive"}`. For stance it is `{0: "dovish", 1: "hawkish", 2:
"neutral"}`. These mappings are defined once in `config.py`
(`SENTIMENT_LABELS`, `STANCE_LABELS`) and every script imports them
from there.

### 5.5 Few-shot subset

Finally, `get_few_shot_subset(dataset_split, k=16)` in
`src/data_loader.py` samples exactly `k` examples per class from a
split, stratified and seeded. For `k=16` and 3 classes this produces
48 training examples. This is the training set used for the few-shot
linear probes in Section 8.

---

## 6. The Loughran-McDonald Lexicon

Before we had transformers, financial-NLP research relied heavily on
hand-crafted **word lists** — curated lists of words that indicate
positive, negative, or uncertain sentiment in financial text. The
**Loughran-McDonald (LM) Financial Sentiment Dictionary** (Loughran
and McDonald 2011, updated 2023) is the canonical example. Their
insight was that general-purpose sentiment lexicons are not
appropriate for financial text, because many words the general lexicon
flags as negative (e.g. "tax", "liability", "cost", "capital") are
perfectly neutral accounting vocabulary in a 10-K filing.

We ship five curated word lists in `src/lexicon.py`:

- `LM_POSITIVE` (~110 words) — e.g. `achieve`, `benefit`, `gain`,
  `robust`, `outperform`, `surge`.
- `LM_NEGATIVE` (~230 words) — e.g. `loss`, `decline`, `risk`,
  `default`, `recession`, `weaken`.
- `LM_UNCERTAINTY` (~90 words) — e.g. `approximate`, `contingent`,
  `uncertain`, `speculative`, `may`, `might`.
- `HAWKISH_WORDS` (~40 words) — monetary-tightening cues, e.g.
  `hike`, `tighten`, `restrictive`, `normalize`, `taper`,
  `overheating`.
- `DOVISH_WORDS` (~50 words) — monetary-easing cues, e.g. `cut`,
  `ease`, `accommodate`, `stimulus`, `patient`, `slack`, `recession`.

The positive/negative/uncertainty lists are curated subsets of the LM
master dictionary. The hawkish/dovish lists are task-specific
additions assembled from central-banking literature for FOMC stance
classification.

### 6.1 The eight extracted features

For each sentence, `extract_lexicon_features` returns a length-8
vector:

1. `positive_count` — number of tokens that appear in `LM_POSITIVE`.
2. `negative_count` — number of tokens in `LM_NEGATIVE`.
3. `uncertainty_count` — number of tokens in `LM_UNCERTAINTY`.
4. `hawkish_count` — number of tokens in `HAWKISH_WORDS`.
5. `dovish_count` — number of tokens in `DOVISH_WORDS`.
6. `net_sentiment` — `(positive_count - negative_count) / total_words`.
7. `net_stance` — `(hawkish_count - dovish_count) / total_words`.
8. `total_words` — total number of tokens (used for scale).

Tokens are produced by a simple regular-expression tokeniser that
keeps only alphabetic runs:

```python
def _tokenize(text):
    return re.findall(r'\b[a-z]+\b', text.lower())
```

Lowercasing before matching makes the lexicon case-insensitive. The
denominator `total_words` in the `net_*` features prevents long
sentences from dominating short ones just because they contain more
matches.

### 6.2 Rule-based classifier

The rule-based classifier in `lexicon_rule_based(...)` makes a
decision purely from counts, with no training:

- **Sentiment.** Classify as `positive` if `net_sentiment > 0.02`,
  `negative` if `net_sentiment < -0.02`, otherwise `neutral`. The
  `0.02` threshold was chosen so that a sentence needs at least one
  strongly polarised word to leave the neutral band.
- **Stance.** Classify as `hawkish` if `hawkish_count > dovish_count`,
  `dovish` if `dovish_count > hawkish_count`, otherwise `neutral`.

The rule-based classifier hits `0.6932` accuracy / `0.5315` macro-F1
on sentiment and `0.4153` accuracy / `0.3885` macro-F1 on stance — see
the full results table in Section 14. On stance it is only slightly
above random (random three-class accuracy is 0.333), because FOMC
sentences routinely contain both hawkish and dovish words in the same
sentence (e.g. "while inflation remains elevated, the Committee
anticipates that moderate growth will support a gradual return to
target"). Counting cannot disambiguate what the sentence actually
argues for.

### 6.3 TF-IDF + lexicon hybrid

A stronger lexicon baseline is `lexicon_plus_tfidf(...)`, which
combines the 8 lexicon features with a full TF-IDF vector and trains a
single Logistic Regression. The TF-IDF vectoriser is the same as in
the baseline (1- and 2-grams, 50k vocab, sublinear TF). The 8 lexicon
features are standardised with `StandardScaler` (mean zero, unit
variance) and then horizontally stacked onto the sparse TF-IDF matrix
with `scipy.sparse.hstack`:

```python
train_combined = hstack([train_tfidf, csr_matrix(train_lex_scaled)])
```

This model achieves `0.8543` accuracy / `0.8050` macro-F1 on sentiment
and `0.6109` accuracy / `0.5863` macro-F1 on stance — slightly below
the plain TF-IDF + LR baseline. The lexicon features help rule-based
classification (large absolute gain over the counting rules) but do
not meaningfully add to what the TF-IDF features already capture.
This is an instructive negative result: once you have TF-IDF n-grams,
the hand-crafted sentiment/stance lists are redundant.

---

## 7. Baseline Models

The baselines in `src/baseline.py` are three linear models trained on
scikit-learn TF-IDF features. They are cheap to train (seconds on
CPU) and set a floor below which no transformer should fall.

### 7.1 TF-IDF + Logistic Regression

The reference baseline `build_baseline_pipeline()` is a two-stage
scikit-learn pipeline:

```python
Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=SEED,
        solver="lbfgs",
    )),
])
```

`max_features=50_000` caps the vocabulary so that rare typos and junk
n-grams do not bloat the matrix. `ngram_range=(1, 2)` includes bigrams
for simple phrase patterns like "rate hike" or "weak demand".
`sublinear_tf=True` applies the `1 + log(tf)` transform from
Section 2.2. `strip_accents="unicode"` normalises accented characters
(irrelevant for the English-only datasets here but a good habit).
`class_weight="balanced"` inverse-weights the three classes so that
the model cannot trivially win by predicting "neutral" everywhere. The
L-BFGS solver is chosen because it is fast and deterministic for
multinomial logistic regression.

This model reaches `0.8720` / `0.8232` on sentiment and
`0.6089` / `0.5873` on stance.

### 7.2 TF-IDF + Linear SVM

`build_tfidf_svm_pipeline()` uses the same vectoriser with
`LinearSVC(C=1.0)` in place of Logistic Regression. The hinge loss is
slightly better suited to high-dimensional sparse text features than
cross-entropy, and on both tasks the SVM edges out LR:
`0.8940` / `0.8534` on sentiment and `0.6331` / `0.6061` on stance.
This is the strongest classical baseline in the whole project. It
trains in under five seconds on a laptop.

### 7.3 TF-IDF (1–3 grams) + Logistic Regression

`build_tfidf_trigram_lr_pipeline()` extends the vectoriser to
trigrams, raises the vocabulary cap to 80,000, and filters any n-gram
that appears in fewer than two training documents (`min_df=2`). The
trigrams slightly help sentiment (`0.8786` / `0.8310`) and stance
(`0.6109` / `0.5914`) over the bigram LR baseline, but do not beat the
SVM.

### 7.4 Baseline summary

| Model                              | Sent. Acc | Sent. F1 | Stance Acc | Stance F1 |
|-----------------------------------|-----------|----------|------------|-----------|
| TF-IDF + LR (bigrams)             | 0.8720    | 0.8232   | 0.6089     | 0.5873    |
| TF-IDF + SVM (LinearSVC, C=1)     | 0.8940    | 0.8534   | 0.6331     | 0.6061    |
| TF-IDF (1-3 grams) + LR           | 0.8786    | 0.8310   | 0.6109     | 0.5914    |
| TF-IDF + LM Lexicon (LR)          | 0.8543    | 0.8050   | 0.6109     | 0.5863    |
| LM Lexicon (rule-based)           | 0.6932    | 0.5315   | 0.4153     | 0.3885    |

The takeaway from this table is that a plain linear classifier over
bag-of-bigrams features already crosses 0.85 macro-F1 on sentiment.
This means the real benefit of transformers on sentiment will only
show up in the last few percentage points. On stance the baselines
sit around 0.59–0.61, leaving a larger gap for transformers to close.

---

## 8. Pre-trained Transformer Evaluation

Before fine-tuning anything, we evaluate off-the-shelf transformers
to see how much of the task is solved by pre-training alone. The
code is in `src/pretrained_eval.py`.

### 8.1 Zero-shot FinBERT (native head)

`evaluate_finbert_native(...)` invokes FinBERT's existing three-way
sentiment head via `transformers.pipeline`. The pipeline runs on CPU
(`device=-1`) because MPS is unstable for the text-classification
pipeline on some `transformers` versions (see Section 13).

- For **sentiment** the output mapping is direct:
  `{"negative": 0, "neutral": 1, "positive": 2}`.
- For **stance** FinBERT does not have a hawkish/dovish head. As a
  proxy we map `negative → dovish`, `positive → hawkish`,
  `neutral → neutral`:

  ```python
  finbert_to_idx = {"negative": 0, "positive": 1, "neutral": 2}
  ```

  (The index values follow our own `STANCE_LABELS` order: dovish=0,
  hawkish=1, neutral=2.) This is a deliberately weak baseline — it
  tests whether sentiment polarity is a usable substitute for
  monetary-policy stance. It is not: zero-shot FinBERT scores
  `0.4980` accuracy / `0.4874` macro-F1 on stance, barely above the
  rule-based lexicon.

On sentiment, however, FinBERT's native head is stunning:
`0.9735` accuracy / `0.9650` macro-F1, without any task-specific
training on our splits. FinBERT was originally fine-tuned on Financial
PhraseBank (a slightly different split) so it was "born knowing" this
task.

### 8.2 Few-shot linear probe

The second evaluation `evaluate_few_shot(...)` is a fair head-to-head
between three pre-trained encoders on the same tiny amount of data:

1. Load the pre-trained encoder (FinBERT, `bert-base-uncased`, or
   `roberta-base`) with no classification head — just `AutoModel`.
2. Freeze the encoder. Encode 16 examples per class (48 total) and
   the whole test set to `[CLS]` embeddings.
3. Train a tiny linear classifier (`Dropout(0.1)` → `Linear(768, 3)`)
   on the 48 training embeddings for 200 epochs with Adam and
   learning rate `1e-3`.
4. Evaluate the linear classifier on the frozen test embeddings.

Because only the final linear layer is trained, any difference in
performance is attributable to the quality of the encoder's
representations — not to the classifier architecture or the optimiser.
This is the fairest possible comparison of "how much does finance-
specific pre-training help?".

Results:

| Encoder (16-shot)   | Sent. Acc | Sent. F1 | Stance Acc | Stance F1 |
|---------------------|-----------|----------|------------|-----------|
| FinBERT             | 0.9779    | 0.9670   | 0.4859     | 0.4534    |
| BERT-base-uncased   | 0.7417    | 0.6500   | 0.3851     | 0.3744    |
| RoBERTa-base        | 0.7682    | 0.6722   | 0.3730     | 0.3600    |

On sentiment FinBERT beats BERT-base by `0.9670 - 0.6500 = 0.3170`
macro-F1 (32 percentage points). This is the single most dramatic
number in the entire project. With only 48 training examples, domain
pre-training produces a model that approaches the fine-tuned ceiling,
while general-purpose models barely pass the rule-based baseline. On
stance, all three encoders are weak, because the pre-training signal
(general English for BERT/RoBERTa, financial-sentiment for FinBERT)
does not include anything hawkish-vs-dovish specific.

---

## 9. Fine-tuning FinBERT and BERT-base

This section covers the two flagship fine-tuning pipelines: standard
fine-tuning of FinBERT (`src/finetune_fineBert.py`) and LLRD-plus-
gradual-unfreezing fine-tuning of BERT-base (`src/finetune_bert.py`).

### 9.1 The `TextClassificationDataset` wrapper

Both fine-tuning scripts wrap a HuggingFace split in a PyTorch
`Dataset`:

```python
class TextClassificationDataset(TorchDataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_SEQ_LENGTH):
        ...
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }
```

The tokeniser pads every sentence to 128 tokens so the collate
function can stack them into a tensor. Truncation is unnecessary in
practice (the longest FOMC sentence is under 128 subwords) but the
flag is there for safety.

### 9.2 `finetune_finbert(...)` — entry point

The function signature is:

```python
def finetune_finbert(train_split, val_split, test_split, label_names,
                     task_name, use_weighted_loss=True):
```

It takes the three splits produced by the data loader, the label
names for pretty-printing, the task name (`"stance"` or
`"sentiment"`), and a flag to enable weighted cross-entropy. For
stance we set `use_weighted_loss=True` (FOMC is imbalanced); for
sentiment we set it to `False` (the weighting hurt more than it
helped in preliminary experiments).

### 9.3 Model initialisation

```python
model = AutoModelForSequenceClassification.from_pretrained(
    FINBERT_MODEL,
    num_labels=num_labels,
    ignore_mismatched_sizes=True,
)
```

`ignore_mismatched_sizes=True` is crucial here. `ProsusAI/finbert` on
the Hub already has a 3-way sentiment head. When we load it with
`num_labels=3` for stance, the label *count* matches but the head is
initialised from the sentiment head's weights. That is the wrong
semantics — we want a fresh randomly initialised head. The flag tells
`transformers` to replace the existing head silently rather than
raise.

### 9.4 Loss, optimiser, and schedule

```python
if use_weighted_loss:
    weights = compute_class_weights(train_split, num_classes=num_labels)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float).to(device)
    )
else:
    criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
total_steps = len(train_loader) * FINETUNE_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * WARMUP_RATIO),
    num_training_steps=total_steps,
)
```

For stance, the class weights are exactly those we computed in
Section 2.9: `[1.272, 1.365, 0.675]`. AdamW is Adam with weight decay
decoupled from the gradient update (Loshchilov and Hutter, 2019). The
linear warmup-then-decay schedule is the canonical choice for BERT
fine-tuning.

### 9.5 Training loop

The loop is intentionally explicit:

```python
for epoch in range(FINETUNE_EPOCHS):
    model.train()
    for batch in train_loader:
        outputs = model(input_ids=..., attention_mask=...)
        loss = criterion(outputs.logits, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    val_metrics = _evaluate_model(model, val_loader, label_names, device)
    if val_metrics["macro_f1"] > best_val_f1:
        best_val_f1 = val_metrics["macro_f1"]
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
```

Gradient clipping (`max_norm=1.0`) prevents occasional exploding-
gradient updates. At the end of each epoch we evaluate on the
validation set and keep the best checkpoint by macro-F1. After all
`FINETUNE_EPOCHS = 5` epochs are done, the best checkpoint is loaded
and used for the test-set evaluation.

### 9.6 Test evaluation and saving

The final test-set numbers are:

- **FinBERT fine-tuned — stance**: `0.6371` accuracy /
  `0.6194` macro-F1.
- **FinBERT fine-tuned — sentiment**: `0.9669` accuracy /
  `0.9459` macro-F1.

The model is saved to `models/finbert_stance/` or
`models/finbert_sentiment/` and can be reloaded by `cli.py`
(`--model finetune`).

### 9.7 BERT-base with LLRD + Gradual Unfreezing

`src/finetune_bert.py` implements the more elaborate recipe. Every
design choice in this file is a deliberate contrast to the FinBERT
pipeline above. The goal was to see how close a careful fine-tune of a
non-domain-specific BERT can get to FinBERT.

#### 9.7.1 Layer groups

`_build_layer_groups(model)` partitions the model's parameters into
14 ordered groups, from top (classifier) to bottom (embeddings):

- Group 0: classifier head + BERT pooler, `lr = 2e-5`.
- Group 1: transformer layer 11, `lr = 2e-5 × 0.9`.
- Group 2: transformer layer 10, `lr = 2e-5 × 0.9²`.
- …
- Group 12: transformer layer 0, `lr = 2e-5 × 0.9¹²`.
- Group 13: embeddings, `lr = 2e-5 × 0.9¹³`.

```python
for layer_idx in range(NUM_BERT_LAYERS - 1, -1, -1):
    depth = (NUM_BERT_LAYERS - 1) - layer_idx
    lr    = LLRD_BASE_LR * (LLRD_DECAY ** (depth + 1))
    groups.append({
        "name":   f"encoder_layer_{layer_idx}",
        "params": list(model.bert.encoder.layer[layer_idx].parameters()),
        "lr":     lr,
    })
```

#### 9.7.2 Gradual unfreezing schedule

`_build_optimizer(model, epoch)` constructs the optimiser for the
current epoch under the unfreezing schedule:

- Epoch 1: head only (1 group active, 593,667 trainable params for a
  3-class head).
- Epoch 2: head + layer 11.
- Epoch 3: head + layer 11 + layer 10.
- …
- Epoch 10: head + all 12 transformer layers (embeddings still
  frozen).

The embeddings are never unfrozen in our 10-epoch run because BERT-
base's token, position, and type embeddings are already well-
calibrated by pre-training; allowing gradients into them too early
risks catastrophic forgetting. Any group that is not currently active
has `requires_grad = False` and is excluded from the optimiser.

#### 9.7.3 Loss function

```python
if task_name == "stance":
    class_weights = compute_class_weights(train_split, num_classes=num_labels)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float).to(device),
        label_smoothing=LABEL_SMOOTHING,
    )
else:
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
```

Stance combines the class weights from Section 2.9 with label
smoothing ε = 0.1. Sentiment uses label smoothing alone. The
combination turns out to be the best regulariser we tried.

#### 9.7.4 No-weight-decay for the head

In `_build_optimizer`:

```python
wd = 0.0 if group["name"] == "head" else WEIGHT_DECAY
```

The classifier head is small and bias-heavy; shrinking its weights
under weight decay can flatten logits and hurt accuracy. We exclude
it from decay.

#### 9.7.5 Results

- **BERT-base LLRD + GUF — stance**: `0.6512` / `0.6383`.
- **BERT-base LLRD + GUF — sentiment**: `0.9757` / `0.9670`.

These numbers essentially **match** FinBERT fine-tuned (stance
`0.6371 / 0.6194`, sentiment `0.9669 / 0.9459`) and even **beat** it
slightly on both tasks. This is a major finding: a careful fine-tune
of a general-purpose BERT on the full training set closes the gap to
domain-specific pre-training almost entirely. See Section 14 for the
interpretation.

---

## 10. Multi-task Learning

The multi-task model in `src/multitask.py` trains a single shared
FinBERT encoder on both tasks simultaneously, with two separate
classification heads. This is the richest model in the project and
the one used by the Gradio demo.

### 10.1 Architecture

```python
class MultiTaskFinBERT(nn.Module):
    def __init__(self, num_stance_labels=3, num_sentiment_labels=3, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(FINBERT_MODEL)
        hidden_size = self.encoder.config.hidden_size  # 768
        self.dropout = nn.Dropout(dropout)
        self.stance_head    = nn.Linear(hidden_size, num_stance_labels)
        self.sentiment_head = nn.Linear(hidden_size, num_sentiment_labels)
```

The encoder is `ProsusAI/finbert`'s transformer (~110 million
parameters). After encoding, we take the `[CLS]` token's final-layer
representation (768 dimensions), apply dropout, and feed it into one
of two linear heads selected by the `task` argument to `forward(...)`.
The total number of trainable parameters is 110M + 768 × 3 + 768 × 3
= about 110M plus a handful of kilobytes of head weights.

### 10.2 Alternating-batch training

```python
while not (stance_done and sentiment_done):
    if not stance_done:
        try:
            batch = next(stance_iter)
            loss = _train_step(model, batch, stance_criterion, optimizer,
                               scheduler, device, task="stance")
            total_stance_loss += loss
        except StopIteration:
            stance_done = True

    if not sentiment_done:
        try:
            batch = next(sentiment_iter)
            loss = _train_step(model, batch, sentiment_criterion, optimizer,
                               scheduler, device, task="sentiment")
            total_sentiment_loss += loss
        except StopIteration:
            sentiment_done = True
```

Each epoch alternates stance and sentiment batches one at a time
until both loaders are exhausted. The FOMC training set has 55
batches of 32 and the FPB training set has 50 batches of 32, so one
multi-task epoch contains about 105 gradient steps in total —
slightly more than a single-task epoch. We use `MULTITASK_EPOCHS = 8`
so that each head sees roughly as many gradient updates as in a
single-task fine-tune.

### 10.3 Loss functions

```python
stance_weights = compute_class_weights(fomc_splits["train"], num_classes=3)
stance_criterion    = nn.CrossEntropyLoss(weight=torch.tensor(stance_weights).to(device))
sentiment_criterion = nn.CrossEntropyLoss()
```

Stance uses weighted cross-entropy. Sentiment uses plain cross-
entropy. No label smoothing for the multi-task model — we reserve
that for the LLRD run.

### 10.4 Validation and checkpointing

At the end of each epoch we evaluate both tasks on their validation
sets, average the two macro-F1 scores, and keep the checkpoint with
the best average:

```python
avg_f1 = (val_stance["macro_f1"] + val_sentiment["macro_f1"]) / 2
if avg_f1 > best_avg_f1:
    best_avg_f1 = avg_f1
    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
```

This prevents over-fitting on one task at the expense of the other.

### 10.5 Test results

- **Multi-task FinBERT — stance**: `0.6492` / `0.6384`.
- **Multi-task FinBERT — sentiment**: `0.9779` / `0.9666`.

Compared to single-task FinBERT (`0.6371 / 0.6194` and
`0.9669 / 0.9459`), multi-task learning adds **+1.90 pp** on stance
macro-F1 and **+2.07 pp** on sentiment macro-F1. The two tasks help
each other: stance training exposes the encoder to monetary-policy
vocabulary that Financial PhraseBank does not contain, and sentiment
training regularises the encoder so it does not over-fit the smaller
FOMC stance dataset.

### 10.6 Per-class F1

For the multi-task model, per-class F1 on the test set is:

- **Stance** — dovish `0.6174`, hawkish `0.5869`, neutral `0.7109`.
- **Sentiment** — negative `0.9500`, neutral `0.9928`, positive
  `0.9569`.

Neutral is the easiest class on both tasks, which is consistent with
its majority prevalence. On stance, hawkish is the hardest class —
the model often confuses hawkish for neutral when inflation or
tightening language is hedged (e.g. "the Committee remains attentive
to inflation risks" is technically hawkish-leaning but uses neutral
procedural syntax).

### 10.7 Error analysis

`error_analysis(...)` in `src/evaluate.py` produces a breakdown of
misclassifications by error type (e.g. `"hawkish → neutral"`,
`"dovish → neutral"`). The top error modes on stance are:

- `neutral → hawkish` and `hawkish → neutral` in roughly equal
  numbers, when monetary-policy language appears in procedural
  sentences.
- `dovish → neutral`, when easing language is present but hedged
  ("the Committee continues to monitor conditions").

On sentiment the errors concentrate on sentences that mix positive
and negative vocabulary in a single sentence (e.g. beat-but-guide-
down earnings reports).

---

## 11. Evaluation Methodology

All evaluation logic lives in `src/evaluate.py`. Four functions
cover everything:

- `compute_metrics(y_true, y_pred, label_names)` — returns
  `accuracy`, `macro_f1`, `per_class_f1` (dict), and a formatted
  scikit-learn `classification_report` string.
- `print_classification_report(metrics, model_name, task_name)` —
  pretty-prints the metrics to the console.
- `plot_confusion_matrix(y_true, y_pred, label_names, model_name,
  task_name)` — draws a seaborn heatmap and saves it to `results/`.
- `error_analysis(texts, y_true, y_pred, label_names, top_n=20)` —
  returns a DataFrame of the 20 longest misclassified examples and a
  dict counting each error type.

### 11.1 Accuracy vs macro-F1

Accuracy is the fraction of test examples classified correctly. It
is easy to interpret but misleading when classes are imbalanced: a
classifier that always predicts "neutral" on the FOMC test set would
score `245/496 = 0.494` accuracy without learning anything.

**Macro-F1** is the unweighted mean of the three per-class F1
scores. Each class contributes equally, so the majority class cannot
dominate the metric. For an imbalanced three-class problem, macro-F1
is the more honest summary statistic and is the metric we use to
pick the best validation checkpoint throughout training. We report
both so readers can see the relationship.

### 11.2 Per-class F1

The F1 for class `c` is the harmonic mean of precision and recall
for that class:

```
F1_c = 2 × (precision_c × recall_c) / (precision_c + recall_c)
```

We report per-class F1 for the two best models in Section 14 so
that readers can see where each model's remaining errors are
concentrated.

### 11.3 Confusion matrices

For each trained model we save a 3×3 confusion matrix as a PNG in
`results/`. Rows are true classes; columns are predicted classes;
the diagonal shows correct predictions and off-diagonal cells show
specific error modes.

### 11.4 JSON results

Every `_evaluate(...)` call writes a JSON file under `results/` with
fields `{model, task, accuracy, macro_f1, per_class_f1, report}`.
At the end of a full run, `run_experiments.py` aggregates all per-
experiment JSON files into `results/all_results_summary.json`, which
is the source of every number in Section 14.

---

## 12. CLI, Demo, and HuggingFace Publishing

Three user-facing deliverables expose the trained models.

### 12.1 `cli.py` — command-line classifier

```text
python cli.py                                        # interactive REPL
python cli.py --text "The Fed signaled further hikes"  # single sentence
python cli.py --file input.txt                       # one sentence per line
python cli.py --model finetune --text "..."          # use single-task FinBERT
                                                     # instead of multi-task
```

By default `cli.py` loads the multi-task model from
`models/multitask_finbert/` and returns both stance and sentiment
for each input. The `--model finetune` flag switches to two separate
fine-tuned FinBERT checkpoints (one per task).

The output for each sentence shows the top prediction, its
confidence, and a bar-chart-style breakdown of all class
probabilities. Confidence is produced by `F.softmax` on the logits.

### 12.2 `demo.py` — Gradio web demo

`demo.py` launches a Gradio Blocks interface on
`http://localhost:7860`. The user pastes any sentence; the interface
shows two `gr.Label` outputs — one for stance probabilities, one for
sentiment probabilities. Eight pre-filled examples from FOMC
statements and financial news are shown below the input box.

The demo uses the multi-task model only. If the model is not trained
yet, `load_model()` raises a clear `FileNotFoundError` with the
command to train it:

```
python run_experiments.py --step 5
```

### 12.3 `data_analysis.py` — 10 analysis plots

`data_analysis.py` is a separate script that produces exploratory
plots of both datasets:

1. Label distribution for FOMC (bar chart).
2. Label distribution for FPB (bar chart).
3. Sentence-length distribution for FOMC.
4. Sentence-length distribution for FPB.
5. Top words per class for FOMC.
6. Top words per class for FPB.
7. Co-occurrence of hawkish and dovish cues.
8. LM lexicon hit-rate per FOMC class.
9. LM lexicon hit-rate per FPB class.
10. Correlation heatmap of lexicon features.

All plots are saved to `analysis/`.

### 12.4 `push_to_hf.py` — HuggingFace upload

`push_to_hf.py` uploads the five trained checkpoints to the
HuggingFace Hub under the project namespace:

1. `finbert_stance/` — single-task FinBERT for stance.
2. `finbert_sentiment/` — single-task FinBERT for sentiment.
3. `bert_llrd_stance/` — BERT-base LLRD for stance.
4. `bert_llrd_sentiment/` — BERT-base LLRD for sentiment.
5. `multitask_finbert/` — multi-task FinBERT (both tasks).

Each repo includes the weights, the tokenizer, and a model card with
the test-set metrics so that readers on the Hub see the numbers
without running anything.

---

## 13. Bugs Encountered

This section documents the dependency and platform issues we hit.
The fixes are folded into `requirements.txt` and the source.

### 13.1 `datasets` 4.x dropped loading scripts

HuggingFace `datasets` 4.0+ removed support for Python-based
`loading_script.py` files on the Hub. Some older datasets (including
earlier versions of Financial PhraseBank) relied on them. We pin to
a `datasets` version that still supports the Parquet-based loaders,
and we use `gtfintechlab/financial_phrasebank_sentences_allagree`
rather than the classic `financial_phrasebank`, because the
`gtfintechlab` mirror ships as Parquet files with no loading script.

### 13.2 scikit-learn 1.8 removed `multi_class`

scikit-learn 1.8 removed the `multi_class` parameter of
`LogisticRegression` and made multinomial the only option. Our code
never passes `multi_class`, but some tutorials and older references
do; reviewers running newer scikit-learn will not hit this.

### 13.3 pandas 3.0 `groupby.apply` drops columns

In pandas 3.0, `groupby(...).apply(...)` drops the grouping columns
from the resulting DataFrame by default (the `include_groups=False`
behaviour became the default). The data-analysis plots originally
used the old behaviour; we pin pandas to a version that preserves
the grouping columns to keep the plotting code working unchanged.

### 13.4 MPS pipeline instability

Hugging Face `transformers.pipeline("text-classification", ...,
device="mps")` is unstable on some PyTorch versions — it returns
wrong probabilities and sometimes hangs. The fix is to run the
zero-shot FinBERT pipeline on CPU (`device=-1`) and let the batched
encoder paths use MPS:

```python
clf = pipeline(
    "text-classification",
    model=FINBERT_MODEL,
    tokenizer=FINBERT_MODEL,
    device=-1,
    ...
)
```

Throughput loss is small because the zero-shot evaluation only runs
on the test set and is fast anyway.

### 13.5 Python 3.14 incompatibility with PyTorch

At the time of submission, PyTorch does not yet publish wheels for
Python 3.14. Any attempt to `pip install torch` under Python 3.14
falls back to a source build that fails without the Xcode command-
line toolchain. Pin Python to 3.12.

---

## 14. Results Analysis

Every number below is read directly from
`results/all_results_summary.json`. Accuracy is the fraction correct
on the held-out test set; macro-F1 is the unweighted mean of the
three per-class F1 scores on the same test set.

### 14.1 Full results table

| Model                           | Sent. Acc | Sent. F1 | Stance Acc | Stance F1 |
|---------------------------------|-----------|----------|------------|-----------|
| LM Lexicon (rule-based)         | 0.6932    | 0.5315   | 0.4153     | 0.3885    |
| TF-IDF + LR                     | 0.8720    | 0.8232   | 0.6089     | 0.5873    |
| TF-IDF + SVM (LinearSVC, C=1)   | 0.8940    | 0.8534   | 0.6331     | 0.6061    |
| TF-IDF (1-3 grams) + LR         | 0.8786    | 0.8310   | 0.6109     | 0.5914    |
| TF-IDF + LM Lexicon (LR)        | 0.8543    | 0.8050   | 0.6109     | 0.5863    |
| FinBERT zero-shot (native head) | 0.9735    | 0.9650   | 0.4980     | 0.4874    |
| FinBERT few-shot (k=16 probe)   | 0.9779    | 0.9670   | 0.4859     | 0.4534    |
| BERT-base few-shot (k=16)       | 0.7417    | 0.6500   | 0.3851     | 0.3744    |
| RoBERTa-base few-shot (k=16)    | 0.7682    | 0.6722   | 0.3730     | 0.3600    |
| FinBERT (fine-tuned)            | 0.9669    | 0.9459   | 0.6371     | 0.6194    |
| BERT-base LLRD + Gradual UF     | 0.9757    | 0.9670   | 0.6512     | 0.6383    |
| Multi-task FinBERT              | 0.9779    | 0.9666   | 0.6492     | 0.6384    |

### 14.2 Per-class F1 of the two best models

**Multi-task FinBERT — stance**

| Class   | F1     |
|---------|--------|
| dovish  | 0.6174 |
| hawkish | 0.5869 |
| neutral | 0.7109 |

**Multi-task FinBERT — sentiment**

| Class    | F1     |
|----------|--------|
| negative | 0.9500 |
| neutral  | 0.9928 |
| positive | 0.9569 |

**BERT-base LLRD — stance**

| Class   | F1     |
|---------|--------|
| dovish  | 0.5831 |
| hawkish | 0.6084 |
| neutral | 0.7235 |

**BERT-base LLRD — sentiment**

| Class    | F1     |
|----------|--------|
| negative | 0.9600 |
| neutral  | 0.9891 |
| positive | 0.9520 |

### 14.3 Observation 1: BERT-base LLRD matches FinBERT and Multi-task FinBERT

Sentiment macro-F1 ties at **0.9670** for BERT-base LLRD and for
the FinBERT 16-shot probe, while the multi-task FinBERT lands at
**0.9666**. Stance macro-F1 is **0.6383** for BERT-base LLRD and
**0.6384** for multi-task FinBERT. A careful fine-tuning recipe on a
general-purpose BERT — Layer-wise Learning Rate Decay, Gradual
Unfreezing, label smoothing ε = 0.1, 10 epochs — substitutes almost
entirely for the domain pre-training of FinBERT. The contrast is
most informative in Observation 4 below.

### 14.4 Observation 2: Multi-task learning helps stance more than single-task

Versus single-task FinBERT, the multi-task model gains **+1.90 pp**
on stance macro-F1 (`0.6194 → 0.6384`) and **+2.07 pp** on sentiment
macro-F1 (`0.9459 → 0.9666`). Both tasks benefit; the shared encoder
absorbs vocabulary and syntactic patterns from both datasets.

### 14.5 Observation 3: Stance is fundamentally harder than sentiment

Best stance macro-F1 is **0.6384** (roughly a 36% macro error rate).
Best sentiment macro-F1 is **0.9670** (roughly a 3% macro error
rate). An order of magnitude of difficulty separates the two tasks
despite the identical input format, similar training-set sizes, and
identical model architectures. Stance judgements depend on implicit
policy reasoning that often is not in the sentence's surface words;
sentiment judgements are typically written with overt polarity cues.

### 14.6 Observation 4: Domain pre-training matters MOST in low-data settings

In the 16-shot setting (48 training examples), FinBERT scores
**0.9670** sentiment macro-F1 while BERT-base scores **0.6500** and
RoBERTa-base scores **0.6722**. The gap is roughly **32 percentage
points**. Once we fine-tune on the full training set the gap
collapses: BERT-base LLRD ties FinBERT fine-tuned on sentiment and
slightly beats it on stance. The practical implication is that if
you have only a handful of labelled examples, a domain-specific pre-
trained model is worth much more than any fine-tuning trick; if you
have a few thousand labels, a general-purpose model with a good
fine-tuning recipe catches up.

### 14.7 Observation 5: Zero-shot FinBERT solves sentiment

With zero task-specific supervision, FinBERT's native head reaches
**0.9735** accuracy / **0.9650** macro-F1 on the Financial
PhraseBank test set. The native head was trained on Financial
PhraseBank during FinBERT's original release, so the task is
essentially pre-solved for this dataset. Practically, if your
downstream task *is* Financial PhraseBank sentiment, you do not need
to train anything.

### 14.8 Observation 6: TF-IDF baselines are strong on sentiment

TF-IDF + SVM reaches **0.8940** accuracy / **0.8534** macro-F1 on
sentiment with under five seconds of CPU training. On a budget, a
classical baseline covers 87% of the sentiment task; transformers
add the last 10 percentage points at a cost of 110 million parameters
and 15+ minutes of GPU time. This trade-off is worth making
consciously.

---

## 15. Key Takeaways

### 15.1 For NLP practitioners

1. **Start with TF-IDF + LR or SVM.** It takes minutes, it is
   interpretable, and it gives you a floor. If your transformer does
   not beat it by at least a few points, something is wrong.
2. **Pick your pre-trained model to match your domain.** Domain-
   specific models give huge returns in low-data settings and non-
   trivial returns even in data-rich ones.
3. **Fine-tuning is not one-size-fits-all.** Layer-wise learning-rate
   decay, gradual unfreezing, and label smoothing are three
   independent knobs that together can make a general-purpose BERT
   competitive with a domain-specific one.
4. **Multi-task learning is cheap insurance.** On related tasks with
   the same input type, training jointly almost always helps a
   little — and occasionally helps a lot.
5. **Always report macro-F1 alongside accuracy.** Imbalanced test
   sets can make a do-nothing classifier look competitive on
   accuracy.

### 15.2 For future work

1. **Larger base models.** `bert-large-uncased` or
   `roberta-large` would likely close the remaining stance-vs-
   sentiment gap, at the cost of four times the memory.
2. **Data augmentation for hawkish/dovish.** The hawkish class is
   the hardest one in stance; targeted paraphrasing or back-
   translation might help.
3. **Calibration.** None of our models are well calibrated — the
   multi-task model often returns 0.99+ confidence on wrong answers.
   Temperature scaling or Platt scaling on the validation set is a
   cheap fix that does not change the argmax.
4. **Chain-of-thought stance.** Stance often requires implicit
   reasoning the surface text does not contain. An LLM prompted to
   reason about monetary policy first and then classify might do
   better than any of the models here.
5. **Cross-lingual adaptation.** The Vietnamese version of this
   document (`DOC_VI.md`) exists because one author is Vietnamese-
   speaking; extending the classifier itself to Vietnamese financial
   text is a natural follow-up.

---

## 16. Complete Code Reference

This section lists every source file and its key public functions
with a one-to-three-line explanation. For full signatures and
docstrings, see the source directly.

### 16.1 `config.py`

All project hyperparameters, device selection, dataset identifiers,
label mappings, and model identifiers.

- `DEVICE` — `torch.device` chosen among MPS / CUDA / CPU at import
  time.
- `SENTIMENT_LABELS`, `STANCE_LABELS` — the canonical label-order
  lists; both are `[lab_0, lab_1, lab_2]`.
- `FINBERT_MODEL`, `BERT_BASE_MODEL`, `ROBERTA_BASE_MODEL` —
  Hugging Face model identifiers.
- Hyperparameter constants (`MAX_SEQ_LENGTH`, `BATCH_SIZE`,
  `LEARNING_RATE`, `WEIGHT_DECAY`, `FINETUNE_EPOCHS`,
  `MULTITASK_EPOCHS`, `FEW_SHOT_K`, `WARMUP_RATIO`, `SEED`,
  `TEST_SIZE`, `VAL_SIZE`).

### 16.2 `src/data_loader.py`

- `load_financial_phrasebank()` — loads the Parquet version of FPB
  (allagree), merges all Hub splits, stratified-splits into train/
  val/test with `SEED=42`, prints statistics.
- `load_fomc_dataset()` — same pipeline for FOMC; falls back to a
  local CSV if the Hub is unreachable.
- `_process_fomc_df(df)` — shared logic for FOMC: rename `sentence`
  to `text`, drop NaNs, normalise string labels to ints, stratify-
  split, wrap in a `DatasetDict`.
- `get_few_shot_subset(dataset_split, k=FEW_SHOT_K)` — samples
  exactly `k` examples per class with `random_state=SEED`.
- `compute_class_weights(dataset_split, num_classes=3)` — returns
  `[N / (C × n_c) for c in range(C)]` for weighted cross-entropy.
- `_print_split_stats(name, splits, label_names)` — prints the
  size-and-class-distribution tables used in Section 5.

### 16.3 `src/baseline.py`

- `build_baseline_pipeline()` — TF-IDF bigrams + balanced
  `LogisticRegression`.
- `train_and_evaluate_baseline(train, test, label_names, task)` —
  fits the baseline pipeline, computes metrics, writes JSON + PNG,
  returns `(metrics, pipeline)`.
- `build_tfidf_svm_pipeline()` — TF-IDF bigrams + `LinearSVC(C=1.0)`.
- `build_tfidf_trigram_lr_pipeline()` — TF-IDF 1-3-grams
  (`max_features=80_000, min_df=2`) + `LogisticRegression`.
- `run_alternative_baselines(fomc, fpb)` — runs the SVM and
  trigram-LR pipelines on both tasks and returns a dict of
  metrics.

### 16.4 `src/lexicon.py`

- `LM_POSITIVE`, `LM_NEGATIVE`, `LM_UNCERTAINTY`, `HAWKISH_WORDS`,
  `DOVISH_WORDS` — the five curated word lists.
- `_tokenize(text)` — lowercase alphabetic-run tokeniser.
- `extract_lexicon_features(texts)` — returns an `(n, 8)` numpy
  array of the eight features defined in Section 6.1.
- `lexicon_rule_based(test, label_names, task_name)` — pure rule-
  based classifier using the `net_sentiment` band for sentiment
  and hawkish-vs-dovish counts for stance.
- `lexicon_plus_tfidf(train, test, label_names, task_name)` —
  TF-IDF + `StandardScaler`-scaled lexicon features stacked with
  `scipy.sparse.hstack`, then a balanced `LogisticRegression`.
- `run_lexicon_experiments(fomc, fpb)` — runs both the rule-based
  and the TF-IDF-hybrid lexicon experiments on both tasks.

### 16.5 `src/pretrained_eval.py`

- `evaluate_finbert_native(test_split, task_name)` — zero-shot
  FinBERT via `transformers.pipeline` on CPU; direct mapping for
  sentiment, proxy mapping (positive→hawkish, negative→dovish) for
  stance.
- `FewShotClassifier(hidden_size, num_labels)` — a
  `Dropout(0.1) + Linear(hidden, num_labels)` head used for the
  linear probe.
- `_encode_texts(tokenizer, model, texts, device)` — encodes a list
  of texts into `[CLS]` embeddings using a frozen model.
- `evaluate_few_shot(model_name, train, test, label_names,
  task_name, k=FEW_SHOT_K)` — 16-shot linear probe: sample
  `k`-per-class training, encode to frozen `[CLS]` embeddings,
  train a linear classifier for 200 epochs, evaluate on test.
- `run_all_pretrained_evaluations(fomc, fpb)` — runs zero-shot
  FinBERT + few-shot probes for all three encoders on both tasks.

### 16.6 `src/finetune_fineBert.py`

- `TextClassificationDataset(texts, labels, tokenizer,
  max_length)` — PyTorch `Dataset` that tokenises on-the-fly,
  padding to `MAX_SEQ_LENGTH`.
- `finetune_finbert(train, val, test, label_names, task_name,
  use_weighted_loss=True)` — loads FinBERT with a fresh head,
  trains for `FINETUNE_EPOCHS=5` with AdamW + linear warmup,
  optionally weighted CE, keeps the best-val-F1 checkpoint,
  evaluates on test, saves under `models/finbert_{task}/`.
- `_evaluate_model(model, loader, label_names, device)` — helper
  that runs the model on a dataloader and returns metrics.
- `_get_predictions(model, loader, device)` — returns
  `(y_true, y_pred)` lists from a dataloader.

### 16.7 `src/finetune_bert.py`

- `LLRD_BASE_LR`, `LLRD_DECAY`, `LABEL_SMOOTHING`, `BERT_EPOCHS`,
  `NUM_BERT_LAYERS` — the LLRD-specific hyperparameters.
- `TextClassificationDataset` — identical in intent to the one in
  `finetune_fineBert.py`.
- `_build_layer_groups(model)` — partitions the model into 14
  ordered groups (head → embeddings) and assigns per-group
  learning rates of `LLRD_BASE_LR × 0.9^depth`.
- `_build_optimizer(model, epoch)` — builds the AdamW optimiser
  for the current epoch, activating the first `min(epoch, 14)`
  groups and freezing the rest; excludes the head from weight
  decay.
- `_train_one_epoch(model, loader, criterion, optimizer, device)` —
  one epoch of training with gradient clipping at `max_norm=1.0`.
- `_get_predictions(model, loader, device)` — inference helper.
- `finetune_bert_llrd(train, val, test, label_names, task_name)` —
  main entry point: BERT-base fine-tuning with LLRD + gradual
  unfreezing, label-smoothed (and, for stance, class-weighted)
  CE, 10 epochs, best-val-F1 checkpoint, saves under
  `models/bert_llrd_{task}/`.

### 16.8 `src/multitask.py`

- `MultiTaskFinBERT(num_stance_labels=3, num_sentiment_labels=3,
  dropout=0.1)` — shared FinBERT encoder with two linear heads;
  `forward(input_ids, attention_mask, task)` picks the head.
- `train_multitask(fomc_splits, fpb_splits)` — alternating-batch
  training for `MULTITASK_EPOCHS=8`, weighted CE on stance, plain
  CE on sentiment, best-avg-macro-F1 checkpoint, evaluates on both
  test sets, writes results, saves under `models/multitask_finbert/`.
- `_train_step(model, batch, criterion, optimizer, scheduler,
  device, task)` — single training step on one batch for one
  task.
- `_evaluate_multitask(model, loader, label_names, device, task)` —
  evaluates the multi-task model on one task.
- `_get_multitask_predictions(model, loader, device, task)` —
  returns `(y_true, y_pred)` for the selected task.

### 16.9 `src/evaluate.py`

- `compute_metrics(y_true, y_pred, label_names)` — accuracy, macro-
  F1, per-class F1, and a text classification report.
- `print_classification_report(metrics, model_name, task_name)` —
  pretty console output.
- `plot_confusion_matrix(y_true, y_pred, label_names, model_name,
  task_name, save_dir=None)` — seaborn heatmap saved to a PNG.
- `error_analysis(texts, y_true, y_pred, label_names, top_n=20)` —
  returns the longest misclassified examples and a dict of error-
  type counts.
- `save_results(results_dict, filename)` — dumps a metrics dict to
  JSON under `results/`.

### 16.10 `run_experiments.py`

- `step1_load_data()` — calls the two loaders from
  `src/data_loader.py`.
- `step2_baseline(fomc, fpb)` — runs TF-IDF + LR, TF-IDF + SVM, and
  TF-IDF trigrams + LR on both tasks.
- `step2b_lexicon(fomc, fpb)` — runs the LM rule-based and
  TF-IDF + lexicon hybrid experiments.
- `step3_pretrained(fomc, fpb)` — zero-shot FinBERT + 16-shot linear
  probes for all three encoders.
- `step4_finetune(fomc, fpb)` — single-task FinBERT fine-tuning on
  both datasets, with weighted CE on stance and plain CE on
  sentiment.
- `step5_multitask(fomc, fpb)` — multi-task FinBERT training.
- `step6_finetune_bert_llrd(fomc, fpb)` — BERT-base LLRD + gradual
  unfreezing on both tasks.
- `print_summary(all_results)` — prints the consolidated results
  table.
- `main()` — argparse entry point with `--step N` (0 = all).

### 16.11 `cli.py`

- `load_multitask_model()` — loads
  `models/multitask_finbert/model.pt` into a `MultiTaskFinBERT`
  instance.
- `load_finetune_models()` — loads the two single-task FinBERT
  checkpoints from `models/finbert_stance/` and
  `models/finbert_sentiment/`.
- `predict_multitask(text, model, tokenizer)` — returns both
  stance and sentiment predictions with softmax probabilities.
- `predict_finetune(text, models)` — same but using separate
  fine-tuned models.
- `format_prediction(results)` — renders the result dict as a
  console-friendly string with bar chart.
- `main()` — argparse interface for interactive / `--text` /
  `--file` / `--model {multitask,finetune}` modes.

### 16.12 `demo.py`

- `load_model()` — loads the multi-task checkpoint, raises a
  helpful error if it is missing.
- `predict(text, model, tokenizer)` — returns two dicts of
  `{label: confidence}` suitable for Gradio's `Label` component.
- `create_demo()` — builds the Gradio Blocks UI with a textbox,
  classify button, two `Label` outputs, and eight pre-filled
  examples.
- Bottom-of-file launcher — `demo.launch(share=False,
  server_name="0.0.0.0", server_port=7860)`.

---

End of document.
