import os
import torch

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# ─── Device ───────────────────────────────────────────────────────────────────
# Apple Silicon M3 Max uses MPS; falls back to CUDA then CPU.
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# ─── Random seed for reproducibility ─────────────────────────────────────────
SEED = 42

# ─── Dataset identifiers ─────────────────────────────────────────────────────
# Financial PhraseBank: sentences with all-annotator agreement (Parquet version)
FPB_DATASET_NAME = "gtfintechlab/financial_phrasebank_sentences_allagree"
FPB_SUBSET = "5768"  # config name required by this dataset version

# FOMC Hawkish-Dovish dataset (gtfintechlab)
FOMC_DATASET_NAME = "gtfintechlab/fomc_communication"

# ─── Label mappings ───────────────────────────────────────────────────────────
# Sentiment labels (Financial PhraseBank)
SENTIMENT_LABELS = ["negative", "neutral", "positive"]
SENTIMENT_ID2LABEL = {i: l for i, l in enumerate(SENTIMENT_LABELS)}
SENTIMENT_LABEL2ID = {l: i for i, l in enumerate(SENTIMENT_LABELS)}

# Stance labels (FOMC Hawkish-Dovish)
STANCE_LABELS = ["dovish", "hawkish", "neutral"]
STANCE_ID2LABEL = {i: l for i, l in enumerate(STANCE_LABELS)}
STANCE_LABEL2ID = {l: i for i, l in enumerate(STANCE_LABELS)}

# ─── Pre-trained model identifiers ───────────────────────────────────────────
FINBERT_MODEL = "ProsusAI/finbert"
BERT_BASE_MODEL = "bert-base-uncased"
ROBERTA_BASE_MODEL = "roberta-base"

# ─── Training hyperparameters ─────────────────────────────────────────────────
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
FINETUNE_EPOCHS = 5
MULTITASK_EPOCHS = 8
FEW_SHOT_K = 16          # number of examples per class for few-shot
WARMUP_RATIO = 0.1

# ─── Evaluation ───────────────────────────────────────────────────────────────
TEST_SIZE = 0.2           # fraction reserved for testing
VAL_SIZE = 0.1            # fraction reserved for validation (from train portion)
