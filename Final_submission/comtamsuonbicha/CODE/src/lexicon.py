import os
import sys
import re
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SEED, DATA_DIR
from src.evaluate import (
    compute_metrics,
    print_classification_report,
    plot_confusion_matrix,
    save_results,
)


# ──────────────────────────────────────────────────────────────────────────────
# Loughran-McDonald word lists
# ──────────────────────────────────────────────────────────────────────────────

LM_POSITIVE = {
    "achieve", "attain", "benefit", "bolster", "boom", "breakthrough", "compliment",
    "constructive", "creative", "delight", "desirable", "diligent", "distinguish",
    "earn", "efficient", "empower", "enable", "encourage", "enhance", "enjoy",
    "enthusiasm", "excellence", "exceptional", "exciting", "exclusive", "favorable",
    "favourable", "gain", "good", "great", "greatest", "guarantee", "highest",
    "honor", "honour", "ideal", "impressive", "improve", "improvement", "incredible",
    "influential", "informative", "ingenuity", "innovate", "innovation", "innovative",
    "insight", "inspire", "integrity", "invent", "leader", "leadership", "lucrative",
    "merit", "notable", "optimal", "optimism", "optimistic", "outperform", "outstanding",
    "overcome", "perfect", "pleasant", "pleasure", "popular", "positive", "proactive",
    "proficiency", "profit", "profitable", "progress", "prominence", "prosper",
    "prosperity", "rebound", "recover", "recovery", "reinforce", "reliability",
    "remarkable", "resilient", "resolve", "reward", "robust", "satisfaction",
    "smooth", "solid", "solution", "stability", "stabilize", "stable", "strength",
    "strengthen", "strong", "succeed", "success", "successful", "superior", "support",
    "surge", "surpass", "sustain", "sustainable", "transform", "tremendous", "triumph",
    "trust", "unmatched", "upgrade", "upturn", "valuable", "versatile", "vibrant",
    "win", "winner", "worthy",
}

LM_NEGATIVE = {
    "abandon", "abolish", "abuse", "accident", "adverse", "allegation", "annul",
    "argue", "attrition", "bad", "bail", "bankrupt", "bankruptcy", "bottleneck",
    "breach", "burden", "catastrophe", "caution", "cease", "challenge", "claim",
    "close", "closure", "collapse", "complain", "complaint", "concern", "condemn",
    "conflict", "confront", "contraction", "controversy", "corrupt", "crisis",
    "critical", "criticism", "cut", "damage", "danger", "debt", "decline", "decrease",
    "default", "defect", "defer", "deficiency", "deficit", "degrade", "delay",
    "delinquent", "demote", "denial", "deny", "deplete", "depreciate", "depress",
    "depression", "deteriorate", "detriment", "difficult", "difficulty", "diminish",
    "disappoint", "disappointment", "disaster", "disclaim", "discontinue", "discount",
    "dispute", "disruption", "dissatisfaction", "dissolve", "distort", "distress",
    "downgrade", "downturn", "drop", "drought", "erode", "erosion", "error", "evict",
    "exacerbate", "excessive", "fail", "failure", "fall", "falter", "fear", "fine",
    "flaw", "flee", "fluctuate", "forbid", "force", "fraud", "hinder", "hostile",
    "hurdle", "idle", "impair", "impairment", "impediment", "impose", "inability",
    "inadequate", "incapable", "incompetent", "ineffective", "inefficiency",
    "inflame", "inflation", "inflationary", "infringe", "injure", "insolvency",
    "instability", "insufficient", "interfere", "interrupt", "investigation",
    "jeopardize", "lack", "lag", "lapse", "layoff", "limit", "liquidate", "litigation",
    "lose", "loss", "losses", "lower", "malfunction", "manipulate", "misappropriate",
    "misconduct", "mislead", "misrepresent", "misstate", "monopoly", "moody",
    "negate", "neglect", "obstacle", "obstruct", "offense", "omission", "oppose",
    "outage", "overdue", "overrun", "overshadow", "panic", "penalty", "peril",
    "pessimism", "pessimistic", "plummet", "poor", "preclude", "pressure",
    "problem", "prohibit", "prosecute", "protest", "questionable", "recession",
    "reckless", "reduce", "reduction", "reject", "relinquish", "reluctance",
    "repossess", "restrain", "restructure", "retaliate", "risk", "risky", "sanction",
    "scandal", "scarcity", "seize", "serious", "setback", "severe", "shortage",
    "shrink", "slippage", "slow", "slowdown", "sluggish", "slump", "stagnant",
    "stagnate", "strain", "stress", "struggle", "subprime", "suffer", "susceptible",
    "suspend", "suspension", "tariff", "tense", "terminate", "threat", "threaten",
    "tighten", "troubled", "turmoil", "unable", "uncertain", "uncertainty",
    "undermine", "underperform", "unfavorable", "unfavourable", "unfortunate",
    "unprofitable", "unresolved", "unstable", "unsuccessful", "violate", "violation",
    "volatile", "volatility", "vulnerable", "warn", "warning", "weak", "weaken",
    "weakness", "worsen", "worst", "worthless", "writedown", "writeoff",
}

LM_UNCERTAINTY = {
    "almost", "ambiguity", "ambiguous", "apparent", "apparently", "appear",
    "approximate", "approximately", "arbitrarily", "assume", "assumption",
    "believe", "conceivable", "conditional", "confuse", "contingency",
    "contingent", "could", "depend", "doubt", "estimate", "exposure", "fluctuate",
    "generally", "hypothetical", "imprecise", "incompleteness", "indefinite",
    "indeterminate", "inexact", "likelihood", "may", "maybe", "might",
    "nearly", "nonassessable", "occasionally", "pending", "perhaps",
    "possibility", "possible", "possibly", "predict", "prediction",
    "preliminary", "presumably", "probabilistic", "probability", "probable",
    "probably", "provisional", "random", "reassess", "recalculate",
    "reconsider", "reexamine", "reinterpret", "revise", "roughly",
    "seem", "seldom", "sometimes", "somewhat", "speculate", "speculation",
    "suggest", "suppose", "tentative", "uncertain", "uncertainty",
    "unclear", "undecided", "undefined", "undetermined", "unforeseeable",
    "unknown", "unlikely", "unpredictable", "unproven", "unquantifiable",
    "unreliable", "unsettled", "unspecified", "unusual", "vague", "variability",
    "variable", "vary",
}

# Hawkish / dovish word lists for monetary policy stance classification
# These capture language typically used in FOMC communications
HAWKISH_WORDS = {
    "hike", "raise", "tighten", "tightening", "inflation", "inflationary",
    "overheating", "overheat", "accelerate", "accelerating", "pressures",
    "upward", "elevated", "hawkish", "restrictive", "restrict", "curb",
    "normalize", "normalization", "tapering", "taper", "reduce",
    "withdrawal", "withdraw", "shrink", "contraction", "vigilant",
    "vigilance", "concern", "concerned", "risk", "risks", "robust",
    "strong", "strength", "strengthen", "surge", "surging", "excessive",
    "above", "target", "overshoot", "higher",
}

DOVISH_WORDS = {
    "cut", "lower", "ease", "easing", "accommodate", "accommodative",
    "accommodation", "stimulus", "stimulate", "support", "supportive",
    "dovish", "patient", "patience", "gradual", "cautious", "caution",
    "monitor", "monitoring", "subdued", "below", "weakness", "weak",
    "weaken", "slowdown", "slowing", "slow", "soft", "soften", "moderate",
    "moderation", "decline", "declining", "recession", "recessionary",
    "downside", "downturn", "slack", "underperform", "underemployment",
    "unemployment", "stagnant", "stagnation", "deflationary", "deflation",
    "disinflation", "expansionary", "expansion", "purchase", "purchasing",
    "inject", "injection", "liquidity", "maintain", "maintaining",
}


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────────────────────

def _tokenize(text):
    """Simple whitespace + punctuation tokeniser, returns lowercased tokens."""
    return re.findall(r'\b[a-z]+\b', text.lower())


def extract_lexicon_features(texts):
    """
    Extract Loughran-McDonald lexicon features from a list of texts.

    For each text, computes:
      - positive_count:    number of LM positive words
      - negative_count:    number of LM negative words
      - uncertainty_count: number of LM uncertainty words
      - hawkish_count:     number of hawkish monetary policy words
      - dovish_count:      number of dovish monetary policy words
      - net_sentiment:     (positive - negative) / total_words
      - net_stance:        (hawkish - dovish) / total_words
      - total_words:       total number of tokens

    Returns a numpy array of shape (n_texts, 8).
    """
    features = []
    for text in texts:
        tokens = _tokenize(text)
        total = max(len(tokens), 1)  # avoid division by zero

        pos_count = sum(1 for t in tokens if t in LM_POSITIVE)
        neg_count = sum(1 for t in tokens if t in LM_NEGATIVE)
        unc_count = sum(1 for t in tokens if t in LM_UNCERTAINTY)
        hawk_count = sum(1 for t in tokens if t in HAWKISH_WORDS)
        dove_count = sum(1 for t in tokens if t in DOVISH_WORDS)

        features.append([
            pos_count,
            neg_count,
            unc_count,
            hawk_count,
            dove_count,
            (pos_count - neg_count) / total,     # net sentiment
            (hawk_count - dove_count) / total,    # net stance
            total,
        ])

    return np.array(features)


# ──────────────────────────────────────────────────────────────────────────────
# Rule-based lexicon classifier
# ──────────────────────────────────────────────────────────────────────────────

def lexicon_rule_based(test_split, label_names, task_name):
    """
    Pure rule-based classification using LM lexicon word counts.

    For sentiment: classify based on net_sentiment (positive - negative)
    For stance:    classify based on hawkish vs dovish word counts
    """
    print(f"\n{'='*60}")
    print(f"  LEXICON RULE-BASED — {task_name}")
    print(f"{'='*60}")

    texts = test_split["text"]
    y_true = test_split["label"]
    features = extract_lexicon_features(texts)

    y_pred = []
    if task_name == "sentiment":
        # Use net sentiment: positive=2, negative=0, neutral=1
        net_sent = features[:, 5]  # net_sentiment column
        for ns in net_sent:
            if ns > 0.02:
                y_pred.append(2)   # positive
            elif ns < -0.02:
                y_pred.append(0)   # negative
            else:
                y_pred.append(1)   # neutral
    else:
        # Use hawkish/dovish counts for stance
        hawk = features[:, 3]
        dove = features[:, 4]
        for h, d in zip(hawk, dove):
            if h > d:
                y_pred.append(1)   # hawkish
            elif d > h:
                y_pred.append(0)   # dovish
            else:
                y_pred.append(2)   # neutral

    metrics = compute_metrics(y_true, y_pred, label_names)
    print_classification_report(metrics, "LM Lexicon (rule-based)", task_name)
    plot_confusion_matrix(y_true, y_pred, label_names, "LM_Lexicon_rules", task_name)
    save_results(
        {"model": "LM Lexicon (rule-based)", "task": task_name, **metrics},
        f"lexicon_rules_{task_name}.json",
    )
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# TF-IDF + Lexicon features classifier
# ──────────────────────────────────────────────────────────────────────────────

def lexicon_plus_tfidf(train_split, test_split, label_names, task_name):
    """
    Combined TF-IDF + LM lexicon features with Logistic Regression.

    This augments the TF-IDF baseline with domain-specific lexicon features,
    demonstrating the value of incorporating structured financial knowledge.
    """
    print(f"\n{'='*60}")
    print(f"  TF-IDF + LM LEXICON FEATURES — {task_name}")
    print(f"{'='*60}")

    train_texts = train_split["text"]
    train_labels = train_split["label"]
    test_texts = test_split["text"]
    test_labels = test_split["label"]

    # Build TF-IDF features
    tfidf = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
    )
    train_tfidf = tfidf.fit_transform(train_texts)
    test_tfidf = tfidf.transform(test_texts)

    # Extract lexicon features
    train_lex = extract_lexicon_features(train_texts)
    test_lex = extract_lexicon_features(test_texts)

    # Scale lexicon features to be comparable with TF-IDF
    scaler = StandardScaler()
    train_lex_scaled = scaler.fit_transform(train_lex)
    test_lex_scaled = scaler.transform(test_lex)

    # Combine TF-IDF + lexicon features
    train_combined = hstack([train_tfidf, csr_matrix(train_lex_scaled)])
    test_combined = hstack([test_tfidf, csr_matrix(test_lex_scaled)])

    # Train classifier
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=SEED,
        solver="lbfgs",
    )
    clf.fit(train_combined, train_labels)
    predictions = clf.predict(test_combined)

    # Evaluate
    metrics = compute_metrics(test_labels, predictions, label_names)
    print_classification_report(metrics, "TF-IDF + LM Lexicon", task_name)
    plot_confusion_matrix(
        test_labels, predictions, label_names, "TFIDF_LM_Lexicon", task_name
    )
    save_results(
        {"model": "TF-IDF + LM Lexicon", "task": task_name, **metrics},
        f"lexicon_tfidf_{task_name}.json",
    )
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Run all lexicon experiments
# ──────────────────────────────────────────────────────────────────────────────

def run_lexicon_experiments(fomc_splits, fpb_splits):
    """Run all lexicon-based experiments on both datasets."""
    from config import STANCE_LABELS, SENTIMENT_LABELS

    results = {}

    # Rule-based lexicon classifier
    results["lexicon_rules_stance"] = lexicon_rule_based(
        fomc_splits["test"], STANCE_LABELS, "stance"
    )
    results["lexicon_rules_sentiment"] = lexicon_rule_based(
        fpb_splits["test"], SENTIMENT_LABELS, "sentiment"
    )

    # TF-IDF + lexicon features
    results["lexicon_tfidf_stance"] = lexicon_plus_tfidf(
        fomc_splits["train"], fomc_splits["test"], STANCE_LABELS, "stance"
    )
    results["lexicon_tfidf_sentiment"] = lexicon_plus_tfidf(
        fpb_splits["train"], fpb_splits["test"], SENTIMENT_LABELS, "sentiment"
    )

    return results
