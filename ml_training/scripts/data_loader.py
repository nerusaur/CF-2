"""
ChildFocus — Shared Data Loader & Evaluation Utilities
ml_training/scripts/data_loader.py

Imported by every individual model file and by evaluate_crossfold.py.
Keeps the data pipeline in one place so all models use identical inputs.
"""

import os
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.model_selection import StratifiedKFold

# ── Constants ──────────────────────────────────────────────────────────────────
CLASS_ORDER   = ["Educational", "Neutral", "Overstimulating"]
N_FOLDS       = 5
RANDOM_STATE  = 42

SCRIPT_DIR  = Path(__file__).parent
OUTPUTS_DIR = SCRIPT_DIR.parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# ── TF-IDF shared config ───────────────────────────────────────────────────────
TFIDF_PARAMS = dict(
    max_features = 10_000,
    ngram_range  = (1, 2),
    sublinear_tf = True,
    min_df       = 2,
)

# ── Data source search order ───────────────────────────────────────────────────
DATA_SOURCES = [
    {"type": "split",
     "train": SCRIPT_DIR / "train_490.csv",
     "test":  SCRIPT_DIR / "test_210.csv"},
    {"type": "full",
     "full":  SCRIPT_DIR / "data" / "processed" / "metadata_labeled.csv"},
    {"type": "full",
     "full":  SCRIPT_DIR / "data" / "raw" / "final_700_dataset.csv"},
]


# ─────────────────────────────────────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Mirrors naive_bayes.py _clean_text() exactly.
    Lowercase, strip URLs, keep only alphanumeric + whitespace.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_text_column(df: pd.DataFrame) -> pd.Series:
    """
    Combine title + tags + first 300 chars of description into one text field.
    Same logic used by the live classify.py pipeline.
    """
    def combine(row):
        title = str(row.get("title", "") or "")
        tags  = str(row.get("tags",  "") or "")
        desc  = str(row.get("description", "") or "")[:300]
        return clean_text(f"{title} {tags} {desc}")
    return df.apply(combine, axis=1)


def normalise_labels(series: pd.Series) -> pd.Series:
    """Map any label variant to the canonical CLASS_ORDER string."""
    mapping = {
        "educational":     "Educational",
        "neutral":         "Neutral",
        "overstimulating": "Overstimulating",
        "overstimulation": "Overstimulating",
    }
    return series.astype(str).str.strip().str.lower().map(
        lambda x: mapping.get(x, x)
    )


def infer_label_column(df: pd.DataFrame) -> str:
    """Auto-detect which column holds the OIR label."""
    for col in ["label", "category", "oir_label", "class"]:
        if col in df.columns:
            return col
    raise ValueError(
        f"Label column not found. Columns present: {list(df.columns)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    """
    Try DATA_SOURCES in order, return a consistent data bundle.

    Returns
    -------
    dict with keys:
      X_train   pd.Series  raw text — training split
      X_test    pd.Series  raw text — test split
      y_train   np.ndarray integer-encoded labels — training
      y_test    np.ndarray integer-encoded labels — test
      X_all     pd.Series  full dataset (train + test)
      y_all     np.ndarray full labels
      le        LabelEncoder fitted on CLASS_ORDER
    """
    le = LabelEncoder()
    le.fit(CLASS_ORDER)

    for cfg in DATA_SOURCES:
        try:
            if cfg["type"] == "split":
                if not cfg["train"].exists() or not cfg["test"].exists():
                    continue
                print(f"[DATA] Using pre-split CSVs  "
                      f"({cfg['train'].name} + {cfg['test'].name})")
                tr = pd.read_csv(cfg["train"])
                te = pd.read_csv(cfg["test"])
                lc = infer_label_column(tr)
                tr[lc] = normalise_labels(tr[lc])
                te[lc] = normalise_labels(te[lc])
                tr = tr[tr[lc].isin(CLASS_ORDER)]
                te = te[te[lc].isin(CLASS_ORDER)]
                X_train = build_text_column(tr)
                X_test  = build_text_column(te)
                y_train = le.transform(tr[lc])
                y_test  = le.transform(te[lc])

            elif cfg["type"] == "full":
                if not cfg["full"].exists():
                    continue
                print(f"[DATA] Using full dataset  ({cfg['full'].name})")
                df = pd.read_csv(cfg["full"])
                lc = infer_label_column(df)
                df[lc] = normalise_labels(df[lc])
                df = df[df[lc].isin(CLASS_ORDER)]
                X_all_raw = build_text_column(df)
                y_all_raw = le.transform(df[lc])
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_all_raw, y_all_raw,
                    test_size=0.30, stratify=y_all_raw,
                    random_state=RANDOM_STATE,
                )

            else:
                continue

            X_all = pd.concat(
                [X_train.reset_index(drop=True),
                 X_test.reset_index(drop=True)],
                ignore_index=True,
            )
            y_all = np.concatenate([y_train, y_test])

            print(f"[DATA] Train={len(y_train)}  Test={len(y_test)}  "
                  f"Total={len(y_all)}")
            print(f"[DATA] Class dist (all): "
                  + "  ".join(
                      f"{cls}={int(np.sum(y_all==i))}"
                      for i, cls in enumerate(le.classes_)
                  ))
            return dict(
                X_train=X_train, X_test=X_test,
                y_train=y_train, y_test=y_test,
                X_all=X_all, y_all=y_all, le=le,
            )

        except Exception as e:
            print(f"[DATA] Source failed: {e}")
            continue

    raise FileNotFoundError(
        "No usable dataset found.\n"
        "Expected one of:\n"
        "  ml_training/scripts/train_490.csv + test_210.csv\n"
        "  ml_training/scripts/data/processed/metadata_labeled.csv"
    )


# ─────────────────────────────────────────────────────────────────────────────
# VECTORISATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def fit_vectorizer(X_train: pd.Series) -> TfidfVectorizer:
    """
    Fit a TF-IDF vectorizer on training data only.
    Fitted on X_train → transforms both train and test (no data leakage).
    """
    vec = TfidfVectorizer(**TFIDF_PARAMS)
    vec.fit(X_train)
    return vec


# ─────────────────────────────────────────────────────────────────────────────
# METRIC HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def bi_decision_accuracy(y_true, y_pred, le) -> float:
    """
    Collapse Educational + Neutral → Safe (0), Overstimulating → Over (1).
    Returns binary accuracy — the thesis operational metric.
    """
    over_idx  = list(le.classes_).index("Overstimulating")
    y_true_bi = (np.array(y_true) == over_idx).astype(int)
    y_pred_bi = (np.array(y_pred) == over_idx).astype(int)
    return float(accuracy_score(y_true_bi, y_pred_bi))


def overstimulating_recall(y_true, y_pred, le) -> float:
    """
    Recall for Overstimulating class only.
    The child-safety primary metric: how many harmful videos did we catch?
    """
    over_idx = list(le.classes_).index("Overstimulating")
    return float(recall_score(
        y_true, y_pred,
        labels=[over_idx], average="macro",
        zero_division=0,
    ))


def full_metrics(y_true, y_pred, le) -> dict:
    """
    Compute all metrics needed for thesis tables.
    Returns a flat dict ready for JSON serialisation.
    """
    classes = list(le.classes_)
    cr      = classification_report(
        y_true, y_pred,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        cls: {
            "precision": round(cr[cls]["precision"], 4),
            "recall":    round(cr[cls]["recall"],    4),
            "f1":        round(cr[cls]["f1-score"],  4),
            "support":   int(cr[cls]["support"]),
        }
        for cls in classes
    }
    cm = confusion_matrix(y_true, y_pred,
                          labels=list(range(len(classes))))
    return dict(
        accuracy         = round(float(accuracy_score(y_true, y_pred)), 4),
        bi_decision_acc  = round(bi_decision_accuracy(y_true, y_pred, le), 4),
        over_recall      = round(overstimulating_recall(y_true, y_pred, le), 4),
        macro_precision  = round(float(cr["macro avg"]["precision"]), 4),
        macro_recall     = round(float(cr["macro avg"]["recall"]),    4),
        macro_f1         = round(float(cr["macro avg"]["f1-score"]),  4),
        per_class        = per_class,
        confusion_matrix = cm.tolist(),
    )


def print_metrics_table(name: str, holdout: dict, cv: dict):
    """Pretty-print a summary block for the terminal."""
    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")
    print(f"  {'Metric':<28} {'Holdout':>10}  {'CV Mean':>10}  {'CV ±':>8}")
    print(f"  {'─'*56}")
    rows = [
        ("Accuracy",             holdout['accuracy'],       cv['mean'],    cv['std']),
        ("Bi-Decision Accuracy", holdout['bi_decision_acc'],cv['bi_mean'], cv['bi_std']),
        ("Over Recall",          holdout['over_recall'],    cv['or_mean'], cv['or_std']),
        ("Macro F1",             holdout['macro_f1'],       cv['f1_mean'], cv['f1_std']),
    ]
    for lbl, h, m, s in rows:
        print(f"  {lbl:<28} {h:>10.4f}  {m:>10.4f}  {s:>8.4f}")
    print(f"{'─'*60}")
