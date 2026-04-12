"""
ChildFocus — Algorithm Evaluation Suite
ml_training/scripts/evaluate_algorithms.py

PURPOSE:
  Produces the complete algorithm comparison required for thesis Chapter 5.
  Runs 5 classifiers, Stratified 5-Fold CV, holdout evaluation, bi-decision
  accuracy, per-class metrics, and confusion matrices.

HOW TO RUN:
  cd ml_training/scripts
  py evaluate_algorithms.py

OUTPUT FILES (saved to ml_training/outputs/):
  algorithm_comparison_results.json  ← loaded by the HTML dashboard
  confusion_matrices/                ← PNG confusion matrices per algorithm
"""

import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless — no display required
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, make_scorer
)
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ─── PATHS ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
OUTPUTS_DIR = SCRIPT_DIR.parent / "outputs"
CM_DIR      = OUTPUTS_DIR / "confusion_matrices"
OUTPUTS_DIR.mkdir(exist_ok=True)
CM_DIR.mkdir(exist_ok=True)

# The three possible data source configs — script tries in order
DATA_SOURCES = [
    # Config 1: pre-split train/test CSVs (your primary setup)
    {
        "train": SCRIPT_DIR / "train_490.csv",
        "test":  SCRIPT_DIR / "test_210.csv",
        "type":  "split",
    },
    # Config 2: processed clean dataset (fallback)
    {
        "full": SCRIPT_DIR / "data" / "processed" / "metadata_labeled.csv",
        "type": "full",
    },
    # Config 3: raw labeled dataset (second fallback)
    {
        "full": SCRIPT_DIR / "data" / "raw" / "final_700_dataset.csv",
        "type": "full",
    },
]

# OIR class order (controls matrix rows/cols ordering)
CLASS_ORDER = ["Educational", "Neutral", "Overstimulating"]

# ─── ALGORITHMS ───────────────────────────────────────────────────────────────
ALGORITHMS = {
    "Complement NB": {
        "model": ComplementNB(alpha=1.0),
        "short": "CNB",
        "color": "#4FC3F7",   # light blue — thesis primary
    },
    "Multinomial NB": {
        "model": MultinomialNB(alpha=1.0),
        "short": "MNB",
        "color": "#81C784",   # green
    },
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                                     random_state=42),
        "short": "LR",
        "color": "#FFB74D",   # orange
    },
    "Linear SVM": {
        "model": LinearSVC(C=1.0, max_iter=2000, random_state=42),
        "short": "SVM",
        "color": "#CE93D8",   # purple
    },
    "Random Forest": {
        "model": RandomForestClassifier(n_estimators=100, random_state=42,
                                        n_jobs=-1),
        "short": "RF",
        "color": "#EF9A9A",   # red
    },
}


# ─── TEXT PREPROCESSING ───────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    """Lowercase + basic clean. Mirrors naive_bayes.py's _clean_text()."""
    import re
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_text_column(df: pd.DataFrame) -> pd.Series:
    """Combine title + tags + description into one text field."""
    def combine(row):
        title = str(row.get("title", "") or "")
        tags  = str(row.get("tags", "") or "")
        desc  = str(row.get("description", "") or "")[:300]
        return preprocess(f"{title} {tags} {desc}")
    return df.apply(combine, axis=1)


def infer_label_column(df: pd.DataFrame) -> str:
    """Detect which column holds the OIR label."""
    for col in ["label", "category", "oir_label", "class"]:
        if col in df.columns:
            return col
    raise ValueError(
        f"Cannot find label column. Available columns: {list(df.columns)}"
    )


# ─── DATA LOADING ─────────────────────────────────────────────────────────────
def load_data():
    """
    Try each DATA_SOURCES config in order.
    Returns (X_train, X_test, y_train, y_test, X_all, y_all, le)
    where X values are raw text Series and y values are integer-encoded labels.
    """
    le = LabelEncoder()
    le.fit(CLASS_ORDER)

    def normalise_labels(series):
        """Map any label variant to the canonical CLASS_ORDER string."""
        mapping = {
            "educational": "Educational",
            "neutral":     "Neutral",
            "overstimulating": "Overstimulating",
            "overstimulation": "Overstimulating",
        }
        return series.str.strip().str.lower().map(
            lambda x: mapping.get(x, x)
        )

    for config in DATA_SOURCES:
        try:
            if config["type"] == "split":
                if not config["train"].exists() or not config["test"].exists():
                    continue
                print(f"[DATA] Loading from pre-split CSVs...")
                train_df = pd.read_csv(config["train"])
                test_df  = pd.read_csv(config["test"])
                lc       = infer_label_column(train_df)
                train_df[lc] = normalise_labels(train_df[lc].astype(str))
                test_df[lc]  = normalise_labels(test_df[lc].astype(str))
                # Keep only valid labels
                train_df = train_df[train_df[lc].isin(CLASS_ORDER)]
                test_df  = test_df[test_df[lc].isin(CLASS_ORDER)]
                X_train = build_text_column(train_df)
                X_test  = build_text_column(test_df)
                y_train = le.transform(train_df[lc])
                y_test  = le.transform(test_df[lc])
                X_all   = pd.concat([X_train, X_test], ignore_index=True)
                y_all   = np.concatenate([y_train, y_test])
                print(f"[DATA] Train: {len(X_train)} | Test: {len(X_test)} | "
                      f"Classes: {dict(zip(*np.unique(y_all, return_counts=True)))}")
                return X_train, X_test, y_train, y_test, X_all, y_all, le

            elif config["type"] == "full":
                if not config["full"].exists():
                    continue
                print(f"[DATA] Loading full dataset: {config['full'].name}...")
                df = pd.read_csv(config["full"])
                lc = infer_label_column(df)
                df[lc] = normalise_labels(df[lc].astype(str))
                df = df[df[lc].isin(CLASS_ORDER)]
                X_all = build_text_column(df)
                y_all = le.transform(df[lc])
                # Reproduce the 70/30 stratified split
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_all, y_all, test_size=0.30, stratify=y_all, random_state=42
                )
                print(f"[DATA] Stratified 70/30 split — "
                      f"Train: {len(X_train)} | Test: {len(X_test)}")
                return X_train, X_test, y_train, y_test, X_all, y_all, le
        except Exception as e:
            print(f"[DATA] Source failed: {e}")
            continue

    raise FileNotFoundError(
        "No usable dataset found. Expected one of:\n"
        "  ml_training/scripts/train_490.csv + test_210.csv\n"
        "  ml_training/scripts/data/processed/metadata_labeled.csv\n"
        "  ml_training/scripts/data/raw/final_700_dataset.csv"
    )


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def bi_decision_accuracy(y_true, y_pred, le):
    """
    Collapse Educational + Neutral → Safe (0) and Overstimulating → Over (1).
    Returns float accuracy on the binary reframing.
    """
    over_idx = list(le.classes_).index("Overstimulating")
    y_true_bi = (y_true == over_idx).astype(int)
    y_pred_bi = (y_pred == over_idx).astype(int)
    return accuracy_score(y_true_bi, y_pred_bi)


def overstimulating_recall(y_true, y_pred, le):
    """Recall for the Overstimulating class only — the thesis safety metric."""
    over_idx = list(le.classes_).index("Overstimulating")
    labels = [over_idx]
    return recall_score(y_true, y_pred, labels=labels, average="macro",
                        zero_division=0)


# ─── CONFUSION MATRIX CHART ───────────────────────────────────────────────────
def save_confusion_matrix(cm, class_names, algo_name, subset_label,
                          accuracy, filename):
    """Save a publication-quality confusion matrix PNG."""
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#0D1B2A")
    ax.set_facecolor("#0D1B2A")

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5, linecolor="#1E3A5F",
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    ax.set_xlabel("Predicted Label", color="white", fontsize=11, labelpad=10)
    ax.set_ylabel("True Label", color="white", fontsize=11, labelpad=10)
    ax.set_title(
        f"{algo_name} — {subset_label}\nAccuracy: {accuracy:.2%}",
        color="white", fontsize=12, pad=12
    )
    ax.tick_params(colors="white", labelsize=9)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", color="white")
    plt.setp(ax.get_yticklabels(), rotation=0, color="white")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [CM] Saved → {filename.name}")


# ─── MAIN EVALUATION ──────────────────────────────────────────────────────────
def run_evaluation():
    print("\n" + "=" * 60)
    print("  ChildFocus — Algorithm Evaluation Suite")
    print("=" * 60 + "\n")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, X_all, y_all, le = load_data()
    class_names = list(le.classes_)   # sorted: Educational, Neutral, Over...

    # ── 2. Fit shared TF-IDF vectorizer on training data ─────────────────────
    # We fit on X_train only (no data leakage) for holdout evaluation.
    # CV folds get their own vectorizer per fold via Pipeline.
    print("[STEP] Fitting shared TF-IDF vectorizer on training data...")
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf  = tfidf.transform(X_test)
    X_all_tfidf   = tfidf.transform(X_all)  # for CV

    # ── 3. Stratified 5-Fold CV setup ─────────────────────────────────────────
    SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    print(f"\n[STEP] Running evaluation for {len(ALGORITHMS)} algorithms...\n")

    for algo_name, algo_info in ALGORITHMS.items():
        model = algo_info["model"]
        short = algo_info["short"]
        print(f"  [{short}] {algo_name}")

        # ── 3a. Holdout evaluation (70/30 split) ─────────────────────────────
        model.fit(X_train_tfidf, y_train)
        y_pred_test  = model.predict(X_test_tfidf)
        y_pred_train = model.predict(X_train_tfidf)

        holdout_acc  = accuracy_score(y_test, y_pred_test)
        train_acc    = accuracy_score(y_train, y_pred_train)
        overfit_gap  = train_acc - holdout_acc
        bi_acc       = bi_decision_accuracy(y_test, y_pred_test, le)
        over_recall  = overstimulating_recall(y_test, y_pred_test, le)
        cr           = classification_report(
            y_test, y_pred_test,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )

        print(f"       Holdout acc: {holdout_acc:.4f}  "
              f"Train: {train_acc:.4f}  "
              f"Gap: {overfit_gap:.4f}  "
              f"Bi-dec: {bi_acc:.4f}")

        # Save holdout confusion matrix
        cm_holdout = confusion_matrix(y_test, y_pred_test,
                                      labels=list(range(len(class_names))))
        save_confusion_matrix(
            cm_holdout, class_names, algo_name,
            "Holdout (30%)", holdout_acc,
            CM_DIR / f"cm_holdout_{short.lower()}.png"
        )

        # ── 3b. Stratified 5-Fold Cross-Validation ────────────────────────────
        # Use Pipeline so vectorizer is refitted inside each fold (no leakage)
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=2,
            )),
            ("clf", algo_info["model"].__class__(
                **{k: v for k, v in
                   algo_info["model"].get_params().items()}
            )),
        ])

        cv_scores = []
        cv_bi_scores = []
        cv_over_recall_scores = []
        cv_cms = []

        all_texts = list(X_all)   # raw text list for Pipeline input

        for fold_i, (train_idx, val_idx) in enumerate(SKF.split(all_texts, y_all)):
            X_fold_train = [all_texts[i] for i in train_idx]
            X_fold_val   = [all_texts[i] for i in val_idx]
            y_fold_train = y_all[train_idx]
            y_fold_val   = y_all[val_idx]

            pipeline.fit(X_fold_train, y_fold_train)
            y_fold_pred = pipeline.predict(X_fold_val)

            fold_acc      = accuracy_score(y_fold_val, y_fold_pred)
            fold_bi       = bi_decision_accuracy(y_fold_val, y_fold_pred, le)
            fold_over_rec = overstimulating_recall(y_fold_val, y_fold_pred, le)
            fold_cm       = confusion_matrix(y_fold_val, y_fold_pred,
                                             labels=list(range(len(class_names))))

            cv_scores.append(fold_acc)
            cv_bi_scores.append(fold_bi)
            cv_over_recall_scores.append(fold_over_rec)
            cv_cms.append(fold_cm.tolist())

        cv_mean  = float(np.mean(cv_scores))
        cv_std   = float(np.std(cv_scores))
        cv_bi    = float(np.mean(cv_bi_scores))
        cv_over  = float(np.mean(cv_over_recall_scores))

        # Aggregate confusion matrix across all 5 folds
        agg_cm = np.sum([np.array(c) for c in cv_cms], axis=0)
        save_confusion_matrix(
            agg_cm, class_names, algo_name,
            "5-Fold CV (Aggregated)", cv_mean,
            CM_DIR / f"cm_cv_{short.lower()}.png"
        )

        print(f"       CV acc: {cv_mean:.4f} ± {cv_std:.4f}  "
              f"CV bi-dec: {cv_bi:.4f}  "
              f"CV over-recall: {cv_over:.4f}")

        # Per-class metrics from holdout
        per_class = {}
        for cls in class_names:
            per_class[cls] = {
                "precision": round(cr[cls]["precision"], 4),
                "recall":    round(cr[cls]["recall"],    4),
                "f1":        round(cr[cls]["f1-score"],  4),
                "support":   int(cr[cls]["support"]),
            }

        results[algo_name] = {
            "short":           short,
            "color":           algo_info["color"],
            # Holdout metrics
            "holdout_acc":     round(holdout_acc,  4),
            "train_acc":       round(train_acc,    4),
            "overfit_gap":     round(overfit_gap,  4),
            "holdout_bi_acc":  round(bi_acc,       4),
            "over_recall":     round(over_recall,  4),
            # CV metrics
            "cv_mean":         round(cv_mean,  4),
            "cv_std":          round(cv_std,   4),
            "cv_bi_acc":       round(cv_bi,    4),
            "cv_over_recall":  round(cv_over,  4),
            "cv_fold_scores":  [round(s, 4) for s in cv_scores],
            "cv_fold_bi":      [round(s, 4) for s in cv_bi_scores],
            # Confusion matrix (holdout, as flat list for JSON)
            "cm_holdout":      cm_holdout.tolist(),
            "cm_cv_agg":       agg_cm.tolist(),
            # Per-class breakdown
            "per_class":       per_class,
            # Macro averages
            "macro_f1":        round(cr["macro avg"]["f1-score"], 4),
            "macro_precision": round(cr["macro avg"]["precision"], 4),
            "macro_recall":    round(cr["macro avg"]["recall"],    4),
        }

        print()

    # ── 4. Dataset summary ────────────────────────────────────────────────────
    label_dist_train = dict(zip(
        class_names,
        [int(np.sum(y_train == i)) for i in range(len(class_names))]
    ))
    label_dist_test = dict(zip(
        class_names,
        [int(np.sum(y_test == i)) for i in range(len(class_names))]
    ))
    label_dist_all = dict(zip(
        class_names,
        [int(np.sum(y_all == i)) for i in range(len(class_names))]
    ))

    # ── 5. Build final output JSON ────────────────────────────────────────────
    output = {
        "meta": {
            "generated_at":   datetime.now().isoformat(),
            "n_train":        int(len(y_train)),
            "n_test":         int(len(y_test)),
            "n_total":        int(len(y_all)),
            "n_folds":        5,
            "class_order":    class_names,
            "label_dist_train": label_dist_train,
            "label_dist_test":  label_dist_test,
            "label_dist_all":   label_dist_all,
        },
        "algorithms": results,
    }

    out_path = OUTPUTS_DIR / "algorithm_comparison_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[DONE] Results saved → {out_path}")

    # ── 6. Print summary table ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  SUMMARY TABLE (Thesis Chapter 5)")
    print("=" * 80)
    header = (f"{'Algorithm':<25} {'HoldoutAcc':>10} {'CV Mean':>9} "
              f"{'CV ±':>7} {'Bi-Dec':>8} {'Ovfit Gap':>10}")
    print(header)
    print("-" * 80)

    # Rank by CV mean (ties broken by bi-decision)
    ranked = sorted(results.items(),
                    key=lambda x: (x[1]["cv_mean"], x[1]["cv_bi_acc"]))
    for name, r in ranked:
        print(f"  {name:<23} {r['holdout_acc']:>10.2%} "
              f"{r['cv_mean']:>9.2%} "
              f"±{r['cv_std']:>5.2%} "
              f"{r['holdout_bi_acc']:>8.2%} "
              f"{r['overfit_gap']:>10.2%}")
    print("=" * 80)

    print(f"\n  Confusion matrices (PNG) → {CM_DIR}\n")
    return output


if __name__ == "__main__":
    run_evaluation()
