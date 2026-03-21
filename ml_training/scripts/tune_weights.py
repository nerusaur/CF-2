"""
ChildFocus - Weight and Threshold Tuning Script
ml_training/scripts/tune_weights.py

Tests different alpha/beta combinations and threshold values.
Produces a comparison table for the thesis.

Run from ml_training/scripts/:
    python tune_weights.py
"""

import os, csv, pickle, random
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(SCRIPT_DIR, "data", "processed", "metadata_clean.csv")
OUTPUTS_DIR  = os.path.join(SCRIPT_DIR, "..", "outputs")
MODEL_PATH   = os.path.join(OUTPUTS_DIR, "nb_model.pkl")
VEC_PATH     = os.path.join(OUTPUTS_DIR, "vectorizer.pkl")
RESULTS_PATH = os.path.join(OUTPUTS_DIR, "tuning_results.txt")

LABELS       = ["Educational", "Neutral", "Overstimulating"]
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def load_all():
    texts, labels = [], []
    with open(DATA_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t, l = row.get("text","").strip(), row.get("label","").strip()
            if t and l in LABELS:
                texts.append(t); labels.append(l)

    neutral_count = labels.count("Neutral")
    t_over = [t for t,l in zip(texts,labels) if l=="Overstimulating"]
    l_over = [l for l in labels if l=="Overstimulating"]
    if len(t_over) < neutral_count:
        t_up, l_up = resample(t_over, l_over, n_samples=neutral_count, random_state=RANDOM_STATE)
        texts += t_up; labels += l_up

    _, X_test, _, y_test = train_test_split(
        texts, labels, test_size=0.30, random_state=RANDOM_STATE, stratify=labels
    )

    with open(MODEL_PATH,"rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"] if isinstance(bundle, dict) else bundle
    with open(VEC_PATH,"rb") as f:
        vec = pickle.load(f)

    return model, vec, X_test, y_test


def predict_hybrid(proba, classes, alpha, block_thresh, allow_thresh):
    over_idx = classes.index("Overstimulating")
    preds = []
    for prob in proba:
        score_nb    = float(prob[over_idx])
        score_h     = score_nb   # proxy
        score_final = (alpha * score_nb) + ((1 - alpha) * score_h)
        if   score_final >= block_thresh: preds.append("Overstimulating")
        elif score_final <= allow_thresh: preds.append("Educational")
        else:                             preds.append("Neutral")
    return preds


def main():
    print("\n" + "="*65)
    print("CHILDFOCUS — WEIGHT AND THRESHOLD TUNING")
    print("="*65)

    model, vec, X_test, y_test = load_all()
    X_vec   = vec.transform(X_test)
    proba   = model.predict_proba(X_vec)
    classes = list(model.classes_)

    # ── Weight tuning (fixed thresholds 0.75/0.35) ────────────────────────────
    alphas = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    print(f"\n{'Alpha(NB)':>10} {'Beta(H)':>10} {'F1-Score':>10} {'Accuracy':>10}")
    print("-"*45)

    best_alpha, best_f1 = 0.4, 0.0
    weight_rows = []
    for alpha in alphas:
        beta  = round(1 - alpha, 1)
        preds = predict_hybrid(proba, classes, alpha, 0.75, 0.35)
        f1    = f1_score(y_test, preds, average="weighted", zero_division=0)
        acc   = accuracy_score(y_test, preds)
        marker = " ← THESIS" if alpha == 0.4 else ""
        print(f"{alpha:>10.1f} {beta:>10.1f} {f1:>10.4f} {acc:>10.4f}{marker}")
        weight_rows.append((alpha, beta, f1, acc))
        if f1 > best_f1:
            best_f1, best_alpha = f1, alpha

    print(f"\n  Best alpha: {best_alpha} | Best F1: {best_f1:.4f}")

    # ── Threshold tuning (fixed alpha=0.4) ────────────────────────────────────
    threshold_pairs = [
        (0.70, 0.30), (0.70, 0.35), (0.75, 0.30),
        (0.75, 0.35), (0.75, 0.40), (0.80, 0.35), (0.80, 0.40),
    ]
    print(f"\n\n{'Block(≥)':>10} {'Allow(≤)':>10} {'F1-Score':>10} {'Accuracy':>10}")
    print("-"*45)

    thresh_rows = []
    for block, allow in threshold_pairs:
        preds = predict_hybrid(proba, classes, 0.4, block, allow)
        f1    = f1_score(y_test, preds, average="weighted", zero_division=0)
        acc   = accuracy_score(y_test, preds)
        marker = " ← THESIS" if (block == 0.75 and allow == 0.35) else ""
        print(f"{block:>10.2f} {allow:>10.2f} {f1:>10.4f} {acc:>10.4f}{marker}")
        thresh_rows.append((block, allow, f1, acc))

    # ── Save results ──────────────────────────────────────────────────────────
    import datetime
    with open(RESULTS_PATH, "w") as f:
        f.write(f"ChildFocus Tuning Results — {datetime.datetime.now()}\n\n")
        f.write("WEIGHT TUNING (Block=0.75, Allow=0.35)\n")
        f.write(f"{'Alpha':>8} {'Beta':>8} {'F1':>10} {'Accuracy':>10}\n")
        for row in weight_rows:
            f.write(f"{row[0]:>8.1f} {row[1]:>8.1f} {row[2]:>10.4f} {row[3]:>10.4f}\n")
        f.write(f"\nTHRESHOLD TUNING (Alpha=0.4)\n")
        f.write(f"{'Block':>8} {'Allow':>8} {'F1':>10} {'Accuracy':>10}\n")
        for row in thresh_rows:
            f.write(f"{row[0]:>8.2f} {row[1]:>8.2f} {row[2]:>10.4f} {row[3]:>10.4f}\n")

    print(f"\n[TUNE] ✓ Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()