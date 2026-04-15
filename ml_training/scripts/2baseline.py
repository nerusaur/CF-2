"""
Step 2 — Baselines
ChildFocus — ml_training/scripts/2baseline.py

Establishes three baselines that your model MUST beat to be meaningful.
All four thesis metrics reported for every baseline:
  Accuracy  — fraction of correct predictions
  Precision — of predicted Overstimulating, how many were correct
  Recall    — of actual Overstimulating, how many were caught
  F1-Score  — harmonic mean of precision and recall

Baselines:
  B1  Random chance          (~33.33% acc — pure chance floor)
  B2  Majority class         (always predicts most common class)
  B3  Keyword heuristic      (simple rule-based title scan)

Test set (test_210.csv) used here because baselines need no training.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report,
    precision_score, recall_score, f1_score,
    confusion_matrix
)

# ── Load data ──────────────────────────────────────────────────────────────
test  = pd.read_csv("test_210.csv")
train = pd.read_csv("train_490.csv")

y_true  = test["label"].values
LABELS  = ["Educational", "Neutral", "Overstimulating"]

def four_metrics(y_true, y_pred, name):
    """Print all four metrics for one baseline."""
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro",
                           zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro",
                        zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro",
                    zero_division=0)
    # Overstimulating-specific recall (child safety metric)
    over_rec = recall_score(y_true, y_pred,
                            labels=["Overstimulating"],
                            average=None, zero_division=0)[0]
    return acc, prec, rec, f1, over_rec


# ── Baseline 1: Random chance ──────────────────────────────────────────────
np.random.seed(42)
y_random = np.random.choice(LABELS, size=len(y_true))
b1 = four_metrics(y_true, y_random, "B1 Random")

# ── Baseline 2: Majority class ────────────────────────────────────────────
majority  = train["label"].value_counts().idxmax()
y_majority = np.full(len(y_true), majority)
b2 = four_metrics(y_true, y_majority, "B2 Majority")

# ── Baseline 3: Keyword heuristic ─────────────────────────────────────────
EDU_WORDS  = ["learn", "educat", "alphabet", "number", "science",
              "math", "abc", "phonics", "lesson", "kids learn"]
OVER_WORDS = ["surprise", "challenge", "fast", "crazy", "loud",
              "unbox", "slime", "screaming", "compilation", "extreme"]

def keyword_baseline(title):
    t = str(title).lower()
    if any(k in t for k in EDU_WORDS):
        return "Educational"
    if any(k in t for k in OVER_WORDS):
        return "Overstimulating"
    return "Neutral"

y_keyword = test["title"].apply(keyword_baseline).values
b3 = four_metrics(y_true, y_keyword, "B3 Keyword")

# ── Print full results ─────────────────────────────────────────────────────
print("=" * 68)
print("  BASELINE RESULTS — All Four Metrics  (test_210.csv, n=210)")
print("=" * 68)
print(f"  {'Baseline':<26} {'Accuracy':>9} {'Precision':>11} "
      f"{'Recall':>8} {'F1':>8} {'Over_Rec':>10}")
print(f"  {'-'*68}")

rows = [
    (f"B1  Random chance",         *b1),
    (f"B2  Majority ('{majority}')", *b2),
    (f"B3  Keyword heuristic",      *b3),
]
for name, acc, prec, rec, f1, ov in rows:
    print(f"  {name:<26} {acc*100:>8.2f}%  {prec*100:>9.2f}%  "
          f"{rec*100:>7.2f}%  {f1*100:>7.2f}%  {ov*100:>9.2f}%")

print("=" * 68)
print(f"\n  NOTE — Macro averages weight all 3 classes equally.")
print(f"  Over_Rec = Overstimulating recall (child safety metric).")

# ── Per-class breakdown for B3 (most informative baseline) ───────────────
print(f"\n{'='*68}")
print(f"  B3 KEYWORD HEURISTIC — Per-Class Detail")
print(f"{'='*68}")
print(classification_report(y_true, y_keyword, target_names=LABELS))

cm = confusion_matrix(y_true, y_keyword, labels=LABELS)
print(f"  Confusion Matrix  (rows=Actual, cols=Predicted):")
print(f"  {'':>22}" + "".join(f"{l[:5]:>8}" for l in LABELS))
for i, rl in enumerate(LABELS):
    print(f"  {rl:>22}" + "".join(f"{cm[i][j]:>8}" for j in range(3)))

# ── Target line ────────────────────────────────────────────────────────────
b3_acc = b3[0]
print(f"\n{'='*68}")
print(f"  YOUR MODEL MUST BEAT (all four metrics over baseline):")
print(f"  Accuracy  > {b3_acc*100:.2f}%    (B3 keyword heuristic)")
print(f"  Precision > {b3[1]*100:.2f}%    (macro)")
print(f"  Recall    > {b3[2]*100:.2f}%    (macro)")
print(f"  F1        > {b3[3]*100:.2f}%    (macro)")
print(f"  Over_Rec  > {b3[4]*100:.2f}%    (Overstimulating recall)")
print(f"{'='*68}")
print("\n-> Run: py 3modelselection.py")
