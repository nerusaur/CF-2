
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

test = pd.read_csv("test_210.csv")
y_true = test["label"].values
classes = ["Educational", "Neutral", "Overstimulating"]

# ── Baseline 1: Random chance ──────────────────────────────────────────────
np.random.seed(42)
y_random = np.random.choice(classes, size=len(y_true))
acc_random = accuracy_score(y_true, y_random)

# ── Baseline 2: Majority class (most frequent in train) ───────────────────
train = pd.read_csv("train_490.csv")
majority = train["label"].value_counts().idxmax()
y_majority = np.full(len(y_true), majority)
acc_majority = accuracy_score(y_true, y_majority)

# ── Baseline 3: Keyword heuristic ─────────────────────────────────────────
def keyword_baseline(title):
    t = str(title).lower()
    if any(k in t for k in ["learn","educat","alphabet","number","science","math","abc","phonics"]):
        return "Educational"
    if any(k in t for k in ["surprise","challenge","fast","crazy","loud","unbox","slime","screaming"]):
        return "Overstimulating"
    return "Neutral"

y_keyword = test["title"].apply(keyword_baseline).values
acc_keyword = accuracy_score(y_true, y_keyword)

# ── Print results ──────────────────────────────────────────────────────────
print("=" * 50)
print("  BASELINE RESULTS (test set, n=210)")
print("=" * 50)
print(f"  B1 Random chance   : {acc_random*100:.2f}%  (expected ~33.33%)")
print(f"  B2 Majority class  : {acc_majority*100:.2f}%  (always predicts '{majority}')")
print(f"  B3 Keyword heuristic: {acc_keyword*100:.2f}%")
print("=" * 50)
print("\nKeyword heuristic detail:")
print(classification_report(y_true, y_keyword, target_names=classes))
print("→ Your model MUST beat 49.05% to be meaningful.")