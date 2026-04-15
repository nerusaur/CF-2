"""
Step 5 — Final Evaluation: Open the holdout exactly once
ChildFocus — ml_training/scripts/5final_eval.py

THIS IS THE ONLY TIME test_210.csv IS OPENED.
Before running: set BEST_ALPHA to the result from Step 4.
"""
import re, numpy as np, pandas as pd
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, recall_score
)

# ========================================================
#  SET THIS to the best alpha from Step 4 before running
BEST_ALPHA = 0.15
# ========================================================

LABELS = ["Educational", "Neutral", "Overstimulating"]

STOP_WORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","it","this","that","was","are","be","as","so","we",
    "he","she","they","you","i","my","your","his","her","its","our","their",
    "what","which","who","will","would","could","should","has","have","had",
    "do","does","did","not","no","if","then","than","when","where","how",
    "all","each","more","also","just","can","up","out","about","into",
    "too","very","s","t","re","ve","ll","d",
}

def build_nb_text(title="", tags=None, description=""):
    title_part = f"{title} " * 3
    if isinstance(tags, list):
        tags_str = " ".join(str(t) for t in tags)
    else:
        tags_str = str(tags) if tags and str(tags) != "nan" else ""
    desc_part = str(description or "")[:300]
    raw = f"{title_part}{tags_str} {desc_part}".lower()
    raw = re.sub(r"https?://\S+|www\.\S+", " ", raw)
    raw = re.sub(r"[^a-z\s]", " ", raw)
    tokens = [t for t in raw.split() if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)

def load_from_csv(path):
    df = pd.read_csv(path)
    df = df[df["label"].isin(LABELS)].reset_index(drop=True)
    X = df.apply(lambda r: build_nb_text(
        title=r.get("title",""),
        tags=r.get("tags",""),
        description=r.get("description","")
    ), axis=1).tolist()
    y = df["label"].tolist()
    return X, y

# ── Load both splits ────────────────────────────────────────────────────────
X_train, y_train = load_from_csv("train_490.csv")
X_test,  y_test  = load_from_csv("test_210.csv")

print(f"Train: {len(X_train)}  <- must be 490")
print(f"Test : {len(X_test)}   <- must be 210")
assert len(X_train) == 490, f"Expected 490, got {len(X_train)}"
assert len(X_test)  == 210, f"Expected 210, got {len(X_test)}"

# ── Pipeline with best alpha ────────────────────────────────────────────────
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1,2), max_features=10000,
        sublinear_tf=True, min_df=2, strip_accents="unicode"
    )),
    ("clf", ComplementNB(alpha=BEST_ALPHA))
])

# ── CV on training only (for thesis table) ─────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=False)
cv_scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="accuracy")
cv_f1     = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="f1_macro")

print(f"\n{'='*55}")
print(f"  CV RESULTS (training pool, n=490)")
print(f"{'='*55}")
for i, (acc, f1) in enumerate(zip(cv_scores, cv_f1)):
    print(f"  Fold {i+1}:  acc={acc*100:.2f}%  f1={f1:.4f}")
print(f"  {'-'*48}")
print(f"  Mean : {cv_scores.mean()*100:.2f}%  +-{cv_scores.std()*100:.2f}%")
print(f"  F1   : {cv_f1.mean():.4f}")

# ── Train on full training set then evaluate on holdout ────────────────────
pipe.fit(X_train, y_train)
y_pred       = pipe.predict(X_test)
y_train_pred = pipe.predict(X_train)

train_acc   = accuracy_score(y_train, y_train_pred)
holdout_acc = accuracy_score(y_test,  y_pred)
overfit_gap = (train_acc - holdout_acc) * 100

# Bi-decision (Overstimulating vs Safe)
def bi(labels):
    return ["Overstimulating" if l=="Overstimulating" else "Safe" for l in labels]
bi_acc = accuracy_score(bi(y_test), bi(y_pred))

# Overstimulating recall
over_rec = recall_score(
    y_test, y_pred, labels=["Overstimulating"],
    average=None, zero_division=0
)[0]

macro_f1 = f1_score(y_test, y_pred, average="macro")

print(f"\n{'='*55}")
print(f"  FINAL HOLDOUT RESULTS  (test_210.csv)")
print(f"{'='*55}")
print(f"  Alpha              : {BEST_ALPHA}")
print(f"  Train accuracy     : {train_acc*100:.2f}%")
print(f"  Holdout accuracy   : {holdout_acc*100:.2f}%")
print(f"  Overfit gap        : {overfit_gap:.2f}%")
print(f"  Bi-decision acc    : {bi_acc*100:.2f}%")
print(f"  Over. recall       : {over_rec*100:.2f}%")
print(f"  CV mean (5-fold)   : {cv_scores.mean()*100:.2f}% +-{cv_scores.std()*100:.2f}%")
print(f"  Macro F1 (holdout) : {macro_f1:.4f}")

print(f"\n{'='*55}")
print("  CLASSIFICATION REPORT (holdout)")
print(f"{'='*55}")
print(classification_report(y_test, y_pred, target_names=LABELS, digits=4))

print("  CONFUSION MATRIX")
cm = confusion_matrix(y_test, y_pred, labels=LABELS)
print(f"  {'':>20}", "  ".join(f"{l[:5]:>7}" for l in LABELS))
for i, rl in enumerate(LABELS):
    print(f"  {rl:>20}", "  ".join(f"{cm[i][j]:>7}" for j in range(3)))

print(f"\n{'='*55}")
print("  THESIS SUMMARY ROW — Complement NB")
print(f"{'='*55}")
print(f"  Holdout: {holdout_acc*100:.2f}% | CV: {cv_scores.mean()*100:.2f}% +- "
      f"{cv_scores.std()*100:.2f}% | Gap: {overfit_gap:.2f}% | "
      f"Bi-Dec: {bi_acc*100:.2f}% | Over Rec: {over_rec*100:.2f}% | "
      f"F1: {macro_f1:.4f}")
print(f"\n-> These are your FINAL numbers. Paste them into 6generate_figures.py")
print("-> Do NOT re-run after this and tune more.")
