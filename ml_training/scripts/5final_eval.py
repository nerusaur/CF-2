"""
Step 5 — Final Evaluation: Open the holdout exactly once
ChildFocus — ml_training/scripts/5final_eval.py

THIS IS THE ONLY TIME test_210.csv IS OPENED.
Do NOT re-run after seeing results. Do NOT tune further after this.

Before running:
  - Set BEST_ALPHA from Step 4 (4tunealpha.py)
  - Set BEST_MAX_FEATURES and BEST_MIN_DF from Step 4b (4b_tune_tfidf.py)
"""
import re
import numpy as np
import pandas as pd
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, recall_score, precision_score
)

# ========================================================
#  SET THESE from Step 4 and Step 4b before running
BEST_ALPHA        = 1.5     # from 4tunealpha.py    (was wrongly 0.15)
BEST_MAX_FEATURES = 5000    # from 4b_tune_tfidf.py (update if different)
BEST_MIN_DF       = 1       # from 4b_tune_tfidf.py (update if different)
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
        title=r.get("title", ""),
        tags=r.get("tags", ""),
        description=r.get("description", "")
    ), axis=1).tolist()
    y = df["label"].tolist()
    return X, y

# ── Load both splits ───────────────────────────────────────────────────────
X_train, y_train = load_from_csv("train_490.csv")
X_test,  y_test  = load_from_csv("test_210.csv")

print(f"Train : {len(X_train)}  <- must be 490")
print(f"Test  : {len(X_test)}   <- must be 210")
assert len(X_train) == 490, f"Expected 490, got {len(X_train)}"
assert len(X_test)  == 210, f"Expected 210, got {len(X_test)}"

# ── Pipeline with tuned hyperparameters ───────────────────────────────────
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=BEST_MAX_FEATURES,
        min_df=BEST_MIN_DF,
        sublinear_tf=True,
        strip_accents="unicode"
    )),
    ("clf", ComplementNB(alpha=BEST_ALPHA))
])

# ── CV on training only (for thesis table — test still sealed here) ────────
skf       = StratifiedKFold(n_splits=5, shuffle=False)
cv_scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="accuracy")
cv_f1     = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="f1_macro")
cv_prec   = cross_val_score(pipe, X_train, y_train, cv=skf,
                             scoring="precision_macro")
cv_rec    = cross_val_score(pipe, X_train, y_train, cv=skf,
                             scoring="recall_macro")

print(f"\n{'='*58}")
print(f"  CV RESULTS (training pool only, n=490, test still sealed)")
print(f"{'='*58}")
print(f"  {'Fold':<8} {'Accuracy':>10} {'Precision':>11} "
      f"{'Recall':>8} {'F1':>8}")
print(f"  {'-'*50}")
for i in range(len(cv_scores)):
    print(f"  Fold {i+1:<4} {cv_scores[i]*100:>9.2f}%  "
          f"{cv_prec[i]*100:>9.2f}%  "
          f"{cv_rec[i]*100:>7.2f}%  "
          f"{cv_f1[i]:.4f}")
print(f"  {'-'*50}")
print(f"  Mean     {cv_scores.mean()*100:>9.2f}%  "
      f"{cv_prec.mean()*100:>9.2f}%  "
      f"{cv_rec.mean()*100:>7.2f}%  "
      f"{cv_f1.mean():.4f}")
print(f"  Std    +-{cv_scores.std()*100:>8.2f}%  "
      f"+-{cv_prec.std()*100:>7.2f}%  "
      f"+-{cv_rec.std()*100:>6.2f}%  "
      f"+-{cv_f1.std():.4f}")

# ── Train on full training set, evaluate on holdout ────────────────────────
pipe.fit(X_train, y_train)
y_pred       = pipe.predict(X_test)
y_train_pred = pipe.predict(X_train)

# ── Four core metrics — holdout ────────────────────────────────────────────
train_acc     = accuracy_score(y_train, y_train_pred)
holdout_acc   = accuracy_score(y_test, y_pred)
overfit_gap   = (train_acc - holdout_acc) * 100

macro_precision = precision_score(y_test, y_pred, average="macro",
                                  zero_division=0)
macro_recall    = recall_score(y_test, y_pred, average="macro",
                               zero_division=0)
macro_f1        = f1_score(y_test, y_pred, average="macro",
                           zero_division=0)

weighted_precision = precision_score(y_test, y_pred, average="weighted",
                                     zero_division=0)
weighted_recall    = recall_score(y_test, y_pred, average="weighted",
                                  zero_division=0)
weighted_f1        = f1_score(y_test, y_pred, average="weighted",
                               zero_division=0)

# Bi-decision accuracy (Overstimulating vs Safe)
def bi(labels):
    return ["Overstimulating" if l == "Overstimulating" else "Safe"
            for l in labels]
bi_acc = accuracy_score(bi(y_test), bi(y_pred))

# Overstimulating recall (child safety metric — must be high)
over_rec = recall_score(
    y_test, y_pred, labels=["Overstimulating"],
    average=None, zero_division=0
)[0]

# Per-class precision, recall, F1
prec_per  = precision_score(y_test, y_pred, labels=LABELS, average=None,
                             zero_division=0)
rec_per   = recall_score(y_test, y_pred, labels=LABELS, average=None,
                          zero_division=0)
f1_per    = f1_score(y_test, y_pred, labels=LABELS, average=None,
                     zero_division=0)
from sklearn.metrics import confusion_matrix as cm_fn
cm = cm_fn(y_test, y_pred, labels=LABELS)
support = cm.sum(axis=1)

# ── Print all results ──────────────────────────────────────────────────────
print(f"\n{'='*58}")
print(f"  FINAL HOLDOUT RESULTS  (test_210.csv — unsealed once)")
print(f"{'='*58}")
print(f"  Alpha              : {BEST_ALPHA}")
print(f"  max_features       : {BEST_MAX_FEATURES}")
print(f"  min_df             : {BEST_MIN_DF}")
print(f"")
print(f"  Train accuracy     : {train_acc*100:.2f}%")
print(f"  Holdout accuracy   : {holdout_acc*100:.2f}%")
print(f"  Overfit gap        : {overfit_gap:.2f}%  (train - holdout)")
print(f"")
print(f"  ── Macro averages (unweighted mean across 3 classes) ──")
print(f"  Macro Precision    : {macro_precision*100:.2f}%  ({macro_precision:.4f})")
print(f"  Macro Recall       : {macro_recall*100:.2f}%  ({macro_recall:.4f})")
print(f"  Macro F1           : {macro_f1*100:.2f}%  ({macro_f1:.4f})")
print(f"")
print(f"  ── Weighted averages (weighted by class support) ──")
print(f"  Weighted Precision : {weighted_precision*100:.2f}%  ({weighted_precision:.4f})")
print(f"  Weighted Recall    : {weighted_recall*100:.2f}%  ({weighted_recall:.4f})")
print(f"  Weighted F1        : {weighted_f1*100:.2f}%  ({weighted_f1:.4f})")
print(f"")
print(f"  ── Child safety metrics ──")
print(f"  Bi-decision acc    : {bi_acc*100:.2f}%  (Overstimulating vs Safe)")
print(f"  Overstimulating recall : {over_rec*100:.2f}%  (missed detections = "
      f"{int(support[LABELS.index('Overstimulating')] * (1-over_rec))}"
      f"/{support[LABELS.index('Overstimulating')]})")
print(f"  CV mean (5-fold)   : {cv_scores.mean()*100:.2f}% "
      f"+-{cv_scores.std()*100:.2f}%")

print(f"\n{'='*58}")
print("  CLASSIFICATION REPORT (holdout, per-class breakdown)")
print(f"{'='*58}")
print(classification_report(y_test, y_pred, target_names=LABELS, digits=4))

print(f"\n{'='*58}")
print("  PER-CLASS METRICS TABLE")
print(f"{'='*58}")
print(f"  {'Class':<20} {'Precision':>10} {'Recall':>8} "
      f"{'F1':>8} {'Support':>9}")
print(f"  {'-'*58}")
for i, lbl in enumerate(LABELS):
    print(f"  {lbl:<20} {prec_per[i]*100:>9.2f}%  "
          f"{rec_per[i]*100:>7.2f}%  "
          f"{f1_per[i]*100:>7.2f}%  "
          f"{support[i]:>7}")

print(f"\n{'='*58}")
print("  CONFUSION MATRIX (rows=actual, columns=predicted)")
print(f"{'='*58}")
print(f"  {'':>22}" + "".join(f"{l[:5]:>10}" for l in LABELS))
for i, rl in enumerate(LABELS):
    print(f"  {rl:>22}" + "".join(f"{cm[i][j]:>10}" for j in range(3)))

print(f"\n{'='*58}")
print("  THESIS SUMMARY ROW — Complement NB  (paste into Chapter 5)")
print(f"{'='*58}")
print(f"  Holdout Acc : {holdout_acc*100:.2f}%")
print(f"  CV Acc      : {cv_scores.mean()*100:.2f}% +-{cv_scores.std()*100:.2f}%")
print(f"  Overfit Gap : {overfit_gap:.2f}%")
print(f"  Macro P     : {macro_precision:.4f}")
print(f"  Macro R     : {macro_recall:.4f}")
print(f"  Macro F1    : {macro_f1:.4f}")
print(f"  Bi-Dec Acc  : {bi_acc*100:.2f}%")
print(f"  Over. Recall: {over_rec*100:.2f}%")
print(f"\n-> These are your FINAL numbers.")
print("-> Do NOT re-run with different parameters after this.")
print("-> Paste results into 6generate_figures.py")
