"""
Step 4 — Hyperparameter Tuning: Alpha search for Complement NB
ChildFocus — ml_training/scripts/4tunealpha.py

Builds text directly from train_490.csv — no join with metadata_clean needed.
Test set (test_210.csv) still SEALED.

All four thesis metrics reported for each alpha value:
  Accuracy  — primary selection metric
  Precision — macro-averaged
  Recall    — macro-averaged (includes Overstimulating recall — safety metric)
  F1-Score  — macro-averaged (used as tiebreaker between close accuracy values)

Best alpha is selected by highest CV accuracy, with F1 as tiebreaker.
"""
import re, numpy as np, pandas as pd
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

# ── Same text formula as preprocess.py ────────────────────────────────────
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

# ── Load train_490.csv ─────────────────────────────────────────────────────
LABELS = ["Educational", "Neutral", "Overstimulating"]

df = pd.read_csv("train_490.csv")
df = df[df["label"].isin(LABELS)].reset_index(drop=True)

X_tr = df.apply(lambda r: build_nb_text(
    title=r.get("title", ""),
    tags=r.get("tags", ""),
    description=r.get("description", "")
), axis=1).tolist()
y_tr = df["label"].tolist()

print(f"Training samples loaded: {len(X_tr)}  <- must be 490")
assert len(X_tr) == 490, f"Expected 490, got {len(X_tr)}"

# ── Alpha sweep via 5-Fold CV — all four metrics ───────────────────────────
alphas  = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00, 1.50, 2.00]
skf     = StratifiedKFold(n_splits=5, shuffle=False)

scoring = {
    "accuracy":  "accuracy",
    "precision": "precision_macro",
    "recall":    "recall_macro",
    "f1":        "f1_macro",
}

print("\n" + "=" * 80)
print(f"  {'Alpha':>8}  {'Acc%':>8}  {'±Acc':>7}  {'Prec%':>7}  "
      f"{'Rec%':>7}  {'F1':>8}")
print("=" * 80)

best_alpha, best_acc, best_f1 = None, 0.0, 0.0
rows = []

for alpha in alphas:
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), max_features=10000,
            sublinear_tf=True, min_df=2, strip_accents="unicode"
        )),
        ("clf", ComplementNB(alpha=alpha))
    ])
    cv   = cross_validate(pipe, X_tr, y_tr, cv=skf, scoring=scoring)
    acc  = cv["test_accuracy"].mean()
    std  = cv["test_accuracy"].std()
    prec = cv["test_precision"].mean()
    rec  = cv["test_recall"].mean()
    f1   = cv["test_f1"].mean()

    rows.append((alpha, acc, std, prec, rec, f1))
    if acc > best_acc or (acc == best_acc and f1 > best_f1):
        best_acc, best_f1, best_alpha = acc, f1, alpha

for alpha, acc, std, prec, rec, f1 in rows:
    marker = "  <- BEST" if alpha == best_alpha else ""
    print(f"  alpha={alpha:<6}  {acc*100:>7.2f}%  "
          f"+-{std*100:.2f}%  {prec*100:>6.2f}%  "
          f"{rec*100:>6.2f}%  {f1:.4f}{marker}")

print("=" * 80)

# ── Per-fold detail for best alpha ────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  BEST ALPHA = {best_alpha} — Per-Fold Detail")
print(f"{'='*70}")
best_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2), max_features=10000,
        sublinear_tf=True, min_df=2, strip_accents="unicode"
    )),
    ("clf", ComplementNB(alpha=best_alpha))
])
cv_best = cross_validate(best_pipe, X_tr, y_tr, cv=skf, scoring=scoring)
print(f"  {'Fold':<8} {'Accuracy':>10} {'Precision':>11} {'Recall':>8} {'F1':>8}")
print(f"  {'-'*48}")
for i in range(5):
    print(f"  Fold {i+1:<4} "
          f"{cv_best['test_accuracy'][i]*100:>9.2f}%  "
          f"{cv_best['test_precision'][i]*100:>9.2f}%  "
          f"{cv_best['test_recall'][i]*100:>7.2f}%  "
          f"{cv_best['test_f1'][i]:.4f}")
print(f"  {'-'*48}")
print(f"  {'Mean':<8} "
      f"{cv_best['test_accuracy'].mean()*100:>9.2f}%  "
      f"{cv_best['test_precision'].mean()*100:>9.2f}%  "
      f"{cv_best['test_recall'].mean()*100:>7.2f}%  "
      f"{cv_best['test_f1'].mean():.4f}")
print(f"  {'±Std':<8} "
      f"+-{cv_best['test_accuracy'].std()*100:>7.2f}%  "
      f"+-{cv_best['test_precision'].std()*100:>7.2f}%  "
      f"+-{cv_best['test_recall'].std()*100:>5.2f}%  "
      f"+-{cv_best['test_f1'].std():.4f}")

print(f"\n{'='*70}")
print(f"  STEP 4 COMPLETE")
print(f"{'='*70}")
print(f"  Best alpha : {best_alpha}")
print(f"  CV Acc     : {best_acc*100:.2f}%")
print(f"  CV F1      : {best_f1:.4f}")
print(f"\n-> Open 4b_tune_tfidf.py and confirm: BEST_ALPHA = {best_alpha}")
print(f"-> Then run: py 4b_tune_tfidf.py")
print(f"-> Test set (test_210.csv) still SEALED.")
