"""
Step 4b — Hyperparameter Tuning: TF-IDF grid search for Complement NB
ChildFocus — ml_training/scripts/4b_tune_tfidf.py

Sweeps max_features and min_df of the TF-IDF vectorizer.
Uses BEST_ALPHA = 1.5 from Step 4 (4tunealpha.py).
CV-only — test set (test_210.csv) still SEALED.

Run from ml_training/scripts/:
    py 4b_tune_tfidf.py

After this: set BEST_MAX_FEATURES and BEST_MIN_DF in 5final_eval.py
"""

import re
import numpy as np
import pandas as pd
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

# ── Best alpha carried forward from Step 4 ────────────────────────────────
BEST_ALPHA = 1.5

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
print(f"Alpha fixed at         : {BEST_ALPHA}  (from Step 4)")
print(f"Test set               : still SEALED\n")

# ── Grid: max_features × min_df ───────────────────────────────────────────
max_features_opts = [5000, 10000, 15000, 20000]
min_df_opts       = [1, 2, 3, 5]

skf = StratifiedKFold(n_splits=5, shuffle=False)

rows = []

print("=" * 65)
print(f"  {'max_feat':>10}  {'min_df':>7}  {'CV Mean':>9}  {'CV Std':>8}")
print("=" * 65)

best_mean   = 0.0
best_cfg    = None

for mf in max_features_opts:
    for md in min_df_opts:
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=mf,
                min_df=md,
                sublinear_tf=True,
                strip_accents="unicode"
            )),
            ("clf", ComplementNB(alpha=BEST_ALPHA))
        ])
        scores = cross_val_score(pipe, X_tr, y_tr, cv=skf, scoring="accuracy")
        mean, std = scores.mean(), scores.std()
        rows.append((mf, md, mean, std))
        if mean > best_mean:
            best_mean = mean
            best_cfg  = (mf, md, mean, std)

# Print all results grouped by max_features for readability
prev_mf = None
for mf, md, mean, std in rows:
    if mf != prev_mf:
        print(f"  {'─'*60}")
        prev_mf = mf
    marker = "  <- BEST" if (mf, md) == (best_cfg[0], best_cfg[1]) else ""
    print(f"  {mf:>10}  {md:>7}  {mean*100:>8.2f}%  +-{std*100:.2f}%{marker}")

print("=" * 65)

best_mf, best_md, best_cv, best_std = best_cfg
print(f"\n  Best max_features : {best_mf}")
print(f"  Best min_df       : {best_md}")
print(f"  CV mean           : {best_cv*100:.2f}%  +-{best_std*100:.2f}%")

# Comparison against Step 4 baseline (max_features=10000, min_df=2)
baseline = next((m for mf, md, m, s in rows if mf == 10000 and md == 2), None)
if baseline is not None:
    delta = (best_cv - baseline) * 100
    print(f"\n  Baseline (10000, min_df=2) : {baseline*100:.2f}%")
    print(f"  Gain over baseline         : {delta:+.2f}%")

print(f"\n-> Open 5final_eval.py and update:")
print(f"     BEST_ALPHA        = {BEST_ALPHA}")
print(f"     BEST_MAX_FEATURES = {best_mf}")
print(f"     BEST_MIN_DF       = {best_md}")
print(f"-> Then run: py 5final_eval.py  (unseals the test set — one time only)")
print(f"-> Test set (test_210.csv) still SEALED.")
