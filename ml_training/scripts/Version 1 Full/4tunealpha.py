"""
Step 4 — Hyperparameter Tuning: Alpha search for Complement NB
ChildFocus — ml_training/scripts/4tunealpha.py

Builds text directly from train_490.csv — no join with metadata_clean needed.
Test set (test_210.csv) still SEALED.
"""
import re, numpy as np, pandas as pd
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
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

# ── Alpha sweep via 5-Fold CV ─────────────────────────────────────────────
alphas = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00, 1.50, 2.00]
skf    = StratifiedKFold(n_splits=5, shuffle=False)

print("\n" + "="*55)
print(f"  {'Alpha':>8}  {'CV Mean':>9}  {'CV Std':>8}")
print("="*55)

best_alpha, best_score, best_std = None, 0.0, 999.0
rows = []

for alpha in alphas:
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1,2), max_features=10000,
            sublinear_tf=True, min_df=2, strip_accents="unicode"
        )),
        ("clf", ComplementNB(alpha=alpha))
    ])
    scores = cross_val_score(pipe, X_tr, y_tr, cv=skf, scoring="accuracy")
    mean, std = scores.mean(), scores.std()
    rows.append((alpha, mean, std))
    if mean > best_score:
        best_score, best_std, best_alpha = mean, std, alpha

for alpha, mean, std in rows:
    marker = "  <- BEST" if alpha == best_alpha else ""
    print(f"  alpha={alpha:<6}  {mean*100:>8.2f}%  +-{std*100:.2f}%{marker}")

print("="*55)
print(f"\n  Best alpha : {best_alpha}")
print(f"  CV mean    : {best_score*100:.2f}%  +-{best_std*100:.2f}%")
print(f"\n-> Open 5final_eval.py and set:  BEST_ALPHA = {best_alpha}")
print("-> Then run: py 5final_eval.py")
print("-> Test set is still SEALED.")
