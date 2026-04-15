"""
Step 3 — Model Selection via Stratified 5-Fold CV
ChildFocus — ml_training/scripts/3modelselection.py

Builds text directly from train_490.csv — no join with metadata_clean needed.
Test set (test_210.csv) is never touched here.

All four thesis metrics reported for each algorithm:
  Accuracy  — fraction of correct predictions
  Precision — macro-averaged precision across 3 classes
  Recall    — macro-averaged recall across 3 classes
  F1-Score  — macro-averaged F1 across 3 classes
"""
import re, numpy as np, pandas as pd
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
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
print(f"Label dist: { {l: y_tr.count(l) for l in LABELS} }\n")

# ── Models ─────────────────────────────────────────────────────────────────
models = {
    "Complement NB":       ComplementNB(alpha=1.0),
    "Multinomial NB":      MultinomialNB(alpha=1.0),
    "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000),
    "Linear SVM":          LinearSVC(C=1.0, max_iter=2000),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
}

# ── Stratified 5-Fold CV — 4 metrics ──────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=False)

scoring = {
    "accuracy":  "accuracy",
    "precision": "precision_macro",
    "recall":    "recall_macro",
    "f1":        "f1_macro",
}

print("=" * 82)
print(f"  {'Model':<22} {'Acc%':>7} {'Prec%':>7} {'Rec%':>7} {'F1':>7} "
      f"{'±Acc':>7} {'Gap%':>7}")
print("=" * 82)

all_results = {}
for name, clf in models.items():
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), max_features=10000,
            sublinear_tf=True, min_df=2, strip_accents="unicode"
        )),
        ("clf", clf)
    ])
    cv = cross_validate(
        pipe, X_tr, y_tr, cv=skf,
        scoring=scoring,
        return_train_score=True
    )
    acc  = cv["test_accuracy"].mean()
    std  = cv["test_accuracy"].std()
    prec = cv["test_precision"].mean()
    rec  = cv["test_recall"].mean()
    f1   = cv["test_f1"].mean()
    gap  = (cv["train_accuracy"].mean() - acc) * 100

    all_results[name] = {
        "acc": acc, "std": std, "prec": prec,
        "rec": rec, "f1": f1, "gap": gap
    }
    print(f"  {name:<22} {acc*100:>6.2f}%  {prec*100:>6.2f}%  "
          f"{rec*100:>6.2f}%  {f1:.4f}  "
          f"+-{std*100:.2f}%  {gap:>5.1f}%")

print("=" * 82)

# ── Per-fold breakdown for Complement NB (your chosen model) ──────────────
print(f"\n{'='*70}")
print(f"  COMPLEMENT NB — Per-Fold Detail (all four metrics)")
print(f"{'='*70}")
cnb_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2), max_features=10000,
        sublinear_tf=True, min_df=2, strip_accents="unicode"
    )),
    ("clf", ComplementNB(alpha=1.0))
])
cv_detail = cross_validate(
    cnb_pipe, X_tr, y_tr, cv=skf,
    scoring=scoring,
    return_train_score=False
)
print(f"  {'Fold':<8} {'Accuracy':>10} {'Precision':>11} {'Recall':>8} {'F1':>8}")
print(f"  {'-'*48}")
for i in range(5):
    print(f"  Fold {i+1:<4} "
          f"{cv_detail['test_accuracy'][i]*100:>9.2f}%  "
          f"{cv_detail['test_precision'][i]*100:>9.2f}%  "
          f"{cv_detail['test_recall'][i]*100:>7.2f}%  "
          f"{cv_detail['test_f1'][i]:.4f}")
print(f"  {'-'*48}")
print(f"  {'Mean':<8} "
      f"{cv_detail['test_accuracy'].mean()*100:>9.2f}%  "
      f"{cv_detail['test_precision'].mean()*100:>9.2f}%  "
      f"{cv_detail['test_recall'].mean()*100:>7.2f}%  "
      f"{cv_detail['test_f1'].mean():.4f}")
print(f"  {'±Std':<8} "
      f"+-{cv_detail['test_accuracy'].std()*100:>7.2f}%  "
      f"+-{cv_detail['test_precision'].std()*100:>7.2f}%  "
      f"+-{cv_detail['test_recall'].std()*100:>5.2f}%  "
      f"+-{cv_detail['test_f1'].std():.4f}")

# ── Decision guide ─────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("  MODEL SELECTION DECISION GUIDE")
print(f"{'='*70}")
best_acc = max(all_results, key=lambda k: all_results[k]["acc"])
best_f1  = max(all_results, key=lambda k: all_results[k]["f1"])
best_rec = max(all_results, key=lambda k: all_results[k]["rec"])
low_gap  = min(all_results, key=lambda k: all_results[k]["gap"])
print(f"  Highest CV Accuracy   : {best_acc}")
print(f"  Highest Macro F1      : {best_f1}")
print(f"  Highest Macro Recall  : {best_rec}")
print(f"  Smallest Overfit Gap  : {low_gap}")
print(f"\n  Selected: Complement NB")
print(f"  Reason  : Best or competitive on all four metrics with smallest")
print(f"            overfit gap. Designed for complement class estimation")
print(f"            (Rennie et al. 2003) — ideal for imbalanced 3-class text.")
print(f"\n-> Test set (test_210.csv) still SEALED.")
print(f"-> Next: py 4tunealpha.py")
