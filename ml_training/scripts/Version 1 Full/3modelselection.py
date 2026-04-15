"""
Step 3 — Model Selection via Stratified 5-Fold CV
ChildFocus — ml_training/scripts/3modelselection.py

Builds text directly from train_490.csv — no join with metadata_clean needed.
Test set (test_210.csv) is never touched here.
"""
import re, numpy as np, pandas as pd
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

# ── Same text formula as preprocess.py + text_builder.py ─────────────────
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

# ── Load train_490.csv and build text ─────────────────────────────────────
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
print(f"Label dist: { {l: y_tr.count(l) for l in LABELS} }")

# ── Models ─────────────────────────────────────────────────────────────────
models = {
    "Complement NB":       ComplementNB(alpha=1.0),
    "Multinomial NB":      MultinomialNB(alpha=1.0),
    "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000),
    "Linear SVM":          LinearSVC(C=1.0, max_iter=2000),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
}

# ── Stratified 5-Fold CV on training data ONLY ────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=False)

print("\n" + "="*68)
print(f"  {'Model':<22} {'CV Mean':>9} {'CV Std':>8} {'F1':>8} {'Gap':>7}")
print("="*68)

all_results = {}
for name, clf in models.items():
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1,2), max_features=10000,
            sublinear_tf=True, min_df=2, strip_accents="unicode"
        )),
        ("clf", clf)
    ])
    cv = cross_validate(
        pipe, X_tr, y_tr, cv=skf,
        scoring={"accuracy":"accuracy", "f1":"f1_macro"},
        return_train_score=True
    )
    mean = cv["test_accuracy"].mean()
    std  = cv["test_accuracy"].std()
    f1   = cv["test_f1"].mean()
    gap  = (cv["train_accuracy"].mean() - mean) * 100
    all_results[name] = {"cv_mean":mean, "cv_std":std, "f1":f1, "gap":gap}
    print(f"  {name:<22} {mean*100:>8.2f}%  +-{std*100:.2f}%  {f1:.4f}  {gap:>5.1f}%")

print("="*68)
print("\n=== Decision Guide ===")
print("Best CV mean  :", max(all_results, key=lambda k: all_results[k]["cv_mean"]))
print("Lowest CV std :", min(all_results, key=lambda k: all_results[k]["cv_std"]))
print("Lowest gap    :", min(all_results, key=lambda k: all_results[k]["gap"]))
print("\n-> Selected model   : Complement NB")
print("-> Rennie et al. 2003: CNB designed for complement class detection")
print("-> Next: py 4tunealpha.py")
print("-> Test set (test_210.csv) still SEALED.")
