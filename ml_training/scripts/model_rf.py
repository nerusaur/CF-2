"""
ChildFocus — Model: Random Forest (RF)
ml_training/scripts/model_rf.py

WHY RF IS INCLUDED AS A COMPARISON:
  Random Forest is an ensemble tree-based classifier that builds multiple
  decision trees on random feature subsets and aggregates their predictions.
  It represents a completely different algorithm family from the probabilistic
  (NB, LR) and margin-based (SVM) approaches.

  For high-dimensional, sparse TF-IDF features (10,000 dimensions), RF is
  generally at a disadvantage compared to text-optimised classifiers.
  Decision trees split on individual features, and finding meaningful splits
  in a 10,000-dimensional sparse space requires many more trees. This is why
  RF typically ranks last in text classification benchmarks.

  Thesis comparison point: "Random Forest's lower performance on sparse TF-IDF
  vectors validates the theoretical expectation that probabilistic models are
  better suited to text classification than tree-based ensembles, which were
  designed for dense feature spaces."

  n_estimators=100 is the standard default; increasing to 200+ improves RF
  slightly but not enough to close the gap with CNB.

RUN STANDALONE:
  cd ml_training/scripts
  py model_rf.py
"""

import json
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from data_loader import (
    load_data, fit_vectorizer, full_metrics,
    print_metrics_table, TFIDF_PARAMS, N_FOLDS,
    RANDOM_STATE, OUTPUTS_DIR,
)

# ── Algorithm identity ─────────────────────────────────────────────────────────
ALGO_NAME  = "Random Forest"
ALGO_SHORT = "RF"
ALGO_COLOR = "#EF9A9A"

# ── Hyperparameters ────────────────────────────────────────────────────────────
# n_estimators = 100 : standard default for benchmarking.
# max_features = "sqrt" : uses sqrt(n_features) per split — standard for
#                         classification tasks with high-dim features.
# n_jobs = -1          : use all CPU cores (RF is parallelisable).
RF_N_ESTIMATORS = 100
RF_MAX_FEATURES = "sqrt"


def build_model() -> RandomForestClassifier:
    """Return a fresh, untrained RandomForest instance."""
    return RandomForestClassifier(
        n_estimators = RF_N_ESTIMATORS,
        max_features = RF_MAX_FEATURES,
        random_state = RANDOM_STATE,
        n_jobs       = -1,
    )


def run(data: dict = None, verbose: bool = True) -> dict:
    """
    Train and evaluate Random Forest on holdout + Stratified 5-Fold CV.

    Parameters
    ----------
    data : dict, optional
        Pre-loaded bundle from data_loader.load_data().
    verbose : bool
        Print progress and summary table.

    Returns
    -------
    dict — all results, JSON-serialisable.

    NOTE: RF on TF-IDF is slower than NB/LR/SVM. Expect ~2-5 minutes
    depending on hardware and dataset size.
    """
    if data is None:
        data = load_data()

    X_train = data["X_train"]
    X_test  = data["X_test"]
    y_train = data["y_train"]
    y_test  = data["y_test"]
    X_all   = data["X_all"]
    y_all   = data["y_all"]
    le      = data["le"]

    if verbose:
        print(f"\n{'='*60}")
        print(f"  {ALGO_NAME} ({ALGO_SHORT})")
        print(f"{'='*60}")
        print("  NOTE: RF on TF-IDF takes longer than other models (~2-5 min).")

    # ── Holdout evaluation ─────────────────────────────────────────────────────
    if verbose:
        print("[STEP 1] Holdout evaluation (70/30 split)...")

    vec        = fit_vectorizer(X_train)
    X_tr_tfidf = vec.transform(X_train)
    X_te_tfidf = vec.transform(X_test)

    model = build_model()
    model.fit(X_tr_tfidf, y_train)

    y_pred_test  = model.predict(X_te_tfidf)
    y_pred_train = model.predict(X_tr_tfidf)

    holdout = full_metrics(y_test, y_pred_test, le)
    holdout["train_accuracy"] = round(
        float((y_pred_train == y_train).mean()), 4
    )
    holdout["overfit_gap"] = round(
        holdout["train_accuracy"] - holdout["accuracy"], 4
    )

    if verbose:
        print(f"  Holdout accuracy   : {holdout['accuracy']:.4f}")
        print(f"  Bi-decision acc    : {holdout['bi_decision_acc']:.4f}")
        print(f"  Overstimulating recall: {holdout['over_recall']:.4f}")
        print(f"  Overfit gap        : {holdout['overfit_gap']:.4f}")

    # ── Stratified 5-Fold CV ───────────────────────────────────────────────────
    if verbose:
        print(f"\n[STEP 2] Stratified {N_FOLDS}-Fold Cross-Validation...")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
        ("clf",   build_model()),
    ])

    skf       = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                random_state=RANDOM_STATE)
    all_texts = list(X_all)

    fold_acc, fold_bi, fold_or, fold_f1, fold_cms = [], [], [], [], []

    for fold_i, (tr_idx, val_idx) in enumerate(skf.split(all_texts, y_all)):
        X_ft = [all_texts[i] for i in tr_idx]
        X_fv = [all_texts[i] for i in val_idx]
        y_ft = y_all[tr_idx]
        y_fv = y_all[val_idx]

        pipeline.fit(X_ft, y_ft)
        y_fp = pipeline.predict(X_fv)

        m = full_metrics(y_fv, y_fp, le)
        fold_acc.append(m["accuracy"])
        fold_bi.append(m["bi_decision_acc"])
        fold_or.append(m["over_recall"])
        fold_f1.append(m["macro_f1"])
        fold_cms.append(m["confusion_matrix"])

        if verbose:
            print(f"  Fold {fold_i+1}/{N_FOLDS}  "
                  f"acc={m['accuracy']:.4f}  "
                  f"bi={m['bi_decision_acc']:.4f}  "
                  f"over_rec={m['over_recall']:.4f}")

    agg_cm = np.sum([np.array(c) for c in fold_cms], axis=0).tolist()

    cv = dict(
        mean    = round(float(np.mean(fold_acc)), 4),
        std     = round(float(np.std(fold_acc)),  4),
        bi_mean = round(float(np.mean(fold_bi)),  4),
        bi_std  = round(float(np.std(fold_bi)),   4),
        or_mean = round(float(np.mean(fold_or)),  4),
        or_std  = round(float(np.std(fold_or)),   4),
        f1_mean = round(float(np.mean(fold_f1)),  4),
        f1_std  = round(float(np.std(fold_f1)),   4),
        fold_acc = [round(v,4) for v in fold_acc],
        fold_bi  = [round(v,4) for v in fold_bi],
        fold_or  = [round(v,4) for v in fold_or],
        fold_f1  = [round(v,4) for v in fold_f1],
        agg_confusion_matrix = agg_cm,
    )

    if verbose:
        print_metrics_table(ALGO_NAME, holdout, cv)

    result = dict(
        algo_name   = ALGO_NAME,
        algo_short  = ALGO_SHORT,
        color       = ALGO_COLOR,
        hyperparams = {
            "n_estimators": RF_N_ESTIMATORS,
            "max_features": RF_MAX_FEATURES,
        },
        holdout     = holdout,
        cv          = cv,
        generated_at = datetime.now().isoformat(),
    )

    out_path = OUTPUTS_DIR / f"result_{ALGO_SHORT.lower()}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    if verbose:
        print(f"\n[SAVED] {out_path}")

    return result


if __name__ == "__main__":
    run()
