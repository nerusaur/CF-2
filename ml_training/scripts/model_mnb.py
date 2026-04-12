"""
ChildFocus — Model: Multinomial Naïve Bayes (MNB)
ml_training/scripts/model_mnb.py

WHY MNB IS INCLUDED AS A COMPARISON:
  Multinomial NB is the classical baseline for text classification.
  It is included to directly justify the choice of CNB over its predecessor.
  MNB assumes each feature is drawn from a multinomial distribution based on
  its raw count (or TF-IDF weight), while CNB uses complement statistics.
  For datasets where some classes are harder to distinguish (e.g. Neutral),
  CNB typically outperforms MNB because it directly models what each class
  is NOT — a better fit for three-class OIR separation.

  In thesis context: MNB is Rank 4 in the comparison table. Its lower
  bi-decision accuracy vs CNB directly justifies CNB selection.

RUN STANDALONE:
  cd ml_training/scripts
  py model_mnb.py
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from data_loader import (
    load_data, fit_vectorizer, full_metrics,
    print_metrics_table, TFIDF_PARAMS, N_FOLDS,
    RANDOM_STATE, OUTPUTS_DIR,
)

# ── Algorithm identity ─────────────────────────────────────────────────────────
ALGO_NAME  = "Multinomial NB"
ALGO_SHORT = "MNB"
ALGO_COLOR = "#81C784"

# ── Hyperparameters ────────────────────────────────────────────────────────────
# alpha = Laplace smoothing. 1.0 is the standard default.
MNB_ALPHA = 1.0


def build_model() -> MultinomialNB:
    """Return a fresh, untrained MNB instance."""
    return MultinomialNB(alpha=MNB_ALPHA)


def run(data: dict = None, verbose: bool = True) -> dict:
    """
    Train and evaluate MNB on holdout + Stratified 5-Fold CV.

    Parameters
    ----------
    data : dict, optional
        Pre-loaded bundle from data_loader.load_data().
    verbose : bool
        Print progress and summary table.

    Returns
    -------
    dict — all results, JSON-serialisable.
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

    # ── Holdout evaluation ─────────────────────────────────────────────────────
    if verbose:
        print("[STEP 1] Holdout evaluation (70/30 split)...")

    vec         = fit_vectorizer(X_train)
    X_tr_tfidf  = vec.transform(X_train)
    X_te_tfidf  = vec.transform(X_test)

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
        hyperparams = {"alpha": MNB_ALPHA},
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
