"""
ChildFocus — Model: Logistic Regression (LR)
ml_training/scripts/model_lr.py

WHY LR IS INCLUDED AS A COMPARISON:
  Logistic Regression is the most common text classification baseline in
  academic NLP research. It models the log-odds of each class as a linear
  combination of TF-IDF features. It is regularised by the C parameter
  (inverse of regularisation strength; higher C = less regularisation).

  LR typically performs competitively with or slightly above NB on holdout
  accuracy for text tasks. However, CNB still leads on bi-decision accuracy
  and has a smaller overfit gap — both more important for ChildFocus than
  raw three-class holdout accuracy.

  Thesis comparison point: "Although LR achieves comparable holdout accuracy,
  CNB outperforms it on the operational bi-decision metric and demonstrates
  better generalisation as measured by the overfit gap."

RUN STANDALONE:
  cd ml_training/scripts
  py model_lr.py
"""

import json
import numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from data_loader import (
    load_data, fit_vectorizer, full_metrics,
    print_metrics_table, TFIDF_PARAMS, N_FOLDS,
    RANDOM_STATE, OUTPUTS_DIR,
)

# ── Algorithm identity ─────────────────────────────────────────────────────────
ALGO_NAME  = "Logistic Regression"
ALGO_SHORT = "LR"
ALGO_COLOR = "#FFB74D"

# ── Hyperparameters ────────────────────────────────────────────────────────────
# C=1.0   : default regularisation — balanced fit vs. generalisation.
# solver  : lbfgs handles multi-class natively (faster than liblinear for 3+).
# max_iter: increased to 1000 to ensure convergence on large TF-IDF vectors.
LR_C        = 1.0
LR_SOLVER   = "lbfgs"
LR_MAX_ITER = 1000


def build_model() -> LogisticRegression:
    """Return a fresh, untrained LR instance."""
    return LogisticRegression(
        C        = LR_C,
        solver   = LR_SOLVER,
        max_iter = LR_MAX_ITER,
        random_state = RANDOM_STATE,
    )


def run(data: dict = None, verbose: bool = True) -> dict:
    """
    Train and evaluate LR on holdout + Stratified 5-Fold CV.

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
        hyperparams = {"C": LR_C, "solver": LR_SOLVER, "max_iter": LR_MAX_ITER},
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
