"""
ChildFocus — Model: Linear SVM (SVM)
ml_training/scripts/model_svm.py

WHY SVM IS INCLUDED AS A COMPARISON:
  Support Vector Machines with a linear kernel are widely regarded as one of
  the strongest traditional text classifiers. SVM maximises the margin between
  class boundaries in the high-dimensional TF-IDF feature space. For text,
  LinearSVC (which implements a primal SVM via coordinate descent) is preferred
  over RBF-kernel SVM because text features are already linearly separable at
  high dimensionality.

  SVM typically achieves near-perfect training accuracy (it maximises margin,
  so it fits training data very tightly). This means its overfit gap is large —
  an important thesis talking point. CNB, by contrast, has a much smaller gap,
  demonstrating better generalisation even if SVM's raw test accuracy is
  comparable.

  Note for committee: SVM does not produce probability estimates by default
  (LinearSVC uses decision_function). For the bi-decision metric we use the
  predicted class label directly, not probability thresholding.

RUN STANDALONE:
  cd ml_training/scripts
  py model_svm.py
"""

import json
import numpy as np
from datetime import datetime
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from data_loader import (
    load_data, fit_vectorizer, full_metrics,
    print_metrics_table, TFIDF_PARAMS, N_FOLDS,
    RANDOM_STATE, OUTPUTS_DIR,
)

# ── Algorithm identity ─────────────────────────────────────────────────────────
ALGO_NAME  = "Linear SVM"
ALGO_SHORT = "SVM"
ALGO_COLOR = "#CE93D8"

# ── Hyperparameters ────────────────────────────────────────────────────────────
# C=1.0     : regularisation strength (higher = tighter fit to training data).
# max_iter  : 2000 ensures convergence on large TF-IDF feature sets.
SVM_C        = 1.0
SVM_MAX_ITER = 2000


def build_model() -> LinearSVC:
    """Return a fresh, untrained LinearSVC instance."""
    return LinearSVC(
        C        = SVM_C,
        max_iter = SVM_MAX_ITER,
        random_state = RANDOM_STATE,
    )


def run(data: dict = None, verbose: bool = True) -> dict:
    """
    Train and evaluate Linear SVM on holdout + Stratified 5-Fold CV.

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
        if holdout["train_accuracy"] >= 0.99:
            print(f"  NOTE: SVM train acc ≈ 100% — this is expected (margin maximisation).")
            print(f"        The high overfit gap is a known SVM characteristic, not an error.")

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
        hyperparams = {"C": SVM_C, "max_iter": SVM_MAX_ITER},
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
