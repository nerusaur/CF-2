"""
ChildFocus — Model: Complement Naïve Bayes (CNB)
ml_training/scripts/model_cnb.py

WHY CNB FOR CHILDFOCUS:
  Complement NB is designed for imbalanced or multi-class text classification.
  Unlike standard Multinomial NB, it trains each class using the complement
  of that class (all other classes), making it more robust when one class
  dominates. Here it achieves the highest bi-decision accuracy and the
  smallest overfit gap — making it the thesis' recommended classifier.

THESIS FORMULA:
  Score_NB = P(Overstimulating | title, tags, description)
  The probability output of CNB becomes Score_NB in the hybrid fusion:
    Score_final = (0.70 × Score_NB) + (0.30 × Score_H)

RUN STANDALONE:
  cd ml_training/scripts
  py model_cnb.py
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from data_loader import (
    load_data, fit_vectorizer, full_metrics,
    print_metrics_table, TFIDF_PARAMS, N_FOLDS,
    RANDOM_STATE, OUTPUTS_DIR, CLASS_ORDER,
)

# ── Algorithm identity ─────────────────────────────────────────────────────────
ALGO_NAME  = "Complement NB"
ALGO_SHORT = "CNB"
ALGO_COLOR = "#4FC3F7"

# ── Hyperparameters ────────────────────────────────────────────────────────────
# alpha = Laplace smoothing parameter.
# 1.0 is the standard default; lower values (e.g. 0.5) can help with very
# sparse text but increase the risk of zero-probability issues.
CNB_ALPHA = 1.0


def build_model() -> ComplementNB:
    """
    Return a fresh, untrained CNB instance.
    Called by evaluate_crossfold.py to ensure consistent hyperparameters.
    """
    return ComplementNB(alpha=CNB_ALPHA)


def run(data: dict = None, verbose: bool = True) -> dict:
    """
    Train and evaluate CNB on holdout + Stratified 5-Fold CV.

    Parameters
    ----------
    data : dict, optional
        Pre-loaded data bundle from data_loader.load_data().
        If None, load_data() is called here (standalone mode).
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

    # ── Step 1: Holdout evaluation (70/30 split) ───────────────────────────────
    # Vectorizer is fitted on X_train only — test set must NOT influence vocab.
    if verbose:
        print("[STEP 1] Holdout evaluation (70/30 split)...")

    vec          = fit_vectorizer(X_train)
    X_tr_tfidf   = vec.transform(X_train)
    X_te_tfidf   = vec.transform(X_test)

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

    # ── Step 2: Stratified 5-Fold Cross-Validation ────────────────────────────
    # Pipeline ensures TF-IDF is re-fitted inside each fold (no data leakage).
    if verbose:
        print(f"\n[STEP 2] Stratified {N_FOLDS}-Fold Cross-Validation...")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
        ("clf",   build_model()),
    ])

    skf        = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                 random_state=RANDOM_STATE)
    all_texts  = list(X_all)

    fold_acc     = []
    fold_bi      = []
    fold_or      = []   # over recall
    fold_f1      = []
    fold_cms     = []

    for fold_i, (tr_idx, val_idx) in enumerate(skf.split(all_texts, y_all)):
        X_fold_tr  = [all_texts[i] for i in tr_idx]
        X_fold_val = [all_texts[i] for i in val_idx]
        y_fold_tr  = y_all[tr_idx]
        y_fold_val = y_all[val_idx]

        pipeline.fit(X_fold_tr, y_fold_tr)
        y_fold_pred = pipeline.predict(X_fold_val)

        m = full_metrics(y_fold_val, y_fold_pred, le)
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

    # Aggregate confusion matrix across all 5 folds
    import numpy as np
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
        fold_acc       = [round(v, 4) for v in fold_acc],
        fold_bi        = [round(v, 4) for v in fold_bi],
        fold_or        = [round(v, 4) for v in fold_or],
        fold_f1        = [round(v, 4) for v in fold_f1],
        agg_confusion_matrix = agg_cm,
    )

    if verbose:
        print_metrics_table(ALGO_NAME, holdout, cv)

    # ── Step 3: Bundle and save ────────────────────────────────────────────────
    result = dict(
        algo_name  = ALGO_NAME,
        algo_short = ALGO_SHORT,
        color      = ALGO_COLOR,
        hyperparams = {"alpha": CNB_ALPHA},
        holdout    = holdout,
        cv         = cv,
        generated_at = datetime.now().isoformat(),
    )

    out_path = OUTPUTS_DIR / f"result_{ALGO_SHORT.lower()}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    if verbose:
        print(f"\n[SAVED] {out_path}")

    return result


# ── Standalone entry point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    run()
