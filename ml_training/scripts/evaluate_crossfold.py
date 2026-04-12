"""
ChildFocus — Complete Cross-Fold Algorithm Evaluation
ml_training/scripts/evaluate_crossfold.py

PURPOSE:
  This is the single entry point for the full algorithm evaluation required
  by thesis Chapter 5. It imports all five individual model modules, runs
  them on the same data with the same Stratified 5-Fold split, and produces
  one consolidated JSON file used by the HTML dashboard.

WHAT IT RUNS:
  1. Loads data once (shared across all models — no inconsistency)
  2. Runs all 5 models in sequence:
       model_cnb.py  → Complement Naïve Bayes   (proposed system component)
       model_mnb.py  → Multinomial Naïve Bayes  (baseline comparison)
       model_lr.py   → Logistic Regression       (NLP standard baseline)
       model_svm.py  → Linear SVM                (strong text classifier)
       model_rf.py   → Random Forest             (tree-based ensemble)
  3. Saves individual JSON per model: result_cnb.json, result_mnb.json, …
  4. Saves the combined comparison: algorithm_comparison_results.json
  5. Prints the complete thesis summary table to terminal

HOW TO RUN:
  cd ml_training/scripts
  py evaluate_crossfold.py

  To run a single model only:
  py model_cnb.py       ← runs CNB standalone
  py model_lr.py        ← runs LR standalone
  (etc.)

OUTPUT FILES (all saved to ml_training/outputs/):
  result_cnb.json                    ← individual CNB results
  result_mnb.json                    ← individual MNB results
  result_lr.json                     ← individual LR results
  result_svm.json                    ← individual SVM results
  result_rf.json                     ← individual RF results
  algorithm_comparison_results.json  ← combined file for the dashboard

THESIS CONTEXT:
  The evaluation follows two validation strategies as required by the panel:

  Strategy 1 — Holdout (70% train / 30% test):
    The same split used to train and evaluate the final system.
    Provides the accuracy, F1, and confusion matrix figures for Chapter 5.

  Strategy 2 — Stratified 5-Fold Cross-Validation:
    The gold standard for evaluating model stability on a 700-sample dataset.
    Each fold preserves class proportions (stratified).
    The ± standard deviation across 5 folds proves the model is stable.
    This directly addresses Sir Rogel's question: "How do you know your
    results are not due to a lucky split?"

  Both strategies use identical TF-IDF preprocessing inside a Pipeline
  object so that the vectorizer is re-fitted inside each fold — preventing
  data leakage from the validation set into the vocabulary.
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Import individual model modules ───────────────────────────────────────────
# Each module exposes: run(data, verbose) → dict
import model_cnb
import model_mnb
import model_lr
import model_svm
import model_rf

from data_loader import load_data, OUTPUTS_DIR, CLASS_ORDER

# ── Ordered for thesis presentation (weakest → strongest) ─────────────────────
# This order matches the comparison table in Chapter 5.
MODEL_MODULES = [
    model_rf,   # Rank 5 — tree-based, weakest on sparse TF-IDF
    model_mnb,  # Rank 4 — classic NB baseline
    model_svm,  # Rank 3 — strong linear classifier, high overfit gap
    model_lr,   # Rank 2 — NLP standard baseline
    model_cnb,  # Rank 1 — proposed system component (best bi-decision)
]


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE PRINTER
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_table(all_results: dict):
    """
    Print the complete Chapter 5 comparison table to the terminal.
    Columns match the thesis table format exactly.
    """
    SEP = "=" * 90
    print(f"\n{SEP}")
    print("  ChildFocus — Algorithm Comparison (Chapter 5 Table)")
    print(f"  Stratified 5-Fold Cross-Validation  |  Holdout 70/30 Split")
    print(SEP)

    # Header
    print(f"\n  {'Algorithm':<22} {'Holdout':>9} {'TrainAcc':>9} "
          f"{'CV Mean':>9} {'CV ±':>7} {'Bi-Dec':>8} "
          f"{'OverRec':>8} {'Gap':>7} {'MacroF1':>8}")
    print(f"  {'-'*86}")

    # Sort by CV mean descending
    ranked = sorted(
        all_results.items(),
        key=lambda x: (x[1]["cv"]["mean"], x[1]["holdout"]["bi_decision_acc"]),
        reverse=True,
    )

    # Track which value is best in each column (highlight with *)
    best = {
        "cv_mean":    max(v["cv"]["mean"]             for v in all_results.values()),
        "holdout":    max(v["holdout"]["accuracy"]     for v in all_results.values()),
        "bi_dec":     max(v["holdout"]["bi_decision_acc"] for v in all_results.values()),
        "over_rec":   max(v["holdout"]["over_recall"]  for v in all_results.values()),
        "macro_f1":   max(v["holdout"]["macro_f1"]     for v in all_results.values()),
        "min_gap":    min(v["holdout"]["overfit_gap"]  for v in all_results.values()),
    }

    for rank, (name, r) in enumerate(ranked, 1):
        h  = r["holdout"]
        cv = r["cv"]

        marker_cv  = "*" if cv["mean"]             == best["cv_mean"]  else " "
        marker_h   = "*" if h["accuracy"]           == best["holdout"]  else " "
        marker_bi  = "*" if h["bi_decision_acc"]    == best["bi_dec"]   else " "
        marker_or  = "*" if h["over_recall"]        == best["over_rec"] else " "
        marker_f1  = "*" if h["macro_f1"]           == best["macro_f1"] else " "
        marker_gap = "*" if h["overfit_gap"]        == best["min_gap"]  else " "

        print(
            f"  [{rank}] {name:<20} "
            f"{h['accuracy']:>8.2%}{marker_h} "
            f"{h['train_accuracy']:>8.2%} "
            f"{cv['mean']:>8.2%}{marker_cv} "
            f"±{cv['std']:>5.2%} "
            f"{h['bi_decision_acc']:>7.2%}{marker_bi} "
            f"{h['over_recall']:>7.2%}{marker_or} "
            f"{h['overfit_gap']:>6.2%}{marker_gap} "
            f"{h['macro_f1']:>7.4f}{marker_f1}"
        )

    print(f"\n  * = best value in column")
    print(f"{SEP}")

    # Per-fold breakdown for the winner
    winner_name, winner_r = ranked[0]
    print(f"\n  5-Fold Breakdown — {winner_name} (Best CV Accuracy)")
    print(f"  {'Fold':<8} {'Accuracy':>10} {'Bi-Decision':>13} {'Over Recall':>13}")
    print(f"  {'─'*46}")
    for i, (acc, bi, or_) in enumerate(zip(
        winner_r["cv"]["fold_acc"],
        winner_r["cv"]["fold_bi"],
        winner_r["cv"]["fold_or"],
    ), 1):
        print(f"  Fold {i:<4} {acc:>10.4f} {bi:>13.4f} {or_:>13.4f}")
    print(f"  {'─'*46}")
    print(f"  Mean     {winner_r['cv']['mean']:>10.4f} "
          f"{winner_r['cv']['bi_mean']:>13.4f} "
          f"{winner_r['cv']['or_mean']:>13.4f}")
    print(f"  Std      {winner_r['cv']['std']:>10.4f} "
          f"{winner_r['cv']['bi_std']:>13.4f} "
          f"{winner_r['cv']['or_std']:>13.4f}")
    print(f"{SEP}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATION RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_all(verbose_models: bool = True) -> dict:
    """
    Run all five models and consolidate results.

    Parameters
    ----------
    verbose_models : bool
        Pass verbose=True to each model's run() so fold progress is printed.

    Returns
    -------
    dict — consolidated results bundle (also saved to JSON).
    """
    print("\n" + "=" * 60)
    print("  ChildFocus — Cross-Fold Algorithm Evaluation")
    print("  All 5 classifiers · Stratified 5-Fold CV")
    print("=" * 60)

    # ── 1. Load data ONCE and share across all models ─────────────────────────
    # This guarantees identical train/test splits for fair comparison.
    print("\n[INIT] Loading dataset (shared across all models)...")
    data = load_data()

    # ── 2. Run each model ─────────────────────────────────────────────────────
    all_results = {}
    total = len(MODEL_MODULES)

    for i, module in enumerate(MODEL_MODULES, 1):
        name = module.ALGO_NAME
        print(f"\n[{i}/{total}] Running {name}...")
        result = module.run(data=data, verbose=verbose_models)
        all_results[name] = result

    # ── 3. Print consolidated summary table ───────────────────────────────────
    print_comparison_table(all_results)

    # ── 4. Build dataset metadata for the dashboard ───────────────────────────
    le = data["le"]
    classes = list(le.classes_)
    y_train = data["y_train"]
    y_test  = data["y_test"]
    y_all   = data["y_all"]

    def class_dist(y):
        return {cls: int(np.sum(y == i)) for i, cls in enumerate(classes)}

    meta = dict(
        generated_at       = datetime.now().isoformat(),
        n_train            = int(len(y_train)),
        n_test             = int(len(y_test)),
        n_total            = int(len(y_all)),
        n_folds            = 5,
        class_order        = classes,
        label_dist_train   = class_dist(y_train),
        label_dist_test    = class_dist(y_test),
        label_dist_all     = class_dist(y_all),
        split_strategy     = "Stratified 70/30 holdout + Stratified 5-Fold CV",
        tfidf_max_features = 10_000,
        tfidf_ngram_range  = [1, 2],
        random_state       = 42,
    )

    # ── 5. Save the consolidated JSON (for the HTML dashboard) ────────────────
    # Flatten each result to match the dashboard's expected schema
    algorithms_flat = {}
    for name, r in all_results.items():
        h  = r["holdout"]
        cv = r["cv"]
        algorithms_flat[name] = dict(
            short            = r["algo_short"],
            color            = r["color"],
            hyperparams      = r.get("hyperparams", {}),
            # Holdout
            holdout_acc      = h["accuracy"],
            train_acc        = h["train_accuracy"],
            overfit_gap      = h["overfit_gap"],
            holdout_bi_acc   = h["bi_decision_acc"],
            over_recall      = h["over_recall"],
            macro_f1         = h["macro_f1"],
            macro_precision  = h["macro_precision"],
            macro_recall     = h["macro_recall"],
            per_class        = h["per_class"],
            cm_holdout       = h["confusion_matrix"],
            # CV
            cv_mean          = cv["mean"],
            cv_std           = cv["std"],
            cv_bi_acc        = cv["bi_mean"],
            cv_over_recall   = cv["or_mean"],
            cv_fold_scores   = cv["fold_acc"],
            cv_fold_bi       = cv["fold_bi"],
            cv_fold_or       = cv["fold_or"],
            cv_fold_f1       = cv["fold_f1"],
            cm_cv_agg        = cv["agg_confusion_matrix"],
        )

    combined = dict(meta=meta, algorithms=algorithms_flat)
    out_path = OUTPUTS_DIR / "algorithm_comparison_results.json"
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"[DONE] Combined results saved → {out_path}")
    print(f"       Open algorithm_dashboard.html to view the thesis dashboard.\n")

    return combined


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_all()
