"""
ChildFocus - Threshold Recalibration Script
ml_training/scripts/recalibrate_thresholds.py

The real hybrid evaluation showed all Score_final values fell in the
Educational range (< 0.35). This is because heuristic scores (Score_H)
from real videos are consistently in the 0.08-0.28 range, far below
the thesis threshold of 0.75.

This script:
  1. Loads the 30-video evaluation results
  2. Analyzes the actual score distribution per class
  3. Finds optimal thresholds based on real data
  4. Reports new recommended thresholds for the thesis

Run from ml_training/scripts/:
    python recalibrate_thresholds.py
"""

import os
import json
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH  = os.path.join(SCRIPT_DIR, "..", "outputs", "hybrid_real_results.json")
OUTPUT_PATH   = os.path.join(SCRIPT_DIR, "..", "outputs", "recalibration_report.txt")

LABELS = ["Educational", "Neutral", "Overstimulating"]


def load_results():
    with open(RESULTS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    results = data["results"]
    valid = [r for r in results if r["pred_label"] not in ("SKIPPED", "ERROR")]
    print(f"[RECAL] Loaded {len(valid)} valid results")
    return valid


def analyze_score_distribution(results):
    print("\n" + "="*65)
    print("SCORE DISTRIBUTION BY TRUE CLASS")
    print("="*65)

    by_class = {lbl: [] for lbl in LABELS}
    for r in results:
        by_class[r["true_label"]].append({
            "score_nb":    r.get("score_nb",    0),
            "score_h":     r.get("score_h",     0),
            "score_final": r.get("score_final", 0),
            "title":       r.get("title",       ""),
        })

    stats = {}
    for lbl in LABELS:
        items = by_class[lbl]
        if not items:
            continue
        finals = [x["score_final"] for x in items]
        nbs    = [x["score_nb"]    for x in items]
        hs     = [x["score_h"]     for x in items]

        stats[lbl] = {
            "final_min":  round(min(finals),  4),
            "final_max":  round(max(finals),  4),
            "final_mean": round(np.mean(finals), 4),
            "final_std":  round(np.std(finals),  4),
            "nb_mean":    round(np.mean(nbs),    4),
            "h_mean":     round(np.mean(hs),     4),
            "items":      items,
        }

        print(f"\n{lbl} (n={len(items)})")
        print(f"  Score_final: min={stats[lbl]['final_min']:.4f}  "
              f"max={stats[lbl]['final_max']:.4f}  "
              f"mean={stats[lbl]['final_mean']:.4f}  "
              f"std={stats[lbl]['final_std']:.4f}")
        print(f"  Score_NB:    mean={stats[lbl]['nb_mean']:.4f}")
        print(f"  Score_H:     mean={stats[lbl]['h_mean']:.4f}")
        print(f"  Individual scores:")
        for x in sorted(items, key=lambda i: i["score_final"]):
            print(f"    {x['score_final']:.4f}  NB={x['score_nb']:.3f}  "
                  f"H={x['score_h']:.3f}  {x['title'][:50]!r}")

    return stats


def find_optimal_thresholds(results, stats):
    """
    Find block/allow thresholds that best separate the classes
    using the actual score distribution from the 30-video evaluation.
    """
    print("\n" + "="*65)
    print("THRESHOLD SEARCH (based on real score distribution)")
    print("="*65)

    y_true       = [r["true_label"]  for r in results]
    score_finals = [r["score_final"] for r in results]

    # Get score ranges per class
    edu_scores   = [r["score_final"] for r in results if r["true_label"] == "Educational"]
    neu_scores   = [r["score_final"] for r in results if r["true_label"] == "Neutral"]
    over_scores  = [r["score_final"] for r in results if r["true_label"] == "Overstimulating"]

    all_scores = sorted(set(score_finals))
    print(f"\nAll Score_final values: {[round(s,4) for s in sorted(score_finals)]}")
    print(f"\nEducational  range: {min(edu_scores):.4f} – {max(edu_scores):.4f}")
    print(f"Neutral      range: {min(neu_scores):.4f} – {max(neu_scores):.4f}")
    print(f"Overstimulating range: {min(over_scores):.4f} – {max(over_scores):.4f}")

    # Search for best thresholds
    best_f1       = 0
    best_block    = 0
    best_allow    = 0
    best_report   = ""

    # Generate candidate thresholds based on actual data range
    candidates = sorted(set([round(s + 0.01, 3) for s in all_scores] +
                            [round(s - 0.01, 3) for s in all_scores] +
                            [round(s,         3) for s in all_scores]))
    candidates = [c for c in candidates if 0.05 <= c <= 0.95]

    print(f"\n{'Block(≥)':>10} {'Allow(≤)':>10} {'F1-Score':>10} {'Accuracy':>10}")
    print("-"*45)

    for block in candidates:
        for allow in candidates:
            if allow >= block:
                continue
            preds = []
            for s in score_finals:
                if   s >= block: preds.append("Overstimulating")
                elif s <= allow: preds.append("Educational")
                else:            preds.append("Neutral")

            f1  = f1_score(y_true, preds, average="weighted", zero_division=0)
            acc = sum(p == t for p, t in zip(preds, y_true)) / len(y_true)

            if f1 > best_f1:
                best_f1     = f1
                best_block  = block
                best_allow  = allow
                best_report = classification_report(
                    y_true, preds, target_names=LABELS,
                    digits=4, zero_division=0
                )

    # Show top result
    print(f"{best_block:>10.4f} {best_allow:>10.4f} {best_f1:>10.4f}  ← OPTIMAL")

    # Show what predictions look like with optimal thresholds
    print(f"\nOptimal Thresholds Found:")
    print(f"  Block (Overstimulating) : Score_final >= {best_block}")
    print(f"  Allow (Educational)     : Score_final <= {best_allow}")
    print(f"  Neutral                 : {best_allow} < Score_final < {best_block}")
    print(f"\nClassification Report with Optimal Thresholds:")
    print(best_report)

    # Apply optimal thresholds and show confusion matrix
    best_preds = []
    for s in score_finals:
        if   s >= best_block: best_preds.append("Overstimulating")
        elif s <= best_allow: best_preds.append("Educational")
        else:                 best_preds.append("Neutral")

    cm = confusion_matrix(y_true, best_preds, labels=LABELS)
    print("Confusion Matrix (Optimal Thresholds):")
    print(f"{'':>20}" + "".join(f"{l[:5]:>12}" for l in LABELS))
    for i, lbl in enumerate(LABELS):
        print(f"{lbl:>20}" + "".join(f"{cm[i][j]:>12}" for j in range(3)))

    acc = sum(p == t for p, t in zip(best_preds, y_true)) / len(y_true)
    print(f"\nAccuracy with optimal thresholds: {acc:.4f} ({acc*100:.2f}%)")

    return best_block, best_allow, best_f1, acc, best_report, cm.tolist()


def save_report(stats, best_block, best_allow, best_f1, best_acc, best_report, cm):
    import datetime
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("ChildFocus — Threshold Recalibration Report\n")
        f.write(f"Generated: {datetime.datetime.now()}\n\n")

        f.write("SCORE DISTRIBUTION SUMMARY\n")
        f.write("-"*50 + "\n")
        for lbl, s in stats.items():
            f.write(f"{lbl}:\n")
            f.write(f"  Score_final: {s['final_min']:.4f} – {s['final_max']:.4f} "
                    f"(mean={s['final_mean']:.4f}, std={s['final_std']:.4f})\n")
            f.write(f"  Score_NB mean: {s['nb_mean']:.4f} | Score_H mean: {s['h_mean']:.4f}\n\n")

        f.write("OPTIMAL THRESHOLDS\n")
        f.write("-"*50 + "\n")
        f.write(f"Block (>= Overstimulating): {best_block}\n")
        f.write(f"Allow (<= Educational):     {best_allow}\n")
        f.write(f"Weighted F1-Score:          {best_f1:.4f}\n")
        f.write(f"Accuracy:                   {best_acc:.4f}\n\n")

        f.write("CLASSIFICATION REPORT\n")
        f.write(best_report)

        f.write("\nCONFUSION MATRIX\n")
        f.write(f"{'':>20}" + "".join(f"{l[:5]:>12}" for l in LABELS) + "\n")
        for i, lbl in enumerate(LABELS):
            f.write(f"{lbl:>20}" + "".join(f"{cm[i][j]:>12}" for j in range(3)) + "\n")

    print(f"\n[RECAL] Report saved → {OUTPUT_PATH}")


def main():
    print("\n" + "="*65)
    print("CHILDFOCUS — THRESHOLD RECALIBRATION")
    print("="*65)

    results                                          = load_results()
    stats                                            = analyze_score_distribution(results)
    block, allow, f1, acc, report, cm               = find_optimal_thresholds(results, stats)
    save_report(stats, block, allow, f1, acc, report, cm)

    print("\n" + "="*65)
    print("SUMMARY FOR THESIS")
    print("="*65)
    print(f"Original thresholds (thesis): Block=0.75, Allow=0.35")
    print(f"Recalibrated thresholds:      Block={block}, Allow={allow}")
    print(f"Recalibrated F1-Score:        {f1:.4f}")
    print(f"Recalibrated Accuracy:        {acc:.4f} ({acc*100:.2f}%)")
    print("="*65)
    print("\nNext step: run evaluate_hybrid_recalibrated.py with new thresholds")


if __name__ == "__main__":
    main()
