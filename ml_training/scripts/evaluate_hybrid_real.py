"""
ChildFocus - Real Hybrid Evaluation Script
ml_training/scripts/evaluate_hybrid_real.py

Runs the ACTUAL hybrid pipeline (real video download + heuristic)
on a stratified sample of 30 videos (10 per class).

This produces real Score_H values from frame_sampler.py,
giving genuine hybrid metrics for the thesis.

Run from ml_training/scripts/:
    python evaluate_hybrid_real.py

NOTE: Requires internet connection. Each video takes ~2-5 minutes.
      Progress is auto-saved after every video.
"""

import os
import sys
import csv
import json
import time
import random
import datetime

# ── Add backend to path ────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
BACKEND_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "backend"))
sys.path.insert(0, BACKEND_PATH)

DATA_PATH    = os.path.join(SCRIPT_DIR, "data", "processed", "metadata_clean.csv")
OUTPUTS_DIR  = os.path.join(SCRIPT_DIR, "..", "outputs")
PROGRESS_PATH = os.path.join(OUTPUTS_DIR, "hybrid_eval_progress.json")
RESULTS_PATH  = os.path.join(OUTPUTS_DIR, "hybrid_real_results.json")
REPORT_PATH   = os.path.join(OUTPUTS_DIR, "hybrid_real_report.txt")

LABELS            = ["Educational", "Neutral", "Overstimulating"]
RANDOM_STATE      = 42
SAMPLES_PER_CLASS = 10   # 10 per class = 30 total

random.seed(RANDOM_STATE)

# ── Fusion config (thesis values) ─────────────────────────────────────────
ALPHA           = 0.4   # NB weight
BETA            = 0.6   # Heuristic weight
THRESHOLD_BLOCK = 0.75
THRESHOLD_ALLOW = 0.35


def load_labeled_videos():
    rows = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label    = row.get("label", "").strip()
            video_id = row.get("video_id", "").strip()
            title    = row.get("title", "").strip()
            text     = row.get("text",  "").strip()
            if label in LABELS and video_id and len(video_id) == 11:
                rows.append({
                    "video_id": video_id,
                    "label":    label,
                    "title":    title,
                    "text":     text,
                })
    print(f"[HYBRID_EVAL] Loaded {len(rows)} videos with valid IDs")
    for lbl in LABELS:
        count = sum(1 for r in rows if r["label"] == lbl)
        print(f"[HYBRID_EVAL]   {lbl}: {count}")
    return rows


def stratified_sample(rows):
    by_class = {lbl: [] for lbl in LABELS}
    for r in rows:
        by_class[r["label"]].append(r)

    sample = []
    for lbl in LABELS:
        pool = by_class[lbl][:]
        random.shuffle(pool)
        selected = pool[:SAMPLES_PER_CLASS]
        sample.extend(selected)
        print(f"[HYBRID_EVAL]   Selected {len(selected)} {lbl} videos")

    random.shuffle(sample)
    return sample


def load_progress():
    """Load previously completed results to resume interrupted run."""
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        completed_ids = {r["video_id"] for r in data}
        print(f"[HYBRID_EVAL] Resuming — {len(data)} already done: {completed_ids}")
        return data, completed_ids
    return [], set()


def save_progress(results):
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def run_hybrid_on_video(video: dict) -> dict:
    """Run the actual classify pipeline on one video."""
    try:
        from app.modules.frame_sampler import sample_video
        from app.modules.heuristic     import compute_heuristic_score
        from app.modules.naive_bayes   import score_from_metadata_dict

        video_id      = video["video_id"]
        video_url     = f"https://www.youtube.com/watch?v={video_id}"
        thumbnail_url = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"
        t_start       = time.time()

        # ── Step 1: Download + sample video ───────────────────────────────
        sample        = sample_video(video_url,
                                     thumbnail_url=thumbnail_url,
                                     hint_title=video.get("title", ""))
        sample_status = sample.get("status", "error")

        if sample_status in ("unavailable", "error"):
            reason = sample.get("reason", sample.get("message", "unavailable"))
            print(f"  ✗ SKIP [{video['label']:>15}] {video_id} — {reason[:60]}")
            return {
                "video_id":   video_id,
                "title":      video.get("title", "")[:60],
                "true_label": video["label"],
                "pred_label": "SKIPPED",
                "status":     sample_status,
                "reason":     reason,
            }

        # ── Step 2: Heuristic score ────────────────────────────────────────
        h_result = compute_heuristic_score(sample)
        score_h  = h_result["score_h"]

        # ── Step 3: NB score ───────────────────────────────────────────────
        nb_result = score_from_metadata_dict({
            "title":       sample.get("video_title", "") or video.get("title", ""),
            "tags":        sample.get("tags", []),
            "description": sample.get("description", ""),
        })
        score_nb = nb_result.score_nb

        # ── Step 4: Fusion ─────────────────────────────────────────────────
        score_final = round((ALPHA * score_nb) + (BETA * score_h), 4)

        if   score_final >= THRESHOLD_BLOCK: pred_label = "Overstimulating"
        elif score_final <= THRESHOLD_ALLOW: pred_label = "Educational"
        else:                                pred_label = "Neutral"

        runtime = round(time.time() - t_start, 2)
        correct = "✓" if pred_label == video["label"] else "✗"
        path    = "FULL" if sample_status == "success" else "THUMB"

        print(f"  {correct} [{video['label']:>15} → {pred_label:>15}] "
              f"NB={score_nb:.3f} H={score_h:.3f} "
              f"Final={score_final:.3f} [{path}] "
              f"({runtime}s) {video['title'][:35]!r}")

        # Segment details for logging
        segments = sample.get("segments", [])
        seg_data = []
        for s in segments:
            if s:
                seg_data.append({
                    "segment_id": s.get("segment_id"),
                    "fcr":        s.get("fcr"),
                    "csv":        s.get("csv"),
                    "att":        s.get("att"),
                    "score_h":    s.get("score_h"),
                })

        return {
            "video_id":    video_id,
            "title":       video.get("title", "")[:80],
            "true_label":  video["label"],
            "pred_label":  pred_label,
            "score_nb":    round(score_nb,    4),
            "score_h":     round(score_h,     4),
            "score_final": score_final,
            "nb_confidence": round(nb_result.confidence, 4),
            "sample_path": sample_status,
            "segments":    seg_data,
            "thumbnail_intensity": sample.get("thumbnail_intensity", 0.0),
            "status":      "success",
            "runtime":     runtime,
        }

    except Exception as e:
        import traceback
        print(f"  ✗ ERROR {video['video_id']}: {e}")
        traceback.print_exc()
        return {
            "video_id":   video["video_id"],
            "title":      video.get("title", "")[:60],
            "true_label": video["label"],
            "pred_label": "ERROR",
            "status":     "error",
            "reason":     str(e),
        }


def compute_metrics(results):
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        precision_recall_fscore_support, accuracy_score
    )

    valid = [r for r in results
             if r["pred_label"] not in ("SKIPPED", "ERROR")]
    skipped = len(results) - len(valid)

    print(f"\n{'='*60}")
    print("REAL HYBRID EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Total videos attempted : {len(results)}")
    print(f"Successfully evaluated : {len(valid)}")
    print(f"Skipped / Error        : {skipped}")

    if len(valid) < 5:
        print("[METRICS] Not enough valid results.")
        return {}

    y_true = [r["true_label"] for r in valid]
    y_pred = [r["pred_label"] for r in valid]

    report = classification_report(y_true, y_pred,
                                   target_names=LABELS,
                                   digits=4,
                                   zero_division=0)
    cm     = confusion_matrix(y_true, y_pred, labels=LABELS)
    acc    = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    prec_per, rec_per, f1_per, sup_per = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )

    print(f"\nFusion: Score_final = ({ALPHA} x NB) + ({BETA} x Heuristic)")
    print(f"Thresholds: Block >= {THRESHOLD_BLOCK} | Allow <= {THRESHOLD_ALLOW}\n")
    print("Classification Report (Real Hybrid):")
    print(report)
    print("Confusion Matrix:")
    header = f"{'':>20}" + "".join(f"{l[:5]:>12}" for l in LABELS)
    print(header)
    for i, row_label in enumerate(LABELS):
        row_str = f"{row_label:>20}" + "".join(f"{cm[i][j]:>12}" for j in range(3))
        print(row_str)
    print(f"\nOverall Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Weighted Precision: {prec:.4f}")
    print(f"Weighted Recall   : {rec:.4f}")
    print(f"Weighted F1-Score : {f1:.4f}")

    # Path breakdown
    full_count  = sum(1 for r in valid if r.get("sample_path") == "success")
    thumb_count = sum(1 for r in valid if r.get("sample_path") == "thumbnail_only")
    print(f"\nEvaluation Path Breakdown:")
    print(f"  Full video analysis  : {full_count}")
    print(f"  Thumbnail-only       : {thumb_count}")

    return {
        "n_total":   len(results),
        "n_valid":   len(valid),
        "n_skipped": skipped,
        "n_full":    full_count,
        "n_thumb":   thumb_count,
        "accuracy":  round(acc,  4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1":        round(f1,   4),
        "per_class": {
            lbl: {
                "precision": round(float(prec_per[i]), 4),
                "recall":    round(float(rec_per[i]),  4),
                "f1":        round(float(f1_per[i]),   4),
                "support":   int(sup_per[i]),
            }
            for i, lbl in enumerate(LABELS)
        },
        "confusion_matrix": cm.tolist(),
        "report":    report,
        "y_true":    y_true,
        "y_pred":    y_pred,
    }


def save_report(results, metrics):
    # ── JSON ──────────────────────────────────────────────────────────────
    json_data = {
        "generated": datetime.datetime.now().isoformat(),
        "config": {
            "alpha_nb":       ALPHA,
            "beta_heuristic": BETA,
            "threshold_block": THRESHOLD_BLOCK,
            "threshold_allow": THRESHOLD_ALLOW,
            "samples_per_class": SAMPLES_PER_CLASS,
        },
        "metrics": {k: v for k, v in metrics.items()
                    if k not in ("report", "y_true", "y_pred")},
        "results": results,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    # ── Human-readable report ─────────────────────────────────────────────
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("ChildFocus — Real Hybrid Model Evaluation Report\n")
        f.write(f"Generated : {datetime.datetime.now()}\n")
        f.write(f"Alpha(NB) : {ALPHA} | Beta(H): {BETA}\n")
        f.write(f"Thresholds: Block >= {THRESHOLD_BLOCK} | Allow <= {THRESHOLD_ALLOW}\n")
        f.write("="*60 + "\n\n")

        if "report" in metrics:
            f.write("CLASSIFICATION REPORT\n")
            f.write(metrics["report"])
            f.write(f"\nAccuracy  : {metrics['accuracy']:.4f}\n")
            f.write(f"Precision : {metrics['precision']:.4f}\n")
            f.write(f"Recall    : {metrics['recall']:.4f}\n")
            f.write(f"F1-Score  : {metrics['f1']:.4f}\n\n")

            f.write("CONFUSION MATRIX\n")
            f.write(f"{'':>20}" + "".join(f"{l[:5]:>12}" for l in LABELS) + "\n")
            cm = metrics["confusion_matrix"]
            for i, lbl in enumerate(LABELS):
                f.write(f"{lbl:>20}" + "".join(f"{cm[i][j]:>12}" for j in range(3)) + "\n")
            f.write("\n")

        f.write("PER-VIDEO RESULTS\n")
        f.write("-"*90 + "\n")
        f.write(f"{'video_id':>12} {'true_label':>16} {'pred':>16} "
                f"{'NB':>7} {'H':>7} {'Final':>7} {'path':>8}  title\n")
        f.write("-"*90 + "\n")
        for r in results:
            mark = "OK" if r["pred_label"] == r["true_label"] else "XX"
            f.write(
                f"{r['video_id']:>12} {r['true_label']:>16} "
                f"{r['pred_label']:>16} "
                f"{r.get('score_nb', 0):>7.4f} "
                f"{r.get('score_h', 0):>7.4f} "
                f"{r.get('score_final', 0):>7.4f} "
                f"{r.get('sample_path','?'):>8} "
                f"[{mark}] {r.get('title','')[:40]}\n"
            )

    print(f"\n[HYBRID_EVAL] Results JSON  → {RESULTS_PATH}")
    print(f"[HYBRID_EVAL] Report TXT    → {REPORT_PATH}")
    print(f"[HYBRID_EVAL] Progress file → {PROGRESS_PATH}")


def main():
    print("\n" + "="*60)
    print("CHILDFOCUS — REAL HYBRID EVALUATION (30 VIDEOS)")
    print("="*60)
    print(f"Config: alpha={ALPHA}, beta={BETA}, "
          f"block>={THRESHOLD_BLOCK}, allow<={THRESHOLD_ALLOW}")
    print(f"Sample: {SAMPLES_PER_CLASS} videos per class = {SAMPLES_PER_CLASS*3} total\n")

    rows   = load_labeled_videos()
    sample = stratified_sample(rows)

    # Resume support
    completed_results, completed_ids = load_progress()
    remaining = [v for v in sample if v["video_id"] not in completed_ids]

    print(f"\n[HYBRID_EVAL] Videos to process: {len(remaining)} "
          f"({len(completed_ids)} already done)\n")

    results = list(completed_results)
    t_total = time.time()

    for i, video in enumerate(remaining, 1):
        idx = len(completed_ids) + i
        print(f"\n[{idx:>2}/{SAMPLES_PER_CLASS*3}] video_id={video['video_id']} "
              f"| class={video['label']}")
        print(f"       title: {video['title'][:60]!r}")

        result = run_hybrid_on_video(video)
        results.append(result)
        completed_ids.add(video["video_id"])
        save_progress(results)

        elapsed = time.time() - t_total
        remaining_count = len(remaining) - i
        avg_time = elapsed / i
        eta = avg_time * remaining_count
        print(f"       Elapsed: {elapsed/60:.1f}min | "
              f"ETA: ~{eta/60:.1f}min remaining")

    print(f"\n[HYBRID_EVAL] All {len(results)} videos processed.")
    metrics = compute_metrics(results)
    save_report(results, metrics)

    # ── Final summary ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("FINAL SUMMARY — PASTE THESE INTO YOUR THESIS")
    print("="*60)
    if metrics:
        print(f"Sample size  : {metrics['n_valid']} valid / {metrics['n_total']} attempted")
        print(f"Accuracy     : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Precision    : {metrics['precision']:.4f}")
        print(f"Recall       : {metrics['recall']:.4f}")
        print(f"F1-Score     : {metrics['f1']:.4f}")
        print(f"\nPer-class:")
        for lbl, m in metrics["per_class"].items():
            print(f"  {lbl:<18}: P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}  n={m['support']}")
        cm = metrics["confusion_matrix"]
        print(f"\nConfusion Matrix:")
        print(f"{'':>20}" + "".join(f"{l[:5]:>12}" for l in LABELS))
        for i, lbl in enumerate(LABELS):
            print(f"{lbl:>20}" + "".join(f"{cm[i][j]:>12}" for j in range(3)))
    print("="*60)
    print("\n[HYBRID_EVAL] Complete. Copy results above for Chapter 5.")


if __name__ == "__main__":
    main()