"""
ChildFocus - Real Hybrid Evaluation Script  (v4 — merged)
ml_training/scripts/evaluate_hybrid_real.py

Evaluates all 210 videos in test_clean.csv — the 30% NB held-out split.
None of these videos were used during NB training (no contamination).

Changes from v3 (_original.py):
  - Added scrape_ytInitialData_keywords() tag enrichment for every video,
    including thumbnail-only fallbacks — mirrors enrich_dataset.py so NB
    input at eval time matches NB input at training time.
    (export from browser via "Get cookies.txt LOCALLY" extension) so yt-dlp
    can bypass region locks and age gates that caused "not available" errors.
  - _fuse() returns 3-tuple (score_final, pred, eff_alpha) — consistent with
    original and avoids double-calculation of eff_alpha.
  - Result dict uses "effective_alpha" key (matches save_report column header).
  - Output files kept as hybrid_full_* for continuity with previous runs.

Run from ml_training/scripts/:
    python evaluate_hybrid_real.py

Delete old progress before re-running after a retrain:
    rm ../outputs/hybrid_full_progress.json
    rm ../outputs/hybrid_full_results.json
"""

import os
import sys
import csv
import json
import time
import random
import datetime

# ── Add backend to path ────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
BACKEND_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "backend"))
sys.path.insert(0, BACKEND_PATH)

# ── Source: test_clean.csv ONLY (NB has never seen these) ─────────────────────
DATA_PATH     = os.path.join(SCRIPT_DIR, "data", "processed", "test_clean.csv")
OUTPUTS_DIR   = os.path.join(SCRIPT_DIR, "..", "outputs")
PROGRESS_PATH = os.path.join(OUTPUTS_DIR, "hybrid_full_progress.json")
RESULTS_PATH  = os.path.join(OUTPUTS_DIR, "hybrid_full_results.json")
REPORT_PATH   = os.path.join(OUTPUTS_DIR, "hybrid_full_report.txt")


# Note: cookies are handled internally by frame_sampler.py
#   COOKIES_PATH = backend/cookies.txt  (no config needed here)


LABELS            = ["Educational", "Neutral", "Overstimulating"]
RANDOM_STATE      = 42
SAMPLES_PER_CLASS = 70   # 70 × 3 = 210 — all of test_clean.csv

random.seed(RANDOM_STATE)

# ── Fusion config v3 (confidence-gated alpha + H-override) ────────────────────
# MUST stay identical to classify.py. If you change classify.py, update here
# then delete progress/results and re-run from scratch.
BASE_ALPHA      = 0.40   # NB weight when nb_confidence >= CONF_THRESH
LOW_ALPHA       = 0.15   # NB weight when nb_confidence <  CONF_THRESH
CONF_THRESH     = 0.40   # confidence boundary for switching alpha
H_OVERRIDE      = 0.07   # Score_H < this → cannot be Overstimulating
THRESHOLD_BLOCK = 0.20   # Score_final >= 0.20 → Overstimulating
THRESHOLD_ALLOW = 0.18   # Score_final <= 0.18 → Educational


def _fuse(score_nb: float, score_h: float, nb_confidence: float) -> tuple:
    """
    Confidence-gated hybrid fusion — identical to classify.py::_fuse().
    Returns (score_final, pred_label, eff_alpha).
    """
    eff_alpha   = LOW_ALPHA if nb_confidence < CONF_THRESH else BASE_ALPHA
    score_final = round((eff_alpha * score_nb) + ((1 - eff_alpha) * score_h), 4)

    if H_OVERRIDE > 0 and score_h < H_OVERRIDE:
        pred = "Educational" if score_final <= THRESHOLD_ALLOW else "Neutral"
    elif score_final >= THRESHOLD_BLOCK:
        pred = "Overstimulating"
    elif score_final <= THRESHOLD_ALLOW:
        pred = "Educational"
    else:
        pred = "Neutral"

    return score_final, pred, eff_alpha


def load_labeled_videos():
    """Load test_clean.csv — the 210-video held-out split."""
    if not os.path.exists(DATA_PATH):
        print(f"[HYBRID_EVAL] ✗ test_clean.csv not found at {DATA_PATH}")
        print(f"[HYBRID_EVAL]   Run: python build_700.py → python preprocess.py")
        sys.exit(1)

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

    print(f"[HYBRID_EVAL] Loaded {len(rows)} videos from test_clean.csv")
    print(f"[HYBRID_EVAL]   (All unseen by NB during training — no contamination)")
    for lbl in LABELS:
        count = sum(1 for r in rows if r["label"] == lbl)
        print(f"[HYBRID_EVAL]   {lbl}: {count}")
    return rows


def stratified_sample(rows):
    """Select SAMPLES_PER_CLASS videos per class, reproducibly."""
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
    """Resume from previous run — skip already-completed videos."""
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        completed_ids = {r["video_id"] for r in data}
        print(f"[HYBRID_EVAL] Resuming — {len(data)} already done.")
        return data, completed_ids
    return [], set()


def save_progress(results):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def run_hybrid_on_video(video: dict) -> dict:
    """
    Run the full v3 hybrid pipeline on one video.

    Step 1 — Download + frame-sample (with optional cookies for region bypass)
    Step 2 — Heuristic score
    Step 3 — NB score, enriched with scrape_ytInitialData_keywords() so that
              the NB input at eval time matches the enriched training data from
              enrich_dataset.py.  This is the critical fix vs _original.py.
    Step 4 — Confidence-gated fusion (identical to classify.py)
    """
    try:
        from app.modules.frame_sampler import sample_video
        from app.modules.heuristic     import compute_heuristic_score
        from app.modules.naive_bayes   import score_from_metadata_dict
        from app.modules.youtube_api   import scrape_ytInitialData_keywords, _merge_tags

        video_id      = video["video_id"]
        video_url     = f"https://www.youtube.com/watch?v={video_id}"
        thumbnail_url = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"
        t_start       = time.time()

        # ── Step 1: Download + frame-sample ───────────────────────────────────
        # Cookies are handled internally by frame_sampler.py via COOKIES_PATH
        # (backend/cookies.txt). sample_video() does NOT accept cookies_file.
        sample        = sample_video(video_url,
                                     thumbnail_url=thumbnail_url,
                                     hint_title=video.get("title", ""))
        sample_status = sample.get("status", "error")

        if sample_status in ("unavailable", "error"):
            reason = sample.get("reason", sample.get("message", "unavailable"))
            print(f"  ✗ SKIP [{video['label']:>15}] {video_id} — {reason[:60]}")
            return {
                "video_id":        video_id,
                "title":           video.get("title", "")[:60],
                "true_label":      video["label"],
                "pred_label":      "SKIPPED",
                "status":          sample_status,
                "reason":          reason,
            }

        # ── Step 2: Heuristic score ────────────────────────────────────────────
        h_result = compute_heuristic_score(sample)
        score_h  = h_result["score_h"]

        # ── Step 3: NB score — enrich tags with ytInitialData scraping ────────
        # Must match classify.py exactly: yt-dlp tags + scraped hidden keywords,
        # same as what the enriched training data contained via enrich_dataset.py.
        # Without this, evaluation NB input differs from training → lower results.
        sample_tags = sample.get("tags", [])
        try:
            scraped_tags = scrape_ytInitialData_keywords(video_id)
            nb_tags      = _merge_tags(sample_tags, scraped_tags)
            if scraped_tags:
                print(f"  [NB] Tags: {len(sample_tags)} sample "
                      f"+ {len(scraped_tags)} scraped = {len(nb_tags)} total")
        except Exception as e:
            print(f"  [NB] ✗ tag enrichment failed: {e}")
            nb_tags = sample_tags

        nb_result     = score_from_metadata_dict({
            "title":       sample.get("video_title", "") or video.get("title", ""),
            "tags":        nb_tags,
            "description": sample.get("description", ""),
        })
        score_nb      = nb_result.score_nb
        nb_confidence = nb_result.confidence

        # ── Step 4: Confidence-gated fusion — identical to classify.py ─────────
        score_final, pred_label, eff_alpha = _fuse(score_nb, score_h, nb_confidence)

        runtime = round(time.time() - t_start, 2)
        correct = "✓" if pred_label == video["label"] else "✗"
        path    = "FULL" if sample_status == "success" else "THUMB"

        print(f"  {correct} [{video['label']:>15} → {pred_label:>15}] "
              f"NB={score_nb:.3f}(conf={nb_confidence:.2f}) "
              f"H={score_h:.3f} α={eff_alpha} "
              f"Final={score_final:.3f} [{path}] "
              f"({runtime}s) {video['title'][:35]!r}")

        segments = sample.get("segments", [])
        seg_data = [
            {
                "segment_id": s.get("segment_id"),
                "fcr":        s.get("fcr"),
                "csv":        s.get("csv"),
                "att":        s.get("att"),
                "score_h":    s.get("score_h"),
            }
            for s in segments if s
        ]

        return {
            "video_id":              video_id,
            "title":                 video.get("title", "")[:80],
            "true_label":            video["label"],
            "pred_label":            pred_label,
            "score_nb":              round(score_nb,        4),
            "score_h":               round(score_h,         4),
            "score_final":           score_final,
            "nb_confidence":         round(nb_confidence,   4),
            "effective_alpha":       eff_alpha,
            "h_overridden":          score_h < H_OVERRIDE,
            "sample_path":           sample_status,
            "segments":              seg_data,
            "thumbnail_intensity":   sample.get("thumbnail_intensity", 0.0),
            "status":                "success",
            "runtime":               runtime,
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

    valid   = [r for r in results if r["pred_label"] not in ("SKIPPED", "ERROR")]
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
                                   target_names=LABELS, digits=4, zero_division=0)
    cm     = confusion_matrix(y_true, y_pred, labels=LABELS)
    acc    = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    prec_per, rec_per, f1_per, sup_per = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )

    print(f"\nFusion v3: confidence-gated alpha ({BASE_ALPHA}/{LOW_ALPHA}), "
          f"conf_thresh={CONF_THRESH}, H_override={H_OVERRIDE}")
    print(f"Thresholds: Block >= {THRESHOLD_BLOCK} | Allow <= {THRESHOLD_ALLOW}\n")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(f"{'':>20}" + "".join(f"{l[:5]:>12}" for l in LABELS))
    for i, lbl in enumerate(LABELS):
        print(f"{lbl:>20}" + "".join(f"{cm[i][j]:>12}" for j in range(3)))

    print(f"\nOverall Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Weighted Precision: {prec:.4f}")
    print(f"Weighted Recall   : {rec:.4f}")
    print(f"Weighted F1-Score : {f1:.4f}")

    full_count  = sum(1 for r in valid if r.get("sample_path") == "success")
    thumb_count = sum(1 for r in valid if r.get("sample_path") == "thumbnail_only")
    h_count     = sum(1 for r in valid if r.get("h_overridden", False))
    low_conf    = sum(1 for r in valid if r.get("effective_alpha") == LOW_ALPHA)
    print(f"\nEvaluation Path Breakdown:")
    print(f"  Full video analysis  : {full_count}")
    print(f"  Thumbnail-only       : {thumb_count}")
    print(f"  H_OVERRIDE triggered : {h_count}  (Score_H < {H_OVERRIDE} → capped at Neutral)")
    print(f"  LOW_ALPHA used       : {low_conf}  (NB confidence < {CONF_THRESH})")

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
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    json_data = {
        "generated": datetime.datetime.now().isoformat(),
        "source":    "test_clean.csv — NB held-out split (no training contamination)",
        "config": {
            "fusion_version":     "v3-confidence-gated",
            "base_alpha_nb":      BASE_ALPHA,
            "low_alpha_nb":       LOW_ALPHA,
            "conf_thresh":        CONF_THRESH,
            "h_override":         H_OVERRIDE,
            "threshold_block":    THRESHOLD_BLOCK,
            "threshold_allow":    THRESHOLD_ALLOW,
            "samples_per_class":  SAMPLES_PER_CLASS,
            "cookies_used":       True,  # handled by frame_sampler.py via COOKIES_PATH
        },
        "metrics": {k: v for k, v in metrics.items()
                    if k not in ("report", "y_true", "y_pred")},
        "results": results,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("ChildFocus — Full Hybrid Evaluation Report  (v4 — test_clean, enriched NB)\n")
        f.write(f"Generated   : {datetime.datetime.now()}\n")
        f.write(f"Source      : test_clean.csv (NB held-out split, no contamination)\n")
        f.write(f"Fusion      : v3 confidence-gated (matches classify.py)\n")
        f.write(f"BASE_ALPHA  : {BASE_ALPHA} (when NB conf >= {CONF_THRESH})\n")
        f.write(f"LOW_ALPHA   : {LOW_ALPHA} (when NB conf <  {CONF_THRESH})\n")
        f.write(f"H_OVERRIDE  : {H_OVERRIDE}\n")
        f.write(f"Thresholds  : Block >= {THRESHOLD_BLOCK} | Allow <= {THRESHOLD_ALLOW}\n")
        f.write("Cookies     : handled by frame_sampler.py via backend/cookies.txt\n")
        f.write("="*70 + "\n\n")

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
        f.write("-"*105 + "\n")
        f.write(f"{'video_id':>12} {'true_label':>16} {'pred':>16} "
                f"{'NB':>7} {'conf':>6} {'effA':>5} {'H':>7} {'Final':>7} "
                f"{'H_ov':>5} {'path':>12}  title\n")
        f.write("-"*105 + "\n")
        for r in results:
            mark = "OK" if r.get("pred_label") == r.get("true_label") else "XX"
            f.write(
                f"{r['video_id']:>12} {r.get('true_label','?'):>16} "
                f"{r.get('pred_label','?'):>16} "
                f"{r.get('score_nb', 0):>7.4f} "
                f"{r.get('nb_confidence', 0):>6.3f} "
                f"{r.get('effective_alpha', 0):>5.2f} "
                f"{r.get('score_h', 0):>7.4f} "
                f"{r.get('score_final', 0):>7.4f} "
                f"{'YES' if r.get('h_overridden') else 'no':>5} "
                f"{r.get('sample_path','?'):>12} "
                f"[{mark}] {r.get('title','')[:38]}\n"
            )

    print(f"\n[HYBRID_EVAL] Results JSON  → {RESULTS_PATH}")
    print(f"[HYBRID_EVAL] Report TXT    → {REPORT_PATH}")
    print(f"[HYBRID_EVAL] Progress file → {PROGRESS_PATH}")


def main():
    print("\n" + "="*65)
    print("CHILDFOCUS — FULL HYBRID EVALUATION  (v4, 210-video test_clean.csv)")
    print("="*65)
    print(f"Source      : test_clean.csv — NB has NEVER seen these videos")
    print(f"Fusion      : v3 confidence-gated (matches classify.py)")
    print(f"BASE_ALPHA  : {BASE_ALPHA} (NB conf >= {CONF_THRESH}) | "
          f"LOW_ALPHA: {LOW_ALPHA} (NB conf < {CONF_THRESH})")
    print(f"H_OVERRIDE  : Score_H < {H_OVERRIDE} → cannot be Overstimulating")
    print(f"Thresholds  : Block >= {THRESHOLD_BLOCK} | Allow <= {THRESHOLD_ALLOW}")
    print(f"Tag enrich  : ytInitialData scraping ENABLED (matches enrich_dataset.py)")
    _cp = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "backend", "cookies.txt"))
    if os.path.isfile(_cp):
        print(f"Cookies     : ✓ {_cp}")
    else:
        print(f"Cookies     : ✗ NOT FOUND at {_cp} — region-locked videos will fallback to thumbnail")
    print(f"Sample      : {SAMPLES_PER_CLASS} videos per class = {SAMPLES_PER_CLASS * 3} total\n")

    rows   = load_labeled_videos()
    sample = stratified_sample(rows)

    completed_results, completed_ids = load_progress()
    remaining = [v for v in sample if v["video_id"] not in completed_ids]

    print(f"\n[HYBRID_EVAL] Videos to process: {len(remaining)} "
          f"({len(completed_ids)} already done)\n")

    results = list(completed_results)
    t_total = time.time()

    for i, video in enumerate(remaining, 1):
        idx = len(completed_ids) + i
        print(f"\n[{idx:>3}/{SAMPLES_PER_CLASS * 3}] "
              f"video_id={video['video_id']} | class={video['label']}")
        print(f"        title: {video['title'][:65]!r}")

        result = run_hybrid_on_video(video)
        results.append(result)
        completed_ids.add(video["video_id"])
        save_progress(results)

        elapsed         = time.time() - t_total
        remaining_count = len(remaining) - i
        avg_time        = elapsed / i
        eta             = avg_time * remaining_count
        print(f"        Elapsed: {elapsed/60:.1f}min | "
              f"ETA: ~{eta/60:.0f}min remaining")

    print(f"\n[HYBRID_EVAL] All {len(results)} videos processed.")
    metrics = compute_metrics(results)
    save_report(results, metrics)

    print("\n" + "="*65)
    print("FINAL SUMMARY — PASTE THESE INTO YOUR THESIS")
    print("="*65)
    if metrics:
        print(f"Source       : test_clean.csv (no NB training contamination)")
        print(f"Sample size  : {metrics['n_valid']} valid / {metrics['n_total']} attempted")
        print(f"Accuracy     : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Precision    : {metrics['precision']:.4f}")
        print(f"Recall       : {metrics['recall']:.4f}")
        print(f"F1-Score     : {metrics['f1']:.4f}")
        print(f"\nPer-class:")
        for lbl, m in metrics["per_class"].items():
            print(f"  {lbl:<18}: P={m['precision']:.4f}  "
                  f"R={m['recall']:.4f}  F1={m['f1']:.4f}  n={m['support']}")
        cm = metrics["confusion_matrix"]
        print(f"\nConfusion Matrix:")
        print(f"{'':>20}" + "".join(f"{l[:5]:>12}" for l in LABELS))
        for i, lbl in enumerate(LABELS):
            print(f"{lbl:>20}" + "".join(f"{cm[i][j]:>12}" for j in range(3)))
    print("="*65)
    print("\n[HYBRID_EVAL] Complete. Copy results above for Chapter 5.")


if __name__ == "__main__":
    main()
