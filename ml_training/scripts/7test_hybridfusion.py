"""
ChildFocus - Hybrid Evaluation Script  (v5 — 210-video, config-aligned)
ml_training/scripts/evaluate_hybrid_real.py

WHAT CHANGED FROM v4:
  - Fusion config constants now match hybrid_fusion.py exactly:
      BASE_ALPHA      = 0.60   (was 0.40)
      LOW_ALPHA       = 0.15   (unchanged)
      CONF_THRESH     = 0.40   (unchanged)
      H_OVERRIDE      = 0.07   (unchanged)
      THRESHOLD_BLOCK = 0.30   (was 0.20)
      THRESHOLD_ALLOW = 0.12   (was 0.18)
  - Added FUSION_VERSION string for report traceability.
  - Added per-class binary (Safe vs Overstimulating) metrics section.
  - Output files renamed to hybrid_210_* to avoid overwriting old 30-video results.

REQUIREMENTS (run on your local machine, not in Claude):
  1. pip install yt-dlp opencv-python librosa scikit-learn pillow requests
  2. backend/cookies.txt must exist (export from browser via "Get cookies.txt LOCALLY")
  3. Run from ml_training/scripts/ :  python evaluate_hybrid_real.py

Expected runtime: ~2-4 hours for 210 videos (30-60s per video).
Progress is saved after every video — safe to interrupt and resume.

DELETE these files before re-running after a retrain:
    rm ../outputs/hybrid_210_progress.json
    rm ../outputs/hybrid_210_results.json
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
PROGRESS_PATH = os.path.join(OUTPUTS_DIR, "hybrid_210_progress.json")
RESULTS_PATH  = os.path.join(OUTPUTS_DIR, "hybrid_210_results.json")
REPORT_PATH   = os.path.join(OUTPUTS_DIR, "hybrid_210_report.txt")

LABELS             = ["Educational", "Neutral", "Overstimulating"]
RANDOM_STATE       = 42
SAMPLES_PER_CLASS  = 70   # 70 × 3 = 210 total

random.seed(RANDOM_STATE)

# ── Fusion config v5 — MUST MATCH backend/app/modules/hybrid_fusion.py ────────
# If you change hybrid_fusion.py, update these constants and delete progress files.
FUSION_VERSION  = "v5-confidence-gated-aligned"
BASE_ALPHA      = 0.60   # NB weight when nb_confidence >= CONF_THRESH
LOW_ALPHA       = 0.15   # NB weight when nb_confidence <  CONF_THRESH
CONF_THRESH     = 0.40   # confidence boundary for switching alpha
H_OVERRIDE      = 0.07   # Score_H < this → cannot be Overstimulating
THRESHOLD_BLOCK = 0.30   # Score_final >= 0.30 → Overstimulating
THRESHOLD_ALLOW = 0.12   # Score_final <= 0.12 → Educational
# Neutral range: 0.12 < Score_final < 0.30


def _fuse(score_nb: float, score_h: float, nb_confidence: float) -> tuple:
    """
    Confidence-gated hybrid fusion — identical to hybrid_fusion.py classify_full().
    Returns (score_final, pred_label, eff_alpha).
    """
    eff_alpha   = LOW_ALPHA if nb_confidence < CONF_THRESH else BASE_ALPHA
    score_final = round((eff_alpha * score_nb) + ((1 - eff_alpha) * score_h), 4)

    if H_OVERRIDE > 0 and score_h < H_OVERRIDE:
        # Very low heuristic score → cannot be Overstimulating regardless of NB
        pred = "Educational" if score_final <= THRESHOLD_ALLOW else "Neutral"
    elif score_final >= THRESHOLD_BLOCK:
        pred = "Overstimulating"
    elif score_final <= THRESHOLD_ALLOW:
        pred = "Educational"
    else:
        pred = "Neutral"

    return score_final, pred, eff_alpha


def load_labeled_videos():
    """Load test_clean.csv — the 210-video held-out NB split."""
    if not os.path.exists(DATA_PATH):
        print(f"[HYBRID] ✗ test_clean.csv not found at {DATA_PATH}")
        print(f"[HYBRID]   Run: python build_700.py → python preprocess.py")
        sys.exit(1)

    rows = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label    = row.get("label", "").strip()
            video_id = row.get("video_id", "").strip()
            title    = row.get("title", "").strip()
            text     = row.get("text", "").strip()
            if label in LABELS and video_id and len(video_id) == 11:
                rows.append({
                    "video_id": video_id,
                    "label":    label,
                    "title":    title,
                    "text":     text,
                })

    print(f"[HYBRID] Loaded {len(rows)} videos from test_clean.csv")
    for lbl in LABELS:
        count = sum(1 for r in rows if r["label"] == lbl)
        print(f"[HYBRID]   {lbl}: {count}")
    return rows


def stratified_sample(rows):
    """Select SAMPLES_PER_CLASS per class reproducibly."""
    by_class = {lbl: [] for lbl in LABELS}
    for r in rows:
        by_class[r["label"]].append(r)

    sample = []
    for lbl in LABELS:
        pool = by_class[lbl][:]
        random.shuffle(pool)
        selected = pool[:SAMPLES_PER_CLASS]
        sample.extend(selected)
        print(f"[HYBRID]   Selected {len(selected)} {lbl} videos")

    random.shuffle(sample)
    return sample


def load_progress():
    """Resume from previous run — skip already-completed videos."""
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        completed_ids = {r["video_id"] for r in data}
        print(f"[HYBRID] Resuming — {len(data)} already done.")
        return data, completed_ids
    return [], set()


def save_progress(results):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def run_hybrid_on_video(video: dict) -> dict:
    """
    Full hybrid pipeline on one video:
      1. Download + frame-sample (yt-dlp + OpenCV)
      2. Heuristic score (FCR, CSV, ATT)
      3. NB score with tag enrichment (matches training input)
      4. Confidence-gated fusion (matches hybrid_fusion.py)
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

        # Step 1: Download + frame-sample
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

        # Step 2: Heuristic score
        h_result = compute_heuristic_score(sample)
        score_h  = h_result["score_h"]

        # Step 3: NB score with enriched tags (must match training input)
        sample_tags = sample.get("tags", [])
        try:
            scraped_tags = scrape_ytInitialData_keywords(video_id)
            nb_tags      = _merge_tags(sample_tags, scraped_tags)
            if scraped_tags:
                print(f"  [NB] Tags: {len(sample_tags)} + {len(scraped_tags)} scraped = {len(nb_tags)}")
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

        # Step 4: Confidence-gated fusion
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
            "video_id":            video_id,
            "title":               video.get("title", "")[:80],
            "true_label":          video["label"],
            "pred_label":          pred_label,
            "score_nb":            round(score_nb,       4),
            "score_h":             round(score_h,        4),
            "score_final":         score_final,
            "nb_confidence":       round(nb_confidence,  4),
            "effective_alpha":     eff_alpha,
            "h_overridden":        score_h < H_OVERRIDE,
            "sample_path":         sample_status,
            "segments":            seg_data,
            "thumbnail_intensity": sample.get("thumbnail_intensity", 0.0),
            "status":              "success",
            "runtime":             runtime,
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

    print(f"\n{'='*65}")
    print("HYBRID EVALUATION RESULTS  (v5 — 210-video test_clean.csv)")
    print(f"{'='*65}")
    print(f"Total attempted  : {len(results)}")
    print(f"Valid results    : {len(valid)}")
    print(f"Skipped / Error  : {skipped}")

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
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_per, rec_per, f1_per, sup_per = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )

    # ── Binary (Safe vs Overstimulating) metrics ──────────────────────────────
    y_true_bin = ["Overstimulating" if l == "Overstimulating" else "Safe" for l in y_true]
    y_pred_bin = ["Overstimulating" if l == "Overstimulating" else "Safe" for l in y_pred]
    bin_acc    = accuracy_score(y_true_bin, y_pred_bin)
    bin_labels = ["Safe", "Overstimulating"]
    bin_report = classification_report(y_true_bin, y_pred_bin,
                                       target_names=bin_labels, digits=4, zero_division=0)
    bin_cm     = confusion_matrix(y_true_bin, y_pred_bin, labels=bin_labels)

    print(f"\nFusion {FUSION_VERSION}")
    print(f"BASE_ALPHA={BASE_ALPHA} (conf>={CONF_THRESH}) | LOW_ALPHA={LOW_ALPHA} | "
          f"H_OVERRIDE={H_OVERRIDE}")
    print(f"Block >= {THRESHOLD_BLOCK} | Allow <= {THRESHOLD_ALLOW}\n")

    print("3-CLASS CLASSIFICATION REPORT:")
    print(report)
    print("3-Class Confusion Matrix:")
    print(f"{'':>20}" + "".join(f"{l[:5]:>14}" for l in LABELS))
    for i, lbl in enumerate(LABELS):
        print(f"{lbl:>20}" + "".join(f"{cm[i][j]:>14}" for j in range(3)))

    print(f"\n── 3-Class Summary ──")
    print(f"Overall Accuracy   : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"Macro Precision    : {macro_prec:.4f}")
    print(f"Macro Recall       : {macro_rec:.4f}")
    print(f"Macro F1           : {macro_f1:.4f}")

    print(f"\n── Binary Summary (Safe vs Overstimulating) ──")
    print(bin_report)
    print("Binary Confusion Matrix:")
    print(f"{'':>20}" + "".join(f"{l:>18}" for l in bin_labels))
    for i, lbl in enumerate(bin_labels):
        print(f"{lbl:>20}" + "".join(f"{bin_cm[i][j]:>18}" for j in range(2)))
    print(f"Binary Accuracy    : {bin_acc:.4f}  ({bin_acc*100:.2f}%)")

    # ── Overstimulating recall (child safety metric) ──────────────────────────
    over_idx   = LABELS.index("Overstimulating")
    over_rec   = rec_per[over_idx]
    over_sup   = sup_per[over_idx]
    over_miss  = int(over_sup - (over_rec * over_sup))
    print(f"\n── Child Safety Metric ──")
    print(f"Overstimulating Recall : {over_rec:.4f} ({over_rec*100:.2f}%)")
    print(f"Missed detections      : {over_miss} out of {int(over_sup)}")

    # ── Evaluation path breakdown ─────────────────────────────────────────────
    full_count  = sum(1 for r in valid if r.get("sample_path") == "success")
    thumb_count = sum(1 for r in valid if r.get("sample_path") == "thumbnail_only")
    h_count     = sum(1 for r in valid if r.get("h_overridden", False))
    low_conf    = sum(1 for r in valid if r.get("effective_alpha") == LOW_ALPHA)
    print(f"\n── Path Breakdown ──")
    print(f"Full video analysis  : {full_count}")
    print(f"Thumbnail-only       : {thumb_count}")
    print(f"H_OVERRIDE triggered : {h_count}  (Score_H < {H_OVERRIDE})")
    print(f"LOW_ALPHA used       : {low_conf}  (NB conf < {CONF_THRESH})")

    return {
        "n_total":        len(results),
        "n_valid":        len(valid),
        "n_skipped":      skipped,
        "n_full":         full_count,
        "n_thumb":        thumb_count,
        "accuracy":       round(acc,        4),
        "macro_precision":round(macro_prec, 4),
        "macro_recall":   round(macro_rec,  4),
        "macro_f1":       round(macro_f1,   4),
        "weighted_precision": round(prec, 4),
        "weighted_recall":    round(rec,  4),
        "weighted_f1":        round(f1,   4),
        "binary_accuracy":    round(bin_acc, 4),
        "overstimulating_recall": round(over_rec, 4),
        "per_class": {
            lbl: {
                "precision": round(float(prec_per[i]), 4),
                "recall":    round(float(rec_per[i]),  4),
                "f1":        round(float(f1_per[i]),   4),
                "support":   int(sup_per[i]),
            }
            for i, lbl in enumerate(LABELS)
        },
        "confusion_matrix":        cm.tolist(),
        "binary_confusion_matrix": bin_cm.tolist(),
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
            "fusion_version":  FUSION_VERSION,
            "base_alpha":      BASE_ALPHA,
            "low_alpha":       LOW_ALPHA,
            "conf_thresh":     CONF_THRESH,
            "h_override":      H_OVERRIDE,
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

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("ChildFocus — Hybrid Evaluation Report  (v5 — 210-video, config-aligned)\n")
        f.write(f"Generated   : {datetime.datetime.now()}\n")
        f.write(f"Source      : test_clean.csv (NB held-out split, no contamination)\n")
        f.write(f"Fusion      : {FUSION_VERSION}\n")
        f.write(f"BASE_ALPHA  : {BASE_ALPHA} (when NB conf >= {CONF_THRESH})\n")
        f.write(f"LOW_ALPHA   : {LOW_ALPHA}  (when NB conf <  {CONF_THRESH})\n")
        f.write(f"H_OVERRIDE  : {H_OVERRIDE}\n")
        f.write(f"Thresholds  : Block >= {THRESHOLD_BLOCK} | Allow <= {THRESHOLD_ALLOW}\n")
        f.write("=" * 70 + "\n\n")

        if "report" in metrics:
            f.write("3-CLASS CLASSIFICATION REPORT\n")
            f.write(metrics["report"])
            f.write(f"\nOverall Accuracy   : {metrics['accuracy']:.4f}\n")
            f.write(f"Macro Precision    : {metrics['macro_precision']:.4f}\n")
            f.write(f"Macro Recall       : {metrics['macro_recall']:.4f}\n")
            f.write(f"Macro F1           : {metrics['macro_f1']:.4f}\n")
            f.write(f"Binary Accuracy    : {metrics['binary_accuracy']:.4f}\n")
            f.write(f"Overst. Recall     : {metrics['overstimulating_recall']:.4f}\n\n")
            f.write("3-CLASS CONFUSION MATRIX\n")
            cm = metrics["confusion_matrix"]
            f.write(f"{'':>20}" + "".join(f"{l[:5]:>14}" for l in LABELS) + "\n")
            for i, lbl in enumerate(LABELS):
                f.write(f"{lbl:>20}" + "".join(f"{cm[i][j]:>14}" for j in range(3)) + "\n")
            f.write("\nBINARY CONFUSION MATRIX (Safe vs Overstimulating)\n")
            bcm  = metrics["binary_confusion_matrix"]
            blbls = ["Safe", "Overstimulating"]
            f.write(f"{'':>20}" + "".join(f"{l:>18}" for l in blbls) + "\n")
            for i, lbl in enumerate(blbls):
                f.write(f"{lbl:>20}" + "".join(f"{bcm[i][j]:>18}" for j in range(2)) + "\n")
            f.write("\n")

        f.write("PER-VIDEO RESULTS\n")
        f.write("-" * 110 + "\n")
        f.write(f"{'video_id':>12} {'true_label':>16} {'pred':>16} "
                f"{'NB':>7} {'conf':>6} {'effA':>5} {'H':>7} {'Final':>7} "
                f"{'H_ov':>5} {'path':>12}  title\n")
        f.write("-" * 110 + "\n")
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

    print(f"\n[HYBRID] Results JSON → {RESULTS_PATH}")
    print(f"[HYBRID] Report TXT  → {REPORT_PATH}")


def main():
    print("\n" + "=" * 65)
    print("CHILDFOCUS — HYBRID EVALUATION  (v5 — 210 videos, config-aligned)")
    print("=" * 65)
    print(f"Source       : test_clean.csv — NB NEVER saw these 210 videos")
    print(f"Fusion       : {FUSION_VERSION}")
    print(f"BASE_ALPHA   : {BASE_ALPHA} (NB conf >= {CONF_THRESH})")
    print(f"LOW_ALPHA    : {LOW_ALPHA} (NB conf <  {CONF_THRESH})")
    print(f"H_OVERRIDE   : Score_H < {H_OVERRIDE} → cannot be Overstimulating")
    print(f"Thresholds   : Block >= {THRESHOLD_BLOCK} | Allow <= {THRESHOLD_ALLOW}")
    _cp = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "backend", "cookies.txt"))
    print(f"Cookies      : {'✓ found' if os.path.isfile(_cp) else '✗ NOT FOUND — region-locked videos will fallback to thumbnail'}")
    print(f"Sample       : {SAMPLES_PER_CLASS} per class = {SAMPLES_PER_CLASS * 3} total\n")

    rows   = load_labeled_videos()
    sample = stratified_sample(rows)

    completed_results, completed_ids = load_progress()
    remaining = [v for v in sample if v["video_id"] not in completed_ids]
    print(f"\n[HYBRID] Videos to process: {len(remaining)} "
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
        print(f"        Elapsed: {elapsed/60:.1f}min | ETA: ~{eta/60:.0f}min remaining")

    print(f"\n[HYBRID] All {len(results)} videos processed.")
    metrics = compute_metrics(results)
    save_report(results, metrics)

    print("\n" + "=" * 65)
    print("THESIS NUMBERS — PASTE INTO CHAPTER 5 SECTION 5.3.3")
    print("=" * 65)
    if metrics:
        print(f"Source        : test_clean.csv ({metrics['n_valid']} valid / {metrics['n_total']} attempted)")
        print(f"Fusion        : {FUSION_VERSION}")
        print(f"3-class acc   : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Macro F1      : {metrics['macro_f1']:.4f}")
        print(f"Binary acc    : {metrics['binary_accuracy']:.4f} ({metrics['binary_accuracy']*100:.2f}%)")
        print(f"Overst. recall: {metrics['overstimulating_recall']:.4f}")
        print(f"\nPer-class breakdown:")
        for lbl, m in metrics["per_class"].items():
            print(f"  {lbl:<18}: P={m['precision']:.4f}  "
                  f"R={m['recall']:.4f}  F1={m['f1']:.4f}  n={m['support']}")
        cm = metrics["confusion_matrix"]
        print(f"\n3-Class Confusion Matrix:")
        print(f"{'':>20}" + "".join(f"{l[:5]:>14}" for l in LABELS))
        for i, lbl in enumerate(LABELS):
            print(f"{lbl:>20}" + "".join(f"{cm[i][j]:>14}" for j in range(3)))
        bcm   = metrics["binary_confusion_matrix"]
        blbls = ["Safe", "Overstimulating"]
        print(f"\nBinary Confusion Matrix:")
        print(f"{'':>20}" + "".join(f"{l:>18}" for l in blbls))
        for i, lbl in enumerate(blbls):
            print(f"{lbl:>20}" + "".join(f"{bcm[i][j]:>18}" for j in range(2)))
    print("=" * 65)
    print("\n[HYBRID] Complete.")


if __name__ == "__main__":
    main()
