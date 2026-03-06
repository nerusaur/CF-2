"""
ChildFocus - Heuristic Analysis Module
backend/app/modules/heuristic.py

What this does:
  - Wraps the frame_sampler.py logic into a clean interface
  - Exposes compute_heuristic_score() → returns Score_H between 0.0 and 1.0
  - Score_H is computed from FCR, CSV, ATT, and Thumbnail Intensity
  - Used by hybrid_fusion.py to compute the final OIR score

Thesis reference (heuristic weights):
  Score_H = (0.35 × FCR) + (0.25 × CSV) + (0.20 × ATT) + (0.20 × Thumb)
  Aggregation: Score_agg = max({Score_S1, Score_S2, Score_S3})
  Then final heuristic: 0.80 × Score_agg + 0.20 × Thumbnail
"""

from app.modules.frame_sampler import (
    sample_video,
    compute_fcr,
    compute_csv,
    compute_att,
    compute_thumbnail_intensity,
    extract_frames,
    fetch_video,
)

# ── Heuristic weights (from thesis) ───────────────────────────────────────────
W_FCR   = 0.35
W_CSV   = 0.25
W_ATT   = 0.20
W_THUMB = 0.20

# ── Thresholds (from thesis) ──────────────────────────────────────────────────
THRESHOLD_HIGH = 0.75   # Overstimulating
THRESHOLD_LOW  = 0.35   # Safe / Educational


def compute_heuristic_score(video_id: str, thumbnail_url: str = "") -> dict:
    """
    Full heuristic analysis of a YouTube video.

    Calls sample_video() from frame_sampler.py which:
      1. Downloads the video (up to 90s)
      2. Extracts 3 segments (beginning, middle, end)
      3. Computes FCR, CSV, ATT per segment
      4. Computes thumbnail intensity
      5. Returns Score_H per segment and aggregate score

    Args:
        video_id:      YouTube video ID (e.g. "dQw4w9WgXcQ")
        thumbnail_url: Optional thumbnail URL (improves Thumb score)

    Returns:
        dict with:
            score_h (float):   Aggregate heuristic score [0.0, 1.0]
            segments (list):   Per-segment FCR, CSV, ATT, score_h
            thumbnail (float): Thumbnail intensity score
            label (str):       Overstimulating / Uncertain / Safe
            status (str):      success / unavailable / error
            runtime_seconds (float)
    """
    result = sample_video(video_id, thumbnail_url)

    if result.get("status") != "success":
        return {
            "score_h":          0.5,
            "segments":         [],
            "thumbnail":        0.0,
            "label":            "Uncertain",
            "status":           result.get("status", "error"),
            "message":          result.get("message", result.get("reason", "Unknown error")),
            "runtime_seconds":  0.0,
        }

    score_h = result.get("aggregate_heuristic_score", 0.5)
    label   = _label_from_score(score_h)

    return {
        "score_h":         score_h,
        "segments":        result.get("segments", []),
        "thumbnail":       result.get("thumbnail_intensity", 0.0),
        "label":           label,
        "video_title":     result.get("video_title", ""),
        "video_duration":  result.get("video_duration_sec", 0),
        "status":          "success",
        "runtime_seconds": result.get("runtime_seconds", 0.0),
    }


def compute_segment_score(fcr: float, csv: float, att: float) -> float:
    """
    Compute heuristic score for a single segment.
    Thesis formula: Score_H = (w1×FCR) + (w2×CSV) + (w3×ATT)
    Note: Thumbnail is factored in separately at the video level.

    Args:
        fcr: Frame-Change Rate [0, 1]
        csv: Color Saturation Variance [0, 1]
        att: Audio Tempo Transitions [0, 1]

    Returns:
        float: Score_H for this segment [0, 1]
    """
    return round(
        (W_FCR * fcr) + (W_CSV * csv) + (W_ATT * att),
        4
    )


def _label_from_score(score: float) -> str:
    """Map a numeric score to an OIR label using thesis thresholds."""
    if score >= THRESHOLD_HIGH:
        return "Overstimulating"
    elif score <= THRESHOLD_LOW:
        return "Safe"
    else:
        return "Uncertain"


def get_feature_weights() -> dict:
    """Return the heuristic feature weights for transparency/logging."""
    return {
        "w_fcr":            W_FCR,
        "w_csv":            W_CSV,
        "w_att":            W_ATT,
        "w_thumb":          W_THUMB,
        "threshold_high":   THRESHOLD_HIGH,
        "threshold_low":    THRESHOLD_LOW,
    }