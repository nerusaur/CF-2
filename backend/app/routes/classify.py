import os
import sqlite3
import time

from flask import Blueprint, jsonify, request

from app.modules.frame_sampler import sample_video
from app.modules.heuristic import compute_heuristic_score
from app.modules.naive_bayes import score_from_metadata_dict, score_metadata

classify_bp = Blueprint("classify", __name__)

DB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "database", "childfocus.db"
)


def extract_video_id(url: str) -> str:
    import re
    for pattern in [r"(?:v=)([a-zA-Z0-9_-]{11})",
                    r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",
                    r"(?:embed/)([a-zA-Z0-9_-]{11})"]:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    if re.fullmatch(r"[a-zA-Z0-9_-]{11}", url.strip()):
        return url.strip()
    return url.strip()


def _nb_only_result(video_id: str, metadata: dict, reason: str, t_start: float) -> dict:
    """
    Absolute last-resort: NB score only, no heuristic.
    score_h = None, score_final = score_nb.
    """
    nb_obj      = score_from_metadata_dict(metadata)
    score_nb    = nb_obj.score_nb
    score_final = round(score_nb, 4)

    if score_final >= 0.75:
        oir_label = "Overstimulating"; action = "block"
    elif score_final <= 0.35:
        oir_label = "Educational";     action = "allow"
    else:
        oir_label = "Neutral";         action = "allow"

    runtime = round(time.time() - t_start, 3)
    print(f"[ROUTE] NB-only ({reason[:60]}) → {video_id} {oir_label} ({score_final}) in {runtime}s")
    return {
        "video_id":        video_id,
        "video_title":     metadata.get("title", ""),
        "oir_label":       oir_label,
        "score_nb":        round(score_nb, 4),
        "score_h":         None,
        "score_final":     score_final,
        "cached":          False,
        "action":          action,
        "runtime_seconds": runtime,
        "status":          "success",
        "fallback_reason": reason[:120],
        "nb_details": {
            "predicted":  nb_obj.predicted_label,
            "confidence": round(nb_obj.confidence, 4),
        },
    }


def _fetch_metadata_only(video_url: str) -> dict:
    try:
        import yt_dlp
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True,
                                "skip_download": True}) as ydl:
            info = ydl.extract_info(video_url, download=False)
        return {
            "title":       info.get("title", ""),
            "tags":        info.get("tags", []) or [],
            "description": info.get("description", "") or "",
        }
    except Exception as e:
        print(f"[META] ✗ {e}")
        return {"title": "", "tags": [], "description": ""}


def _save_to_db(result: dict):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur  = conn.cursor()
        cur.execute("""
            INSERT OR REPLACE INTO videos
            (video_id, label, final_score, last_checked, checked_by,
             video_title, nb_score, heuristic_score, runtime_seconds)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?)
        """, (
            result["video_id"], result.get("oir_label", ""),
            result.get("score_final", 0.0), "hybrid_full",
            result.get("video_title", ""), result.get("score_nb", 0.0),
            result.get("score_h") or 0.0, result.get("runtime_seconds", 0.0),
        ))
        for seg in result.get("heuristic_details", {}).get("segments", []):
            cur.execute("""
                INSERT INTO segments
                (video_id, segment_id, offset_seconds, length_seconds,
                 fcr, csv, att, score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (result["video_id"], seg.get("segment_id"),
                  seg.get("offset_seconds"), seg.get("length_seconds"),
                  seg.get("fcr"), seg.get("csv"), seg.get("att"), seg.get("score_h")))
        conn.commit()
        conn.close()
        print(f"[DB] ✓ Saved {result['video_id']}")
    except Exception as e:
        print(f"[DB] ✗ {e}")


def _check_cache(video_id: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur  = conn.cursor()
        cur.execute("SELECT label, final_score, last_checked FROM videos WHERE video_id = ?",
                    (video_id,))
        row = cur.fetchone()
        conn.close()
        return row
    except Exception as e:
        print(f"[CACHE] {e}")
        return None


# ── /classify_fast ────────────────────────────────────────────────────────────

@classify_bp.route("/classify_fast", methods=["POST"])
def classify_fast():
    data = request.get_json(silent=True) or {}
    title = data.get("title", "")
    if not title:
        return jsonify({"error": "title is required", "status": "error"}), 400
    try:
        result = score_metadata(title=title, tags=data.get("tags", []),
                                description=data.get("description", ""))
        return jsonify({
            "score_nb":   result["score_nb"],
            "oir_label":  result["label"],
            "label":      result["label"],
            "confidence": result.get("confidence", 0.0),
            "status":     "success",
        }), 200
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500


# ── /classify_full ────────────────────────────────────────────────────────────

@classify_bp.route("/classify_full", methods=["POST"])
def classify_full():
    """
    Full hybrid classification. sample_video() handles the fallback chain:

      Normal download
        → Cookie-authenticated retry   (if age-restricted + cookies.txt exists)
          → Thumbnail-only heuristic   (CSV + thumb intensity; FCR/ATT = 0)
            → status="unavailable"     (NB-only here as absolute last resort)

    This route handles all three outcome statuses:
      "success"        → full hybrid   (video frames + NB)
      "thumbnail_only" → partial hybrid (thumbnail heuristic + NB)
      "unavailable"    → NB-only

    `hint_title` from classify_by_title ensures NB always has a title to score.
    """
    data          = request.get_json(silent=True) or {}
    video_url     = data.get("video_url", "").strip()
    thumbnail_url = data.get("thumbnail_url", "")
    hint_title    = data.get("hint_title", "").strip()

    if not video_url:
        return jsonify({"error": "video_url is required", "status": "error"}), 400

    video_id = extract_video_id(video_url)

    cached = _check_cache(video_id)
    if cached:
        label, final_score, last_checked = cached
        print(f"[CACHE] ✓ Hit for {video_id} → {label}")
        return jsonify({
            "video_id":     video_id,
            "oir_label":    label,
            "score_final":  final_score,
            "last_checked": last_checked,
            "cached":       True,
            "action":       "block" if label == "Overstimulating" else "allow",
            "status":       "success",
        }), 200

    t_start = time.time()

    try:
        print(f"[ROUTE] /classify_full → {video_id}")
        sample = sample_video(video_url, thumbnail_url=thumbnail_url,
                              hint_title=hint_title)
        sample_status = sample.get("status", "error")

        # ── Absolute fallback: video AND thumbnail both failed ─────────────────
        if sample_status in ("unavailable", "error"):
            reason = sample.get("reason", sample.get("message", "unavailable"))
            print(f"[ROUTE] ✗ Fully unavailable — NB-only for {video_id}")
            metadata = _fetch_metadata_only(video_url)
            if not metadata["title"] and hint_title:
                metadata["title"] = hint_title
            result = _nb_only_result(video_id, metadata, reason, t_start)
            _save_to_db(result)
            return jsonify(result), 200

        # ── Heuristic available (full video OR thumbnail-only) ─────────────────
        # compute_heuristic_score() receives identical dict structure in both cases
        h_result  = compute_heuristic_score(sample)
        score_h   = h_result["score_h"]
        h_details = h_result.get("details", {})

        # NB: prefer actual video title, fall back to hint_title
        nb_obj = score_from_metadata_dict({
            "title":       sample.get("video_title", "") or hint_title,
            "tags":        sample.get("tags", []),
            "description": sample.get("description", ""),
        })
        score_nb        = nb_obj.score_nb
        predicted_label = nb_obj.predicted_label
        score_final     = round((0.4 * score_nb) + (0.6 * score_h), 4)

        path_label = "full" if sample_status == "success" else "thumbnail-only"
        print(f"[ROUTE] [{path_label}] nb={score_nb:.4f} h={score_h:.4f} "
              f"final={score_final:.4f}")

        if score_final >= 0.75:
            oir_label = "Overstimulating"; action = "block"
        elif score_final <= 0.35:
            oir_label = "Educational";     action = "allow"
        else:
            oir_label = "Neutral";         action = "allow"

        runtime = round(time.time() - t_start, 3)
        result = {
            "video_id":          video_id,
            "video_title":       sample.get("video_title", "") or hint_title,
            "oir_label":         oir_label,
            "score_nb":          round(score_nb, 4),
            "score_h":           round(score_h, 4),
            "score_final":       score_final,
            "cached":            False,
            "action":            action,
            "runtime_seconds":   runtime,
            "status":            "success",
            "sample_path":       sample_status,   # "success" | "thumbnail_only"
            "heuristic_details": h_details,
            "nb_details": {
                "predicted":  predicted_label,
                "confidence": round(nb_obj.confidence, 4),
            },
        }
        _save_to_db(result)
        print(f"[ROUTE] /classify_full {video_id} → {oir_label} "
              f"({score_final}) [{path_label}] in {runtime}s")
        return jsonify(result), 200

    except Exception as e:
        print(f"[ROUTE] /classify_full error for {video_id}: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e), "status": "error"}), 500


# ── /classify_by_title ────────────────────────────────────────────────────────

@classify_bp.route("/classify_by_title", methods=["POST"])
def classify_by_title():
    data  = request.get_json(silent=True) or {}
    title = data.get("title", "").strip()

    if not title:
        return jsonify({"error": "title is required", "status": "error"}), 400
    if len(title.split()) < 2:
        print(f"[TITLE_ROUTE] Rejected: {title!r}")
        return jsonify({"error": "Title too short", "status": "error"}), 400

    print(f"[TITLE_ROUTE] Searching for: {title!r}")

    try:
        import yt_dlp
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True,
                                "extract_flat": True}) as ydl:
            info = ydl.extract_info(f"ytsearch1:{title}", download=False)

        entries = info.get("entries", [])
        if not entries:
            return jsonify({"error": "No video found", "status": "error"}), 404

        video_id  = entries[0].get("id", "")
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        thumb_url = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"
        print(f"[TITLE_ROUTE] Resolved: {title!r} → {video_id}")

        cached = _check_cache(video_id)
        if cached:
            label, final_score, last_checked = cached
            return jsonify({
                "video_id":     video_id,
                "oir_label":    label,
                "score_final":  final_score,
                "last_checked": last_checked,
                "cached":       True,
                "action":       "block" if label == "Overstimulating" else "allow",
                "status":       "success",
            }), 200

        from flask import current_app
        with current_app.test_request_context(
            "/classify_full", method="POST",
            json={"video_url": video_url, "thumbnail_url": thumb_url,
                  "hint_title": title},
        ):
            return classify_full()

    except Exception as e:
        print(f"[TITLE_ROUTE] Error: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e), "status": "error"}), 500


# ── /health ───────────────────────────────────────────────────────────────────

@classify_bp.route("/health", methods=["GET"])
def health():
    from app.modules.naive_bayes import model_status
    from app.modules.frame_sampler import COOKIES_PATH, _has_cookies
    return jsonify({
        "status":       "ok",
        "nb_model":     model_status(),
        "db_path":      DB_PATH,
        "db_exists":    os.path.exists(DB_PATH),
        "cookies_path": COOKIES_PATH,
        "cookies_ok":   _has_cookies(),   # quick way to confirm cookies loaded
    }), 200
