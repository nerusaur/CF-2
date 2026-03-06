"""
ChildFocus - Classification Routes
backend/app/routes/classify.py

Endpoints:
  POST /classify_fast  → metadata-only quick scan (no download)
  POST /classify_full  → full hybrid analysis (downloads video)
"""

from flask import Blueprint, request, jsonify
from app.modules.hybrid_fusion import classify_fast as _classify_fast, classify_full as _classify_full
from app.modules.youtube_api   import get_video_metadata, extract_video_id, get_thumbnail_url
from app.utils.validators      import validate_video_url
from app.utils.logger          import log_classification

classify_bp = Blueprint("classify", __name__)


@classify_bp.route("/classify_fast", methods=["POST"])
def classify_fast():
    """
    Fast classification using only metadata + Naïve Bayes.
    No video download. Returns preliminary OIR within ~1-2 seconds.

    Request body:
        { "video_url": "https://youtube.com/watch?v=..." }

    Response:
        {
            "video_id":          "...",
            "score_nb":          0.72,
            "preliminary_label": "Overstimulating",
            "action":            "pending_full_analysis",
            "nb_probabilities":  {...},
            "status":            "success"
        }
    """
    data = request.get_json()
    if not data or "video_url" not in data:
        return jsonify({"error": "Missing video_url in request body"}), 400

    video_url = data["video_url"]
    error     = validate_video_url(video_url)
    if error:
        return jsonify({"error": error}), 400

    video_id = extract_video_id(video_url)

    # Fetch metadata from YouTube API
    metadata = get_video_metadata(video_id)
    if "error" in metadata:
        return jsonify({"error": metadata["error"], "video_id": video_id}), 404

    # Run fast NB-only classification
    result = _classify_fast(
        video_id    = video_id,
        title       = metadata.get("title", ""),
        tags        = metadata.get("tags", []),
        description = metadata.get("description", ""),
    )

    # Attach metadata to response
    result["metadata"] = {
        "title":        metadata.get("title", ""),
        "channel":      metadata.get("channel", ""),
        "thumbnail_url": metadata.get("thumbnail_url", ""),
        "duration":     metadata.get("duration", ""),
        "view_count":   metadata.get("view_count", 0),
    }

    log_classification(video_id, result.get("preliminary_label"), "fast")
    return jsonify(result), 200


@classify_bp.route("/classify_full", methods=["POST"])
def classify_full():
    """
    Full hybrid classification: downloads video + runs heuristic + NB fusion.
    Takes 20-60 seconds depending on video length and hardware.

    Request body:
        {
            "video_url":     "https://youtube.com/watch?v=...",
            "thumbnail_url": "https://i.ytimg.com/vi/.../hqdefault.jpg"  (optional)
        }

    Response:
        {
            "video_id":    "...",
            "score_nb":    0.72,
            "score_h":     0.68,
            "score_final": 0.698,
            "oir_label":   "Overstimulating",
            "action":      "block",
            "heuristic_details": { "segments": [...], "thumbnail": 0.5 },
            "nb_details":  { "probabilities": {...} },
            "status":      "success"
        }
    """
    data = request.get_json()
    if not data or "video_url" not in data:
        return jsonify({"error": "Missing video_url in request body"}), 400

    video_url = data["video_url"]
    error     = validate_video_url(video_url)
    if error:
        return jsonify({"error": error}), 400

    video_id = extract_video_id(video_url)

    # Get thumbnail URL — try provided first, then fetch from API
    thumbnail_url = data.get("thumbnail_url", "")
    if not thumbnail_url:
        thumbnail_url = get_thumbnail_url(video_id)

    # Fetch metadata for NB scoring
    metadata = get_video_metadata(video_id)
    if "error" in metadata:
        # Still proceed with heuristic only if metadata fails
        title, tags, description = "", [], ""
    else:
        title       = metadata.get("title", "")
        tags        = metadata.get("tags", [])
        description = metadata.get("description", "")

    # Run full hybrid classification
    result = _classify_full(
        video_id      = video_id,
        thumbnail_url = thumbnail_url,
        title         = title,
        tags          = tags,
        description   = description,
    )

    log_classification(video_id, result.get("oir_label"), "full")
    return jsonify(result), 200