from flask import Blueprint, request, jsonify
from app.modules.frame_sampler import sample_video
from app.modules.youtube_api import get_video_metadata, extract_video_id

classify_bp = Blueprint("classify", __name__)

@classify_bp.route("/classify_fast", methods=["POST"])
def classify_fast():
    """
    Fast classification using only metadata + snapshot heuristic.
    Body: { "video_url": "https://youtube.com/watch?v=..." }
    """
    data = request.get_json()
    if not data or "video_url" not in data:
        return jsonify({"error": "Missing video_url"}), 400

    video_id = extract_video_id(data["video_url"])
    metadata = get_video_metadata(video_id)

    if "error" in metadata:
        return jsonify(metadata), 404

    # Sprint 1: return metadata + signal that full analysis is needed
    return jsonify({
        "video_id": video_id,
        "metadata": metadata,
        "status": "fast_scan_complete",
        "next": "POST /classify_full for heuristic analysis"
    })


@classify_bp.route("/classify_full", methods=["POST"])
def classify_full():
    """
    Full classification: downloads video, extracts frames, runs heuristic.
    Body: { "video_url": "...", "thumbnail_url": "..." }
    """
    data = request.get_json()
    if not data or "video_url" not in data:
        return jsonify({"error": "Missing video_url"}), 400

    video_id = extract_video_id(data["video_url"])
    thumbnail_url = data.get("thumbnail_url", "")

    result = sample_video(video_id, thumbnail_url)
    return jsonify(result)