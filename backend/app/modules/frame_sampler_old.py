import os
import cv2
import numpy as np
import yt_dlp
import tempfile
import librosa

# ── Constants ──────────────────────────────────────────────────────────────────
SNAPSHOT_DURATION = 1.0      # seconds per fast snapshot (T_snap)
SEGMENT_DURATION  = 20       # seconds per deep-analysis segment
NUM_SEGMENTS      = 3        # S1 (beginning), S2 (middle), S3 (end)
FRAME_SAMPLE_RATE = 1        # extract 1 frame per second


def download_video_stream(video_id: str, max_duration: int = 90) -> str:
    """
    Downloads a YouTube video to a temp file using yt-dlp.
    Returns the path to the downloaded file.
    Only downloads up to max_duration seconds (default 90s).
    """
    output_path = tempfile.mktemp(suffix=".mp4")
    ydl_opts = {
        "format": "worst[ext=mp4]/worst",   # lowest quality for speed
        "outtmpl": output_path,
        "quiet": True,
        "no_warnings": True,
        "external_downloader_args": ["-t", str(max_duration)],
    }
    url = f"https://www.youtube.com/watch?v={video_id}"
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_path


def extract_frames_from_video(video_path: str, start_sec: int, duration: int) -> list:
    """
    Extracts frames from a video file between start_sec and start_sec+duration.
    Returns a list of numpy arrays (BGR frames).
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps

    start_frame = int(start_sec * fps)
    end_frame   = int(min((start_sec + duration) * fps, total_frames))

    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    sample_every = int(fps * FRAME_SAMPLE_RATE)  # 1 frame per second
    frame_idx = start_frame

    while frame_idx < end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_idx += sample_every

    cap.release()
    return frames, video_duration


def compute_frame_change_rate(frames: list) -> float:
    """
    FCR = min(1, cuts_per_second / C_max) where C_max = 4
    Uses frame-difference threshold to detect scene cuts.
    """
    if len(frames) < 2:
        return 0.0

    C_MAX = 4.0
    cuts = 0
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

    for frame in frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        mean_diff = np.mean(diff)
        if mean_diff > 25:   # threshold for scene cut detection
            cuts += 1
        prev_gray = gray

    duration_sec = len(frames) / FRAME_SAMPLE_RATE
    cuts_per_sec = cuts / max(duration_sec, 1)
    return min(1.0, cuts_per_sec / C_MAX)


def compute_color_saturation_variance(frames: list) -> float:
    """
    CSV = std(saturation) / S_max
    Converts frames to HSV and measures saturation variance.
    """
    if not frames:
        return 0.0

    S_MAX = 128.0
    saturations = []

    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        saturations.append(np.mean(sat))

    std_sat = np.std(saturations)
    return min(1.0, std_sat / S_MAX)


def compute_audio_activity_proxy(video_path: str, start_sec: int, duration: int) -> float:
    """
    ATTprox = normalized spectral flux
    Uses librosa to extract audio and compute spectral flux.
    """
    try:
        y, sr = librosa.load(video_path, offset=start_sec, duration=duration, sr=22050)
        if len(y) == 0:
            return 0.0
        # Spectral flux = mean of onset strength envelope
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        flux = float(np.mean(onset_env))
        # Normalize: typical range 0-10, cap at 1
        return min(1.0, flux / 10.0)
    except Exception:
        return 0.0


def compute_thumbnail_intensity(thumbnail_url: str) -> float:
    """
    Thumb = (w_s × mean_sat) + (w_t × text_density)
    Downloads thumbnail and computes saturation mean.
    Text density approximated via edge detection.
    """
    import requests
    from io import BytesIO
    from PIL import Image

    W_S = 0.7   # weight for saturation
    W_T = 0.3   # weight for text/edge density

    try:
        resp = requests.get(thumbnail_url, timeout=5)
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Saturation
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        mean_sat = np.mean(hsv[:, :, 1]) / 255.0

        # Text density (edge proxy)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        text_density = np.sum(edges > 0) / edges.size

        return min(1.0, (W_S * mean_sat) + (W_T * text_density))
    except Exception:
        return 0.0


def sample_video(video_id: str, thumbnail_url: str = "") -> dict:
    """
    Main Sprint 1 function: Downloads video, extracts 3 segments,
    computes all heuristic features per segment.
    Returns a dict of segment scores and raw features.
    """
    try:
        video_path = download_video_stream(video_id)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        video_duration = total_frames / fps
        cap.release()

        # Define 3 segment start times: beginning, middle, end
        seg_starts = [
            0,
            max(0, int(video_duration / 2) - SEGMENT_DURATION // 2),
            max(0, int(video_duration) - SEGMENT_DURATION)
        ]

        segments = []
        for i, start in enumerate(seg_starts):
            frames, _ = extract_frames_from_video(video_path, start, SEGMENT_DURATION)

            fcr = compute_frame_change_rate(frames)
            csv = compute_color_saturation_variance(frames)
            att = compute_audio_activity_proxy(video_path, start, SEGMENT_DURATION)

            # Heuristic Score per segment
            score_h = (0.35 * fcr) + (0.25 * csv) + (0.20 * att)

            segments.append({
                "segment_id": f"S{i+1}",
                "offset_seconds": start,
                "length_seconds": SEGMENT_DURATION,
                "fcr": round(fcr, 4),
                "csv": round(csv, 4),
                "att": round(att, 4),
                "score_h": round(score_h, 4)
            })

        # Thumbnail analysis
        thumb_score = compute_thumbnail_intensity(thumbnail_url) if thumbnail_url else 0.0

        # Aggregate: max score across segments (conservative blocking)
        max_score = max(s["score_h"] for s in segments)

        # Cleanup temp file
        os.remove(video_path)

        return {
            "video_id": video_id,
            "video_duration_sec": round(video_duration, 1),
            "thumbnail_intensity": round(thumb_score, 4),
            "segments": segments,
            "aggregate_heuristic_score": round(max_score, 4),
            "status": "success"
        }

    except Exception as e:
        return {"video_id": video_id, "status": "error", "message": str(e)}