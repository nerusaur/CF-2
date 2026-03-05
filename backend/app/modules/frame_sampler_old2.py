"""
ChildFocus - Sprint 1: Frame Sampling Module
backend/app/modules/frame_sampler.py

Fixes applied:
  1. Bot detection     → cookiesfrombrowser + User-Agent headers
  2. ATT = 0.0        → ffmpeg-based audio extraction fallback + soundfile
  3. thumbnail = 0.0  → Pillow import fix + robust error handling
"""

import os
import cv2
import numpy as np
import tempfile
import subprocess
import requests
from io import BytesIO

# ── Optional imports with graceful fallback ────────────────────────────────────
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("[WARN] librosa not installed. ATT will use ffmpeg fallback.")

try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False
    print("[WARN] Pillow not installed. Thumbnail intensity will be 0.0.")

try:
    import yt_dlp
    YTDLP_AVAILABLE = True
except ImportError:
    YTDLP_AVAILABLE = False
    print("[ERROR] yt-dlp not installed. Run: pip install yt-dlp")


# ── Constants ──────────────────────────────────────────────────────────────────
SNAPSHOT_DURATION = 1.0      # seconds per fast snapshot (T_snap)
SEGMENT_DURATION  = 20       # seconds per deep-analysis segment
NUM_SEGMENTS      = 3        # S1 (beginning), S2 (middle), S3 (end)
FRAME_SAMPLE_RATE = 1        # extract 1 frame per second
C_MAX             = 4.0      # max cuts/sec for FCR normalization
S_MAX             = 128.0    # max saturation std for CSV normalization


# ── Video Download ─────────────────────────────────────────────────────────────
def download_video_stream(video_id: str, max_duration: int = 90) -> str:
    """
    Downloads a YouTube video to a temp .mp4 file using yt-dlp.
    Includes browser cookies + User-Agent to bypass bot detection.
    Returns the path to the downloaded file.
    """
    if not YTDLP_AVAILABLE:
        raise RuntimeError("yt-dlp is not installed.")

    output_path = tempfile.mktemp(suffix=".mp4")

    # ── FIX 1: Browser cookies + proper User-Agent to bypass bot detection ──
    ydl_opts = {
        "format": "worst[ext=mp4]/worst",
        "outtmpl": output_path,
        "quiet": True,
        "no_warnings": True,

        # Browser cookies bypass YouTube's bot detection
        # Uses your locally logged-in Chrome session
        "cookiesfrombrowser": ("chrome",),

        # Realistic browser headers
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },

        # Limit download to max_duration seconds for efficiency
        "download_sections": [f"*0-{max_duration}"],
        "postprocessors": [],
    }

    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        # Fallback: try without cookies if Chrome not available
        print(f"[WARN] Download with cookies failed: {e}. Retrying without cookies...")
        ydl_opts_fallback = {
            "format": "worst[ext=mp4]/worst",
            "outtmpl": output_path,
            "quiet": True,
            "no_warnings": True,
            "http_headers": {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            },
            "download_sections": [f"*0-{max_duration}"],
        }
        with yt_dlp.YoutubeDL(ydl_opts_fallback) as ydl:
            ydl.download([url])

    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Downloaded file not found at {output_path}")

    return output_path


# ── Frame Extraction ───────────────────────────────────────────────────────────
def extract_frames_from_video(video_path: str, start_sec: int, duration: int):
    """
    Extracts frames from a video file between start_sec and start_sec+duration.
    Returns (list of numpy BGR frames, total video duration in seconds).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], 0.0

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_dur    = total_frames / fps

    end_frame   = int(min((start_sec + duration) * fps, total_frames))
    start_frame = int(start_sec * fps)
    sample_every = max(1, int(fps * FRAME_SAMPLE_RATE))

    frames    = []
    frame_idx = start_frame

    while frame_idx < end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_idx += sample_every

    cap.release()
    return frames, video_dur


# ── Heuristic Feature: FCR ─────────────────────────────────────────────────────
def compute_frame_change_rate(frames: list) -> float:
    """
    FCR = min(1, cuts_per_second / C_max)  where C_max = 4
    Detects scene cuts using mean absolute pixel difference between frames.
    """
    if len(frames) < 2:
        return 0.0

    cuts = 0
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

    for frame in frames[1:]:
        gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff     = cv2.absdiff(prev_gray, gray)
        mean_diff = np.mean(diff)
        if mean_diff > 25:   # threshold for scene cut
            cuts += 1
        prev_gray = gray

    duration_sec = len(frames) / max(FRAME_SAMPLE_RATE, 1)
    cuts_per_sec = cuts / max(duration_sec, 1)
    return round(min(1.0, cuts_per_sec / C_MAX), 4)


# ── Heuristic Feature: CSV ─────────────────────────────────────────────────────
def compute_color_saturation_variance(frames: list) -> float:
    """
    CSV = std(saturation_means) / S_max
    High variance = rapidly shifting, visually intense colors.
    """
    if not frames:
        return 0.0

    saturations = []
    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturations.append(float(np.mean(hsv[:, :, 1])))

    std_sat = float(np.std(saturations))
    return round(min(1.0, std_sat / S_MAX), 4)


# ── Heuristic Feature: ATT ─────────────────────────────────────────────────────
def compute_audio_activity_proxy(video_path: str, start_sec: int, duration: int) -> float:
    """
    ATTprox = normalized spectral flux

    FIX 2: Three-layer approach to get ATT working:
      Layer 1 — librosa directly on the video file
      Layer 2 — ffmpeg extracts audio to WAV, then librosa reads it
      Layer 3 — ffmpeg-only RMS energy fallback (no librosa needed)
    """
    # ── Layer 1: Try librosa directly ────────────────────────────────────────
    if LIBROSA_AVAILABLE:
        try:
            y, sr = librosa.load(
                video_path,
                offset=float(start_sec),
                duration=float(duration),
                sr=22050,
                mono=True
            )
            if len(y) > 100:
                onset_env = librosa.onset.onset_strength(y=y, sr=sr)
                flux = float(np.mean(onset_env))
                return round(min(1.0, flux / 10.0), 4)
        except Exception as e:
            print(f"[ATT] librosa direct failed: {e}. Trying ffmpeg extraction...")

    # ── Layer 2: Extract audio with ffmpeg, then use librosa ─────────────────
    wav_path = tempfile.mktemp(suffix=".wav")
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(start_sec),
                "-t",  str(duration),
                "-i",  video_path,
                "-vn",                      # no video
                "-acodec", "pcm_s16le",     # uncompressed WAV
                "-ar", "22050",             # sample rate
                "-ac", "1",                 # mono
                wav_path
            ],
            capture_output=True,
            timeout=30
        )

        if result.returncode == 0 and os.path.exists(wav_path) and os.path.getsize(wav_path) > 1000:
            if LIBROSA_AVAILABLE:
                y, sr = librosa.load(wav_path, sr=22050)
                if len(y) > 100:
                    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
                    flux = float(np.mean(onset_env))
                    os.remove(wav_path)
                    return round(min(1.0, flux / 10.0), 4)

            # ── Layer 3: Pure numpy RMS if librosa unavailable ────────────────
            import wave
            import struct
            with wave.open(wav_path, 'rb') as wf:
                frames_data = wf.readframes(wf.getnframes())
                samples = np.frombuffer(frames_data, dtype=np.int16).astype(np.float32)
                samples /= 32768.0  # normalize to [-1, 1]

            if len(samples) > 100:
                # Compute RMS energy in chunks as proxy for tempo activity
                chunk_size = 2205  # ~0.1 seconds at 22050 Hz
                rms_values = []
                for i in range(0, len(samples) - chunk_size, chunk_size):
                    chunk = samples[i:i + chunk_size]
                    rms_values.append(float(np.sqrt(np.mean(chunk ** 2))))

                if rms_values:
                    # Variance of RMS = audio tempo activity proxy
                    att = float(np.std(rms_values)) * 10.0
                    os.remove(wav_path)
                    return round(min(1.0, att), 4)

    except FileNotFoundError:
        print("[ATT] ffmpeg not found. Install ffmpeg: https://ffmpeg.org/download.html")
    except Exception as e:
        print(f"[ATT] ffmpeg extraction failed: {e}")
    finally:
        if os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception:
                pass

    print("[ATT] All methods failed. Returning 0.0")
    return 0.0


# ── Heuristic Feature: Thumbnail Intensity ────────────────────────────────────
def compute_thumbnail_intensity(thumbnail_url: str) -> float:
    """
    Thumb = (w_s × mean_sat) + (w_t × text_density)
    FIX 3: Proper Pillow import + robust error handling + OpenCV fallback
    """
    if not thumbnail_url:
        return 0.0

    W_S = 0.7   # saturation weight
    W_T = 0.3   # text/edge density weight

    try:
        resp = requests.get(thumbnail_url, timeout=8)
        resp.raise_for_status()
        img_bytes = resp.content

        # ── Try Pillow first ──────────────────────────────────────────────────
        if PILLOW_AVAILABLE:
            img_pil  = Image.open(BytesIO(img_bytes)).convert("RGB")
            img_cv   = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        else:
            # OpenCV fallback — decode directly from bytes
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img_cv    = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img_cv is None:
                return 0.0

        # Saturation from HSV
        hsv      = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        mean_sat = float(np.mean(hsv[:, :, 1])) / 255.0

        # Text density proxy via Canny edge detection
        gray         = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        edges        = cv2.Canny(gray, 100, 200)
        text_density = float(np.sum(edges > 0)) / float(edges.size)

        result = min(1.0, (W_S * mean_sat) + (W_T * text_density))
        return round(result, 4)

    except requests.exceptions.RequestException as e:
        print(f"[THUMB] Network error fetching thumbnail: {e}")
    except Exception as e:
        print(f"[THUMB] Error computing thumbnail intensity: {e}")

    return 0.0


# ── Main Sprint 1 Function ─────────────────────────────────────────────────────
def sample_video(video_id: str, thumbnail_url: str = "") -> dict:
    """
    Main Sprint 1 function:
      1. Downloads YouTube video (with bot-detection bypass)
      2. Extracts 3 segments (beginning / middle / end)
      3. Computes FCR, CSV, ATT per segment
      4. Computes thumbnail intensity
      5. Returns all features + aggregate heuristic score
    """
    video_path = None
    try:
        print(f"[SAMPLER] Starting analysis for video: {video_id}")

        # ── Step 1: Download ──────────────────────────────────────────────────
        video_path = download_video_stream(video_id, max_duration=90)
        print(f"[SAMPLER] Downloaded to: {video_path}")

        # ── Step 2: Get video duration ────────────────────────────────────────
        cap           = cv2.VideoCapture(video_path)
        fps           = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames  = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        video_duration = total_frames / fps
        cap.release()
        print(f"[SAMPLER] Video duration: {video_duration:.1f}s")

        # ── Step 3: Define 3 segment start times ─────────────────────────────
        half      = max(0, int(video_duration / 2) - SEGMENT_DURATION // 2)
        end_start = max(0, int(video_duration) - SEGMENT_DURATION)
        seg_starts = [0, half, end_start]

        # ── Step 4: Compute features per segment ─────────────────────────────
        segments = []
        for i, start in enumerate(seg_starts):
            print(f"[SAMPLER] Analyzing segment S{i+1} at {start}s...")
            frames, _ = extract_frames_from_video(video_path, start, SEGMENT_DURATION)

            if not frames:
                print(f"[SAMPLER] No frames extracted for S{i+1}, skipping.")
                segments.append({
                    "segment_id":     f"S{i+1}",
                    "offset_seconds": start,
                    "length_seconds": SEGMENT_DURATION,
                    "fcr":   0.0,
                    "csv":   0.0,
                    "att":   0.0,
                    "score_h": 0.0
                })
                continue

            fcr = compute_frame_change_rate(frames)
            csv = compute_color_saturation_variance(frames)
            att = compute_audio_activity_proxy(video_path, start, SEGMENT_DURATION)

            # Heuristic score formula from thesis:
            # Score_H = (0.35×FCR) + (0.25×CSV) + (0.20×ATT) + (0.20×Thumb)
            # Thumb applied at aggregate level below
            score_h = round((0.35 * fcr) + (0.25 * csv) + (0.20 * att), 4)

            print(f"[SAMPLER] S{i+1} → FCR={fcr}, CSV={csv}, ATT={att}, Score_H={score_h}")

            segments.append({
                "segment_id":     f"S{i+1}",
                "offset_seconds": start,
                "length_seconds": SEGMENT_DURATION,
                "fcr":            fcr,
                "csv":            csv,
                "att":            att,
                "score_h":        score_h
            })

        # ── Step 5: Thumbnail intensity ───────────────────────────────────────
        print(f"[SAMPLER] Computing thumbnail intensity...")
        thumb_score = compute_thumbnail_intensity(thumbnail_url) if thumbnail_url else 0.0
        print(f"[SAMPLER] Thumbnail intensity: {thumb_score}")

        # ── Step 6: Aggregate score (conservative — use max across segments) ──
        max_seg_score = max(s["score_h"] for s in segments) if segments else 0.0

        # Final heuristic score includes thumbnail (thesis formula weight 0.20)
        aggregate_score = round(
            (0.80 * max_seg_score) + (0.20 * thumb_score), 4
        )

        # ── Step 7: Preliminary OIR classification (heuristic only) ──────────
        if aggregate_score >= 0.75:
            preliminary_label = "Overstimulating"
        elif aggregate_score <= 0.35:
            preliminary_label = "Safe"
        else:
            preliminary_label = "Uncertain — needs NB classification (Sprint 2)"

        return {
            "video_id":                  video_id,
            "video_duration_sec":        round(video_duration, 1),
            "thumbnail_url":             thumbnail_url,
            "thumbnail_intensity":       thumb_score,
            "segments":                  segments,
            "aggregate_heuristic_score": aggregate_score,
            "preliminary_label":         preliminary_label,
            "status":                    "success"
        }

    except Exception as e:
        print(f"[SAMPLER] Fatal error: {e}")
        return {
            "video_id": video_id,
            "status":   "error",
            "message":  str(e)
        }

    finally:
        # Always clean up temp file
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
                print(f"[SAMPLER] Cleaned up temp file.")
            except Exception:
                pass
