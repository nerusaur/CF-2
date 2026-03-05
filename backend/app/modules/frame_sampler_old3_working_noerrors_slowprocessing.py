"""
ChildFocus - Sprint 1: Frame Sampling Module
backend/app/modules/frame_sampler.py

Fixes in this version:
  1. Chrome cookie error    → removed cookiesfrombrowser (not needed for public videos)
  2. PySoundFile warning    → ffmpeg extracts WAV first, librosa reads WAV cleanly
  3. FutureWarning audioread → eliminated by WAV pipeline
  4. Video not available    → validate video BEFORE downloading
  5. Thumbnail scraper      → robust Pillow + OpenCV fallback
  6. JS runtime             → explicit Node.js path + remote components for challenge solving
  7. Geo-block              → extractor_args force multiple player clients
"""

import os
import cv2
import numpy as np
import tempfile
import subprocess
import requests

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("[WARN] librosa not installed. ATT will use numpy RMS fallback.")

try:
    from PIL import Image
    from io import BytesIO
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False
    print("[WARN] Pillow not installed. Using OpenCV for thumbnail.")

try:
    import yt_dlp
    YTDLP_AVAILABLE = True
except ImportError:
    YTDLP_AVAILABLE = False
    print("[ERROR] yt-dlp not installed.")

# ── Constants (thesis formulas) ────────────────────────────────────────────────
SEGMENT_DURATION  = 20
NUM_SEGMENTS      = 3
FRAME_SAMPLE_RATE = 1
C_MAX             = 4.0
S_MAX             = 128.0

# ── Node.js path (explicit, for yt-dlp JS challenge solving) ──────────────────
NODE_PATH = r"C:\Program Files\nodejs\node.exe"


def _base_ydl_opts() -> dict:
    """
    Shared yt-dlp options used by both validate and download.
    - Explicit Node.js path so yt-dlp doesn't fail to auto-detect v24
    - remote_components downloads the EJS challenge solver from GitHub
    - extractor_args tries multiple player clients so geo-restricted
      videos that work in browser can be reached via android_vr / tv_embedded
    """
    return {
        "quiet":              True,
        "no_warnings":        True,
        "geo_bypass":         True,
        "geo_bypass_country": "US",
        "js_runtimes": {"node": {"path": NODE_PATH}},
        "remote_components":  ["ejs:github"],
        "extractor_args": {
            "youtube": {
                "player_client": [
                    "web",
                    "web_safari",
                    "android_vr",
                    "tv_embedded",
                    "web_embedded",
                ]
            }
        },
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": (
                "text/html,application/xhtml+xml,"
                "application/xml;q=0.9,*/*;q=0.8"
            ),
        },
    }


# ── Step 0: Validate video before downloading ──────────────────────────────────
def validate_video(video_id: str, is_short: bool = False) -> dict:
    """
    Checks if a YouTube video is publicly available before attempting download.
    Tries both watch and Shorts URLs. Returns the URL that succeeded so
    download_video_stream can reuse it without a second round-trip.
    """
    if not YTDLP_AVAILABLE:
        return {"available": False, "reason": "yt-dlp not installed"}

    ydl_opts = {**_base_ydl_opts(), "skip_download": True}

    urls_to_try = (
        [
            f"https://www.youtube.com/shorts/{video_id}",
            f"https://www.youtube.com/watch?v={video_id}",
        ]
        if is_short else
        [
            f"https://www.youtube.com/watch?v={video_id}",
            f"https://www.youtube.com/shorts/{video_id}",
        ]
    )

    last_error = None
    for url in urls_to_try:
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return {
                    "available": True,
                    "title":     info.get("title", "Unknown"),
                    "duration":  info.get("duration", 0),
                    "uploader":  info.get("uploader", "Unknown"),
                    "url_used":  url,
                    "reason":    "ok",
                }
        except Exception as e:
            last_error = e
            continue

    msg = str(last_error).lower()
    if "not available" in msg:
        reason = "Video is not available in this region or has been removed"
    elif "private" in msg:
        reason = "Video is private"
    elif "age" in msg:
        reason = "Video is age-restricted"
    else:
        reason = str(last_error)
    return {"available": False, "reason": reason}


# ── Step 1: Download video ─────────────────────────────────────────────────────
def download_video_stream(
    video_id: str, max_duration: int = 90, url_used: str = None
) -> str:
    """
    Downloads a YouTube video to a temp .mp4 file.
    Reuses the URL that already passed validation to avoid a second check.
    """
    if not YTDLP_AVAILABLE:
        raise RuntimeError("yt-dlp is not installed.")

    output_path = tempfile.mktemp(suffix=".mp4")

    ydl_opts = {
        **_base_ydl_opts(),
        "format":            "worst[ext=mp4]/worst",
        "outtmpl":           output_path,
        "download_sections": [f"*0-{max_duration}"],
        "postprocessors":    [],
    }

    url = url_used or f"https://www.youtube.com/watch?v={video_id}"

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Downloaded file not found at {output_path}")

    return output_path


# ── Audio extraction via ffmpeg ────────────────────────────────────────────────
def extract_audio_to_wav(video_path: str, start_sec: int, duration: int) -> str | None:
    """
    FIX for PySoundFile warning + FutureWarning:
    ffmpeg extracts a clean PCM WAV → librosa reads it without any warnings.
    The warnings only appear when librosa tries to read MP4 audio directly.
    """
    wav_path = tempfile.mktemp(suffix=".wav")
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(start_sec),
                "-t",  str(duration),
                "-i",  video_path,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "22050",
                "-ac", "1",
                wav_path,
            ],
            capture_output=True,
            timeout=30,
            text=True,
        )
        if (
            result.returncode == 0
            and os.path.exists(wav_path)
            and os.path.getsize(wav_path) > 500
        ):
            return wav_path
        print(f"[AUDIO] ffmpeg error (code {result.returncode}): {result.stderr[:150]}")
        return None
    except FileNotFoundError:
        print("[AUDIO] ffmpeg not found. Add to PATH: https://www.gyan.dev/ffmpeg/builds/")
        return None
    except Exception as e:
        print(f"[AUDIO] Exception: {e}")
        return None


# ── Frame extraction ───────────────────────────────────────────────────────────
def extract_frames_from_video(video_path: str, start_sec: int, duration: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], 0.0

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_dur    = total_frames / fps
    end_frame    = int(min((start_sec + duration) * fps, total_frames))
    start_frame  = int(start_sec * fps)
    sample_every = max(1, int(fps * FRAME_SAMPLE_RATE))

    frames = []
    idx    = start_frame
    while idx < end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        idx += sample_every

    cap.release()
    return frames, video_dur


# ── FCR ────────────────────────────────────────────────────────────────────────
def compute_frame_change_rate(frames: list) -> float:
    """FCR = min(1, cuts_per_second / C_max) where C_max = 4"""
    if len(frames) < 2:
        return 0.0
    cuts      = 0
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for frame in frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if float(np.mean(cv2.absdiff(prev_gray, gray))) > 25:
            cuts += 1
        prev_gray = gray
    return round(
        min(1.0, (cuts / max(len(frames) / FRAME_SAMPLE_RATE, 1)) / C_MAX), 4
    )


# ── CSV ────────────────────────────────────────────────────────────────────────
def compute_color_saturation_variance(frames: list) -> float:
    """CSV = std(saturation_means) / S_max"""
    if not frames:
        return 0.0
    sats = [
        float(np.mean(cv2.cvtColor(f, cv2.COLOR_BGR2HSV)[:, :, 1]))
        for f in frames
    ]
    return round(min(1.0, float(np.std(sats)) / S_MAX), 4)


# ── ATT ────────────────────────────────────────────────────────────────────────
def compute_audio_activity_proxy(
    video_path: str, start_sec: int, duration: int
) -> float:
    """
    ATTprox = normalized spectral flux
    Pipeline: ffmpeg → WAV → librosa (clean, no warnings)
    """
    wav_path = None
    try:
        wav_path = extract_audio_to_wav(video_path, start_sec, duration)
        if wav_path is None:
            return 0.0

        if LIBROSA_AVAILABLE:
            y, sr = librosa.load(wav_path, sr=None, mono=True)
            if len(y) > 100:
                onset_env = librosa.onset.onset_strength(y=y, sr=sr)
                return round(min(1.0, float(np.mean(onset_env)) / 10.0), 4)

        # Numpy RMS fallback
        import wave
        with wave.open(wav_path, "rb") as wf:
            samples = (
                np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
                .astype(np.float32) / 32768.0
            )
        if len(samples) > 100:
            chunk = 2205
            rms = [
                float(np.sqrt(np.mean(samples[i : i + chunk] ** 2)))
                for i in range(0, len(samples) - chunk, chunk)
            ]
            if rms:
                return round(min(1.0, float(np.std(rms)) * 10.0), 4)

    except Exception as e:
        print(f"[ATT] Error: {e}")
    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception:
                pass
    return 0.0


# ── Thumbnail Intensity ────────────────────────────────────────────────────────
def compute_thumbnail_intensity(thumbnail_url: str) -> float:
    """Thumb = (0.7 × mean_sat) + (0.3 × edge_density)"""
    if not thumbnail_url:
        return 0.0
    try:
        resp = requests.get(thumbnail_url, timeout=8)
        resp.raise_for_status()
        img_bytes = resp.content

        if PILLOW_AVAILABLE:
            img_cv = cv2.cvtColor(
                np.array(Image.open(BytesIO(img_bytes)).convert("RGB")),
                cv2.COLOR_RGB2BGR,
            )
        else:
            img_cv = cv2.imdecode(
                np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if img_cv is None:
                return 0.0

        mean_sat = (
            float(np.mean(cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)[:, :, 1])) / 255.0
        )
        gray         = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        text_density = float(np.sum(cv2.Canny(gray, 100, 200) > 0)) / float(gray.size)

        return round(min(1.0, (0.7 * mean_sat) + (0.3 * text_density)), 4)
    except Exception as e:
        print(f"[THUMB] Error: {e}")
        return 0.0


# ── Main Function ──────────────────────────────────────────────────────────────
def sample_video(video_id: str, thumbnail_url: str = "") -> dict:
    """
    Sprint 1 main function:
    Validate → Download → Extract → Compute FCR/CSV/ATT → Score → Return OIR
    """
    video_path = None
    try:
        print(f"\n[SAMPLER] ══════════════════════════════════════")
        print(f"[SAMPLER] Analyzing video_id: {video_id}")

        # Step 0: Validate
        print(f"[SAMPLER] Checking availability...")
        val = validate_video(video_id)
        if not val["available"]:
            print(f"[SAMPLER] ✗ Unavailable: {val['reason']}")
            return {
                "video_id": video_id,
                "status":   "unavailable",
                "reason":   val["reason"],
                "message":  f"Video cannot be analyzed: {val['reason']}",
            }
        print(f"[SAMPLER] ✓ '{val['title']}' ({val['duration']}s)")

        # Step 1: Download — reuse the URL that already worked in validation
        print(f"[SAMPLER] Downloading (first 90s)...")
        video_path = download_video_stream(
            video_id,
            max_duration=90,
            url_used=val.get("url_used"),
        )

        # Step 2: Duration
        cap            = cv2.VideoCapture(video_path)
        video_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (
            cap.get(cv2.CAP_PROP_FPS) or 30
        )
        cap.release()
        print(f"[SAMPLER] ✓ Duration: {video_duration:.1f}s")

        # Step 3: Segment starts
        seg_starts = [
            0,
            max(0, int(video_duration / 2) - SEGMENT_DURATION // 2),
            max(0, int(video_duration) - SEGMENT_DURATION),
        ]

        # Step 4: Per-segment features
        segments = []
        for i, start in enumerate(seg_starts):
            print(f"[SAMPLER] S{i+1} at {start}s...")
            frames, _ = extract_frames_from_video(video_path, start, SEGMENT_DURATION)
            if not frames:
                segments.append(
                    {
                        "segment_id":     f"S{i+1}",
                        "offset_seconds": start,
                        "length_seconds": SEGMENT_DURATION,
                        "fcr": 0.0, "csv": 0.0, "att": 0.0, "score_h": 0.0,
                    }
                )
                continue

            fcr     = compute_frame_change_rate(frames)
            csv     = compute_color_saturation_variance(frames)
            att     = compute_audio_activity_proxy(video_path, start, SEGMENT_DURATION)
            score_h = round((0.35 * fcr) + (0.25 * csv) + (0.20 * att), 4)

            print(f"[SAMPLER] S{i+1} FCR={fcr} | CSV={csv} | ATT={att} | H={score_h}")
            segments.append(
                {
                    "segment_id":     f"S{i+1}",
                    "offset_seconds": start,
                    "length_seconds": SEGMENT_DURATION,
                    "fcr": fcr, "csv": csv, "att": att, "score_h": score_h,
                }
            )

        # Step 5: Thumbnail
        thumb = compute_thumbnail_intensity(thumbnail_url) if thumbnail_url else 0.0
        print(f"[SAMPLER] Thumbnail: {thumb}")

        # Step 6: OIR
        max_seg   = max(s["score_h"] for s in segments) if segments else 0.0
        agg_score = round((0.80 * max_seg) + (0.20 * thumb), 4)
        label     = (
            "Overstimulating" if agg_score >= 0.75
            else "Safe"       if agg_score <= 0.35
            else "Uncertain"
        )

        print(f"[SAMPLER] ✓ Score: {agg_score} → {label}")
        print(f"[SAMPLER] ══════════════════════════════════════\n")

        return {
            "video_id":                  video_id,
            "video_title":               val.get("title", ""),
            "video_duration_sec":        round(video_duration, 1),
            "thumbnail_url":             thumbnail_url,
            "thumbnail_intensity":       thumb,
            "segments":                  segments,
            "aggregate_heuristic_score": agg_score,
            "preliminary_label":         label,
            "status":                    "success",
        }

    except Exception as e:
        print(f"[SAMPLER] ✗ Fatal: {e}")
        import traceback
        traceback.print_exc()
        return {"video_id": video_id, "status": "error", "message": str(e)}
    finally:
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception:
                pass