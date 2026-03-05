"""
ChildFocus - Frame Sampling Module
backend/app/modules/frame_sampler.py

Architecture:
  1. fetch_metadata()     → lightweight yt-dlp call, no download, gets title/duration/url
  2. fetch_segment_only() → downloads ONLY the 20s window needed per segment
  3. ThreadPoolExecutor   → 3 segment downloads + thumbnail ALL run in parallel
  4. Analyze immediately  → each segment is analyzed the moment its download finishes
  5. Total download       → 60s of video (3×20s) instead of 90s full download
  6. Short video safe     → handles Shorts and videos < 20s correctly
  7. Runtime timer        → per-segment + total runtime logged and returned
"""

import os
import time
import warnings
import cv2
import numpy as np
import tempfile
import subprocess
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from PIL import Image
    from io import BytesIO
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False

try:
    import yt_dlp
    YTDLP_AVAILABLE = True
except ImportError:
    YTDLP_AVAILABLE = False
    print("[ERROR] yt-dlp not installed.")

# ── Constants ──────────────────────────────────────────────────────────────────
SEGMENT_DURATION  = 20
FRAME_SAMPLE_RATE = 1
C_MAX             = 4.0
S_MAX             = 128.0
FRAME_WIDTH       = 320
NODE_PATH         = r"C:\Program Files\nodejs\node.exe"


# ── yt-dlp shared options ──────────────────────────────────────────────────────
def _ydl_opts(extra: dict = None) -> dict:
    opts = {
        "quiet":              True,
        "no_warnings":        True,
        "noprogress":         True,
        "geo_bypass":         True,
        "geo_bypass_country": "US",
        "js_runtimes":        {"node": {"path": NODE_PATH}},
        "remote_components":  ["ejs:github"],
        "extractor_args": {
            "youtube": {
                "player_client": ["web", "web_safari", "android_vr", "tv_embedded"]
            }
        },
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        },
    }
    if extra:
        opts.update(extra)
    return opts


# ── Step 0: Metadata only (no download) ───────────────────────────────────────
def fetch_metadata(video_id: str) -> dict:
    """
    Lightweight yt-dlp call — validates availability and gets duration/title.
    No video bytes downloaded. Used to calculate segment windows before
    launching parallel downloads.
    """
    if not YTDLP_AVAILABLE:
        return {"ok": False, "reason": "yt-dlp not installed"}

    urls_to_try = [
        f"https://www.youtube.com/watch?v={video_id}",
        f"https://www.youtube.com/shorts/{video_id}",
    ]

    last_error = None
    for url in urls_to_try:
        try:
            opts = _ydl_opts({"skip_download": True})
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
            return {
                "ok":       True,
                "title":    info.get("title",    "Unknown"),
                "duration": info.get("duration", 0),
                "uploader": info.get("uploader", "Unknown"),
                "url":      url,
            }
        except Exception as e:
            last_error = e
            continue

    msg = str(last_error).lower()
    reason = (
        "Video is not available in this region or has been removed" if "not available" in msg
        else "Video is private"        if "private" in msg
        else "Video is age-restricted" if "age"     in msg
        else str(last_error)
    )
    return {"ok": False, "reason": reason}


# ── Step 1: Download only the needed time window for one segment ───────────────
def fetch_segment_only(
    video_id: str, url: str, start: int, seg_dur: int, seg_id: str
) -> dict:
    """
    Downloads ONLY the 20s window needed for this segment.
    Each of the 3 segments runs this in parallel — total download = 60s not 90s.
    Analysis starts immediately after this segment's download finishes,
    not after all segments finish downloading.
    """
    output_path = tempfile.mktemp(suffix=f"_{seg_id}.mp4")
    t = time.time()
    try:
        opts = _ydl_opts({
            "format":            "worst[ext=mp4]/worst",
            "outtmpl":           output_path,
            "download_sections": [f"*{start}-{start + seg_dur}"],
            "postprocessors":    [],
        })
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.extract_info(url, download=True)

        if not os.path.exists(output_path):
            return {"ok": False, "seg_id": seg_id, "reason": "File missing after download"}

        print(f"[SAMPLER] {seg_id} downloaded in {time.time()-t:.1f}s")
        return {
            "ok":      True,
            "path":    output_path,
            "seg_id":  seg_id,
            "start":   start,
            "seg_dur": seg_dur,
        }
    except Exception as e:
        if os.path.exists(output_path):
            try: os.remove(output_path)
            except Exception: pass
        return {"ok": False, "seg_id": seg_id, "reason": str(e)}


# ── Frame extraction (resized for speed) ──────────────────────────────────────
def extract_frames(video_path: str, start_sec: int, duration: int) -> list:
    """
    1fps frames resized to 320px wide.
    When called after fetch_segment_only(), start_sec=0 since the
    downloaded file already starts at the segment window.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame    = int(min((start_sec + duration) * fps, total_frames))
    start_frame  = int(start_sec * fps)
    step         = max(1, int(fps * FRAME_SAMPLE_RATE))

    frames = []
    idx = start_frame
    while idx < end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        frame = cv2.resize(
            frame, (FRAME_WIDTH, int(h * FRAME_WIDTH / w)),
            interpolation=cv2.INTER_LINEAR
        )
        frames.append(frame)
        idx += step

    cap.release()
    return frames


# ── FCR ───────────────────────────────────────────────────────────────────────
def compute_fcr(frames: list) -> float:
    if len(frames) < 2:
        return 0.0
    cuts = 0
    prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for f in frames[1:]:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if np.mean(cv2.absdiff(prev, gray)) > 25:
            cuts += 1
        prev = gray
    return round(min(1.0, (cuts / max(len(frames), 1)) / C_MAX), 4)


# ── CSV ───────────────────────────────────────────────────────────────────────
def compute_csv(frames: list) -> float:
    if not frames:
        return 0.0
    sats = [np.mean(cv2.cvtColor(f, cv2.COLOR_BGR2HSV)[:, :, 1]) for f in frames]
    return round(min(1.0, float(np.std(sats)) / S_MAX), 4)


# ── ATT ───────────────────────────────────────────────────────────────────────
def compute_att(video_path: str, start_sec: int, duration: int) -> float:
    """
    Fast path: librosa reads directly from the segment mp4 file.
    start_sec=0 when called after fetch_segment_only() since the file
    already starts at the segment window.
    Fallback: ffmpeg WAV extraction if direct read fails.
    """
    if LIBROSA_AVAILABLE:
        try:
            y, sr = librosa.load(
                video_path,
                offset=float(start_sec),
                duration=float(duration),
                sr=22050,
                mono=True,
            )
            if len(y) > 100:
                return round(min(1.0, float(np.mean(
                    librosa.onset.onset_strength(y=y, sr=sr)
                )) / 10.0), 4)
        except Exception:
            pass

    # Fallback: ffmpeg WAV
    wav_path = tempfile.mktemp(suffix=".wav")
    try:
        r = subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(start_sec), "-t", str(duration),
                "-i", video_path,
                "-vn", "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "1",
                wav_path,
            ],
            capture_output=True, timeout=30,
        )
        if r.returncode == 0 and os.path.exists(wav_path) and os.path.getsize(wav_path) > 500:
            if LIBROSA_AVAILABLE:
                y, sr = librosa.load(wav_path, sr=None, mono=True)
                if len(y) > 100:
                    return round(min(1.0, float(np.mean(
                        librosa.onset.onset_strength(y=y, sr=sr)
                    )) / 10.0), 4)
            import wave
            with wave.open(wav_path, "rb") as wf:
                samples = np.frombuffer(
                    wf.readframes(wf.getnframes()), dtype=np.int16
                ).astype(np.float32) / 32768.0
            if len(samples) > 100:
                chunk = 2205
                rms = [
                    float(np.sqrt(np.mean(samples[i:i+chunk]**2)))
                    for i in range(0, len(samples) - chunk, chunk)
                ]
                if rms:
                    return round(min(1.0, float(np.std(rms)) * 10.0), 4)
    except Exception as e:
        print(f"[ATT] Fallback error: {e}")
    finally:
        if os.path.exists(wav_path):
            try: os.remove(wav_path)
            except Exception: pass
    return 0.0


# ── Thumbnail ─────────────────────────────────────────────────────────────────
def compute_thumbnail_intensity(url: str) -> float:
    if not url:
        return 0.0
    try:
        raw = requests.get(url, timeout=6).content
        img = (
            cv2.cvtColor(
                np.array(Image.open(BytesIO(raw)).convert("RGB")),
                cv2.COLOR_RGB2BGR
            )
            if PILLOW_AVAILABLE
            else cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        )
        if img is None:
            return 0.0
        mean_sat = float(np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1])) / 255.0
        gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edge_den = float(np.sum(cv2.Canny(gray, 100, 200) > 0)) / float(gray.size)
        return round(min(1.0, 0.7 * mean_sat + 0.3 * edge_den), 4)
    except Exception as e:
        print(f"[THUMB] {e}")
        return 0.0


# ── Analyze one downloaded segment file ───────────────────────────────────────
def _analyze_segment(seg_path: str, seg_id: str, seg_dur: int) -> dict:
    """
    Called immediately after fetch_segment_only() finishes for this segment.
    offset=0 because the downloaded file already starts at the segment window.
    """
    t      = time.time()
    frames = extract_frames(seg_path, 0, seg_dur)
    fcr    = compute_fcr(frames)
    csv    = compute_csv(frames)
    att    = compute_att(seg_path, 0, seg_dur)
    score_h = round(0.35 * fcr + 0.25 * csv + 0.20 * att, 4)
    print(f"[SAMPLER] {seg_id} FCR={fcr} | CSV={csv} | ATT={att} | H={score_h} ({time.time()-t:.1f}s)")
    return {
        "segment_id":     seg_id,
        "length_seconds": seg_dur,
        "fcr": fcr, "csv": csv, "att": att, "score_h": score_h,
    }


# ── Download + analyze one segment (chained, runs concurrently) ───────────────
def _fetch_and_analyze(
    video_id: str, url: str, seg_id: str, start: int, seg_dur: int
) -> tuple[int, dict]:
    """
    Downloads the segment window then immediately analyzes it.
    Returns (index, result) so the caller can place it correctly.
    Timeline per worker:
      [=== download 20s ===][=== analyze ===]
    All 3 workers run simultaneously:
      Worker 1: [=== dl S1 ===][analyze S1]
      Worker 2: [=== dl S2 ===][analyze S2]
      Worker 3: [=== dl S3 ===][analyze S3]
    """
    idx_map = {"S1": 0, "S2": 1, "S3": 2}

    dl = fetch_segment_only(video_id, url, start, seg_dur, seg_id)

    if not dl["ok"]:
        print(f"[SAMPLER] {seg_id} download failed: {dl.get('reason', '?')}")
        return idx_map[seg_id], {
            "segment_id":     seg_id,
            "offset_seconds": start,
            "length_seconds": seg_dur,
            "fcr": 0.0, "csv": 0.0, "att": 0.0, "score_h": 0.0,
        }

    try:
        result = _analyze_segment(dl["path"], seg_id, seg_dur)
        result["offset_seconds"] = start
        return idx_map[seg_id], result
    finally:
        if os.path.exists(dl["path"]):
            try: os.remove(dl["path"])
            except Exception: pass


# ── Main ──────────────────────────────────────────────────────────────────────
def sample_video(video_id: str, thumbnail_url: str = "") -> dict:
    t_start = time.time()

    try:
        print(f"\n[SAMPLER] ══════════════════════════════════════")
        print(f"[SAMPLER] Analyzing: {video_id}")

        # Step 1: Metadata only (fast — no download)
        t0   = time.time()
        meta = fetch_metadata(video_id)
        print(f"[SAMPLER] Metadata: {time.time()-t0:.1f}s")

        if not meta["ok"]:
            print(f"[SAMPLER] ✗ {meta['reason']}")
            return {
                "video_id": video_id,
                "status":   "unavailable",
                "reason":   meta["reason"],
                "message":  f"Video cannot be analyzed: {meta['reason']}",
            }

        video_duration = meta["duration"] or 0
        print(f"[SAMPLER] ✓ '{meta['title']}' ({video_duration}s)")

        # Step 2: Calculate segment windows
        actual_dur = min(video_duration, 90)

        if actual_dur <= SEGMENT_DURATION:
            effective_seg_dur = max(1, int(actual_dur))
            seg_starts        = [("S1", 0), ("S2", 0), ("S3", 0)]
        else:
            effective_seg_dur = SEGMENT_DURATION
            mid = max(0, int(actual_dur / 2) - effective_seg_dur // 2)
            end = max(0, int(actual_dur) - effective_seg_dur)
            seen = []
            for v in [0, mid, end]:
                if v not in seen:
                    seen.append(v)
            while len(seen) < 3:
                seen.append(seen[-1])
            seg_starts = list(zip(["S1", "S2", "S3"], seen))

        print(f"[SAMPLER] Segments: {seg_starts} | seg_dur={effective_seg_dur}s")

        # Step 3: All 3 segment downloads+analysis + thumbnail run concurrently
        # Each worker: downloads its 20s window → immediately analyzes → done
        # Thumbnail fetched in parallel with all of this
        t0       = time.time()
        segments = [None, None, None]
        thumb    = 0.0

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {}

            # Submit 3 download+analyze workers
            for sid, start in seg_starts:
                f = pool.submit(
                    _fetch_and_analyze,
                    video_id, meta["url"], sid, start, effective_seg_dur
                )
                futures[f] = sid

            # Submit thumbnail concurrently
            thumb_future = pool.submit(compute_thumbnail_intensity, thumbnail_url)
            futures[thumb_future] = "thumb"

            for future in as_completed(futures):
                key = futures[future]
                if key == "thumb":
                    thumb = future.result()
                    print(f"[SAMPLER] Thumbnail: {thumb}")
                else:
                    idx, seg_result = future.result()
                    segments[idx]   = seg_result

        print(f"[SAMPLER] Total download+analysis: {time.time()-t0:.1f}s")

        # Step 4: OIR score
        max_seg   = max(s["score_h"] for s in segments if s)
        agg_score = round(0.80 * max_seg + 0.20 * thumb, 4)
        label     = (
            "Overstimulating" if agg_score >= 0.75
            else "Safe"       if agg_score <= 0.35
            else "Uncertain"
        )

        total = time.time() - t_start
        print(f"[SAMPLER] ✓ Score: {agg_score} → {label}")
        print(f"[SAMPLER] ✓ Total runtime: {total:.1f}s")
        print(f"[SAMPLER] ══════════════════════════════════════\n")

        return {
            "video_id":                  video_id,
            "video_title":               meta.get("title", ""),
            "video_duration_sec":        round(float(video_duration), 1),
            "thumbnail_url":             thumbnail_url,
            "thumbnail_intensity":       thumb,
            "segments":                  segments,
            "aggregate_heuristic_score": agg_score,
            "preliminary_label":         label,
            "status":                    "success",
            "runtime_seconds":           round(total, 2),
        }

    except Exception as e:
        print(f"[SAMPLER] ✗ Fatal: {e}")
        import traceback; traceback.print_exc()
        return {"video_id": video_id, "status": "error", "message": str(e)}