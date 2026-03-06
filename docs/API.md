# ChildFocus Backend API Documentation

Base URL: `http://localhost:5000` (development) / `http://<your-ip>:5000` (device testing)

---

## Endpoints

### `GET /health`
Health check. Used by Android app to verify backend connectivity.

**Response:**
```json
{ "status": "ok", "service": "ChildFocus Backend", "version": "1.0.0" }
```

---

### `GET /metadata?video_url=<url>`
Fetch YouTube video metadata.

**Query Params:**
- `video_url` — Full YouTube URL or 11-char video ID

**Response:**
```json
{
  "video_id":     "dQw4w9WgXcQ",
  "title":        "Video Title",
  "description":  "...",
  "tags":         ["tag1", "tag2"],
  "channel":      "Channel Name",
  "thumbnail_url": "https://i.ytimg.com/vi/.../hqdefault.jpg",
  "view_count":   1000000
}
```

---

### `POST /classify_fast`
Fast classification using Naïve Bayes metadata scoring only. No video download. Returns in ~1–2 seconds.

**Request body:**
```json
{ "video_url": "https://youtube.com/watch?v=dQw4w9WgXcQ" }
```

**Response:**
```json
{
  "video_id":          "dQw4w9WgXcQ",
  "score_nb":           0.72,
  "nb_label":          "Overstimulating",
  "nb_confidence":      0.72,
  "nb_probabilities":  { "Educational": 0.08, "Neutral": 0.20, "Overstimulating": 0.72 },
  "preliminary_label": "Overstimulating",
  "action":            "block",
  "status":            "success",
  "runtime_seconds":   1.2
}
```

**Action values:**
- `block` — NB is confident this is overstimulating (score_nb ≥ 0.75)
- `allow` — NB is confident this is safe (score_nb ≤ 0.35)
- `pending_full_analysis` — Uncertain, run `/classify_full`

---

### `POST /classify_full`
Full hybrid classification. Downloads video, runs heuristic analysis + NB fusion. Takes 20–60 seconds.

**Request body:**
```json
{
  "video_url":     "https://youtube.com/watch?v=dQw4w9WgXcQ",
  "thumbnail_url": "https://i.ytimg.com/vi/dQw4w9WgXcQ/hqdefault.jpg"
}
```

**Response:**
```json
{
  "video_id":    "dQw4w9WgXcQ",
  "video_title": "Video Title",
  "score_nb":    0.72,
  "score_h":     0.68,
  "score_final": 0.696,
  "fusion_weights": { "alpha_nb": 0.4, "beta_heuristic": 0.6 },
  "oir_label":   "Overstimulating",
  "action":      "block",
  "thresholds":  { "block": 0.75, "allow": 0.35 },
  "nb_details": {
    "label":         "Overstimulating",
    "confidence":    0.72,
    "probabilities": { "Educational": 0.08, "Neutral": 0.20, "Overstimulating": 0.72 }
  },
  "heuristic_details": {
    "segments": [
      { "segment_id": "S1", "offset_seconds": 0,  "fcr": 0.62, "csv": 0.55, "att": 0.70, "score_h": 0.63 },
      { "segment_id": "S2", "offset_seconds": 35, "fcr": 0.70, "csv": 0.60, "att": 0.75, "score_h": 0.68 },
      { "segment_id": "S3", "offset_seconds": 70, "fcr": 0.65, "csv": 0.50, "att": 0.65, "score_h": 0.60 }
    ],
    "thumbnail":      0.75,
    "video_duration": 90.0,
    "runtime":        42.5
  },
  "status":          "success",
  "runtime_seconds": 45.2
}
```

---

## OIR Labels

| Score_final | Label           | Action |
|-------------|-----------------|--------|
| ≥ 0.75      | Overstimulating | block  |
| 0.36–0.74   | Neutral         | allow  |
| ≤ 0.35      | Educational     | allow  |

---

## Fusion Formula

```
Score_final = (0.4 × Score_NB) + (0.6 × Score_H)
```

- **Score_NB** — Naïve Bayes probability of "Overstimulating" from metadata
- **Score_H** — Heuristic score from FCR (0.35) + CSV (0.25) + ATT (0.20) + Thumb (0.20)

---

## Running the Backend

```bash
cd backend
pip install -r requirements.txt
python run.py
```

Make sure `YOUTUBE_API_KEY` is set in `backend/.env`:
```
YOUTUBE_API_KEY=your_key_here
```
