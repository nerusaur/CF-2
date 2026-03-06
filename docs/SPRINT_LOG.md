# ChildFocus — Sprint Log

---

## Sprint 1 — Frame Sampling Module ✅ COMPLETE

**Duration:** Jan 16 – Feb 15  
**Responsible:** Neil (Feature 1), Allen (Feature 2), Dennis (Feature 3)

### Deliverables
- `backend/app/modules/frame_sampler.py` — fully implemented and optimized
  - `fetch_video()` — single yt-dlp call (validate + download combined)
  - `extract_frames()` — 1fps, resized to 320px for speed
  - `compute_fcr()` — Frame-Change Rate
  - `compute_csv()` — Color Saturation Variance
  - `compute_att()` — Audio Tempo (librosa direct read + ffmpeg fallback)
  - `compute_thumbnail_intensity()` — saturation + edge density
  - `sample_video()` — main entry point with ThreadPoolExecutor concurrency
  - Short video / YouTube Shorts handling (deduplication)
  - 6 previous versions showing iterative improvement
- `backend/app/modules/youtube_api.py` — fully implemented
  - `get_video_metadata()` — title, tags, description, duration, stats
  - `get_thumbnail_url()` — fast direct URL + API fallback
  - `scrape_thumbnail_batch()` — batch 50 videos per API call
  - `search_child_videos()` — safeSearch strict
- `ml_training/scripts/collect_metadata.py` — 10 search queries
- `ml_training/scripts/data/raw/metadata_raw.csv` — 500 rows collected
- `backend/app/routes/classify.py` — `/classify_fast` + `/classify_full` endpoints
- Android foundation: `LandingScreen.kt`, `AppDatabase.kt`, all Room entities/DAOs
- `database/schema.sql` — 4 tables: videos, segments, users, logs

### Notes
- yt-dlp integration required 6 iterations due to YouTube bot detection changes
- librosa direct MP4 read saves ~3-6s vs ffmpeg subprocess per segment
- ThreadPoolExecutor cuts segment analysis from ~30s to ~10s

---

## Sprint 2 — Heuristic + Naïve Bayes Classifier 🔄 IN PROGRESS

**Duration:** Jan 22 – Feb 25  
**Responsible:** Neil (Features 1-2), Dennis (Feature 3)

### Deliverables
- `ml_training/scripts/preprocess.py` — auto-labeling + TF-IDF vectorization
- `ml_training/scripts/train_nb.py` — ComplementNB training, 70/30 split, metrics
- `backend/app/modules/naive_bayes.py` — `score_metadata()` → Score_NB [0,1]
- `backend/app/modules/heuristic.py` — clean interface wrapping frame_sampler
- `backend/app/modules/hybrid_fusion.py` — `classify_fast()` + `classify_full()`
- `backend/app/routes/classify.py` — updated with full hybrid logic
- `backend/app/routes/metadata.py` — `/metadata`, `/health`, `/config` endpoints
- `backend/app/config.py` — Flask configuration
- `backend/app/__init__.py` — updated with metadata blueprint
- `backend/app/utils/logger.py` — classification event logging
- `backend/app/utils/validators.py` — YouTube URL validation
- `backend/tests/test_heuristic.py` — pytest suite for heuristic module
- `backend/tests/test_naive_bayes.py` — pytest suite for NB module
- `backend/tests/test_hybrid.py` — pytest suite for fusion module
- Android: `ClassificationResult.kt`, `Video.kt`, `Segment.kt` — data models
- Android: `ChildFocusApi.kt`, `YouTubeApi.kt` — updated with proper types
- Android: `VideoRepository.kt` — caching + API calls
- Android: `SafetyViewModel.kt` — UI state management
- Android: `SafetyModeScreen.kt` — video URL input + result display
- Android: `ResultScreen.kt` — detailed classification report
- `docs/API.md` — full endpoint documentation

### TODO before Sprint 2 close
- [ ] Label `metadata_raw.csv` (500 rows, 0 labeled) — run `preprocess.py` (auto-labels by query)
- [ ] Run `train_nb.py` and verify F1 ≥ 0.70
- [ ] Replace placeholder `nb_model.pkl` and `vectorizer.pkl` with trained versions
- [ ] Run pytest suite: `cd backend && python -m pytest tests/ -v`
- [ ] Test `/classify_fast` and `/classify_full` endpoints with Postman or curl

---

## Sprint 3 — Hybrid Filtering Logic ⏳ UPCOMING

**Duration:** Feb 28 – Mar 16

### Planned
- Fine-tune fusion weights (α, β) based on Sprint 2 evaluation results
- Implement conservative aggregation (max of S1, S2, S3 scores)
- Add caching layer (skip re-analysis of already-seen video IDs)
- Threshold calibration from labeled dataset

---

## Sprint 4 — UI + Backend Integration ⏳ UPCOMING

**Duration:** Mar 16 – Apr 16

### Planned
- Connect Android app to live Flask backend
- Implement screen-time control
- Implement website blocking
- Content restriction UI
- Parental alert notifications

---

## Sprint 5 — Accuracy Testing & Optimization ⏳ UPCOMING

**Duration:** Mar 24 – Apr 21

### Planned
- Evaluate full hybrid model: Precision, Recall, F1, Confusion Matrix
- UAT with parents and educators
- Adjust thresholds based on UAT feedback
- Performance testing on real Android device
