"""
ChildFocus - Dataset Enrichment Script
ml_training/scripts/enrich_dataset.py

Backfills hidden YouTube keywords (ytInitialData 'keywords' array) for all
700 rows in final_700_dataset.csv. These tags are invisible to viewers but
embedded in raw page HTML — the most complete tag source available.

Problem this solves:
  - 346/700 rows (49.4%) have EMPTY tags in final_700_dataset.csv
  - 354 rows that "have" tags average only 1.1 tags (stored as raw string)
  - YouTube Data API v3 returns empty tags when creator marks them private
  - NB model was effectively trained on title + description only for ~50% of data

What this script does:
  1. Reads  data/raw/final_700_dataset.csv          (700 rows, original)
  2. Scrapes ytInitialData 'keywords' array for each video_id
  3. Merges scraped keywords with any existing tags (deduplicates)
  4. Saves data/raw/final_700_enriched.csv           (700 rows, enriched)
  5. Checkpoints progress every 10 rows → safe to Ctrl+C and resume

Run order after this script:
  1. python enrich_dataset.py     →  data/raw/final_700_enriched.csv
  2. python preprocess.py         →  data/processed/metadata_clean.csv
  3. python train_nb.py           →  outputs/nb_model.pkl + vectorizer.pkl

Estimated runtime: 15–35 minutes (700 HTTP requests, ~1.5s delay each)
Rate limit: 1.5s between requests — stays well within YouTube's tolerance.

Resume: if interrupted, re-run the same command. Progress is checkpointed
to enrich_checkpoint.json and the script will skip already-processed rows.
"""

import os
import re
import csv
import json
import time
import random
import requests

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW_PATH        = "data/raw/final_700_dataset.csv"
ENRICHED_PATH   = "data/raw/final_700_enriched.csv"
CHECKPOINT_PATH = "data/raw/enrich_checkpoint.json"

# ── HTTP settings ──────────────────────────────────────────────────────────────
REQUEST_DELAY   = 1.5    # seconds between requests — safe for YouTube
REQUEST_TIMEOUT = 12     # seconds per request
MAX_RETRIES     = 2      # retries per video on network failure

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ── CSV columns (must match final_700_dataset.csv exactly) ────────────────────
FIELDNAMES = ["video_id", "title", "description", "tags", "channel",
              "query_used", "label", "trust_level"]


# ══════════════════════════════════════════════════════════════════════════════
# Core scraper
# ══════════════════════════════════════════════════════════════════════════════

def scrape_keywords(video_id: str) -> list:
    """
    Fetches the raw YouTube page and extracts the hidden 'keywords' array
    from the ytInitialData JSON blob injected into every watch page.

    Tries /watch?v= first, then /shorts/ as fallback (for Shorts videos).
    Returns a list of keyword strings, or [] on failure.
    """
    urls = [
        f"https://www.youtube.com/watch?v={video_id}",
        f"https://www.youtube.com/shorts/{video_id}",
    ]
    for url in urls:
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = requests.get(url, headers=HEADERS,
                                    timeout=REQUEST_TIMEOUT)
                if resp.status_code == 429:
                    wait = 30 + random.uniform(5, 15)
                    print(f"  [!] Rate limited — waiting {wait:.0f}s...")
                    time.sleep(wait)
                    continue
                if resp.status_code != 200:
                    break  # non-retryable error (404, 403, etc.)

                match = re.search(r'"keywords"\s*:\s*(\[[^\]]*\])', resp.text)
                if match:
                    keywords = json.loads(match.group(1))
                    return [str(k).strip() for k in keywords if str(k).strip()]
                # Page loaded but no keywords array — try Shorts URL
                break

            except requests.exceptions.Timeout:
                if attempt < MAX_RETRIES:
                    time.sleep(3)
                    continue
                break
            except Exception:
                break

    return []


def merge_tags(existing_raw: str, scraped: list) -> str:
    """
    Merges existing tag string with scraped keywords list.
    Deduplicates case-insensitively, preserves original casing.
    Returns a comma-separated string for CSV storage.
    """
    seen   = set()
    merged = []

    # Parse existing tags (comma, pipe, or space-separated)
    existing = []
    if existing_raw and existing_raw.strip():
        s = existing_raw.strip()
        if "," in s:
            existing = [t.strip() for t in s.split(",") if t.strip()]
        elif "|" in s:
            existing = [t.strip() for t in s.split("|") if t.strip()]
        else:
            existing = [s]

    for tag in existing + scraped:
        key = tag.lower()
        if key and key not in seen:
            seen.add(key)
            merged.append(tag)

    return ", ".join(merged)


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_checkpoint() -> dict:
    """Returns {video_id: merged_tags_string} for already-processed rows."""
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_checkpoint(done: dict):
    """Persists progress so the script can resume after interruption."""
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(done, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def enrich():
    if not os.path.exists(RAW_PATH):
        print(f"[ENRICH] ✗ Source file not found: {RAW_PATH}")
        print(f"[ENRICH]   Expected: {os.path.abspath(RAW_PATH)}")
        return

    os.makedirs("data/raw", exist_ok=True)

    # ── Load source rows ───────────────────────────────────────────────────────
    with open(RAW_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"[ENRICH] Loaded {len(rows)} rows from {RAW_PATH}")

    # ── Load checkpoint (resume support) ──────────────────────────────────────
    done = load_checkpoint()
    if done:
        print(f"[ENRICH] Resuming — {len(done)} rows already processed, "
              f"{len(rows) - len(done)} remaining")

    # ── Scraping loop ──────────────────────────────────────────────────────────
    stats = {"scraped": 0, "empty": 0, "already_had_tags": 0,
             "new_tags_added": 0, "failed": 0}

    t_start = time.time()

    for i, row in enumerate(rows, 1):
        video_id = row.get("video_id", "").strip()
        if not video_id:
            continue

        # Skip already-processed rows
        if video_id in done:
            existing_had = bool(row.get("tags", "").strip())
            if existing_had:
                stats["already_had_tags"] += 1
            continue

        existing_tags = row.get("tags", "").strip()
        had_tags      = bool(existing_tags)

        # Progress display
        elapsed  = time.time() - t_start
        per_item = elapsed / max(i - len(done) - 1, 1)
        remaining = (len(rows) - i) * per_item
        print(f"[{i:>3}/{len(rows)}] {video_id}  "
              f"{'(had tags)' if had_tags else '(no tags) '}  "
              f"ETA: {remaining/60:.1f}min", end="  ")

        scraped = scrape_keywords(video_id)

        if scraped:
            merged = merge_tags(existing_tags, scraped)
            added  = len(scraped) if not had_tags else max(0, len(scraped) - 1)
            print(f"✓ {len(scraped)} keywords scraped")
            stats["scraped"] += 1
            if not had_tags:
                stats["new_tags_added"] += 1
        else:
            merged = existing_tags  # keep original (even if empty)
            print(f"✗ none found")
            stats["failed"] += 1
            stats["empty"] += 1

        if had_tags and scraped:
            stats["already_had_tags"] += 1

        done[video_id] = merged

        # Checkpoint every 10 rows
        if i % 10 == 0:
            save_checkpoint(done)
            print(f"[ENRICH] Checkpoint saved ({i}/{len(rows)})")

        # Rate limit
        time.sleep(REQUEST_DELAY + random.uniform(0, 0.5))

    # Final checkpoint save
    save_checkpoint(done)

    # ── Write enriched CSV ─────────────────────────────────────────────────────
    enriched_rows = []
    for row in rows:
        video_id = row.get("video_id", "").strip()
        new_row  = dict(row)
        if video_id in done:
            new_row["tags"] = done[video_id]
        enriched_rows.append(new_row)

    with open(ENRICHED_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(enriched_rows)

    # ── Summary ────────────────────────────────────────────────────────────────
    total_time = time.time() - t_start
    had_no_tags_before = sum(1 for r in rows if not r.get("tags", "").strip())
    now_have_tags = sum(
        1 for r in enriched_rows
        if r.get("tags", "").strip()
    )

    print(f"\n[ENRICH] ══════════════════════════════════════")
    print(f"[ENRICH] Enrichment complete in {total_time/60:.1f} min")
    print(f"[ENRICH] ══════════════════════════════════════")
    print(f"[ENRICH] Rows originally without tags : {had_no_tags_before}")
    print(f"[ENRICH] Rows with tags after enrich  : {now_have_tags}")
    print(f"[ENRICH] Keywords successfully scraped: {stats['scraped']}")
    print(f"[ENRICH] Previously empty → now filled: {stats['new_tags_added']}")
    print(f"[ENRICH] Scrape failures (kept orig.)  : {stats['failed']}")
    print(f"[ENRICH] ✓ Saved → {ENRICHED_PATH}")
    print(f"\n[ENRICH] Next steps:")
    print(f"[ENRICH]   1. python preprocess.py   (reads final_700_enriched.csv)")
    print(f"[ENRICH]   2. python train_nb.py      (retrains NB on enriched text)")


if __name__ == "__main__":
    enrich()
