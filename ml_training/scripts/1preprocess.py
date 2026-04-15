"""
ChildFocus - Data Preprocessing Script
ml_training/scripts/preprocess.py

Steps:
  1. Read data/raw/final_700_dataset.csv  ← output of merge_datasets.py (700 rows)
  2. Clean and combine text fields using build_nb_text() — title×3 + tags + description[:300]
  3. Remove stop words, URLs, special characters  (identical to text_builder.py in backend)
  4. Save labeled + cleaned CSV → data/processed/metadata_clean.csv  (700 rows)

IMPORTANT — Run order:
  1. python merge_datasets.py          →  data/raw/final_700_dataset.csv  (700 rows)
  2. python enrich_dataset.py          →  data/raw/final_700_enriched.csv (700 rows + scraped keywords)
  3. python preprocess.py              →  data/processed/metadata_clean.csv
  4. python train_nb.py                →  outputs/nb_model.pkl + vectorizer.pkl  (490 train / 210 test)

WHY final_700_dataset.csv (not merged_dataset.csv):
  merged_dataset.csv has 1,702 rows — training on all of them would produce a
  train/test split of ~1191/511, not the 490/210 stated in the thesis manuscript.
  final_700_dataset.csv is the curated 700-row set (230 Educational / 235 Neutral /
  235 Overstimulating) produced by merge_datasets.py, which is the exact corpus
  described in the thesis.

TEXT FORMULA — must stay identical to backend/app/modules/text_builder.py:
  build_nb_text(title, tags, description)
    = (title × 3)  +  tags  +  description[:300]
  → lowercased, URLs removed, non-alpha stripped, stop words removed

Run:
    python preprocess.py
"""

import os
import re
import csv

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW_PATH       = "data/raw/final_700_enriched.csv"  # ← output of enrich_dataset.py
#               = "data/raw/final_700_dataset.csv"  # ← fallback: original (no scraped keywords)
PROCESSED_DIR  = "data/processed"
PROCESSED_PATH = "data/processed/metadata_clean.csv"

# ── Stop words (MUST match backend/app/modules/text_builder.py exactly) ───────
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "this", "that", "was", "are",
    "be", "as", "so", "we", "he", "she", "they", "you", "i", "my", "your",
    "his", "her", "its", "our", "their", "what", "which", "who", "will",
    "would", "could", "should", "has", "have", "had", "do", "does", "did",
    "not", "no", "if", "then", "than", "when", "where", "how", "all",
    "each", "more", "also", "just", "can", "up", "out", "about", "into",
    "too", "very", "s", "t", "re", "ve", "ll", "d",
}

# ── Valid labels ───────────────────────────────────────────────────────────────
VALID_LABELS = {"Educational", "Neutral", "Overstimulating"}


def build_nb_text(title: str = "", tags=None, description: str = "") -> str:
    """
    Canonical text representation for NB classification.

    ⚠ THIS FUNCTION MUST STAY BYTE-FOR-BYTE IDENTICAL TO
      backend/app/modules/text_builder.py :: build_nb_text()
    Any change here MUST also be applied there, and the model must be retrained.

    Formula:
      - title repeated 3× (high signal density)
      - tags joined with spaces (medium signal)
      - description truncated to 300 chars (low signal)
    Then: lowercase → strip URLs → strip non-alpha → remove stop words
    """
    title_part = f"{title} " * 3
    tags_str   = " ".join(str(t) for t in tags) if isinstance(tags, list) else (tags or "")
    desc_part  = (description or "")[:300]

    raw = f"{title_part}{tags_str} {desc_part}".lower()

    # Remove URLs
    raw = re.sub(r"https?://\S+|www\.\S+", " ", raw)
    # Remove non-alpha characters (no digits — consistent with text_builder.py)
    raw = re.sub(r"[^a-z\s]", " ", raw)
    # Remove stop words and single-char tokens
    tokens = [t for t in raw.split() if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


def parse_tags(raw_tags) -> list:
    """
    Convert the tags column value to a clean list of tag strings.
    Handles: comma-separated strings, pipe-separated, already-list, or empty.
    """
    if not raw_tags or not str(raw_tags).strip():
        return []
    s = str(raw_tags).strip()
    # Try comma-separated first (most common from merged_dataset.csv)
    if "," in s:
        return [t.strip() for t in s.split(",") if t.strip()]
    # Try pipe-separated
    if "|" in s:
        return [t.strip() for t in s.split("|") if t.strip()]
    # Single tag or space-separated
    return [s] if s else []


def preprocess():
    # ── Guard: check source file exists ───────────────────────────────────────
    if not os.path.exists(RAW_PATH):
        print(f"[PREPROCESS] ✗ Source file not found: {RAW_PATH}")
        print(f"[PREPROCESS]   Run 'python enrich_dataset.py' first to generate it.")
        print(f"[PREPROCESS]   (or switch RAW_PATH back to final_700_dataset.csv to skip enrichment)")
        return

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    with open(RAW_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"[PREPROCESS] Loaded {len(rows)} rows from {RAW_PATH}")

    # ── Warn if row count deviates from thesis 700 ────────────────────────────
    if len(rows) != 700:
        print(f"[PREPROCESS] ⚠  Expected 700 rows (thesis spec), got {len(rows)}.")
        print(f"[PREPROCESS]    This will change the 490/210 train/test split in train_nb.py.")

    processed    = []
    label_counts = {lbl: 0 for lbl in VALID_LABELS}
    skipped      = 0

    for row in rows:
        label = row.get("label", "").strip()

        # Normalise capitalisation (e.g. "educational" → "Educational")
        label = label.title() if label else ""

        if label not in VALID_LABELS:
            skipped += 1
            continue

        tags = parse_tags(row.get("tags", ""))
        text = build_nb_text(
            title       = row.get("title", ""),
            tags        = tags,
            description = row.get("description", ""),
        )

        if not text.strip():
            skipped += 1
            continue

        label_counts[label] += 1
        processed.append({
            "video_id": row.get("video_id", ""),
            "text":     text,
            "label":    label,
            "title":    row.get("title", ""),
        })

    # ── Write processed CSV ────────────────────────────────────────────────────
    with open(PROCESSED_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "text", "label", "title"])
        writer.writeheader()
        writer.writerows(processed)

    total = len(processed)
    print(f"[PREPROCESS] ✓ {total} rows saved → {PROCESSED_PATH}")
    if skipped:
        print(f"[PREPROCESS]   {skipped} rows skipped (invalid label or empty text)")
    print(f"[PREPROCESS] Label distribution:")
    for label, count in label_counts.items():
        pct = count / total * 100 if total else 0
        print(f"[PREPROCESS]   {label}: {count} ({pct:.1f}%)")

    # ── Thesis split preview ───────────────────────────────────────────────────
    train_n = round(total * 0.70)
    test_n  = total - train_n
    print(f"\n[PREPROCESS] Expected split (train_nb.py will confirm):")
    print(f"[PREPROCESS]   Train (70%): {train_n}  |  Test (30%): {test_n}")
    print(f"[PREPROCESS] Next step: python train_nb.py")


if __name__ == "__main__":
    preprocess()
