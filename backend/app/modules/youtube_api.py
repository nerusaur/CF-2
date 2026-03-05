"""
ChildFocus - YouTube API Module
backend/app/modules/youtube_api.py

Features:
  - get_video_metadata()       → fetch title, tags, description, duration
  - get_thumbnail_url()        → best available thumbnail URL
  - scrape_thumbnail_batch()   → collect thumbnails for dataset building
  - extract_video_id()         → parse video ID from any YouTube URL
  - search_child_videos()      → search YouTube for child-directed content
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

# YouTube thumbnail quality levels (highest to lowest)
THUMBNAIL_QUALITY = ["maxres", "standard", "high", "medium", "default"]


# ── Utility ────────────────────────────────────────────────────────────────────
def extract_video_id(url_or_id: str) -> str:
    """Extracts video ID from any YouTube URL format, or returns as-is if already an ID."""
    if "v=" in url_or_id:
        return url_or_id.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url_or_id:
        return url_or_id.split("youtu.be/")[1].split("?")[0]
    elif "shorts/" in url_or_id:
        return url_or_id.split("shorts/")[1].split("?")[0]
    return url_or_id.strip()


def get_best_thumbnail_url(thumbnails: dict) -> str:
    """Returns the highest quality thumbnail URL from a thumbnails dict."""
    for quality in THUMBNAIL_QUALITY:
        if quality in thumbnails:
            return thumbnails[quality]["url"]
    return ""


# ── Core API Functions ─────────────────────────────────────────────────────────
def get_video_metadata(video_id: str) -> dict:
    """
    Fetches full metadata for a single YouTube video.
    Returns title, description, tags, duration, thumbnails, stats.
    """
    if not API_KEY:
        return {"error": "YOUTUBE_API_KEY not set in .env"}

    url    = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        "part": "snippet,contentDetails,statistics",
        "id":   video_id,
        "key":  API_KEY
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {e}"}

    if not data.get("items"):
        return {"error": f"Video not found: {video_id}"}

    item    = data["items"][0]
    snippet = item["snippet"]
    stats   = item.get("statistics", {})

    thumbnail_url = get_best_thumbnail_url(snippet.get("thumbnails", {}))

    return {
        "video_id":       video_id,
        "title":          snippet.get("title", ""),
        "description":    snippet.get("description", "")[:500],  # truncate for storage
        "tags":           snippet.get("tags", []),
        "channel":        snippet.get("channelTitle", ""),
        "published_at":   snippet.get("publishedAt", ""),
        "duration":       item["contentDetails"].get("duration", ""),
        "category_id":    snippet.get("categoryId", ""),
        "default_lang":   snippet.get("defaultLanguage", ""),
        "thumbnail_url":  thumbnail_url,
        "view_count":     int(stats.get("viewCount", 0)),
        "like_count":     int(stats.get("likeCount", 0)),
        "comment_count":  int(stats.get("commentCount", 0)),
    }


def get_thumbnail_url(video_id: str) -> str:
    """
    Returns the best available thumbnail URL for a video.
    Tries YouTube Data API first, falls back to direct URL construction.
    """
    # Fast path: construct URL directly (no API quota used)
    # YouTube thumbnail URL pattern is predictable
    direct_urls = [
        f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg",
        f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
        f"https://i.ytimg.com/vi/{video_id}/mqdefault.jpg",
        f"https://i.ytimg.com/vi/{video_id}/default.jpg",
    ]

    for thumb_url in direct_urls:
        try:
            resp = requests.head(thumb_url, timeout=5)
            # maxresdefault returns 404 for some videos — skip those
            if resp.status_code == 200 and int(resp.headers.get("content-length", 0)) > 5000:
                return thumb_url
        except Exception:
            continue

    # API fallback (uses quota)
    if API_KEY:
        meta = get_video_metadata(video_id)
        return meta.get("thumbnail_url", "")

    return f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"  # last resort


# ── Batch Thumbnail Scraper ────────────────────────────────────────────────────
def scrape_thumbnail_batch(video_ids: list) -> list:
    """
    Thumbnail scraper for dataset building.
    Given a list of video IDs, returns a list of dicts with:
      - video_id, thumbnail_url, title, tags (for NB classifier)

    Used in Sprint 1 data collection to enrich the metadata CSV.
    """
    if not API_KEY:
        print("[SCRAPER] No API key — using direct thumbnail URL construction.")
        return [
            {
                "video_id":      vid,
                "thumbnail_url": get_thumbnail_url(vid),
                "title":         "",
                "tags":          [],
                "source":        "direct_url"
            }
            for vid in video_ids
        ]

    results      = []
    # YouTube API allows up to 50 IDs per request
    batch_size   = 50
    total        = len(video_ids)

    for batch_start in range(0, total, batch_size):
        batch = video_ids[batch_start:batch_start + batch_size]
        print(f"[SCRAPER] Fetching batch {batch_start // batch_size + 1} "
              f"({len(batch)} videos)...")

        url    = "https://www.googleapis.com/youtube/v3/videos"
        params = {
            "part": "snippet",
            "id":   ",".join(batch),
            "key":  API_KEY
        }

        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            print(f"[SCRAPER] Batch failed: {e}")
            # Add fallback entries for failed batch
            for vid in batch:
                results.append({
                    "video_id":      vid,
                    "thumbnail_url": get_thumbnail_url(vid),
                    "title":         "",
                    "tags":          [],
                    "source":        "direct_url_fallback"
                })
            continue

        found_ids = set()
        for item in data.get("items", []):
            vid_id    = item["id"]
            snippet   = item["snippet"]
            thumb_url = get_best_thumbnail_url(snippet.get("thumbnails", {}))
            found_ids.add(vid_id)

            results.append({
                "video_id":      vid_id,
                "thumbnail_url": thumb_url,
                "title":         snippet.get("title", ""),
                "tags":          snippet.get("tags", []),
                "channel":       snippet.get("channelTitle", ""),
                "source":        "youtube_api"
            })

        # Handle IDs not returned by API (private/deleted)
        for vid in batch:
            if vid not in found_ids:
                results.append({
                    "video_id":      vid,
                    "thumbnail_url": "",
                    "title":         "",
                    "tags":          [],
                    "source":        "not_found"
                })

    print(f"[SCRAPER] Done. {len(results)}/{total} videos processed.")
    return results


# ── Search for Child-Directed Videos ──────────────────────────────────────────
def search_child_videos(query: str, max_results: int = 50) -> list:
    """
    Searches YouTube for child-directed videos matching the query.
    Returns list of video_ids for dataset collection.
    """
    if not API_KEY:
        return []

    url    = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part":       "snippet",
        "q":          query,
        "type":       "video",
        "maxResults": min(max_results, 50),
        "safeSearch": "strict",       # filter adult content
        "relevanceLanguage": "en",
        "key":        API_KEY
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        items = resp.json().get("items", [])
        return [item["id"]["videoId"] for item in items if "videoId" in item.get("id", {})]
    except Exception as e:
        print(f"[SEARCH] Error: {e}")
        return []
