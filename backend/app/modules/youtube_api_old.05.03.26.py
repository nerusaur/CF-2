import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

def get_video_metadata(video_id: str) -> dict:
    """
    Fetches title, tags, description, and duration from YouTube Data API v3.
    Returns a dict or raises an exception on failure.
    """
    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        "part": "snippet,contentDetails,statistics",
        "id": video_id,
        "key": API_KEY
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    if not data.get("items"):
        return {"error": "Video not found"}

    item = data["items"][0]
    snippet = item["snippet"]

    return {
        "video_id": video_id,
        "title": snippet.get("title", ""),
        "description": snippet.get("description", ""),
        "tags": snippet.get("tags", []),
        "channel": snippet.get("channelTitle", ""),
        "duration": item["contentDetails"].get("duration", ""),
        "view_count": item["statistics"].get("viewCount", 0),
        "thumbnail_url": snippet["thumbnails"]["high"]["url"]
    }


def extract_video_id(url: str) -> str:
    """Extracts video ID from a YouTube URL."""
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return url  # assume it's already an ID