"""
Sprint 1 Data Collection:
Collects metadata from 1,000 child-oriented YouTube video IDs.
Run this script to build your raw dataset.
"""
import csv
import time
import requests
import os
from dotenv import load_dotenv

load_dotenv("../../backend/.env")
API_KEY = os.getenv("YOUTUBE_API_KEY")

# Sample child-directed search queries
SEARCH_QUERIES = [
    "kids educational videos",
    "children cartoon episodes",
    "nursery rhymes for toddlers",
    "kids science experiments",
    "children's music videos",
    "animated stories for kids",
    "kids yoga and exercise",
    "baby sensory videos",
    "kids fast cartoon compilation",   # likely overstimulating
    "surprise eggs unboxing kids",     # likely overstimulating
]

def search_youtube(query, max_results=50):
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "videoCategoryId": "27",  # Education category
        "maxResults": max_results,
        "key": API_KEY
    }
    resp = requests.get(url, params=params)
    return resp.json().get("items", [])

def collect_dataset(output_path="data/raw/metadata_raw.csv"):
    os.makedirs("data/raw", exist_ok=True)
    collected = []

    for query in SEARCH_QUERIES:
        print(f"Searching: {query}")
        items = search_youtube(query, max_results=100)
        for item in items:
            video_id = item["id"].get("videoId", "")
            snippet = item["snippet"]
            collected.append({
                "video_id": video_id,
                "title": snippet.get("title", ""),
                "description": snippet.get("description", ""),
                "channel": snippet.get("channelTitle", ""),
                "query_used": query,
                "label": ""   # to be labeled in Sprint 2
            })
        time.sleep(1)  # respect API rate limits

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=collected[0].keys())
        writer.writeheader()
        writer.writerows(collected)

    print(f"Collected {len(collected)} videos → {output_path}")

if __name__ == "__main__":
    collect_dataset()