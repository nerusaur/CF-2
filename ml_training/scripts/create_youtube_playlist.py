"""
YouTube Playlist Creator from CSV
==================================
Creates a YouTube playlist in your channel from handpicked_playlist_validated.csv

SETUP (one-time):
1. Go to https://console.cloud.google.com/
2. Create a project → Enable "YouTube Data API v3"
3. Go to APIs & Services → Credentials → Create OAuth 2.0 Client ID (Desktop App)
4. Download the JSON → save it as "client_secret.json" in this same folder
5. pip install google-auth google-auth-oauthlib google-api-python-client pandas

USAGE:
    python create_youtube_playlist.py
    python create_youtube_playlist.py --filter Educational
    python create_youtube_playlist.py --filter Neutral
    python create_youtube_playlist.py --filter Overstimulating
    python create_youtube_playlist.py --name "My Custom Playlist Name"
    python create_youtube_playlist.py --privacy unlisted   (public / private / unlisted)
"""

import os
import sys
import argparse
import pandas as pd
import time

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pickle

# ── Config ──────────────────────────────────────────────────────────────────

CSV_PATH         = "handpicked_playlist_validated.csv"   # path to your CSV
CLIENT_SECRET    = "client_secret.json"                  # OAuth credentials file
TOKEN_PICKLE     = "token.pkl"                           # cached token (auto-created)
SCOPES           = ["https://www.googleapis.com/auth/youtube"]

# ── Auth ─────────────────────────────────────────────────────────────────────

def get_authenticated_service():
    creds = None

    # Load cached token if it exists
    if os.path.exists(TOKEN_PICKLE):
        with open(TOKEN_PICKLE, "rb") as f:
            creds = pickle.load(f)

    # Refresh or re-authenticate
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CLIENT_SECRET):
                sys.exit(
                    f"❌  '{CLIENT_SECRET}' not found.\n"
                    "    Download it from Google Cloud Console → APIs & Services → Credentials."
                )
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(TOKEN_PICKLE, "wb") as f:
            pickle.dump(creds, f)
        print("✅  Authenticated and token cached.")

    return build("youtube", "v3", credentials=creds)


# ── Playlist creation ────────────────────────────────────────────────────────

def create_playlist(youtube, title, description, privacy):
    response = youtube.playlists().insert(
        part="snippet,status",
        body={
            "snippet": {
                "title": title,
                "description": description,
            },
            "status": {
                "privacyStatus": privacy  # "public", "private", or "unlisted"
            }
        }
    ).execute()
    playlist_id = response["id"]
    print(f"✅  Playlist created: '{title}'")
    print(f"    🔗 https://www.youtube.com/playlist?list={playlist_id}\n")
    return playlist_id


def add_video_to_playlist(youtube, playlist_id, video_id):
    youtube.playlistItems().insert(
        part="snippet",
        body={
            "snippet": {
                "playlistId": playlist_id,
                "resourceId": {
                    "kind": "youtube#video",
                    "videoId": video_id
                }
            }
        }
    ).execute()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Create a YouTube playlist from CSV.")
    parser.add_argument("--csv",      default=CSV_PATH,         help="Path to the CSV file")
    parser.add_argument("--filter",   default=None,             help="Filter by label: Educational | Neutral | Overstimulating")
    parser.add_argument("--name",     default=None,             help="Custom playlist name (overrides auto-name)")
    parser.add_argument("--privacy",  default="private",        help="Playlist privacy: public | private | unlisted (default: private)")
    args = parser.parse_args()

    # ── Load CSV ──────────────────────────────────────────────────────────────
    if not os.path.exists(args.csv):
        sys.exit(f"❌  CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    print(f"📄  Loaded {len(df)} videos from '{args.csv}'")

    # ── Optional filter ───────────────────────────────────────────────────────
    if args.filter:
        valid_labels = ["Educational", "Neutral", "Overstimulating"]
        if args.filter not in valid_labels:
            sys.exit(f"❌  --filter must be one of: {valid_labels}")
        df = df[df["label"] == args.filter]
        print(f"🔍  Filtered to '{args.filter}': {len(df)} videos")

    if df.empty:
        sys.exit("❌  No videos to add after filtering.")

    # ── Playlist name ─────────────────────────────────────────────────────────
    if args.name:
        playlist_title = args.name
    elif args.filter:
        playlist_title = f"Handpicked – {args.filter} Videos"
    else:
        playlist_title = "Handpicked Validated Playlist"

    description = (
        f"Auto-generated playlist from handpicked validated dataset.\n"
        f"Total videos: {len(df)}"
        + (f"\nFilter: {args.filter}" if args.filter else "")
    )

    # ── Authenticate ──────────────────────────────────────────────────────────
    print("\n🔐  Authenticating with YouTube...")
    youtube = get_authenticated_service()

    # ── Create playlist ───────────────────────────────────────────────────────
    playlist_id = create_playlist(youtube, playlist_title, description, args.privacy)

    # ── Add videos ────────────────────────────────────────────────────────────
    success, failed = 0, []

    for i, row in enumerate(df.itertuples(), 1):
        video_id = row.video_id
        title    = getattr(row, "title", video_id)
        try:
            add_video_to_playlist(youtube, playlist_id, video_id)
            print(f"  [{i:>3}/{len(df)}] ✅  {title[:70]}")
            success += 1
            time.sleep(0.3)   # polite rate limiting

        except HttpError as e:
            reason = e.reason if hasattr(e, "reason") else str(e)
            print(f"  [{i:>3}/{len(df)}] ❌  FAILED ({reason}) — {video_id}")
            failed.append({"video_id": video_id, "title": title, "error": reason})

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"✅  Added:  {success} videos")
    if failed:
        print(f"❌  Failed: {len(failed)} videos")
        fail_df = pd.DataFrame(failed)
        fail_df.to_csv("failed_videos.csv", index=False)
        print("    Saved failed videos to 'failed_videos.csv'")
    print(f"\n🎬  Playlist URL: https://www.youtube.com/playlist?list={playlist_id}")


if __name__ == "__main__":
    main()
