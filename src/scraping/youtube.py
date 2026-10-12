"""Scrape comments from Tunisian YouTube channels via the official Data API v3.

Setup: enable "YouTube Data API v3" in Google Cloud, create an API key, put it in .env
as YOUTUBE_API_KEY. Add channel IDs (UC...) to config/config.yaml under scraping.youtube.

Run:  python -m src.scraping.youtube
"""
from __future__ import annotations

import time
from pathlib import Path

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tqdm import tqdm

from ..config import RAW, env, ensure_data_dirs, settings
from .storage import make_comment, write_jsonl

OUT = RAW / "youtube.jsonl"


def _client():
    key = env("YOUTUBE_API_KEY", required=True)
    return build("youtube", "v3", developerKey=key, cache_discovery=False)


def get_uploads_playlist(yt, channel_id: str) -> str | None:
    resp = yt.channels().list(part="contentDetails", id=channel_id).execute()
    items = resp.get("items", [])
    if not items:
        return None
    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]


def get_video_ids(yt, playlist_id: str, max_videos: int) -> list[str]:
    ids: list[str] = []
    page = None
    while len(ids) < max_videos:
        resp = yt.playlistItems().list(
            part="contentDetails", playlistId=playlist_id,
            maxResults=50, pageToken=page,
        ).execute()
        ids += [it["contentDetails"]["videoId"] for it in resp.get("items", [])]
        page = resp.get("nextPageToken")
        if not page:
            break
    return ids[:max_videos]


def get_comments(yt, video_id: str, max_comments: int, order: str, sleep: float):
    """Yield top-level comment threads for a video (no replies, no usernames)."""
    count = 0
    page = None
    while count < max_comments:
        try:
            resp = yt.commentThreads().list(
                part="snippet", videoId=video_id, maxResults=100,
                order=order, textFormat="plainText", pageToken=page,
            ).execute()
        except HttpError as e:
            # Comments disabled / video removed: skip quietly.
            if e.resp.status in (403, 404):
                return
            raise
        for item in resp.get("items", []):
            top = item["snippet"]["topLevelComment"]
            sn = top["snippet"]
            yield make_comment(
                id=f"yt_{top['id']}",
                platform="youtube",
                text=sn.get("textOriginal", ""),
                source=video_id,
                timestamp=(sn.get("publishedAt") or "")[:10] or None,
            )
            count += 1
            if count >= max_comments:
                break
        page = resp.get("nextPageToken")
        if not page:
            break
        time.sleep(sleep)


def scrape(out: Path = OUT) -> int:
    ensure_data_dirs()
    cfg = settings()["scraping"]
    yc = cfg["youtube"]
    sleep = cfg["request_sleep_seconds"]
    channels = yc.get("channel_ids") or []
    if not channels:
        print("No channel_ids in config/config.yaml (scraping.youtube.channel_ids). Nothing to do.")
        return 0

    yt = _client()
    total = 0
    for ch in channels:
        uploads = get_uploads_playlist(yt, ch)
        if not uploads:
            print(f"  channel {ch}: no uploads playlist, skipping")
            continue
        videos = get_video_ids(yt, uploads, yc["max_videos_per_channel"])
        for vid in tqdm(videos, desc=f"channel {ch}"):
            rows = list(get_comments(yt, vid, yc["max_comments_per_video"],
                                     yc["order"], sleep))
            total += write_jsonl(out, rows)
            time.sleep(sleep)
    print(f"Wrote {total} new comments -> {out}")
    return total


if __name__ == "__main__":
    scrape()
