"""Scrape comments from Tunisian YouTube channels via the official Data API v3.

Setup: enable "YouTube Data API v3" in Google Cloud, create an API key, put it in .env
as YOUTUBE_API_KEY. List channels in config/config.yaml under scraping.youtube.channels —
each entry may be a channel ID (UC...), an @handle, or a channel URL.

Run:  python -m src.scraping.youtube
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from urllib.parse import urlparse

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tqdm import tqdm

from ..config import RAW, env, ensure_data_dirs, settings
from .storage import make_comment, write_jsonl

OUT = RAW / "youtube.jsonl"


def _client():
    key = env("YOUTUBE_API_KEY", required=True)
    return build("youtube", "v3", developerKey=key, cache_discovery=False)


_UC_ID = re.compile(r"^UC[\w-]{22}$")


def _parse_ref(ref: str) -> tuple[str, str]:
    """Classify a channel reference into (kind, value).

    kind is one of: "id" (UC...), "handle" (@name), "username" (legacy /user/),
    or "query" (a custom /c/ URL or bare name we must resolve via search).
    """
    ref = ref.strip()
    if _UC_ID.match(ref):
        return "id", ref
    if ref.startswith("@"):
        return "handle", ref
    if "youtube.com" in ref or "youtu.be" in ref:
        parts = [p for p in urlparse(ref).path.split("/") if p]
        if not parts:
            return "query", ref
        head = parts[0]
        if head == "channel" and len(parts) > 1 and _UC_ID.match(parts[1]):
            return "id", parts[1]
        if head.startswith("@"):
            return "handle", head
        if head == "user" and len(parts) > 1:
            return "username", parts[1]
        if head == "c" and len(parts) > 1:
            return "query", parts[1]
        return "query", head
    # bare token: treat as a handle (the API's forHandle accepts names without '@')
    return "handle", ref


def resolve_channel(yt, ref: str) -> tuple[str, str, str] | None:
    """Resolve an ID / @handle / URL to (channel_id, uploads_playlist, title).

    Returns None (and prints a note) if the channel can't be found — the caller skips it.
    """
    kind, value = _parse_ref(ref)
    params = {"part": "contentDetails,snippet"}
    if kind == "id":
        params["id"] = value
    elif kind == "handle":
        params["forHandle"] = value.lstrip("@")
    elif kind == "username":
        params["forUsername"] = value
    else:  # query: find the channel by search, then look it up by id
        sr = yt.search().list(part="snippet", q=value, type="channel", maxResults=1).execute()
        hits = sr.get("items", [])
        if not hits:
            print(f"  could not resolve channel ref {ref!r} (no search match), skipping")
            return None
        params["id"] = hits[0]["snippet"]["channelId"]

    resp = yt.channels().list(**params).execute()
    items = resp.get("items", [])
    if not items:
        print(f"  could not resolve channel ref {ref!r}, skipping")
        return None
    ch = items[0]
    return (
        ch["id"],
        ch["contentDetails"]["relatedPlaylists"]["uploads"],
        ch["snippet"]["title"],
    )


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
    channels = yc.get("channels") or []
    if not channels:
        print("No channels in config/config.yaml (scraping.youtube.channels). Nothing to do.")
        return 0

    yt = _client()
    total = 0
    for ref in channels:
        resolved = resolve_channel(yt, ref)
        if not resolved:
            continue
        channel_id, uploads, title = resolved
        videos = get_video_ids(yt, uploads, yc["max_videos_per_channel"])
        for vid in tqdm(videos, desc=title):
            rows = list(get_comments(yt, vid, yc["max_comments_per_video"],
                                     yc["order"], sleep))
            total += write_jsonl(out, rows)
            time.sleep(sleep)
    print(f"Wrote {total} new comments -> {out}")
    return total


if __name__ == "__main__":
    scrape()
