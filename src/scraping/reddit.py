"""Scrape comments from r/Tunisia (and friends) via the Reddit API (PRAW).

Setup: create a "script" app at https://www.reddit.com/prefs/apps, then set
REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET / REDDIT_USER_AGENT in .env.

Run:  python -m src.scraping.reddit
"""
from __future__ import annotations

import time
from pathlib import Path

import praw
from tqdm import tqdm

from ..config import RAW, env, ensure_data_dirs, settings
from .storage import make_comment, write_jsonl

OUT = RAW / "reddit.jsonl"


def _client() -> praw.Reddit:
    return praw.Reddit(
        client_id=env("REDDIT_CLIENT_ID", required=True),
        client_secret=env("REDDIT_CLIENT_SECRET", required=True),
        user_agent=env("REDDIT_USER_AGENT", required=True),
        check_for_async=False,
    )


def _listing(subreddit, name: str, limit: int, time_filter: str):
    if name == "top":
        return subreddit.top(time_filter=time_filter, limit=limit)
    if name == "new":
        return subreddit.new(limit=limit)
    return subreddit.hot(limit=limit)


def scrape(out: Path = OUT) -> int:
    ensure_data_dirs()
    cfg = settings()["scraping"]
    rc = cfg["reddit"]
    sleep = cfg["request_sleep_seconds"]
    reddit = _client()

    total = 0
    for sub_name in rc["subreddits"]:
        sub = reddit.subreddit(sub_name)
        for listing in rc["listings"]:
            posts = _listing(sub, listing, rc["posts_per_listing"], rc["top_time_filter"])
            for post in tqdm(posts, desc=f"r/{sub_name} {listing}"):
                post.comments.replace_more(limit=0)
                rows = [
                    make_comment(
                        id=f"rd_{c.id}",
                        platform="reddit",
                        text=c.body or "",
                        source=sub_name,
                        timestamp=time.strftime("%Y-%m-%d", time.gmtime(c.created_utc)),
                    )
                    for c in post.comments.list()
                    if c.body not in ("[deleted]", "[removed]", None)
                ]
                total += write_jsonl(out, rows)
                time.sleep(sleep)
    print(f"Wrote {total} new comments -> {out}")
    return total


if __name__ == "__main__":
    scrape()
