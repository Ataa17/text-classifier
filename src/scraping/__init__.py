"""Scrapers. Each writes JSONL to data/raw/ with a shared schema.

Comment schema (no PII — we deliberately drop usernames):
    {
      "id": str,            # platform-prefixed, stable
      "platform": str,      # "youtube" | "reddit" | "news"
      "text": str,          # raw comment text
      "source": str,        # video_id / subreddit / site name
      "timestamp": str | None,  # ISO date (coarse), or null
    }
"""
