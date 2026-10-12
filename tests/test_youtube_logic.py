"""Mocked-API tests for the YouTube scraper logic (no network, no key).

Verifies channel resolution, video-id pagination, and comment parsing by faking the
googleapiclient client. Skipped until scraping deps are installed.
"""
from unittest.mock import MagicMock

import pytest

pytest.importorskip("googleapiclient")

from src.scraping.youtube import (  # noqa: E402
    get_comments,
    get_video_ids,
    resolve_channel,
)


def test_resolve_channel_handle():
    yt = MagicMock()
    yt.channels.return_value.list.return_value.execute.return_value = {
        "items": [{
            "id": "UC12345678901234567890AB",
            "contentDetails": {"relatedPlaylists": {"uploads": "UU_uploads"}},
            "snippet": {"title": "Test Channel"},
        }]
    }
    result = resolve_channel(yt, "@foo")
    assert result == ("UC12345678901234567890AB", "UU_uploads", "Test Channel")
    # a handle is passed via forHandle, with the leading '@' stripped
    assert yt.channels.return_value.list.call_args.kwargs["forHandle"] == "foo"


def test_resolve_channel_not_found():
    yt = MagicMock()
    yt.channels.return_value.list.return_value.execute.return_value = {"items": []}
    assert resolve_channel(yt, "@missing") is None


def test_get_video_ids_paginates():
    yt = MagicMock()
    yt.playlistItems.return_value.list.return_value.execute.side_effect = [
        {"items": [{"contentDetails": {"videoId": "v1"}},
                   {"contentDetails": {"videoId": "v2"}}],
         "nextPageToken": "p2"},
        {"items": [{"contentDetails": {"videoId": "v3"}}]},
    ]
    assert get_video_ids(yt, "UU_uploads", max_videos=10) == ["v1", "v2", "v3"]


def test_get_video_ids_respects_max():
    yt = MagicMock()
    yt.playlistItems.return_value.list.return_value.execute.return_value = {
        "items": [{"contentDetails": {"videoId": f"v{i}"}} for i in range(50)],
    }
    assert get_video_ids(yt, "UU_uploads", max_videos=3) == ["v0", "v1", "v2"]


def test_get_comments_parses_and_drops_pii():
    yt = MagicMock()
    yt.commentThreads.return_value.list.return_value.execute.return_value = {
        "items": [{
            "snippet": {"topLevelComment": {
                "id": "c1",
                "snippet": {"textOriginal": "behi barcha",
                            "publishedAt": "2024-05-06T10:00:00Z"},
            }}
        }]
    }
    rows = list(get_comments(yt, "vid1", max_comments=10, order="time", sleep=0))
    assert len(rows) == 1
    row = rows[0]
    assert row == {"id": "yt_c1", "platform": "youtube", "text": "behi barcha",
                   "source": "vid1", "timestamp": "2024-05-06"}
    # schema carries no username / author field
    assert "author" not in row and set(row) == {
        "id", "platform", "text", "source", "timestamp"}
