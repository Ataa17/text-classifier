"""Tests for YouTube channel-reference parsing.

Skipped until scraping deps (google-api-python-client) are installed.
"""
import pytest

pytest.importorskip("googleapiclient")

from src.scraping.youtube import _parse_ref  # noqa: E402


def test_parse_channel_id():
    assert _parse_ref("UC12345678901234567890AB") == ("id", "UC12345678901234567890AB")


def test_parse_handle():
    assert _parse_ref("@nessmatv") == ("handle", "@nessmatv")


def test_parse_bare_name_is_handle():
    assert _parse_ref("nessmatv") == ("handle", "nessmatv")


def test_parse_url_channel_id():
    ref = "https://www.youtube.com/channel/UC12345678901234567890AB"
    assert _parse_ref(ref) == ("id", "UC12345678901234567890AB")


def test_parse_url_handle():
    assert _parse_ref("https://youtube.com/@Flayou") == ("handle", "@Flayou")


def test_parse_url_legacy_user():
    assert _parse_ref("https://www.youtube.com/user/SomeOldName") == ("username", "SomeOldName")


def test_parse_url_custom_c():
    assert _parse_ref("https://www.youtube.com/c/SomeCustom") == ("query", "SomeCustom")
