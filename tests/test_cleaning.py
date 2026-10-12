"""Tests for the deterministic cleaning pieces."""
from src.cleaning.normalize import collapse_repeats, normalize
from src.cleaning.script_detect import detect_script


def test_collapse_repeats():
    assert collapse_repeats("barchaaaa", 2) == "barchaa"
    assert collapse_repeats("cool", 2) == "cool"


def test_normalize_strips_url_and_tatweel():
    out = normalize("شكراااا http://x.co ــ")
    assert "http" not in out
    assert "ـ" not in out
    assert "شكرا" in out  # diacritic-free, repeats collapsed


def test_normalize_unifies_letters():
    # أهلا -> اهلا (alef-hamza unified), ة -> ه
    assert normalize("أهلا") == "اهلا"


def test_detect_script():
    assert detect_script("behi barcha") == "latin"
    assert detect_script("مرحبا بيك") == "arabic"
    assert detect_script("behi برشا") == "mixed"
    assert detect_script("👍👍 123") == "other"
