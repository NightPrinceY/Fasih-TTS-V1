"""Golden tests for the Arabic text front-end (deterministic, no model)."""

from tts.text.diacritics import coverage, needs_diacritization
from tts.text.normalize import HARAKAT, normalize, strip_diacritics

DIAC = "السَّلَامُ عَلَيْكُمْ وَرَحْمَةُ اللَّهِ"
PLAIN = "السلام عليكم ورحمة الله"


def test_normalize_preserves_diacritics():
    out = normalize(DIAC)
    assert any(c in HARAKAT for c in out), "diacritics must be preserved"
    assert out == DIAC  # already clean


def test_normalize_removes_tatweel_and_collapses_ws():
    assert normalize("الســـلام   عليكم") == "السلام عليكم"


def test_normalize_ellipsis():
    assert normalize("انتظر...") == "انتظر…"


def test_strip_diacritics_is_analysis_only():
    assert strip_diacritics(DIAC) == PLAIN


def test_coverage_levels():
    assert coverage(DIAC).level == "full"
    assert coverage(PLAIN).level == "none"


def test_needs_diacritization():
    assert needs_diacritization(PLAIN)
    assert not needs_diacritization(DIAC)


def test_empty():
    assert normalize("") == ""
    assert coverage("").level == "none"
