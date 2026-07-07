"""Tests for the production text front-end: chunking + number expansion."""

from tts.text.chunk import chunk_text
from tts.text.numbers import expand_numbers


def test_short_text_single_chunk():
    assert chunk_text("نَعَمْ.", 160) == ["نَعَمْ."]


def test_long_text_all_chunks_within_limit():
    long = ("الصَّلَاةُ عِمَادُ الدِّينِ، مَنْ أَقَامَهَا فَقَدْ أَقَامَ الدِّينَ. "
            "وَهِيَ أَوَّلُ مَا يُحَاسَبُ عَلَيْهِ الْعَبْدُ يَوْمَ الْقِيَامَةِ.") * 3
    chunks = chunk_text(long, 160)
    assert len(chunks) > 1
    assert all(len(c) <= 160 for c in chunks)


def test_chunks_reconstruct_all_words():
    text = "جُمْلَةٌ أُولَى. جُمْلَةٌ ثَانِيَةٌ، وَفِيهَا تَفْصِيلٌ. وَخِتَامٌ."
    joined = " ".join(chunk_text(text, 40))
    for word in text.replace(".", " ").replace("،", " ").split():
        assert word in joined


def test_empty():
    assert chunk_text("", 160) == []


def test_expand_western_digits():
    assert "خمسة" in expand_numbers("عندي 5 كتب")


def test_expand_arabic_indic_digits():
    out = expand_numbers("الآيةُ ٧")
    assert "٧" not in out and "سبعة" in out
