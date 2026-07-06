from __future__ import annotations

import pytest

from indices.inverted.text_chunker import TextChunker


def test_chunker_splits_paragraphs_on_blank_lines() -> None:
    chunker = TextChunker()

    chunks = chunker.split("first paragraph\n\nsecond paragraph\n\n\nthird one")

    assert chunks == ["first paragraph", "second paragraph", "third one"]


def test_chunker_keeps_single_paragraph_whole() -> None:
    chunker = TextChunker()

    chunks = chunker.split("single paragraph with\na line break")

    assert chunks == ["single paragraph with\na line break"]


def test_chunker_falls_back_to_full_text_when_no_paragraphs() -> None:
    chunker = TextChunker()

    assert chunker.split("") == [""]
    assert chunker.split("   \n\n   ") == ["   \n\n   "]


def test_chunker_windows_long_single_line_text() -> None:
    chunker = TextChunker(window_words=4)

    chunks = chunker.split("w1 w2 w3 w4 w5 w6 w7 w8 w9 w10")

    assert chunks == ["w1 w2 w3 w4", "w5 w6 w7 w8", "w9 w10"]


def test_chunker_keeps_short_single_line_text_whole() -> None:
    chunker = TextChunker(window_words=4)

    chunks = chunker.split("w1 w2 w3 w4")

    assert chunks == ["w1 w2 w3 w4"]


def test_chunker_prefers_paragraphs_over_window_fallback() -> None:
    chunker = TextChunker(window_words=2)

    chunks = chunker.split("one two three four\n\nfive six seven")

    assert chunks == ["one two three four", "five six seven"]


def test_chunker_rejects_non_positive_window() -> None:
    with pytest.raises(ValueError):
        TextChunker(window_words=0)
