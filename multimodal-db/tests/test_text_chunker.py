from __future__ import annotations

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
