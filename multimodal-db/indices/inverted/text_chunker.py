from __future__ import annotations

import re

PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
DEFAULT_WINDOW_WORDS = 100


class TextChunker:

    def __init__(self, window_words: int = DEFAULT_WINDOW_WORDS) -> None:
        if window_words < 1:
            raise ValueError("window_words must be positive")
        self.window_words = window_words

    def split(self, text: str) -> list[str]:
        paragraphs = [part.strip() for part in PARAGRAPH_BREAK.split(text)]
        paragraphs = [part for part in paragraphs if part]
        if not paragraphs:
            # Sin párrafos se usa el texto completo
            return [text]
        if len(paragraphs) > 1:
            return paragraphs
        return self._split_by_window(paragraphs[0])

    # Corta un texto sin párrafos en ventanas de palabras seguidas
    def _split_by_window(self, text: str) -> list[str]:
        words = text.split()
        if len(words) <= self.window_words:
            return [text]
        return [
            " ".join(words[start:start + self.window_words])
            for start in range(0, len(words), self.window_words)
        ]


DEFAULT_CHUNKER = TextChunker()
