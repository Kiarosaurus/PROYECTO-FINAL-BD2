from __future__ import annotations

import re

PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


class TextChunker:

    def split(self, text: str) -> list[str]:
        paragraphs = [part.strip() for part in PARAGRAPH_BREAK.split(text)]
        paragraphs = [part for part in paragraphs if part]
        if not paragraphs:
            # Sin párrafos se usa el texto completo
            return [text]
        return paragraphs


DEFAULT_CHUNKER = TextChunker()
