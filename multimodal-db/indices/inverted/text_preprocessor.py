from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

try:
    from nltk.corpus import stopwords
    from nltk.stem import SnowballStemmer
except ImportError:
    stopwords = None
    SnowballStemmer = None


DEFAULT_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "with",
    }
)


class _Stemmer(Protocol):

    def stem(self, word: str) -> str:
        ...


class _RuleStemmer:

    def stem(self, word: str) -> str:
        for suffix in ("ization", "ational", "fulness", "ousness", "iveness"):
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[: -len(suffix)]
        for suffix in ("ingly", "edly", "ments", "ment", "ness", "able", "ible"):
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[: -len(suffix)]
        for suffix in ("ing", "ers", "ies", "ied", "ed", "ly"):
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                base = word[: -len(suffix)]
                if suffix in {"ies", "ied"}:
                    return base + "y"
                return self._trim_double_consonant(base)
        if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
            return word[:-1]
        return word

    def _trim_double_consonant(self, word: str) -> str:
        if len(word) < 2:
            return word
        if word[-1] == word[-2] and word[-1] not in "aeiou":
            return word[:-1]
        return word


def _load_stopwords() -> frozenset[str]:
    if stopwords is None:
        return DEFAULT_STOPWORDS
    try:
        return frozenset(stopwords.words("english"))
    except LookupError:
        return DEFAULT_STOPWORDS


def _load_stemmer() -> _Stemmer:
    if SnowballStemmer is None:
        return _RuleStemmer()
    return SnowballStemmer("english")


@dataclass(frozen=True)
class TextPreprocessor:
    stopwords: frozenset[str] = field(default_factory=_load_stopwords)
    stemmer: _Stemmer = field(default_factory=_load_stemmer)
    min_token_length: int = 2
    _token_pattern: re.Pattern[str] = field(
        default=re.compile(r"[a-z0-9]+"),
        init=False,
        repr=False,
    )

    def tokenize(self, text: str) -> list[str]:
        terms: list[str] = []
        for token in self._token_pattern.findall(text.lower()):
            if len(token) < self.min_token_length or token in self.stopwords:
                continue
            stem = self.stemmer.stem(token)
            if stem and stem not in self.stopwords:
                terms.append(stem)
        return terms


DEFAULT_PREPROCESSOR = TextPreprocessor()
