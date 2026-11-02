"""Tokenization shared by BM25 indexing and BM25 queries (both must use exactly the same rules).

Music-specific choices:
- keep one-character words: note names (C, D, E...) are single letters;
- English + French stopwords, but never the note letters a-g ("a", "c", "d" are stopwords
  by default);
- lowercase, no stemming: identical treatment for both languages, easy to explain.
"""

import bm25s
from bm25s.stopwords import STOPWORDS_EN, STOPWORDS_FRENCH

NOTE_LETTERS = set("abcdefg")
STOPWORDS = sorted((set(STOPWORDS_EN) | set(STOPWORDS_FRENCH)) - NOTE_LETTERS)
TOKEN_PATTERN = r"(?u)\b\w+\b"


def bm25_tokenize(texts: list[str]) -> bm25s.tokenization.Tokenized:
    return bm25s.tokenize(
        texts, lower=True, token_pattern=TOKEN_PATTERN, stopwords=STOPWORDS, show_progress=False
    )


def bm25_terms(text: str) -> list[str]:
    """Readable list of the terms BM25 sees for one text (for tests and debugging)."""
    return bm25s.tokenize(
        [text],
        lower=True,
        token_pattern=TOKEN_PATTERN,
        stopwords=STOPWORDS,
        return_ids=False,
        show_progress=False,
    )[0]
