"""Two chunking strategies, measured in tokens of the embedding model.

- `chunk_fixed`: sliding window of `size` tokens with `overlap`, ignoring the document structure.
- `chunk_sections`: packs whole paragraphs into chunks of at most `max_tokens`, never crossing a
  section boundary; long paragraphs are split on sentences; tiny chunks are merged into the previous
  one of the same article.

Token counting goes through a small `Tokenizer` protocol (character spans of each token), so tests
can use a whitespace tokenizer and production uses the real bge-m3 tokenizer.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from musicteacher.ingestion.clean import Section

SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


class Tokenizer(Protocol):
    def token_spans(self, text: str) -> list[tuple[int, int]]:
        """Character (start, end) of each token in `text`."""
        ...


class WhitespaceTokenizer:
    """One token per whitespace-separated word. For tests and quick experiments."""

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        return [m.span() for m in re.finditer(r"\S+", text)]


class HFTokenizer:
    """Tokenizer of a Hugging Face model (e.g. BAAI/bge-m3), so sizes match what the model sees."""

    def __init__(self, model_name: str) -> None:
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(model_name)
        self._tok.model_max_length = 10**9  # we only count tokens here, no truncation warning

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        encoded = self._tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return [tuple(span) for span in encoded["offset_mapping"]]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    article_id: str
    language: str
    title: str
    section: str
    text: str
    n_tokens: int


def load_raw_articles(raw_dir: Path) -> list[tuple[dict, str]]:
    """(metadata, raw text) for every article saved by `fetch_wikipedia`."""
    articles = []
    for meta_path in sorted(raw_dir.glob("*/*.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        text = meta_path.with_suffix(".txt").read_text(encoding="utf-8")
        articles.append((meta, text))
    return articles


def _make_chunks(meta: dict, pieces: list[tuple[str, str]], tok: Tokenizer) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"{meta['id']}#{i:03d}",
            article_id=meta["id"],
            language=meta["language"],
            title=meta["title"],
            section=section,
            text=text,
            n_tokens=len(tok.token_spans(text)),
        )
        for i, (section, text) in enumerate(pieces)
    ]


def _windows(text: str, tok: Tokenizer, size: int, overlap: int) -> list[tuple[int, int]]:
    """Character ranges of token windows of `size` tokens, each overlapping the previous one."""
    spans = tok.token_spans(text)
    ranges = []
    step = size - overlap
    for start in range(0, len(spans), step):
        window = spans[start : start + size]
        ranges.append((window[0][0], window[-1][1]))
        if start + size >= len(spans):
            break
    return ranges


def chunk_fixed(
    meta: dict, sections: list[Section], tok: Tokenizer, size: int = 300, overlap: int = 50
) -> list[Chunk]:
    full, section_starts = "", []
    for s in sections:
        section_starts.append((len(full), s.path))
        full += s.text + "\n"

    pieces = []
    for start, end in _windows(full, tok, size, overlap):
        section = [path for offset, path in section_starts if offset <= start][-1]
        pieces.append((section, full[start:end]))
    return _make_chunks(meta, pieces, tok)


def _split_long_paragraph(paragraph: str, tok: Tokenizer, max_tokens: int) -> list[str]:
    """Pack sentences up to `max_tokens`; a single over-long sentence is cut by tokens."""
    parts, current = [], ""
    for sentence in SENTENCE_END.split(paragraph):
        if len(tok.token_spans(sentence)) > max_tokens:
            if current:
                parts.append(current)
                current = ""
            parts += [sentence[a:b] for a, b in _windows(sentence, tok, max_tokens, 0)]
            continue
        candidate = f"{current} {sentence}".strip()
        if current and len(tok.token_spans(candidate)) > max_tokens:
            parts.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def chunk_sections(
    meta: dict,
    sections: list[Section],
    tok: Tokenizer,
    max_tokens: int = 300,
    min_tokens: int = 64,
) -> list[Chunk]:
    pieces: list[tuple[str, str]] = []
    for s in sections:
        current: list[str] = []
        current_tokens = 0
        for paragraph in s.text.split("\n"):
            n = len(tok.token_spans(paragraph))
            if n > max_tokens:
                if current:
                    pieces.append((s.path, "\n".join(current)))
                    current, current_tokens = [], 0
                pieces += [(s.path, p) for p in _split_long_paragraph(paragraph, tok, max_tokens)]
            elif current and current_tokens + n > max_tokens:
                pieces.append((s.path, "\n".join(current)))
                current, current_tokens = [paragraph], n
            else:
                current.append(paragraph)
                current_tokens += n
        if current:
            pieces.append((s.path, "\n".join(current)))

    # Merge tiny pieces (a one-line section, a short trailing paragraph) into the previous piece.
    merged: list[tuple[str, str]] = []
    for section, text in pieces:
        if merged and len(tok.token_spans(text)) < min_tokens:
            prev_section, prev_text = merged[-1]
            merged[-1] = (prev_section, f"{prev_text}\n{text}")
        else:
            merged.append((section, text))
    return _make_chunks(meta, merged, tok)
