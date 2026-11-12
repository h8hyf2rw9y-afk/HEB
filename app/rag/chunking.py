"""Split extracted blocks into overlapping chunks that never cross page boundaries."""

from __future__ import annotations

import hashlib
from itertools import groupby

from .extraction import ExtractedDocument
from .models import Chunk


def _chunk_id(document: str, page: int, ordinal: int, text: str) -> str:
    digest = hashlib.sha1(f"{document}|{page}|{ordinal}|{text}".encode("utf-8")).hexdigest()[:10]
    return f"{document}:p{page}:{ordinal}:{digest}"


def chunk_document(
    extracted: ExtractedDocument, *, max_words: int = 160, overlap_words: int = 40
) -> list[Chunk]:
    """Group consecutive blocks that share (page, section) and window them by words.

    Keeping each chunk inside a single page/section guarantees exact citations.
    """
    if max_words <= 0:
        raise ValueError("max_words debe ser > 0")
    if not 0 <= overlap_words < max_words:
        raise ValueError("overlap_words debe estar en [0, max_words)")

    chunks: list[Chunk] = []

    def add(page: int, section: str | None, text: str, kind: str = "text", label: str | None = None) -> None:
        ordinal = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(extracted.document, page, ordinal, text),
                document=extracted.document,
                page=page,
                section=section,
                text=text,
                kind=kind,
                label=label,
            )
        )

    # Table rows are atomic chunks (one row = one citable fact); prose blocks that
    # share (page, section) are concatenated and windowed by words.
    def key(b):
        return (b.page, b.section, b.kind == "text" or id(b))

    for (page, section, _), group in groupby(extracted.blocks, key=key):
        blocks = list(group)
        if blocks[0].kind != "text":
            for b in blocks:
                add(page, section, b.text, b.kind, b.label)
            continue
        words = " ".join(b.text for b in blocks).split()
        if not words:
            continue
        step = max_words - overlap_words
        start = 0
        while True:
            add(page, section, " ".join(words[start : start + max_words]))
            if start + max_words >= len(words):
                break
            start += step
    return chunks
