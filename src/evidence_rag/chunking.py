from __future__ import annotations

import re

from .models import ChunkSpan

_SENTENCE_END = re.compile(r"[.!?](?:\s|$)")


def chunk_text(
    text: str,
    max_chars: int = 2000,
    overlap: int = 200,
) -> list[ChunkSpan]:
    """Split ``text`` into overlapping, boundary-aware chunks.

    Boundaries prefer a blank line, then a sentence end, within the second half
    of the window. Overlap is measured in characters. Offsets always refer to
    the original text, so a result can be traced back to the source.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be >= 0 and < max_chars")
    if not text or not text.strip():
        return []

    spans: list[ChunkSpan] = []
    start = 0
    idx = 0
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            floor = start + max_chars // 2
            paragraph = text.rfind("\n\n", floor, end)
            if paragraph != -1:
                end = paragraph + 2
            else:
                match = None
                for match in _SENTENCE_END.finditer(text, floor, end):
                    pass
                if match is not None:
                    end = match.end()

        raw = text[start:end]
        lead = len(raw) - len(raw.lstrip())
        trail = len(raw) - len(raw.rstrip())
        piece_start = start + lead
        piece_end = end - trail
        piece = text[piece_start:piece_end]

        if piece:
            spans.append(ChunkSpan(idx=idx, start=piece_start, end=piece_end, text=piece))
            idx += 1

        if end >= n:
            break
        next_start = max(end - overlap, start + 1)
        start = next_start

    return spans
