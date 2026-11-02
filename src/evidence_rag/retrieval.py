from __future__ import annotations

import math
import re
import struct
from typing import Sequence

from .embeddings import Embedder
from .models import SearchHit
from .store import Store

_TOKEN = re.compile(r"\w+", re.UNICODE)


def _tokens(query: str) -> list[str]:
    seen: list[str] = []
    for token in _TOKEN.findall(query.casefold()):
        if token not in seen:
            seen.append(token)
        if len(seen) >= 32:
            break
    return seen


def _match_query(tokens: Sequence[str], operator: str) -> str:
    return f" {operator} ".join(f'"{token}"' for token in tokens)


def keyword_search(store: Store, query: str, limit: int = 8) -> list[SearchHit]:
    """FTS5 keyword retrieval with a documented AND-then-OR fallback."""
    tokens = _tokens(query)
    if not tokens or limit <= 0:
        return []
    hits = store.search_keyword(_match_query(tokens, "AND"), limit)
    if hits:
        return hits
    return store.search_keyword(_match_query(tokens, "OR"), limit)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def vector_search(
    store: Store,
    query: str,
    embedder: Embedder,
    limit: int = 8,
) -> list[SearchHit]:
    """Brute-force cosine similarity over locally stored chunk embeddings."""
    if limit <= 0:
        return []
    rows = store.embedded_chunks(embedder.name)
    if not rows:
        return []
    query_vector = embedder.embed([query])[0]
    scored: list[tuple[float, object]] = []
    for row in rows:
        dim = int(row["dim"])
        vector = list(struct.unpack(f"<{dim}f", row["vector"]))
        scored.append((_cosine(query_vector, vector), row))
    scored.sort(key=lambda item: item[0], reverse=True)
    hits: list[SearchHit] = []
    for score, row in scored[:limit]:
        if score <= 0:
            continue
        hits.append(
            SearchHit(
                chunk_id=row["chunk_id"],
                doc_id=row["doc_id"],
                title=row["title"],
                source=row["source"],
                idx=row["idx"],
                text=row["text"],
                start=row["start"],
                end=row["end"],
                score=score,
            )
        )
    return hits
