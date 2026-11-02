from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkSpan:
    """A slice of a source document."""

    idx: int
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class IngestResult:
    doc_id: str
    source: str
    title: str
    chunks: int
    unchanged: bool


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    doc_id: str
    title: str
    source: str
    idx: int
    text: str
    start: int
    end: int
    score: float


@dataclass(frozen=True)
class EntityHit:
    name: str
    type: str
    mentions: int


@dataclass(frozen=True)
class RelationHit:
    subject: str
    predicate: str
    object: str
    evidence: str
    chunk_id: str
    source: str
    title: str
    idx: int


@dataclass(frozen=True)
class Stats:
    documents: int
    chunks: int
    entities: int
    relations: int
    embeddings: int
