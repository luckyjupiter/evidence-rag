from __future__ import annotations

from pathlib import Path

import pytest

from evidence_rag.embeddings import HashingEmbedder
from evidence_rag.ingest import ingest_file, ingest_text
from evidence_rag.retrieval import keyword_search, vector_search
from evidence_rag.store import Store

CORPUS = Path(__file__).resolve().parents[1] / "examples" / "corpus"


@pytest.fixture()
def store(tmp_path):
    with Store(tmp_path / "test.db") as instance:
        yield instance


def test_keyword_search_prefers_the_matching_document(store):
    ingest_file(store, CORPUS / "north-orchard-trial.md", chunk_chars=400, overlap=40)
    ingest_file(store, CORPUS / "harbor-point-calibration.md", chunk_chars=400, overlap=40)
    hits = keyword_search(store, "vent tube salt creep", limit=3)
    assert hits
    assert hits[0].title == "Tide Gauge Calibration Notes: Harbor Point"


def test_keyword_search_falls_back_to_or(store):
    ingest_text(store, "The gravel layer drains quickly.", source="x")
    hits = keyword_search(store, "gravel absentterm", limit=3)
    assert hits


def test_keyword_search_punctuation_is_safe(store):
    ingest_text(store, "Signal drift and filter checks.", source="x")
    assert keyword_search(store, 'signal "drift" (filter)', limit=3)


def test_vector_search_uses_stored_embeddings(store):
    embedder = HashingEmbedder()
    ingest_file(
        store,
        CORPUS / "north-orchard-trial.md",
        chunk_chars=300,
        overlap=30,
        embedder=embedder,
    )
    ingest_file(
        store,
        CORPUS / "harbor-point-calibration.md",
        chunk_chars=300,
        overlap=30,
        embedder=embedder,
    )
    hits = vector_search(store, "soil moisture irrigation", embedder, limit=3)
    assert hits
    assert hits[0].title == "North Orchard Drip Irrigation Trial"
    assert store.stats().embeddings > 0


def test_embedding_is_idempotent(store):
    embedder = HashingEmbedder()
    ingest_text(store, "One two three.", source="same", embedder=embedder)
    count = store.stats().embeddings
    ingest_text(store, "One two three.", source="same", embedder=embedder)
    assert store.stats().embeddings == count


def test_empty_query_returns_nothing(store):
    ingest_text(store, "Anything.", source="x")
    assert keyword_search(store, "   ", limit=5) == []
    assert vector_search(store, "anything", HashingEmbedder(), limit=0) == []
