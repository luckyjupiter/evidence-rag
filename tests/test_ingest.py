from __future__ import annotations

from pathlib import Path

import pytest

from evidence_rag.ingest import ingest_file, ingest_text
from evidence_rag.retrieval import keyword_search
from evidence_rag.store import Store

CORPUS = Path(__file__).resolve().parents[1] / "examples" / "corpus"


@pytest.fixture()
def store(tmp_path):
    with Store(tmp_path / "test.db") as instance:
        yield instance


def test_ingests_corpus_with_stable_source_metadata(store):
    result = ingest_file(store, CORPUS / "north-orchard-trial.md", chunk_chars=400, overlap=40)
    assert result.chunks >= 2
    assert not result.unchanged

    hit = keyword_search(store, "gravel layer", limit=1)[0]
    assert hit.title == "North Orchard Drip Irrigation Trial"
    assert hit.source.endswith("north-orchard-trial.md")
    source_text = (CORPUS / "north-orchard-trial.md").read_text(encoding="utf-8")
    assert source_text[hit.start : hit.end] == hit.text
    assert hit.idx >= 0


def test_reingesting_identical_content_is_a_noop(store):
    first = ingest_text(store, "Alpha beta gamma.", source="note:1")
    before = store.stats()
    second = ingest_text(store, "Alpha beta gamma.", source="note:1")
    assert first.doc_id == second.doc_id
    assert second.unchanged is True
    assert store.stats() == before


def test_changed_document_replaces_its_chunks(store, tmp_path):
    path = tmp_path / "evolving.md"
    path.write_text("The first version mentions zeppelin.", encoding="utf-8")
    first = ingest_file(store, path, chunk_chars=400, overlap=20)
    assert keyword_search(store, "zeppelin")

    path.write_text("The second version mentions submarine.", encoding="utf-8")
    second = ingest_file(store, path, chunk_chars=400, overlap=20)
    assert first.doc_id == second.doc_id
    assert second.unchanged is False
    assert keyword_search(store, "submarine")
    assert keyword_search(store, "zeppelin") == []
    assert store.stats().documents == 1


def test_different_sources_are_distinct_documents(store):
    ingest_text(store, "Shared wording about basins.", source="a")
    ingest_text(store, "Shared wording about basins.", source="b")
    assert store.stats().documents == 2
