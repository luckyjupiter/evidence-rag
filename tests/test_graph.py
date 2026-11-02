from __future__ import annotations

from pathlib import Path

import pytest

from evidence_rag.extract import Entity, Extraction, Relation, parse_extraction
from evidence_rag.ingest import ingest_file, ingest_text
from evidence_rag.store import Store

CORPUS = Path(__file__).resolve().parents[1] / "examples" / "corpus"


class FakeExtractor:
    def __init__(self) -> None:
        self.calls = 0

    def extract(self, text: str) -> Extraction:
        self.calls += 1
        if "Mira Chen" not in text:
            return Extraction()
        return Extraction(
            entities=[
                Entity("Mira Chen", "person"),
                Entity("North Orchard", "location"),
            ],
            relations=[
                Relation(
                    "Mira Chen",
                    "designed",
                    "North Orchard",
                    "Mira Chen designed the trial",
                )
            ],
        )


@pytest.fixture()
def store(tmp_path):
    with Store(tmp_path / "test.db") as instance:
        yield instance


def test_relations_keep_chunk_and_source_evidence(store):
    ingest_file(
        store,
        CORPUS / "north-orchard-trial.md",
        chunk_chars=800,
        overlap=80,
        extractor=FakeExtractor(),
    )
    rows = store.relations_for("Mira Chen")
    assert rows
    row = rows[0]
    assert row.predicate == "designed"
    assert row.object == "North Orchard"
    assert row.evidence == "Mira Chen designed the trial"
    assert row.source.endswith("north-orchard-trial.md")
    assert row.idx >= 0
    assert store.find_entities("mira")[0].mentions >= 1


def test_relations_created_for_entities_missing_from_entity_list(store):
    class RelationOnly:
        def extract(self, text: str) -> Extraction:
            return Extraction(
                entities=[],
                relations=[Relation("Alpha", "links_to", "Beta", "Alpha links Beta")],
            )

    ingest_text(store, "Alpha links Beta.", source="x", extractor=RelationOnly())
    rows = store.relations_for("Alpha")
    assert rows and rows[0].object == "Beta"


def test_unchanged_document_skips_extraction(store):
    extractor = FakeExtractor()
    ingest_file(store, CORPUS / "north-orchard-trial.md", extractor=extractor)
    first_calls = extractor.calls
    ingest_file(store, CORPUS / "north-orchard-trial.md", extractor=extractor)
    assert extractor.calls == first_calls


def test_changed_document_drops_old_relations(store):
    class First:
        def extract(self, text: str) -> Extraction:
            return Extraction(
                entities=[Entity("Widget", "product")],
                relations=[Relation("Widget", "has_state", "red", "Widget is red")],
            )

    class Second:
        def extract(self, text: str) -> Extraction:
            return Extraction()

    ingest_text(store, "Widget is red.", source="x", extractor=First())
    assert store.relations_for("Widget")
    ingest_text(store, "Something else entirely.", source="x", extractor=Second())
    assert store.relations_for("Widget") == []


def test_parse_extraction_handles_malformed_output():
    assert parse_extraction(None).entities == []
    assert parse_extraction("not json at all").relations == []
    assert parse_extraction({"entities": "wrong", "relations": 5}).entities == []

    result = parse_extraction(
        {
            "entities": [
                {"name": "  Ada   Lovelace ", "type": "Person"},
                {"name": "", "type": "person"},
                {"name": "Mystery", "type": "not-a-type"},
            ],
            "relations": [
                {"subject": "Ada Lovelace", "predicate": "Wrote Notes", "object": "Mystery"},
                {"subject": "X", "predicate": "self", "object": "X"},
                {"subject": "OnlySubject", "predicate": "missing_object"},
            ],
        }
    )
    assert [entity.name for entity in result.entities] == ["Ada Lovelace", "Mystery"]
    assert result.entities[1].type == "other"
    assert len(result.relations) == 1
    assert result.relations[0].predicate == "wrote_notes"


def test_parse_extraction_strips_code_fence():
    payload = '```json\n{"entities": [{"name": "A"}], "relations": []}\n```'
    assert parse_extraction(payload).entities[0].name == "A"


def test_parse_extraction_handles_braces_inside_quoted_evidence():
    payload = (
        'Here is the result: {"entities": [], "relations": ['
        '{"subject": "A", "predicate": "mentions", "object": "B", '
        '"evidence": "the set {x} and a closing } inside a quote"}]}'
    )
    relation = parse_extraction(payload).relations[0]
    assert relation.evidence == "the set {x} and a closing } inside a quote"


def test_extraction_can_be_added_after_initial_ingestion(store):
    text = "Mira Chen designed the trial at North Orchard."
    ingest_text(store, text, source="late-graph")
    extractor = FakeExtractor()
    result = ingest_text(store, text, source="late-graph", extractor=extractor)
    assert result.unchanged
    assert extractor.calls == 1
    assert store.relations_for("Mira Chen")


def test_failed_extraction_is_retried_without_changing_source(store):
    class RetryExtractor(FakeExtractor):
        def extract(self, text):
            if self.calls == 0:
                self.calls += 1
                raise RuntimeError("temporary provider failure")
            return super().extract(text)

    extractor = RetryExtractor()
    text = "Mira Chen designed the trial at North Orchard."
    with pytest.raises(RuntimeError, match="temporary provider failure"):
        ingest_text(store, text, source="retry", extractor=extractor)
    ingest_text(store, text, source="retry", extractor=extractor)
    assert store.relations_for("Mira Chen")


def test_reextract_replaces_previous_graph_without_duplicates(store):
    extractor = FakeExtractor()
    text = "Mira Chen designed the trial at North Orchard."
    ingest_text(store, text, source="rerun", extractor=extractor)
    before = store.stats()
    ingest_text(store, text, source="rerun", extractor=extractor, reextract=True)
    assert extractor.calls == 2
    assert store.stats() == before


def test_changing_extractor_replaces_graph_on_unchanged_text(store):
    text = "Mira Chen designed the trial at North Orchard."
    ingest_text(store, text, source="new-extractor", extractor=FakeExtractor())
    assert store.relations_for("Mira Chen")

    class EmptyExtractor:
        def extract(self, text):
            return Extraction()

    ingest_text(store, text, source="new-extractor", extractor=EmptyExtractor())
    assert store.relations_for("Mira Chen") == []
