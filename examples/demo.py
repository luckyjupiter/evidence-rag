#!/usr/bin/env python3
"""Offline demo: ingest the sample corpus and show retrieval plus linked evidence.

Uses a deterministic extractor so no model, key, or network call is required.
The extractor only recognizes a few facts in the sample documents; replace it
with OpenAICompatibleExtractor or your own implementation for real data.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from evidence_rag import (
    Entity,
    Extraction,
    Relation,
    Store,
    ingest_file,
    keyword_search,
)
from evidence_rag.models import RelationHit


class DemoExtractor:
    """Tiny deterministic extractor for the bundled sample documents."""

    def extract(self, text: str) -> Extraction:
        entities: list[Entity] = []
        relations: list[Relation] = []

        if "Mira Chen designed the trial" in text:
            entities += [
                Entity("Mira Chen", "person"),
                Entity("North Orchard", "location"),
            ]
            relations.append(
                Relation(
                    "Mira Chen",
                    "designed",
                    "North Orchard",
                    "Mira Chen designed the trial",
                )
            )
        if "HG-14" in text:
            entities += [
                Entity("Harbor Point", "location"),
                Entity("HG-14", "product"),
            ]
            relations.append(
                Relation(
                    "Harbor Point",
                    "uses_sensor",
                    "HG-14",
                    "Harbor Point station uses a vented pressure sensor, model HG-14",
                )
            )
        if "Box C-14 contains Harbor Point" in text:
            entities += [
                Entity("Box C-14", "document"),
                Entity("Harbor Point", "location"),
            ]
            relations.append(
                Relation(
                    "Box C-14",
                    "contains",
                    "Harbor Point",
                    "Box C-14 contains Harbor Point calibration records for 2021 and 2022.",
                )
            )
        return Extraction(entities=entities, relations=relations)


def main() -> None:
    corpus = Path(__file__).resolve().parent / "corpus"
    with tempfile.TemporaryDirectory() as tmp:
        with Store(Path(tmp) / "demo.db") as store:
            for path in sorted(corpus.glob("*.md")):
                result = ingest_file(store, path, extractor=DemoExtractor())
                print(f"ingested {result.title}  chunks={result.chunks}")

            stats = store.stats()
            print(
                f"\nstats: {stats.documents} documents, {stats.chunks} chunks, "
                f"{stats.entities} entities, {stats.relations} relations"
            )

            question = "Why did Harbor Point read low in March?"
            print(f"\nquestion: {question}\n")
            for number, hit in enumerate(keyword_search(store, question, limit=2), start=1):
                print(f"[{number}] {hit.title}  chunk {hit.idx}  score {hit.score:.2f}")
                print(f"    {hit.source}  chars {hit.start}-{hit.end}")
                print(f"    {hit.text[:220]}...\n")

            relation: RelationHit
            print("linked evidence for 'Harbor Point':")
            for relation in store.relations_for("Harbor Point"):
                print(
                    f"  {relation.subject} --{relation.predicate}--> {relation.object}"
                )
                print(f"    {relation.evidence}")
                print(f"    {relation.source}  [chunk {relation.idx}]")


if __name__ == "__main__":
    main()
