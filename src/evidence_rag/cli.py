from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from .embeddings import OpenAICompatibleEmbedder
from .extract import OpenAICompatibleExtractor
from .ingest import ingest_file
from .retrieval import keyword_search, vector_search
from .store import Store

DEFAULT_DB = "evidence-rag.db"


def _db_path(value: str | None) -> str:
    return value or os.environ.get("EVIDENCE_RAG_DB") or DEFAULT_DB


def _print_hits(hits) -> None:
    if not hits:
        print("(no matching evidence)")
        return
    for number, hit in enumerate(hits, start=1):
        print(f"{number}. {hit.title}  [chunk {hit.idx}, score {hit.score:.3f}]")
        print(f"   {hit.source}  chars {hit.start}-{hit.end}")
        print(f"   {hit.text[:400]}")
        print()


def _print_evidence(rows) -> None:
    if not rows:
        print("(no linked evidence)")
        return
    for row in rows:
        print(f"{row.subject} --{row.predicate}--> {row.object}")
        if row.evidence:
            print(f"   evidence: {row.evidence}")
        print(f"   {row.source}  [chunk {row.idx}]")
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="evidence-rag",
        description="Retrieval and evidence-linked knowledge graph over your documents.",
    )
    parser.add_argument("--db", help=f"SQLite database (default: {DEFAULT_DB})")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="ingest text/Markdown documents")
    ingest.add_argument("paths", nargs="+")
    ingest.add_argument("--title", help="override title (single document only)")
    ingest.add_argument("--chunk-chars", type=int, default=2000)
    ingest.add_argument("--overlap", type=int, default=200)
    ingest.add_argument(
        "--extract",
        action="store_true",
        help="extract entities and relations with the configured LLM endpoint",
    )
    ingest.add_argument(
        "--embed",
        action="store_true",
        help="store embeddings with the configured embedding endpoint",
    )

    search = sub.add_parser("search", help="retrieve evidence for a query")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=8)
    search.add_argument("--vector", action="store_true")
    search.add_argument("--json", action="store_true")

    entities = sub.add_parser("entities", help="find extracted entities")
    entities.add_argument("query")
    entities.add_argument("--limit", type=int, default=20)
    entities.add_argument("--json", action="store_true")

    evidence = sub.add_parser("evidence", help="show relations linked to an entity")
    evidence.add_argument("name")
    evidence.add_argument("--limit", type=int, default=50)
    evidence.add_argument("--json", action="store_true")

    stats = sub.add_parser("stats", help="show database counts")
    stats.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    store = Store(_db_path(args.db))

    try:
        if args.command == "ingest":
            if args.title and len(args.paths) != 1:
                parser.error("--title can only be used with one document")
            extractor = OpenAICompatibleExtractor.from_env() if args.extract else None
            embedder = OpenAICompatibleEmbedder.from_env() if args.embed else None
            for path in args.paths:
                if not Path(path).exists():
                    print(f"missing: {path}", file=sys.stderr)
                    return 2
                result = ingest_file(
                    store,
                    path,
                    title=args.title,
                    chunk_chars=args.chunk_chars,
                    overlap=args.overlap,
                    extractor=extractor,
                    embedder=embedder,
                )
                state = "unchanged" if result.unchanged else "ingested"
                print(f"{state}: {result.source}  chunks={result.chunks}  title={result.title!r}")
            return 0

        if args.command == "search":
            if args.vector:
                embedder = OpenAICompatibleEmbedder.from_env()
                hits = vector_search(store, args.query, embedder, args.limit)
            else:
                hits = keyword_search(store, args.query, args.limit)
            if args.json:
                print(json.dumps([asdict(hit) for hit in hits], indent=2))
            else:
                _print_hits(hits)
            return 0

        if args.command == "entities":
            rows = store.find_entities(args.query, args.limit)
            if args.json:
                print(json.dumps([asdict(row) for row in rows], indent=2))
            elif not rows:
                print("(no entities)")
            else:
                for row in rows:
                    print(f"{row.name}  [{row.type}]  mentions={row.mentions}")
            return 0

        if args.command == "evidence":
            rows = store.relations_for(args.name, args.limit)
            if args.json:
                print(json.dumps([asdict(row) for row in rows], indent=2))
            else:
                _print_evidence(rows)
            return 0

        if args.command == "stats":
            stats_data = store.stats()
            if args.json:
                print(json.dumps(asdict(stats_data), indent=2))
            else:
                for key, value in asdict(stats_data).items():
                    print(f"{key:12} {value}")
            return 0
    finally:
        store.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
