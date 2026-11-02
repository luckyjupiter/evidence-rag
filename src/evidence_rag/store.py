from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .models import ChunkSpan, EntityHit, RelationHit, SearchHit, Stats

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    source      TEXT NOT NULL UNIQUE,
    title       TEXT NOT NULL,
    sha256      TEXT NOT NULL,
    bytes       INTEGER NOT NULL,
    fetched_at  TEXT NOT NULL,
    text        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id          TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    idx         INTEGER NOT NULL,
    text        TEXT NOT NULL,
    start       INTEGER NOT NULL,
    end         INTEGER NOT NULL,
    sha256      TEXT NOT NULL,
    UNIQUE (doc_id, idx)
);

CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    text,
    tokenize = 'unicode61'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts (chunk_id, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    DELETE FROM chunks_fts WHERE chunk_id = old.id;
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    DELETE FROM chunks_fts WHERE chunk_id = old.id;
    INSERT INTO chunks_fts (chunk_id, text) VALUES (new.id, new.text);
END;

CREATE TABLE IF NOT EXISTS entities (
    id    TEXT PRIMARY KEY,
    name  TEXT NOT NULL UNIQUE COLLATE NOCASE,
    type  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS mentions (
    chunk_id  TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    snippet   TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (chunk_id, entity_id)
);

CREATE TABLE IF NOT EXISTS relations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    predicate  TEXT NOT NULL,
    object_id  TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    chunk_id   TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    evidence   TEXT NOT NULL DEFAULT '',
    UNIQUE (subject_id, predicate, object_id, chunk_id)
);

CREATE INDEX IF NOT EXISTS relations_subject ON relations(subject_id);
CREATE INDEX IF NOT EXISTS relations_object ON relations(object_id);

CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    model    TEXT NOT NULL,
    dim      INTEGER NOT NULL,
    vector   BLOB NOT NULL
);
"""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def document_id(source: str) -> str:
    """Stable document identity: the source string, not the current content."""
    return _sha256(source)


def chunk_id(doc_id: str, idx: int, text: str) -> str:
    return _sha256(f"{doc_id}\x00{idx}\x00{text}")


def _entity_id(name: str) -> str:
    return _sha256(name.casefold())


class Store:
    """SQLite-backed documents, chunks, entities, relations, and vectors."""

    def __init__(self, path: str | Path = "evidence-rag.db") -> None:
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        if self.path != ":memory:":
            self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---------------------------------------------------------------- ingest
    def get_document(self, source: str) -> sqlite3.Row | None:
        cur = self.conn.execute("SELECT * FROM documents WHERE source = ?", (source,))
        return cur.fetchone()

    def replace_document(
        self,
        *,
        source: str,
        title: str,
        text: str,
        spans: Sequence[ChunkSpan],
    ) -> tuple[str, bool]:
        """Insert or replace a document and its chunks.

        Returns ``(doc_id, unchanged)``. Re-ingesting identical content is a
        no-op; changed content replaces the document's chunks and any graph
        evidence attached to them.
        """
        digest = _sha256(text)
        doc_id = document_id(source)
        existing = self.get_document(source)
        if existing is not None and existing["sha256"] == digest:
            return doc_id, True

        now = datetime.now(timezone.utc).isoformat()
        with self.conn:
            if existing is not None:
                self.conn.execute("DELETE FROM documents WHERE id = ?", (existing["id"],))
            self.conn.execute(
                """
                INSERT INTO documents (id, source, title, sha256, bytes, fetched_at, text)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (doc_id, source, title, digest, len(text.encode("utf-8")), now, text),
            )
            for span in spans:
                self.conn.execute(
                    """
                    INSERT INTO chunks (id, doc_id, idx, text, start, end, sha256)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk_id(doc_id, span.idx, span.text),
                        doc_id,
                        span.idx,
                        span.text,
                        span.start,
                        span.end,
                        _sha256(span.text),
                    ),
                )
        return doc_id, False

    def chunk_ids(self, doc_id: str) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                "SELECT id, idx, text FROM chunks WHERE doc_id = ? ORDER BY idx",
                (doc_id,),
            )
        )

    # ----------------------------------------------------------------- graph
    def upsert_entity(self, name: str, type_: str) -> str:
        entity_id = _entity_id(name)
        self.conn.execute(
            """
            INSERT INTO entities (id, name, type) VALUES (?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET type = excluded.type
            """,
            (entity_id, name, type_),
        )
        return entity_id

    def add_mention(self, chunk: str, entity_id: str, snippet: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO mentions (chunk_id, entity_id, snippet) VALUES (?, ?, ?)",
            (chunk, entity_id, snippet[:500]),
        )

    def add_relation(
        self,
        subject_id: str,
        predicate: str,
        object_id: str,
        chunk: str,
        evidence: str,
    ) -> None:
        self.conn.execute(
            """
            INSERT OR IGNORE INTO relations
                (subject_id, predicate, object_id, chunk_id, evidence)
            VALUES (?, ?, ?, ?, ?)
            """,
            (subject_id, predicate, object_id, chunk, evidence[:500]),
        )

    def commit(self) -> None:
        self.conn.commit()

    # -------------------------------------------------------------- retrieval
    def search_keyword(self, match_query: str, limit: int) -> list[SearchHit]:
        rows = self.conn.execute(
            """
            SELECT c.id, c.doc_id, d.title, d.source, c.idx, c.text, c.start, c.end,
                   bm25(chunks_fts) AS rank
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.chunk_id
            JOIN documents d ON d.id = c.doc_id
            WHERE chunks_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (match_query, limit),
        ).fetchall()
        return [
            SearchHit(
                chunk_id=r["id"],
                doc_id=r["doc_id"],
                title=r["title"],
                source=r["source"],
                idx=r["idx"],
                text=r["text"],
                start=r["start"],
                end=r["end"],
                score=-float(r["rank"]),
            )
            for r in rows
        ]

    def embedded_chunks(self, model: str) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                """
                SELECT e.chunk_id, e.dim, e.vector, c.doc_id, d.title, d.source,
                       c.idx, c.text, c.start, c.end
                FROM embeddings e
                JOIN chunks c ON c.id = e.chunk_id
                JOIN documents d ON d.id = c.doc_id
                WHERE e.model = ?
                """,
                (model,),
            )
        )

    def embedded_ids(self, model: str) -> set[str]:
        return {
            row[0]
            for row in self.conn.execute(
                "SELECT chunk_id FROM embeddings WHERE model = ?", (model,)
            )
        }

    def set_embedding(self, chunk: str, model: str, vector: Sequence[float]) -> None:
        import struct

        blob = struct.pack(f"<{len(vector)}f", *vector)
        self.conn.execute(
            """
            INSERT INTO embeddings (chunk_id, model, dim, vector) VALUES (?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                model = excluded.model, dim = excluded.dim, vector = excluded.vector
            """,
            (chunk, model, len(vector), blob),
        )

    # ------------------------------------------------------------------ graph
    def find_entities(self, text: str, limit: int = 20) -> list[EntityHit]:
        rows = self.conn.execute(
            """
            SELECT e.name, e.type, COUNT(m.chunk_id) AS mentions
            FROM entities e
            LEFT JOIN mentions m ON m.entity_id = e.id
            WHERE e.name LIKE ?
            GROUP BY e.id
            ORDER BY mentions DESC, e.name
            LIMIT ?
            """,
            (f"%{text}%", limit),
        ).fetchall()
        return [
            EntityHit(name=r["name"], type=r["type"], mentions=r["mentions"])
            for r in rows
        ]

    def relations_for(self, name: str, limit: int = 50) -> list[RelationHit]:
        rows = self.conn.execute(
            """
            SELECT s.name AS subject, r.predicate, o.name AS object, r.evidence,
                   r.chunk_id, d.source, d.title, c.idx
            FROM relations r
            JOIN entities s ON s.id = r.subject_id
            JOIN entities o ON o.id = r.object_id
            JOIN chunks c ON c.id = r.chunk_id
            JOIN documents d ON d.id = c.doc_id
            WHERE s.name = ? COLLATE NOCASE OR o.name = ? COLLATE NOCASE
            ORDER BY d.title, c.idx
            LIMIT ?
            """,
            (name, name, limit),
        ).fetchall()
        return [
            RelationHit(
                subject=r["subject"],
                predicate=r["predicate"],
                object=r["object"],
                evidence=r["evidence"],
                chunk_id=r["chunk_id"],
                source=r["source"],
                title=r["title"],
                idx=r["idx"],
            )
            for r in rows
        ]

    def stats(self) -> Stats:
        def count(table: str) -> int:
            return int(self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

        return Stats(
            documents=count("documents"),
            chunks=count("chunks"),
            entities=count("entities"),
            relations=count("relations"),
            embeddings=count("embeddings"),
        )

    def dump_manifest(self) -> str:
        """Small JSON inventory useful for backups and test assertions."""
        rows = self.conn.execute(
            "SELECT id, source, title, sha256 FROM documents ORDER BY source"
        ).fetchall()
        return json.dumps([dict(r) for r in rows], indent=2, sort_keys=True)
