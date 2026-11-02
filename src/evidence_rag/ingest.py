from __future__ import annotations

import json
import re
from pathlib import Path

from .chunking import chunk_text
from .embeddings import Embedder
from .extract import Extractor
from .models import IngestResult
from .store import Store

TEXT_SUFFIXES = {".md", ".markdown", ".txt"}


def _default_title(text: str, source: str) -> str:
    for line in text.splitlines()[:40]:
        stripped = line.strip()
        if not stripped:
            continue
        candidate = re.sub(r"^#+\s*", "", stripped).strip()
        if candidate:
            return candidate[:120]
    return Path(source).name or source


def read_document(path: str | Path) -> str:
    """Read a text or Markdown document. PDF requires the optional ``pypdf``."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ValueError("PDF support requires: pip install 'evidence-rag[pdf]'") from exc
        reader = PdfReader(str(path))
        pages = []
        for number, page in enumerate(reader.pages, start=1):
            pages.append(f"[page {number}]\n{page.extract_text() or ''}")
        return "\n\n".join(pages)
    if suffix and suffix not in TEXT_SUFFIXES:
        raise ValueError(f"unsupported document type: {suffix}")
    return path.read_text(encoding="utf-8")


def _snippet(text: str, name: str) -> str:
    lowered = text.casefold()
    pos = lowered.find(name.casefold())
    if pos == -1:
        return ""
    start = max(0, pos - 80)
    end = min(len(text), pos + len(name) + 80)
    return re.sub(r"\s+", " ", text[start:end]).strip()


def ingest_text(
    store: Store,
    text: str,
    *,
    source: str,
    title: str | None = None,
    chunk_chars: int = 2000,
    overlap: int = 200,
    extractor: Extractor | None = None,
    embedder: Embedder | None = None,
    reextract: bool = False,
) -> IngestResult:
    """Ingest text under a stable source key, returning what changed."""
    title = title or _default_title(text, source)
    spans = chunk_text(text, max_chars=chunk_chars, overlap=overlap)
    doc_id, unchanged = store.replace_document(
        source=source, title=title, text=text, spans=spans
    )
    chunks = store.chunk_ids(doc_id)

    if embedder is not None:
        already = store.embedded_ids(embedder.name)
        pending = [row for row in chunks if row["id"] not in already]
        if pending:
            vectors = embedder.embed([row["text"] for row in pending])
            if len(vectors) != len(pending):
                raise RuntimeError("embedder returned the wrong number of vectors")
            with store.conn:
                for row, vector in zip(pending, vectors):
                    store.set_embedding(row["id"], embedder.name, vector)

    if extractor is not None:
        # Custom extractors may supply a versioned name. Provider URL/model
        # distinguish different configurations without storing API keys.
        extractor_key = json.dumps([
            getattr(extractor, "name", None) or
            f"{type(extractor).__module__}.{type(extractor).__qualname__}",
            getattr(extractor, "base_url", None),
            getattr(extractor, "model", None),
        ])
        completed = set() if reextract else store.extracted_ids(extractor_key)
        pending = [row for row in chunks if row["id"] not in completed]
        extracted = [(row, extractor.extract(row["text"])) for row in pending]
        with store.conn:
            for row, result in extracted:
                store.clear_chunk_graph(row["id"])
                entity_ids: dict[str, str] = {}
                for entity in result.entities:
                    entity_ids[entity.name.casefold()] = store.upsert_entity(
                        entity.name, entity.type
                    )
                    store.add_mention(
                        row["id"], entity_ids[entity.name.casefold()], _snippet(row["text"], entity.name)
                    )
                for relation in result.relations:
                    subject_id = entity_ids.get(relation.subject.casefold())
                    if subject_id is None:
                        subject_id = store.upsert_entity(relation.subject, "other")
                        entity_ids[relation.subject.casefold()] = subject_id
                    object_id = entity_ids.get(relation.object.casefold())
                    if object_id is None:
                        object_id = store.upsert_entity(relation.object, "other")
                        entity_ids[relation.object.casefold()] = object_id
                    store.add_relation(
                        subject_id,
                        relation.predicate,
                        object_id,
                        row["id"],
                        relation.evidence,
                    )
                store.mark_extracted(row["id"], extractor_key)

    return IngestResult(
        doc_id=doc_id,
        source=source,
        title=title,
        chunks=len(chunks),
        unchanged=unchanged,
    )


def ingest_file(
    store: Store,
    path: str | Path,
    *,
    title: str | None = None,
    source: str | None = None,
    chunk_chars: int = 2000,
    overlap: int = 200,
    extractor: Extractor | None = None,
    embedder: Embedder | None = None,
    reextract: bool = False,
) -> IngestResult:
    path = Path(path)
    text = read_document(path)
    return ingest_text(
        store,
        text,
        source=source or str(path),
        title=title,
        chunk_chars=chunk_chars,
        overlap=overlap,
        extractor=extractor,
        embedder=embedder,
        reextract=reextract,
    )
