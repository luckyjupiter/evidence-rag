"""Evidence-linked retrieval and a small knowledge graph over your own documents."""

from .chunking import chunk_text
from .embeddings import HashingEmbedder, OpenAICompatibleEmbedder
from .extract import (
    Entity,
    Extraction,
    Relation,
    NoopExtractor,
    OpenAICompatibleExtractor,
    parse_extraction,
)
from .ingest import ingest_file, ingest_text, read_document
from .retrieval import keyword_search, vector_search
from .store import Store

__all__ = [
    "Entity",
    "Extraction",
    "HashingEmbedder",
    "NoopExtractor",
    "OpenAICompatibleEmbedder",
    "OpenAICompatibleExtractor",
    "Relation",
    "Store",
    "chunk_text",
    "ingest_file",
    "ingest_text",
    "keyword_search",
    "parse_extraction",
    "read_document",
    "vector_search",
]

__version__ = "0.1.3"
