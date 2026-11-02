# evidence-rag

[![CI](https://github.com/luckyjupiter/evidence-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/luckyjupiter/evidence-rag/actions/workflows/ci.yml)

Retrieval over your own documents with an evidence-linked knowledge graph.
Documents and chunks get stable IDs; extracted entities and relations keep the
chunk and quote they came from, so a claim can be traced back to its source.

The package is a library and CLI. It needs no service, model, or API key to
store documents, chunk them, and run keyword retrieval. LLM extraction and
embedding search are optional adapters.

Use it when a small document collection needs inspectable retrieval and
source-linked graph records without a hosted service. It is the retrieval layer
of a RAG application; answer generation is separate. [Worked example and design](docs/design.md).

## Install

```sh
git clone https://github.com/luckyjupiter/evidence-rag.git
cd evidence-rag
python -m venv .venv
# Windows: .venv\Scripts\activate
. .venv/bin/activate
python -m pip install -e .
# optional PDF support
python -m pip install -e '.[pdf]'
```

Python 3.10+. SQLite with FTS5 (included in normal CPython builds).
Install from this GitHub checkout; this release is not published on PyPI.

## Quickstart

```sh
evidence-rag ingest examples/corpus/*.md
evidence-rag search "why did Harbor Point read low in March"
```

Search output includes the source, character offsets, chunk index, and score:

```text
1. Tide Gauge Calibration Notes: Harbor Point  [chunk 0, score 0.520]
   examples/corpus/harbor-point-calibration.md  chars 0-1372
   # Tide Gauge Calibration Notes: Harbor Point
   ...
   On 3 March 2023 the station read 42 millimetres below the reference staff. The
   technician found a partial blockage in the vent tube, likely salt creep.
```

Retrieval is keyword-based (SQLite FTS5 with BM25 ranking). The query is treated
as terms joined with AND first, then retried with OR if there are no matches.
This is deliberate and inspectable, not a claim of semantic search.

## Evidence-linked graph

`examples/demo.py` uses a small deterministic extractor so the demo runs
offline. It shows the shape of the result:

```sh
python examples/demo.py
```

```text
ingested Cedar Room Archive Inventory  chunks=1
ingested Tide Gauge Calibration Notes: Harbor Point  chunks=1
ingested North Orchard Drip Irrigation Trial  chunks=1

stats: 3 documents, 3 chunks, 5 entities, 3 relations

question: Why did Harbor Point read low in March?

[1] Tide Gauge Calibration Notes: Harbor Point  chunk 0  score 0.52
    examples/corpus/harbor-point-calibration.md  chars 0-1372
    ...On 3 March 2023 the station read 42 millimetres below the reference staff...

linked evidence for 'Harbor Point':
  Box C-14 --contains--> Harbor Point
    Box C-14 contains Harbor Point calibration records for 2021 and 2022.
    examples/corpus/cedar-room-inventory.md  [chunk 0]
  Harbor Point --uses_sensor--> HG-14
    Harbor Point station uses a vented pressure sensor, model HG-14
    examples/corpus/harbor-point-calibration.md  [chunk 0]
```

In code:

```python
from evidence_rag import Store, ingest_file, keyword_search

with Store("notes.db") as store:
    ingest_file(store, "notes.md")
    for hit in keyword_search(store, "vent tube blockage", limit=5):
        print(hit.title, hit.idx, hit.start, hit.end)
```

## Optional model adapters

Extraction and embeddings use any OpenAI-compatible HTTP endpoint. Nothing is
called unless you ask for it.

```sh
export EVIDENCE_RAG_LLM_URL=http://localhost:8000/v1
export EVIDENCE_RAG_LLM_MODEL=your-model
export EVIDENCE_RAG_EMBED_URL=http://localhost:8000/v1
export EVIDENCE_RAG_EMBED_MODEL=your-embedding-model

evidence-rag ingest notes.md --extract --embed
evidence-rag search "vent tube" --vector
evidence-rag evidence "Harbor Point"
```

`evidence-rag ingest` is idempotent: identical content under the same source is
unchanged when the title and resulting chunk layout match too. Changed content
or chunk settings replace that document’s chunks and their graph evidence.
Extraction completion is tracked per chunk and extractor configuration, so you
can add `--extract` later or retry after an extraction request fails. Successful
empty extraction is cached too. Use `--extract --reextract` to replace graph
records even when the source and extractor are unchanged. Custom extractors may
provide a versioned `name` for cache invalidation. Entities are deduplicated by
case-folded name; this is name matching, not entity disambiguation.

## How it works

- **Document ID** = SHA-256 of the source key. The key is stable across content
  edits; the content hash is stored separately.
- **Chunk ID** = SHA-256 of document ID, chunk index, and chunk text.
- **Chunks** are character windows with paragraph/sentence-aware breaks and
  configurable overlap. Offsets refer to the original text.
- **Retrieval** is FTS5 `bm25`; scores are reported as `-bm25` (higher is
  better).
- **Graph** tables (`entities`, `mentions`, `relations`) reference the chunk
  that produced them. Relations carry the extractor's evidence quote.
- **Vectors** are stored per chunk in SQLite and searched by brute-force cosine
  similarity. That is fine for a local corpus and honest about its limits.

Schema: `documents → chunks → mentions → entities`, plus `relations` and
`embeddings`. Deleting or replacing a document cascades to its chunks,
mentions, relations, and vectors.

## Tests

```sh
python -m pip install -e '.[dev]'
python -m pytest
```

The suite covers repeat ingestion, changed documents, source traceability,
keyword and vector ranking, malformed extraction output, and evidence links.
Tests use no network and no paid calls.

## Evaluate the example

```sh
python examples/evaluate.py
```

The authored three-document fixture contains seven answerable keyword queries
and one disjoint-vocabulary negative control. With 500-character chunks and
80-character overlap, all seven put a chunk with the expected source **and
supporting quote** at rank 1; the control returns no result. The script prints
Hit@1 and MRR@3 and runs in CI. This tiny, deliberately lexical regression set
is not an external benchmark or evidence of production retrieval accuracy.
Paraphrase, ambiguous questions, scale, and model extraction need evaluation on
your own corpus. See [the worked example](docs/design.md).

## Limits and non-goals

- Keyword retrieval is literal; synonyms and paraphrase need the vector adapter
  or a better retriever.
- Vector search is brute-force and single-model.
- The package retrieves evidence. It does not generate an answer and does not
  verify an extractor's claims. A generated answer would be a separate layer,
  and the evidence must remain visible.
- Extraction quality depends entirely on the configured model; malformed output
  is dropped rather than guessed at.
- PDF extraction needs `pypdf` and is not tested against scanned documents.
- This is not an access-control or multi-tenant boundary.

## Help and contributions

Report reproducible bugs in [GitHub Issues](https://github.com/luckyjupiter/evidence-rag/issues).
See [CONTRIBUTING](CONTRIBUTING.md) for development and [SECURITY](SECURITY.md)
for private vulnerability reports.

## License

MIT. The bundled sample corpus is original text released under CC0 1.0.
