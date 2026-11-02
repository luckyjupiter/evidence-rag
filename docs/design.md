# Worked example: retrieve a record and keep its evidence

## Problem

A document search result should show where it came from. If an extraction
model links a sensor to a station, the source chunk and its quoted evidence
should remain inspectable when someone asks why that link exists.

The bundled field notes are original fictional documents, not client records.
This walkthrough demonstrates the reusable engine, not QCC's proprietary system
or a production accuracy result.

## Decision

Keep documents, source offsets, FTS5 text search, graph records, and optional
vectors in SQLite. A source key determines the document ID; chunk text and index
determine chunk IDs. The default install can run without a model, server, API
key, or paid call. The tradeoff is literal keyword ranking and a brute-force
vector path rather than a large-corpus search service.

Extraction records retain the originating chunk, source, and extractor-supplied
quote. Those links make a claim inspectable; they do not prove it true. The
engine neither validates the model's reasoning nor generates an answer.

## Reproduce it

Follow the [README](../README.md#install), then run:

```sh
python examples/demo.py
python examples/evaluate.py
python -m pytest
```

The demo asks why Harbor Point read low in March. Keyword search retrieves the
calibration note, whose text identifies a blocked vent tube. An independent
graph query links Harbor Point to its sensor and archive record, with the
source quote beside each relation. The demo extractor recognizes a few known
facts; it is deliberately not presented as a general language model.

The [evaluation fixture](../examples/evaluation.json) tests seven keyword queries
and one disjoint-vocabulary control across three documents. Each positive case
requires both the correct source and a supporting quote inside a retrieved
chunk. At 500 characters with 80-character overlap, the current fixture gives
Hit@1 = 1.0, MRR@3 = 1.0, and an empty result for the control. These are small,
authored regression checks, not an external benchmark. The terms are chosen
from the corpus; they say nothing about general paraphrase performance.

## Updates and failure recovery

Changing a document or its chunk layout rebuilds chunks and removes stale
linked graph evidence. Extraction completion is independent of text ingestion:
adding extraction later and retrying a failed provider request both work on
unchanged documents. Completed empty extractions are cached. Versioned custom
extractor names and `--extract --reextract` support deliberate replacement.
The [graph tests](../tests/test_graph.py) reproduce these cases without a model.

Remaining limits include ambiguous names, unverified extraction quotes,
paraphrases, scanned PDFs, large corpora, access control, and retrieval quality
on real data. The included evaluation is intentionally separate from model
answer evaluation, following the distinction in
[Haystack's evaluation documentation](https://docs.haystack.deepset.ai/docs/evaluation).
