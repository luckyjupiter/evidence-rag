#!/usr/bin/env python3
"""Small, authored retrieval regression set; not a production quality benchmark.

Checks both the expected source and a supporting quote inside a returned chunk.
No LLM judge, model, network, or paid calls. Run from a source checkout.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from evidence_rag import Store, ingest_file, keyword_search


def evaluate() -> dict[str, float | int]:
    examples = Path(__file__).resolve().parent
    cases = json.loads((examples / "evaluation.json").read_text(encoding="utf-8"))
    answerable = top_one = 0
    reciprocal_rank = 0.0
    negative_controls = negative_passed = 0
    with tempfile.TemporaryDirectory() as directory:
        with Store(Path(directory) / "evaluation.db") as store:
            for path in sorted((examples / "corpus").glob("*.md")):
                ingest_file(store, path, source=path.name, chunk_chars=500, overlap=80)
            for case in cases:
                hits = keyword_search(store, case["query"], limit=3)
                if case["source"] is None:
                    negative_controls += 1
                    negative_passed += not hits
                    print(f"{'PASS' if not hits else 'FAIL'} negative control: {case['query']}")
                    continue
                answerable += 1
                ranks = [rank for rank, hit in enumerate(hits, start=1)
                         if hit.source == case["source"] and case["quote"] in hit.text]
                rank = min(ranks) if ranks else None
                top_one += rank == 1
                reciprocal_rank += 1 / rank if rank else 0
                print(f"rank={rank or 'miss'}  {case['query']}")
    return {
        "answerable_queries": answerable,
        "hit_at_1": top_one / answerable,
        "mrr_at_3": reciprocal_rank / answerable,
        "negative_controls": negative_controls,
        "negative_controls_passed": negative_passed,
    }


if __name__ == "__main__":
    metrics = evaluate()
    print(json.dumps(metrics, indent=2))
    # This is a fixed lexical regression fixture, so fail CI on a changed
    # baseline rather than printing a reassuring report and exiting success.
    if (metrics["hit_at_1"] != 1.0 or
            metrics["negative_controls_passed"] != metrics["negative_controls"]):
        raise SystemExit(1)
