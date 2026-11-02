from __future__ import annotations

import pytest

from evidence_rag.chunking import chunk_text


def test_offsets_trace_back_to_source():
    text = "First paragraph.\n\nSecond paragraph with more words.\n\nThird."
    spans = chunk_text(text, max_chars=30, overlap=5)
    assert spans
    for span in spans:
        assert text[span.start : span.end] == span.text
        assert span.text.strip() == span.text
    assert [span.idx for span in spans] == list(range(len(spans)))


def test_prefers_paragraph_boundaries():
    text = "A" * 50 + "\n\n" + "B" * 50
    spans = chunk_text(text, max_chars=80, overlap=10)
    assert spans[0].text == "A" * 50


def test_empty_and_whitespace():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        chunk_text("text", max_chars=0)
    with pytest.raises(ValueError):
        chunk_text("text", max_chars=10, overlap=10)
