from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol

ENTITY_TYPES = {
    "person",
    "organization",
    "location",
    "method",
    "product",
    "event",
    "document",
    "date",
    "concept",
    "other",
}

EXTRACTION_PROMPT = """Extract entities and relations from the TEXT below.

Return ONLY a JSON object with this shape:
{
  "entities": [{"name": "...", "type": "person|organization|location|method|product|event|document|date|concept|other"}],
  "relations": [{"subject": "...", "predicate": "short_snake_case", "object": "...", "evidence": "verbatim quote under 300 characters"}]
}

Rules:
- Use names exactly as they appear when possible.
- Only emit a relation when both subject and object appear in the text.
- Keep evidence as a short quote from the text.
- Return empty lists when nothing is worth extracting.

TEXT:
"""


@dataclass(frozen=True)
class Entity:
    name: str
    type: str


@dataclass(frozen=True)
class Relation:
    subject: str
    predicate: str
    object: str
    evidence: str


@dataclass
class Extraction:
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)


class Extractor(Protocol):
    def extract(self, text: str) -> Extraction: ...


def _normalize_name(value: object) -> str:
    name = re.sub(r"\s+", " ", str(value or "")).strip()
    return name[:200]


def _normalize_type(value: object) -> str:
    type_ = str(value or "other").strip().lower().replace(" ", "_")
    return type_ if type_ in ENTITY_TYPES else "other"


def _normalize_predicate(value: object) -> str:
    predicate = re.sub(r"\s+", "_", str(value or "").strip().lower())
    predicate = re.sub(r"[^a-z0-9_]", "", predicate)
    return predicate[:64]


def _json_object(raw: str) -> dict | None:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for pos in range(start, len(text)):
        char = text[pos]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : pos + 1])
                except json.JSONDecodeError:
                    return None
    return None


def parse_extraction(payload: dict | str | None) -> Extraction:
    """Validate and normalize provider output. Malformed input yields empty results."""
    data = _json_object(payload) if isinstance(payload, str) else payload
    if not isinstance(data, dict):
        return Extraction()

    entities: dict[str, Entity] = {}
    raw_entities = data.get("entities", [])
    if isinstance(raw_entities, list):
        for item in raw_entities[:500]:
            if not isinstance(item, dict):
                continue
            name = _normalize_name(item.get("name"))
            if not name:
                continue
            entity = Entity(name=name, type=_normalize_type(item.get("type")))
            entities.setdefault(name.casefold(), entity)

    relations: list[Relation] = []
    raw_relations = data.get("relations", [])
    if isinstance(raw_relations, list):
        for item in raw_relations[:500]:
            if not isinstance(item, dict):
                continue
            subject = _normalize_name(item.get("subject"))
            predicate = _normalize_predicate(item.get("predicate"))
            object_ = _normalize_name(item.get("object"))
            if not (subject and predicate and object_):
                continue
            if subject.casefold() == object_.casefold():
                continue
            evidence = re.sub(r"\s+", " ", str(item.get("evidence") or "")).strip()[:500]
            relations.append(
                Relation(
                    subject=subject,
                    predicate=predicate,
                    object=object_,
                    evidence=evidence,
                )
            )

    return Extraction(entities=list(entities.values()), relations=relations)


class NoopExtractor:
    """Default extractor: stores text and retrieval data without calling a model."""

    def extract(self, text: str) -> Extraction:
        return Extraction()


class OpenAICompatibleExtractor:
    """Minimal extractor for any OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "OpenAICompatibleExtractor":
        base_url = os.environ.get("EVIDENCE_RAG_LLM_URL")
        model = os.environ.get("EVIDENCE_RAG_LLM_MODEL")
        if not base_url or not model:
            raise ValueError(
                "Set EVIDENCE_RAG_LLM_URL and EVIDENCE_RAG_LLM_MODEL to use extraction"
            )
        return cls(base_url, model, os.environ.get("EVIDENCE_RAG_LLM_KEY"))

    def extract(self, text: str) -> Extraction:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": EXTRACTION_PROMPT + text},
                ],
                "temperature": 0,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"extraction request failed: {exc}") from exc

        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("extraction response did not contain message content") from exc
        return parse_extraction(content)
