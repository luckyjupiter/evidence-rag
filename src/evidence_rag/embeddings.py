from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.error
import urllib.request
from typing import Protocol, Sequence

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Deterministic bag-of-words baseline.

    This is not a semantic model. It exists so vector storage and ranking can be
    exercised offline and in tests without a provider or API key.
    """

    name = "hashing-256-v1"

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dim
            for token in _TOKEN.findall(text.casefold()):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] & 1 else -1.0
                vector[bucket] += sign
            norm = math.sqrt(sum(value * value for value in vector))
            vectors.append([value / norm for value in vector] if norm else vector)
        return vectors


class OpenAICompatibleEmbedder:
    """Embedding client for any OpenAI-compatible ``/embeddings`` endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.name = f"openai-compatible:{model}"

    @classmethod
    def from_env(cls) -> "OpenAICompatibleEmbedder":
        base_url = os.environ.get("EVIDENCE_RAG_EMBED_URL")
        model = os.environ.get("EVIDENCE_RAG_EMBED_MODEL")
        if not base_url or not model:
            raise ValueError(
                "Set EVIDENCE_RAG_EMBED_URL and EVIDENCE_RAG_EMBED_MODEL to use vector search"
            )
        return cls(base_url, model, os.environ.get("EVIDENCE_RAG_EMBED_KEY"))

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        body = json.dumps({"model": self.model, "input": list(texts)}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}/embeddings", data=body, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"embedding request failed: {exc}") from exc

        try:
            return [list(map(float, row["embedding"])) for row in payload["data"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("embedding response did not contain vectors") from exc
