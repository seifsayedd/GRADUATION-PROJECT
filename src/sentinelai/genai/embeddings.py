"""Text embeddings behind one interface.

* ``MiniLMEmbedder``: ``sentence-transformers/all-MiniLM-L6-v2`` (384 dimensions), run locally.
  The model is downloaded once into the Hugging Face cache.
* ``HashEmbedder``: deterministic token hashing, no download. Weak retrieval; meant for tests.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

_TOKEN = re.compile(r"[a-z0-9_]+")


class HashEmbedder:
    name = "hash"

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for tok in _TOKEN.findall(t.lower()):
                h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little")
                out[i, h % self.dim] += 1.0 if (h >> 32) & 1 else -1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(n == 0, 1, n)


class MiniLMEmbedder:
    name = "minilm"
    model_id = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, dim: int = 384, device: str | None = None):
        self.dim, self.device, self._m = dim, device, None

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._m is None:
            from sentence_transformers import SentenceTransformer
            self._m = SentenceTransformer(self.model_id, device=self.device)
            got = self._m.get_sentence_embedding_dimension()
            if got != self.dim:
                raise ValueError(f"{self.model_id} gives {got} dimensions, settings say {self.dim}")
        return np.asarray(self._m.encode(texts, normalize_embeddings=True, batch_size=64), dtype="float32")


def get_embedder(settings) -> HashEmbedder | MiniLMEmbedder:
    if settings.embedder == "hash":
        return HashEmbedder(settings.embed_dim)
    if settings.embedder == "minilm":
        return MiniLMEmbedder(settings.embed_dim)
    raise ValueError(f"unknown embedder {settings.embedder!r} (minilm | hash)")
