"""Dense channel for hybrid retrieval. Local only.

Priority: embeddings from the *same local server* that serves the LLM
(/api/embed, /v1/embeddings). If unavailable, a stateless signed hashing
embedder (word stems + char trigrams, sublinear TF, L2 norm) provides a
deterministic second channel with no dependencies and no network.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .textutil import tokenize

DIM = 256
CHAR_NGRAMS = (3, 4)


def _hash_index(feature: str, dim: int = DIM) -> int:
    h = int(hashlib.blake2b(feature.encode("utf-8"), digest_size=8).hexdigest(), 16)
    return h % dim


def _sign(feature: str) -> float:
    h = int(hashlib.blake2b((feature + "#s").encode("utf-8"), digest_size=8).hexdigest(), 16)
    return 1.0 if h % 2 == 0 else -1.0


def _char_ngrams(word: str) -> list[str]:
    padded = f"^{word}$"
    out = []
    for n in CHAR_NGRAMS:
        if len(padded) >= n:
            out += [f"c{n}:{padded[i:i + n]}" for i in range(len(padded) - n + 1)]
    return out


class HashEmbedder:
    """Stateless signed feature-hashing embedder (deterministic)."""

    mode = "hash-local"

    def __init__(self, dim: int = DIM):
        self.dim = dim

    def encode_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        words = tokenize(text, stem_tokens=True)
        for w in words:
            vec[_hash_index("w:" + w, self.dim)] += _sign("w:" + w)
        for i in range(len(words) - 1):
            bg = f"b:{words[i]}_{words[i + 1]}"
            vec[_hash_index(bg, self.dim)] += _sign(bg) * 0.6
        for w in words:
            for ng in _char_ngrams(w):
                vec[_hash_index(ng, self.dim)] += _sign(ng) * 0.25
        sub = np.sign(vec) * np.log1p(np.abs(vec))
        norm = float(np.linalg.norm(sub))
        return sub / norm if norm else sub

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self.encode_one(t) for t in texts])


class Embedder:
    def __init__(self, provider=None, dim: int = DIM):
        self.provider = provider
        self.fallback = HashEmbedder(dim)
        self.last_mode = "unset"
        self.provider_failures = 0

    def encode(self, texts: list[str]) -> tuple[np.ndarray, str]:
        if not texts:
            return np.zeros((0, self.fallback.dim), dtype=np.float32), self.fallback.mode
        if self.provider is not None and self.provider_failures < 2:
            try:
                rows = self.provider.embed(texts)
                if rows and len(rows) == len(texts):
                    mat = np.asarray(rows, dtype=np.float32)
                    norms = np.linalg.norm(mat, axis=1, keepdims=True)
                    mat = mat / np.where(norms == 0, 1.0, norms)
                    self.last_mode = "provider"
                    return mat, "provider"
            except Exception:
                self.provider_failures += 1
        from . import semantic
        rows = semantic.embed_texts(texts)
        if rows and len(rows) == len(texts):
            mat = np.asarray(rows, dtype=np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            mat = mat / np.where(norms == 0, 1.0, norms)
            self.last_mode = "semantic-model"
            return mat, "semantic-model"
        mat = self.fallback.encode(texts)
        self.last_mode = self.fallback.mode
        return mat, self.last_mode

    def encode_one(self, text: str) -> tuple[np.ndarray, str]:
        mat, mode = self.encode([text])
        return mat[0], mode
