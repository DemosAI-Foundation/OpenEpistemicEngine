"""Optional local non-LLM retrieval: TF-IDF keyword index over a local corpus directory."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .rules import normalize

SUFFIXES = {".txt", ".md", ".markdown", ".json", ".csv", ".log", ".rst", ".tsv"}


def tokenize(text: str) -> list[str]:
    return [t for t in normalize(text).split() if len(t) > 2]


@dataclass
class Hit:
    source: str
    score: float
    text: str


@dataclass
class Retriever:
    corpus_dir: str = ""
    max_chunk: int = 1400
    ready: bool = False
    docs: list[tuple[str, str]] = ()
    vocab: dict[str, int] = ()
    matrix: object = None
    norms: object = None
    reason: str = ""

    def build(self) -> "Retriever":
        path = Path(self.corpus_dir) if self.corpus_dir else None
        if path is None or not path.is_dir():
            self.reason = "no local corpus directory configured"
            return self
        docs: list[tuple[str, str]] = []
        for file in sorted(path.rglob("*")):
            if file.suffix.lower() not in SUFFIXES or not file.is_file():
                continue
            try:
                raw = file.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for chunk in self._chunks(raw):
                docs.append((str(file), chunk))
        if not docs:
            self.reason = "corpus directory contains no readable documents"
            return self
        vocab: dict[str, int] = {}
        for _, chunk in docs:
            for tok in set(tokenize(chunk)):
                vocab[tok] = vocab.get(tok, 0) + 1
        index = {tok: i for i, tok in enumerate(sorted(vocab))}
        matrix = np.zeros((len(docs), len(index)), dtype=np.float32)
        for row, (_, chunk) in enumerate(docs):
            counts: dict[str, int] = {}
            for tok in tokenize(chunk):
                if tok in index:
                    counts[tok] = counts.get(tok, 0) + 1
            for tok, count in counts.items():
                idf = math.log((len(docs) + 1) / (vocab[tok] + 1)) + 1.0
                matrix[row, index[tok]] = (1.0 + math.log(count)) * idf
        norms = np.linalg.norm(matrix, axis=1) + 1e-9
        self.docs = docs
        self.vocab = index
        self.matrix = matrix
        self.norms = norms
        self.ready = True
        self.reason = ""
        return self

    def _chunks(self, raw: str) -> list[str]:
        raw = raw.strip()
        if not raw:
            return []
        parts = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
        if len(parts) == 1:
            parts = re.findall(r".{1,%d}(?:\s|$)" % self.max_chunk, raw, flags=re.S)
        out: list[str] = []
        buf = ""
        for part in parts:
            while len(part) > self.max_chunk:
                if buf:
                    out.append(buf)
                    buf = ""
                out.append(part[: self.max_chunk])
                part = part[self.max_chunk :]
            if len(buf) + len(part) > self.max_chunk:
                if buf:
                    out.append(buf)
                buf = part
            else:
                buf = f"{buf}\n{part}".strip()
        if buf:
            out.append(buf)
        return out[:1200]

    def search(self, query: str, k: int = 8) -> list[Hit]:
        if not self.ready:
            return []
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        for tok in tokenize(query):
            idx = self.vocab.get(tok)
            if idx is not None:
                vec[idx] += 1.0
        if not vec.any():
            return []
        scores = (self.matrix @ vec) / (self.norms * (np.linalg.norm(vec) + 1e-9))
        order = np.argsort(-scores)[:k]
        return [
            Hit(source=self.docs[int(i)][0], score=round(float(scores[int(i)]), 3), text=self.docs[int(i)][1])
            for i in order
            if scores[int(i)] > 0.02
        ]

    def context_block(self, query: str, k: int = 8) -> str:
        hits = self.search(query, k)
        if not hits:
            return "RETRIEVAL: none available. Internal knowledge only; lower evidence quality accordingly."
        lines = ["RETRIEVAL (local index, non-LLM):"]
        for hit in hits:
            lines.append(f"[{hit.source} | score {hit.score}] {hit.text}")
        return "\n".join(lines)

    def status(self) -> dict:
        return {
            "available": self.ready,
            "corpus_dir": self.corpus_dir,
            "documents": len(self.docs),
            "reason": self.reason,
        }
