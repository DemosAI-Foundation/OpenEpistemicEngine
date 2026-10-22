"""Phase 5 post-generation scrubber: deterministic Python, no LLM.

1. Meta-Instruction Scrubber (regex): delete leaked prompt artifacts directed at the model, not the user.
2. Semantic Deduplication: TF-IDF cosine similarity over paragraphs; above the threshold, the later paragraph goes.
3. Format Normalization: fix concatenated labels and ensure bullet points render as bullets.
"""

from __future__ import annotations

import math
import re

import numpy as np

from .rules import STOPWORDS, normalize

DEDUPE_THRESHOLD = 0.85

# Any instruction directed at the model, not the user, is a leak and gets deleted.
META_INSTRUCTION_PATTERNS = [
    r"^\s*(?:[-*>#\s]*)?The answer must include\b[^\n]*$",
    r"^\s*(?:[-*>#\s]*)?Label as weak or rejected\b[^\n]*$",
    r"^\s*(?:[-*>#\s]*)?(?:Step|Fase)\s*\d{1,2}\s*[.):][^\n]*$",
    r"^\s*(?:[-*>#\s]*)?No termination warning\b[^\n]*$",
    r"^\s*(?:[-*>#\s]*)?The Writer should\b[^\n]*$",
    r"^\s*(?:[-*>#\s]*)?The (?:Critic|Editor|Skeptic|Triage|Writer|Researcher)\b(?: must| should| will| has to)\b[^\n]*$",
    r"^\s*(?:[-*>#\s]*)?REWRITE INSTRUCTION\b[^\n]*$",
    r"^\s*(?:[-*>#\s]*)?(?:EDITO|R) DIRECTIVE\b[^\n]*$",
    r"^\s*(?:[-*>#\s]*)?Output (?:exactly|only)\b[^\n]*$",
    r"^\s*(?:[-*>#\s]*)?(?:Fields to output|Output no field twice)\b[^\n]*$",
    r"^\s*(?:[-*>#\s]*)?Do not (?:adopt|answer|treat|use|refuse|flatter|moralize|create)\b[^\n]*$",
    r"^\s*(?:[-*>#\s]*)?Write (?:one|only|the)\b[^\n]*$",
    r"^\s*(?:[-*>#\s]*)?(?:THINKING_PROCESS|RESPONSE|FINAL[ _-]RESPONSE)\s*[:!]?\s*$",
    r"^\s*(?:[-*>#\s]*)?LOGIC ANCHOR\b[^\n]*$",
    r"^\s*(?:[-*>#\s]*)?STEP CONTRACT\b[^\n]*$",
    r"^\s*(?:[-*>#\s]*)?CONTINUATION MODE\b[^\n]*$",
    r"^\s*(?:[-*>#\s]*)?(?:This|The previous) draft\b[^\n]*$",
]

META_RES = [re.compile(pattern, re.I | re.M) for pattern in META_INSTRUCTION_PATTERNS]

MIN_TOKENS = 2
STOP = STOPWORDS


def tokenize(text: str) -> list[str]:
    return [t for t in normalize(text).split() if len(t) > 2 and t not in STOP]


def _tfidf_matrix(paragraphs: list[str]) -> np.ndarray:
    vocab: dict[str, int] = {}
    docs: list[list[str]] = []
    for para in paragraphs:
        tokens = tokenize(para)
        docs.append(tokens)
        for tok in set(tokens):
            vocab[tok] = vocab.get(tok, 0) + 1
    if not vocab:
        return np.zeros((len(paragraphs), 0), dtype=np.float64)
    index = {tok: i for i, tok in enumerate(sorted(vocab))}
    matrix = np.zeros((len(paragraphs), len(index)), dtype=np.float64)
    for row, tokens in enumerate(docs):
        counts: dict[str, int] = {}
        for tok in tokens:
            counts[tok] = counts.get(tok, 0) + 1
        for tok, count in counts.items():
            idf = math.log((len(paragraphs) + 1) / (vocab[tok] + 1)) + 1.0
            matrix[row, index[tok]] = (1.0 + math.log(count)) * idf
    norms = np.linalg.norm(matrix, axis=1) + 1e-9
    return matrix / norms[:, None]


def max_cosine(text: str, others: list[str]) -> float:
    if not tokenize(text) and not any(tokenize(o) for o in others):
        return 1.0
    rows = [text] + others
    matrix = _tfidf_matrix(rows)
    if matrix.shape[1] == 0:
        return 0.0
    scores = matrix[1:] @ matrix[0]
    return float(np.max(scores)) if len(scores) else 0.0


def dedupe_paragraphs(text: str, threshold: float = DEDUPE_THRESHOLD) -> tuple[str, list[str]]:
    """If cosine similarity exceeds the threshold between two paragraphs, the later one is deleted."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    kept: list[str] = []
    dropped: list[str] = []
    for para in paras:
        if len(para) < 120:
            kept.append(para)
            continue
        if max_cosine(para, kept) > threshold:
            dropped.append(para)
            continue
        kept.append(para)
    return "\n\n".join(kept), dropped


def scrub_meta(text: str) -> tuple[str, list[str]]:
    """Delete leaked prompt artifacts and model-directed instructions from the user-facing draft."""
    removed: list[str] = []
    lines = (text or "").splitlines()
    kept: list[str] = []
    for line in lines:
        hit = next((rx for rx in META_RES if rx.search(line.strip())), None)
        if hit:
            removed.append(line.strip())
            continue
        kept.append(line)
    return "\n".join(kept), removed


CONCAT_LABEL = re.compile(r"([A-Z][A-Za-z &/+-]{2,40}:\s*[^\n:]{2,80}?)\1")


def normalize_format(text: str) -> tuple[str, int]:
    """Fix concatenated labels ('Mode: XMode: X' -> 'Mode: X') and unbroken bullet runs."""
    fixes = 0
    out = text or ""
    for _ in range(6):
        hit = CONCAT_LABEL.search(out)
        if not hit:
            break
        out = out.replace(hit.group(1) + hit.group(1), hit.group(1), 1)
        fixes += 1
    fixed_lines: list[str] = []
    for line in out.splitlines():
        if re.search(r"\.\s+[-*\u2022]\s+\S", line) and not line.strip().startswith(("-", "*", "|", "#")):
            pieces = re.split(r"(?<=[.;])\s+(?=[-*\u2022]\s)", line)
            if len(pieces) > 1:
                fixes += len(pieces) - 1
                fixed_lines.extend(piece.strip() for piece in pieces if piece.strip())
                continue
        fixed_lines.append(line)
    return "\n".join(fixed_lines), fixes


def scrub(text: str) -> dict:
    """Run the full post-generation scrub. Returns the clean text plus the mechanical audit."""
    text, removed_meta = scrub_meta(text)
    text, dropped = dedupe_paragraphs(text)
    text, format_fixes = normalize_format(text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return {
        "text": text,
        "removed_meta_instructions": removed_meta,
        "dropped_duplicate_paragraphs": dropped,
        "format_fixes": format_fixes,
    }
