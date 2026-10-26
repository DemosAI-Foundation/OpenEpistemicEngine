"""Token estimation, sentence/paragraph segmentation, lexical normalization."""

from __future__ import annotations

import re

WORD_RE = re.compile(r"[A-Za-z0-9_'\u00c0-\u024f]+")
SENT_RE = re.compile(r"(?<=[.!?…])\s+(?=[\"'“(\[]?[A-Z0-9])")
PARA_SPLIT = re.compile(r"\n\s*\n+")

STOPWORDS = set(
    """a an the and or but if then than that this these those of to in on at by for with
    without from into over under about as is are was were be been being do does did done
    have has had will would shall should can could may might must not no nor so very
    it its it's i you he she they we me my your his her their our us them
    what which who whom when where why how all any some each other same too
    s t just also there here again once more
    """.split()
)

SUFFIXES = (
    "ational", "iveness", "fulness", "ousness", "ization", "ations", "ement",
    "ments", "ingly", "edly", "ance", "ence", "ness", "ment", "able", "ible",
    "ing", "ies", "ied", "ers", "est", "ed", "ly", "er", "es", "s",
)


def stem(word: str) -> str:
    w = word.lower()
    if len(w) <= 3:
        return w
    if w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    for suf in SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            base = w[: -len(suf)]
            if base.endswith("b") and suf.startswith("i"):
                base = base[:-1] + "ble"
            return base
    return w


def tokenize(text: str, stem_tokens: bool = True) -> list[str]:
    fn = stem if stem_tokens else (lambda w: w)
    return [fn(m.group(0)) for m in WORD_RE.finditer(text) if m.group(0).lower() not in STOPWORDS]


def estimate_tokens(text: str) -> int:
    """Cheap, dependency-free proxy for subword token count."""
    if not text:
        return 0
    words = len(WORD_RE.findall(text))
    chars = len(text)
    cjk = len(re.findall(r"[\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]", text))
    return int(max(words * 1.35, chars / 4.0) + cjk)


def sentences(text: str) -> list[str]:
    parts = [p.strip() for p in SENT_RE.split(text) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in PARA_SPLIT.split(text) if p.strip()]


def truncate_chars(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + " …[truncated]"


def quote_span(text: str, max_chars: int = 220) -> str:
    """Deterministic representative span for citation tooltips."""
    clean = " ".join(text.split())
    if len(clean) <= max_chars:
        return clean
    mid = len(clean) // 2
    start = max(0, mid - max_chars // 2)
    return "…" + clean[start : start + max_chars] + "…"
