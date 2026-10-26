"""Optional semantic models (MiniLM-style similarity, NLI, real embeddings)
with deterministic, dependency-free fallbacks. Every call degrades to a lexical
measure and reports which path was used, so a machine without torch keeps
working and stays honest about what was checked."""

from __future__ import annotations

import math
import os
import re

SIM_MODEL = os.getenv("OEE_SEMANTIC_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
NLI_MODEL = os.getenv("OEE_NLI_MODEL", "cross-encoder/nli-deberta-v3-small")
EMBED_MODEL = os.getenv("OEE_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
MODE = os.getenv("OEE_SEMANTIC", "auto")  # auto | off | on

_state: dict = {"tried": False, "sim": None, "nli": None, "embed": None,
                "notes": []}


def _load():
    if _state["tried"]:
        return _state
    _state["tried"] = True
    if MODE == "off":
        _state["notes"].append("semantic models disabled (OEE_SEMANTIC=off)")
        return _state
    try:
        from sentence_transformers import SentenceTransformer, CrossEncoder  # type: ignore
    except Exception as exc:
        _state["notes"].append(
            "semantic models unavailable (sentence-transformers not installed); "
            "lexical fallbacks in use")
        return _state
    try:
        _state["sim"] = SentenceTransformer(SIM_MODEL)
        _state["notes"].append(f"similarity model loaded: {SIM_MODEL}")
    except Exception as exc:
        _state["notes"].append(f"similarity model failed to load: {type(exc).__name__}")
    try:
        _state["nli"] = CrossEncoder(NLI_MODEL)
        _state["notes"].append(f"NLI model loaded: {NLI_MODEL}")
    except Exception as exc:
        _state["notes"].append(f"NLI model failed to load: {type(exc).__name__}")
    try:
        _state["embed"] = SentenceTransformer(EMBED_MODEL)
        _state["notes"].append(f"embedding model loaded: {EMBED_MODEL}")
    except Exception as exc:
        _state["notes"].append(f"embedding model failed to load: {type(exc).__name__}")
    return _state


def status() -> dict:
    st = _load()
    return {"mode": MODE,
            "similarity": "model" if st["sim"] else "lexical",
            "nli": "model" if st["nli"] else "unavailable",
            "embeddings": "model" if st["embed"] else "unavailable",
            "notes": list(st["notes"])}


STOPWORDS = set(
    "the a an of to in on for and or is are was were be been that this it with as at "
    "by from into not no nor so if then than them they there these those we you your "
    "his her its their our my i he she do does did".split())


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def lexical_similarity(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    return inter / math.sqrt(len(ta) * len(tb))


def similarity(a: str, b: str) -> tuple[float, str]:
    """Cosine similarity when a model is loaded, otherwise a deterministic
    token-overlap measure. Returns (score, path)."""
    st = _load()
    if st["sim"] is not None:
        try:
            import numpy as np
            va = st["sim"].encode([a], normalize_embeddings=True)[0]
            vb = st["sim"].encode([b], normalize_embeddings=True)[0]
            return float(np.dot(va, vb)), "model"
        except Exception:
            pass
    return lexical_similarity(a, b), "lexical"


def similarity_best(a: str, parts: list[str]) -> tuple[float, str]:
    """Best cosine of `a` against a list of candidate spans. Paraphrase checks
    compare a short phrase against the clauses of the source, not the whole
    document, so length dilution does not hide a real paraphrase."""
    parts = [p.strip() for p in parts if p and p.strip()]
    if not parts:
        return 0.0, "none"
    st = _load()
    if st["sim"] is not None:
        try:
            import numpy as np
            va = st["sim"].encode([a], normalize_embeddings=True)[0]
            vb = st["sim"].encode(parts, normalize_embeddings=True)
            return float(max(float(np.dot(va, v)) for v in vb)), "model"
        except Exception:
            pass
    return max((lexical_similarity(a, p) for p in parts), default=0.0), "lexical"


NLI_LABELS = ("contradiction", "entailment", "neutral")


def nli(premise: str, hypothesis: str) -> dict | None:
    """Entailment check. None when no NLI model is available: the caller keeps
    the substring verification and says so, rather than inventing a label."""
    st = _load()
    if st["nli"] is None:
        return None
    if not (premise or "").strip() or not (hypothesis or "").strip():
        return None
    try:
        scores = st["nli"].predict([(premise, hypothesis)])
        row = scores[0] if hasattr(scores, "shape") and scores.ndim > 1 else scores
        probs = _softmax(row)
        idx = max(range(len(probs)), key=lambda i: probs[i])
        return {"label": NLI_LABELS[idx % 3], "score": round(float(probs[idx]), 4),
                "path": "model"}
    except Exception:
        return None


def _softmax(row) -> list[float]:
    import math as _m
    vals = [float(v) for v in row]
    top = max(vals)
    exps = [_m.exp(v - top) for v in vals]
    total = sum(exps) or 1.0
    return [e / total for e in exps]


def embed_texts(texts: list[str]) -> list[list[float]] | None:
    st = _load()
    if st["embed"] is None:
        return None
    try:
        return [list(map(float, v)) for v in
                st["embed"].encode(texts, normalize_embeddings=True)]
    except Exception:
        return None
