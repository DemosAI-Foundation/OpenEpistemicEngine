"""Phase 2 - Universal S2A (System 2 Attention). Runs on every turn, for every
model, so the user always sees the same de-biasing step in the trace."""

from __future__ import annotations

import re
import time

from . import lexicon as lx
from . import prompts as P
from . import semantic
from .jsonx import extract_json, parse_and_repair
from .llm import ProviderError
from .security import strip_reference_delimiter_collisions
from .textutil import tokenize

ANSWERISH = re.compile(r"(?i)^(the answer is|to answer|answering:|in summary,? the answer)")
MAX_LEN = 900


def deterministic_sanitize(raw: str, has_reference: bool) -> tuple[str, list[str]]:
    """Fallback sanitizer used when the model's Pass 1 output is unusable."""
    bias = detect_bias(raw)
    text = raw
    for pat in lx.PRESSURE_PATTERNS:
        text = re.sub(pat, " ", text)
    for term in sorted(bias, key=len, reverse=True):
        text = re.sub(re.escape(term), " ", text, flags=re.I)
    text = re.sub(r"\s{2,}", " ", text).strip(" ?.!")
    if has_reference:
        out = f"Does the provided reference support the following: {text}?"
    else:
        out = (text + "?") if not text.endswith("?") else text
    return out[:MAX_LEN], bias


def detect_bias(text: str) -> list[str]:
    low = text.lower()
    found: list[str] = []
    for term in lx.LOADED_TERMS:
        if term in low and term not in found:
            found.append(term)
    for pat in lx.PRESSURE_PATTERNS:
        m = re.search(pat, low)
        if m and m.group(0).strip() not in found:
            found.append(m.group(0).strip())
    return found


def sanitize(provider, raw_clean: str, doc_ids: list[str]) -> dict:
    has_reference = bool(doc_ids)
    pointer = (
        f"The user has attached reference(s): {', '.join(doc_ids)}. Refer to them only "
        "as 'the provided reference'."
        if has_reference else "none"
    )
    safe_raw, delim_flags = strip_reference_delimiter_collisions(raw_clean)
    user = P.PASS1_USER_TEMPLATE.format(raw_text=safe_raw, reference_pointer=pointer)
    notes: list[str] = []
    raw_output = ""
    latency = 0.0
    model = provider.settings.model
    sanitized = ""
    bias: list[str] = []
    intent_preserved = True
    intent_reported = False
    flags: list[str] = list(delim_flags)
    t0 = time.perf_counter()
    try:
        resp = provider.chat(P.PASS1_SYSTEM, user, schema=P.PASS1_SCHEMA,
                             kind="pass1", temperature=0.0)
        latency = resp.latency_ms
        raw_output = resp.text
        model = resp.model
        pre = extract_json(resp.text)
        if isinstance(pre, dict):
            for probe in ("intent_preserved", "preserved_intent", "intent"):
                if probe in pre:
                    intent_reported = True
                    intent_preserved = bool(pre[probe])
                    break
        obj = parse_and_repair(resp.text, P.PASS1_SCHEMA, notes)
        sanitized = (obj.get("sanitized_request") or "").strip()
        bias = list(obj.get("bias_removed") or [])
        if intent_reported:
            intent_preserved = bool(obj.get("intent_preserved", intent_preserved))
    except (ProviderError, ValueError) as exc:
        flags.append(f"DEGRADED: Pass 1 model call failed ({exc}); deterministic fallback used")

    deterministic_bias = detect_bias(raw_clean)
    if not sanitized:
        sanitized, _ = deterministic_sanitize(raw_clean, has_reference)
        flags.append("FALLBACK: Pass 1 returned no sanitized_request")
    if ANSWERISH.search(sanitized):
        sanitized, _ = deterministic_sanitize(raw_clean, has_reference)
        flags.append("BLOCKED: Pass 1 answered the question; output replaced with a neutral request")
    if intent_reported and not intent_preserved:
        sanitized, _ = deterministic_sanitize(raw_clean, has_reference)
        flags.append("FALLBACK: Pass 1 reported intent_preserved=false; deterministic rewrite used")
    elif not intent_reported:
        flags.append("POLICY: Pass 1 did not report intent_preserved; kept the model's rewrite "
                     "and cross-checked it against the input")

    missed = [t for t in deterministic_bias if not _covered(t, bias)]
    if missed:
        flags.append(
            "CROSS-CHECK: deterministic lexicon found loaded terms Pass 1 did not report: "
            + ", ".join(missed[:10])
        )
    fabricated, semantic_hits = [], []
    for t in bias:
        if _present_in(t, raw_clean):
            continue
        score, path = semantic.similarity_best(t, _clauses(raw_clean))
        if score >= 0.45:
            semantic_hits.append(f"'{t}' ~ {score:.2f} ({path})")
        else:
            fabricated.append(t)
    if fabricated:
        flags.append(
            "CROSS-CHECK: Pass 1 reported terms absent from the input: " + ", ".join(fabricated[:6])
        )
    if semantic_hits:
        flags.append(
            "S2A_CROSSCHECK_SEMANTIC: Pass 1 reported paraphrases, not literal strings; "
            "matched semantically: " + ", ".join(semantic_hits[:6])
        )
    if has_reference and "reference" not in sanitized.lower() and _has_reference_pointer(raw_clean):
        flags.append("POLICY: Pass 1 dropped the reference pointer; re-anchored by fallback clause")
        sanitized = f"{sanitized} (Answer using the provided reference.)"
    if len(sanitized) > MAX_LEN:
        sanitized = sanitized[: MAX_LEN - 2].rsplit(" ", 1)[0] + " …"
        flags.append(f"TRUNCATED: sanitized_request exceeded {MAX_LEN} chars")

    return {
        "sanitized_request": sanitized,
        "bias_removed": bias,
        "deterministic_bias": deterministic_bias,
        "intent_preserved": intent_preserved,
        "system_prompt": P.PASS1_SYSTEM,
        "user_prompt": user,
        "raw_output": raw_output,
        "latency_ms": round(latency, 2),
        "wall_ms": round((time.perf_counter() - t0) * 1000, 2),
        "model": model,
        "flags": flags,
        "repairs": notes,
    }


def _covered(term: str, reported: list[str]) -> bool:
    low = term.lower()
    for r in reported:
        rl = r.lower()
        if rl in low or low in rl:
            return True
        head = low.split()[0]
        if head and head in rl:
            return True
    return False


CLAUSE_SPLIT = re.compile(r"[.!?;:\n\r]+")


def _clauses(text: str, max_clauses: int = 300) -> list[str]:
    """Candidate spans for the paraphrase cross-check: clause boundaries plus
    sliding 4-12 word windows, so a short reported phrase is compared with the
    phrase-sized region of the input instead of the whole message."""
    out = [c.strip() for c in CLAUSE_SPLIT.split(text or "") if len(c.split()) >= 2]
    words = (text or "").split()
    for n in range(4, 13):
        out.extend(" ".join(words[i:i + n]) for i in range(len(words) - n + 1))
    seen: set[str] = set()
    out = [c for c in out if c and not (c in seen or seen.add(c))]
    return out[:max_clauses]


def _present_in(term: str, text: str) -> bool:
    """A reported bias item counts as real if it quotes the input, uses the
    input's words, or shares content stems with it (models report bias as
    sentences, not bare terms)."""
    low = text.lower()
    tl = term.lower()
    if tl in low:
        return True
    quoted = re.findall(r"['\"“‘]([^'\"”’]{3,60})['\"”’]", term)
    for frag in quoted:
        if frag.lower().strip() in low:
            return True
        if _overlap(frag, text):
            return True
    if quoted:
        return False
    return _overlap(tl, text)


def _overlap(source: str, text: str) -> bool:
    a = [w for w in tokenize(source) if len(w) > 4]
    if not a:
        return False
    b = set(tokenize(text))
    hits = sum(1 for w in a if w in b)
    return hits / len(a) >= 0.6


def _has_reference_pointer(text: str) -> bool:
    return bool(re.search(
        r"(?i)\b(this|the|that|above|following|attached|pasted|provided)\s+"
        r"(article|essay|paper|text|document|study|report|post|chapter|book|passage|excerpt)\b",
        text,
    ))
