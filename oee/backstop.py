"""Phase 5 - Grounding & backstop. Deterministic post-hoc guardrails
(corrective RAG / lexical provenance checks) applied to the model's output."""

from __future__ import annotations

import re

from . import lexicon as lx
from . import semantic
from .security import REDACTION

LLM_ISMS = list(lx.LLM_ISMS)

MODAL_TAG_RE = re.compile(r"\[(FACT|PREDICTION|HYPOTHETICAL|POSSIBILITY|CONSTRAINT|AUTHOR_CLAIM)\]")
CITATION_RE = re.compile(r"\[(R\d+-C\d+|R\d+-S\d+)\]")
CARD_CITATION_RE = re.compile(r"\[R\d+-CARD[0-9]*\]")


def epistemic_backstop(chat_reply: str, has_references: bool,
                       retrieved_ids: set[str] | None = None,
                       citations: list[dict] | None = None,
                       chunk_texts: dict[str, str] | None = None,
                       rewrite: bool = True) -> tuple[str, list]:
    """Strips banned phrases and checks provenance rules."""
    flags: list[str] = []

    chat_reply, opener_flags = scrub_sycophantic_openers(chat_reply)
    flags.extend(opener_flags)

    for pattern in LLM_ISMS:
        if re.search(pattern, chat_reply):
            if rewrite:
                chat_reply = re.sub(pattern, "", chat_reply)
            flags.append(f"SCRUBBED: LLM-ism '{pattern}'")

    if not has_references and re.search(lx.EVIDENCE_PHRASE, chat_reply):
        flags.append(
            "WARNING: Claimed 'evidence' without provided references. Review required."
        )
        if rewrite:
            chat_reply = re.sub(lx.EVIDENCE_PHRASE, "Parametric synthesis suggests", chat_reply)

    if has_references and re.search(lx.EVIDENCE_PHRASE, chat_reply):
        cited = re.findall(CITATION_RE, chat_reply)
        if not cited:
            flags.append(
                "WARNING: 'Evidence' phrasing used with no [R#-C#] citation in the "
                "chat_reply; provenance rule 1 requires a chunk citation."
            )

    if retrieved_ids is not None:
        for cid in dict.fromkeys(re.findall(CITATION_RE, chat_reply)):
            if cid not in retrieved_ids:
                flags.append(
                    f"UNVERIFIED CITATION: [{cid}] was not in the retrieved set "
                    f"({len(retrieved_ids)} nodes were); the quote cannot be checked."
                )
        for row in citations or []:
            cid = row.get("chunk_id", "")
            quote = (row.get("quote") or "").strip()
            if not cid or not quote:
                continue
            if cid not in retrieved_ids:
                flags.append(f"UNVERIFIED CITATION: sidecar cites unknown [{cid}]")
                continue
            source = (chunk_texts or {}).get(cid, "")
            if not _quote_in(quote, source):
                flags.append(
                    f"QUOTE MISMATCH: sidecar quote for [{cid}] is not present in the "
                    f"stored chunk (paraphrase, not quotation)."
                )

    if citations:
        flags.extend(nli_check_citations(citations, chunk_texts or {}, chat_reply))

    if not MODAL_TAG_RE.search(chat_reply):
        flags.append(
            "POLICY: chat_reply contains no modal tags; rule 3 requires [FACT]/"
            "[PREDICTION]/[HYPOTHETICAL]/[POSSIBILITY]/[CONSTRAINT]/[AUTHOR_CLAIM]."
        )

    chat_reply = re.sub(r"\s{2,}", " ", chat_reply).strip()
    return chat_reply, flags



FORBIDDEN_OPENERS = re.compile(
    r"^\s*(?:absolutely|certainly|of course|you'?re right|great question|"
    r"good question|i agree|indeed|well said|exactly right)[,!:!?\s]+",
    re.IGNORECASE)

OPPOSITION_MARKERS = re.compile(
    r"\b(however|but|conversely|nevertheless|yet|although|on the contrary|"
    r"against this|the counter\b|objection)\b", re.IGNORECASE)


def scrub_sycophantic_openers(text: str) -> tuple[str, list[str]]:
    """Rule 8: the answer must not open with agreement. Deterministic strip,
    never a re-prompt."""
    flags: list[str] = []
    out = text
    for _ in range(3):
        m = FORBIDDEN_OPENERS.match(out)
        if not m:
            break
        flags.append(f"SCRUBBED: sycophantic opener {m.group(0).strip()!r}")
        out = out[m.end():]
    if flags:
        out = out.strip()
        if out:
            out = out[0].upper() + out[1:]
    return out, flags


def dialectic_opposition_check(thesis: str, antithesis: str) -> str | None:
    """Strawman guard: if the thesis entails the antithesis, the "counter-position"
    is a consequence of the thesis, not an opposition. Deterministic NLI when the
    model is loaded, skipped (not faked) when it is not."""
    result = semantic.nli(thesis, antithesis)
    if result is None:
        return None
    if result["label"] == "entailment" and result["score"] >= 0.5:
        return ("antithesis is entailed by the thesis "
                f"(NLI entailment, score {result['score']}) - a strawman, not a "
                "counter-position")
    return None


def check_dialectic_quality(dialectic) -> list[str]:
    flags: list[str] = []
    if not isinstance(dialectic, dict):
        return ["DIALECTIC INCOMPLETE: the model returned no dialectic object."]
    thesis = (dialectic.get("thesis") or "").strip()
    antithesis = (dialectic.get("antithesis") or "").strip()
    synthesis = (dialectic.get("synthesis") or "").strip()
    for name, value in (("thesis", thesis), ("antithesis", antithesis),
                        ("synthesis", synthesis)):
        if not value:
            flags.append(f"DIALECTIC INCOMPLETE: {name} is empty.")
    if antithesis and not OPPOSITION_MARKERS.search(antithesis):
        flags.append("DIALECTIC WEAK: the antithesis carries no opposition marker "
                     "('however', 'but', 'conversely', ...) - it restates rather than opposes.")
    if antithesis and len(antithesis) < 40:
        flags.append(f"DIALECTIC WEAK: antithesis is {len(antithesis)} chars - "
                     "too short to be a real counter-position.")
    if antithesis and thesis and len(antithesis) * 4 < len(thesis):
        flags.append("DIALECTIC WEAK: the antithesis is far shorter than the thesis "
                     "(strawman shape).")
    if antithesis and thesis:
        straw = dialectic_opposition_check(thesis, antithesis)
        if straw:
            flags.append(f"DIALECTIC WEAK: {straw}.")
    return flags


MARKER_NOISE_RE = re.compile(
    r"\[(?:R\d+-[A-Z0-9-]+|FACT|PREDICTION|HYPOTHETICAL|POSSIBILITY|CONSTRAINT|"
    r"AUTHOR_CLAIM|INFERENCE|SPECULATION)\]")


def clean_hypothesis(text: str) -> str:
    """Citation and modality markers are formatting, not content: left in, the
    NLI model reads '[AUTHOR_CLAIM]' as part of the claim and answers 'neutral'
    for an obvious entailment."""
    out = " ".join(MARKER_NOISE_RE.sub(" ", text or "").split())
    return re.sub(r"\s+([.,;:!?])", r"\1", out).strip()


def claim_sentences(reply: str, cid: str) -> str:
    """The sentences of the reply that carry this citation: what the model is
    actually asserting from the chunk, used when the sidecar row has no claim."""
    if not reply or not cid:
        return ""
    hits = [s for s in re.split(r"(?<=[.!?])\s+", reply) if f"[{cid}]" in s]
    return clean_hypothesis(" ".join(hits))


def nli_check_citations(citations: list[dict], chunk_texts: dict[str, str],
                        reply: str = "") -> list[str]:
    """Semantic layer over the citation rows. The substring check stays; NLI is
    an extra, labelled pass, and is reported as unavailable rather than faked.
    premise = stored chunk, hypothesis = the claim (row claim, else the citing
    sentence of chat_reply)."""
    flags: list[str] = []
    checked = 0
    for row in citations:
        cid = row.get("chunk_id", "")
        claim = clean_hypothesis((row.get("claim") or "").strip()) \
            or claim_sentences(reply, cid)
        source = (chunk_texts or {}).get(cid, "")
        if not claim:
            continue
        result = semantic.nli(source, claim)
        if not (source or "").strip():
            row["nli_label"] = "no-chunk-text"
            continue
        if result is None:
            row["nli_label"] = "unavailable"
            continue
        row["nli_hypothesis"] = claim[:300]
        checked += 1
        row["nli_label"] = result["label"]
        row["nli_score"] = result["score"]
        if result["label"] == "contradiction":
            flags.append(f"CLAIM_CONTRADICTS_SOURCE: [{cid}] is labelled contradiction "
                         f"against the stored chunk (score {result['score']}).")
        elif result["label"] == "neutral":
            flags.append(f"CLAIM_UNSUPPORTED: [{cid}] is labelled neutral - the chunk "
                         f"neither supports nor contradicts the claim (score {result['score']}).")
    return flags


def critic(sidecar: dict) -> dict:
    """Deterministic critic over the generated payload. High severity means the
    answer is missing the structure the schema exists to force."""
    violations: list[str] = []
    severity = "low"
    reply = (sidecar.get("chat_reply") or "").strip()
    bottom = (sidecar.get("bottom_line") or "").strip()
    if not bottom:
        violations.append("bottom_line is empty: the answer has no plain-language direct answer")
        severity = "high"
    elif len(bottom) > 600:
        violations.append("bottom_line is over 600 chars: it must be 1-3 sentences")
    if not reply:
        violations.append("chat_reply is empty")
        severity = "high"
    if FORBIDDEN_OPENERS.match(reply):
        violations.append("chat_reply opens with an agreement marker")
        severity = "high"
    dialectic_flags = check_dialectic_quality(sidecar.get("dialectic"))
    for f in dialectic_flags:
        violations.append(f.replace("DIALECTIC ", "dialectic: "))
    if any("INCOMPLETE" in f for f in dialectic_flags):
        severity = "high"
    retrieval = sidecar.get("retrieval") or {}
    has_references = bool(retrieval.get("has_references",
                                        retrieval.get("leaves")
                                        or retrieval.get("sections")))
    if has_references and not (sidecar.get("citations") or []):
        violations.append("no citations were emitted on a reference-grounded turn")
    if not (sidecar.get("what_would_change") or []):
        violations.append("what_would_change is empty on an analytical question")
    for c in sidecar.get("methodological_constraints", []) or []:
        if re.search(r"\bimpossible\b", str(c), re.I):
            violations.append(f"methodological constraint says 'impossible': {str(c)[:80]}")
            severity = "high"
    card_cited = bool(CARD_CITATION_RE.search(reply)) or any(
        str(c.get("chunk_id", "")).endswith("-CARD")
        for c in sidecar.get("citations") or [])
    if card_cited and "[AUTHOR_CLAIM]" not in reply:
        violations.append(
            "chat_reply cites the reference but never uses [AUTHOR_CLAIM]: sentences "
            "reporting what the source authors assert must carry [AUTHOR_CLAIM] under "
            "modality rule 3. Add the tags; change nothing else.")
        severity = "high"
    return {"passed": not violations, "severity": severity, "violations": violations}


def score_candidate(sidecar: dict, flags: list[str]) -> tuple[float, dict]:
    """Best-of-2 selection score: fewer guardrail violations and more completed
    structure wins. Deterministic and explainable."""
    hard = ("SCRUBBED", "UNVERIFIED", "QUOTE MISMATCH", "INCONSISTENT", "INCOMPLETE",
            "WEAK", "CONTRADICT", "UNSUPPORTED", "WARNING")
    penalties = sum(3.0 for f in flags if any(h in f for h in hard))
    dialectic = sidecar.get("dialectic") or {}
    parts = {
        "penalties": -penalties,
        "bottom_line": 2.0 if (sidecar.get("bottom_line") or "").strip() else 0.0,
        "dialectic": (1.0 if (dialectic.get("thesis") or "").strip() else 0.0)
                     + (1.5 if OPPOSITION_MARKERS.search(dialectic.get("antithesis") or "") else 0.0)
                     + (1.0 if (dialectic.get("synthesis") or "").strip() else 0.0),
        "citations": sum(1.0 for c in sidecar.get("citations", []) if c.get("verified"))
                     - 1.0 * sum(1 for c in sidecar.get("citations", [])
                                 if c.get("verified") is False),
        "falsifiability": 1.0 if (sidecar.get("what_would_change") or []) else 0.0,
        "tags": 1.0 if MODAL_TAG_RE.search(sidecar.get("chat_reply") or "") else 0.0,
        "length": min(1.0, len(sidecar.get("chat_reply") or "") / 400.0),
    }
    return round(sum(parts.values()), 3), parts


def _quote_in(quote: str, source: str) -> bool:
    q = _norm(quote)
    s = _norm(source)
    if not q:
        return False
    q = q.strip("…").strip()
    if len(q) < 12:
        return q in s
    probe = q[: min(60, len(q))]
    if probe in s:
        return True
    words = [w for w in probe.split() if len(w) > 4]
    if not words:
        return False
    hits = sum(1 for w in words if w in s)
    return hits / len(words) >= 0.75


def _norm(text: str) -> str:
    text = (text or "").replace(REDACTION, " ")
    text = re.sub(r"[“”«»\"'`]", "", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    return re.sub(r"\s{2,}", " ", text).strip()


def sidecar_backstop(sidecar: dict, has_references: bool, retrieved_ids: set[str]) -> list:
    flags = []
    flags.extend(check_dialectic_quality(sidecar.get("dialectic")))
    if not (sidecar.get("bottom_line") or "").strip():
        flags.append("INCOMPLETE: bottom_line is empty; the schema requires a direct answer.")
    for c in sidecar.get("methodological_constraints", []) or []:
        if re.search(r"\bimpossible\b", str(c), re.I):
            flags.append(f"CONSTRAINT RULE: methodological_constraints says 'impossible' "
                         f"({str(c)[:60]}...) - rule 13 forbids that wording.")
    basis = sidecar.get("basis", [])
    if has_references and "reference" not in basis and sidecar.get("citations"):
        flags.append(
            "INCONSISTENT: citations present but basis omits 'reference' "
            "(auto-corrected in the sidecar)."
        )
    if not has_references and "reference" in basis:
        flags.append(
            "INCONSISTENT: basis claims 'reference' grounding but no references exist."
        )
    if not has_references and sidecar.get("citations"):
        flags.append(
            "INCONSISTENT: citations emitted with no references in the corpus."
        )
    for row in sidecar.get("citations", []):
        if row.get("chunk_id") not in retrieved_ids:
            flags.append(f"UNVERIFIED CITATION: sidecar [{row.get('chunk_id')}] not retrieved.")
    return flags
