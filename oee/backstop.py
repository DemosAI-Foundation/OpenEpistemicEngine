"""Phase 5 - Grounding & backstop. Deterministic post-hoc guardrails
(corrective RAG / lexical provenance checks) applied to the model's output."""

from __future__ import annotations

import re

from . import lexicon as lx
from . import semantic
from .prompts import MODAL_TAGS
from .security import REDACTION

LLM_ISMS = list(lx.LLM_ISMS)

MODAL_TAG_RE = re.compile(
    r"\[(FACT|PREDICTION|HYPOTHETICAL|POSSIBILITY|CONSTRAINT|PRESUPPOSITION|"
    r"AUTHOR_CLAIM|INFERRED ATTRIBUTION|INFERRED VALUE)\]")

# Modal tags a rewrite emitted WITHOUT their [ ] brackets (the model copies the
# answer and drops the brackets, silently disabling the tag colouring). Longest
# first so "INFERRED ATTRIBUTION" matches before "INFERRED".
_BARE_TAG_RE = re.compile(
    r"(?<!\[)\b(" + "|".join(sorted(MODAL_TAGS, key=len, reverse=True)) + r")\b(?!\])")


def restore_modal_tags(text: str) -> tuple[str, int]:
    """Re-bracket bare modal tags. Idempotent: already-bracketed tags are left
    untouched. Returns (fixed_text, n_tags_restored)."""
    n = 0

    def sub(m):
        nonlocal n
        n += 1
        return "[" + m.group(0) + "]"

    return _BARE_TAG_RE.sub(sub, text or ""), n
CITATION_RE = re.compile(r"\[(R\d+-C\d+|R\d+-S\d+)\]")
CARD_CITATION_RE = re.compile(r"\[R\d+-CARD[0-9]*\]")


def epistemic_backstop(chat_reply: str, has_references: bool,
                       retrieved_ids: set[str] | None = None,
                       citations: list[dict] | None = None,
                       chunk_texts: dict[str, str] | None = None,
                       rewrite: bool = True,
                       bottom_line: str = "") -> tuple[str, list, str]:
    """Strips banned phrases and checks provenance rules. Returns the cleaned
    chat_reply, the flag list, and the cleaned bottom_line: validation-first
    framing is severed from the summary the user reads first."""
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

    # Ghost-document guard. With no reference ingested, any chunk ID in the reply
    # is one the model invented for text that only ever sat in the prompt as
    # inline user data. Flag it loudly and strip the fake provenance from the
    # prose: the analysis of the inline text stays, the citation does not.
    if not has_references:
        valid_ids = set(retrieved_ids or set())
        cited_ids = {m.group(1) for m in CITATION_RE.finditer(chat_reply)} | {
            m.group(0)[1:-1] for m in CARD_CITATION_RE.finditer(chat_reply)}
        ghosts = sorted(cid for cid in cited_ids if cid not in valid_ids)
        if ghosts:
            flags.append(
                "WARNING: model invented reference IDs for inline prompt text "
                f"({', '.join(ghosts)}): no document was attached, so those chunks "
                "do not exist. The citation tags are stripped; the inline text is "
                "user input, not a reference, and must not be cited as one."
            )
            if rewrite:
                for cid in ghosts:
                    chat_reply = chat_reply.replace(f"[{cid}]", " ")

    if not MODAL_TAG_RE.search(chat_reply):
        flags.append(
            "POLICY: chat_reply contains no modal tags; rule 3 requires [FACT]/"
            "[PREDICTION]/[HYPOTHETICAL]/[POSSIBILITY]/[CONSTRAINT]/[PRESUPPOSITION]/"
            "[AUTHOR_CLAIM]."
        )

    chat_reply = re.sub(r"\s{2,}", " ", chat_reply).strip()

    # Sever validation-first framing from the bottom line: strip the congratulatory
    # opener and promote the conclusion. Deterministic - RLHF agreeableness is not
    # reliably curbed by prompt instructions alone, so the summary gets a guillotine.
    if bottom_line:
        m = VALIDATION_OPENER_RE.match(bottom_line)
        if m:
            flags.append(
                f"SCRUBBED: removed validation-first opener "
                f"('{m.group(0).strip()}') from bottom_line")
            bottom_line = bottom_line[m.end():].strip()
            if bottom_line:
                bottom_line = bottom_line[0].upper() + bottom_line[1:]
            bottom_line = re.sub(r"\s{2,}", " ", bottom_line).strip()

    return chat_reply, flags, bottom_line



FORBIDDEN_OPENERS = re.compile(
    r"^\s*(?:absolutely|certainly|of course|you'?re right|great question|"
    r"good question|i agree|indeed|well said|exactly right)[,!:!?\s]+",
    re.IGNORECASE)

# Validation-first framing: a phrase that congratulates the user's premise before
# delivering the analysis. Respect is shown by rigor, not by conversational padding.
VALIDATION_OPENER_RE = re.compile(
    r"^\s*(?:your (?:point|take|framing|argument|concern|observation)\s+"
    r"(?:is|seems|sounds|reads)\s+(?:reasonable|valid|strong|well[- ]taken|fair|good|"
    r"correct|spot[- ]on)|your (?:point|take|argument)\s+(?:is\s+)?valid|"
    r"the (?:poster|user|author|question|framing)\s+is\s+"
    r"(?:right|correct)\s+(?:that|in\s+saying|to\s+say|in\s+noting)?|"
    r"the (?:poster|user|author)\s+is\s+(?:right|correct)\b|"
    r"the (?:poster|user|author|argument|case|point)\W{0,2}s\s+strongest\s+point\s+is|"
    r"the (?:core|main|central|underlying|basic)\s+(?:argument|case|point|claim)\s+is\s+"
    r"(?:sound|valid|strong|correct|reasonable)|"
    r"you\s+(?:make|raise|have)\s+a\s+(?:good|valid|fair|strong|compelling)\s+"
    r"(?:point|case|argument|critique|observation|question)|"
    r"that(?:'s| is)\s+a\s+(?:valid|fair|good|strong|reasonable|good)\s+"
    r"(?:point|observation|critique|concern|question)|"
    r"i\s+agree\s+(?:with|that)|you\s+are\s+(?:right|correct)\s+(?:that|to\s+say|in)|"
    r"you'?re\s+(?:right|correct)\s+(?:that|to\s+say|in)|"
    r"fair\s+point|good\s+question|great\s+question|well[- ]?posed|"
    r"absolutely|certainly|of\s+course|indeed|exactly\s+right)[,.!?:\s]+",
    re.IGNORECASE)

BOTTOM_VALIDATION_OPENER = re.compile(
    r"^\s*(?:your (?:point|take|framing|argument|concern)\s+\w+|"
    r"you(?:'re| are) (?:right|correct|making a|raise a|make a|spot[- ]on)|"
    r"you make a (?:good|strong|fair|valid) case|you (?:make|raise) a (?:good|strong|fair)|"
    r"fair point|good question|great question|"
    r"that(?:'s| is) (?:a )?(?:good|strong|fair|valid|reasonable|well)|"
    r"this is (?:a )?(?:good|strong|fair|valid|reasonable)|"
    r"agreed|well[- ]?posed|nice (?:point|question)|i (?:agree|appreciate|think)|"
    r"absolutely|certainly|of course)[,.!?:\s]",
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
        if straw is None and semantic.status()["nli"] == "unavailable":
            flags.append("FALLBACK: NLI model unavailable; the strawman (entailment) "
                         "check is skipped - only the opposition-marker rules apply.")
        elif straw:
            flags.append(f"DIALECTIC WEAK: {straw}.")
    return flags


CAPABILITY_DENIAL_RE = re.compile(
    r"(?i)\b(cannot reliably|cannot|can't|can not|is not able|are not able|is unable"
    r"|are unable|not reliably|is unreliable|are unreliable|fails to|fail to"
    r"|lacks? the (?:ability|capacity|means)|without the (?:ability|capacity)"
    r"|no reliable|should not be able)\b")

_AFFIRM_SPLITS = [
    (r"(?i)\bcannot reliably\b", "can reliably"),
    (r"(?i)\bshould not be able to\b", "should be able to"),
    (r"(?i)\bshould not be able\b", "should be able"),
    (r"(?i)\bcannot\b", "can"), (r"(?i)\bcan not\b", "can"), (r"(?i)\bcan't\b", "can"),
    (r"(?i)\bis not able to\b", "is able to"), (r"(?i)\bare not able to\b", "are able to"),
    (r"(?i)\bis unable to\b", "is able to"), (r"(?i)\bare unable to\b", "are able to"),
    (r"(?i)\bnot reliably\b", "reliably"), (r"(?i)\bis unreliable\b", "is reliable"),
    (r"(?i)\bare unreliable\b", "are reliable"), (r"(?i)\bno reliable\b", "reliable"),
    (r"(?i)\bfails? to\b", ""),
    (r"(?i)\blacks? the (ability|capacity|means)\b", r"has the \1"),
    (r"(?i)\bwithout the (ability|capacity)\b", r"with the \1"),
]


def _affirm_capability(text: str) -> str:
    out = text
    for pat, rep in _AFFIRM_SPLITS:
        out = re.sub(pat, rep, out)
    return " ".join(out.split())


def self_referential_consistency_check(sidecar: dict) -> list[str]:
    """Self-undermining-output guard. Compares what the payload declares impossible
    (methodological_constraints, stakes_and_policy) and what it flagged as a
    presupposition (Rule 17, in question_decomposition) against what the synthesis
    and bottom_line actually prescribe. NLI decides; when the model is unavailable
    the check is skipped, never guessed (no false positives)."""
    flags: list[str] = []
    dialectic = sidecar.get("dialectic")
    synth = (dialectic.get("synthesis") if isinstance(dialectic, dict) else "") or ""
    prescriptive = " ".join(x for x in (synth, sidecar.get("bottom_line") or "")
                            if x.strip()).strip()
    if not prescriptive:
        return flags
    limits = [c for c in (list(sidecar.get("methodological_constraints") or [])
                          + [sidecar.get("stakes_and_policy") or ""])
              if CAPABILITY_DENIAL_RE.search(c)]
    for sent in limits:
        cap = _affirm_capability(sent)
        if cap.strip().lower() == sent.strip().lower():
            continue
        res = semantic.nli(prescriptive, cap)
        if res and res["label"] == "entailment" and res["score"] >= 0.6:
            flags.append(
                "SELF-CONTRADICTION: synthesis prescribes a capability the constraints "
                f"declare unreliable - constraints say '{sent[:140]}' while the synthesis "
                "assumes it.")
            break
    claims = [str(e.get("presupposition", "")).strip()
              for e in (sidecar.get("presupposition_audit") or [])
              if isinstance(e, dict) and str(e.get("presupposition", "")).strip()]
    for item in (sidecar.get("question_decomposition") or []):  # legacy shape
        text = str(item).strip()
        if text.lower().startswith("presupposition:"):
            claims.append(text.split(":", 1)[1].strip())
    for claim in claims:
        if len(claim) < 12:
            continue
        res = semantic.nli(prescriptive, claim)
        if res and res["label"] == "entailment" and res["score"] >= 0.6:
            flags.append(
                "ASSUMPTION CONFLICT: the analysis assumes the very premise the "
                f"question/data questions - it builds on the presupposition '{claim[:140]}'.")
            break
    return flags


MARKER_NOISE_RE = re.compile(
    r"\[(?:R\d+-[A-Z0-9-]+|FACT|PREDICTION|HYPOTHETICAL|POSSIBILITY|CONSTRAINT|"
    r"PRESUPPOSITION|AUTHOR_CLAIM|INFERENCE|SPECULATION|INFERRED ATTRIBUTION|"
    r"INFERRED VALUE)\]")


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
        if (source or "").strip() and _quote_in(claim, source):
            # a verbatim quote is entailed by identity: decided lexically,
            # no model call. Small NLI cross-encoders are unreliable on long
            # premises and label verbatim substrings as contradictions.
            row["nli_label"] = "entailment"
            row["nli_score"] = 1.0
            row["nli_path"] = "lexical-verbatim"
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
        row["nli_path"] = result.get("path", "model")
        if result["label"] == "contradiction":
            flags.append(f"CLAIM_CONTRADICTS_SOURCE: [{cid}] is labelled contradiction "
                         f"against the stored chunk (score {result['score']}).")
        elif result["label"] == "neutral":
            flags.append(f"CLAIM_UNSUPPORTED: [{cid}] is labelled neutral - the chunk "
                         f"neither supports nor contradicts the claim (score {result['score']}).")
    if any(row.get("nli_label") == "unavailable" for row in citations):
        flags.append("FALLBACK: NLI model unavailable; citation verification stays on the "
                     "lexical quote check (substring match against the stored chunk). "
                     "No semantic entailment label was invented.")
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
    if bottom and BOTTOM_VALIDATION_OPENER.match(bottom):
        violations.append("bottom_line opens by validating the user (rule 6b); it must lead "
                          "with the analytical conclusion")
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
    for f in self_referential_consistency_check(sidecar):
        violations.append(f)
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
