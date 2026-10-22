"""Phase 6 (deterministic layer): the mechanical half of the Presentation Critic's checklist.

The checks that a Python rule can decide are decided here, not asked of the LLM: section contract,
provenance vocabulary, temporal disclaimer, modal tags, tier justifications, landmark status,
bottom-line shape, scope caps, leaks, and redundancy. The LLM critic receives these verdicts as
settled and judges only the checks that need reading: question fidelity, symmetry, fairness.
"""

from __future__ import annotations

import re

from . import guardrails, rules, scrubber

# The output contract (Part 3), in mandatory order.
SECTION_TITLES = [
    ("bottom", "Bottom Line"),
    ("status", "Epistemic & Temporal Status"),
    ("decomposition", "Question Decomposition"),
    ("core", "Core Analysis"),
    ("matrix", "Theory-Indexed Feasibility Matrix"),
    ("constraints", "Methodological Constraints"),
    ("unknowns", "Key Unknowns"),
    ("change", "What Would Change This Answer"),
    ("stakes", "Stakes & Policy"),
]

SECTION_TITLES_BY_ID = dict(SECTION_TITLES)

LABEL_ALIASES = {
    "bottom line": "bottom",
    "bottom-line": "bottom",
    "epistemic & temporal status": "status",
    "epistemic and temporal status": "status",
    "epistemic status": "status",
    "temporal status": "status",
    "question decomposition": "decomposition",
    "decomposition": "decomposition",
    "sub questions": "decomposition",
    "core analysis": "core",
    "analysis": "core",
    "theory indexed feasibility matrix": "matrix",
    "theory indexed matrix": "matrix",
    "feasibility matrix": "matrix",
    "matrix": "matrix",
    "methodological constraints": "constraints",
    "methodology constraints": "constraints",
    "constraints": "constraints",
    "key unknowns": "unknowns",
    "unknowns": "unknowns",
    "what would change this answer": "change",
    "what would change": "change",
    "stakes & policy": "stakes",
    "stakes and policy": "stakes",
    "stakes": "stakes",
}

def label_key(text: str) -> str:
    """Canonical form of a section label: the same normalisation the headers get."""
    return " ".join(rules.normalize(str(text)).split())


LABEL_LOOKUP = {label_key(alias): sid for alias, sid in LABEL_ALIASES.items()}

HEADER = re.compile(r"^\s*(?:#{1,6}\s*|[-*>\s]*)?([A-Za-z][A-Za-z&/ ,+\-]{2,60}?)\s*:\s*(.*)$")

DIRECT_ANSWER_HEADS = ("yes", "no", "not currently", "not yet", "not with", "it depends", "it depend", "no -", "no,", "not ")

SENTENCE_SPLIT = re.compile(r"(?<=[.?!])\s+")


def split_sections(text: str) -> dict[str, str]:
    """Map the drafted answer onto the output-contract sections by header."""
    sections: dict[str, list[str]] = {}
    current = "preamble"
    for line in (text or "").splitlines():
        hit = HEADER.match(line)
        if hit:
            label = label_key(hit.group(1))
            if label in LABEL_LOOKUP:
                current = LABEL_LOOKUP[label]
                rest = hit.group(2).strip()
                sections.setdefault(current, [])
                if rest:
                    sections[current].append(rest)
                continue
        sections.setdefault(current, []).append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items() if k != "preamble" and "\n".join(v).strip()}


def split_ordered(text: str) -> list[tuple[str, str, list[str]]]:
    """[(section_id, header_line, body lines)] in document order; unlabelled text keeps id ''."""
    items: list[tuple[str, str, list[str]]] = []
    current_id, current_header, current_lines = "", "", []
    for line in (text or "").splitlines():
        hit = HEADER.match(line)
        label = label_key(hit.group(1)) if hit else ""
        if hit and label in LABEL_LOOKUP:
            items.append((current_id, current_header, current_lines))
            current_id, current_header, current_lines = LABEL_LOOKUP[label], line.strip(), []
            continue
        current_lines.append(line)
    items.append((current_id, current_header, current_lines))
    return items


def splice_section(text: str, section_id: str, new_section: str) -> str:
    """Replace one section (header + body) in place; append it if the section was missing entirely."""
    items = split_ordered(text)
    rebuilt: list[str] = []
    replaced = False
    lines = [line for line in (new_section or "").strip().splitlines()]
    new_header = lines[0].strip() if lines and lines[0].strip().rstrip().endswith(":") else ""
    new_body = "\n".join(lines[1:]) if new_header else "\n".join(lines)
    for sec_id, header, body in items:
        if sec_id == section_id and sec_id and not replaced:
            rebuilt.append("\n".join([new_header or header, new_body]).strip())
            replaced = True
            continue
        rebuilt.append(("\n".join([header, *body]).strip() if header else "\n".join(body).strip()))
    if not replaced:
        rebuilt.append(f"{SECTION_TITLES_BY_ID.get(section_id, section_id)}:\n{new_body.strip()}")
    return "\n\n".join(part for part in rebuilt if part).strip()


def required_sections(stakes_high: bool) -> list[str]:
    ids = [sid for sid, _ in SECTION_TITLES if sid != "stakes"]
    if stakes_high:
        ids.append("stakes")
    return ids


def bottom_line_shape(text: str) -> tuple[int, bool, str]:
    """(sentence_count, direct_opening, first_words)."""
    body = text.strip()
    sentences = [s.strip() for s in SENTENCE_SPLIT.split(body) if s.strip() and re.search(r"[A-Za-z]", s)]
    head = body[:40].strip().lower()
    direct = any(head.startswith(prefix) for prefix in DIRECT_ANSWER_HEADS)
    return len(sentences), direct, body[:60]


def untagged_prediction_sentences(text: str) -> list[str]:
    """Sentences that state a future behaviour without a modal tag: the modal-collapse signature."""
    hits: list[str] = []
    for sentence in SENTENCE_SPLIT.split(text or ""):
        if re.search(r"\b(will|would|going to)\s+(report|experience|be conscious|be aware|feel|be sentient)\b", sentence, re.I):
            if not any(tag.lower() in sentence.lower() for tag in rules.MODAL_TAGS) and not re.search(
                r"\b(predicts?|theory|framework|under\b|claims?)\b", sentence, re.I
            ):
                hits.append(sentence.strip())
    return hits


def conscious_conflation(text: str) -> list[str]:
    """Sentences that say 'is conscious' without 'under Theory X' / criteria qualification."""
    hits: list[str] = []
    for sentence in SENTENCE_SPLIT.split(text or ""):
        if re.search(r"\b(is|are|was|were|would be|will be)\s+(a\s+)?conscious\b", sentence, re.I):
            if not re.search(r"\b(under|according to|criteria|predicts?|theory|framework|claims?|proposes?)\b", sentence, re.I):
                hits.append(sentence.strip())
    return hits


def landmark_gaps(landmarks: list[str], required: list[str]) -> list[str]:
    """Each mandatory landmark must be ENGAGED (where) or DISMISSED (with a one-line justification)."""
    gaps: list[str] = []
    stated: dict[str, str] = {}
    for entry in landmarks:
        parts = rules.split_entry(entry, 2)
        if len(parts) < 2:
            continue
        concept, status = parts[0], parts[1]
        stated[rules.normalize(concept)] = status
    for concept in required:
        needle = rules.normalize(concept)
        key = next((k for k in stated if k == needle or needle in k or k in needle), None)
        if key is None:
            gaps.append(f"{concept}: not engaged and not dismissed")
            continue
        status = stated[key].upper()
        if status.startswith("ENGAGED"):
            if len(rules.content_tokens(stated[key])) <= 1:
                gaps.append(f"{concept}: ENGAGED without stating where")
        elif status.startswith("DISMISSED"):
            if len(rules.content_tokens(stated[key])) <= 2:
                gaps.append(f"{concept}: DISMISSED without a justification")
        else:
            gaps.append(f"{concept}: status must be ENGAGED or DISMISSED, got '{stated[key]}'")
    return gaps


def run(case) -> list[dict]:
    """Deterministic checklist verdicts: [{id, name, pass, note, section}]."""
    text = case.final_response or ""
    sections = split_sections(text)
    core = sections.get("core", "") + "\n" + sections.get("matrix", "")
    stakes_high = str((case.route or {}).get("STAKES", "")) == rules.STAKES_HIGH
    scope = str((case.route or {}).get("INTENT_SCOPE", rules.HIGH_LEVEL_SCOPE))
    ontology = case.ontology or {}
    checks: list[dict] = []

    def add(check_id: int, name: str, ok: bool, note: str = "", section: str = "") -> None:
        checks.append({"id": check_id, "name": name, "pass": bool(ok), "note": note, "section": section})

    missing = [sid for sid in required_sections(stakes_high) if sid not in sections]
    add(13, "Output contract: all mandatory sections present (What Would Change This Answer included)",
        not missing and "change" in sections,
        ("missing sections: " + ", ".join(missing)) if missing else "", "core")

    add(12, "Temporal disclaimer present",
        "parametric knowledge up to" in rules.normalize(text),
        "", "status")

    add(11, "No banned live-empirical vocabulary",
        not rules.banned_vocabulary(text),
        "; ".join(rules.banned_vocabulary(text)[:6]), "core")

    tags = [tag for tag in rules.MODAL_TAGS if tag.lower() in rules.normalize(core + sections.get("bottom", ""))]
    untagged = untagged_prediction_sentences(core + sections.get("bottom", ""))
    add(4, "Theory predictions are tagged as predictions, not facts",
        bool(tags) and not untagged,
        (f"modal tags present: {', '.join(tags)}; " if tags else "no modal tag found; ")
        + ("untagged prediction sentences: " + " | ".join(untagged[:3]) if untagged else ""),
        "core")

    uses_syllogism = bool(re.search(r"\b(premise \d|conclusion:)\b", text, re.I))
    anchor_status = str(ontology.get("ANCHOR_MODAL_STATUS", "")).strip()
    labeled = bool(anchor_status) or bool(
        re.search(r"hypothetical|theory[- ]internal|necessary[- ]condition|actual state", text, re.I)
    )
    add(5, "Every syllogism is labelled with its modal status",
        (not uses_syllogism) or labeled,
        "anchor modal status: " + (anchor_status or "stated in prose") if labeled else "syllogism used with no modal status label",
        "core")

    conflated = conscious_conflation(text)
    add(6, "'Satisfies criteria under Theory X' is never written as 'is conscious'",
        not conflated, "; ".join(conflated[:3]), "core")

    gaps = guardrails.tier_justification_gaps(list(ontology.get("TIERS") or []))
    tier_in_text = any(rules.normalize(tier) in rules.normalize(text) for tier in rules.TIERS)
    add(10, "Qualitative tiers assigned and each carries a one-line justification",
        not gaps and (tier_in_text or not ontology.get("TIERS")),
        ("tier gaps: " + "; ".join(gaps[:4])) if gaps else ("" if tier_in_text else "no tier label appears in the answer"),
        "core")

    scope_hits = guardrails.scope_violations(scope, sections)
    add(14, f"Output respects the intent scope ({scope})", not scope_hits, "; ".join(scope_hits[:4]), "core")

    leaked = scrubber.scrub_meta(text)[1]
    add(15, "No leaked prompt directives in the prose", not leaked, "; ".join(leaked[:4]), "core")

    dupes = scrubber.dedupe_paragraphs(text)[1]
    add(16, "No semantic redundancy above the threshold", not dupes, f"{len(dupes)} duplicate paragraph(s) present", "core")

    count, direct, head = bottom_line_shape(sections.get("bottom", ""))
    add(17, "Bottom Line is direct and within the 2-4 sentence range",
        sections.get("bottom", "") != "" and direct and 2 <= count <= 4,
        f"{count} sentences, direct opening: {direct} ('{head}...')", "bottom")

    required = rules.landmark_concepts(str((case.route or {}).get("DEPTH_TRIGGER", "NONE")))
    lm_gaps = landmark_gaps(list(ontology.get("LANDMARKS") or []), required)
    add(18, "Mandatory landmark concepts are engaged or explicitly dismissed with justification",
        not lm_gaps, "; ".join(lm_gaps[:4]), "core")

    if stakes_high:
        stakes_text = sections.get("stakes", "")
        add(19, "High Moral / Ontological Stakes flagged: Stakes & Policy section present",
            bool(stakes_text), "section missing" if not stakes_text else "", "stakes")
        words = rules.content_tokens(stakes_text)
        ethical = words & {"moral", "patienthood", "patient", "welfare", "ethical", "obligation", "obligations", "rights", "suffering"}
        add(20, "Stakes section addresses moral patienthood, welfare, or ethical obligations",
            len(ethical) >= 2, "few ethical terms in the section" if len(ethical) < 2 else "", "stakes")

    objections = (case.guardrails or {}).get("universal_objections") or []
    if objections:
        shared_words: set[str] = set()
        for obj in objections:
            shared_words |= rules.content_tokens(obj["flaw"])
        in_answer = rules.content_tokens(sections.get("core", "")) | rules.content_tokens(sections.get("unknowns", ""))
        present = len(shared_words & in_answer) >= 2
        add(9, "The measurement problem is acknowledged as a universal constraint across all frameworks",
            present,
            "universal objections found by the guardrail are not acknowledged in the answer"
            if not present else "",
            "core")
    return checks


def failed(checks: list[dict]) -> list[dict]:
    return [c for c in checks if not c.get("pass")]


def checklist_digest(checks: list[dict]) -> str:
    lines = []
    for c in checks:
        lines.append(f"Check {c['id']}: {'PASS' if c.get('pass') else 'FAIL'}" + (f" - {c.get('note')}" if c.get("note") else ""))
    return "\n".join(lines)
