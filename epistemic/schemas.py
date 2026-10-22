"""Field specifications for each role packet (Phases 0-6 of the parametric architecture)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class FieldSpec:
    name: str
    kind: str = "text"
    hint: str = ""
    required: bool = True


ROUTES = ("SIMPLE", "REASONING", "FACTUAL", "ENGINE", "DIVERGENT")
REASONING_ROUTES = {"ENGINE", "REASONING", "DIVERGENT"}
TERMINAL_ROUTES = {"SIMPLE", "FACTUAL"}

# Phase 0: triage, scoping & stakes.
TRIAGE = [
    FieldSpec("ROUTE", hint="SIMPLE, FACTUAL, REASONING (aka ENGINE / DIVERGENT)"),
    FieldSpec("REASON"),
    FieldSpec("COMPLEXITY", hint="LOW, MEDIUM or HIGH"),
    FieldSpec("RISK_LEVEL", hint="LOW, MEDIUM or HIGH"),
    FieldSpec("DEPTH_TRIGGER", hint="the deep-frame trigger name, or NONE"),
    FieldSpec("INTENT_SCOPE", hint="High-Level Feasibility, Applied Methodology, or Conceptual Clarification"),
    FieldSpec("STAKES", hint="'High Moral / Ontological Stakes' or 'Standard'"),
    FieldSpec("TERMINATION_SIGNAL", hint="NONE, or a short uppercase signal phrase"),
]

# Phase 1: skeptic & question decomposition.
SKEPTIC = [
    FieldSpec("NEUTRAL_QUESTION"),
    FieldSpec(
        "SUB_QUESTIONS",
        "list",
        "the decomposed sub-problems, one per line, each 'label: question', e.g. 'Design Methodology: ...'",
    ),
    FieldSpec(
        "CLAIM_TYPOLOGY",
        "list",
        "every extracted premise tagged: 'claim | Definitional | Conceptual/Philosophical | Empirical/Scientific | Technical/Engineering'",
    ),
    FieldSpec("REQUESTED_REGISTER"),
    FieldSpec("REQUESTED_TASK_SHAPE"),
    FieldSpec("HIDDEN_AXIOMS", "list"),
    FieldSpec("EPISTEMIC_STANCE"),
    FieldSpec("CAUSAL_ATTRIBUTIONS", "list"),
    FieldSpec("MODAL_BOUNDARIES", "list"),
    FieldSpec("EXCLUDED_OPTIONS", "list"),
    FieldSpec("LOADED_TERMS", "list"),
    FieldSpec(
        "VOCABULARY_CONTRACT",
        "list",
        "one line per substitution: 'original term | substituted term | reason | changes the question (YES/NO) | original question still addressed (YES/NO)'",
    ),
    FieldSpec("FALLACY_FLAGS", "list"),
    FieldSpec("TERMINATION_SIGNAL"),
]

# Phase 2: parametric ontology mapper (map pass, then distillation pass).
ONTOLOGY_MAP = [
    FieldSpec(
        "ONTOLOGY_MAP",
        "list",
        "one line per theory, landmark concept, active frontier, and option; each line states the claim and what it predicts or excludes",
    ),
]

ONTOLOGY = [
    FieldSpec("SUMMARY"),
    FieldSpec(
        "LANDMARKS",
        "list",
        "one line per mandatory landmark concept: 'concept | ENGAGED: where and how' or 'concept | DISMISSED: the one-line justification'",
    ),
    FieldSpec(
        "THEORIES",
        "list",
        "one line per theory: 'Theory: Name | Core: core claim | Requires: necessary conditions | Excludes: what it rules out | Predicts: prediction (tagged as prediction, not fact)'",
    ),
    FieldSpec("FRONTIERS", "list", "ongoing debates and active research frontiers, framed as unresolved at the cutoff, never as settled results"),
    FieldSpec(
        "TIERS",
        "list",
        "one line per major claim: 'claim | tier | one-line justification'; tiers are exactly: Established Consensus; Theoretical Plausibility; Highly Contested; Conceptually Open; Architecturally Implausible (Current Frameworks)",
    ),
    FieldSpec(
        "MODAL_CLAIMS",
        "list",
        "one line per major reasoning claim: 'claim | modal tag' where the tag is one of: Actual Fact; Theory Prediction; Hypothetical Schema; Conceptual Possibility; Architectural Constraint",
    ),
    FieldSpec(
        "ANCHOR_MODAL_STATUS",
        hint="the modal status of the logic anchor: actual state, hypothetical schema, theory-internal consequence, or necessary-condition schema",
        required=False,
    ),
    FieldSpec(
        "LOGICAL_SYLLOGISM",
        "list",
        "the causal chain in formal form, one line per move: 'Premise 1: ...', 'Premise 2: ...', 'Conclusion: ...'",
        required=False,
    ),
    FieldSpec(
        "PSEUDO_CODE",
        "list",
        "only when no syllogism fits: numbered causal steps, e.g. '1. measure X', '2. if X then model A else model B'",
        required=False,
    ),
    FieldSpec("KEY_UNKNOWNS", "list", "at most five, prioritized"),
    FieldSpec("WHAT_WOULD_CHANGE", "list", "specific results, breakthroughs, or tests that would overturn the conclusion"),
    FieldSpec(
        "PROPOSED_TESTS",
        "list",
        "candidate tests and measurements, one per line; name the target substrate or architecture when relevant (e.g. 'PCI applied to a feedforward transformer')",
    ),
    FieldSpec("LIVE_OPTIONS", "list"),
    FieldSpec("REJECTED_OR_WEAK_OPTIONS", "list"),
    FieldSpec("CLAIMS_TO_AVOID", "list"),
    FieldSpec(
        "THEORY_OBJECTIONS",
        "list",
        "one line per objection: 'theory | flaw the theory is rejected for' - the symmetry filter reads these",
    ),
]

# Phase 3: dialectical engine & modal logic.
DIALECTIC = [
    FieldSpec("THESIS", hint="the primary framework's position, in one paragraph"),
    FieldSpec("THESIS_AXIOMS", "list", "the core axioms of the thesis, one per line"),
    FieldSpec("ANTITHESIS", hint="a position that explicitly negates at least one thesis axiom"),
    FieldSpec("ANTITHESIS_NEGATES", "list", "which thesis axioms the antithesis denies, quoted"),
    FieldSpec("SYNTHESIS", hint="what survives of thesis and antithesis; state the tension that remains unresolved"),
    FieldSpec("TENSION_STATUS", hint="RESOLVED, PRESERVED, or UNRESOLVED - never hidden"),
]

WRITER = [
    FieldSpec(
        "THINKING_PROCESS",
        hint="private deliberation as numbered moves: 'Step 1: ...', 'Step 2: ...', each step one move of the chain",
        required=False,
    ),
    FieldSpec("RESPONSE", "list", "the final user-facing answer, section by section, in the output-contract order"),
]

# Phase 6: presentation critic. CHECKS reuses the step-verdict parsing: one verdict per checklist id.
CRITIC = [
    FieldSpec(
        "CHECKS",
        "steps",
        'one object per checklist item: {"step": 1, "valid": true}, '
        '{"step": 7, "valid": false, "reason": "Rejected Theory A for a flaw Theory B shares"}',
        required=False,
    ),
    FieldSpec("RESULT", hint="PASS if every checklist item passes, else FAIL"),
    FieldSpec("VIOLATIONS", "list"),
    FieldSpec("REWRITE_REQUIRED", hint="YES or NO"),
    FieldSpec("REWRITE_INSTRUCTION"),
    FieldSpec(
        "SECTION_TARGETS",
        "list",
        "the section names whose content must be rewritten, chosen from: bottom, status, decomposition, core, matrix, constraints, unknowns, change, stakes",
        required=False,
    ),
]

CONTINUATION = [
    FieldSpec("NEW_EVIDENCE", hint="YES or NO"),
    FieldSpec("NEW_CONSTRAINT", hint="YES or NO"),
    FieldSpec("PUSHBACK_ONLY", hint="YES or NO"),
    FieldSpec("NEW_QUESTION", hint="YES if the user asks something materially new, else NO"),
    FieldSpec("MODE", hint="evidence-bounded, user-forced, hypothetical, speculative or preference-based"),
    FieldSpec("REASON"),
]


STEP_VERDICT = re.compile(
    r"^\s*(?:[-*#>\s]*)?(?:step|check|item|rule)\s*(?P<step>\d{1,2})\s*[.):=\-]?\s*(?P<rest>.*)$",
    re.I,
)
VALID_WORDS = {"valid", "true", "ok", "sound", "pass", "passed", "yes", "fine", "accepted"}
INVALID_WORDS = {"invalid", "false", "fail", "failed", "no", "broken", "unsound", "rejected", "flawed"}


def _as_step_no(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    hit = re.search(r"\d{1,2}", str(value or ""))
    return int(hit.group(0)) if hit else None


def _as_valid(value, rest: str = "") -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = " ".join([str(value or ""), rest]).strip().lower()
    text = re.sub(r"^[^a-z]+", "", text)
    if not text:
        return None
    head = re.split(r"[\s:,\-\u2192]+", text)[0]
    if head in INVALID_WORDS or text.startswith(("not valid", "not sound", "not pass")):
        return False
    if head in VALID_WORDS or text.startswith(("is valid", "is sound", "is pass")):
        return True
    if "invalid" in text or "not valid" in text:
        return False
    return True


def _loads(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"```\s*$", "", text.strip())
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    for open_c, close_c in (("[", "]"), ("{", "}")):
        start, end = text.find(open_c), text.rfind(close_c)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None


def _split_verdict(rest: str) -> tuple[str, str]:
    parts = re.split(r"\s+[-:\u2192]\s+|\s+because\s+|\s{2,}", rest, maxsplit=1)
    if len(parts) == 1:
        return parts[0].strip(), ""
    return parts[0].strip(), parts[1].strip()


def parse_step_verdicts(raw: str | list) -> list[dict]:
    """Coerce model output into [{step, valid, reason}] from JSON-like items or from 'Step N: INVALID - reason' lines."""
    items: list = []
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("[") or text.startswith("{"):
            data = _loads(text)
            if isinstance(data, dict):
                data = [data]
            items = list(data) if isinstance(data, list) else text.splitlines()
        else:
            items = text.splitlines()
    else:
        items = list(raw or [])
    out: list[dict] = []
    for item in items:
        if isinstance(item, dict):
            step = _as_step_no(item.get("step", item.get("Step", item.get("check"))))
            valid = _as_valid(item.get("valid", item.get("Valid", item.get("pass"))))
            reason = str(item.get("reason") or item.get("reason_for_failure") or "").strip()
        else:
            line = str(item).strip().strip("{}[]\"' ")
            if not line:
                continue
            hit = STEP_VERDICT.match(line)
            if not hit:
                continue
            step = _as_step_no(hit.group("step"))
            head, reason = _split_verdict(hit.group("rest").strip().strip("{}\"'"))
            valid = _as_valid(head)
        if step is None or valid is None:
            continue
        out.append({"step": step, "valid": valid, "reason": reason})
    return out


# The presentation critic reads checklist ids, not reasoning steps; the parsing is identical.
parse_check_verdicts = parse_step_verdicts


def validate_fields(data: dict, spec: list[FieldSpec], partial: bool = False) -> dict | None:
    """Fill the packet. partial=True accepts a truncated packet: a field with some content is kept,
    an empty required field becomes an empty default instead of failing the whole call."""
    if not isinstance(data, dict):
        return None
    out: dict = {}
    for f in spec:
        val = data.get(f.name)
        if val is None:
            if not f.required or partial:
                out[f.name] = [] if f.kind in {"list", "steps"} else ""
                continue
            return None
        if f.kind == "steps":
            out[f.name] = parse_step_verdicts(val if isinstance(val, (list, str)) else [val])
        elif f.kind == "list":
            if isinstance(val, str):
                items = [p.strip() for p in val.splitlines() if p.strip()]
            elif isinstance(val, list):
                items = [str(p).strip() for p in val if str(p).strip()]
            else:
                if not partial:
                    return None
                items = []
            out[f.name] = [i.lstrip("-").strip() for i in items if i.lstrip("- ").upper() not in {"NONE", "N/A", ""}]
        else:
            out[f.name] = str(val).strip()
    if not partial and not any(out[f.name] for f in spec if f.required):
        return None
    return out


def text_format_block(spec: list[FieldSpec]) -> str:
    lines = ["Output exactly these fields, one per line, using the exact field names:"]
    for f in spec:
        hint = f" ({f.hint})" if f.hint else ""
        if f.kind == "list":
            lines.append(f"{f.name}: <one item per line, each line prefixed with '-'>{hint}")
        elif f.kind == "steps":
            lines.append(
                f"{f.name}: <one line per checklist item: 'Check 1: PASS', "
                f"'Check 7: FAIL - <the reason in one line>'>{hint}"
            )
        else:
            lines.append(f"{f.name}: <value>{hint}")
    lines.append("If a field has no content, write NONE.")
    lines.append("Output no field twice. Output no extra fields.")
    return "\n".join(lines)
