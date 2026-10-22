"""Phase 4: the Deterministic Editor - non-LLM logical rules applied to the generated content.

This is where the system stops being a language model and becomes a logic engine:
Guardrail 1: theory-indexed option matrix (options are conditional objects, never global).
Guardrail 2: substrate-test compatibility filter (architecturally invalid tests are blocked).
Guardrail 3: universal objection filter (asymmetric critique is rejected).
Guardrail 4: epistemic labeling enforcement (banned live-empirical vocabulary).
Guardrail 5: scope enforcement (no bloat on high-level feasibility questions).
"""

from __future__ import annotations

import re

from . import rules

MATRIX_VIABLE = "viable"
MATRIX_EXCLUDED = "excluded"
MATRIX_IRRELEVANT = "irrelevant"

TEST_BLOCK = "BLOCK"
TEST_FLAG = "FLAG AS THEORY-LADEN"
TEST_OK = "OK"

SCOPE_CAPS = {rules.HIGH_LEVEL_SCOPE: rules.HIGH_LEVEL_OPTION_CAP}


def parse_theory(line: str) -> dict:
    """Parse 'Theory: X | Core: ... | Requires: ... | Excludes: ... | Predicts: ...' into a dict."""
    parts = rules.split_entry(line)
    out = {"name": "", "core": "", "requires": "", "excludes": "", "predicts": "", "raw": str(line)}
    if not parts:
        return out
    key_map = {"theory": "name", "core": "core", "claim": "core", "requires": "requires", "necessary": "requires",
               "excludes": "excludes", "excluded": "excludes", "predicts": "predicts", "prediction": "predicts"}
    for part in parts:
        hit = re.match(r"^(?:theory|core|claim|requires|necessary conditions?|excludes?|excluded|predicts?|prediction)\s*[:\-]\s*(.+)$", part, re.I)
        if hit:
            label = part.split(":", 1)[0].strip().lower().split()[0]
            key = key_map.get(label, "core")
            out[key] = (out[key] + " " + hit.group(1).strip()).strip()
        elif not out["name"]:
            out["name"] = part
    if not out["name"]:
        out["name"] = str(line)[:60]
    return out


def theory_names(theories: list[str]) -> list[str]:
    return [parse_theory(t)["name"] for t in theories if str(t).strip()]


def _overlap(a: str, b: str) -> int:
    return len(rules.content_tokens(a) & rules.content_tokens(b))


def build_option_matrix(options: list[str], theories: list[str]) -> list[dict]:
    """Guardrail 1: every option is tagged [Viable under / Excluded under / Irrelevant under]. No global options."""
    parsed = [parse_theory(t) for t in theories if str(t).strip()]
    rows: list[dict] = []
    for option in options:
        if not str(option).strip():
            continue
        viable: list[str] = []
        excluded: list[str] = []
        irrelevant: list[str] = []
        for theory in parsed:
            name = theory["name"] or "unspecified theory"
            affinity = _overlap(option, theory["requires"]) + _overlap(option, theory["core"])
            friction = _overlap(option, theory["excludes"])
            contact = _overlap(option, theory["raw"])
            if friction >= 2 and friction > affinity:
                excluded.append(name)
            elif affinity >= 2:
                viable.append(name)
            elif contact <= 1:
                irrelevant.append(name)
        rows.append(
            {
                "option": str(option).strip(),
                "viable": viable,
                "excluded": excluded,
                "irrelevant": irrelevant,
                "indexed": bool(viable or excluded),
            }
        )
    return rows


def render_matrix(rows: list[dict]) -> list[str]:
    lines = ["| Engineering Approach | Viable Under | Excluded Under | Irrelevant Under |", "|---|---|---|---|"]
    for row in rows:
        lines.append(
            f"| {row['option']} | {'; '.join(row['viable']) or '-'} | {'; '.join(row['excluded']) or '-'} "
            f"| {'; '.join(row['irrelevant']) or '-'} |"
        )
    return lines


# Guardrail 2 rules: (test pattern, substrate pattern, status, reason)
SUBSTRATE_RULES = [
    (r"\bpci\b", r"feedforward|feed forward|transformer|non recurrent|non-recurrent", TEST_BLOCK,
     "PCI requires history-dependent recurrent plasticity; it is an invalid instrument for a feedforward architecture."),
    (r"\bpci\b", r"\btransformer\b", TEST_BLOCK,
     "PCI requires history-dependent recurrent plasticity; it is an invalid instrument for a transformer substrate."),
    (r"compute (?:exact )?\u03a6|calculate (?:exact )?\u03a6|exact \u03a6|integrated information.*exact|exact.*integrated information",
     r"scale|system size|large|whole brain|real[- ]scale|industrial|above toy|non[- ]toy|beyond toy|network of|millions|billions",
     TEST_BLOCK,
     "Computing exact Phi above toy-model scale is NP-hard; no current framework provides a tractable path. "
     "Replace with proxy metrics or theoretical bounds, tier: Architecturally Implausible (Current Frameworks)."),
    (r"tms[- ]?eeg|tms-peg|pms", r"silicon|software|digital|in[- ]silico|machine", TEST_FLAG,
     "The TMS-EEG protocol is not substrate-neutral: it encodes assumptions about recurrent cortical dynamics. "
     "Any silicon analog is theory-laden and must be labelled as such."),
]


def substrate_test_flags(tests: list[str], context: str = "") -> list[dict]:
    """Guardrail 2: cross-reference proposed tests with target architectures; block architecturally invalid tests."""
    blob = rules.normalize(context + " " + " ".join(str(t) for t in tests))
    flags: list[dict] = []
    for test in tests:
        text = str(test).strip()
        if not text:
            continue
        low = rules.normalize(text)
        status, reason = TEST_OK, ""
        for test_pattern, substrate_pattern, rule_status, rule_reason in SUBSTRATE_RULES:
            if re.search(test_pattern, low) and (re.search(substrate_pattern, low) or re.search(substrate_pattern, blob)):
                status, reason = rule_status, rule_reason
                break
        flags.append({"test": text, "status": status, "reason": reason})
    return flags


def universal_objections(theory_objections: list[str], theories: list[str], selected: list[str]) -> list[dict]:
    """Guardrail 3: if Theory A is rejected for flaw X, check whether the favored theories share flaw X.

    When they do, the critique is asymmetric and the draft must acknowledge that the flaw is universal.
    """
    favored = list(selected) or theories
    found: list[dict] = []
    for entry in theory_objections:
        parts = rules.split_entry(entry, 2)
        if len(parts) < 2:
            continue
        theory, flaw = parts[0], parts[1]
        flaw_words = rules.content_tokens(flaw)
        if not flaw_words:
            continue
        shared = [
            parse_theory(t)["name"]
            for t in favored
            if parse_theory(t)["name"] != theory and _overlap(flaw, t) >= 2
        ]
        if shared:
            found.append(
                {
                    "flaw": flaw,
                    "rejected_theory": theory,
                    "also_shared_by": shared,
                    "acknowledgment": (
                        f"The flaw '{flaw}' is not specific to {theory}: "
                        + ", ".join(shared)
                        + " share it. Any objection built on it is a universal constraint, not a refutation of one theory."
                    ),
                }
            )
    return found


def label_violations(text: str) -> list[str]:
    """Guardrail 4: epistemic labeling enforcement - banned live-empirical vocabulary in parametric output."""
    return rules.banned_vocabulary(text)


def scope_violations(scope: str, sections: dict[str, str]) -> list[str]:
    """Guardrail 5: High-Level Feasibility output is capped: at most 4 strategic directions, at most 5 key unknowns."""
    violations: list[str] = []
    cap = SCOPE_CAPS.get(scope)
    if cap:
        core = sections.get("core", "")
        bullets = _bullet_count(core)
        if bullets > cap:
            violations.append(
                f"Scope creep: intent scope is '{scope}' but the Core Analysis carries {bullets} bullet directions (cap {cap})."
            )
    unknowns = _bullet_count(sections.get("unknowns", ""))
    if unknowns > rules.KEY_UNKNOWN_CAP:
        violations.append(
            f"Key Unknowns must be a prioritized list of at most {rules.KEY_UNKNOWN_CAP}, got {unknowns}."
        )
    return violations


def _bullet_count(text: str) -> int:
    if not text:
        return 0
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    bullets = [line for line in lines if re.match(r"^[-*\u2022]|\d{1,2}[.)]\s", line)]
    return len(bullets) if bullets else (1 if text.strip() else 0)


def tier_justification_gaps(tiers: list[str]) -> list[str]:
    """Every tier label must carry a one-line justification; the Editor rejects tier labels without one."""
    gaps: list[str] = []
    for entry in tiers:
        parts = rules.split_entry(entry, 3)
        if len(parts) < 2:
            gaps.append(str(entry))
            continue
        claim, tier_raw = parts[0], parts[1]
        tier = rules.parse_tier(tier_raw)
        justification = parts[2] if len(parts) > 2 else ""
        if not tier:
            gaps.append(f"{claim} (unrecognized tier '{tier_raw}')")
        elif not justification:
            gaps.append(f"{claim} (tier '{tier}' has no justification)")
    return gaps


def modal_tag_gaps(modal_claims: list[str]) -> list[str]:
    valid = {tag.lower() for tag in rules.MODAL_TAGS}
    gaps: list[str] = []
    for entry in modal_claims:
        parts = rules.split_entry(entry, 2)
        if len(parts) < 2:
            gaps.append(str(entry))
            continue
        claim, tag = parts[0], parts[1]
        if tag.strip().lower() not in valid:
            gaps.append(f"{claim} (modal tag '{tag}' is not one of {rules.MODAL_TAGS})")
    return gaps
