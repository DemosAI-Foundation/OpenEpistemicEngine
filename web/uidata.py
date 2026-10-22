"""Case -> UI payload mapping for the web interface."""

from __future__ import annotations

import re

from epistemic import checks as contract
from epistemic.casefile import CaseFile, title_of
from epistemic.pipeline import FORCING_MODES

VOCABULARY_ISSUES = {
    "definitely": "Asserts certainty the evidence does not support.",
    "all": "Universal quantifier over a contested set.",
    "replace": "Assumes total substitution rather than partial automation.",
    "survive": "Catastrophic framing of an adaptive decision.",
    "feel like": "Treats a feeling as evidence.",
    "just know": "Treats an unverified belief as evidence.",
    "down": "States a direction of change without a measurement.",
    "convince": "Persuasion goal in place of an evidence-based recommendation.",
    "persuade": "Persuasion goal in place of an evidence-based recommendation.",
    "bad": "Evaluative label with no measured criterion.",
    "immediately": "Urgency in place of verification.",
    "require": "Mandates a policy before the cause is tested.",
    "mandate": "Mandates a policy before the cause is tested.",
    "lazy": "Character attribute in place of a measured performance problem.",
    "toxic culture": "Broad label with no specific evidence.",
    "killing innovation": "Causal claim stated as a metaphor.",
    "blunder": "Implies a known solution was missed; may be an unsolved theoretical limit.",
    "design": "Assumes the target is an engineering object.",
    "crash": "Presents a forecast as settled.",
    "hate": "Attributes a group sentiment without measurement.",
    "ruining": "Single-cause harm claim.",
    "because of": "States a causal relation as established.",
}

WARNING_MARKERS = ("termination warning", "epistemic warning", "warning:")

BULLETED_SECTIONS = {"decomposition", "constraints", "unknowns", "change", "stakes"}


TIER_STRENGTH = {
    "Established Consensus": "settled at the cutoff",
    "Theoretical Plausibility": "plausible inside a theory, not established",
    "Highly Contested": "contested across frameworks",
    "Conceptually Open": "open question, no framework settles it",
    "Architecturally Implausible (Current Frameworks)": "implausible under current architectures",
}


def tier_label(tier: str | None) -> str:
    key = str(tier or "").strip()
    if not key:
        return "Untiered"
    return TIER_STRENGTH.get(key, key)


def classify_warning(text: str) -> tuple[str, str, str]:
    low = text.lower()
    if low.startswith("this requires external factual verification"):
        return "knowledge-boundary", "medium", "Outside the parametric boundary"
    if low.startswith("overthinking monitor"):
        return "overthinking", "medium", "Overthinking stream aborted"
    if low.startswith("termination warning"):
        if "repeated" in low:
            return "termination", "medium", "Repeated request"
        if "binary" in low:
            return "termination", "medium", "Forced binary answer"
        if "user-forced" in low or "hypothetical" in low or "mode" in low:
            return "user-override", "medium", "User override"
        return "termination", "soft", "Premature closure"
    if "intuition" in low or "feeling" in low or "anecdote" in low:
        return "epistemic", "medium", "Weak evidence standard"
    if "evidence quality" in low:
        return "evidence-gap", "soft", "Evidence gap"
    if "certainty" in low:
        return "epistemic", "soft", "Certainty demand"
    if "caus" in low:
        return "causal", "medium", "Causal claim unverified"
    if "option" in low or "solution space" in low:
        return "modal", "soft", "Narrow solution space"
    if "unproven" in low or "rejected" in low:
        return "premise", "medium", "Premise unverified"
    return "epistemic", "soft", "Epistemic note"


def build_warnings(case: CaseFile) -> list[dict]:
    out = []
    for text in case.termination_warnings:
        kind, severity, title = classify_warning(text)
        out.append(
            {
                "type": kind,
                "severity": severity,
                "title": title,
                "text": text,
                "blocking": False,
                "actions": ["add-evidence", "reframe", "continue-anyway"],
            }
        )
    over = case.overthinking or {}
    if over.get("aborted"):
        text = (
            f"Overthinking monitor: {over.get('reason', '')}. The stream was stopped and the answer truncated "
            f"at the last stable paragraph ({over.get('dropped_paragraphs', 0)} dropped)."
        )
        kind, severity, title = classify_warning(text)
        out.append({"type": kind, "severity": severity, "title": title, "text": text, "blocking": False, "actions": ["reframe"]})
    for repair in case.repairs:
        section = str(repair.get("section") or "core")
        text = (
            f"Surgical section repair: the '{section}' section failed the presentation critic ({repair.get('reason')}). "
            "That section was rewritten in place; every other section is unchanged."
        )
        out.append({"type": "reasoning-repair", "severity": "soft", "title": "One section rewritten", "text": text, "blocking": False, "actions": []})
    banner_text = str(case.warning_banner or "").strip()
    if banner_text:
        kind, severity, title = classify_warning("Termination warning: " + banner_text)
        out.append(
            {
                "type": kind,
                "severity": severity,
                "title": "Unresolved critic violations",
                "text": banner_text,
                "blocking": False,
                "actions": ["add-evidence", "reframe", "continue-anyway"],
            }
        )
    if str(case.ontology.get("_logic") or "") == "missing" and (case.route or {}).get("ROUTE") in {"ENGINE", "REASONING"}:
        text = (
            "No formal logic anchor: the Ontology pass produced neither a LOGICAL_SYLLOGISM nor PSEUDO_CODE, "
            "so the causal chain is unstated and is carried as a hypothesis."
        )
        out.append({"type": "causal", "severity": "medium", "title": "Logic chain unstated", "text": text, "blocking": False, "actions": ["reframe"]})
    for text in list(case.editor.get("GUARDRAIL_VIOLATIONS") or [])[:4]:
        kind, severity, title = classify_warning(text)
        out.append({"type": kind, "severity": severity, "title": title, "text": text, "blocking": False, "actions": ["reframe"]})
    for text in list(case.editor.get("EPISTEMIC_WARNINGS") or [])[:4]:
        kind, severity, title = classify_warning(text)
        out.append({"type": kind, "severity": severity, "title": title, "text": text, "blocking": False, "actions": ["add-evidence", "reframe"]})
    return out


def build_sections(case: CaseFile) -> tuple[list[dict], dict]:
    """Map the final answer onto the output-contract sections the Critic checked."""
    banner: dict = {}
    sections: list[dict] = []
    preamble: list[str] = []
    for sid, _header, lines in contract.split_ordered(case.final_response or ""):
        body = "\n".join(lines).strip()
        if sid:
            title = contract.SECTION_TITLES_BY_ID.get(sid, sid)
            items = bullet_list(body)
            if sid in BULLETED_SECTIONS and len(items) > 1:
                sections.append({"id": sid, "title": title, "items": items})
            else:
                sections.append({"id": sid, "title": title, "text": strip_label(body, title)})
            continue
        if body:
            preamble.append(body)
    rest: list[str] = []
    for para in preamble:
        for chunk in [p.strip() for p in re.split(r"\n\s*\n", para) if p.strip()]:
            if not banner and any(marker in chunk.lower() for marker in WARNING_MARKERS):
                banner = {"text": chunk, "type": classify_warning("Termination warning: " + chunk)[0]}
                continue
            rest.append(chunk)
    if rest:
        sections.insert(min(len(sections), 1), {"id": "answer", "title": "Analysis", "text": "\n\n".join(rest)})
    return sections, banner


def strip_label(text: str, title: str) -> str:
    label = re.escape(title.lower())
    return re.sub(rf"^{label}\s*:\s*", "", text.strip(), count=1, flags=re.I).strip()


def strip_heading(text: str) -> str:
    return re.sub(r"^[^:]{0,60}:\s*", "", text, count=1).strip()


def bullet_list(text: str) -> list[str]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    items: list[str] = []
    for line in lines:
        parts = re.split(r"\s+(?=\d{1,2}[.)]\s)|(?<=[.;])\s{2,}", line)
        for part in parts:
            cleaned = part.strip().lstrip("-*").strip()
            cleaned = re.sub(r"^key unknowns:\s*", "", cleaned, flags=re.I)
            cleaned = re.sub(r"^next (best )?steps:\s*", "", cleaned, flags=re.I)
            cleaned = re.sub(r"^\d{1,2}[.)]\s*", "", cleaned)
            cleaned = cleaned.strip(" .;")
            if cleaned:
                items.append(cleaned)
    return items[:24]


ROUTE_LABELS = {
    "SIMPLE": "Simple",
    "ENGINE": "Engine",
    "REASONING": "Reasoning",
    "FACTUAL": "Factual: outside the knowledge boundary",
}


def build_triage(case: CaseFile) -> dict:
    route = case.route or {}
    depth = str(route.get("DEPTH_TRIGGER") or "NONE")
    label = route.get("ROUTE", "-")
    return {
        "route": label,
        "route_label": ROUTE_LABELS.get(str(label), str(label)),
        "reason": route.get("REASON", ""),
        "complexity": route.get("COMPLEXITY", "-"),
        "risk": route.get("RISK_LEVEL", "-"),
        "depth_trigger": depth,
        "depth_active": depth.upper() not in {"NONE", ""},
        "intent_scope": route.get("INTENT_SCOPE", "-"),
        "stakes": route.get("STAKES", "-"),
        "high_stakes": str(route.get("STAKES") or "") == "High Moral / Ontological Stakes",
        "termination_signal": route.get("TERMINATION_SIGNAL", "NONE"),
        "source": route.get("_source", "-"),
        "forced": bool(route.get("_forced")),
        "is_factual": str(label) == "FACTUAL",
        "refused": bool(route.get("_refused")),
        "escalated_from_factual": route.get("_escalated", ""),
    }


def assumption_status(case: CaseFile) -> list[dict]:
    axioms = list(case.checkpoint.get("assumptions_detected") or case.skeptic.get("HIDDEN_AXIOMS") or [])
    premise_text = " ".join(case.editor.get("PREMISE_WARNINGS") or [])
    meta = case.editor.get("_meta", {})
    tier = str(meta.get("tier") or "")
    unverified_tier = tier in {
        "Conceptually Open",
        "Highly Contested",
        "Theoretical Plausibility",
        "Architecturally Implausible (Current Frameworks)",
    }
    forced = case.mode in FORCING_MODES
    out = []
    for axiom in axioms:
        key = axiom.strip(".").strip().lower()
        status = "accepted"
        for chunk in case.editor.get("PREMISE_WARNINGS") or []:
            if key[:40] and key[:40] in chunk.lower():
                status = "rejected" if "rejected" in chunk.lower() else "unverified"
                break
        if status == "accepted" and (premise_text or unverified_tier):
            status = "unverified"
        if forced and status == "unverified":
            status = "user-forced"
        out.append({"text": axiom, "status": status, "tier": tier})
    return out


def build_evidence(case: CaseFile) -> dict:
    """The Phase 2 ontology packet: the landscape the answer is indexed against."""
    ontology = case.ontology or {}
    meta = case.editor.get("_meta", {})
    tiers = [tier_row(entry) for entry in (ontology.get("TIERS") or [])]
    return {
        "summary": str(ontology.get("SUMMARY", "")),
        "tier": str(meta.get("tier") or ""),
        "tier_label": tier_label(meta.get("tier") or ""),
        "tier_entry": str(meta.get("tier_entry") or ""),
        "tier_table": [{"claim": t["claim"], "tier": t["tier"], "justification": t["justification"]} for t in tiers],
        "anchor_modal_status": str(meta.get("anchor_modal_status") or ontology.get("ANCHOR_MODAL_STATUS") or ""),
        "scope": str(meta.get("scope") or ""),
        "stakes": str(meta.get("stakes") or ""),
        "logic_anchor": logic_anchor_kind(ontology),
        "logic_lines": logic_lines(ontology),
        "ontology_map": list(ontology.get("ONTOLOGY_MAP") or []),
        "map_used": int(ontology.get("ONTOLOGY_MAP_SIZE") or len(ontology.get("ONTOLOGY_MAP") or [])),
        "map_raw": int(ontology.get("ONTOLOGY_MAP_RAW") or 0),
        "degraded": str(ontology.get("_degraded") or ""),
        "landmarks": list(ontology.get("LANDMARKS") or []),
        "theories": list(ontology.get("THEORIES") or []),
        "frontiers": list(ontology.get("FRONTIERS") or []),
        "modal_claims": list(ontology.get("MODAL_CLAIMS") or []),
        "key_unknowns": list(ontology.get("KEY_UNKNOWNS") or []),
        "what_would_change": list(ontology.get("WHAT_WOULD_CHANGE") or []),
        "proposed_tests": list(ontology.get("PROPOSED_TESTS") or []),
        "live_options": list(ontology.get("LIVE_OPTIONS") or []),
        "weak_options": list(ontology.get("REJECTED_OR_WEAK_OPTIONS") or []),
        "claims_to_avoid": list(ontology.get("CLAIMS_TO_AVOID") or []),
        "theory_objections": list(ontology.get("THEORY_OBJECTIONS") or []),
    }


def tier_row(entry: str) -> dict:
    parts = [p.strip() for p in str(entry).split("|")]
    claim = parts[0] if parts else ""
    tier = parts[1] if len(parts) > 1 else ""
    justification = " | ".join(parts[2:]) if len(parts) > 2 else ""
    return {"claim": claim, "tier": tier, "justification": justification}


def build_dialectic(case: CaseFile) -> dict:
    """The Phase 3 dialectical engine: thesis, negating antithesis, and what survives."""
    d = case.dialectic or {}
    return {
        "thesis": str(d.get("THESIS", "")),
        "thesis_axioms": list(d.get("THESIS_AXIOMS") or []),
        "antithesis": str(d.get("ANTITHESIS", "")),
        "antithesis_negates": list(d.get("ANTITHESIS_NEGATES") or []),
        "synthesis": str(d.get("SYNTHESIS", "")),
        "tension_status": str(d.get("TENSION_STATUS", "")),
        "similarity": d.get("_similarity", 0.0),
        "forced_antithesis": bool(d.get("_forced")),
        "degraded": str(d.get("_degraded") or ""),
    }


def build_guardrails(case: CaseFile) -> dict:
    """The deterministic Phase 4 guardrail output: matrix, test flags, symmetry objections."""
    g = case.guardrails or {}
    flags = list(g.get("test_flags") or [])
    return {
        "matrix": list(g.get("matrix") or []),
        "matrix_table": list(case.editor.get("MATRIX_TABLE") or []),
        "test_flags": flags,
        "blocked_tests": [f for f in flags if f.get("status") == "BLOCK"],
        "flagged_tests": [f for f in flags if str(f.get("status", "")).startswith("FLAG")],
        "universal_objections": list(g.get("universal_objections") or []),
        "tier_gaps": list(g.get("tier_gaps") or []),
        "modal_gaps": list(g.get("modal_gaps") or []),
        "violations": list(g.get("violations") or []),
    }


def build_causal(case: CaseFile) -> dict:
    """Theory bookkeeping: which mapped theories the Editor selected to argue, which it archived."""
    meta = case.editor.get("_meta", {})
    tier = str(meta.get("tier") or "")
    selected = list(case.editor.get("SELECTED_THEORIES") or [])
    archived = list(case.editor.get("ARCHIVED_THEORIES") or [])
    claims_to_avoid = list((case.ontology or {}).get("CLAIMS_TO_AVOID") or [])
    models = []
    for theory in (case.ontology or {}).get("THEORIES") or []:
        name = str(theory).split("|")[0].strip()
        chosen = any(overlap(theory, s) >= 3 for s in selected)
        archived_hit = any(overlap(theory, a) >= 3 for a in archived)
        avoided = any(overlap(theory, claim) >= 3 for claim in claims_to_avoid)
        status = "rejected" if avoided else ("live" if chosen or not archived_hit else "archived")
        models.append(
            {
                "name": name,
                "status": status,
                "placement": "argued in the answer" if chosen else "archived in the trace",
                "note": "Indexed in the theory-indexed feasibility matrix." if chosen else "Not carried into the prose; kept visible in the trace.",
            }
        )
    for cause in case.skeptic.get("CAUSAL_ATTRIBUTIONS") or []:
        models.insert(
            0,
            {
                "name": str(cause),
                "status": "user-proposed (unverified)" if tier != "Established Consensus" else "user-proposed (tiered as consensus)",
                "placement": "under test",
                "note": f"Central claim tier: {tier}." if tier else "The central claim carries no tier label.",
            },
        )
    return {
        "user_causes": list(case.skeptic.get("CAUSAL_ATTRIBUTIONS") or []),
        "corrections": list(case.editor.get("CAUSAL_CORRECTIONS") or []),
        "selected": selected,
        "archived": archived,
        "models": models,
        "tier": tier,
    }


def overlap(a: str, b: str) -> int:
    from epistemic.rules import content_tokens

    return len(content_tokens(a) & content_tokens(b))


def build_modal(case: CaseFile) -> dict:
    modal_claims = [str(entry) for entry in ((case.ontology or {}).get("MODAL_CLAIMS") or [])]
    return {
        "user_options": list(case.skeptic.get("MODAL_BOUNDARIES") or []),
        "expanded": list(case.editor.get("MODAL_CORRECTIONS") or []),
        "excluded": list(case.skeptic.get("EXCLUDED_OPTIONS") or []),
        "live_options": list((case.ontology or {}).get("LIVE_OPTIONS") or []),
        "modal_claims": modal_claims,
        "modal_gaps": list((case.guardrails or {}).get("modal_gaps") or []),
        "allowed_task_shape": case.editor.get("ALLOWED_TASK_SHAPE", ""),
    }


def build_vocabulary(case: CaseFile) -> list[dict]:
    rows = []
    for sub in case.editor.get("VOCABULARY_SUBSTITUTIONS") or []:
        parts = re.split(r"->|→", sub)
        if len(parts) != 2:
            continue
        term = parts[0].strip().strip("'\"")
        replacement = parts[1].strip().strip("'\"")
        rows.append({"term": term, "issue": VOCABULARY_ISSUES.get(term.lower(), "Carries an unverified commitment."), "substitution": replacement})
    return rows


def build_termination(case: CaseFile) -> list[dict]:
    rows = []
    for text in case.termination_warnings:
        kind, severity, title = classify_warning(text)
        rows.append(
            {
                "signal": title,
                "type": kind,
                "reason": text.replace("Termination warning:", "").strip(),
                "severity": severity,
                "action": "Warning shown. Response continued in the labeled mode.",
                "override": "None",
            }
        )
    for signal in (case.route.get("TERMINATION_SIGNAL"), case.skeptic.get("TERMINATION_SIGNAL")):
        if signal and str(signal).upper() != "NONE":
            rows.append(
                {
                    "signal": str(signal).replace("_", " ").capitalize(),
                    "type": "signal",
                    "reason": "Detected during Triage and Skeptic analysis of the request.",
                    "severity": "soft",
                    "action": "Recorded. Analysis continued.",
                    "override": "None",
                }
            )
    for override in case.user_override_history:
        rows.append(
            {
                "signal": "User override",
                "type": "user-override",
                "reason": str(override.get("note") or override.get("checkpoint")),
                "severity": "medium",
                "action": f"Continued in {override.get('mode')} mode with an explicit label.",
                "override": str(override.get("checkpoint")),
            }
        )
    return rows


def logic_lines(ontology: dict) -> list[str]:
    lines = [str(x).strip() for x in (list(ontology.get("LOGICAL_SYLLOGISM") or []) + list(ontology.get("PSEUDO_CODE") or [])) if str(x).strip()]
    return lines


def logic_anchor_kind(ontology: dict) -> str:
    if ontology.get("_logic"):
        return str(ontology.get("_logic"))
    if ontology.get("LOGICAL_SYLLOGISM"):
        return "syllogism"
    if ontology.get("PSEUDO_CODE"):
        return "pseudo-code"
    return "missing"


def build_repairs(case: CaseFile) -> list[dict]:
    """Surgical section repairs: one section rewritten, the rest of the answer untouched."""
    titles = contract.SECTION_TITLES_BY_ID
    return [
        {
            "section": r.get("section", ""),
            "title": titles.get(str(r.get("section", "")).lower(), str(r.get("section", ""))),
            "checks": list(r.get("checks") or []),
            "reason": r.get("reason", ""),
            "before": r.get("before", ""),
            "after": r.get("after", ""),
            "round": r.get("round"),
        }
        for r in case.repairs
    ]


def build_overthinking(case: CaseFile) -> dict:
    over = dict(case.overthinking or {})
    over["report"] = bool(over)
    over["negative_prompt"] = bool((case.packets_meta or {}).get("Writer", {}).get("negative_prompt", False))
    return over


def build_critic(case: CaseFile) -> dict:
    """Phase 6: the presentation critic's checklist verdicts over the final text."""
    critic = case.critic or {}
    verdicts = list(case.checks or [])
    failed = [c for c in verdicts if not c.get("pass")]
    rewrites = max(0, len(case.critic_rounds) - 1) if case.critic_rounds else 0
    return {
        "result": critic.get("RESULT", "Not run"),
        "mode": "presentation contract over the final text: " + (f"{len(verdicts)} checklist items" if verdicts else "not run"),
        "checks": [
            {
                "id": c.get("id"),
                "name": c.get("name", ""),
                "pass": bool(c.get("pass")),
                "note": c.get("note", ""),
                "section": c.get("section", ""),
                "section_title": contract.SECTION_TITLES_BY_ID.get(str(c.get("section", "")).lower(), str(c.get("section", ""))),
            }
            for c in verdicts
        ],
        "checks_total": len(verdicts),
        "checks_failed": len(failed),
        "failed_sections": sorted({str(c.get("section") or "core") for c in failed}),
        "violations": list(critic.get("VIOLATIONS") or []),
        "rewrite_required": critic.get("REWRITE_REQUIRED", "-"),
        "rewrite_instruction": critic.get("REWRITE_INSTRUCTION", ""),
        "section_targets": [str(t) for t in (critic.get("SECTION_TARGETS") or [])],
        "deliberation_characters": len(case.deliberation),
        "rounds": len(case.critic_rounds or []),
        "rewrites": rewrites,
        "repairs": len(case.repairs),
        "banner": str(case.warning_banner or ""),
        "drift_note": "The Critic found remaining violations after the surgical repair. Review before relying on the answer."
        if critic.get("RESULT") == "FAIL" and not case.repairs
        else "",
    }


def build_deliberation(case: CaseFile) -> dict:
    text = case.deliberation or ""
    return {
        "text": text,
        "rewrite_text": case.deliberation_rewrite or "",
        "characters": len(text),
        "estimated_tokens": len(text) // 4,
        "lines": len([line for line in text.splitlines() if line.strip()]),
    }


def build_answer(case: CaseFile) -> dict:
    sections, banner = build_sections(case)
    meta = case.editor.get("_meta", {})
    annotations = [
        {"term": row["term"], "note": f"User term: “{row['term']}” → neutral substitution: “{row['substitution']}”"}
        for row in build_vocabulary(case)
    ]
    footer = {
        "tier": str(meta.get("tier") or "untiered"),
        "tier_label": tier_label(meta.get("tier")),
        "confidence": tier_label(meta.get("tier")),
        "anchor_modal_status": str(meta.get("anchor_modal_status") or ""),
        "scope": str(meta.get("scope") or ""),
        "stakes": str(meta.get("stakes") or ""),
        "mode": case.mode,
        "label": (
            f"Mode: {case.mode}. This response follows the user's requested framing despite unresolved evidence gaps."
            if case.mode in FORCING_MODES
            else f"Mode: {case.mode}. Continued with the best bounded answer."
        ),
    }
    if case.refusal:
        text = str(case.refusal.get("text") or "")
        sections = [
            {"id": "refusal", "title": "Epistemic refusal", "text": text},
            {
                "id": "boundary",
                "title": "Why the engine was not run",
                "text": str(case.refusal.get("reason") or "") + " " + str(case.refusal.get("boundary") or ""),
            },
            {"id": "next", "title": "How to get the answer", "items": [str(case.refusal.get("suggestion") or "")]},
        ]
        banner = {"text": "Routed FACTUAL: the request needs external verification, and this system does not guess facts.", "type": "knowledge-boundary"}
        footer = {
            "tier": "refused at the knowledge boundary",
            "tier_label": "Refused: outside the parametric boundary",
            "confidence": "Refused",
            "anchor_modal_status": "",
            "scope": str(case.route.get("INTENT_SCOPE") or ""),
            "stakes": str(case.route.get("STAKES") or ""),
            "mode": "knowledge-boundary",
            "label": "Route: FACTUAL. Refused at the knowledge boundary; the Engine was not run and nothing was guessed.",
        }
        annotations = []
    return {"banner": banner, "sections": sections, "footer": footer, "annotations": annotations}


def build_overview(case: CaseFile) -> dict:
    over = case.overthinking or {}
    ontology = case.ontology or {}
    guardrails = case.guardrails or {}
    dialectic = case.dialectic or {}
    meta = case.editor.get("_meta", {})
    return {
        "route": case.route.get("ROUTE", "-"),
        "route_label": ROUTE_LABELS.get(str(case.route.get("ROUTE")), str(case.route.get("ROUTE", "-"))),
        "complexity": case.route.get("COMPLEXITY", "-"),
        "risk": case.route.get("RISK_LEVEL", "-"),
        "depth_trigger": case.route.get("DEPTH_TRIGGER", "NONE"),
        "intent_scope": case.route.get("INTENT_SCOPE", "-"),
        "stakes": case.route.get("STAKES", "-"),
        "continuation_count": case.continuation_count,
        "mode": case.mode,
        "checkpoint": describe_checkpoint(case),
        "editor_source": case.editor.get("_source", "-"),
        "tier": str(meta.get("tier") or "untiered"),
        "logic_anchor": logic_anchor_kind(ontology),
        "overthinking": (
            f"aborted: {over.get('reason', '')}; truncated at the last stable paragraph ({over.get('dropped_paragraphs', 0)} dropped)"
            if over.get("aborted")
            else f"output cut at the token budget: {over.get('reason', '')}; the answer is a partial draft assembled from the ontology packet"
            if over.get("truncated")
            else f"clear: {over.get('hesitation_markers', 0)} hesitation markers, {over.get('stance_flips', 0)} stance flips"
            if over
            else "not run"
        ),
        "negative_prompt": "on: the raw biased prompt is sent as the API negative_prompt for the Writer"
        if (case.packets_meta or {}).get("Writer", {}).get("negative_prompt")
        else "off",
        "repairs": f"{len(case.repairs)} section(s) rewritten in place from Critic checklist verdicts" if case.repairs else "none",
        "refusal": str(case.refusal.get("text")) if case.refusal else "",
        "ontology_packet": "degraded fallback (the ontology passes returned no readable packet)"
        if ontology.get("_degraded")
        else f"ontology map of {len(ontology.get('ONTOLOGY_MAP') or [])} entries distilled into "
        f"{len(ontology.get('THEORIES') or [])} theories, {len(ontology.get('LANDMARKS') or [])} landmarks, "
        f"{len(ontology.get('FRONTIERS') or [])} frontiers",
        "dialectic": f"tension {dialectic.get('TENSION_STATUS', '-')}: {len(dialectic.get('THESIS_AXIOMS') or [])} thesis axioms, "
        f"{len(dialectic.get('ANTITHESIS_NEGATES') or [])} negated"
        + (" (forced antithesis)" if dialectic.get("_forced") else ""),
        "guardrails": f"{len(guardrails.get('matrix') or [])} matrix rows, "
        f"{sum(1 for f in guardrails.get('test_flags') or [] if f.get('status') == 'BLOCK')} blocked tests, "
        f"{len(guardrails.get('universal_objections') or [])} universal objections, "
        f"{len(guardrails.get('violations') or [])} violations carried into the directive",
        "raw_prompt_exposed_to_writer": bool(case.leak.get("raw_prompt_in_context", False)),
        "shared_word_run": case.leak.get("longest_shared_word_run", 0),
        "final_status": f"Continued in {case.mode} mode",
        "continuation_analysis": case.continuation,
    }


def describe_checkpoint(case: CaseFile) -> str:
    record = case.checkpoint or {}
    action = record.get("action", "pending")
    if action == "confirm":
        return "User confirmed"
    if action == "edit":
        return "User edited assumptions"
    if action in {"force-original", "override"}:
        return "User forced original framing"
    if action == "auto-continue":
        return "Auto-approved"
    if action == "skipped":
        return "Skipped (simple route)"
    return action


def build_ui(case: CaseFile) -> dict:
    answer = build_answer(case)
    payload = {
        "meta": {
            "id": case.case_id,
            "title": case.title or title_of(case.raw_prompt),
            "created_at": case.created_at,
            "mode": case.mode,
            "continuation_count": case.continuation_count,
            "settings": case.settings,
            "status": case_status_from(case),
            "error": case.error,
        },
        "prompt": case.raw_prompt,
        "steps": case.steps,
        "triage": build_triage(case),
        "refusal": case.refusal,
        "checkpoint": {
            "payload": {
                "neutral_question": case.skeptic.get("NEUTRAL_QUESTION", ""),
                "assumptions": list(case.skeptic.get("HIDDEN_AXIOMS") or []),
                "epistemic_stance": case.skeptic.get("EPISTEMIC_STANCE", ""),
                "causal_attributions": list(case.skeptic.get("CAUSAL_ATTRIBUTIONS") or []),
                "modal_boundaries": list(case.skeptic.get("MODAL_BOUNDARIES") or []),
                "excluded_options": list(case.skeptic.get("EXCLUDED_OPTIONS") or []),
                "loaded_terms": list(case.skeptic.get("LOADED_TERMS") or []),
                "requested_register": case.skeptic.get("REQUESTED_REGISTER", ""),
                "requested_task_shape": case.skeptic.get("REQUESTED_TASK_SHAPE", ""),
                "fallacy_flags": list(case.skeptic.get("FALLACY_FLAGS") or []),
                "termination_signal": case.skeptic.get("TERMINATION_SIGNAL", ""),
                "depth_trigger": case.route.get("DEPTH_TRIGGER", "NONE"),
            },
            "record": case.checkpoint,
            "open": any(s.get("status") == "waiting" for s in case.steps),
        },
        "warnings": build_warnings(case),
        "answer": answer,
        "deliberation": build_deliberation(case),
        "evidence": build_evidence(case),
        "trace": {
            "overview": build_overview(case),
            "assumptions": assumption_status(case),
            "evidence": build_evidence(case),
            "dialectic": build_dialectic(case),
            "guardrails": build_guardrails(case),
            "causal": build_causal(case),
            "modal": build_modal(case),
            "vocabulary": build_vocabulary(case),
            "termination": build_termination(case),
            "critic": build_critic(case),
            "repairs": build_repairs(case),
            "overthinking": build_overthinking(case),
            "deliberation": build_deliberation(case),
            "register_override": case.editor.get("REGISTER_OVERRIDE", ""),
            "editor": {k: v for k, v in case.editor.items() if not str(k).startswith("_")},
        },
        "raw_packets": case.packets_raw,
        "user_overrides": case.user_override_history,
        "final_response": case.final_response,
        "termination_warnings": case.termination_warnings,
        "packet_meta": case.packets_meta,
        "llm_calls": case.llm_calls,
        "continuation": {
            "count": case.continuation_count,
            "last_analysis": case.continuation,
            "modes": sorted(FORCING_MODES | {"evidence-bounded"}),
            "placeholder": "Add new evidence, constraints, or ask for a deeper analysis.",
            "pushback_placeholder": "Add new evidence or specify a new constraint.",
            "pushback": (case.continuation or {}).get("PUSHBACK_ONLY") == "YES",
            "suggestions": [
                {"id": "add-evidence", "label": "Add evidence"},
                {"id": "challenge-premise", "label": "Challenge a premise"},
                {"id": "expand-options", "label": "Expand solution space"},
                {"id": "compare-theories", "label": "Compare theories"},
                {"id": "continue-anyway", "label": "Continue anyway", "mode": "hypothetical"},
                {"id": "more-concise", "label": "Make it more concise"},
                {"id": "show-deliberation", "label": "Show the deliberation"},
                {"id": "show-trace", "label": "Show trace"},
            ],
        },
    }
    return payload


def case_status_from(case: CaseFile) -> str:
    from web.store import case_status

    return case_status(case.to_dict())
