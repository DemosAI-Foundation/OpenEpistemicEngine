"""User display: final response plus the epistemic trace panel."""

from __future__ import annotations

from .casefile import CaseFile
from .checks import SECTION_TITLES_BY_ID
from .schemas import FieldSpec

WIDTH = 78


def rule(char: str = "-") -> str:
    return char * WIDTH


def checks_title(section: str) -> str:
    key = str(section or "").strip().lower()
    return SECTION_TITLES_BY_ID.get(key, key or "core")


def _bullets(items: list[str], indent: str = "  ") -> str:
    if not items:
        return f"{indent}NONE"
    return "\n".join(f"{indent}- {i}" for i in items)


def logic_panel(ontology: dict) -> str:
    lines = [str(x).strip() for x in (list(ontology.get("LOGICAL_SYLLOGISM") or []) + list(ontology.get("PSEUDO_CODE") or [])) if str(x).strip()]
    if not lines:
        return "LOGIC CHAIN: NONE STATED (the packet carries no syllogism and no pseudo-code)"
    return "LOGIC CHAIN THE ANSWER MUST MAP ONTO:\n" + _bullets(lines)


def checkpoint_view(skeptic: dict) -> str:
    axioms = list(skeptic.get("HIDDEN_AXIOMS") or [])
    lines = [rule("="), "HUMAN CHECKPOINT", rule()]
    if axioms:
        lines.append("The system detected the following assumptions:")
        lines.append("\n".join(f"  {i + 1}. {a}" for i, a in enumerate(axioms)))
    else:
        lines.append("The system detected no hidden assumptions to confirm.")
    lines.append("")
    lines.append("The system will analyze the neutral question:")
    lines.append(f'  "{skeptic.get("NEUTRAL_QUESTION", "")}"')
    if skeptic.get("CAUSAL_ATTRIBUTIONS"):
        lines.append("")
        lines.append("Causal attribution under test: " + "; ".join(skeptic["CAUSAL_ATTRIBUTIONS"]))
    if skeptic.get("TERMINATION_SIGNAL"):
        lines.append("Termination signal: " + skeptic["TERMINATION_SIGNAL"])
    lines.append("")
    lines.append("Choices: [1] confirm neutral analysis  [2] edit assumptions  [3] force original framing (with warning)  [4] continue automatically")
    lines.append(rule())
    return "\n".join(lines)


def trace_panel(case: CaseFile) -> str:
    editor = case.editor
    meta = editor.get("_meta", {})
    lines = [rule("="), "EPISTEMIC TRACE", rule()]
    lines.append(f"ROUTE: {case.route.get('ROUTE', 'ENGINE')} ({case.route.get('_source', 'llm')})")
    lines.append(f"COMPLEXITY: {case.route.get('COMPLEXITY', '-')}  RISK: {case.route.get('RISK_LEVEL', '-')}")
    lines.append(f"TRIAGE TERMINATION SIGNAL: {case.route.get('TERMINATION_SIGNAL', 'NONE')}")
    if case.refusal:
        lines.append("KNOWLEDGE BOUNDARY REFUSAL: " + str(case.refusal.get("text", "")))
        lines.append("  boundary: " + str(case.refusal.get("boundary", "")))
        lines.append("  remedy: " + str(case.refusal.get("suggestion", "")))
        lines.append(rule())
        return "\n".join(lines)
    lines.append(f"NEUTRAL QUESTION: {case.skeptic.get('NEUTRAL_QUESTION', '-')}")
    lines.append(f"ALLOWED TASK SHAPE: {editor.get('ALLOWED_TASK_SHAPE', '-')}")
    lines.append(f"CONTINUATION COUNT: {case.continuation_count}  MODE: {case.mode}")
    lines.append(f"CHECKPOINT: {case.checkpoint.get('action', '-')}  auto_approved={case.checkpoint.get('auto_approved', False)}")
    lines.append(f"EDITOR SOURCE: {editor.get('_source', '-')}")
    lines.append(f"CENTER CLAIM TIER: {meta.get('tier', 'untiered')}  ANCHOR MODAL STATUS: {meta.get('anchor_modal_status', 'unlabelled')}")
    lines.append(f"LOGIC ANCHOR: {meta.get('logic_anchor', '-')}")
    lines.append(logic_panel(case.ontology))
    lines.append(f"RAW PROMPT EXPOSED TO WRITER: {case.leak.get('raw_prompt_in_context', False)} (shared run {case.leak.get('longest_shared_word_run', 0)})")
    lines.append(f"WRITER DELIBERATION: {len(case.deliberation)} characters (kept out of the answer)")
    over = case.overthinking or {}
    if over:
        state = "ABORTED" if over.get("aborted") else ("CUT AT TOKEN BUDGET" if over.get("truncated") else "clear")
        lines.append(
            f"OVERThinking MONITOR: {state}  "
            f"hesitation markers {over.get('hesitation_markers', 0)} (limit {over.get('limit', '-')})  "
            f"stance flips {over.get('stance_flips', 0)} (limit {over.get('flip_limit', '-')})"
        )
        if over.get("aborted"):
            lines.append("  " + str(over.get("reason", "")) + f"  truncated at the last stable paragraph ({over.get('dropped_paragraphs', 0)} dropped)")
        if over.get("truncated"):
            lines.append("  " + str(over.get("reason", "")) + "  the missing sections were assembled from the ontology packet")
    lines.append(
        f"WRITER NEGATIVE PROMPT: {bool(case.packets_meta.get('Writer', {}).get('negative_prompt', False))} "
        "(the user's biased wording is penalized at the API, never pasted into the prompt)"
    )
    lines.append("SELECTED THEORIES (argued in the answer):")
    lines.append(_bullets(list(editor.get("SELECTED_THEORIES") or [])))
    lines.append("ARCHIVED THEORIES (kept in the trace):")
    lines.append(_bullets(list(editor.get("ARCHIVED_THEORIES") or [])))
    lines.append("")
    lines.append("REGISTER OVERRIDE: " + str(editor.get("REGISTER_OVERRIDE", "NONE")))
    lines.append("PREMISE REJECTIONS:")
    lines.append(_bullets(list(editor.get("PREMISE_WARNINGS") or [])))
    lines.append("CAUSAL CORRECTIONS:")
    lines.append(_bullets(list(editor.get("CAUSAL_CORRECTIONS") or [])))
    lines.append("MODAL EXPANSION:")
    lines.append(_bullets(list(editor.get("MODAL_CORRECTIONS") or [])))
    lines.append("EPISTEMIC WARNINGS:")
    lines.append(_bullets(list(editor.get("EPISTEMIC_WARNINGS") or [])))
    lines.append("VOCABULARY SUBSTITUTIONS:")
    lines.append(_bullets(list(editor.get("VOCABULARY_SUBSTITUTIONS") or [])))
    lines.append("TERMINATION WARNINGS:")
    lines.append(_bullets(list(case.termination_warnings)))
    lines.append("DIALECTIC TENSION: " + str(case.dialectic.get("TENSION_STATUS", "-")) + "  antithesis negates: " + "; ".join(str(a) for a in (case.dialectic.get("ANTITHESIS_NEGATES") or []))[:160])
    lines.append("LIVE OPTIONS FROM THE ONTOLOGY:")
    lines.append(_bullets(list(case.ontology.get("LIVE_OPTIONS") or []), indent="  "))
    lines.append("USER OVERRIDES:")
    lines.append(_bullets([str(o.get("note") or o.get("checkpoint")) for o in case.user_override_history]))
    if case.critic:
        failed = [c for c in case.checks if not c.get("pass")]
        lines.append(
            f"CRITIC (presentation contract over {len(case.checks)} checks, final text read): "
            f"{case.critic.get('RESULT', '-')} over {len(case.critic_rounds)} round(s) "
            f"rewrite_required={case.critic.get('REWRITE_REQUIRED', '-')}"
        )
        if failed:
            lines.append("FAILED CHECKS:")
            lines.append(
                _bullets(
                    [
                        f"check {c.get('id')} [{c.get('section', 'core')}]: {c.get('note') or c.get('name')}"
                        for c in failed
                    ]
                )
            )
        lines.append(_bullets(list(case.critic.get("VIOLATIONS") or [])))
    if case.repairs:
        lines.append(f"SURGICAL SECTION REPAIR: {len(case.repairs)} section(s) rewritten, the rest untouched")
        lines.append(
            _bullets(
                [
                    f"{checks_title(r.get('section', 'core'))} (round {r.get('round', 1)}): {r.get('reason')}"
                    for r in case.repairs
                ]
            )
        )
    scrub = case.scrub or {}
    lines.append(
        f"SCRUBBER: {len(scrub.get('removed_meta_instructions') or [])} meta instructions removed, "
        f"{len(scrub.get('dropped_duplicate_paragraphs') or [])} duplicate paragraphs dropped, "
        f"{scrub.get('format_fixes', 0)} format fixes"
    )
    lines.append(f"FINAL STATUS: continued in {case.mode} mode")
    lines.append(rule())
    return "\n".join(lines)


def render(case: CaseFile) -> str:
    out = [rule("="), "RESPONSE", rule(), case.final_response, "", trace_panel(case)]
    return "\n".join(out)


def plain(case: CaseFile) -> str:
    return case.final_response


def format_packet(packet: dict, spec: list[FieldSpec]) -> str:
    lines = []
    for f in spec:
        val = packet.get(f.name)
        if f.kind == "list":
            lines.append(f"{f.name}:")
            lines.append(_bullets([str(v) for v in (val or [])], indent="   ") if val else "   NONE")
        else:
            lines.append(f"{f.name}: {val or 'NONE'}")
    return "\n".join(lines)
