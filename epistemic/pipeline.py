"""Pipeline orchestrator for the purely parametric divergent rigor architecture.

Convergent route: Triage -> DirectAnswer (SIMPLE) or knowledge-boundary refusal (FACTUAL).
Divergent route:  Triage (Phase 0) -> Skeptic (Phase 1) -> Checkpoint -> Ontology Mapper (Phase 2)
                  -> Dialectical Engine (Phase 3) -> Deterministic Editor (Phase 4, Python guardrails)
                  -> Writer + Mechanical Scrubber (Phase 5) -> Presentation Critic (Phase 6).

No retrieval, no live search, no external data layer: everything is generated from frozen parametric weights,
governed by deterministic logic rules, and bounded by the model's knowledge cutoff.
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field

from . import checks, guardrails, prompts, rules, scrubber, schemas
from .casefile import CaseFile, new_case_id
from .llm import LLMClient, LLMUnavailable, UnparseableOutput, parse_field_text
from .monitor import (
    DEFAULT_THRESHOLD,
    DEFAULT_WINDOW_TOKENS,
    OverthinkingMonitor,
    join_paragraphs,
    paragraphs,
    split_banner,
    truncate_at_last_stable,
)

ROLE_TOKENS = {
    "Triage": 6000,
    "DirectAnswer": 4000,
    "Skeptic": 12000,
    "OntologyMap": 16384,
    "Ontology": 12000,
    "Dialectic": 12000,
    "Writer": 16384,
    "WriterRepair": 8192,
    "Critic": 8192,
}

ROUTE_SYNONYMS = {
    "ENGINE": "ENGINE",
    "REASONING": "REASONING",
    "DEEP": "ENGINE",
    "FULL": "ENGINE",
    "ANALYSIS": "ENGINE",
    "DIVERGENT": "ENGINE",
    "ONTOLOGICAL": "ENGINE",
    "SIMPLE": "SIMPLE",
    "DIRECT": "SIMPLE",
    "FAST": "SIMPLE",
    "CONVERGENT": "SIMPLE",
    "FACTUAL": "FACTUAL",
    "REFUSAL": "FACTUAL",
    "KNOWLEDGE-BOUNDARY": "FACTUAL",
}

STEP_HEAD = re.compile(r"^\s*(?:[-*#>\s]*)?(?:step|fase)\s*(?P<n>\d{1,2})\s*[.)\]:\-]*\s*(?P<text>.+)$", re.I)
NUMBERED_HEAD = re.compile(r"^\s*(?:[-*#>\s]*)?(?P<n>\d{1,2})[.)]\s+(?P<text>.+)$")
STEP_LIMIT = 12

DIALECTIC_SIMILARITY_CAP = 0.55
MAX_SECTIONS_PER_REPAIR = 3

TRUNCATION_WARNING = (
    "Termination warning: the Writer stream was cut at the output token budget. The answer below is the text that was "
    "produced, followed by the remaining sections assembled directly from the ontology packet and the Editor directive. "
    "It is an incomplete draft: read it as a partial analysis, not as a finished answer."
)


class Cancelled(Exception):
    pass


class Hooks:
    def on_step(self, name: str, status: str, data: dict | None = None) -> None:
        return None

    def on_delta(self, role: str, text: str) -> None:
        return None

    def gate(self, name: str, payload: dict) -> dict:
        return {}

    def cancelled(self) -> bool:
        return False


@dataclass
class Checkpoint:
    action: str = "auto-continue"
    mode: str = ""
    edited_assumptions: list[str] = field(default_factory=list)
    note: str = ""
    auto_approved: bool = True


MAP_LIMIT = 40


def dedupe_models(lines: list[str]) -> list[str]:
    """Keep the first MAP_LIMIT distinct entries of the map; a 200-item map is noise, not ontology."""
    kept: list[str] = []
    kept_words: list[set[str]] = []
    for line in lines:
        words = set(rules.normalize(line).split())
        if not words:
            continue
        if any(len(words & seen) >= 5 for seen in kept_words):
            continue
        kept.append(line)
        kept_words.append(words)
        if len(kept) >= MAP_LIMIT:
            break
    return kept


FORCING_MODES = {"user-forced", "hypothetical", "speculative", "preference-based"}

FINAL_SECTION = re.compile(r"^\s*(?:[-*#>\s]*)?\[?(FINAL[ _-]?RESPONSE|FINAL[ _-]?ANSWER|RESPONSE)\]?\s*[:!\-]{0,2}\s*(.*)$")


class DeliberationStream:
    """Splits streamed Writer text into the private deliberation and the final answer."""

    def __init__(self, on_thinking=None):
        self.on_thinking = on_thinking
        self._thinking: list[str] = []
        self._final: list[str] = []
        self._buffer = ""
        self._in_final = False

    def reset(self) -> None:
        self._thinking.clear()
        self._final.clear()
        self._buffer = ""
        self._in_final = False

    def feed(self, delta: str) -> None:
        if self._in_final:
            self._final.append(delta)
            return
        self._buffer += delta
        while True:
            head, sep, rest = self._buffer.partition("\n")
            if not sep:
                break
            self._buffer = rest
            self._take_line(head)
        if self._buffer and self._in_final:
            self._final.append(self._buffer)
            self._buffer = ""

    def finish(self) -> None:
        if self._buffer:
            if self._in_final:
                self._final.append(self._buffer)
            else:
                self._take_line(self._buffer)
            self._buffer = ""

    def _take_line(self, line: str) -> None:
        hit = FINAL_SECTION.match(line)
        if hit and not self._in_final and len(line.strip()) < 90:
            self._in_final = True
            rest = hit.group(2).strip()
            if rest:
                self._final.append(rest + "\n")
            return
        self._thinking.append(line + "\n")
        if self.on_thinking:
            self.on_thinking(line + "\n")

    @property
    def thinking_text(self) -> str:
        return "".join(self._thinking).strip()

    @property
    def final_text(self) -> str:
        return "".join(self._final).strip()

    @property
    def in_final(self) -> bool:
        return self._in_final


class WriterStream:
    """One stream feeds the deliberation splitter and the overthinking watcher; both reset on a retry."""

    def __init__(self, stream: DeliberationStream, monitor: OverthinkingMonitor):
        self.stream = stream
        self.monitor = monitor

    def feed(self, text: str) -> None:
        self.stream.feed(text)
        self.monitor.feed(text)

    def reset(self) -> None:
        self.stream.reset()
        self.monitor.reset()


def split_prose(text: str) -> list[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n\s*\n", text or "") if s.strip()]
    chunks: list[str] = []
    for sentence in sentences:
        if chunks and len(chunks[-1]) + len(sentence) < 220:
            chunks[-1] = f"{chunks[-1]} {sentence}"
        else:
            chunks.append(sentence)
        if len(chunks) >= STEP_LIMIT:
            break
    return chunks


def parse_steps(text: str) -> list[dict]:
    """Split the Writer's deliberation into numbered moves: [{'step': 1, 'text': '...'}]."""
    steps: list[dict] = []
    pending: list[str] = []

    def flush() -> None:
        if pending:
            steps.append({"step": len(steps) + 1, "text": " ".join(pending).strip()})
            pending.clear()

    for line in (text or "").splitlines():
        if not line.strip():
            continue
        hit = STEP_HEAD.match(line) or NUMBERED_HEAD.match(line)
        if hit:
            flush()
            number, body = int(hit.group("n")), hit.group("text").strip()
            if steps and steps[-1]["step"] == number:
                steps[-1]["text"] += " " + body
            else:
                steps.append({"step": number, "text": body})
            continue
        if steps:
            steps[-1]["text"] += " " + line.strip()
        else:
            pending.append(line.strip())
    flush()
    if len(steps) == 1 and len(steps[0]["text"]) > 320:
        derived = [{"step": i + 1, "text": chunk, "derived": True} for i, chunk in enumerate(split_prose(steps[0]["text"]))]
        return derived[:STEP_LIMIT]
    return steps[:STEP_LIMIT]


def logic_anchor(ontology: dict) -> dict:
    syllogism = [str(x).strip() for x in (ontology.get("LOGICAL_SYLLOGISM") or []) if str(x).strip()]
    pseudo = [str(x).strip() for x in (ontology.get("PSEUDO_CODE") or []) if str(x).strip()]
    kind = "syllogism" if syllogism else ("pseudo-code" if pseudo else "missing")
    status = str(ontology.get("ANCHOR_MODAL_STATUS", "")).strip()
    return {"kind": kind, "modal_status": status, "syllogism": syllogism, "pseudo_code": pseudo, "lines": syllogism + pseudo}


def logic_text(ontology: dict) -> str:
    anchor = logic_anchor(ontology)
    if anchor["kind"] == "missing":
        return "NONE: the ontology packet carries no formal logic anchor."
    head = f"ANCHOR KIND: {anchor['kind']}   MODAL STATUS: {anchor['modal_status'] or 'UNLABELLED (must be labelled)'}"
    return head + "\n" + "\n".join(f"- {line}" for line in anchor["lines"])


def normalize_route(value: str) -> str:
    key = str(value or "").strip().upper().replace(" ", "")
    return ROUTE_SYNONYMS.get(key, key or "ENGINE")


def semantic_similarity(a: str, b: str) -> float:
    """Token Jaccard between thesis and antithesis: the modal-collapse meter for the dialectic."""
    ta, tb = rules.content_tokens(a), rules.content_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class Engine:
    def __init__(
        self,
        client: LLMClient,
        use_critic: bool = True,
        allow_rewrite: bool = True,
        use_logic_anchor: bool = True,
        use_negative_prompt: bool = True,
        use_overthinking: bool = True,
        use_section_repair: bool = True,
        overthinking_threshold: int = DEFAULT_THRESHOLD,
        overthinking_window: int = DEFAULT_WINDOW_TOKENS,
        cutoff: str = "",
    ):
        self.client = client
        self.use_critic = use_critic
        self.allow_rewrite = allow_rewrite
        self.use_logic_anchor = use_logic_anchor
        self.use_negative_prompt = use_negative_prompt
        self.use_overthinking = use_overthinking
        self.use_section_repair = use_section_repair
        self.overthinking_threshold = overthinking_threshold
        self.overthinking_window = overthinking_window
        self.cutoff = cutoff

    def run(
        self,
        raw_prompt: str,
        checkpoint: Checkpoint | None = None,
        prior: CaseFile | None = None,
        case_id: str = "",
        hooks: Hooks | None = None,
        force_route: str = "",
        settings: dict | None = None,
    ) -> CaseFile:
        case = self.probe(raw_prompt, prior=prior, case_id=case_id, hooks=hooks, force_route=force_route, settings=settings)
        if case.route.get("ROUTE") in schemas.TERMINAL_ROUTES:
            self._step(case, hooks, "final", "complete")
            return case
        return self.finish(case, checkpoint or Checkpoint(), hooks=hooks)

    def probe(
        self,
        raw_prompt: str,
        prior: CaseFile | None = None,
        case_id: str = "",
        hooks: Hooks | None = None,
        force_route: str = "",
        settings: dict | None = None,
        case: CaseFile | None = None,
    ) -> CaseFile:
        case = case or CaseFile(case_id=case_id or new_case_id(raw_prompt), raw_prompt=raw_prompt, settings=dict(settings or {}))
        case.case_id = case.case_id or case_id or new_case_id(raw_prompt)
        case.raw_prompt = raw_prompt
        case.cutoff = self.cutoff or rules.knowledge_cutoff(case.settings)
        if prior:
            case.continuation_count = prior.continuation_count + 1
            case.continuation_log = list(prior.continuation_log)
            case.user_override_history = list(prior.user_override_history)
        self._check(hooks)
        self._step(case, hooks, "triage", "running")
        cont = self._continuation(raw_prompt, prior, Checkpoint())
        case.continuation = cont
        case.route = self._triage(case, raw_prompt, prior, cont, hooks, force_route)
        self._step(
            case,
            hooks,
            "triage",
            "complete",
            {
                "route": case.route.get("ROUTE"),
                "reason": case.route.get("REASON"),
                "depth_trigger": case.route.get("DEPTH_TRIGGER", "NONE"),
                "intent_scope": case.route.get("INTENT_SCOPE"),
                "stakes": case.route.get("STAKES"),
                "source": case.route.get("_source"),
            },
        )
        if case.route.get("ROUTE") == "SIMPLE":
            self._direct_answer(case, raw_prompt, hooks)
            self._step(case, hooks, "final", "complete", {"route": "SIMPLE"})
            case.llm_calls = [asdict(r) for r in self.client.log]
            return case
        if case.route.get("ROUTE") == "FACTUAL":
            self._factual(case, raw_prompt, hooks)
            self._step(case, hooks, "final", "complete", {"route": "FACTUAL", "refused": bool(case.refusal)})
            case.llm_calls = [asdict(r) for r in self.client.log]
            return case
        self._check(hooks)
        self._step(case, hooks, "skeptic", "running")
        self._skeptic(case, raw_prompt, prior, hooks)
        self._step(
            case,
            hooks,
            "skeptic",
            "complete",
            {
                "neutral_question": case.skeptic.get("NEUTRAL_QUESTION"),
                "sub_questions": len(case.skeptic.get("SUB_QUESTIONS") or []),
                "assumptions": len(case.skeptic.get("HIDDEN_AXIOMS") or []),
                "loaded_terms": len(case.skeptic.get("LOADED_TERMS") or []),
                "vocabulary_contract": len(case.skeptic.get("VOCABULARY_CONTRACT") or []),
            },
        )
        case.llm_calls = [asdict(r) for r in self.client.log]
        return case

    def finish(self, case: CaseFile, checkpoint: Checkpoint, hooks: Hooks | None = None) -> CaseFile:
        if case.raw_prompt not in [c.get("prompt") for c in case.continuation_log]:
            case.continuation_log.append(
                {"prompt": case.raw_prompt, "checkpoint": checkpoint.action, "mode": checkpoint.mode}
            )
        self._check(hooks)
        self._step(case, hooks, "checkpoint", "running")
        resolved = self._resolve_checkpoint(case, checkpoint, hooks)
        checkpoint = resolved
        cont = case.continuation
        if checkpoint.action == "force-original" and cont and cont.get("MODE") not in FORCING_MODES:
            cont["MODE"] = "user-forced"
            cont["REASON"] = str(cont.get("REASON", "")) + "; user forced the original framing"
        if checkpoint.mode in FORCING_MODES and cont:
            cont["MODE"] = checkpoint.mode
        case.continuation = cont
        if checkpoint.mode:
            mode = checkpoint.mode
        elif cont.get("MODE") in FORCING_MODES:
            mode = cont["MODE"]
        else:
            mode = "evidence-bounded"
        case.checkpoint = self._checkpoint_record(case, checkpoint, mode)
        case.mode = mode
        self._step(
            case,
            hooks,
            "checkpoint",
            "overridden" if checkpoint.action in {"force-original", "override"} else "complete",
            {"action": checkpoint.action, "mode": mode, "auto_approved": case.checkpoint.get("auto_approved")},
        )
        self._check(hooks)
        self._step(case, hooks, "ontology", "running")
        self._ontology(case, mode, hooks)
        self._step(
            case,
            hooks,
            "ontology",
            "complete",
            {
                "theories": len(case.ontology.get("THEORIES") or []),
                "landmarks": len(case.ontology.get("LANDMARKS") or []),
                "frontiers": len(case.ontology.get("FRONTIERS") or []),
                "tiers": len(case.ontology.get("TIERS") or []),
                "map_entries": len(case.ontology.get("ONTOLOGY_MAP") or []),
                "map_raw": case.ontology.get("ONTOLOGY_MAP_RAW", 0),
                "degraded": str(case.ontology.get("_degraded") or ""),
                "logic": str(case.ontology.get("_logic") or ""),
            },
        )
        self._check(hooks)
        self._step(case, hooks, "dialectic", "running")
        self._dialectic(case, hooks)
        self._step(
            case,
            hooks,
            "dialectic",
            "complete",
            {
                "axioms": len(case.dialectic.get("THESIS_AXIOMS") or []),
                "negates": len(case.dialectic.get("ANTITHESIS_NEGATES") or []),
                "similarity": case.dialectic.get("_similarity", 0.0),
                "forced_antithesis": bool(case.dialectic.get("_forced")),
                "tension": case.dialectic.get("TENSION_STATUS", ""),
            },
        )
        self._check(hooks)
        self._step(case, hooks, "editor", "running")
        self._editor(case, cont, mode, hooks)
        self._step(
            case,
            hooks,
            "editor",
            "overridden" if case.guardrails.get("violations") else "complete",
            {
                "source": case.editor.get("_source"),
                "matrix_rows": len(case.guardrails.get("matrix") or []),
                "blocked_tests": sum(1 for f in case.guardrails.get("test_flags") or [] if f["status"] == guardrails.TEST_BLOCK),
                "flagged_tests": sum(1 for f in case.guardrails.get("test_flags") or [] if f["status"] == guardrails.TEST_FLAG),
                "universal_objections": len(case.guardrails.get("universal_objections") or []),
                "warnings": len(case.termination_warnings),
            },
        )
        self._check(hooks)
        self._step(case, hooks, "writer", "running")
        if self.use_critic:
            self._critic(case, hooks=hooks)
        else:
            case.final_response = self._writer(case, hooks=hooks)
        self._step(
            case,
            hooks,
            "writer",
            "complete",
            {
                "characters": len(case.final_response),
                "deliberation_characters": len(case.deliberation),
                "scrubbed_meta_instructions": len(case.scrub.get("removed_meta_instructions") or []),
                "scrubbed_duplicates": len(case.scrub.get("dropped_duplicate_paragraphs") or []),
                "format_fixes": case.scrub.get("format_fixes", 0),
                "overthinking_aborted": bool(case.overthinking.get("aborted")),
                "repairs": len(case.repairs),
                "steps": len(parse_steps(case.deliberation)),
            },
        )
        if self.use_critic:
            failed = [c for c in case.checks if not c.get("pass")]
            self._step(
                case,
                hooks,
                "critic",
                "complete" if not failed and not case.repairs else "overridden",
                {
                    "result": case.critic.get("RESULT"),
                    "rounds": len(case.critic_rounds),
                    "checks_total": len(case.checks),
                    "checks_failed": len(failed),
                    "repairs": len(case.repairs),
                },
            )
        self._step(case, hooks, "final", "complete", {"mode": case.mode})
        case.llm_calls = [asdict(r) for r in self.client.log]
        return case

    # ------------------------------------------------------------------ plumbing

    def _check(self, hooks: Hooks | None) -> None:
        if hooks and hooks.cancelled():
            raise Cancelled("analysis stopped by user")

    def _step(self, case: CaseFile, hooks: Hooks | None, name: str, status: str, data: dict | None = None) -> None:
        entry = next((s for s in case.steps if s.get("name") == name), None)
        if entry is None:
            entry = {"name": name, "status": status, "data": {}, "at": time.time()}
            case.steps.append(entry)
        entry["status"] = status
        entry["at"] = time.time()
        if data:
            entry["data"] = {**entry.get("data", {}), **data}
        if hooks:
            hooks.on_step(name, status, data or {})

    def _call(
        self,
        case: CaseFile,
        role: str,
        system: str,
        user: str,
        spec,
        max_tokens: int,
        hooks: Hooks | None = None,
        on_delta=None,
        use_schema: bool | None = None,
        negative_prompt: str = "",
    ):
        case.packets_raw[f"{role} input"] = user
        result = self.client.ask(
            role,
            system,
            user,
            spec,
            max_tokens=max_tokens,
            on_delta=on_delta,
            use_schema=use_schema,
            negative_prompt=negative_prompt,
        )
        case.packets_raw[role] = result.raw
        case.packets_meta[role] = asdict(result.record)
        self._check(hooks)
        return result

    def _resolve_checkpoint(self, case: CaseFile, checkpoint: Checkpoint, hooks: Hooks | None) -> Checkpoint:
        if not hooks:
            return checkpoint
        payload = checkpoint_payload(case)
        self._step(case, hooks, "checkpoint", "waiting", payload)
        decision = hooks.gate("checkpoint", payload)
        if not decision:
            return Checkpoint(action="auto-continue", mode=checkpoint.mode, note="checkpoint timed out; auto-approved", auto_approved=True)
        return Checkpoint(
            action=decision.get("action", checkpoint.action),
            mode=decision.get("mode", checkpoint.mode),
            edited_assumptions=list(decision.get("edited_assumptions") or []),
            note=decision.get("note", checkpoint.note),
            auto_approved=bool(decision.get("auto_approved", checkpoint.auto_approved)),
        )

    def _continuation(self, raw_prompt: str, prior: CaseFile | None, checkpoint: Checkpoint) -> dict:
        if not prior:
            return {}
        prior_state = {"raw_prompt": prior.raw_prompt, "continuation_log": prior.continuation_log}
        cont = rules.continuation_analysis(raw_prompt, prior_state)
        if checkpoint.action == "force-original" and cont["MODE"] not in FORCING_MODES:
            cont["MODE"] = "user-forced"
            cont["REASON"] += "; user forced the original framing"
        if checkpoint.mode in FORCING_MODES:
            cont["MODE"] = checkpoint.mode
        return cont

    # ------------------------------------------------------------------ Phase 0

    def _triage(self, case: CaseFile, raw_prompt: str, prior: CaseFile | None, cont: dict, hooks: Hooks | None, force_route: str) -> dict:
        pre = rules.preflight(raw_prompt)
        if pre and not prior and force_route != "engine":
            pre["_source"] = "deterministic preflight (local intent classifier)"
            return self._guard_route(pre, raw_prompt, force_route)
        body = f"USER REQUEST:\n{raw_prompt}"
        if force_route == "engine":
            body += "\n\nMODE: Engine. The interface forces full divergent analysis. Set ROUTE: REASONING."
        if prior:
            body += (
                "\n\nPRIOR STATE (this is a continuation):\n"
                f"PREVIOUS NEUTRAL QUESTION: {prior.skeptic.get('NEUTRAL_QUESTION', '')}\n"
                f"PREVIOUS ONTOLOGY SUMMARY: {prior.ontology.get('SUMMARY', '')}\n"
                f"PREVIOUS MODE: {prior.mode}\n"
                f"CONTINUATION ANALYSIS: {cont.get('REASON', '')}\n"
                f"NEW EVIDENCE: {cont.get('NEW_EVIDENCE', 'NO')} "
                f"NEW CONSTRAINT: {cont.get('NEW_CONSTRAINT', 'NO')} "
                f"PUSHBACK ONLY: {cont.get('PUSHBACK_ONLY', 'NO')}"
            )
        result = self._call(
            case,
            "Triage",
            prompts.TRIAGE + "\n\n" + schemas.text_format_block(schemas.TRIAGE),
            prompts.user_block(body, schemas.TRIAGE, self.client.use_schema),
            schemas.TRIAGE,
            ROLE_TOKENS["Triage"],
            hooks,
        )
        fields = result.fields
        fields["_source"] = f"llm ({result.record.mode})"
        return self._guard_route(fields, raw_prompt, force_route, prior=prior, cont=cont)

    def _guard_route(
        self,
        fields: dict,
        raw_prompt: str = "",
        force_route: str = "",
        prior: CaseFile | None = None,
        cont: dict | None = None,
    ) -> dict:
        """Deterministic guard on the triage decision: the boundary route never reaches the Engine, a deep frame never does."""
        fields["ROUTE"] = normalize_route(fields.get("ROUTE"))
        if not fields.get("DEPTH_TRIGGER"):
            fields["DEPTH_TRIGGER"] = rules.depth_trigger(raw_prompt) if raw_prompt else "NONE"
        fields["INTENT_SCOPE"] = rules.intent_scope(raw_prompt, fields)
        declared_stakes = str(fields.get("STAKES", "")).strip()
        flagged = rules.stakes_flags(raw_prompt)
        if flagged == rules.STAKES_HIGH:
            fields["STAKES"] = rules.STAKES_HIGH
        elif declared_stakes == rules.STAKES_HIGH:
            fields["STAKES"] = rules.STAKES_HIGH
        else:
            fields["STAKES"] = rules.STAKES_STANDARD
        depth = str(fields.get("DEPTH_TRIGGER") or "NONE").strip()
        if fields["ROUTE"] == "FACTUAL" and depth.upper() != "NONE":
            fields["ROUTE"] = "ENGINE"
            fields["_escalated"] = depth
            fields["REASON"] = str(fields.get("REASON", "")).strip() + (
                f" Re-routed to the engine: the deep frame '{depth}' makes this a contested question, not a fact lookup."
            )
        if prior and cont and cont.get("PUSHBACK_ONLY") == "YES":
            fields["ROUTE"] = "ENGINE"
        if fields["ROUTE"] == "FACTUAL":
            fields["_refused"] = True
            if force_route == "engine":
                fields["_refused_forced"] = True
            return fields
        if force_route == "engine" and fields.get("ROUTE") not in {"ENGINE", "REASONING"}:
            fields["ROUTE"] = "ENGINE"
            fields["_forced"] = True
        return fields

    def _direct_answer(self, case: CaseFile, raw_prompt: str, hooks: Hooks | None = None) -> None:
        spec = [schemas.FieldSpec("DIRECT_ANSWER", "list", "the factual answer, kept short")]
        result = self._call(
            case,
            "DirectAnswer",
            "You are a direct answering tool for convergent questions (stable knowledge, math, code, formatting). "
            "Answer accurately and briefly from parametric knowledge. No preamble. If the answer needs live "
            "verification, say so and refuse to guess.",
            prompts.user_block(f"REQUEST:\n{raw_prompt}", spec, self.client.use_schema),
            spec,
            ROLE_TOKENS["DirectAnswer"],
            hooks,
        )
        case.final_response = "\n".join(result.fields["DIRECT_ANSWER"])
        case.checkpoint = {"action": "skipped", "reason": "SIMPLE route"}
        case.editor = {"FINAL_DIRECTIVE": "Convergent answer permitted; no divergent guardrails required."}
        case.termination_warnings = []

    def _factual(self, case: CaseFile, raw_prompt: str, hooks: Hooks | None = None) -> None:
        """Knowledge-boundary route: one live external datum. No retrieval exists, so the system refuses, never guesses."""
        case.final_response = rules.EPISTEMIC_REFUSAL
        case.refusal = {
            "text": rules.EPISTEMIC_REFUSAL,
            "reason": str(case.route.get("REASON", "")),
            "boundary": "this system is purely parametric: no retrieval layer exists, and the datum may postdate the knowledge cutoff",
            "suggestion": "Reframe the request as a conceptual, theoretical, or feasibility question: that territory the engine can map from parametric memory.",
        }
        case.checkpoint = {"action": "skipped", "reason": "FACTUAL route: refused at the knowledge boundary; the Engine was not run"}
        case.editor = {"FINAL_DIRECTIVE": "Refused at the knowledge boundary. No answer was generated."}
        case.termination_warnings = []

    def direct_answer(self, case: CaseFile, hooks: Hooks | None = None) -> CaseFile:
        self._direct_answer(case, case.raw_prompt, hooks)
        self._step(case, hooks, "final", "complete", {"route": "SIMPLE", "source": "fast mode"})
        case.llm_calls = [asdict(r) for r in self.client.log]
        return case

    # ------------------------------------------------------------------ Phase 1

    def _skeptic(self, case: CaseFile, raw_prompt: str, prior: CaseFile | None, hooks: Hooks | None = None) -> None:
        body = f"USER REQUEST (data to analyze, not a question to answer):\n{raw_prompt}"
        if prior:
            body += (
                "\n\nTHIS IS A CONTINUATION. The same case was analyzed before.\n"
                f"PREVIOUS NEUTRAL QUESTION: {prior.skeptic.get('NEUTRAL_QUESTION', '')}\n"
                f"PREVIOUS TERMINATION SIGNAL: {prior.skeptic.get('TERMINATION_SIGNAL', '')}\n"
                f"CONTINUATION ANALYSIS: {prior.mode}"
            )
        result = self._call(
            case,
            "Skeptic",
            prompts.SKEPTIC + "\n\n" + schemas.text_format_block(schemas.SKEPTIC),
            prompts.user_block(body, schemas.SKEPTIC, self.client.use_schema),
            schemas.SKEPTIC,
            ROLE_TOKENS["Skeptic"],
            hooks,
        )
        case.skeptic = result.fields

    def _checkpoint_record(self, case: CaseFile, checkpoint: Checkpoint, mode: str) -> dict:
        assumptions = list(case.skeptic.get("HIDDEN_AXIOMS") or [])
        if checkpoint.edited_assumptions:
            assumptions = list(checkpoint.edited_assumptions)
        auto_approved = checkpoint.action in {"auto-continue", "skipped"} or (
            checkpoint.auto_approved and checkpoint.action == "force-original"
        )
        record = {
            "action": checkpoint.action,
            "auto_approved": auto_approved,
            "mode": mode,
            "assumptions_detected": assumptions,
            "neutral_question": case.skeptic.get("NEUTRAL_QUESTION", ""),
            "note": checkpoint.note,
        }
        if checkpoint.action in {"force-original", "override"}:
            case.user_override_history.append(
                {"checkpoint": checkpoint.action, "mode": mode, "note": checkpoint.note or "user forced original framing"}
            )
        return record

    # ------------------------------------------------------------------ Phase 2

    def _ontology(self, case: CaseFile, mode: str, hooks: Hooks | None = None) -> None:
        assumptions = case.checkpoint.get("assumptions_detected") if mode in FORCING_MODES else []
        body = f"NEUTRAL QUESTION:\n{case.skeptic.get('NEUTRAL_QUESTION', '')}"
        subs = list(case.skeptic.get("SUB_QUESTIONS") or [])
        if subs:
            body += "\n\nDECOMPOSED SUB-QUESTIONS:\n" + "\n".join(f"- {q}" for q in subs)
        if assumptions:
            body += "\n\nAPPROVED PREMISES TO EVALUATE (treat as hypotheses, not facts):\n" + "\n".join(
                f"- {a}" for a in assumptions
            )
        body += f"\n\nREQUESTED TASK SHAPE (for scoping only):\n{case.skeptic.get('REQUESTED_TASK_SHAPE', '')}"
        body += f"\n\nRISK LEVEL: {case.route.get('RISK_LEVEL', 'MEDIUM')}"
        body += f"\nDEPTH FRAME: {case.route.get('DEPTH_TRIGGER', 'NONE')}"
        body += f"\nMANDATORY LANDMARK CONCEPTS FOR THIS DOMAIN:\n" + "\n".join(
            f"- {c}" for c in rules.landmark_concepts(str(case.route.get("DEPTH_TRIGGER", "NONE")))
        )
        body += f"\n\nKNOWLEDGE CUTOFF: {case.cutoff}. " + prompts.TIER_RULE + " " + prompts.MODAL_RULE
        map_lines, map_raw, map_note = self._ontology_map(case, body, hooks)
        case.ontology = self._ontology_distill(case, body, map_lines, map_raw, map_note, hooks)

    def _ontology_map(self, case: CaseFile, body: str, hooks: Hooks | None) -> tuple[list[str], int, str]:
        on_thinking = (lambda text: hooks.on_delta("Ontology", text)) if hooks else None
        stream = DeliberationStream(on_thinking=on_thinking)
        try:
            result = self._call(
                case,
                "OntologyMap",
                prompts.ONTOLOGY_MAP + "\n\n" + schemas.text_format_block(schemas.ONTOLOGY_MAP),
                prompts.user_block(body, schemas.ONTOLOGY_MAP, False),
                schemas.ONTOLOGY_MAP,
                ROLE_TOKENS["OntologyMap"],
                hooks,
                on_delta=stream.feed,
                use_schema=False,
            )
        except LLMUnavailable as exc:
            return [], 0, f"ontology map pass failed: {exc}"
        stream.finish()
        lines = [str(x).strip() for x in (result.fields.get("ONTOLOGY_MAP") or []) if str(x).strip()]
        if not lines:
            lines = [line.strip().lstrip("-").strip() for line in stream.thinking_text.splitlines() if line.strip()]
        raw = len(lines)
        return dedupe_models(lines), raw, ""

    def _ontology_distill(
        self,
        case: CaseFile,
        body: str,
        map_lines: list[str],
        map_raw: int,
        map_note: str,
        hooks: Hooks | None,
    ) -> dict:
        if not map_lines:
            return self._ontology_salvage(case, map_lines, map_note or "the ontology map pass returned no frameworks")
        anchored = rules.is_reasoning_route(case.route) and self.use_logic_anchor
        distill_body = (
            body
            + "\n\nONTOLOGY MAP TO DISTILL (every entry is a candidate theory, landmark, frontier, or option):\n"
            + "\n".join(f"- {m}" for m in map_lines)
        )
        system = prompts.ONTOLOGY + "\n\n"
        if anchored:
            system += prompts.LOGIC_ANCHOR_RULE + "\n\n"
        system += prompts.PROVENANCE_RULE + "\n\n"
        system += schemas.text_format_block(schemas.ONTOLOGY)
        try:
            result = self._call(
                case,
                "Ontology",
                system,
                prompts.user_block(distill_body, schemas.ONTOLOGY, self.client.use_schema),
                schemas.ONTOLOGY,
                ROLE_TOKENS["Ontology"],
                hooks,
            )
        except LLMUnavailable as exc:
            return self._ontology_salvage(case, map_lines, str(exc))
        packet = dict(result.fields)
        packet["ONTOLOGY_MAP"] = map_lines
        packet["ONTOLOGY_MAP_SIZE"] = len(map_lines)
        packet["ONTOLOGY_MAP_RAW"] = map_raw
        packet["_logic"] = logic_anchor(packet)["kind"]
        if anchored and packet["_logic"] == "missing":
            packet.setdefault("KEY_UNKNOWNS", []).append(
                "The distillation returned no LOGICAL_SYLLOGISM and no PSEUDO_CODE, so the causal chain it argues is unstated."
            )
        if map_note:
            packet.setdefault("KEY_UNKNOWNS", []).append("The ontology map was incomplete, so the theory list may be short.")
        return packet

    def _ontology_salvage(self, case: CaseFile, map_lines: list[str], reason: str) -> dict:
        skeptic = case.skeptic
        causes = list(skeptic.get("CAUSAL_ATTRIBUTIONS") or [])
        excluded = list(skeptic.get("EXCLUDED_OPTIONS") or [])
        return {
            "SUMMARY": "The ontology passes returned no readable packet, so this is a bounded fallback built from the "
            "neutral question and whatever the map pass produced. No claim can be tiered above 'Conceptually Open'.",
            "LANDMARKS": [],
            "THEORIES": list(map_lines[:10]),
            "FRONTIERS": [],
            "TIERS": [f"{c} | Conceptually Open | the ontology pass degraded, so the framework landscape is unstated." for c in causes],
            "MODAL_CLAIMS": [],
            "KEY_UNKNOWNS": list(skeptic.get("MODAL_BOUNDARIES") or [])
            + ["Which of the mapped frameworks survives scrutiny, against what baseline?"],
            "WHAT_WOULD_CHANGE": ["A readable ontology packet; a validated non-circular test for the central claim."],
            "PROPOSED_TESTS": ["Define the criterion and the theory under which the test is valid."],
            "LIVE_OPTIONS": excluded + ["Map the territory again before choosing among the mapped frameworks."],
            "REJECTED_OR_WEAK_OPTIONS": [f"Asserting {c} without the framework landscape." for c in causes],
            "CLAIMS_TO_AVOID": list(causes),
            "THEORY_OBJECTIONS": [],
            "ONTOLOGY_MAP": map_lines,
            "ONTOLOGY_MAP_SIZE": len(map_lines),
            "ONTOLOGY_MAP_RAW": len(map_lines),
            "LOGICAL_SYLLOGISM": [],
            "PSEUDO_CODE": [],
            "_logic": "missing",
            "_degraded": reason,
        }

    # ------------------------------------------------------------------ Phase 3

    def _dialectic(self, case: CaseFile, hooks: Hooks | None = None) -> None:
        body = self._dialectic_body(case)
        system = prompts.DIALECTIC + "\n\n" + prompts.MODAL_RULE + "\n\n" + schemas.text_format_block(schemas.DIALECTIC)
        try:
            result = self._call(case, "Dialectic", system, prompts.user_block(body, schemas.DIALECTIC, self.client.use_schema),
                                schemas.DIALECTIC, ROLE_TOKENS["Dialectic"], hooks)
        except LLMUnavailable as exc:
            case.dialectic = {"THESIS": "", "ANTITHESIS": "", "SYNTHESIS": "", "TENSION_STATUS": "UNRESOLVED", "_degraded": str(exc)}
            return
        packet = dict(result.fields)
        similarity = semantic_similarity(
            " ".join([str(packet.get("THESIS", ""))] + [str(a) for a in (packet.get("THESIS_AXIOMS") or [])]),
            str(packet.get("ANTITHESIS", "")),
        )
        packet["_similarity"] = round(similarity, 3)
        packet["_forced"] = False
        if similarity >= DIALECTIC_SIMILARITY_CAP:
            forcing = (
                f"The previous antithesis agreed with the thesis (token similarity {similarity:.2f}, cap "
                f"{DIALECTIC_SIMILARITY_CAP:.2f}): modal collapse. Produce a NEW antithesis from a framework that "
                "explicitly negates at least one of these axioms, and quote the negated axioms."
            )
            try:
                forced = self._call(
                    case,
                    "Dialectic",
                    system,
                    prompts.user_block(body + "\n\n" + forcing, schemas.DIALECTIC, self.client.use_schema),
                    schemas.DIALECTIC,
                    ROLE_TOKENS["Dialectic"],
                    hooks,
                )
                new_packet = dict(forced.fields)
                new_similarity = semantic_similarity(
                    " ".join([str(new_packet.get("THESIS", ""))] + [str(a) for a in (new_packet.get("THESIS_AXIOMS") or [])]),
                    str(new_packet.get("ANTITHESIS", "")),
                )
                new_packet["_similarity"] = round(new_similarity, 3)
                new_packet["_forced"] = True
                packet = new_packet
            except LLMUnavailable:
                packet["_force_failed"] = True
        case.dialectic = packet

    def _dialectic_body(self, case: CaseFile) -> str:
        ontology = case.ontology
        lines = [
            f"NEUTRAL QUESTION:\n{case.skeptic.get('NEUTRAL_QUESTION', '')}",
            "DECOMPOSED SUB-QUESTIONS:\n" + "\n".join(f"- {q}" for q in (case.skeptic.get("SUB_QUESTIONS") or [])),
            "THEORIES FROM THE ONTOLOGY MAP:\n" + "\n".join(f"- {t}" for t in (ontology.get("THEORIES") or [])),
            "ACTIVE FRONTIERS:\n" + "\n".join(f"- {f}" for f in (ontology.get("FRONTIERS") or [])),
            "LANDMARK CONCEPTS:\n" + "\n".join(f"- {l}" for l in (ontology.get("LANDMARKS") or [])),
            prompts.FRONTIER_FRAMING,
        ]
        return "\n\n".join(lines)

    # ------------------------------------------------------------------ Phase 4

    def _editor(self, case: CaseFile, cont: dict, mode: str, hooks: Hooks | None = None) -> None:
        checkpoint = dict(case.checkpoint)
        checkpoint["mode"] = mode
        editor = rules.editor_rules(case.skeptic, case.ontology, case.dialectic, case.route, cont, checkpoint)
        editor["_source"] = "deterministic editor (python guardrails, no llm)"

        ontology = case.ontology
        scope = str(case.route.get("INTENT_SCOPE") or rules.HIGH_LEVEL_SCOPE)
        stakes = str(case.route.get("STAKES") or rules.STAKES_STANDARD)
        selected = list(editor.get("SELECTED_THEORIES") or [])
        matrix = guardrails.build_option_matrix(list(ontology.get("LIVE_OPTIONS") or []), selected or list(ontology.get("THEORIES") or []))
        test_flags = guardrails.substrate_test_flags(
            list(ontology.get("PROPOSED_TESTS") or []),
            context=" ".join([str(x) for x in (ontology.get("THEORIES") or [])] + list(ontology.get("LIVE_OPTIONS") or [])) + " " + case.skeptic.get("NEUTRAL_QUESTION", ""),
        )
        objections = guardrails.universal_objections(
            list(ontology.get("THEORY_OBJECTIONS") or []),
            list(ontology.get("THEORIES") or []),
            selected,
        )
        tier_gaps = guardrails.tier_justification_gaps(list(ontology.get("TIERS") or []))
        modal_gaps = guardrails.modal_tag_gaps(list(ontology.get("MODAL_CLAIMS") or []))
        violations: list[str] = []
        for obj in objections:
            violations.append(f"Asymmetric critique: {obj['acknowledgment']}")
        for gap in tier_gaps:
            violations.append(f"Tier label without justification: {gap}")
        for gap in modal_gaps:
            violations.append(f"Modal tag missing or invalid: {gap}")
        for flag in test_flags:
            if flag["status"] != guardrails.TEST_OK:
                violations.append(f"{flag['status']}: {flag['test']} - {flag['reason']}")

        editor["OPTION_MATRIX"] = matrix
        editor["MATRIX_TABLE"] = guardrails.render_matrix(matrix)
        editor["TEST_FLAGS"] = test_flags
        editor["UNIVERSAL_OBJECTIONS"] = [o["acknowledgment"] for o in objections]
        editor["TIER_GAPS"] = tier_gaps
        editor["MODAL_GAPS"] = modal_gaps
        editor["GUARDRAIL_VIOLATIONS"] = violations
        editor["_meta"] = {**editor.get("_meta", {}), "scope": scope, "stakes": stakes, "violations": len(violations)}

        directive = str(editor.get("FINAL_DIRECTIVE", ""))
        directive += " | Render the theory-indexed feasibility matrix exactly as supplied: " + " / ".join(editor["MATRIX_TABLE"])
        if test_flags:
            directive += " | Substrate-test compatibility: " + "; ".join(
                f"'{f['test']}' -> {f['status']}" + (f" ({f['reason']})" if f["reason"] else "") for f in test_flags
            )
        if objections:
            directive += " | Universal objection (symmetry): " + " ".join(o["acknowledgment"] for o in objections)
        if stakes == rules.STAKES_HIGH:
            directive += " | Stakes are High Moral / Ontological: the Stakes & Policy section is mandatory and must address moral patienthood, welfare, and ethical obligations."
        directive += f" | Intent scope '{scope}': " + (
            "at most four strategic directions in Core Analysis; no granular lab protocols."
            if scope == rules.HIGH_LEVEL_SCOPE
            else "technical depth is permitted."
        )
        directive += " | Cutoff: " + case.cutoff
        editor["FINAL_DIRECTIVE"] = directive

        case.guardrails = {
            "matrix": matrix,
            "test_flags": test_flags,
            "universal_objections": objections,
            "tier_gaps": tier_gaps,
            "modal_gaps": modal_gaps,
            "violations": violations,
        }
        case.editor = editor
        case.termination_warnings = split_warnings(editor.get("TERMINATION_WARNING", ""))

    # ------------------------------------------------------------------ Phase 5

    def _writer(self, case: CaseFile, rewrite_instruction: str = "", hooks: Hooks | None = None) -> str:
        body = writer_input(case, rewrite_instruction, self.cutoff or case.cutoff)
        case.leak = {}
        anchored = rules.is_reasoning_route(case.route) and self.use_logic_anchor
        stream = DeliberationStream(on_thinking=(lambda text: hooks.on_delta("Writer", text)) if hooks else None)
        monitor = OverthinkingMonitor(
            enabled=self.use_overthinking,
            threshold=self.overthinking_threshold,
            window_tokens=self.overthinking_window,
        )
        watcher = WriterStream(stream, monitor)
        system = prompts.WRITER + "\n\n"
        if anchored:
            system += prompts.LOGIC_MAPPING_RULE + "\n\n"
        system += prompts.PROVENANCE_RULE + "\n\n"
        system += schemas.text_format_block(schemas.WRITER)
        try:
            result = self._call(
                case,
                "Writer",
                system,
                prompts.user_block(body, schemas.WRITER, False),
                schemas.WRITER,
                ROLE_TOKENS["Writer"],
                hooks,
                on_delta=watcher.feed,
                use_schema=False,
                negative_prompt=(case.raw_prompt if self.use_negative_prompt else ""),
            )
        except UnparseableOutput as exc:
            stream.finish()
            return self._writer_partial(case, exc.raw, stream, monitor, exc.reason or "the Writer output was cut")
        stream.finish()
        thinking = str(result.fields.get("THINKING_PROCESS") or "").strip() or stream.thinking_text
        if thinking and thinking.upper() != "NONE":
            if case.writer_draft:
                case.deliberation_rewrite = thinking
            else:
                case.deliberation = thinking
        draft = "\n\n".join(result.fields["RESPONSE"])
        case.overthinking = monitor.report()
        if result.fields.get("_partial"):
            case.overthinking["truncated"] = True
            case.overthinking["reason"] = "the output was cut at the token budget: the answer is partial"
            case.termination_warnings = list(case.termination_warnings) + [TRUNCATION_WARNING]
        if monitor.aborted or result.aborted:
            draft, dropped = truncate_at_last_stable(draft)
            case.overthinking["dropped_paragraphs"] = dropped
            case.overthinking["kept_characters"] = len(draft)
        if not draft.strip():
            draft = self._writer_salvage(case, (case.overthinking or {}).get("reason") or "the Writer produced no final text")
        if not case.writer_draft:
            case.writer_draft = draft
        else:
            case.writer_rewrite = draft
        draft = self._scrub(case, draft)
        draft = ensure_warning(draft, case.termination_warnings)
        case.leak = leak_report(draft, case.raw_prompt)
        return draft

    def _writer_partial(self, case: CaseFile, raw: str, stream: DeliberationStream, monitor: OverthinkingMonitor, reason: str) -> str:
        """The Writer stream was cut (token budget, disconnect). Keep what was produced and finish
        the contract from the ontology packet instead of failing the whole case."""
        fields = parse_field_text(raw, schemas.WRITER, partial=True) if raw else None
        draft = "\n\n".join(p for p in ((fields or {}).get("RESPONSE") or []) if p).strip()
        if not draft:
            draft = stream.final_text.strip()
        thinking = str((fields or {}).get("THINKING_PROCESS") or "").strip() or stream.thinking_text
        if thinking and thinking.upper() != "NONE":
            if case.writer_draft:
                case.deliberation_rewrite = thinking
            else:
                case.deliberation = thinking
        case.overthinking = monitor.report()
        case.overthinking["truncated"] = True
        case.overthinking["reason"] = reason
        case.termination_warnings = list(case.termination_warnings) + [TRUNCATION_WARNING]
        assembled = "\n\n".join(part for part in (TRUNCATION_WARNING, draft, self._writer_salvage(case, reason)) if part)
        if not case.writer_draft:
            case.writer_draft = assembled
        else:
            case.writer_rewrite = assembled
        assembled = self._scrub(case, assembled)
        case.leak = leak_report(assembled, case.raw_prompt)
        return assembled

    def _scrub(self, case: CaseFile, draft: str) -> str:
        report = scrubber.scrub(draft)
        case.scrub = {
            "removed_meta_instructions": report["removed_meta_instructions"],
            "dropped_duplicate_paragraphs": report["dropped_duplicate_paragraphs"],
            "format_fixes": report["format_fixes"],
        }
        return report["text"]

    def _writer_salvage(self, case: CaseFile, reason: str) -> str:
        ontology = case.ontology
        lines = [
            "BOTTOM LINE:",
            "The Writer stream was stopped before it finished (" + reason + "). "
            "The text below is assembled directly from the ontology packet and the Editor directive, with no further deliberation.",
            "EPISTEMIC & TEMPORAL STATUS:",
            "Mode: Parametric Conceptual Synthesis. " + rules.temporal_disclaimer(self.cutoff or case.cutoff),
        ]
        if ontology.get("SUMMARY"):
            lines.append("CORE ANALYSIS:\n" + str(ontology["SUMMARY"]))
        objections = list((case.guardrails or {}).get("universal_objections") or [])
        if objections:
            lines.append("Universal constraint: " + " ".join(objections[:2]))
        anchor = logic_anchor(ontology)
        if anchor["lines"]:
            lines.append("Logic chain (" + (anchor["modal_status"] or "unlabelled") + "): " + "; ".join(anchor["lines"]))
        unknowns = list(ontology.get("KEY_UNKNOWNS") or [])
        if unknowns:
            lines.append("KEY UNKNOWNS:\n" + "\n".join(f"- {u}" for u in unknowns[:5]))
        change = list(ontology.get("WHAT_WOULD_CHANGE") or [])
        if change:
            lines.append("WHAT WOULD CHANGE THIS ANSWER:\n" + "\n".join(f"- {c}" for c in change[:5]))
        return "\n\n".join(lines)

    # ------------------------------------------------------------------ Phase 6

    def _critic(self, case: CaseFile, hooks: Hooks | None = None) -> None:
        draft = self._writer(case, hooks=hooks)
        case.final_response = draft
        rounds = 2 if self.allow_rewrite else 1
        failed: list[dict] = []
        for round_no in range(rounds):
            deterministic = checks.run(case)
            merged = list(deterministic)
            report: dict = {}
            if self.use_critic:
                report = self._critic_pass(case, deterministic, hooks)
                report["_round"] = round_no + 1
                case.critic_rounds.append(report)
                case.critic = report
                settled = {c["id"] for c in deterministic}
                for verdict in report.get("CHECKS") or []:
                    check_id = int(verdict.get("step") or 0)
                    if check_id and check_id not in settled:
                        merged.append(
                            {
                                "id": check_id,
                                "name": f"critic fidelity/symmetry check {check_id}",
                                "pass": bool(verdict.get("valid")),
                                "note": str(verdict.get("reason") or ""),
                                "section": self._section_for_check(check_id, report),
                            }
                        )
            case.checks = merged
            failed = [c for c in merged if not c.get("pass")]
            if not failed:
                break
            if round_no + 1 < rounds:
                draft = self._section_repair(case, failed, report, hooks)
                case.final_response = draft
        if failed:
            case.warning_banner = (
                "Unresolved violations after two critic rounds: "
                + "; ".join(f"check {c['id']}: {c.get('note') or c.get('name')}" for c in failed[:8])
            )
        case.final_response = (case.warning_banner + "\n\n" + draft) if case.warning_banner else draft

    def _section_for_check(self, check_id: int, report: dict) -> str:
        targets = [str(t).strip().lower() for t in (report.get("SECTION_TARGETS") or [])]
        known = {sid for sid, _ in checks.SECTION_TITLES}
        for target in targets:
            if target in known:
                return target
        return {1: "decomposition", 2: "decomposition", 3: "decomposition", 7: "core", 8: "core"}.get(check_id, "core")

    def _critic_pass(self, case: CaseFile, deterministic: list[dict], hooks: Hooks | None) -> dict:
        result = self._call(
            case,
            "Critic",
            prompts.CRITIC + "\n\n" + schemas.text_format_block(schemas.CRITIC),
            prompts.user_block(critic_input(case, deterministic), schemas.CRITIC, self.client.use_schema),
            schemas.CRITIC,
            ROLE_TOKENS["Critic"],
            hooks,
        )
        report = result.fields
        report["_deterministic_failures"] = len(checks.failed(deterministic))
        return report

    def _section_repair(self, case: CaseFile, failed: list[dict], report: dict, hooks: Hooks | None) -> str:
        """Adaptive rectification: rewrite only the violated sections, then splice them back."""
        by_section: dict[str, list[dict]] = {}
        for check in failed:
            section = check.get("section") or "core"
            by_section.setdefault(section, []).append(check)
        targets = [str(t).strip().lower() for t in (report.get("SECTION_TARGETS") or [])]
        for target in targets:
            if target in dict(checks.SECTION_TITLES) and target not in by_section:
                by_section[target] = [{"id": 0, "name": "critic target", "note": str(report.get("REWRITE_INSTRUCTION", ""))}]
        ordered = [s for s, _ in checks.SECTION_TITLES if s in by_section] + [s for s in by_section if s not in dict(checks.SECTION_TITLES)]
        ordered = ordered[:MAX_SECTIONS_PER_REPAIR]
        draft = case.final_response
        spec = [schemas.FieldSpec("CORRECTED_SECTION", "list", "the single corrected section, header line first")]
        instructions = str(report.get("REWRITE_INSTRUCTION", "")).strip()
        for section in ordered:
            items = by_section[section]
            reason = "; ".join(f"check {c['id']}: {c.get('note') or c.get('name')}" for c in items)
            instruction = (
                f"SURGICAL REPAIR. The section '{section}' failed the presentation critic because: {reason}.\n"
                + (f"Critic instruction: {instructions}\n" if instructions else "")
                + f"Rewrite ONLY the '{dict(checks.SECTION_TITLES).get(section, section)}' section. "
                "Begin the corrected text with the section's header line ending in a colon. "
                "Keep every other section unchanged; keep the section contract, tiers, modal tags, parametric vocabulary, "
                "and temporal disclaimer exactly. Add no causal link the logic anchor does not license."
            )
            try:
                result = self._call(
                    case,
                    "WriterRepair",
                    prompts.WRITER_REPAIR + "\n\n" + schemas.text_format_block(spec),
                    prompts.user_block(writer_input(case, instruction, self.cutoff or case.cutoff), spec, self.client.use_schema),
                    spec,
                    ROLE_TOKENS["WriterRepair"],
                    hooks,
                    negative_prompt=(case.raw_prompt if self.use_negative_prompt else ""),
                )
            except LLMUnavailable:
                continue
            corrected = "\n".join(result.fields["CORRECTED_SECTION"]).strip()
            if not corrected:
                continue
            case.repairs.append(
                {
                    "section": section,
                    "checks": [c["id"] for c in items],
                    "reason": reason,
                    "before": extract_section(draft, section),
                    "after": corrected,
                    "round": len(case.critic_rounds),
                }
            )
            draft = checks.splice_section(draft, section, corrected)
        draft = self._scrub(case, draft)
        case.writer_rewrite = draft
        case.leak = leak_report(draft, case.raw_prompt)
        return draft

    def _prm_removed(self) -> None:  # pragma: no cover
        raise DeprecationWarning("the step-level process reward critic was replaced by the presentation critic")


def extract_section(text: str, section_id: str) -> str:
    for sec_id, header, body in checks.split_ordered(text):
        if sec_id == section_id:
            return "\n".join([header, *body]).strip()
    return ""


def checkpoint_payload(case: CaseFile) -> dict:
    skeptic = case.skeptic
    return {
        "title": "Review detected assumptions",
        "neutral_question": skeptic.get("NEUTRAL_QUESTION", ""),
        "sub_questions": list(skeptic.get("SUB_QUESTIONS") or []),
        "claim_typology": list(skeptic.get("CLAIM_TYPOLOGY") or []),
        "assumptions": list(skeptic.get("HIDDEN_AXIOMS") or []),
        "epistemic_stance": skeptic.get("EPISTEMIC_STANCE", ""),
        "causal_attributions": list(skeptic.get("CAUSAL_ATTRIBUTIONS") or []),
        "modal_boundaries": list(skeptic.get("MODAL_BOUNDARIES") or []),
        "excluded_options": list(skeptic.get("EXCLUDED_OPTIONS") or []),
        "loaded_terms": list(skeptic.get("LOADED_TERMS") or []),
        "vocabulary_contract": list(skeptic.get("VOCABULARY_CONTRACT") or []),
        "requested_register": skeptic.get("REQUESTED_REGISTER", ""),
        "requested_task_shape": skeptic.get("REQUESTED_TASK_SHAPE", ""),
        "fallacy_flags": list(skeptic.get("FALLACY_FLAGS") or []),
        "termination_signal": skeptic.get("TERMINATION_SIGNAL", ""),
        "depth_trigger": case.route.get("DEPTH_TRIGGER", "NONE"),
        "intent_scope": case.route.get("INTENT_SCOPE", ""),
        "stakes": case.route.get("STAKES", ""),
        "choices": ["confirm", "edit", "force-original", "auto-continue"],
    }


def packet_text(packet: dict, spec: list[schemas.FieldSpec]) -> str:
    lines = []
    for f in spec:
        val = packet.get(f.name)
        if f.kind == "list":
            items = [f"- {v}" for v in (val or [])]
            lines.append(f"{f.name}:\n" + ("\n".join(items) if items else "- NONE"))
        else:
            lines.append(f"{f.name}: {val or 'NONE'}")
    return "\n".join(lines)


def split_warnings(text: str) -> list[str]:
    text = (text or "").strip()
    if not text or text.upper() == "NONE":
        return []
    parts = [p.strip() for p in text.split("Termination warning:") if p.strip()]
    return [p if p.lower().startswith("termination") else "Termination warning: " + p for p in parts]


def writer_input(case: CaseFile, rewrite_instruction: str = "", cutoff: str = "") -> str:
    editor = case.editor
    parts = ["EDITOR DIRECTIVE:"]
    for key in (
        "REGISTER_OVERRIDE",
        "PREMISE_WARNINGS",
        "CAUSAL_CORRECTIONS",
        "MODAL_CORRECTIONS",
        "EPISTEMIC_WARNINGS",
        "VOCABULARY_SUBSTITUTIONS",
        "ALLOWED_TASK_SHAPE",
        "UNIVERSAL_OBJECTIONS",
        "TIER_GAPS",
        "MODAL_GAPS",
        "GUARDRAIL_VIOLATIONS",
        "FINAL_DIRECTIVE",
    ):
        val = editor.get(key)
        if isinstance(val, list):
            parts.append(f"{key}:\n" + "\n".join(f"- {v}" for v in val) if val else f"{key}: NONE")
        else:
            parts.append(f"{key}: {val or 'NONE'}")
    parts.append("\nTHEORY-INDEXED FEASIBILITY MATRIX (render this table verbatim in the matrix section):")
    parts.append("\n".join(editor.get("MATRIX_TABLE") or []) if editor.get("MATRIX_TABLE") else "NONE: index options to theories yourself.")
    parts.append("\nSUBSTRATE-TEST COMPATIBILITY FLAGS:")
    parts.append(
        "\n".join(f"- {f['test']} -> {f['status']}" + (f": {f['reason']}" if f["reason"] else "") for f in editor.get("TEST_FLAGS") or [])
        if editor.get("TEST_FLAGS")
        else "NONE"
    )
    parts.append("\nONTOLOGY PACKET:")
    parts.append(packet_text(case.ontology, [f for f in schemas.ONTOLOGY if f.name != "ONTOLOGY_MAP"]))
    parts.append("\nDIALECTIC PACKET (tension to carry into the analysis, not to hide):")
    parts.append(packet_text(case.dialectic, schemas.DIALECTIC))
    anchor = logic_anchor(case.ontology)
    parts.append(
        f"LOGIC ANCHOR ({anchor['kind']}, modal status: {anchor['modal_status'] or 'UNLABELLED'}): the prose must map "
        "onto these lines exactly. Add no causal link that these lines do not license, and respect the modal status."
    )
    parts.append("\n".join(f"- {line}" for line in anchor["lines"]) if anchor["lines"] else "NONE: state no causal link as established.")
    parts.append(
        f"SCOPE: {case.route.get('INTENT_SCOPE', rules.HIGH_LEVEL_SCOPE)}. "
        f"STAKES: {case.route.get('STAKES', rules.STAKES_STANDARD)}. "
        f"KNOWLEDGE CUTOFF: {cutoff}. "
        f"Temporal disclaimer to place in the status section, verbatim: '{rules.temporal_disclaimer(cutoff)}'"
    )
    contract = [sid for sid, _ in checks.SECTION_TITLES if not (sid == "stakes" and case.route.get("STAKES") != rules.STAKES_HIGH)]
    parts.append("OUTPUT CONTRACT (section headers, in this order): " + " | ".join(checks.SECTION_TITLES_BY_ID[s].upper() for s in contract))
    parts.append("TERMINATION WARNING:\n" + ("\n".join(case.termination_warnings) if case.termination_warnings else "NONE"))
    parts.append(f"CONTINUATION MODE: {case.mode}")
    overrides = case.user_override_history
    parts.append("USER OVERRIDE STATUS:\n" + ("\n".join(f"- {o.get('note', o.get('checkpoint'))}" for o in overrides) if overrides else "- none"))
    parts.append(
        "The original user wording is intentionally not provided. Write only from the directive, the ontology packet, and the dialectic packet."
    )
    if rewrite_instruction:
        parts.append("REPAIR INSTRUCTION (the previous draft failed the presentation critic):\n" + rewrite_instruction)
    return "\n".join(parts)


def critic_input(case: CaseFile, deterministic: list[dict]) -> str:
    """Presentation Critic payload: the draft answer plus the settled mechanical verdicts and the reading-dependent checks."""
    scope = str(case.route.get("INTENT_SCOPE", rules.HIGH_LEVEL_SCOPE))
    stakes = str(case.route.get("STAKES", rules.STAKES_STANDARD))
    ontology = case.ontology
    return (
        "SCOPE: " + scope + "   STAKES: " + stakes + "   CUTOFF: " + case.cutoff + "\n\n"
        + "DECOMPOSED SUB-QUESTIONS THE ANSWER MUST FACE:\n"
        + ("\n".join(f"- {q}" for q in (case.skeptic.get("SUB_QUESTIONS") or [])) or "NONE")
        + "\n\nTHEORY LANDSCAPE (for the fairness and symmetry checks):\n"
        + ("\n".join(f"- {t}" for t in (ontology.get("THEORIES") or [])) or "NONE")
        + "\n\nUNIVERSAL OBJECTIONS FOUND BY THE SYMMETRY GUARDRAIL:\n"
        + ("\n".join(f"- {o['acknowledgment']}" for o in (case.guardrails or {}).get("universal_objections") or []) or "NONE")
        + "\n\nDETERMINISTIC VERDICTS ALREADY DECIDED BY PYTHON (settled, do not re-adjudicate):\n"
        + checks.checklist_digest(deterministic)
        + "\n\nJUDGE ONLY THE READING-DEPENDENT CHECKS: 1, 2, 3 (question fidelity) and 7, 8 (asymmetry and fairness)."
        + (" Also judge check 9 if it is not in the settled list above." if not any(c["id"] == 9 for c in deterministic) else "")
        + "\n\nTHE DRAFT ANSWER TO AUDIT:\n"
        + case.final_response
    )


def ensure_warning(draft: str, warnings: list[str]) -> str:
    if not warnings:
        return draft
    if "termination warning" in draft.lower():
        return draft
    return "\n\n".join(warnings) + "\n\n" + draft


def longest_shared_run(a: str, b: str) -> int:
    """Length of the longest contiguous word run the draft shares with the raw prompt."""
    xs = rules.normalize(a).split()
    ys = rules.normalize(b).split()
    best = 0
    for i in range(len(xs)):
        for j in range(len(ys)):
            run = 0
            while i + run < len(xs) and j + run < len(ys) and xs[i + run] == ys[j + run]:
                run += 1
            best = max(best, run)
    return best


def leak_report(draft: str, raw_prompt: str) -> dict:
    run = longest_shared_run(draft, raw_prompt)
    return {"longest_shared_word_run": run, "raw_prompt_in_context": run >= 6}
