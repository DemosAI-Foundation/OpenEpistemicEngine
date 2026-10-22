"""Background analysis jobs, checkpoint gates, and event streams for the web UI."""

from __future__ import annotations

import queue
import threading
import time
import traceback
from dataclasses import dataclass, field

from epistemic import Checkpoint, Engine, LLMClient
from epistemic.casefile import CaseFile, new_case_id, title_of
from epistemic.llm import LLMUnavailable, MAX_CONTEXT_TOKENS, MAX_OUTPUT_TOKENS, MIN_OUTPUT_TOKENS
from epistemic.pipeline import Cancelled, Hooks
from web.store import Store

DEPTH_TOKENS = {"light": 8192, "standard": 16384, "deep": MAX_OUTPUT_TOKENS}
CHECKPOINT_TIMEOUT = 1800.0


class WebHooks(Hooks):
    def __init__(self, job: "Job"):
        self.job = job

    def on_step(self, name: str, status: str, data: dict | None = None) -> None:
        self.job.publish({"type": "step", "name": name, "status": status, "data": data or {}})
        self.job.snapshot()

    def on_delta(self, role: str, text: str) -> None:
        self.job.on_delta(role, text)

    def gate(self, name: str, payload: dict) -> dict:
        policy = self.job.settings.get("checkpoint_policy", "assumptions")
        if policy == "never":
            return {"action": "auto-continue", "auto_approved": True, "note": "checkpoint disabled in settings"}
        if policy == "assumptions" and not payload.get("assumptions"):
            return {"action": "auto-continue", "auto_approved": True, "note": "no assumptions detected; checkpoint skipped"}
        return self.job.wait_checkpoint(payload)

    def cancelled(self) -> bool:
        return self.job.cancel.is_set()


@dataclass
class Job:
    case_id: str
    store: Store
    manager: "Manager"
    case: CaseFile
    settings: dict
    mode: str
    prompt: str
    prior: CaseFile | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    checkpoint_event: threading.Event = field(default_factory=threading.Event)
    checkpoint_decision: dict = field(default_factory=dict)
    listeners: set = field(default_factory=set)
    lock: threading.RLock = field(default_factory=threading.RLock)
    state: dict = field(default_factory=dict)
    thread: threading.Thread | None = None
    finished: bool = False
    delta_buffer: str = ""
    delta_at: float = 0.0
    delta_chars: int = 0
    budget_tokens: int = 0

    def on_delta(self, role: str, text: str) -> None:
        now = time.time()
        with self.lock:
            self.delta_buffer += text
            self.delta_chars += len(text)
            if len(self.delta_buffer) < 700 and (now - self.delta_at) < 1.2:
                return
            payload = self.delta_buffer
            self.delta_buffer = ""
            self.delta_at = now
        self.state["deliberation_characters"] = self.delta_chars
        self.publish(
            {
                "type": "delta",
                "role": role,
                "text": payload,
                "characters": self.delta_chars,
                "tokens": self.delta_chars // 4,
                "budget_tokens": self.budget_tokens,
            }
        )
        self.snapshot()

    def publish(self, event: dict) -> None:
        event = dict(event)
        event["case_id"] = self.case_id
        event["at"] = time.time()
        with self.lock:
            if event.get("type") == "step":
                self.state["step"] = event.get("name")
                self.state["step_status"] = event.get("status")
            for listener in list(self.listeners):
                listener.put(event)

    def snapshot(self) -> None:
        self.store.save_case(self.case)

    def subscribe(self) -> queue.Queue:
        out: queue.Queue = queue.Queue()
        with self.lock:
            self.listeners.add(out)
        return out

    def unsubscribe(self, listener: queue.Queue) -> None:
        with self.lock:
            self.listeners.discard(listener)

    def wait_checkpoint(self, payload: dict) -> dict:
        self.checkpoint_decision = {}
        self.checkpoint_event.clear()
        self.state["checkpoint"] = payload
        self.case.steps = [
            {**s, "status": "waiting", "data": {**s.get("data", {}), "payload": payload}}
            if s.get("name") == "checkpoint"
            else s
            for s in self.case.steps
        ]
        self.snapshot()
        self.publish({"type": "checkpoint", "payload": payload})
        resolved = self.checkpoint_event.wait(CHECKPOINT_TIMEOUT)
        if self.cancel.is_set():
            return {}
        if not resolved:
            self.publish({"type": "note", "text": "Checkpoint timed out; the system continued with auto-approval."})
            return {"action": "auto-continue", "auto_approved": True, "note": "checkpoint timed out; auto-approved"}
        decision = dict(self.checkpoint_decision)
        self.publish({"type": "checkpoint-resolved", "decision": decision})
        return decision

    def resolve_checkpoint(self, decision: dict) -> bool:
        if not self.state.get("checkpoint"):
            return False
        self.checkpoint_decision = decision
        self.state["checkpoint"] = None
        self.checkpoint_event.set()
        return True

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, name=f"case-{self.case_id}", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        hooks = WebHooks(self)
        engine = self.manager.build_engine(self.settings)
        self.budget_tokens = int(self.settings.get("max_tokens") or 0)
        try:
            force = "engine" if self.mode == "engine" else ""
            self.case = engine.probe(
                self.prompt,
                prior=self.prior,
                case_id=self.case_id,
                hooks=hooks,
                force_route=force,
                settings=self.settings,
                case=self.case,
            )
            self.publish({"type": "triage", "triage": self.case.route})
            route = self.case.route.get("ROUTE")
            if route in {"SIMPLE", "FACTUAL"}:
                if self.case.refusal:
                    self.publish(
                        {
                            "type": "note",
                            "text": "Knowledge boundary: "
                            + str(self.case.refusal.get("text", ""))
                            + " "
                            + str(self.case.refusal.get("suggestion", "")),
                        }
                    )
                self.snapshot()
            elif self.mode == "fast" and fast_path_ok(self.case):
                self.publish({"type": "note", "text": "Fast mode: direct answer. No deep-frame, causal, persuasion, or certainty markers detected."})
                engine.direct_answer(self.case, hooks)
                self.snapshot()
            else:
                if self.mode == "fast":
                    self.publish({"type": "note", "text": "Fast mode escalated to the full engine: markers of a loaded request were detected."})
                self.case = engine.finish(self.case, Checkpoint(action="pending"), hooks=hooks)
                self.snapshot()
            self.report_degradation()
            self.check_repairs(engine)
            self.flush_deltas()
            self.publish(
                {
                    "type": "final",
                    "mode": self.case.mode,
                    "status": "complete",
                    "deliberation_characters": len(self.case.deliberation),
                    "completion_tokens": sum(c.get("completion_tokens", 0) for c in self.case.llm_calls),
                }
            )
        except Cancelled:
            self.case.error = "stopped by user"
            self.mark_failed()
            self.snapshot()
            self.publish({"type": "cancelled"})
        except LLMUnavailable as exc:
            self.case.error = f"local LLM unavailable: {exc}"
            self.mark_failed()
            self.snapshot()
            self.publish({"type": "error", "text": self.case.error, "kind": "llm-offline"})
        except Exception as exc:
            self.case.error = f"{type(exc).__name__}: {exc}"
            self.mark_failed()
            self.snapshot()
            self.publish({"type": "error", "text": self.case.error, "kind": "step-failure", "trace": traceback.format_exc()[-1500:]})
        finally:
            self.flush_deltas()
            self.finished = True
            self.state["checkpoint"] = None
            self.snapshot()

    def flush_deltas(self) -> None:
        with self.lock:
            tail = self.delta_buffer
            self.delta_buffer = ""
        if not tail:
            return
        self.publish(
            {
                "type": "delta",
                "role": "Writer",
                "text": tail,
                "characters": self.delta_chars,
                "tokens": self.delta_chars // 4,
                "budget_tokens": self.budget_tokens,
                "final": True,
            }
        )

    def report_degradation(self) -> None:
        ontology = self.case.ontology or {}
        degraded = str(ontology.get("_degraded") or "")
        if degraded:
            self.publish(
                {
                    "type": "note",
                    "text": "Ontology pass degraded: "
                    + degraded
                    + ". The analysis continues from a bounded fallback packet, and the raw ontology map is kept in the Evidence tab.",
                }
            )
        route = str((self.case.route or {}).get("ROUTE") or "")
        if route in {"ENGINE", "REASONING"} and str(ontology.get("_logic") or "") == "missing":
            self.publish(
                {
                    "type": "note",
                    "text": "Formal logic anchor missing: the Ontology pass returned no LOGICAL_SYLLOGISM and no PSEUDO_CODE, "
                    "so the causal chain is unstated and the Writer must not assert it.",
                }
            )
        tension = str((self.case.dialectic or {}).get("TENSION_STATUS") or "")
        if (self.case.dialectic or {}).get("_degraded"):
            self.publish(
                {
                    "type": "note",
                    "text": "Dialectical engine degraded: "
                    + str((self.case.dialectic or {}).get("_degraded"))
                    + ". No antithesis was generated, so the thesis was not negated and the tension is recorded as UNRESOLVED.",
                }
            )
        elif tension.upper() == "UNRESOLVED":
            self.publish(
                {
                    "type": "note",
                    "text": "Dialectical tension left UNRESOLVED: thesis and antithesis cannot be reconciled at the knowledge cutoff; the answer carries the tension instead of hiding it.",
                }
            )
        over = self.case.overthinking or {}
        if over.get("aborted"):
            self.publish(
                {
                    "type": "note",
                    "text": "Overthinking monitor: "
                    + str(over.get("reason", ""))
                    + ". The stream was aborted and the answer truncated at the last stable paragraph "
                    + f"({over.get('dropped_paragraphs', 0)} paragraph(s) dropped).",
                }
            )
        for repair in self.case.repairs:
            self.publish(
                {
                    "type": "note",
                    "text": f"Surgical section repair: the '{repair.get('section')}' section failed the presentation critic ({repair.get('reason')}). "
                    "That section was rewritten in place; every other section was left untouched.",
                }
            )

    def check_repairs(self, engine: Engine) -> None:
        text_by_design = {"Writer", "WriterRepair", "OntologyMap"}
        for role, meta in (self.case.packets_meta or {}).items():
            if role in text_by_design:
                continue
            if meta.get("mode") == "text" and meta.get("attempts"):
                self.publish(
                    {
                        "type": "note",
                        "text": f"The model produced an unstructured response for {role}. A field-format repair pass was used.",
                        "role": role,
                    }
                )

    def mark_failed(self) -> None:
        for step in self.case.steps:
            if step.get("status") in {"running", "waiting"}:
                step["status"] = "failed"


def fast_path_ok(case: CaseFile) -> bool:
    route = case.route or {}
    if str(route.get("ROUTE", "")).upper() in {"FACTUAL", "REFUSAL"}:
        return False
    if str(route.get("COMPLEXITY", "")).upper() != "LOW":
        return False
    if str(route.get("RISK_LEVEL", "")).upper() != "LOW":
        return False
    if str(route.get("DEPTH_TRIGGER", "NONE")).upper() != "NONE":
        return False
    return str(route.get("TERMINATION_SIGNAL", "NONE")).upper() == "NONE"


class Manager:
    def __init__(self, store: Store, url: str = "http://127.0.0.1:8082"):
        self.store = store
        self.url = url
        self.jobs: dict[str, Job] = {}
        self.health_cache: dict = {"online": False, "latency_ms": None, "last_ok": 0.0, "model": "", "error": "not checked"}
        self.health_at = 0.0
        self.lock = threading.RLock()

    def health(self, max_age: float = 5.0) -> dict:
        now = time.time()
        if now - self.health_at < max_age:
            return self.health_cache
        settings = self.store.settings()
        started = time.time()
        try:
            import requests

            data = requests.get(f"{self.url}/health", timeout=5).json()
            models = requests.get(f"{self.url}/v1/models", timeout=5).json()
            ids = [m.get("id") for m in models.get("data", [])]
            self.health_cache = {
                "online": bool(data.get("loaded")),
                "degraded": not bool(data.get("loaded")),
                "latency_ms": round((time.time() - started) * 1000),
                "last_ok": now,
                "model": data.get("model") or (ids[0] if ids else ""),
                "model_alias": data.get("service") or "",
                "max_context": data.get("max_context") or MAX_CONTEXT_TOKENS,
                "endpoint": self.url,
                "error": "",
            }
        except Exception as exc:
            self.health_cache = {
                "online": False,
                "degraded": False,
                "latency_ms": None,
                "last_ok": self.health_cache.get("last_ok", 0.0),
                "model": "",
                "endpoint": self.url,
                "error": f"{type(exc).__name__}: {exc}",
            }
        self.health_at = now
        return self.health_cache

    def build_engine(self, settings: dict) -> Engine:
        depth = settings.get("analysis_depth", "standard")
        max_tokens = int(settings.get("max_tokens") or DEPTH_TOKENS.get(depth, 24000))
        max_tokens = max(MIN_OUTPUT_TOKENS, min(MAX_OUTPUT_TOKENS, max_tokens))
        client = LLMClient(
            base_url=self.url,
            max_tokens=max_tokens,
            max_output_tokens=max_tokens,
            context_tokens=min(MAX_CONTEXT_TOKENS, int(settings.get("max_context") or MAX_CONTEXT_TOKENS)),
            use_schema=bool(settings.get("structured_output", True)),
        )
        return Engine(
            client=client,
            use_critic=bool(settings.get("use_critic", True)),
            allow_rewrite=bool(settings.get("allow_rewrite", True)),
            use_logic_anchor=bool(settings.get("logic_anchor", True)),
            use_negative_prompt=bool(settings.get("negative_prompt", True)),
            use_overthinking=bool(settings.get("overthinking_monitor", True)),
            use_section_repair=bool(settings.get("section_repair", True)),
            overthinking_threshold=int(settings.get("overthinking_threshold", 3) or 3),
            overthinking_window=int(settings.get("overthinking_window", 500) or 500),
            cutoff=str(settings.get("knowledge_cutoff", "") or ""),
        )

    def module_status(self) -> dict:
        settings = self.store.settings()
        return {
            "deterministic_editor": "on (Python guardrails; the LLM is never asked for the base directives)",
            "structured_output": "on" if settings.get("structured_output", True) else "off",
            "critic": "on (presentation contract over the final text)" if settings.get("use_critic", True) else "off",
            "knowledge_boundary": "on (3-way triage: SIMPLE, REASONING, FACTUAL)",
            "logic_anchor": "on" if settings.get("logic_anchor", True) else "off",
            "negative_prompt": "on" if settings.get("negative_prompt", True) else "off",
            "overthinking_monitor": "on" if settings.get("overthinking_monitor", True) else "off",
            "section_repair": "on" if settings.get("section_repair", True) else "off",
            "external_apis": "disabled (purely parametric: no retrieval, no live search)",
        }

    def _settings_for(self, overrides: dict | None) -> dict:
        settings = self.store.settings()
        for key, value in (overrides or {}).items():
            if key in settings:
                settings[key] = value
        return settings

    def start(self, prompt: str, overrides: dict | None = None) -> Job:
        settings = self._settings_for(overrides)
        mode = str(settings.get("mode", "auto"))
        case = CaseFile(case_id=new_case_id(prompt), raw_prompt=prompt, title=title_of(prompt), settings=settings)
        job = Job(case_id=case.case_id, store=self.store, manager=self, case=case, settings=settings, mode=mode, prompt=prompt)
        self.store.save_case(case)
        with self.lock:
            self.jobs[case.case_id] = job
        job.start()
        return job

    def continue_case(self, case_id: str, prompt: str, overrides: dict | None = None) -> Job:
        prior = self.store.load_case(case_id)
        if prior is None:
            raise KeyError(case_id)
        settings = self._settings_for(overrides)
        mode = str(settings.get("mode", "auto"))
        case = CaseFile(
            case_id=new_case_id(prompt),
            raw_prompt=prompt,
            title=title_of(prompt),
            settings={**settings, "parent": case_id},
            continuation_count=prior.continuation_count + 1,
            continuation_log=list(prior.continuation_log) + [{"prompt": prompt, "checkpoint": "pending", "mode": ""}],
            user_override_history=list(prior.user_override_history),
        )
        job = Job(case_id=case.case_id, store=self.store, manager=self, case=case, settings=settings, mode=mode, prompt=prompt, prior=prior)
        self.store.save_case(case)
        with self.lock:
            self.jobs[case.case_id] = job
        job.start()
        return job

    def retry(self, case_id: str) -> Job:
        case = self.store.load_case(case_id)
        if case is None:
            raise KeyError(case_id)
        fresh = CaseFile(
            case_id=case_id,
            raw_prompt=case.raw_prompt,
            title=case.title,
            settings=case.settings,
            continuation_count=case.continuation_count,
            continuation_log=case.continuation_log[:-1] if case.continuation_log else [],
            user_override_history=case.user_override_history,
        )
        settings = self._settings_for(case.settings)
        job = Job(case_id=case_id, store=self.store, manager=self, case=fresh, settings=settings, mode=str(settings.get("mode", "auto")), prompt=case.raw_prompt)
        self.store.save_case(fresh)
        with self.lock:
            self.jobs[case_id] = job
        job.start()
        return job

    def get(self, case_id: str) -> Job | None:
        with self.lock:
            return self.jobs.get(case_id)

    def resolve_checkpoint(self, case_id: str, decision: dict) -> bool:
        job = self.get(case_id)
        return bool(job and job.resolve_checkpoint(decision))

    def stop(self, case_id: str) -> bool:
        job = self.get(case_id)
        if not job:
            return False
        job.cancel.set()
        job.checkpoint_event.set()
        return True
