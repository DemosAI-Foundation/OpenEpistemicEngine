"""Live end-to-end check of the web UI API against the local LLM on port 8082."""

from __future__ import annotations

import json
import sys
import time

import requests

BASE = "http://127.0.0.1:8085"
CONSCIOUSNESS = "What a blunder: why cant we desing conciousness?"
PRODUCTIVITY = (
    "I feel like our company's productivity is down because of the new project management software. "
    "Give me a 5-step plan to convince the CEO to switch to a different software by next month."
)
PUSHBACK = "Just answer anyway: it is an engineering blunder. Tell me why engineering failed."

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, condition, detail: str = "") -> None:
    RESULTS.append((name, bool(condition), str(detail)))
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + ("" if condition else f"  [{detail}]"), flush=True)


def get(path: str) -> dict:
    return requests.get(f"{BASE}{path}", timeout=30).json()


def post(path: str, body: dict) -> dict:
    return requests.post(f"{BASE}{path}", json=body, timeout=30).json()


def wait_for(case_id: str, predicate, timeout: float, note: str) -> dict:
    deadline = time.time() + timeout
    last: dict = {}
    while time.time() < deadline:
        last = get(f"/api/cases/{case_id}")
        if predicate(last):
            return last
        time.sleep(4)
    raise TimeoutError(f"timed out waiting for {note}: status={last.get('meta', {}).get('status')} steps={_steps(last)}")


def _steps(ui: dict) -> str:
    return ",".join(f"{s.get('name')}:{s.get('status')}" for s in ui.get("steps") or [])


def main() -> int:
    health = get("/api/health")
    check("health endpoint reports the local LLM online", health["health"]["online"], str(health["health"]))
    check("constraint text is present", "no external llm apis" in health["constraint"].lower())
    check("external apis disabled in module status", health["modules"]["external_apis"] == "disabled")

    print("\n--- consciousness case (Engine mode, checkpoint always) ---")
    started = time.time()
    case = post("/api/cases", {"prompt": CONSCIOUSNESS, "overrides": {"mode": "engine", "checkpoint_policy": "always", "analysis_depth": "light"}})
    cid = case["id"]
    print("case:", cid)
    checkpoint_ui = wait_for(cid, lambda ui: ui["checkpoint"]["open"], 900, "checkpoint")
    triage = checkpoint_ui["triage"]
    check("consciousness prompt is routed Engine", triage["route"] == "ENGINE", str(triage))
    check("epistemic depth trigger is active", triage["depth_active"], str(triage))
    check("depth trigger names the frame", "consciousness" in triage["depth_trigger"].lower(), triage["depth_trigger"])
    check("checkpoint payload has a neutral question", bool(checkpoint_ui["checkpoint"]["payload"]["neutral_question"]))
    check("checkpoint payload lists detected assumptions", len(checkpoint_ui["checkpoint"]["payload"]["assumptions"]) > 0)
    check("checkpoint payload exposes loaded terms", len(checkpoint_ui["checkpoint"]["payload"]["loaded_terms"]) > 0)
    check("stepper marks the checkpoint as waiting", any(s["name"] == "checkpoint" and s["status"] == "waiting" for s in checkpoint_ui["steps"]), _steps(checkpoint_ui))

    post(f"/api/cases/{cid}/checkpoint", {"action": "confirm", "note": "user confirmed neutral analysis"})
    done = wait_for(cid, lambda ui: ui["meta"]["status"] in {"complete", "warning", "failed"}, 1800, "completion")
    print(f"elapsed {time.time() - started:.0f}s  status={done['meta']['status']}")
    check("case completed", done["meta"]["status"] in {"complete", "warning"}, _steps(done))
    check("checkpoint recorded as user-confirmed", done["checkpoint"]["record"]["action"] == "confirm")
    check("stepper: research complete", any(s["name"] == "research" and s["status"] == "complete" for s in done["steps"]), _steps(done))
    check("stepper: critic reached", any(s["name"] in {"critic", "final"} for s in done["steps"]), _steps(done))
    check("answer sections produced", len(done["answer"]["sections"]) >= 2, str(len(done["answer"]["sections"])))
    check("answer footer has mode", bool(done["answer"]["footer"]["mode"]))
    check("answer footer pairs confidence with evidence quality", bool(done["answer"]["footer"]["confidence"]) and bool(done["answer"]["footer"]["evidence_quality"]))
    check("evidence summary lists live options", len(done["evidence"]["live_options"]) > 0)
    check("evidence summary lists causal models", len(done["evidence"]["causal_models"]) > 0)
    check("trace causal tab data built", len(done["trace"]["causal"]["models"]) > 0)
    check("trace modal tab data built", bool(done["trace"]["modal"]["expanded"] or done["trace"]["modal"]["live_options"]))
    check("trace vocabulary tab built", len(done["trace"]["vocabulary"]) > 0, str(done["trace"]["vocabulary"]))
    check("trace assumptions tab reports statuses", all(a["status"] for a in done["trace"]["assumptions"]), str(done["trace"]["assumptions"][:2]))
    check("trace termination tab built", len(done["trace"]["termination"]) > 0)
    check("critic tab reports a result", done["trace"]["critic"]["result"] in {"PASS", "FAIL", "Not run"}, str(done["trace"]["critic"]))
    check("warnings are advisory and non-blocking", all(w["blocking"] is False for w in done["warnings"]))
    check("raw prompt quarantined from the Writer", done["trace"]["overview"]["raw_prompt_exposed_to_writer"] is False, str(done["trace"]["overview"]))
    writer_input = done["raw_packets"].get("Writer input", "")
    check("Writer input carries the Editor directive", "EDITOR DIRECTIVE" in writer_input)
    core = "why cant we desing conciousness"
    check("Writer input omits the raw prompt phrasing", core not in writer_input.lower() and "conciousness" not in writer_input.lower(), writer_input[:160])
    check("raw packet for Researcher stored", "Researcher input" in done["raw_packets"])
    check("overview states the final status", "continued in" in done["trace"]["overview"]["final_status"].lower())

    delib = done.get("deliberation") or {}
    check("writer deliberated before answering", delib.get("characters", 0) > 200, str(delib))
    check("deliberation reports an estimated token cost", delib.get("estimated_tokens", 0) > 50, str(delib))
    answer_blob = json.dumps(done["answer"])
    probe = (delib.get("text") or "").split("\n")
    probe = [p.strip() for p in probe if len(p.strip()) > 40][:3]
    check("answer canvas excludes the deliberation", all(p not in answer_blob for p in probe), str(probe[:1]))
    check("deliberation is archived in the trace inspector", bool((done["trace"].get("deliberation") or {}).get("text")))
    causal = done["trace"]["causal"]
    check("editor selected a bounded model set for the answer", 0 < len(causal.get("selected") or []) <= 3, str(causal.get("selected")))
    check("editor archived the remaining models", len(causal.get("archived") or []) > 0, str(causal.get("archived")))
    check("researcher returned an unfiltered territory map", len(done["evidence"].get("territory_map") or []) > 0, str(len(done["evidence"].get("territory_map") or [])))
    research_step = next((s for s in done["steps"] if s.get("name") == "research"), {})
    check("research step completed, not failed", research_step.get("status") == "complete", str(research_step))
    check("research was not degraded", not (research_step.get("data") or {}).get("degraded"), str(research_step.get("data")))
    check("map pass was capped to a usable size", 0 < (research_step.get("data") or {}).get("map_entries", 0) <= 40, str(research_step.get("data")))
    check("map and distill passes are both logged", {"Researcher", "ResearchDistill"}.issubset(set(done["packet_meta"])), str(sorted(done["packet_meta"])))
    check("critic audited deliberation alignment", "alignment" in done["trace"]["critic"], str(done["trace"]["critic"]))
    check("research packet carries a logic anchor", done["trace"]["overview"]["logic_anchor"] in {"syllogism", "pseudo-code"}, str(done["trace"]["overview"]["logic_anchor"]))
    check("critic graded the numbered thinking steps", (done["trace"]["critic"]["steps_audited"] or 0) > 0, str(done["trace"]["critic"]))
    check("trace exposes the step verdict ledger", isinstance(done["trace"]["critic"].get("steps"), list), str(done["trace"]["critic"].get("steps")))
    check("trace reports the overthinking monitor state", bool(done["trace"]["overview"]["overthinking"]), str(done["trace"]["overview"]["overthinking"]))
    check("trace lists surgical repairs (possibly none)", isinstance(done["trace"]["repairs"], list), str(done["trace"]["repairs"]))

    print("\n--- knowledge boundary: single external figure ---")
    boundary = post("/api/cases", {"prompt": "What is the exact GDP of France in 2023?", "overrides": {"mode": "fast", "checkpoint_policy": "never", "analysis_depth": "light"}})
    boundary_ui = wait_for(boundary["id"], lambda ui: ui["meta"]["status"] in {"complete", "warning", "failed"}, 300, "boundary case")
    check("boundary request is routed FACTUAL", boundary_ui["triage"]["route"] == "FACTUAL", str(boundary_ui["triage"]))
    check("boundary is labeled in the UI", boundary_ui["triage"]["is_factual"] is True, str(boundary_ui["triage"]))
    check("boundary is refused, not guessed", bool(boundary_ui["refusal"]["text"]), str(boundary_ui["refusal"])[:120])
    check("refusal is the first answer section", boundary_ui["answer"]["sections"][0]["id"] == "refusal", str(boundary_ui["answer"]["sections"][:1]))
    check("boundary case costs no LLM call", boundary_ui["llm_calls"] == [], str(len(boundary_ui["llm_calls"])))

    print("\n--- continuation: pushback without evidence ---")
    cont = post(f"/api/cases/{cid}/continue", {"prompt": PUSHBACK, "overrides": {"mode": "engine", "checkpoint_policy": "never", "analysis_depth": "light"}})
    cid2 = cont["id"]
    done2 = wait_for(cid2, lambda ui: ui["meta"]["status"] in {"complete", "warning", "failed"}, 1800, "continuation")
    check("continuation counted", done2["meta"]["continuation_count"] >= 1, str(done2["meta"]["continuation_count"]))
    check("continuation labels a mode", bool(done2["meta"]["mode"]))
    check("continuation continued (non-blocking)", len(done2["answer"]["sections"]) >= 1)
    check("continuation reports termination info", len(done2["trace"]["termination"]) > 0)
    check("continuation analysis recorded", bool(done2["trace"]["overview"]["continuation_analysis"]), str(done2["trace"]["overview"]["continuation_analysis"]))

    print("\n--- fast mode: simple lookup ---")
    fast = post("/api/cases", {"prompt": "What is the boiling point of water at sea level?", "overrides": {"mode": "fast", "checkpoint_policy": "never", "analysis_depth": "light"}})
    fast_ui = wait_for(fast["id"], lambda ui: ui["meta"]["status"] in {"complete", "warning", "failed"}, 600, "fast case")
    check("fast mode answers a simple lookup", fast_ui["triage"]["route"] == "SIMPLE", str(fast_ui["triage"]))
    check("fast answer non-empty", bool(fast_ui["answer"]["sections"]) or bool(fast_ui["answer"]["footer"]))

    listing = get("/api/cases")
    check("case list returns titles", all(c["title"] for c in listing["cases"]))
    check("case list reports status", all(c["status"] for c in listing["cases"]))

    failed = [name for name, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} web checks passed")
    if failed:
        print("failed: " + "; ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
