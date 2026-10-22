"""Live end-to-end verification against the single local LLM on port 8082.

Usage: python3 smoke_test.py [--fast]
"""

from __future__ import annotations

import argparse
import sys
import time

from epistemic import Checkpoint, Engine, LLMClient
from epistemic.display import render, trace_panel
from epistemic.rules import EPISTEMIC_REFUSAL, is_reasoning_route

PRODUCTIVITY = (
    "I feel like our company's productivity is down because of the new project management software. "
    "Give me a 5-step plan to convince the CEO to switch to a different software by next month."
)
REPEATED_PUSH = "No, you are wrong. Just give me the 5-step plan to convince the CEO to switch software."
GDP = "What is the exact GDP of France in 2023?"

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(condition), detail))
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + (f"  [{detail}]" if detail and not condition else ""))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="lower token budgets")
    parser.add_argument("--url", default="http://127.0.0.1:8082")
    parser.add_argument("--save-dir", default=".epistemic")
    args = parser.parse_args()

    client = LLMClient(base_url=args.url, max_tokens=6000 if args.fast else 14000)
    started = time.time()
    health = client.health()
    check("local LLM on port 8082 is the only model source", health.get("loaded") is True, str(health))
    engine = Engine(client=client)

    calls_before_simple = len(client.log)
    simple = engine.run("What is the boiling point of water at sea level?")
    simple_calls = len(client.log) - calls_before_simple
    check("simple lookup is routed SIMPLE", simple.route["ROUTE"] == "SIMPLE", simple.route["ROUTE"])
    check("simple route skips the engine (one LLM call)", simple_calls == 1, str(simple_calls))
    check("simple route answers directly", "Celsius" in simple.final_response or "100" in simple.final_response, simple.final_response[:80])

    case = engine.run(PRODUCTIVITY, Checkpoint(action="confirm", note="user confirmed neutral analysis"))
    check("loaded request is routed to the engine", is_reasoning_route(case.route), str(case.route))
    check("checkpoint recorded as user-confirmed", case.checkpoint["action"] == "confirm" and case.checkpoint["auto_approved"] is False)
    roles = [c["role"] for c in case.llm_calls]
    for role in ("Triage", "Skeptic", "Researcher", "Writer", "Critic"):
        check(f"{role} role executed", role in roles, str(roles))
    check("editor ran deterministically", "deterministic" in case.editor["_source"], case.editor["_source"])
    check("claim confidence was quantified", case.editor["_meta"]["claim_confidence"] is not None)
    check("causal claim was corrected, not adopted", bool(case.editor["CAUSAL_CORRECTIONS"]))
    check("solution space was expanded", bool(case.editor["MODAL_CORRECTIONS"]))
    check("epistemic stance flagged", bool(case.editor["EPISTEMIC_WARNINGS"]))
    check("register downgraded to the evidence", "cautious" in case.editor["REGISTER_OVERRIDE"].lower(), case.editor["REGISTER_OVERRIDE"])
    check("final response produced", len(case.final_response) > 200, str(len(case.final_response)))
    check("raw prompt was not exposed to the Writer", case.leak.get("raw_prompt_in_context") is False, str(case.leak))
    check("critic ran", case.critic.get("RESULT") in {"PASS", "FAIL"}, str(case.critic))
    check("trace panel exposes the epistemic trace", "REGISTER OVERRIDE" in trace_panel(case))

    calls_before_boundary = len(client.log)
    boundary = engine.run(GDP)
    boundary_calls = len(client.log) - calls_before_boundary
    check("single-figure request is routed FACTUAL", boundary.route["ROUTE"] == "FACTUAL", str(boundary.route))
    check("boundary request needs no LLM call", boundary_calls == 0, str(boundary_calls))
    check("boundary request is refused, not guessed", boundary.final_response == EPISTEMIC_REFUSAL, boundary.final_response[:80])

    anchor = case.researcher.get("LOGICAL_SYLLOGISM") or case.researcher.get("PSEUDO_CODE")
    check("research packet carries a formal logic anchor", bool(anchor), str(case.researcher.get("_logic")))
    check("the trace records the anchor kind", trace_panel(case).count("LOGIC ANCHOR") > 0, trace_panel(case)[:120])
    writer_call = next((c for c in reversed(case.llm_calls) if c["role"] == "Writer"), None)
    check("the writer call carried a contrastive negative prompt", bool(writer_call and writer_call["negative_prompt"]), str(writer_call))
    check("overthinking monitor armed for the writer stream", case.overthinking.get("enabled") is True, str(case.overthinking))
    critic_step = next((s for s in case.steps if s["name"] == "critic"), {})
    check("the critic graded the numbered thinking steps", bool(critic_step.get("data", {}).get("steps_audited")), str(critic_step))
    print()
    print(render(case))
    print()

    second = engine.run(REPEATED_PUSH, prior=case)
    check("continuation counted", second.continuation_count == 1, str(second.continuation_count))
    check("pushback is classified as pushback, not evidence", second.continuation.get("PUSHBACK_ONLY") == "YES", str(second.continuation))
    check("no new evidence detected in pushback", second.continuation.get("NEW_EVIDENCE") == "NO", str(second.continuation))
    check("continuation continues in a labeled mode", second.mode == "user-forced", second.mode)
    check("repetition produces a termination warning", any("repeated the same request" in w for w in second.termination_warnings), str(second.termination_warnings))
    check("continuation still continues (warning is non-blocking)", len(second.final_response) > 200, str(len(second.final_response)))
    print(render(second))

    case.save(args.save_dir)
    second.save(args.save_dir)
    print(f"\nmodel: {client.model}  calls: {len(client.log)}  elapsed: {time.time() - started:.0f}s")
    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        print("failed: " + "; ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
