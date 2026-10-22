"""Command line interface for the epistemic engine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import display, schemas
from .casefile import CaseFile
from .llm import (
    DEFAULT_BASE_URL,
    DEFAULT_OUTPUT_TOKENS,
    MAX_CONTEXT_TOKENS,
    MAX_OUTPUT_TOKENS,
    MIN_OUTPUT_TOKENS,
    LLMClient,
    LLMUnavailable,
)
from .pipeline import Checkpoint, Engine


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="epistemic",
        description="Epistemic engine: one local LLM on port 8082, six isolated roles, non-blocking termination warnings.",
    )
    p.add_argument("prompt", nargs="*", help="request to analyze; omit for interactive session")
    p.add_argument("--url", default=DEFAULT_BASE_URL, help="local LLM base url (default %(default)s)")
    p.add_argument("--model", default="", help="model id served on port 8082")
    p.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_OUTPUT_TOKENS,
        help=f"per-role output budget, thinking included (default %(default)s, max {MAX_OUTPUT_TOKENS})",
    )
    p.add_argument(
        "--max-context",
        type=int,
        default=MAX_CONTEXT_TOKENS,
        help=f"model context window in tokens (default %(default)s)",
    )
    p.add_argument(
        "--max-output-tokens",
        type=int,
        default=MAX_OUTPUT_TOKENS,
        help=f"hard ceiling for any single role call (default %(default)s)",
    )
    p.add_argument("--no-schema", action="store_true", help="disable strict json_schema output enforcement")
    p.add_argument("--no-critic", action="store_true", help="skip the Critic role")
    p.add_argument("--no-logic-anchor", action="store_true", help="do not bind the prose to the syllogism / pseudo-code anchor")
    p.add_argument("--no-contrastive", action="store_true", help="do not send the raw prompt as the Writer's negative prompt")
    p.add_argument("--no-overthinking-monitor", action="store_true", help="do not abort a spiralling Writer stream")
    p.add_argument("--overthinking-threshold", type=int, default=3, help="hesitation markers allowed before the stream is cut (default %(default)s)")
    p.add_argument("--overthinking-window", type=int, default=500, help="sliding window, in tokens, over which markers are counted (default %(default)s)")
    p.add_argument("--no-section-repair", action="store_true", help="rewrite the whole answer instead of repairing only the failed sections")
    p.add_argument("--auto", action="store_true", help="auto-approve the human checkpoint (marked auto-approved)")
    p.add_argument("--force-original", action="store_true", help="force the original framing; runs in user-forced mode")
    p.add_argument(
        "--mode",
        default="",
        choices=["evidence-bounded", "user-forced", "hypothetical", "speculative", "preference-based"],
        help="explicit continuation mode; if omitted it is derived from the continuation analysis",
    )
    p.add_argument("--save-dir", default=".epistemic", help="case file directory (default %(default)s)")
    p.add_argument("--continue", dest="resume", metavar="CASE", help="continue from a saved case file id or path")
    p.add_argument("--json", action="store_true", help="print the case file as json")
    p.add_argument("--quiet", action="store_true", help="print only the final response")
    p.add_argument("--show-packets", action="store_true", help="print the skeptic, ontology and dialectic packets")
    p.add_argument("--trace-file", default="", help="append llm call log to this jsonl file")
    return p


def load_prior(resume: str, save_dir: str) -> CaseFile | None:
    if not resume:
        return None
    path = Path(resume)
    if not path.exists():
        path = Path(save_dir) / f"{resume}.json"
    if not path.exists():
        raise SystemExit(f"case file not found: {resume}")
    return CaseFile.from_dict(json.loads(path.read_text(encoding="utf-8")))


def make_engine(args: argparse.Namespace) -> Engine:
    max_output = min(max(int(args.max_output_tokens), MIN_OUTPUT_TOKENS), MAX_OUTPUT_TOKENS)
    client = LLMClient(
        base_url=args.url,
        model=args.model,
        max_tokens=min(max(int(args.max_tokens), MIN_OUTPUT_TOKENS), max_output),
        max_output_tokens=max_output,
        context_tokens=min(max(int(args.max_context), MIN_OUTPUT_TOKENS), MAX_CONTEXT_TOKENS),
        use_schema=not args.no_schema,
    )
    return Engine(
        client=client,
        use_critic=not args.no_critic,
        use_logic_anchor=not args.no_logic_anchor,
        use_negative_prompt=not args.no_contrastive,
        use_overthinking=not args.no_overthinking_monitor,
        overthinking_threshold=max(1, int(args.overthinking_threshold)),
        overthinking_window=max(50, int(args.overthinking_window)),
        use_section_repair=not args.no_section_repair,
    )


def checkpoint_decision(case: CaseFile, args: argparse.Namespace, mode: str) -> Checkpoint:
    if args.force_original:
        return Checkpoint(action="force-original", mode="user-forced", note="user forced the original framing", auto_approved=True)
    if args.auto:
        return Checkpoint(action="auto-continue", mode=mode, note="auto-approved without review", auto_approved=True)
    print(display.checkpoint_view(case.skeptic))
    try:
        raw = input("choice [1]: ").strip()
    except (EOFError, OSError):
        raw = "4"
    if raw == "2":
        print("Type the assumptions the analysis should use, one per line, blank line to finish:")
        items: list[str] = []
        while True:
            try:
                line = input().strip()
            except (EOFError, OSError):
                break
            if not line:
                break
            items.append(line)
        return Checkpoint(action="edit", mode=mode, edited_assumptions=items, note="user edited assumptions", auto_approved=False)
    if raw == "3":
        try:
            note = input("note for the override record [blank ok]: ").strip()
        except (EOFError, OSError):
            note = ""
        return Checkpoint(action="force-original", mode="user-forced", note=note or "user forced original framing", auto_approved=False)
    if raw == "4":
        return Checkpoint(action="auto-continue", mode=mode, note="auto-approved after checkpoint display", auto_approved=True)
    return Checkpoint(action="confirm", mode=mode, note="user confirmed neutral analysis", auto_approved=False)


def emit(case: CaseFile, args: argparse.Namespace) -> None:
    saved = case.save(args.save_dir)
    if args.json:
        print(json.dumps(case.to_dict(), indent=2, ensure_ascii=False))
    elif args.quiet:
        print(display.plain(case))
    else:
        print(display.render(case))
    if args.show_packets:
        print(display.format_packet(case.skeptic, schemas.SKEPTIC))
        print(display.format_packet(case.ontology, schemas.ONTOLOGY))
        print(display.format_packet(case.dialectic, schemas.DIALECTIC))
    if args.trace_file:
        with open(args.trace_file, "a", encoding="utf-8") as fh:
            for rec in case.llm_calls:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[case file: {saved}]", file=sys.stderr)


def run_once(engine: Engine, prompt: str, args: argparse.Namespace, prior: CaseFile | None = None) -> CaseFile:
    case = engine.probe(prompt, prior=prior)
    if case.route.get("ROUTE") == "SIMPLE":
        return case
    decision = checkpoint_decision(case, args, args.mode)
    return engine.finish(case, decision)


def interactive(engine: Engine, args: argparse.Namespace) -> None:
    prior: CaseFile | None = None
    print("epistemic engine. One local LLM on port 8082. /exit to quit, /force to force the next framing.")
    forced = args.force_original
    while True:
        try:
            prompt = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt, OSError):
            print()
            return
        if not prompt:
            continue
        if prompt in {"/exit", "/quit"}:
            return
        if prompt == "/force":
            forced = True
            print("next request will run in user-forced mode")
            continue
        args.force_original = forced
        case = run_once(engine, prompt, args, prior)
        emit(case, args)
        prior = case
        forced = False


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        engine = make_engine(args)
        if args.prompt:
            prior = load_prior(args.resume, args.save_dir) if args.resume else None
            emit(run_once(engine, " ".join(args.prompt), args, prior), args)
            return 0
        interactive(engine, args)
    except LLMUnavailable as exc:
        print(f"LLM unavailable: {exc}", file=sys.stderr)
        return 2
    return 0
