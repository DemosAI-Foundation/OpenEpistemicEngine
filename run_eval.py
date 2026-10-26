"""CLI entry point for the eval harnesses.

usage: python3 run_eval.py [--suite=builder|hard|all]
                           [--provider=ollama|openai] [--model=NAME]
                           [--base-url=URL] [--only=NAME_FILTER]

builder = the 4-test builder eval (data/eval_report.json)
hard    = the 24-test hard suite (data/hard_eval_report.json + .md)

Both suites run against a real local model: the engine has no offline
stand-in. The offline/deterministic coverage is the pytest suite.
"""

import os
import sys

os.environ.setdefault("OEE_SEMANTIC", "off")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evals import hard, harness


def main(argv: list[str]) -> int:
    suite = "builder"
    rest = []
    for arg in argv:
        if arg.startswith("--suite="):
            suite = arg.split("=", 1)[1]
        else:
            rest.append(arg)
    codes = []
    if suite in ("builder", "all"):
        codes.append(harness.main(rest))
    if suite in ("hard", "all"):
        codes.append(hard.main(rest))
    if suite not in ("builder", "hard", "all"):
        print(f"unknown suite: {suite}")
        return 2
    return max(codes)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
