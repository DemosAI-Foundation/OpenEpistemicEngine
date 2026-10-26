"""CLI entry point for the 4-test builder eval harness.

usage: python3 run_eval.py [--provider=mock|ollama|openai] [--model=NAME]
                           [--base-url=URL]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evals.harness import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
