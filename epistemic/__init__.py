"""Epistemic engine: one local LLM (port 8082), purely parametric.

Pipeline: Triage -> Skeptic -> Human checkpoint -> Ontology (map, then distill) -> Dialectical Engine
-> deterministic Editor (Python guardrails) -> Writer + mechanical scrubber -> Presentation Critic.
Termination detection produces warnings, never blocks. No retrieval, no live search, no external APIs.
"""

from .casefile import CaseFile
from .llm import DEFAULT_BASE_URL, LLMClient, LLMUnavailable, StreamAbort, UnparseableOutput
from .pipeline import Checkpoint, Engine

__all__ = [
    "CaseFile",
    "Checkpoint",
    "Engine",
    "LLMClient",
    "LLMUnavailable",
    "StreamAbort",
    "UnparseableOutput",
    "DEFAULT_BASE_URL",
]
