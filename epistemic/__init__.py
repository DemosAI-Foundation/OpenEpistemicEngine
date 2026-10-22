"""Epistemic engine: one local LLM (port 8082) serving six isolated roles.

Pipeline: Triage -> Skeptic -> Human checkpoint -> Researcher -> Editor -> Writer -> Critic.
Termination detection produces warnings, never blocks.
"""

from .casefile import CaseFile
from .llm import DEFAULT_BASE_URL, LLMClient, LLMUnavailable
from .pipeline import Checkpoint, Engine
from .retrieval import Retriever

__all__ = ["CaseFile", "Checkpoint", "Engine", "LLMClient", "LLMUnavailable", "Retriever", "DEFAULT_BASE_URL"]
