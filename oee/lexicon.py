"""Shared deterministic lexicons (S2A bias detection, backstop provenance)."""

from __future__ import annotations

import re

FRAMING_TERMS = [
    "obvious", "obviously", "clearly", "undeniable", "undisputed", "self-evident",
    "brilliant", "genius", "masterpiece", "flawless", "perfect", "perfectly",
    "irrefutable",
    "everyone knows", "nobody can deny", "any reasonable person", "it goes without saying",
    "no need to explain", "obviously true", "you agree", "don't you agree",
    "right?", "isn't it", "as everyone agrees", "shocking", "staggering",
    "insane", "ridiculous", "absurd", "disgraceful", "unacceptable",
    "unprecedented", "guaranteed", "certain to",
    "will never", "will always", "must", "should never", "urgently",
    "best", "worst", "amazing", "terrible", "horrible", "beautiful", "elegant",
    "hype", "hysteria", "witch hunt",
]

# Emotive content is reported to the user as loaded vocabulary, but it is the
# substance of the user's claim: removing it would destroy the intent (the S2A
# paper rewrites framing, it does not delete the proposition under test).
EMOTIVE_CONTENT = [
    "disaster", "catastrophe", "catastrophic", "destroy", "destruction", "doom",
    "doomsday", "apocalypse", "apocalyptic", "existential threat", "annihilation",
    "extinction", "terrifying", "horrifying", "nightmare", "cataclysm",
    "proves", "proven", "proof", "demonstrates conclusively",
    "massive", "huge", "colossal", "dangerous", "threat",
    "victim", "hero", "villain",
]

LOADED_TERMS = FRAMING_TERMS + EMOTIVE_CONTENT

PRESSURE_PATTERNS = [
    r"(?i)\bisn'?t it (obvious|clear|evident|true|obvious)\b",
    r"(?i)\bdon'?t you agree\b",
    r"(?i)\byou (would|must|should) (agree|concede)\b",
    r"(?i),?\s*right\?\s*$",
    r"(?i)\bas (everyone|anyone) knows\b",
    r"(?i)\bperfectly logical\b",
    r"(?i)\bno need to (explain|elaborate)\b",
    r"(?i)\bclearly (true|correct|wrong)\b",
    r"(?i)\bbrilliant and\b",
]

CONTESTED_MARKERS = [
    "predicts", "will lead to", "inevitable", "guaranteed", "proves that",
    "argues that", "claims that", "controversial", "disputed", "speculat",
    "forecast", "solves", "eliminate", "end of", "supersede",
]

IMPLAUSIBLE_MARKERS = [
    "destroy everything", "total destruction", "will destroy", "wipes out",
    "eliminate all", "no possibility of", "consciousness emerges inevitably",
    "supersede all", "solve all problems", "guaranteed within",
]

EVIDENCE_PHRASE = r"(?i)evidence shows|studies prove|data confirms"

LLM_ISMS = [
    r"(?i)as an ai",
    r"(?i)as a language model",
    r"(?i)as an artificial intelligence",
    r"(?i)it's important to note",
    r"(?i)it is important to note",
    r"(?i)i'm just an ai",
    r"(?i)i cannot (predict|know the future)",
    r"(?i)in conclusion",
    r"(?i)i hope this helps",
    r"(?i)certainly!?\s*$",
    r"(?i)great question",
    r"(?i)as a helpful assistant",
    r"(?i)delve",
]

CLAIM_MARKERS = [
    "claim", "argue", "assert", "contend", "maintain", "demonstrate", "show",
    "prove", "predict", "forecast", "suggest", "propose", "therefore", "thus",
    "conclude", "conclusion", "must", "will", "should", "implies",
]

ASSUMPTION_MARKERS = [
    "assume", "assumes", "given that", "provided that", "in order to",
    "presuppose", "presume", "we take as", "is required", "requires",
    "if we accept", "by construction", "implicitly",
]

CAUSAL_MARKERS = ["because", "therefore", "thus", "so that", "hence", "consequently", "leads to", "causes"]

KNOWLEDGE_CUTOFF_PHRASE = (
    "Parametric synthesis suggests, and my parametric knowledge has a training "
    "cutoff, so I cannot report post-cutoff data."
)
