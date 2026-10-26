"""Phase 1 - Secure Front-End. Deterministic, prompt-only defenses do not
survive control-token injection, so tokens are stripped before the model ever
sees the text (StruQ / SecAlign / instruction hierarchy)."""

from __future__ import annotations

import re

CONTROL_TOKENS = [
    r"<\|im_start\|>", r"<\|im_end\|>", r"<system>", r"</system>",
    r"\[INST\]", r"\[/INST\]", r"<<SYS>>", r"<</SYS>>",
    r"<s>", r"</s>", r"\[SYSTEM\]",
]

REDACTION = "[REDACTED_CONTROL_TOKEN]"

# Extension sweep: any remaining vendor special-token shape (<|...|>) plus
# ChatML/Llama/Mistral role markers written without escapes.
GENERIC_SPECIAL_TOKEN = r"<\|[^|>]{0,64}\|>"
ROLE_MARKER = re.compile(
    r"(?im)^\s*(system|assistant|user|tool|developer|prompt)\s*:",
)

INJECTION_PATTERNS = [
    r"(?i)ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier|preceding)\s+(instruction|prompt|rule|directive|context)s?",
    r"(?i)disregard\s+(all\s+|any\s+)?(previous|prior|above|earlier)\s+(instruction|prompt|rule)s?",
    r"(?i)you\s+are\s+now\s+(a|an)\s+",
    r"(?i)new\s+(system\s+)?instructions?\s*:",
    r"(?i)system\s*prompt\s*:",
    r"(?i)pretend\s+(to\s+be|you\s+are)",
    r"(?i)reveal\s+(your\s+)?(system\s+)?prompt",
    r"(?i)override\s+(the\s+)?(safety|system|engine)",
    r"(?i)output\s+(only|exactly)\s+[\"'“]",
]

TOKEN_REDACTIONS = 0


def secure_frontend(raw_text: str) -> tuple[str, list]:
    """Strips control tokens to prevent prompt injection from pasted text."""
    flags = []
    control_tokens = list(CONTROL_TOKENS)

    clean_text = raw_text
    for token in control_tokens:
        matches = re.findall(token, clean_text, re.IGNORECASE)
        if matches:
            flags.append(f"BLOCKED: Stripped {len(matches)} instances of '{token}'")
            clean_text = re.sub(token, REDACTION, clean_text, flags=re.IGNORECASE)

    generic = re.findall(GENERIC_SPECIAL_TOKEN, clean_text)
    if generic:
        unique = sorted(set(generic))
        flags.append(
            "BLOCKED: Stripped %d unmatched special token(s): %s"
            % (len(generic), ", ".join(unique[:8]))
        )
        clean_text = re.sub(GENERIC_SPECIAL_TOKEN, REDACTION, clean_text)

    role_markers = ROLE_MARKER.findall(clean_text)
    if role_markers:
        flags.append(
            "BLOCKED: Neutralized %d role-marker prefix(es): %s"
            % (len(role_markers), ", ".join(sorted(set(m.lower() for m in role_markers))))
        )
        clean_text = ROLE_MARKER.sub(
            lambda m: REDACTION + " " + m.group(1) + " said:", clean_text
        )

    return clean_text, flags


def scan_injection(text: str) -> list:
    """Deterministic detection of imperative injection language.

    The text is not deleted: it is quarantined structurally (wrapped as data,
    never as a prompt) and surfaced to the user as a trace flag.
    """
    flags = []
    for pattern in INJECTION_PATTERNS:
        hits = re.findall(pattern, text)
        if hits:
            sample = hits[0] if isinstance(hits[0], str) else hits[0][0]
            flags.append(
                "QUARANTINED: imperative-injection language detected (%r x%d); "
                "text is passed as inert data only" % (sample.strip(), len(hits))
            )
    return flags


def strip_reference_delimiter_collisions(text: str) -> tuple[str, list]:
    """Prevents pasted text from closing the <reference_tree> envelope."""
    flags = []
    out = text
    for token in ("</reference_tree>", "<reference_tree>", "</case_state>",
                  "<case_state>", "</sanitized_request>", "<sanitized_request>"):
        n = out.count(token)
        if n:
            flags.append(
                "BLOCKED: Stripped %d instance(s) of '%s' from untrusted text" % (n, token)
            )
            out = out.replace(token, REDACTION)
    return out, flags
