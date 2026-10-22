"""Overthinking monitor: a stream watcher that aborts a generation that is talking itself into a wrong answer.

The watcher counts hesitation markers and polarity reversals in the streamed text. Past the threshold it raises
StreamAbort, which the LLM client catches: the stream is closed and everything streamed so far is kept.
"""

from __future__ import annotations

import math
import re

from .llm import StreamAbort

HESITATION_MARKERS = [
    "wait",
    "actually",
    "on second thought",
    "on further reflection",
    "on reflection",
    "however",
    "that said",
    "having said that",
    "conversely",
    "i was wrong",
    "i made a mistake",
    "i misspoke",
    "correction:",
    "let me reconsider",
    "reconsider",
    "no,",
    "in fact",
    "to be clear",
    "i need to revisit",
    "i previously said",
    "on the other hand",
    "i take that back",
]

STANCE_POSITIVE = [
    "is correct",
    "is right",
    "is supported",
    "is justified",
    "is sound",
    "is established",
    "will work",
    "is likely",
    "the evidence supports",
    "this holds",
    "is the best model",
    "conclude that",
]

STANCE_NEGATIVE = [
    "is wrong",
    "is not correct",
    "is not supported",
    "is unjustified",
    "is unsound",
    "is not established",
    "will not work",
    "is unlikely",
    "the evidence does not support",
    "this fails",
    "cannot conclude",
    "i retract",
]


def _phrase_re(markers: list[str]) -> re.Pattern[str]:
    parts = []
    for marker in markers:
        needle = re.sub(r"[^a-z0-9 ]+", " ", marker.lower())
        needle = " ".join(needle.split())
        if not needle:
            continue
        parts.append(r"(?<!\w)" + re.escape(needle) + r"(?!\w)")
    return re.compile("|".join(parts))


HESITATION_RE = _phrase_re(HESITATION_MARKERS)
POSITIVE_RE = _phrase_re(STANCE_POSITIVE)
NEGATIVE_RE = _phrase_re(STANCE_NEGATIVE)

DEFAULT_THRESHOLD = 3
DEFAULT_WINDOW_TOKENS = 500
DEFAULT_MIN_TOKENS = 200
DEFAULT_MAX_FLIPS = 2
FLIP_WINDOW_TOKENS = 120
TAIL_CHARS = 64


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower())


class OverthinkingMonitor:
    """Counts hesitation markers and stance flips over a token window and aborts the stream past the threshold."""

    def __init__(
        self,
        enabled: bool = True,
        threshold: int = DEFAULT_THRESHOLD,
        window_tokens: int = DEFAULT_WINDOW_TOKENS,
        min_tokens: int = DEFAULT_MIN_TOKENS,
        max_flips: int = DEFAULT_MAX_FLIPS,
    ) -> None:
        self.enabled = enabled
        self.threshold = threshold
        self.window_tokens = window_tokens
        self.min_tokens = min_tokens
        self.max_flips = max_flips
        self.reset()

    def reset(self) -> None:
        self.chars = 0
        self.markers = 0
        self.flips = 0
        self.aborted = False
        self.abort_reason = ""
        self._pending = ""
        self._tail = ""
        self._tail_hits = 0
        self._flip_pending = ""
        self._sign = 0
        self._full = ""

    def feed(self, text: str) -> None:
        """Consume one stream delta. Raises StreamAbort when the overthinking budget is spent."""
        if not text:
            return
        self.chars += len(text)
        if not self.enabled:
            return
        chunk = normalize_text(text)
        self._pending += chunk
        self._flip_pending += chunk
        self._full += chunk
        counted = self._scan_markers()
        self._scan_flips()
        if counted and self.tokens >= self.min_tokens and self.markers > self.allowed:
            self._raise(f"{self.markers} hesitation markers in ~{self.tokens} tokens (limit {self.allowed})")
        if self.flips > self.max_flips:
            self._raise(f"{self.flips} stance reversals while writing (limit {self.max_flips})")

    def _raise(self, reason: str) -> None:
        self.aborted = True
        self.abort_reason = reason
        markers = [hit.group(0) for hit in HESITATION_RE.finditer(self._full)]
        raise StreamAbort(f"overthinking monitor: {reason}", markers=list(dict.fromkeys(markers)))

    @property
    def tokens(self) -> int:
        return self.chars // 4

    @property
    def allowed(self) -> int:
        windows = max(1, math.ceil(self.tokens / max(1, self.window_tokens)))
        return self.threshold * windows

    def _scan_markers(self) -> bool:
        """Count new marker hits. A marker split across two deltas is counted once, at completion."""
        if not self._pending:
            return False
        window = self._tail + self._pending
        self._pending = ""
        before = self._tail_hits
        after = len(HESITATION_RE.findall(window))
        self._tail = window[-TAIL_CHARS:]
        self._tail_hits = len(HESITATION_RE.findall(self._tail))
        new = after - before
        self.markers = max(0, self.markers + new)
        return new > 0

    def _scan_flips(self) -> None:
        """Track the polarity of the running argument: a flip is a reversal of the dominant stance in one window."""
        room = FLIP_WINDOW_TOKENS * 4
        while len(self._flip_pending) >= room:
            body, self._flip_pending = self._flip_pending[:room], self._flip_pending[room:]
            score = len(POSITIVE_RE.findall(body)) - len(NEGATIVE_RE.findall(body))
            sign = 1 if score > 0 else -1 if score < 0 else 0
            if sign and self._sign and sign != self._sign:
                self.flips += 1
            if sign:
                self._sign = sign

    def report(self) -> dict:
        markers = [hit.group(0) for hit in HESITATION_RE.finditer(self._full)]
        return {
            "enabled": self.enabled,
            "aborted": self.aborted,
            "reason": self.abort_reason,
            "characters": self.chars,
            "tokens": self.tokens,
            "hesitation_markers": len(markers),
            "markers_found": list(dict.fromkeys(markers)),
            "limit": self.allowed,
            "threshold": self.threshold,
            "window_tokens": self.window_tokens,
            "stance_flips": self.flips,
            "flip_limit": self.max_flips,
        }


def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]


def join_paragraphs(items: list[str]) -> str:
    return "\n\n".join(p.strip() for p in items if p.strip())


def split_banner(text: str) -> tuple[str, str]:
    """Separate a leading termination-warning paragraph from the body it precedes."""
    items = paragraphs(text)
    if items and items[0].lower().startswith("termination warning"):
        return items[0], join_paragraphs(items[1:])
    return "", join_paragraphs(items)


def truncate_at_last_stable(text: str) -> tuple[str, int]:
    """Cut the stream at the last stable paragraph before the hesitation spiral."""
    items = paragraphs(text)
    if len(items) <= 1:
        return text, 0
    marker_at = None
    for idx, para in enumerate(items):
        if HESITATION_RE.search(normalize_text(para)):
            marker_at = idx
            break
    if marker_at == 0:
        return text, 0
    kept = items[:marker_at] if marker_at is not None else items[:-1]
    if not kept:
        return text, 0
    return join_paragraphs(kept), len(items) - len(kept)
