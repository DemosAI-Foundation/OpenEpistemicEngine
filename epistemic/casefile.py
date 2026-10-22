"""Tool working state. Not LLM chat memory."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class CaseFile:
    case_id: str = ""
    title: str = ""
    created_at: float = field(default_factory=time.time)
    raw_prompt: str = ""
    route: dict = field(default_factory=dict)
    skeptic: dict = field(default_factory=dict)
    checkpoint: dict = field(default_factory=dict)
    ontology: dict = field(default_factory=dict)
    dialectic: dict = field(default_factory=dict)
    editor: dict = field(default_factory=dict)
    guardrails: dict = field(default_factory=dict)
    termination_warnings: list[str] = field(default_factory=list)
    writer_draft: str = ""
    writer_rewrite: str = ""
    deliberation: str = ""
    deliberation_rewrite: str = ""
    critic: dict = field(default_factory=dict)
    critic_rounds: list[dict] = field(default_factory=list)
    repairs: list[dict] = field(default_factory=list)
    overthinking: dict = field(default_factory=dict)
    refusal: dict = field(default_factory=dict)
    final_response: str = ""
    continuation_count: int = 0
    continuation: dict = field(default_factory=dict)
    continuation_log: list[dict] = field(default_factory=list)
    user_override_history: list[dict] = field(default_factory=list)
    mode: str = "evidence-bounded"
    leak: dict = field(default_factory=dict)
    checks: list[dict] = field(default_factory=list)
    scrub: dict = field(default_factory=dict)
    warning_banner: str = ""
    cutoff: str = ""
    steps: list[dict] = field(default_factory=list)
    packets_raw: dict = field(default_factory=dict)
    packets_meta: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)
    error: str = ""
    llm_calls: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CaseFile":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, directory: str | Path = ".epistemic") -> Path:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{self.case_id}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path


def title_of(prompt: str, max_words: int = 7) -> str:
    words = [w.strip(".,;:?!()\"'") for w in prompt.split()]
    words = [w for w in words if w][:max_words]
    title = " ".join(words).strip()
    if len(prompt.split()) > max_words:
        title += "..."
    return title or "Untitled case"


def new_case_id(prompt: str) -> str:
    slug = "".join(c if c.isalnum() else "-" for c in prompt.lower()[:32])
    slug = "-".join(p for p in slug.split("-") if p)
    return f"{int(time.time())}-{slug[:40] or 'case'}"
