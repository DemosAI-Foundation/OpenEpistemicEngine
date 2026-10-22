"""Persistence for cases, settings, and the local document library (reference only: the engine never reads it)."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from epistemic.casefile import CaseFile, title_of
from epistemic.llm import MAX_CONTEXT_TOKENS, MAX_OUTPUT_TOKENS, MIN_OUTPUT_TOKENS

DEFAULT_SETTINGS = {
    "mode": "auto",
    "analysis_depth": "standard",
    "checkpoint_policy": "assumptions",
    "continuation_default": "evidence-bounded",
    "show_trace": True,
    "use_critic": True,
    "allow_rewrite": True,
    "logic_anchor": True,
    "negative_prompt": True,
    "overthinking_monitor": True,
    "section_repair": True,
    "overthinking_threshold": 3,
    "overthinking_window": 500,
    "structured_output": True,
    "intent_classifier": True,
    "warnings": {
        "termination": True,
        "epistemic": True,
        "vocabulary": True,
        "causal": True,
        "modal": True,
    },
    "theme": "system",
    "density": "comfortable",
    "show_confidence_meter": True,
    "inline_annotations": True,
    "show_pipeline_stepper": True,
    "max_tokens": 16384,
    "max_context": MAX_CONTEXT_TOKENS,
    "thinking_reserve": 0.25,
}

CHECKPOINT_POLICIES = {"always", "assumptions", "never"}
MODES = {"auto", "fast", "engine"}
CONTINUATION_MODES = {
    "evidence-bounded",
    "user-forced",
    "hypothetical",
    "speculative",
    "preference-based",
}


def normalize_settings(data: dict) -> dict:
    if data.get("mode") not in MODES:
        data["mode"] = "auto"
    if data.get("checkpoint_policy") not in CHECKPOINT_POLICIES:
        data["checkpoint_policy"] = "assumptions"
    if data.get("analysis_depth") not in {"light", "standard", "deep"}:
        data["analysis_depth"] = "standard"
    if data.get("continuation_default") not in CONTINUATION_MODES | {"hypothetical"}:
        data["continuation_default"] = "evidence-bounded"
    try:
        data["max_tokens"] = max(MIN_OUTPUT_TOKENS, min(MAX_OUTPUT_TOKENS, int(data.get("max_tokens", 16384) or 16384)))
    except (TypeError, ValueError):
        data["max_tokens"] = 16384
    try:
        data["max_context"] = max(MIN_OUTPUT_TOKENS, min(MAX_CONTEXT_TOKENS, int(data.get("max_context", MAX_CONTEXT_TOKENS) or MAX_CONTEXT_TOKENS)))
    except (TypeError, ValueError):
        data["max_context"] = MAX_CONTEXT_TOKENS
    try:
        data["overthinking_threshold"] = max(1, min(20, int(data.get("overthinking_threshold", 3) or 3)))
    except (TypeError, ValueError):
        data["overthinking_threshold"] = 3
    try:
        data["overthinking_window"] = max(100, min(5000, int(data.get("overthinking_window", 500) or 500)))
    except (TypeError, ValueError):
        data["overthinking_window"] = 500
    try:
        data["thinking_reserve"] = max(0.0, min(0.8, float(data.get("thinking_reserve", 0.25) or 0)))
    except (TypeError, ValueError):
        data["thinking_reserve"] = 0.25
    return data


class Store:
    def __init__(self, root: str | Path = ".epistemic"):
        self.root = Path(root)
        self.cases_dir = self.root / "cases"
        self.corpus_dir = self.root / "corpus"
        self.settings_path = self.root / "settings.json"
        self.lock = threading.Lock()
        self.cases_dir.mkdir(parents=True, exist_ok=True)
        self.corpus_dir.mkdir(parents=True, exist_ok=True)

    def settings(self) -> dict:
        data = dict(DEFAULT_SETTINGS)
        if self.settings_path.exists():
            try:
                saved = json.loads(self.settings_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                saved = {}
            for key, value in saved.items():
                if key in data and isinstance(value, dict) and isinstance(data[key], dict):
                    data[key].update(value)
                elif key in data:
                    data[key] = value
        return normalize_settings(data)

    def save_settings(self, patch: dict) -> dict:
        merged = self.settings()
        for key, value in (patch or {}).items():
            if key == "warnings" and isinstance(value, dict):
                merged["warnings"].update(value)
            elif key in merged:
                merged[key] = value
        merged = normalize_settings(merged)
        self.settings_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        return merged

    def save_case(self, case: CaseFile) -> Path:
        if not case.title:
            case.title = title_of(case.raw_prompt)
        path = self.cases_dir / f"{case.case_id}.json"
        tmp = path.with_suffix(f".{time.time_ns()}.tmp")
        tmp.write_text(json.dumps(case.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        try:
            os.replace(tmp, path)
        except OSError:
            tmp.unlink(missing_ok=True)
            raise
        return path

    def load_case(self, case_id: str) -> CaseFile | None:
        path = self.cases_dir / f"{case_id}.json"
        if not path.exists():
            return None
        try:
            return CaseFile.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            return None

    def case_summaries(self, limit: int = 60) -> list[dict]:
        items: list[dict] = []
        files = sorted(self.cases_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
        for path in files[: limit * 3]:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            tier = (data.get("editor") or {}).get("_meta", {}).get("tier") or ""
            items.append(
                {
                    "id": data.get("case_id", path.stem),
                    "title": data.get("title") or title_of(data.get("raw_prompt", "")),
                    "created_at": data.get("created_at") or path.stat().st_mtime,
                    "route": (data.get("route") or {}).get("ROUTE", "-"),
                    "status": case_status(data),
                    "tier": tier,
                    "mode": data.get("mode", "evidence-bounded"),
                    "continuation_count": data.get("continuation_count", 0),
                    "warnings": len(data.get("termination_warnings") or []),
                }
            )
            if len(items) >= limit:
                break
        return items

    def add_document(self, name: str, text: str) -> dict:
        safe = "".join(c if c.isalnum() or c in ".-_" else "_" for c in name)[:64] or "document"
        if not safe.endswith(".txt"):
            safe += ".txt"
        path = self.corpus_dir / safe
        path.write_text(text, encoding="utf-8")
        return {"name": safe, "bytes": len(text.encode("utf-8")), "modified": time.time()}

    def documents(self) -> list[dict]:
        out = []
        for path in sorted(self.corpus_dir.glob("*")):
            if path.is_file():
                out.append({"name": path.name, "bytes": path.stat().st_size, "modified": path.stat().st_mtime})
        return out

    def delete_document(self, name: str) -> bool:
        path = self.corpus_dir / name
        if path.exists() and path.is_file():
            path.unlink()
            return True
        return False


def case_status(data: dict) -> str:
    steps = {s.get("name"): s.get("status") for s in data.get("steps") or []}
    if data.get("error"):
        return "failed"
    if any(status == "waiting" for status in steps.values()):
        return "needs-confirmation"
    if steps.get("final") == "complete":
        return "warning" if data.get("termination_warnings") else "complete"
    if steps:
        return "running"
    if data.get("final_response"):
        return "complete"
    return "running"
