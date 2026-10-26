"""Robust JSON recovery + jsonschema validation for small local models."""

from __future__ import annotations

import json
import re

from jsonschema import Draft7Validator

from .prompts import SCHEMA_ALIASES

FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str):
    if text is None:
        raise ValueError("empty model output")
    candidates = [text.strip()]
    candidates += [m.strip() for m in FENCE_RE.findall(text)]
    for chunk in candidates:
        if not chunk:
            continue
        try:
            obj = json.loads(chunk)
            if isinstance(obj, (dict, list)):
                return obj
        except json.JSONDecodeError:
            pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = min(
            [i for i in (text.find(opener), text.find(opener)) if i >= 0] or [-1]
        )
        if start < 0:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    snippet = text[start : i + 1]
                    try:
                        return json.loads(snippet)
                    except json.JSONDecodeError:
                        break
    raise ValueError("no JSON object found in model output")


def validate(obj, schema: dict) -> list[str]:
    validator = Draft7Validator(schema)
    return [f"{e.json_path}: {e.message}" for e in validator.iter_errors(obj)]


def _default_for(schema: dict):
    if "x_default" in schema:
        return schema["x_default"]
    t = schema.get("type")
    if t == "string":
        if "enum" in schema:
            return schema["enum"][0]
        return ""
    if t == "boolean":
        return False
    if t in ("array", "object"):
        return [] if t == "array" else {}
    if t in ("integer", "number"):
        return 0
    if "enum" in schema:
        return schema["enum"][0]
    return None


def _norm_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def _flatten_nested(obj: dict, schema: dict, notes: list[str]) -> dict:
    """Models often nest the sidecar fields, e.g. {"chat_reply": ..., "metadata":
    {...}}. A nested object is merged upward when it carries schema keys, and
    nested alias names (e.g. "provenance") are renamed on the way up."""
    props = schema.get("properties", {})
    accepted: dict[str, str] = {}
    for prop in props:
        accepted[_norm_key(prop)] = prop
    for prop in props:
        for alias in SCHEMA_ALIASES.get(prop, []):
            accepted.setdefault(_norm_key(alias), prop)
    out = dict(obj)
    for key, value in list(obj.items()):
        if not isinstance(value, dict):
            continue
        if _norm_key(key) in accepted:
            continue
        moved = []
        for inner_key, inner_val in value.items():
            known = _norm_key(inner_key) in accepted
            target = accepted.get(_norm_key(inner_key), inner_key)
            if target in out:
                continue
            out[target] = inner_val
            if known:
                moved.append(f"{inner_key}->{target}" if target != inner_key else inner_key)
            else:
                moved.append(f"{inner_key} (kept as extra)")
        if moved:
            out.pop(key, None)
            notes.append(f"flattened nested object '{key}' -> {moved[:6]}")
    return out


def alias_keys(obj: dict, schema: dict, notes: list[str]) -> dict:
    """Map a model's near-miss keys onto the schema's names, losslessly."""
    props = schema.get("properties", {})
    if not props:
        return obj
    norm = {_norm_key(p): p for p in props}
    out = dict(obj)
    for key in list(obj):
        nk = _norm_key(key)
        if nk in norm:
            continue
        target = None
        for pn, prop in norm.items():
            if pn and nk and pn.rstrip("s") == nk.rstrip("s"):
                target = prop
                break
        if target is None:
            for prop, cands in SCHEMA_ALIASES.items():
                if prop in props and any(_norm_key(c) == nk for c in cands):
                    target = prop
                    break
        if target is None:
            kt = set(re.findall(r"[a-z]+", str(key).lower()))
            best, best_score = None, 0
            for prop in props:
                score = len(kt & set(re.findall(r"[a-z]+", prop.lower())))
                if score > best_score:
                    best, best_score = prop, score
            if best_score >= 2:
                target = best
        if target and target not in obj:
            out[target] = obj[key]
            out.pop(key)
            notes.append(f"aliased model key '{key}' -> '{target}'")
    return out


def _exact_alias(obj: dict, schema: dict, notes: list[str]) -> dict:
    """Rename keys that exactly match a declared alias of a schema property.
    Runs first, so a container under an alias name (e.g. "provenance" for
    "citations") arrives at its own field's normaliser intact."""
    props = schema.get("properties", {})
    out = dict(obj)
    for key in list(obj):
        if key in props:
            continue
        for prop, cands in SCHEMA_ALIASES.items():
            if prop in props and any(_norm_key(c) == _norm_key(key) for c in cands):
                if prop not in out:
                    out[prop] = obj[key]
                    out.pop(key)
                    notes.append(f"aliased model key '{key}' -> '{prop}'")
                break
    return out


def coerce(obj, schema: dict, notes: list[str] | None = None) -> dict:
    """Best-effort repair so a valid-shaped object always reaches the UI."""
    notes = notes if notes is not None else []
    if not isinstance(obj, dict):
        obj = {"_raw": obj}
        notes.append("model returned non-object; wrapped under _raw")
    obj = _exact_alias(obj, schema, notes)
    obj = _flatten_nested(obj, schema, notes)
    obj = alias_keys(obj, schema, notes)
    props = schema.get("properties", {})
    out: dict = {}
    for key, spec in props.items():
        if key in obj:
            if key == "citations" and (spec.get("items") or {}).get("type") == "object":
                out[key] = _clean(normalise_citations(obj[key], notes), spec, f"{key}", notes)
            elif key == "dialectic" and spec.get("type") == "object":
                out[key] = _clean(normalise_dialectic(obj[key], notes), spec, f"{key}", notes)
            elif key in ROW_KEYS and spec.get("type") == "array":
                out[key] = _clean(normalise_rows(key, obj[key], notes), spec, f"{key}", notes)
            else:
                out[key] = _clean(obj[key], spec, f"{key}", notes)
        else:
            out[key] = _default_for(spec)
            if key in schema.get("required", []):
                notes.append(f"missing required field '{key}' was defaulted")
    for key in obj:
        if key not in props:
            out[key] = obj[key]
    for req in schema.get("required", []):
        out.setdefault(req, _default_for(props.get(req, {})))
    return out


CITED_PREFIX = re.compile(r"^\s*[\[({]?\s*(R\d+-[CS]\d+)\s*[\])}?]?\s*[:\-–]?\s*(.*)$", re.S)


ID_LIST_KEYS = ("reference_used", "references_used", "chunk_ids", "ids", "chunk_id",
                "cited", "citations", "references")
NESTED_LIST_KEYS = ("items", "rows", "list", "citations", "references", "quotes")


def _citation_container(value: dict, notes: list[str]) -> list:
    """Models emit citations as {"provenance": {"reference_used": ["R1-C1"],
    "grounding_note": "..."}}. Lift every recoverable id, keep the note."""
    out: list = []
    for key in ID_LIST_KEYS:
        raw = value.get(key)
        if isinstance(raw, str):
            out.append(raw)
        elif isinstance(raw, list):
            out += [r for r in raw if isinstance(r, (str, dict))]
    for key in NESTED_LIST_KEYS:
        raw = value.get(key)
        if isinstance(raw, list) and key not in ID_LIST_KEYS:
            out += [r for r in raw if isinstance(r, (str, dict))]
    if not out:
        notes.append("citations: container carried no chunk ids")
    return out


def normalise_citations(value, notes: list[str]) -> list[dict]:
    """Local models emit citations as strings like "R1-C1: the quoted line".
    Convert to {chunk_id, quote} rows; never invent a chunk id."""
    if not value:
        return []
    if isinstance(value, dict):
        value = _citation_container(value, notes)
    rows: list[dict] = []
    for item in value:
        if isinstance(item, dict):
            row = {k: v for k, v in item.items()}
            if "chunk_id" not in row:
                for probe in ("id", "chunk", "node_id", "reference"):
                    if probe in row:
                        row["chunk_id"] = row.pop(probe)
                        break
            if "quote" not in row:
                for probe in ("text", "quoted_text", "excerpt", "claim"):
                    if probe in row:
                        row["quote"] = row.pop(probe)
                        break
            if row.get("chunk_id"):
                rows.append({"chunk_id": str(row.get("chunk_id", "")).strip(),
                             "quote": str(row.get("quote", "")).strip()})
            continue
        text = str(item).strip()
        if not text:
            continue
        m = CITED_PREFIX.match(text)
        if m and m.group(1):
            rows.append({"chunk_id": m.group(1), "quote": m.group(2).strip()})
        else:
            notes.append("citations: dropped a citation with no recoverable chunk id")
    if value and not rows:
        notes.append("citations: no usable citation rows recovered from the model output")
    return rows


DIALECTIC_MARKERS = {
    "thesis": r"thesis|primary position|position|pro",
    "antithesis": r"antithesis|counter-?argument|opposition|objection|con",
    "synthesis": r"synthesis|resolution|reconciliation|way forward|outcome",
}


def normalise_dialectic(value, notes: list[str]) -> dict:
    """Models emit the dialectic as prose, a list, or with drifted keys."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        spans = []
        for key, pat in DIALECTIC_MARKERS.items():
            m = re.search(r"(?i)(?:^|[.\n;]|\*\*)\s*" + pat + r"\s*[:\-–]\s*", text)
            if m:
                spans.append((m.start(), m.end(), key))
        if not spans:
            notes.append("dialectic: prose carried no thesis/antithesis/synthesis markers")
            return {"thesis": text, "antithesis": "", "synthesis": ""}
        spans.sort()
        out = {}
        for i, (_start, end, key) in enumerate(spans):
            stop = spans[i + 1][0] if i + 1 < len(spans) else len(text)
            out[key] = text[end:stop].strip(" .;\n")
        for key in DIALECTIC_MARKERS:
            out.setdefault(key, "")
        notes.append("dialectic: split prose into thesis/antithesis/synthesis")
        return out
    if isinstance(value, list):
        keys = ["thesis", "antithesis", "synthesis"]
        out = {}
        for i, item in enumerate(value[:3]):
            if isinstance(item, dict):
                out.update(normalise_dialectic(item, notes))
            else:
                out[keys[i]] = str(item).strip()
        for key in keys:
            out.setdefault(key, "")
        notes.append(f"dialectic: positional list mapped to {keys}")
        return out
    notes.append(f"dialectic: unusable type {type(value).__name__}")
    return {}


ROW_KEYS = {
    "vocabulary_contract": (
        {"original": ("original", "term", "word", "user_term", "original_term"),
         "interpreted_as": ("interpreted_as", "interpreted", "meaning", "interpretation"),
         "reason": ("reason", "why", "rationale")},
    ),
    "theory_matrix": (
        {"option": ("option", "theory", "hypothesis", "position"),
         "viable_under": ("viable_under", "supported_by", "supports", "consistent_with"),
         "excluded_under": ("excluded_under", "excluded_by", "excludes", "ruled_out_by")},
    ),
    "claim_typology": (
        {"claim": ("claim", "statement", "text", "assertion"),
         "type": ("type", "kind", "classification", "claim_type")},
    ),
}


def normalise_rows(key: str, value, notes: list[str]) -> list:
    """Models return these as dicts keyed by term, as "a -> b" strings, or as
    plain strings. Normalise to a list of row dicts."""
    spec = ROW_KEYS.get(key)
    if not spec:
        return value if isinstance(value, list) else []
    fields = spec[0]
    out: list = []
    if isinstance(value, dict):
        for k, v in value.items():
            row = {}
            names = list(fields)
            if key == "vocabulary_contract":
                row = {"original": str(k), "interpreted_as": str(v), "reason": ""}
            elif key == "theory_matrix":
                row = {"option": str(k)}
                if isinstance(v, (list, str)):
                    row["viable_under"] = v if isinstance(v, list) else [v]
                    row["excluded_under"] = []
                elif isinstance(v, dict):
                    row.update(v)
            elif key == "claim_typology":
                row = {"claim": str(k), "type": str(v) if not isinstance(v, (list, dict)) else ""}
            out.append(row)
    elif isinstance(value, str):
        for line in re.split(r"\n|;", value):
            line = line.strip(" -\t")
            if not line:
                continue
            m = re.match(r"^(.*?)\s*(?:->|→|=|:\s*)\s*(.*)$", line)
            if m and key == "vocabulary_contract":
                out.append({"original": m.group(1), "interpreted_as": m.group(2),
                            "reason": ""})
            elif m and key == "claim_typology":
                out.append({"claim": m.group(1), "type": m.group(2)})
            elif m and key == "theory_matrix":
                out.append({"option": m.group(1),
                            "viable_under": [x.strip() for x in
                                             re.split(r",|;", m.group(2)) if x.strip()],
                            "excluded_under": []})
            else:
                out.append({list(fields)[0]: line})
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                row = {}
                for field, cands in fields.items():
                    for cand in cands:
                        if cand in item:
                            row[field] = item[cand]
                            break
                out.append(row or {list(fields)[0]: item})
            elif isinstance(item, str):
                out.append({list(fields)[0]: item})
    return out


def _enum_match(value: str, choices: list[str]) -> str | None:
    low = re.sub(r"[^a-z]", "", str(value).lower())
    for c in choices:
        if re.sub(r"[^a-z]", "", c.lower()) == low:
            return c
    tokens = set(re.findall(r"[a-z]+", str(value).lower()))
    best, best_score = None, 0
    for c in choices:
        score = len(tokens & set(re.findall(r"[a-z]+", c.lower())))
        if score > best_score:
            best, best_score = c, score
    return best if best_score >= 1 else None


def _clean(value, spec: dict, path: str, notes: list[str]):
    t = spec.get("type")
    if t == "string":
        if not isinstance(value, str):
            value = "" if value is None else json.dumps(value) if isinstance(value, (dict, list)) else str(value)
            notes.append(f"{path}: coerced to string")
        if "enum" in spec and value not in spec["enum"]:
            match = _enum_match(value, spec["enum"])
            if match:
                notes.append(f"{path}: {value!r} mapped to enum {match!r}")
                return match
            fallback = spec.get("x_default", spec["enum"][0])
            notes.append(f"{path}: {value!r} not in enum, set to {fallback!r}")
            return fallback
        return value.strip()
    if t == "boolean":
        if isinstance(value, str):
            return value.strip().lower() in ("true", "yes", "1")
        return bool(value)
    if t == "array":
        if isinstance(value, str):
            value = [v.strip() for v in re.split(r"[;\n]|,\s(?=\w)", value) if v.strip()]
            notes.append(f"{path}: split string into array")
        elif not isinstance(value, (list, tuple)):
            value = [] if value is None else [value]
            notes.append(f"{path}: wrapped scalar into array")
        items = spec.get("items", {})
        if items.get("enum"):
            kept = [v for v in value if isinstance(v, str) and v.strip() in items["enum"]]
            if len(kept) != len(list(value)):
                dropped = [v for v in value if v not in kept]
                notes.append(f"{path}: dropped non-enum values {dropped!r} "
                             f"(allowed: {items['enum']})")
            value = kept
        cleaned = [_clean(v, items, f"{path}[{i}]", notes) for i, v in enumerate(value)]
        if "maxItems" in spec and len(cleaned) > spec["maxItems"]:
            notes.append(f"{path}: truncated {len(cleaned)} -> {spec['maxItems']} items")
            cleaned = cleaned[: spec["maxItems"]]
        return [c for c in cleaned if c not in ("", None, [])]
    if t == "object":
        if not isinstance(value, dict):
            notes.append(f"{path}: expected object, got {type(value).__name__}")
            return {}
        return coerce(value, spec, notes)
    return value


def parse_and_repair(text: str, schema: dict, notes: list[str] | None = None) -> dict:
    notes = notes if notes is not None else []
    obj = extract_json(text)
    obj = coerce(obj, schema, notes)
    errors = validate(obj, schema)
    if errors:
        notes.append("schema issues after repair: " + "; ".join(errors[:4]))
    return obj
