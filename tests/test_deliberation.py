import json
import time
from dataclasses import dataclass

import pytest

from epistemic import schemas
from epistemic.llm import LLMClient, LLMUnavailable, RoleResult, parse_field_text
from epistemic.pipeline import MAP_LIMIT, Checkpoint, DeliberationStream, Engine, ROLE_TOKENS, dedupe_models
from epistemic.rules import editor_rules
from web.jobs import Manager
from web.store import Store
from web import uidata

from tests.conftest import (
    CRITIC_PASS,
    EDITOR_PACKET,
    RESEARCHER_B,
    SKEPTIC_B,
    TRIAGE_ENGINE,
    WRITER_B,
    FakeLLM,
)

THINKING_MARK = "Attack: I almost wrote"
PROMPT = (
    "I feel like our company's productivity is down because of the new project management software. "
    "Give me a 5-step plan to convince the CEO to switch to a different software by next month."
)


def script(**over):
    base = {
        "Triage": [TRIAGE_ENGINE],
        "Skeptic": [SKEPTIC_B],
        "Researcher": [RESEARCHER_B],
        "Writer": [WRITER_B],
        "Critic": [CRITIC_PASS],
    }
    base.update(over)
    return base


def run(prompt=PROMPT, **engine_kwargs):
    client = FakeLLM(script=script())
    case = Engine(client=client, **engine_kwargs).run(prompt, Checkpoint(action="confirm"))
    return client, case


def test_writer_deliberation_is_recorded_and_kept_out_of_the_answer():
    client, case = run()
    assert case.deliberation
    assert THINKING_MARK in case.deliberation
    assert THINKING_MARK not in case.final_response
    assert "Next steps" in case.final_response


def test_writer_is_asked_for_a_private_monologue_in_text_mode():
    client, _ = run()
    call = next(c for c in client.calls if c["role"] == "Writer")
    assert "THINKING_PROCESS" in call["spec"]
    assert call["use_schema"] is False
    assert schemas.text_format_block(schemas.WRITER).count("THINKING_PROCESS") == 1


def test_the_answer_canvas_never_renders_the_deliberation():
    _, case = run()
    ui = uidata.build_ui(case)
    rendered = json.dumps(ui["answer"])
    assert THINKING_MARK not in rendered
    assert ui["deliberation"]["characters"] == len(case.deliberation)
    assert THINKING_MARK in ui["deliberation"]["text"]
    assert THINKING_MARK in json.dumps(ui["trace"]["deliberation"])


def test_thinking_stream_is_split_from_the_final_answer():
    received = []
    stream = DeliberationStream(on_thinking=received.append)
    raw = (
        "THINKING_PROCESS:\n- the register must stay cautious\n"
        "- Attack: I almost wrote 'caused' and then fixed it\n\nRESPONSE:\nWhat can be responsibly concluded: not established.\n"
    )
    for i in range(0, len(raw), 7):
        stream.feed(raw[i : i + 7])
    stream.finish()
    assert THINKING_MARK in stream.thinking_text
    assert "not established" in stream.final_text
    assert "not established" not in stream.thinking_text
    assert "".join(received).strip() == stream.thinking_text
    stream.reset()
    assert stream.thinking_text == "" and stream.final_text == ""


def test_bracketed_sections_are_understood():
    stream = DeliberationStream()
    stream.feed("[THINKING_PROCESS]\nplan the structure\n[FINAL_RESPONSE]\nThe bounded answer.")
    stream.finish()
    assert "plan the structure" in stream.thinking_text
    assert stream.final_text == "The bounded answer."


def test_role_budgets_are_large_and_the_timeout_is_generous():
    assert ROLE_TOKENS["ResearchMap"] == 16384
    assert ROLE_TOKENS["ResearchDistill"] == 8192
    assert ROLE_TOKENS["Writer"] == 16384
    assert ROLE_TOKENS["Critic"] == 8192
    assert LLMClient().timeout >= 300


def test_streaming_request_is_sent_and_deltas_are_forwarded():
    seen = {}

    def fake_post(payload, on_delta=None):
        seen["payload"] = payload
        for piece in ["THINKING_PROCESS:\n", "plan\n", "RESPONSE:\n", "The bounded answer."]:
            if on_delta:
                on_delta(piece)
        return "THINKING_PROCESS:\nplan\nRESPONSE:\nThe bounded answer.", 120, 400

    client = LLMClient(use_schema=False)
    client.resolve_model = lambda: "strata"
    client._post = fake_post
    result = client.ask("Writer", "sys", "user", schemas.WRITER, max_tokens=16384, on_delta=lambda text: None)
    assert result.record.streamed is True
    assert result.fields["RESPONSE"] == ["The bounded answer."]
    assert result.fields["THINKING_PROCESS"] == "plan"


def test_editor_selects_the_top_models_and_archives_the_rest():
    editor = editor_rules(SKEPTIC_B, RESEARCHER_B, TRIAGE_ENGINE, {}, {"mode": "evidence-bounded"})
    assert len(editor["SELECTED_MODELS"]) == 3
    assert editor["ARCHIVED_MODELS"]
    assert set(editor["SELECTED_MODELS"]).issubset(set(RESEARCHER_B["CAUSAL_MODELS"]))
    assert "Training" not in " ".join(editor["SELECTED_MODELS"])
    ui = uidata.build_ui(_case_with(editor))
    assert ui["trace"]["causal"]["selected"] == editor["SELECTED_MODELS"]
    assert ui["trace"]["causal"]["archived"]
    assert any(m["placement"] == "in the answer" for m in ui["trace"]["causal"]["models"])


def _case_with(editor):
    from epistemic.casefile import CaseFile

    return CaseFile(
        case_id="sel",
        raw_prompt=PROMPT,
        route=dict(TRIAGE_ENGINE),
        skeptic=dict(SKEPTIC_B),
        researcher=dict(RESEARCHER_B),
        editor=editor,
        final_response="What can be responsibly concluded: not established.",
    )


class StreamingLLM(FakeLLM):
    def ask(self, role, system, user, spec, max_tokens=None, on_delta=None, use_schema=None, **kw):
        result = super().ask(role, system, user, spec, max_tokens=max_tokens, on_delta=on_delta, use_schema=use_schema, **kw)
        if role == "Writer" and on_delta:
            text = result.fields.get("THINKING_PROCESS") or ""
            for i in range(0, len(text), 40):
                on_delta(text[i : i + 40])
        return result


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "store")


class StreamManager(Manager):
    def __init__(self, store):
        super().__init__(store, url="http://127.0.0.1:9")

    def build_engine(self, settings):
        client = StreamingLLM(script=script())
        self.client = client
        return Engine(client=client, retriever=None, use_critic=True, use_rule_engine=True, allow_rewrite=False)


def test_the_job_flushes_the_thinking_stream_in_chunks(store):
    manager = StreamManager(store)
    store.save_settings({"checkpoint_policy": "never", "mode": "engine"})
    job = manager.start(PROMPT, {"checkpoint_policy": "never", "mode": "engine"})
    listener = job.subscribe()
    for i in range(60):
        job.on_delta("Writer", f"deliberation line {i}\n")
    job.flush_deltas()
    events = []
    while not listener.empty():
        events.append(listener.get_nowait())
    job.unsubscribe(listener)
    deltas = [e for e in events if e.get("type") == "delta"]
    assert 2 <= len(deltas) < 60, "flushes must be batched, not one event per token"
    assert "".join(d["text"] for d in deltas).count("deliberation line") == 60
    assert deltas[-1]["characters"] == job.delta_chars


@dataclass
class FailingLLM(FakeLLM):
    fail_on: str = ""

    def ask(self, role, system, user, spec, max_tokens=None, on_delta=None, use_schema=None, **kw):
        if role == self.fail_on:
            self.calls.append({"role": role, "system": system, "user": user, "spec": [f.name for f in spec]})
            raise LLMUnavailable(f"unparseable text output for {role}")
        return super().ask(role, system, user, spec, max_tokens=max_tokens, on_delta=on_delta, use_schema=use_schema, **kw)


def test_lowercase_and_spaced_field_headers_still_parse():
    raw = "summary: Only weak evidence exists.\nkey facts:\n- baseline missing\n- metric undefined\nclaim confidence: 15\n"
    fields = parse_field_text(raw, schemas.RESEARCHER)
    assert fields is not None
    assert fields["SUMMARY"].startswith("Only weak evidence")
    assert fields["KEY_FACTS"] == ["baseline missing", "metric undefined"]
    assert fields["CLAIM_CONFIDENCE"] == "15"
    assert fields["_format"] == "text"


def test_partial_output_with_one_readable_field_is_salvaged():
    raw = "Some preamble prose the model wrote first.\n\n- EVIDENCE QUALITY: very weak, no baseline at all\nMore prose with no header."
    fields = parse_field_text(raw, schemas.RESEARCHER)
    assert fields is not None
    assert "very weak" in fields["EVIDENCE_QUALITY"]


def test_a_map_written_in_prose_style_is_parsed():
    raw = "territory map:\n- Software friction: predicts cycle-time rise at rollout.\n- Training gaps: predicts low competence scores."
    fields = parse_field_text(raw, schemas.RESEARCH_MAP)
    assert fields is not None
    assert len(fields["TERRITORY_MAP"]) == 2


def test_an_unreadable_distillation_degrades_research_instead_of_failing_the_run():
    client = FailingLLM(script=script(Editor=[EDITOR_PACKET]), fail_on="ResearchDistill")
    case = Engine(client=client).run(PROMPT, Checkpoint(action="confirm"))
    assert not case.error
    assert case.researcher["_degraded"]
    assert case.researcher["EVIDENCE_QUALITY"].startswith("unknown")
    assert case.researcher["CAUSAL_MODELS"]
    assert len(case.final_response) > 80
    assert [s["name"] for s in case.steps][-1] == "final"
    ui = uidata.build_ui(case)
    assert ui["evidence"]["degraded"]
    assert "degraded fallback" in ui["trace"]["overview"]["research_packet"]
    assert ui["answer"]["sections"]


def test_an_unreadable_map_degrades_research_instead_of_failing_the_run():
    client = FailingLLM(script=script(Editor=[EDITOR_PACKET]), fail_on="Researcher")
    case = Engine(client=client).run(PROMPT, Checkpoint(action="confirm"))
    assert not case.error
    assert "map pass failed" in case.researcher["_degraded"]
    assert case.final_response
    step = next(s for s in case.steps if s["name"] == "research")
    assert step["data"]["degraded"]


def test_the_map_is_deduplicated_and_capped():
    lines = [f"model {i}: predicts outcome number {i}" for i in range(60)]
    assert len(dedupe_models(lines)) == MAP_LIMIT
    noisy = ["Software friction reduced throughput: predicts a cycle-time rise at rollout."] * 5 + [
        "Training gaps: predicts low competence scores."
    ]
    assert dedupe_models(noisy) == [noisy[0], noisy[5]]


def test_the_map_pass_is_its_own_streamed_call():
    seen = []

    class Spy(FakeLLM):
        def ask(self, role, system, user, spec, max_tokens=None, on_delta=None, use_schema=None, **kw):
            if role == "Researcher" and on_delta:
                for line in ["model a: predicts X\n", "model b: predicts Y\n"]:
                    on_delta(line)
            seen.append((role, [f.name for f in spec], use_schema, on_delta is not None))
            return super().ask(role, system, user, spec, max_tokens=max_tokens, on_delta=on_delta, use_schema=use_schema, **kw)

    case = Engine(client=Spy(script=script())).run(PROMPT, Checkpoint(action="confirm"))
    entry = next(e for e in seen if e[0] == "Researcher")
    assert entry[1] == ["TERRITORY_MAP"]
    assert entry[2] is False
    assert entry[3] is True
    assert any(e[0] == "ResearchDistill" for e in seen)
    assert case.researcher["TERRITORY_MAP"] == RESEARCHER_B["TERRITORY_MAP"]
    assert case.researcher["TERRITORY_MAP_SIZE"] == len(RESEARCHER_B["TERRITORY_MAP"])
    assert case.researcher["TERRITORY_MAP_RAW"] == len(RESEARCHER_B["TERRITORY_MAP"])
    assert case.researcher["CAUSAL_MODELS"] == RESEARCHER_B["CAUSAL_MODELS"]


def test_the_ui_streams_the_deliberation_and_shows_only_the_answer(store):
    manager = StreamManager(store)
    store.save_settings({"checkpoint_policy": "never", "mode": "engine"})
    job = manager.start(PROMPT, {"checkpoint_policy": "never", "mode": "engine"})
    listener = job.subscribe()
    events = []
    deadline = time.time() + 12
    while time.time() < deadline:
        try:
            event = listener.get(timeout=0.2)
        except Exception:
            event = None
        if event:
            events.append(event)
            if event.get("type") in {"final", "error"}:
                break
    job.unsubscribe(listener)
    deltas = [e for e in events if e.get("type") == "delta"]
    assert deltas, "the writer must stream its deliberation"
    assert deltas[0]["tokens"] >= 0
    streamed = "".join(d["text"] for d in deltas)
    assert THINKING_MARK in streamed
    final = [e for e in events if e.get("type") == "final"]
    assert final and final[-1]["deliberation_characters"] > 0
    ui = uidata.build_ui(store.load_case(job.case_id))
    assert ui["deliberation"]["characters"] == len(store.load_case(job.case_id).deliberation)
    assert THINKING_MARK not in json.dumps(ui["answer"])
    assert ui["trace"]["critic"]["alignment"] == "ALIGNED"
