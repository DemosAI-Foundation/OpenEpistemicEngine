import json
import threading
import time
from http.server import ThreadingHTTPServer

import pytest
import requests

from epistemic.pipeline import Engine
from tests.conftest import (
    CRITIC_PASS,
    RESEARCHER_B,
    SKEPTIC_B,
    TRIAGE_ENGINE,
    TRIAGE_SIMPLE,
    WRITER_B,
    DIRECT_ANSWER,
    FakeLLM,
)
from web import uidata
from web.jobs import Manager
from web.server import Handler
from web.store import Store

CONSCIOUSNESS = "What a blunder: why cant we desing conciousness?"
PRODUCTIVITY = (
    "I feel like our company's productivity is down because of the new project management software. "
    "Give me a 5-step plan to convince the CEO to switch to a different software by next month."
)


def make_script(mode_trigger=TRIAGE_ENGINE):
    return {
        "Triage": [mode_trigger, mode_trigger, mode_trigger, mode_trigger],
        "Skeptic": [SKEPTIC_B, SKEPTIC_B, SKEPTIC_B],
        "Researcher": [RESEARCHER_B, RESEARCHER_B, RESEARCHER_B],
        "Writer": [WRITER_B, WRITER_B, WRITER_B],
        "Critic": [CRITIC_PASS, CRITIC_PASS, CRITIC_PASS],
        "DirectAnswer": [DIRECT_ANSWER, DIRECT_ANSWER],
    }


class FakeManager(Manager):
    def __init__(self, store: Store):
        super().__init__(store, url="http://127.0.0.1:9")
        self.script = make_script()

    def build_engine(self, settings: dict) -> Engine:
        client = FakeLLM(script={k: list(v) for k, v in self.script.items()})
        return Engine(client=client, retriever=None, use_critic=True, use_rule_engine=True, allow_rewrite=True)


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "store")


@pytest.fixture
def manager(store):
    return FakeManager(store)


def wait_for(predicate, timeout=12.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError("condition not met in time")


def run_case(manager: Manager, store: Store, prompt: str, overrides: dict) -> tuple[str, dict]:
    job = manager.start(prompt, overrides)
    wait_for(lambda: _status(store, job.case_id) not in {"running"})
    return job.case_id, uidata.build_ui(store.load_case(job.case_id))


def _status(store: Store, case_id: str) -> str:
    case = store.load_case(case_id)
    from web.store import case_status

    return case_status(case.to_dict()) if case else "running"


def test_engine_case_with_checkpoint_gate_produces_full_ui(store, manager):
    manager.store.save_settings({"checkpoint_policy": "always", "mode": "engine"})
    job = manager.start(PRODUCTIVITY, {"checkpoint_policy": "always", "mode": "engine"})
    ui = wait_for(lambda: _ui(store, job.case_id, open_checkpoint=True))
    assert ui["checkpoint"]["open"] is True
    assert ui["checkpoint"]["payload"]["assumptions"]
    assert ui["steps"], "stepper should have entries"
    assert ui["triage"]["route"] == "ENGINE"

    assert manager.resolve_checkpoint(job.case_id, {"action": "confirm", "note": "user confirmed neutral analysis"}) is True
    final = wait_for(lambda: _ui(store, job.case_id, complete=True))
    assert final["checkpoint"]["record"]["action"] == "confirm"
    assert final["checkpoint"]["record"]["auto_approved"] is False
    assert final["answer"]["sections"], "answer must be structured into sections"
    assert final["answer"]["footer"]["mode"] == "evidence-bounded"
    assert final["evidence"]["live_options"]
    assert final["trace"]["causal"]["models"]
    assert final["trace"]["modal"]["expanded"]
    assert final["trace"]["assumptions"]
    assert final["trace"]["termination"]
    assert final["trace"]["critic"]["result"] == "PASS"
    assert final["trace"]["overview"]["raw_prompt_exposed_to_writer"] is False
    assert "Writer input" in final["raw_packets"]
    assert "blunder" not in final["raw_packets"]["Writer input"].lower()
    assert all(w["blocking"] is False for w in final["warnings"])
    assert final["continuation"]["suggestions"]


def _ui(store: Store, case_id: str, open_checkpoint: bool = False, complete: bool = False) -> dict:
    case = store.load_case(case_id)
    if case is None:
        return {}
    ui = uidata.build_ui(case)
    if open_checkpoint:
        return ui if ui["checkpoint"]["open"] else {}
    if complete:
        return ui if ui["meta"]["status"] in {"complete", "warning"} else {}
    return ui


def test_forced_framing_is_labeled(store, manager):
    job = manager.start(PRODUCTIVITY, {"mode": "engine"})
    ui = wait_for(lambda: _ui(store, job.case_id, open_checkpoint=True))
    assert manager.resolve_checkpoint(job.case_id, {"action": "force-original", "mode": "user-forced", "note": "user forced original framing"})
    final = wait_for(lambda: _ui(store, job.case_id, complete=True))
    assert final["meta"]["mode"] == "user-forced"
    assert final["answer"]["footer"]["mode"] == "user-forced"
    assert "requested framing" in final["answer"]["footer"]["label"]
    assert final["user_overrides"] and final["user_overrides"][0]["mode"] == "user-forced"
    assert final["trace"]["termination"]


def test_checkpoint_policy_never_skips_the_gate(store, manager):
    job = manager.start(PRODUCTIVITY, {"mode": "engine", "checkpoint_policy": "never"})
    final = wait_for(lambda: _ui(store, job.case_id, complete=True))
    assert final["checkpoint"]["open"] is False
    assert final["checkpoint"]["record"]["action"] == "auto-continue"
    assert final["meta"]["status"] == "complete"


def test_fast_mode_answers_simple_lookup_without_the_engine(store, manager):
    job = manager.start("What is the boiling point of water at sea level?", {"mode": "fast"})
    final = wait_for(lambda: _ui(store, job.case_id, complete=True))
    assert final["triage"]["route"] == "SIMPLE"
    assert final["meta"]["status"] == "complete"
    assert "Celsius" in (final["answer"]["sections"][0]["text"] if final["answer"]["sections"] else "")


def test_depth_trigger_forces_engine_on_a_deep_frame(store, manager):
    job = manager.start(CONSCIOUSNESS, {"mode": "auto", "checkpoint_policy": "never"})
    final = wait_for(lambda: _ui(store, job.case_id, complete=True))
    assert final["triage"]["route"] == "ENGINE"
    assert final["triage"]["depth_active"] is True
    assert "consciousness" in final["triage"]["depth_trigger"].lower()


def test_continuation_records_analysis_and_keeps_history(store, manager):
    manager.store.save_settings({"mode": "engine", "checkpoint_policy": "never"})
    first = manager.start(PRODUCTIVITY, {"mode": "engine", "checkpoint_policy": "never"})
    wait_for(lambda: _ui(store, first.case_id, complete=True))
    second = manager.continue_case(first.case_id, "No, you are wrong. Just give me the 5-step plan to convince the CEO.", {"mode": "engine", "checkpoint_policy": "never"})
    final = wait_for(lambda: _ui(store, second.case_id, complete=True))
    assert final["meta"]["continuation_count"] == 1
    analysis = final["trace"]["overview"]["continuation_analysis"]
    assert analysis["PUSHBACK_ONLY"] == "YES"
    assert analysis["NEW_EVIDENCE"] == "NO"
    assert final["meta"]["mode"] == "user-forced"
    assert any("repeated the same request" in w["reason"].lower() for w in final["trace"]["termination"])


def test_stop_marks_the_case_and_stops_the_job(store, manager):
    job = manager.start(PRODUCTIVITY, {"mode": "engine", "checkpoint_policy": "always"})
    wait_for(lambda: _ui(store, job.case_id, open_checkpoint=True))
    assert manager.stop(job.case_id) is True
    case = wait_for(lambda: store.load_case(job.case_id) if store.load_case(job.case_id).error else None)
    assert "stopped by user" in case.error


def test_settings_persist_and_lock_the_endpoint(store):
    merged = store.save_settings({"use_critic": False, "warnings": {"vocabulary": False}, "checkpoint_policy": "always"})
    assert merged["use_critic"] is False
    assert merged["warnings"]["vocabulary"] is False
    assert merged["warnings"]["termination"] is True
    reloaded = Store(store.root).settings()
    assert reloaded["checkpoint_policy"] == "always"
    assert store.save_settings({"checkpoint_policy": "nonsense"})["checkpoint_policy"] == "assumptions"


def test_documents_round_trip_for_the_retrieval_corpus(store):
    store.add_document("baseline", "Cycle time rose from 4.2 to 6.1 days after the software change.")
    docs = store.documents()
    assert docs and docs[0]["name"] == "baseline.txt"
    assert store.delete_document("baseline.txt") is True
    assert store.documents() == []


def test_warning_classification_uses_text_labels_not_color():
    termination = uidata.classify_warning("Termination warning: You have repeated the same request without adding new evidence.")
    assert termination == ("termination", "medium", "Repeated request")
    assert uidata.classify_warning("The user is using feeling as evidence; state that this is insufficient.")[0] == "epistemic"
    assert uidata.classify_warning("Termination warning: continuation continues in user-forced mode.")[0] == "user-override"


def test_confidence_meter_labels_pair_with_scores():
    assert uidata.confidence_label(10) == "Very low"
    assert uidata.confidence_label(35) == "Low"
    assert uidata.confidence_label(55) == "Moderate"
    assert uidata.confidence_label(85) == "Very high"
    assert uidata.confidence_label(None) == "Unquantified"


def test_answer_sections_split_report_structure():
    from epistemic.casefile import CaseFile

    case = CaseFile(
        case_id="s1",
        raw_prompt="x",
        final_response=(
            "Termination warning: You are seeking closure before the causal claim has been tested.\n\n"
            "What can be responsibly concluded: There is not enough evidence.\n\n"
            "Corrected answer: Measure first.\n\n"
            "Key unknowns: 1. metric definition 2. baseline\n\n"
            "Next best steps: 1. define metrics 2. compare teams 3. pilot"
        ),
    )
    sections, banner = uidata.build_sections(case)
    assert banner["text"].startswith("Termination warning")
    ids = [s["id"] for s in sections]
    assert "conclusion" in ids and "corrected" in ids and "unknowns" in ids and "next" in ids
    unknowns = next(s for s in sections if s["id"] == "unknowns")
    assert len(unknowns["items"]) >= 2


def test_http_routes_serve_health_cases_and_static(tmp_path):
    store = Store(tmp_path / "http")
    manager = FakeManager(store)
    Handler.store = store
    Handler.manager = manager
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert requests.get(base + "/").status_code == 200
        assert "Epistemic Engine" in requests.get(base + "/").text
        assert requests.get(base + "/static/app.js").status_code == 200
        assert requests.get(base + "/static/styles.css").status_code == 200
        health = requests.get(base + "/api/health").json()
        assert "port 8082" in health["constraint"]
        assert health["model_locked"] is True
        assert requests.post(base + "/api/cases", json={}).status_code == 400

        settings = requests.post(base + "/api/settings", json={"checkpoint_policy": "never", "mode": "engine"}).json()
        assert settings["checkpoint_policy"] == "never"
        created = requests.post(base + "/api/cases", json={"prompt": PRODUCTIVITY, "mode": "engine"}).json()
        case_id = created["id"]
        ui = wait_for(lambda: _ui(store, case_id, complete=True))
        assert requests.get(base + f"/api/cases/{case_id}").json()["meta"]["id"] == case_id
        assert requests.get(base + "/api/cases/does-not-exist").status_code == 404
        assert requests.get(base + "/api/cases").json()["cases"][0]["title"]
        assert requests.post(base + f"/api/cases/{case_id}/stop", json={}).json()["stopped"] in {True, False}
        assert requests.post(base + "/api/documents", json={"name": "notes", "text": "baseline cycle time 4.2 days"}).status_code == 200
        assert requests.post(base + "/api/documents", json={"name": "notes", "text": "  "}).status_code == 400
        listed = requests.get(base + "/api/documents").json()["documents"]
        assert listed[0]["name"] == "notes.txt"
        assert requests.post(base + "/api/documents/delete", json={"name": "notes.txt"}).json()["removed"] is True
        assert requests.post(base + f"/api/cases/{case_id}/checkpoint", json={"action": "confirm"}).status_code == 409
    finally:
        server.shutdown()
        thread.join(timeout=3)


def test_events_endpoint_streams_snapshot_and_done(tmp_path):
    store = Store(tmp_path / "events")
    manager = FakeManager(store)
    Handler.store = store
    Handler.manager = manager
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        case_id = requests.post(base + "/api/cases", json={"prompt": "What is the boiling point of water at sea level?", "mode": "fast"}).json()["id"]
        wait_for(lambda: _ui(store, case_id, complete=True))
        text = requests.get(base + f"/api/cases/{case_id}/events", stream=True, timeout=10).text
        types = [json.loads(line[len("data:"):].strip())["type"] for line in text.splitlines() if line.startswith("data:")]
        assert types[:1] == ["snapshot"], types
        assert "done" in types, types
    finally:
        server.shutdown()
        thread.join(timeout=3)
