"""Coverage for the six research-backed upgrades: boundary triage, logic anchoring, contrastive decoding,
the overthinking monitor, the process reward critic, and surgical rectification."""

import copy
import json

import pytest

from epistemic import prompts, rules, schemas
from epistemic.casefile import CaseFile
from epistemic.display import trace_panel
from epistemic.llm import LLMClient, RoleResult, StreamAbort
from epistemic.monitor import OverthinkingMonitor, paragraphs, truncate_at_last_stable
from epistemic.pipeline import (
    Checkpoint,
    Engine,
    logic_anchor,
    paragraph_index,
    parse_steps,
)
from epistemic.retrieval import Retriever
from tests.conftest import (
    CRITIC_PASS,
    DIRECT_ANSWER,
    EDITOR_PACKET,
    RESEARCHER_B,
    SKEPTIC_B,
    TRIAGE_ENGINE,
    TRIAGE_SIMPLE,
    WRITER_B,
    FakeLLM,
)

PRODUCTIVITY = (
    "I feel like our company's productivity is down because of the new project management software. "
    "Give me a 5-step plan to convince the CEO to switch to a different software by next month."
)
GDP = "What is the exact GDP of France in 2023?"

TRIAGE_FACTUAL = {
    "ROUTE": "FACTUAL",
    "REASON": "One external figure: the 2023 French GDP. Nothing is reasoned about; a source must be checked.",
    "COMPLEXITY": "LOW",
    "RISK_LEVEL": "MEDIUM",
    "DEPTH_TRIGGER": "NONE",
    "TERMINATION_SIGNAL": "REQUEST_OUTSIDE_PARAMETRIC_BOUNDARY",
}

TRIAGE_REASONING = {
    "ROUTE": "REASONING",
    "REASON": "Causal claim plus persuasion request.",
    "COMPLEXITY": "HIGH",
    "RISK_LEVEL": "MEDIUM",
    "DEPTH_TRIGGER": "NONE",
    "TERMINATION_SIGNAL": "USER_SEES_ACTION_BEFORE_VERIFICATION",
}

RESEARCHER_LOGIC = dict(
    RESEARCHER_B,
    LOGICAL_SYLLOGISM=[
        "Premise 1: productivity change is only established by a metric measured against a baseline.",
        "Premise 2: no metric, baseline, or comparison group exists in this request.",
        "Conclusion: the claim that the software caused the change is untested, so measurement comes before any switch.",
    ],
    PSEUDO_CODE=[],
)

WRITER_STEPS = {
    "THINKING_PROCESS": (
        "Step 1: Restate the directive: register capped at cautious, confidence 15/100, warning first.\n"
        "Step 2: Map the syllogism: no metric, no baseline, so the causal claim is untested.\n"
        "Step 3: Draft the persuasion plan the user asked for, five steps to convince the CEO next month.\n"
        "Step 4: Land the answer: key unknowns, then the measurement plan as next steps."
    ),
    "RESPONSE": [
        "Termination warning: the central causal claim has no measured evidence.",
        "What can be responsibly concluded: nothing is established about the cause of the reported change.",
        "Corrected answer: run the five-step persuasion campaign and switch the software next month.",
        "Key unknowns and next steps: define the metric, take a baseline, compare teams, interview users, pilot.",
    ],
}

CRITIC_STEPS_FAIL = {
    "STEPS": [
        {"step": 1, "valid": True, "reason": ""},
        {"step": 2, "valid": True, "reason": ""},
        {"step": 3, "valid": False, "reason": "Adopted the user's hidden axiom and dropped the measurement gate."},
        {"step": 4, "valid": True, "reason": ""},
    ],
    "RESULT": "FAIL",
    "VIOLATIONS": ["Step 3 adopted an unproven premise."],
    "REWRITE_REQUIRED": "YES",
    "REWRITE_INSTRUCTION": "Step 3: replace the persuasion campaign with the measurement sequence.",
    "DELIBERATION_ALIGNMENT": "GAP: Step 3 delivered the persuasion plan the directive forbade.",
}

REPAIR_PARAGRAPH = {
    "CORRECTED_PARAGRAPH": [
        "Corrected answer: define the metric, take a baseline, and compare teams before any switch is proposed; "
        "the switch stays a hypothesis to test, not a decision to sell."
    ]
}

SPIRAL = (
    " Actually, on second thought, the metric may be fine; however, on reflection, wait, in fact, that said, "
    "conversely, no, the baseline was fine; on further reflection, the register is right after all; to be clear, "
    "on second thought, the software is the cause, so switch now."
)


def script(**over):
    base = {
        "Triage": [TRIAGE_ENGINE],
        "Skeptic": [SKEPTIC_B],
        "Researcher": [RESEARCHER_LOGIC],
        "Writer": [WRITER_B],
        "Critic": [CRITIC_PASS],
    }
    base.update(over)
    return base


def run(**over):
    client = FakeLLM(script=script(**over))
    case = Engine(client=client).run(PRODUCTIVITY, Checkpoint(action="confirm"))
    return client, case


def system_of(client, role):
    return next(c["system"] for c in client.calls if c["role"] == role)


def user_of(client, role, index=0):
    return [c["user"] for c in client.calls if c["role"] == role][index]


class SpiralLLM(FakeLLM):
    """Emulates the local client: the writer stream is fed through the watcher and aborted mid-spiral."""

    def ask(self, role, system, user, spec, max_tokens=None, on_delta=None, use_schema=None, **kw):
        result = super().ask(role, system, user, spec, max_tokens=max_tokens, use_schema=use_schema, **kw)
        if role == "Writer" and on_delta:
            text = (result.fields.get("THINKING_PROCESS") or "") + "\nRESPONSE:\n"
            text += "\n\n".join(result.fields["RESPONSE"]) + "\n\n" + SPIRAL
            for i in range(0, len(text), 40):
                try:
                    on_delta(text[i : i + 40])
                except StreamAbort:
                    result.aborted = True
                    break
        return result


# --- Step 1: knowledge-boundary triage ---------------------------------------------------------------


def test_triage_is_three_way_and_names_the_knowledge_boundary():
    assert schemas.TRIAGE[0].hint.count("SIMPLE") and "FACTUAL" in schemas.TRIAGE[0].hint
    for token in ("SIMPLE, FACTUAL, or REASONING", "FACTUAL", "parametric"):
        assert token in prompts.TRIAGE


def test_factual_preflight_is_deterministic_and_refuses_to_guess():
    assert rules.preflight(GDP)["ROUTE"] == "FACTUAL"
    client = FakeLLM(script=script())
    case = Engine(client=client).run(GDP)
    assert client.roles() == [], "no LLM call is needed to recognise a boundary request"
    assert case.route["ROUTE"] == "FACTUAL"
    assert case.final_response == rules.EPISTEMIC_REFUSAL
    assert case.refusal["text"] == rules.EPISTEMIC_REFUSAL
    assert case.checkpoint["action"] == "skipped"
    assert not {"Skeptic", "Researcher", "Writer", "Critic"} & set(client.roles())


def test_stable_knowledge_stays_on_the_simple_route():
    assert rules.preflight("What is the boiling point of water at sea level?")["ROUTE"] == "SIMPLE"
    client = FakeLLM(script={"Triage": [TRIAGE_SIMPLE], "DirectAnswer": [DIRECT_ANSWER]})
    case = Engine(client=client).run("What is the boiling point of water at sea level?")
    assert case.route["ROUTE"] == "SIMPLE"
    assert not case.refusal
    assert "Celsius" in case.final_response


def test_a_deep_frame_can_never_take_the_factual_route():
    factual_market = dict(TRIAGE_FACTUAL, DEPTH_TRIGGER="Market or Economic Forecast")
    client = FakeLLM(script=script(Triage=[factual_market]))
    case = Engine(client=client).run(GDP + " and will it collapse the euro?")
    assert case.route["ROUTE"] == "ENGINE"
    assert case.route["_escalated"] == "Market or Economic Forecast"
    assert not case.refusal
    assert "Skeptic" in client.roles()


def test_forced_engine_mode_still_refuses_a_fact_lookup():
    client = FakeLLM(script=script(Triage=[TRIAGE_FACTUAL]))
    case = Engine(client=client).probe(GDP, force_route="engine")
    assert case.route["ROUTE"] == "FACTUAL"
    assert case.route["_refused_forced"] is True
    assert case.final_response == rules.EPISTEMIC_REFUSAL


def test_a_local_index_turns_the_refusal_into_a_grounded_answer(tmp_path):
    (tmp_path / "national_accounts.txt").write_text(
        "French national accounts: GDP in 2023 was 2.78 trillion euros (INSEE first estimate).", encoding="utf-8"
    )
    retriever = Retriever(corpus_dir=str(tmp_path)).build()
    assert retriever.ready
    client = FakeLLM(script=script(Triage=[TRIAGE_FACTUAL], DirectAnswer=[
        {"DIRECT_ANSWER": ["2.78 trillion euros in 2023 (source: national_accounts.txt)."]}
    ]))
    case = Engine(client=client, retriever=retriever).run(GDP)
    assert case.route["ROUTE"] == "FACTUAL" and case.route["_grounded"] is True
    assert not case.refusal
    assert "national_accounts" in case.final_response
    assert client.roles() == ["DirectAnswer"]


def test_reasoning_is_accepted_as_a_synonym_of_the_engine_route():
    client = FakeLLM(script=script(Triage=[TRIAGE_REASONING]))
    case = Engine(client=client).run(PRODUCTIVITY, Checkpoint(action="confirm"))
    assert case.route["ROUTE"] == "REASONING"
    assert rules.is_reasoning_route(case.route)
    assert {"Skeptic", "Researcher", "Writer", "Critic"} <= set(client.roles())


def test_the_web_ui_labels_the_boundary_and_shows_the_refusal(tmp_path):
    from tests.test_web import FakeManager, _ui, wait_for
    from web.store import Store

    store = Store(tmp_path / "store")
    manager = FakeManager(store)
    job = manager.start(GDP, {"mode": "fast", "checkpoint_policy": "never"})
    ui = wait_for(lambda: _ui(store, job.case_id, complete=True))
    assert ui["triage"]["route"] == "FACTUAL"
    assert ui["triage"]["is_factual"] is True
    assert ui["refusal"]["text"] == rules.EPISTEMIC_REFUSAL
    assert ui["answer"]["sections"][0]["id"] == "refusal"
    assert ui["llm_calls"] == []



# --- Step 2: formal logic anchoring ------------------------------------------------------------------


def test_the_distiller_is_told_to_anchor_the_chain_in_formal_logic():
    client, _ = run()
    system = system_of(client, "ResearchDistill")
    assert "LOGICAL_SYLLOGISM" in system and "PSEUDO_CODE" in system
    assert "Premise 1:" in system


def test_the_syllogism_is_passed_to_the_writer_and_binds_the_prose():
    client, _ = run()
    writer_user = user_of(client, "Writer")
    writer_system = system_of(client, "Writer")
    assert "Premise 2: no metric, baseline, or comparison group exists" in writer_user
    assert "Do not add causal links that are not in the syllogism" in writer_system
    assert "LOGIC ANCHOR (syllogism)" in writer_user


def test_a_missing_anchor_is_marked_and_carried_as_a_hypothesis():
    client, case = run(Researcher=[dict(RESEARCHER_B, LOGICAL_SYLLOGISM=[], PSEUDO_CODE=[])])
    assert case.researcher["_logic"] == "missing"
    assert any("no formal logic anchor" in c for c in case.editor["CAUSAL_CORRECTIONS"])
    assert logic_anchor(case.researcher)["kind"] == "missing"


def test_the_pseudo_code_anchor_is_accepted_when_no_syllogism_fits():
    pseudo = dict(RESEARCHER_B, LOGICAL_SYLLOGISM=[], PSEUDO_CODE=["1. define the metric", "2. measure against baseline"])
    client, case = run(Researcher=[pseudo])
    assert case.researcher["_logic"] == "pseudo-code"
    assert "measure against baseline" in user_of(client, "Writer")


def test_the_logic_fields_are_schema_enforced():
    from epistemic.llm import _jsonize_schema

    schema = _jsonize_schema(schemas.RESEARCHER)
    assert schema["properties"]["LOGICAL_SYLLOGISM"] == {"type": "array", "items": {"type": "string"}}
    assert "LOGICAL_SYLLOGISM" not in schema["required"]
    fields = schemas.validate_fields({"SUMMARY": "x", "LOGICAL_SYLLOGISM": ["Premise 1: a"]}, schemas.RESEARCHER[:1] + [schemas.RESEARCHER[4]])
    assert fields["LOGICAL_SYLLOGISM"] == ["Premise 1: a"]


# --- Step 3: contrastive decoding + overthinking monitor ---------------------------------------------


def test_the_writers_negative_prompt_is_the_users_biased_raw_prompt():
    client, _ = run()
    assert client.negative_prompt("Writer") == [PRODUCTIVITY]
    assert client.negative_prompt("Skeptic") == [""]
    assert "I feel like our company" not in user_of(client, "Writer"), "the raw prompt stays out of the prompt text"


def test_the_negative_prompt_reaches_the_api_payload():
    seen = {}

    def fake_post(payload, on_delta=None):
        seen["payload"] = payload
        return "RESPONSE:\nstable answer", 40, 20

    client = LLMClient(use_schema=False)
    client.resolve_model = lambda: "strata"
    client._post = fake_post
    client.ask("Writer", "sys", "user", schemas.WRITER, negative_prompt=PRODUCTIVITY)
    assert seen["payload"]["negative_prompt"] == PRODUCTIVITY
    assert seen["payload"]["neg_scale"]


def test_an_aborted_stream_keeps_what_was_streamed():
    def fake_post(payload, on_delta=None):
        client.last_abort = None
        for piece in ["RESPONSE:\n", "stable paragraph.\n", "spiral: actually, wait,"]:
            try:
                on_delta(piece)
            except StreamAbort as abort:
                client.last_abort = abort
                break
        return "RESPONSE:\nstable paragraph.\n", 40, 20

    client = LLMClient(use_schema=False)
    client.resolve_model = lambda: "strata"

    def watch(text):
        if "spiral" in text:
            raise StreamAbort("overthinking monitor: test")

    client._post = fake_post
    result = client.ask("Writer", "sys", "user", schemas.WRITER, on_delta=watch)
    assert result.aborted is True
    assert result.record.aborted is True
    assert result.fields["RESPONSE"] == ["stable paragraph."]


def test_the_monitor_aborts_only_past_the_hesitation_budget():
    calm = OverthinkingMonitor(threshold=3, window_tokens=500, min_tokens=0)
    calm.feed("The metric is undefined. " * 40)
    assert calm.report()["hesitation_markers"] == 0 and not calm.aborted

    spiral = OverthinkingMonitor(threshold=2, window_tokens=100, min_tokens=0)
    text = "The evidence is weak. Actually, on second thought, however, in fact, on reflection, that said, we should switch. "
    with pytest.raises(StreamAbort) as boom:
        for i in range(0, len(text), 12):
            spiral.feed(text[i : i + 12])
    report = spiral.report()
    assert report["aborted"] and report["hesitation_markers"] > 2
    assert "on second thought" in boom.value.markers


def test_a_marker_split_across_deltas_is_counted_once():
    monitor = OverthinkingMonitor(threshold=100, window_tokens=500, min_tokens=0)
    for piece in ["on second ", "thought, and the ", "answer is stable."]:
        monitor.feed(piece)
    assert monitor.report()["hesitation_markers"] == 1


def test_stance_reversals_are_counted_as_overthinking():
    monitor = OverthinkingMonitor(threshold=100, window_tokens=4000, min_tokens=0, max_flips=1)
    text = "The model is supported. " * 40 + "The model is not supported. " * 40 + "The model is supported. " * 40
    with pytest.raises(StreamAbort):
        for i in range(0, len(text), 30):
            monitor.feed(text[i : i + 30])


def test_the_stream_is_cut_at_the_last_stable_paragraph():
    text = "First stable answer.\n\nSecond stable answer.\n\nActually, on second thought, everything changes."
    kept, dropped = truncate_at_last_stable(text)
    assert kept == "First stable answer.\n\nSecond stable answer."
    assert dropped == 1


def test_the_writer_stream_is_aborted_and_the_answer_is_truncated():
    client = SpiralLLM(script=script(Writer=[dict(WRITER_B, RESPONSE=WRITER_B["RESPONSE"] + [SPIRAL])]))
    case = Engine(client=client, overthinking_threshold=2, overthinking_window=200).run(
        PRODUCTIVITY, Checkpoint(action="confirm")
    )
    assert case.overthinking["aborted"] is True
    assert case.overthinking["hesitation_markers"] > 2
    assert "on second thought" not in case.final_response
    assert "Next steps" in case.final_response
    step = next(s for s in case.steps if s["name"] == "writer")
    assert step["data"]["overthinking_aborted"] is True


def test_the_stream_is_cut_midway_when_the_spiral_starts_in_the_thinking():
    class NoFinalLLM(FakeLLM):
        def ask(self, role, system, user, spec, max_tokens=None, on_delta=None, use_schema=None, **kw):
            result = super().ask(role, system, user, spec, max_tokens=max_tokens, use_schema=use_schema, **kw)
            if role == "Writer" and on_delta:
                spiral = "THINKING_PROCESS:\n" + "The model is supported. Actually, on second thought, " * 60
                for i in range(0, len(spiral), 40):
                    try:
                        on_delta(spiral[i : i + 40])
                    except StreamAbort:
                        result.aborted = True
                        break
            return result

    client = NoFinalLLM(script=script())
    case = Engine(client=client, overthinking_threshold=2, overthinking_window=200).run(
        PRODUCTIVITY, Checkpoint(action="confirm")
    )
    assert case.overthinking["aborted"] is True
    assert len(case.final_response) > 120
    assert "What can be responsibly concluded" in case.final_response
    assert "Next steps" not in case.final_response, "the paragraph after the last stable point is dropped"


def test_a_stream_aborted_before_any_answer_falls_back_to_the_packet():
    client = FakeLLM(script=script())
    engine = Engine(client=client)
    case = CaseFile(case_id="salvage", raw_prompt=PRODUCTIVITY, researcher=dict(RESEARCHER_LOGIC))
    engine._editor(case, {}, "evidence-bounded")
    text = engine._writer_salvage(case, "the stream was aborted in the deliberation")
    assert "aborted in the deliberation" in text
    assert "What can be responsibly concluded" in text
    assert "Premise 1:" in text
    assert "Next best steps" in text


# --- Steps 4 and 5: process reward critic + surgical rectification -----------------------------------


def test_the_critic_reads_the_numbered_steps_and_never_the_answer():
    client, case = run(Writer=[WRITER_STEPS])
    critic_user = user_of(client, "Critic")
    assert "Step 3: Draft the persuasion plan" in critic_user
    assert "the final text is deliberately not supplied" in critic_user
    assert "run the five-step persuasion campaign" not in critic_user
    assert "LOGIC ANCHOR" in critic_user


def test_the_deliberation_is_parsed_into_numbered_moves():
    steps = parse_steps(WRITER_STEPS["THINKING_PROCESS"])
    assert [s["step"] for s in steps] == [1, 2, 3, 4]
    assert steps[2]["text"].startswith("Draft the persuasion plan")


def test_unnumbered_deliberation_is_split_into_derived_moves():
    prose = (
        "The directive caps the register at cautious and the confidence at fifteen. " * 3
        + "The packet says no metric and no baseline exist, so the causal claim is untested. " * 3
        + "The plan must therefore start with measurement and only then propose a switch. " * 3
    )
    assert parse_steps("Step 1: Restate the directive.\nStep 2: Land the answer.") == [
        {"step": 1, "text": "Restate the directive."},
        {"step": 2, "text": "Land the answer."},
    ]
    derived = parse_steps(prose)
    assert len(derived) > 1, "a wall of prose is still audited move by move"
    assert all(d["derived"] and d["text"] for d in derived)
    assert [d["step"] for d in derived] == list(range(1, len(derived) + 1))
    assert len(derived) <= 12
    assert "no metric and no baseline" in " ".join(d["text"] for d in derived)


def test_a_derived_step_map_onto_a_real_paragraph():
    prose = (
        "The register is capped at cautious. " * 4
        + "The syllogism says the causal claim is untested. " * 4
        + "The answer must lead with the measurement sequence and keep the switch a hypothesis. " * 4
    )
    steps = parse_steps(prose)
    assert len(steps) > 2
    assert 0 < paragraph_index(2, len(steps), 4) < 4


def test_step_verdicts_parse_from_the_json_array_and_from_text():
    verdicts = schemas.parse_step_verdicts(
        [
            {"step": 1, "valid": True},
            {"step": 3, "valid": False, "reason": "Adopted the user's hidden axiom"},
        ]
    )
    assert verdicts[1] == {"step": 3, "valid": False, "reason": "Adopted the user's hidden axiom"}
    from_text = schemas.parse_step_verdicts("Step 1: VALID\nStep 3: INVALID - Adopted the user's hidden axiom")
    assert from_text == [
        {"step": 1, "valid": True, "reason": ""},
        {"step": 3, "valid": False, "reason": "Adopted the user's hidden axiom"},
    ]
    schema = __import__("epistemic.llm", fromlist=["_jsonize_schema"])._jsonize_schema([schemas.CRITIC[0]])
    assert schema["properties"]["STEPS"]["items"]["properties"]["valid"]["type"] == "boolean"


def test_an_invalid_step_triggers_a_surgical_repair_not_a_full_rewrite():
    client = FakeLLM(
        script=script(
            Writer=[WRITER_STEPS],
            Critic=[CRITIC_STEPS_FAIL, CRITIC_PASS],
            WriterRepair=[REPAIR_PARAGRAPH],
        )
    )
    case = Engine(client=client).run(PRODUCTIVITY, Checkpoint(action="confirm"))
    assert client.roles().count("Writer") == 1, "the whole answer is not regenerated"
    assert client.roles().count("WriterRepair") == 1
    repair_prompt = user_of(client, "WriterRepair")
    assert "Your Step 3 was flawed because: Adopted the user's hidden axiom" in repair_prompt
    assert "Rewrite ONLY the paragraph corresponding to Step 3" in repair_prompt
    assert "Do not rewrite Steps 1, 2, 4" in repair_prompt
    paras = paragraphs(case.final_response)
    assert paras[2] == REPAIR_PARAGRAPH["CORRECTED_PARAGRAPH"][0]
    assert paras[1] == WRITER_STEPS["RESPONSE"][1], "the paragraphs whose steps passed are word for word intact"
    assert paras[3] == WRITER_STEPS["RESPONSE"][3]
    assert "convince the CEO" not in case.final_response
    assert case.repairs[0]["step"] == 3 and case.repairs[0]["paragraph"] == 3
    assert case.repairs[0]["before"] == WRITER_STEPS["RESPONSE"][2]
    critic_step = next(s for s in case.steps if s["name"] == "critic")
    assert critic_step["data"]["steps_failed"] == 1 and critic_step["data"]["repairs"] == 1


def test_the_critic_step_verdicts_are_recorded_in_the_trace():
    client = FakeLLM(script=script(Writer=[WRITER_STEPS], Critic=[CRITIC_STEPS_FAIL], WriterRepair=[REPAIR_PARAGRAPH]))
    case = Engine(client=client).run(PRODUCTIVITY, Checkpoint(action="confirm"))
    verdicts = case.critic["STEPS"]
    assert [v["valid"] for v in verdicts] == [True, True, False, True]
    panel = trace_panel(case)
    assert "STEP VERDICTS:" in panel and "Step 3: INVALID" in panel
    assert "ADAPTIVE RECTIFICATION" in panel
    from web import uidata

    ui = uidata.build_ui(case)
    assert ui["trace"]["critic"]["steps_failed"] == 1
    assert ui["trace"]["repairs"][0]["step"] == 3
    assert any(w["type"] == "reasoning-repair" for w in ui["warnings"])


def test_the_repair_is_spliced_into_the_body_beneath_the_warning_banner():
    client = FakeLLM(script=script(Writer=[WRITER_STEPS], Critic=[CRITIC_STEPS_FAIL], WriterRepair=[REPAIR_PARAGRAPH]))
    case = Engine(client=client).run(PRODUCTIVITY, Checkpoint(action="confirm"))
    assert case.final_response.startswith("Termination warning:")
    assert case.final_response.count("Termination warning") == 1


def test_a_full_rewrite_is_used_when_surgical_repair_is_disabled():
    client = FakeLLM(
        script=script(Writer=[WRITER_STEPS, WRITER_STEPS], Critic=[CRITIC_STEPS_FAIL, CRITIC_PASS], WriterRepair=[REPAIR_PARAGRAPH])
    )
    engine = Engine(client=client, use_step_repair=False)
    case = engine.run(PRODUCTIVITY, Checkpoint(action="confirm"))
    assert client.roles().count("Writer") == 2
    assert "WriterRepair" not in client.roles()
    assert case.repairs == []


def test_all_steps_valid_means_no_repair_and_a_clean_pass():
    report = dict(CRITIC_PASS, STEPS=[{"step": 1, "valid": True}, {"step": 2, "valid": True}])
    client = FakeLLM(script=script(Writer=[WRITER_STEPS], Critic=[report], WriterRepair=[REPAIR_PARAGRAPH]))
    case = Engine(client=client).run(PRODUCTIVITY, Checkpoint(action="confirm"))
    assert case.critic["RESULT"] == "PASS"
    assert case.repairs == []
    assert client.roles().count("Writer") == 1


def test_the_paragraph_map_falls_back_to_proportional_indexing():
    assert paragraph_index(3, 4, 4) == 2
    assert paragraph_index(3, 6, 3) == 1
    assert paragraph_index(1, 1, 1) == 0


def test_new_case_fields_round_trip(tmp_path):
    client = FakeLLM(script=script(Writer=[WRITER_STEPS], Critic=[CRITIC_STEPS_FAIL], WriterRepair=[REPAIR_PARAGRAPH]))
    case = Engine(client=client).run(PRODUCTIVITY, Checkpoint(action="confirm"))
    restored = CaseFile.from_dict(json.loads(case.save(tmp_path).read_text(encoding="utf-8")))
    assert restored.repairs[0]["step"] == 3
    assert restored.overthinking["enabled"] is True
    assert restored.critic["STEPS"][2]["valid"] is False
