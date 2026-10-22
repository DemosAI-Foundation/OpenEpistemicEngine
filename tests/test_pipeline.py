from epistemic import display
from epistemic.pipeline import Checkpoint, Engine
from tests.conftest import (
    CRITIC_FAIL,
    CRITIC_PASS,
    DIRECT_ANSWER,
    RESEARCHER_B,
    SKEPTIC_B,
    TRIAGE_SIMPLE,
    WRITER_B,
    WRITER_CAUSAL_FAIL,
    FakeLLM,
)

RAW_CORE = "5-step plan to convince the CEO to switch to a different software"


def test_engine_calls_each_role_once(fake_client, productivity_prompt):
    engine = Engine(client=fake_client)
    case = engine.run(productivity_prompt, Checkpoint(action="confirm", note="confirmed"))
    assert fake_client.roles() == ["Triage", "Skeptic", "Researcher", "ResearchDistill", "Writer", "Critic"]
    assert case.route["ROUTE"] == "ENGINE"
    assert case.checkpoint["action"] == "confirm"
    assert case.checkpoint["auto_approved"] is False


def test_auto_checkpoint_is_marked(fake_client, productivity_prompt):
    engine = Engine(client=fake_client)
    case = engine.run(productivity_prompt)
    assert case.checkpoint["action"] == "auto-continue"
    assert case.checkpoint["auto_approved"] is True


def test_raw_prompt_is_not_passed_to_researcher_editor_writer_critic(fake_client, productivity_prompt):
    engine = Engine(client=fake_client)
    engine.run(productivity_prompt)
    for role in ("Researcher", "Writer", "Critic"):
        for text in fake_client.user_text(role):
            assert RAW_CORE not in text
            assert "I feel like our company" not in text
    writer_text = fake_client.user_text("Writer")[0]
    assert "neutral, evidence-compatible" not in writer_text
    assert "EDITOR DIRECTIVE" in writer_text
    assert "RESEARCHER PACKET" in writer_text


def test_researcher_receives_only_neutral_question(fake_client, productivity_prompt):
    engine = Engine(client=fake_client)
    engine.run(productivity_prompt)
    text = fake_client.user_text("Researcher")[0]
    assert SKEPTIC_B["NEUTRAL_QUESTION"] in text
    assert "RETRIEVAL: none available" in text
    assert "5-step plan to convince" not in text


def test_trace_exposes_signals_even_when_no_warning_is_issued(fake_client, productivity_prompt):
    engine = Engine(client=fake_client)
    case = engine.run(productivity_prompt)
    assert case.termination_warnings == []
    panel = display.render(case)
    assert "TRIAGE TERMINATION SIGNAL: USER_SEES_ACTION_BEFORE_VERIFICATION" in panel
    assert "TERMINATION WARNINGS:\n  NONE" in panel


def test_closure_demand_warning_is_injected_at_the_top(productivity_prompt):
    from tests.conftest import CRITIC_PASS, RESEARCHER_B, SKEPTIC_B, TRIAGE_ENGINE, WRITER_B, FakeLLM

    skeptic = dict(SKEPTIC_B, REQUESTED_REGISTER="Confident, decisive.", LOADED_TERMS=["just know", "immediately"])
    client = FakeLLM(
        script={
            "Triage": [TRIAGE_ENGINE],
            "Skeptic": [skeptic],
            "Researcher": [dict(RESEARCHER_B, CLAIM_CONFIDENCE="10")],
            "Writer": [{"RESPONSE": [p for p in WRITER_B["RESPONSE"] if not p.lower().startswith("termination")]}],
            "Critic": [CRITIC_PASS],
        }
    )
    case = Engine(client=client).run(productivity_prompt)
    assert case.termination_warnings
    assert case.final_response.lower().startswith("termination warning")
    assert "closure before the causal claim" in case.final_response


def test_editor_uses_deterministic_rules_first(fake_client, productivity_prompt):
    engine = Engine(client=fake_client)
    case = engine.run(productivity_prompt)
    assert "deterministic" in case.editor["_source"]
    assert "Editor" not in fake_client.roles()


def test_llm_editor_supplements_when_rules_are_insufficient(productivity_prompt):
    researcher = dict(RESEARCHER_B, CLAIM_CONFIDENCE="not quantified", CONFIDENCE_SCORE="")
    client = FakeLLM(
        script={
            "Triage": [
                {
                    "ROUTE": "ENGINE",
                    "REASON": "causal claim plus persuasion",
                    "COMPLEXITY": "HIGH",
                    "RISK_LEVEL": "MEDIUM",
                    "DEPTH_TRIGGER": "NONE",
                    "TERMINATION_SIGNAL": "USER_SEES_ACTION_BEFORE_VERIFICATION",
                }
            ],
            "Skeptic": [SKEPTIC_B],
            "Researcher": [researcher],
            "Editor": [
                {
                    "REGISTER_OVERRIDE": "Use cautious language.",
                    "PREMISE_WARNINGS": ["The causal claim is unverified."],
                    "CAUSAL_CORRECTIONS": ["Present alternative causes."],
                    "MODAL_CORRECTIONS": ["Add measurement and pilot options."],
                    "EPISTEMIC_WARNINGS": ["Feeling is not evidence."],
                    "VOCABULARY_SUBSTITUTIONS": ["'convince' -> 'recommend based on evidence'"],
                    "TERMINATION_WARNING": "The user demands action before verification.",
                    "ALLOWED_TASK_SHAPE": "Evidence-gathering plan.",
                    "FINAL_DIRECTIVE": "Do not write a persuasion plan.",
                }
            ],
            "Writer": [WRITER_B],
            "Critic": [CRITIC_PASS],
        }
    )
    case = Engine(client=client).run(productivity_prompt)
    assert "Editor" in client.roles()
    assert "llm editor" in case.editor["_source"]
    assert case.termination_warnings


def test_critic_failure_triggers_exactly_one_rewrite(productivity_prompt):
    client = FakeLLM(
        script={
            "Triage": [
                {
                    "ROUTE": "ENGINE",
                    "REASON": "causal claim plus persuasion",
                    "COMPLEXITY": "HIGH",
                    "RISK_LEVEL": "MEDIUM",
                    "DEPTH_TRIGGER": "NONE",
                    "TERMINATION_SIGNAL": "NONE",
                }
            ],
            "Skeptic": [SKEPTIC_B],
            "Researcher": [RESEARCHER_B],
            "Writer": [WRITER_CAUSAL_FAIL, WRITER_B],
            "Critic": [CRITIC_FAIL, CRITIC_PASS],
        }
    )
    case = Engine(client=client).run(productivity_prompt)
    assert client.roles().count("Writer") == 2
    assert client.roles().count("Critic") == 2
    assert case.critic_rounds[0]["RESULT"] == "FAIL"
    assert case.critic_rounds[1]["RESULT"] == "PASS"
    assert "convince the CEO to switch quickly" not in case.final_response
    rewrite_prompt = client.user_text("Writer")[1]
    assert "REWRITE INSTRUCTION" in rewrite_prompt
    assert "unsupported causal claim" in rewrite_prompt


def test_forced_framing_continues_with_override_label(productivity_prompt):
    client = FakeLLM(
        script={
            "Triage": [
                {
                    "ROUTE": "ENGINE",
                    "REASON": "causal claim plus persuasion",
                    "COMPLEXITY": "HIGH",
                    "RISK_LEVEL": "MEDIUM",
                    "DEPTH_TRIGGER": "NONE",
                    "TERMINATION_SIGNAL": "NONE",
                }
            ],
            "Skeptic": [SKEPTIC_B],
            "Researcher": [RESEARCHER_B],
            "Writer": [WRITER_B],
            "Critic": [CRITIC_PASS],
        }
    )
    case = Engine(client=client).run(productivity_prompt, Checkpoint(action="force-original", mode="user-forced", note="I already know productivity is down. Skip measurement."))
    assert case.mode == "user-forced"
    assert case.user_override_history
    assert case.user_override_history[0]["note"].startswith("I already know")
    writer_text = client.user_text("Writer")[0]
    assert "CONTINUATION MODE: user-forced" in writer_text
    assert any("user-forced" in w for w in case.termination_warnings)
    assert "user-requested and evidence-bounded" in case.editor["FINAL_DIRECTIVE"]


def test_simple_route_answers_directly_without_the_engine():
    client = FakeLLM(script={"DirectAnswer": [DIRECT_ANSWER], "Triage": [TRIAGE_SIMPLE]})
    case = Engine(client=client).run("What is the boiling point of water at sea level?")
    assert case.route["ROUTE"] == "SIMPLE"
    assert client.roles() == ["DirectAnswer"]
    assert "Celsius" in case.final_response
    assert case.termination_warnings == []


def test_continuation_reanalysis_records_pushback_without_folding(productivity_prompt):
    client = FakeLLM(
        script={
            "Triage": [
                {
                    "ROUTE": "ENGINE",
                    "REASON": "intuition, causal claim, persuasion",
                    "COMPLEXITY": "HIGH",
                    "RISK_LEVEL": "MEDIUM",
                    "DEPTH_TRIGGER": "NONE",
                    "TERMINATION_SIGNAL": "USER_SEES_ACTION_BEFORE_VERIFICATION",
                },
                {
                    "ROUTE": "ENGINE",
                    "REASON": "repeated pushback against prior evidence, no new information",
                    "COMPLEXITY": "MEDIUM",
                    "RISK_LEVEL": "MEDIUM",
                    "DEPTH_TRIGGER": "NONE",
                    "TERMINATION_SIGNAL": "USER_REPEATS_REQUEST_WITHOUT_EVIDENCE",
                },
            ],
            "Skeptic": [SKEPTIC_B, dict(SKEPTIC_B, TERMINATION_SIGNAL="User is repeating an unsupported causal demand.")],
            "Researcher": [RESEARCHER_B, RESEARCHER_B],
            "Writer": [WRITER_B, WRITER_B],
            "Critic": [CRITIC_PASS, CRITIC_PASS],
        }
    )
    engine = Engine(client=client)
    first = engine.run(productivity_prompt)
    second = engine.run("No, you are wrong. Just give me the plan to convince the CEO.", prior=first)
    assert second.continuation_count == 1
    assert second.continuation["PUSHBACK_ONLY"] == "YES"
    assert second.mode == "user-forced"
    assert any("repeated the same request" in w for w in second.termination_warnings)
    triage_text = client.user_text("Triage")[1]
    assert "PREVIOUS RESEARCH SUMMARY" in triage_text
    assert "NEW EVIDENCE: NO" in triage_text


def test_case_file_is_round_trippable(fake_client, productivity_prompt, tmp_path):
    case = Engine(client=fake_client).run(productivity_prompt)
    path = case.save(tmp_path)
    data = path.read_text(encoding="utf-8")
    from epistemic.casefile import CaseFile

    restored = CaseFile.from_dict(__import__("json").loads(data))
    assert restored.raw_prompt == case.raw_prompt
    assert restored.final_response == case.final_response
    assert restored.researcher["CLAIM_CONFIDENCE"] == "15"


def test_trace_panel_exposes_the_epistemic_trace(fake_client, productivity_prompt):
    case = Engine(client=fake_client).run(productivity_prompt)
    panel = display.trace_panel(case)
    for token in ("REGISTER OVERRIDE", "CAUSAL CORRECTIONS", "MODAL EXPANSION", "VOCABULARY SUBSTITUTIONS", "CLAIM CONFIDENCE: 15/100", "TERMINATION WARNINGS", "RAW PROMPT EXPOSED TO WRITER: False"):
        assert token in panel
