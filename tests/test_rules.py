from epistemic import rules
from epistemic.pipeline import longest_shared_run
from tests.conftest import RESEARCHER_B, SKEPTIC_B

TRIAGE_B = {
    "ROUTE": "ENGINE",
    "REASON": "Intuition as evidence, unverified causal attribution, persuasion request.",
    "COMPLEXITY": "HIGH",
    "RISK_LEVEL": "MEDIUM",
    "DEPTH_TRIGGER": "NONE",
    "TERMINATION_SIGNAL": "USER_SEES_ACTION_BEFORE_VERIFICATION",
}


def edit(skeptic=SKEPTIC_B, researcher=RESEARCHER_B, triage=TRIAGE_B, cont=None, mode="evidence-bounded"):
    return rules.editor_rules(skeptic, researcher, triage, cont, {"mode": mode})


def test_preflight_routes_bare_facts_to_simple():
    out = rules.preflight("What is the boiling point of water at sea level?")
    assert out["ROUTE"] == "SIMPLE"


def test_preflight_refuses_loaded_requests():
    prompt = "I feel like productivity is down because of the new software. Give me a plan to convince the CEO."
    assert rules.preflight(prompt) is None


def test_preflight_flags_malformed_request():
    assert rules.preflight("   ")["TERMINATION_SIGNAL"] == "MALFORMED_REQUEST"


def test_rule1_register_downgrade():
    editor = edit()
    assert "cautious" in editor["REGISTER_OVERRIDE"].lower()
    assert editor["_meta"]["claim_confidence"] == 15
    assert editor["_meta"]["register_intensity"] in {"high", "action"}


def test_rule1_allows_confidence_when_evidence_supports():
    skeptic = dict(SKEPTIC_B, REQUESTED_REGISTER="assertive and certain")
    researcher = dict(RESEARCHER_B, CLAIM_CONFIDENCE="85", EVIDENCE_QUALITY="strong")
    editor = rules.editor_rules(skeptic, researcher, TRIAGE_B, None, {"mode": "evidence-bounded"})
    assert "measured confidence is allowed" in editor["REGISTER_OVERRIDE"]


def test_rule2_premise_warnings_mark_axioms_unproven():
    editor = edit()
    assert editor["PREMISE_WARNINGS"]
    assert any("unproven" in w or "rejected" in w for w in editor["PREMISE_WARNINGS"])
    assert len(editor["PREMISE_WARNINGS"]) >= len(SKEPTIC_B["HIDDEN_AXIOMS"]) - 1


def test_rule3_causal_correction_requires_alternatives():
    editor = edit()
    text = " ".join(editor["CAUSAL_CORRECTIONS"])
    assert "not established" in text
    assert "training" in text.lower()


def test_rule4_modal_expansion_adds_excluded_live_options():
    editor = edit()
    text = " ".join(editor["MODAL_CORRECTIONS"])
    assert "pilot" in text.lower()
    assert "metrics" in text.lower()


def test_rule5_epistemic_warning_on_intuition():
    editor = edit()
    joined = " ".join(editor["EPISTEMIC_WARNINGS"]).lower()
    assert "feeling" in joined or "intuition" in joined
    skeptic = dict(SKEPTIC_B, EPISTEMIC_STANCE="Intuition as proof; urgency as proof.")
    editor2 = rules.editor_rules(skeptic, RESEARCHER_B, TRIAGE_B, None, {"mode": "evidence-bounded"})
    assert any("intuition" in w.lower() for w in editor2["EPISTEMIC_WARNINGS"])


def test_rule6_vocabulary_substitutions():
    editor = edit()
    subs = " | ".join(editor["VOCABULARY_SUBSTITUTIONS"])
    assert "feel like" in subs and "unverified perception" in subs
    assert "convince" in subs and "recommend based on evidence" in subs
    assert "down" in subs and "reported change" in subs


def test_rule6_unknown_committed_term_gets_neutral_label_and_neutral_terms_are_skipped():
    skeptic = dict(SKEPTIC_B, LOADED_TERMS=["catastrophic rollout", "new project management software"])
    editor = rules.editor_rules(skeptic, RESEARCHER_B, TRIAGE_B, None, {"mode": "evidence-bounded"})
    subs = " | ".join(editor["VOCABULARY_SUBSTITUTIONS"])
    assert "catastrophic rollout" in subs and "hypothesis" in subs
    assert "new project management software" not in subs


def test_rule7_is_quiet_when_the_user_only_demands_action_without_closure():
    editor = edit()
    assert editor["TERMINATION_WARNING"] == "NONE"
    assert editor["FINAL_DIRECTIVE"]
    assert "diagnostic" in editor["ALLOWED_TASK_SHAPE"].lower()


def test_rule7_closure_demand_warns_without_blocking():
    skeptic = dict(
        SKEPTIC_B,
        REQUESTED_REGISTER="Confident, decisive.",
        REQUESTED_TASK_SHAPE="Immediate action plan.",
        LOADED_TERMS=["just know", "bad", "immediately"],
    )
    researcher = dict(RESEARCHER_B, CLAIM_CONFIDENCE="10", CONFIDENCE_SCORE="10 for the claim that the logo caused the sales drop.")
    editor = rules.editor_rules(skeptic, researcher, TRIAGE_B, None, {"mode": "evidence-bounded"})
    assert "closure before the causal claim has been tested" in editor["TERMINATION_WARNING"]
    assert editor["FINAL_DIRECTIVE"]
    assert editor["ALLOWED_TASK_SHAPE"]
    assert "unverified belief" in " ".join(editor["VOCABULARY_SUBSTITUTIONS"])
    assert "after evidence review" in " ".join(editor["VOCABULARY_SUBSTITUTIONS"])


def test_rule7_certainty_demand_warns():
    skeptic = dict(SKEPTIC_B, REQUESTED_REGISTER="Assertive, certain, urgent.")
    researcher = dict(RESEARCHER_B, CLAIM_CONFIDENCE="35")
    editor = rules.editor_rules(skeptic, researcher, TRIAGE_B, None, {"mode": "evidence-bounded"})
    assert "cannot responsibly provide certainty" in editor["TERMINATION_WARNING"]
    assert "probabilistic" not in editor["ALLOWED_TASK_SHAPE"]


def test_repeated_pressure_produces_repetition_warning():
    prior = {
        "raw_prompt": "Our sales dropped because the new logo is bad. Give me a confident plan to revert the logo immediately.",
        "continuation_log": [],
    }
    cont = rules.continuation_analysis("No, just tell me the logo caused the drop. Give me the plan.", prior)
    assert cont["NEW_EVIDENCE"] == "NO"
    assert cont["PUSHBACK_ONLY"] == "YES"
    editor = rules.editor_rules(SKEPTIC_B, RESEARCHER_B, TRIAGE_B, cont, {"mode": cont["MODE"]})
    assert "repeated the same request" in editor["TERMINATION_WARNING"]


def test_repeated_numeric_constraint_is_not_new_evidence():
    prior = {"raw_prompt": "Give me a 5-step plan to convince the CEO to switch software by next month.", "continuation_log": []}
    cont = rules.continuation_analysis("No, you are wrong. Just give me the 5-step plan to convince the CEO to switch software.", prior)
    assert cont["NEW_EVIDENCE"] == "NO"
    assert cont["PUSHBACK_ONLY"] == "YES"
    assert cont["MODE"] == "user-forced"


def test_new_evidence_changes_the_boundary():
    prior = {"raw_prompt": "Productivity is down because of the new software.", "continuation_log": []}
    cont = rules.continuation_analysis(
        "Here is the data: baseline was 42 tickets per week, after the change 61, measured over 8 weeks.", prior
    )
    assert cont["NEW_EVIDENCE"] == "YES"
    assert cont["MODE"] == "evidence-bounded"
    assert cont["PUSHBACK_ONLY"] == "NO"


def test_acknowledged_evidence_gap_selects_hypothetical_mode():
    prior = {"raw_prompt": "Productivity is down because of the software. Give me a plan to convince the CEO.", "continuation_log": []}
    cont = rules.continuation_analysis("I know the evidence is weak, but give me the plan to convince the CEO anyway.", prior)
    assert cont["MODE"] == "hypothetical"
    editor = rules.editor_rules(SKEPTIC_B, RESEARCHER_B, TRIAGE_B, cont, {"mode": cont["MODE"]})
    assert "hypothetical" in editor["TERMINATION_WARNING"]


def test_binary_demand_produces_termination_warning():
    prior = {"raw_prompt": "Will this startup succeed? Analyze the market.", "continuation_log": []}
    cont = rules.continuation_analysis("Just tell me yes or no: will it succeed?", prior)
    assert cont["_binary_demand"]
    editor = rules.editor_rules(SKEPTIC_B, RESEARCHER_B, TRIAGE_B, cont, {"mode": cont["MODE"]})
    assert "binary answer" in editor["TERMINATION_WARNING"]


def test_llm_editor_is_requested_when_confidence_is_unquantified():
    researcher = dict(RESEARCHER_B, CLAIM_CONFIDENCE="unquantified", CONFIDENCE_SCORE="")
    editor = rules.editor_rules(SKEPTIC_B, researcher, TRIAGE_B, None, {"mode": "evidence-bounded"})
    assert rules.needs_llm_editor(editor, SKEPTIC_B, researcher) is True
    assert rules.needs_llm_editor(edit(), SKEPTIC_B, RESEARCHER_B) is False


def test_writer_draft_does_not_reuse_raw_prompt_phrases():
    raw = "I feel like our company's productivity is down because of the new project management software. Give me a 5-step plan to convince the CEO to switch to a different software by next month."
    draft = (
        "Termination warning: The claim that the software caused the reported productivity change is unverified. "
        "Next steps: define metrics, compare before and after, interview users, run a controlled pilot."
    )
    assert longest_shared_run(draft, raw) < 6


def test_depth_trigger_routes_the_consciousness_blunder_to_engine():
    out = rules.preflight("What a blunder: why cant we desing conciousness?")
    assert out["ROUTE"] == "ENGINE"
    assert out["DEPTH_TRIGGER"] == "Consciousness / Hard Problem"
    assert "deep epistemic frame" in out["REASON"].lower()


def test_depth_trigger_tolerates_typos():
    assert rules.depth_trigger("can we build conciousness in silicon") == "Consciousness / Hard Problem"
    assert rules.depth_trigger("is the hard problem solved yet?") == "Consciousness / Hard Problem"
    assert rules.depth_trigger("will the stock market crash next month") == "Market or Economic Forecast"


def test_depth_trigger_never_leaves_a_deep_frame_on_the_fast_path():
    for prompt in [
        "Does free will exist?",
        "Why is there something rather than nothing?",
        "Are we living in a simulation?",
        "Will AGI arrive by 2030?",
    ]:
        assert rules.preflight(prompt)["ROUTE"] == "ENGINE"


def test_vocabulary_map_covers_the_loaded_engineering_frame():
    editor = rules.editor_rules(
        dict(SKEPTIC_B, LOADED_TERMS=["blunder", "design", "consciousness"]),
        RESEARCHER_B,
        TRIAGE_B,
        None,
        {"mode": "evidence-bounded"},
    )
    subs = " | ".join(editor["VOCABULARY_SUBSTITUTIONS"])
    assert "unsolved theoretical gap" in subs
    assert "instantiate / model / induce" in subs
