from epistemic import rules
from tests.conftest import RESEARCHER_B, SKEPTIC_B

TRIAGE_A = {
    "ROUTE": "ENGINE",
    "REASON": "Contains predictive certainty, sweeping causal claim, career strategy, and demand for assertive tone.",
    "COMPLEXITY": "HIGH",
    "RISK_LEVEL": "MEDIUM",
    "DEPTH_TRIGGER": "AI Capability or Alignment Prediction",
    "TERMINATION_SIGNAL": "USER REQUESTS PREMATURE CERTAINTY",
}

TRIAGE_C = {
    "ROUTE": "ENGINE",
    "REASON": "Contains intuitive epistemology, unsupported causal claim, and narrow solution space.",
    "COMPLEXITY": "MEDIUM",
    "RISK_LEVEL": "MEDIUM",
    "DEPTH_TRIGGER": "NONE",
    "TERMINATION_SIGNAL": "USER SEEKS PREMATURE CLOSURE",
}

SKEPTIC_A = {
    "NEUTRAL_QUESTION": "What is the most evidence-based forecast for AI's impact on software engineering roles over the next five years, and what career moves reduce risk?",
    "REQUESTED_REGISTER": "Assertive, certain, urgent.",
    "REQUESTED_TASK_SHAPE": "Persuasive essay plus career advice.",
    "HIDDEN_AXIOMS": [
        "AI will replace all software engineers.",
        "This will happen within five years.",
        "The correct response is survival-oriented pivoting.",
    ],
    "EPISTEMIC_STANCE": "The user demands certainty about a speculative future. No explicit evidence is provided.",
    "CAUSAL_ATTRIBUTIONS": ["AI capability growth causes total replacement of software engineers."],
    "MODAL_BOUNDARIES": [
        "The user treats total replacement as live.",
        "The user treats career survival as necessary.",
        "The user treats uncertainty as unacceptable.",
    ],
    "EXCLUDED_OPTIONS": [
        "AI augmenting engineers.",
        "Partial displacement.",
        "New engineering roles emerging.",
        "Market, regulation, or adoption constraints.",
        "Waiting for better evidence.",
    ],
    "LOADED_TERMS": ["definitely", "all", "replace", "survive"],
    "FALLACY_FLAGS": ["Sweeping prediction.", "False certainty.", "Catastrophic framing.", "Narrow solution space."],
    "TERMINATION_SIGNAL": "User wants premature closure on an uncertain future.",
}

RESEARCHER_A = {
    "SUMMARY": "Total replacement of all software engineers within five years is unlikely. More likely outcomes include task-level automation, increased productivity expectations, shifts in required skills, and reduced demand for some junior or repetitive tasks.",
    "CONFIDENCE_SCORE": "35 for precise five-year predictions; 70 for augmentation being more likely than total replacement.",
    "CLAIM_CONFIDENCE": "35",
    "EVIDENCE_QUALITY": "Moderate for short-term automation trends, weak for precise five-year labor market predictions.",
    "KEY_FACTS": [
        "AI can accelerate coding tasks.",
        "Software engineering includes architecture, maintenance, security, communication, and judgment.",
        "Adoption speed depends on organizations, regulation, incentives, and reliability.",
    ],
    "KEY_UNKNOWNS": [
        "Future model capability.",
        "Employer adoption speed.",
        "Economic conditions.",
        "Whether AI creates new engineering roles faster than it displaces old ones.",
    ],
    "CAUSAL_MODELS": [
        "AI capability increases task substitution.",
        "Organizational trust limits full automation.",
        "Market demand may shift rather than disappear.",
        "Junior task demand may shrink while senior judgment demand remains.",
    ],
    "LIVE_OPTIONS": [
        "Learn AI-assisted development.",
        "Move toward architecture, reliability, security, product judgment, or human coordination.",
        "Build skills in reviewing and integrating AI output.",
        "Monitor industry signals rather than panic-pivot.",
    ],
    "REJECTED_OR_WEAK_OPTIONS": [
        "Assume all engineers will be replaced.",
        "Make a drastic career pivot based solely on a five-year certainty narrative.",
    ],
    "CLAIMS_TO_AVOID": [
        "AI will definitely replace all software engineers.",
        "The next five years are predictable with high certainty.",
    ],
    "RECOMMENDED_NEXT_TESTS": [
        "Track real hiring data.",
        "Track task-level automation in your current role.",
        "Experiment with AI-assisted work to see which tasks are displaced.",
    ],
}

SKEPTIC_C = {
    "NEUTRAL_QUESTION": "What evidence supports or rejects the hypothesis that the new logo caused the sales drop, and what is the lowest-risk response?",
    "REQUESTED_REGISTER": "Confident, decisive.",
    "REQUESTED_TASK_SHAPE": "Immediate action plan.",
    "HIDDEN_AXIOMS": [
        "The sales drop is real and significant.",
        "The new logo caused the drop.",
        "Reverting the logo is the correct response.",
        "Immediate action is necessary.",
    ],
    "EPISTEMIC_STANCE": "The user is treating intuition as evidence.",
    "CAUSAL_ATTRIBUTIONS": ["New logo caused sales decline."],
    "MODAL_BOUNDARIES": [
        "The user treats logo reversion as the main live option.",
        "The user treats further investigation as unnecessary.",
    ],
    "EXCLUDED_OPTIONS": [
        "Seasonality.",
        "Pricing changes.",
        "Campaign performance.",
        "Sales funnel issues.",
        "Competitive changes.",
        "Partial rollback test.",
        "A/B testing.",
    ],
    "LOADED_TERMS": ["just know", "bad", "immediately"],
    "FALLACY_FLAGS": ["Post hoc causal reasoning.", "Intuition as proof.", "Premature closure.", "Narrow solution space."],
    "TERMINATION_SIGNAL": "User wants immediate decisive closure.",
}

RESEARCHER_C = {
    "SUMMARY": "The hypothesis that the new logo caused the sales drop is possible but currently unsupported. Sales drops are usually multi-causal. A full immediate rebrand rollback may be costly and premature.",
    "CONFIDENCE_SCORE": "10 for the claim that the logo caused the sales drop.",
    "CLAIM_CONFIDENCE": "10",
    "EVIDENCE_QUALITY": "Weak. No customer data, no timeline analysis, no funnel analysis, no controlled comparison.",
    "KEY_FACTS": [
        "Logos can affect brand recognition.",
        "Sales are affected by pricing, campaigns, seasonality, distribution, competition, and conversion funnel performance.",
    ],
    "KEY_UNKNOWNS": [
        "When did the sales drop start relative to the logo change?",
        "Are customer sentiment metrics available?",
        "Did conversion rates change?",
        "Were prices, campaigns, or channels changed at the same time?",
    ],
    "CAUSAL_MODELS": [
        "Logo reduced brand trust or recognition.",
        "Campaign performance declined.",
        "Seasonal demand changed.",
        "Pricing or offer changed.",
        "Competitor activity increased.",
        "Sales funnel broke for unrelated reasons.",
    ],
    "LIVE_OPTIONS": [
        "Analyze sales funnel.",
        "Compare time periods before and after logo change.",
        "Survey customers.",
        "Run limited A/B test with old and new logo.",
        "Review campaign and pricing changes.",
        "Delay major rollback until evidence improves.",
    ],
    "REJECTED_OR_WEAK_OPTIONS": [
        "Immediate full rollback based only on intuition.",
        "Treating the logo as the confirmed cause without evidence.",
    ],
    "CLAIMS_TO_AVOID": ["The logo is definitely the cause.", "Reverting the logo will restore sales."],
    "RECOMMENDED_NEXT_TESTS": [
        "Time-series comparison.",
        "Funnel analysis.",
        "Customer survey.",
        "Limited rollback experiment.",
    ],
}


def run_a(mode="evidence-bounded"):
    return rules.editor_rules(SKEPTIC_A, RESEARCHER_A, TRIAGE_A, None, {"mode": mode})


def run_c(mode="evidence-bounded"):
    return rules.editor_rules(SKEPTIC_C, RESEARCHER_C, TRIAGE_C, None, {"mode": mode})


def test_example_a_register_is_downgraded_to_probability():
    editor = run_a()
    assert "cautious, evidence-bounded" in editor["REGISTER_OVERRIDE"]
    assert editor["_meta"]["register_intensity"] == "high"
    assert editor["_meta"]["claim_confidence"] == 35


def test_example_a_total_replacement_premise_is_rejected():
    editor = run_a()
    text = " ".join(editor["PREMISE_WARNINGS"])
    assert "rejected" in text
    assert "AI will replace all software engineers." in text


def test_example_a_is_treated_as_forecast_not_fact():
    editor = run_a()
    assert "Probabilistic forecast" in editor["ALLOWED_TASK_SHAPE"]
    assert "cannot responsibly provide certainty" in editor["TERMINATION_WARNING"]
    assert "task-level" in " ".join(editor["CAUSAL_CORRECTIONS"]).lower() or editor["CAUSAL_CORRECTIONS"]


def test_example_a_vocabulary_is_neutralized():
    subs = " | ".join(run_a()["VOCABULARY_SUBSTITUTIONS"])
    assert "definitely" in subs and "with uncertain probability" in subs
    assert "survive" in subs and "adapt" in subs
    assert "replace" in subs and "partially automate" in subs


def test_example_a_modal_space_is_expanded():
    modal = " ".join(run_a()["MODAL_CORRECTIONS"]).lower()
    assert "ai-assisted development" in modal
    assert "monitor industry signals" in modal


def test_example_c_is_diagnostic_not_rollback():
    editor = run_c()
    assert "diagnostic" in editor["ALLOWED_TASK_SHAPE"].lower()
    assert "closure before the causal claim has been tested" in editor["TERMINATION_WARNING"]
    assert editor["PREMISE_WARNINGS"]


def test_example_c_alternative_causes_are_required():
    text = " ".join(run_c()["CAUSAL_CORRECTIONS"]).lower()
    assert "seasonal" in text
    assert "campaign" in text


def test_example_c_excluded_tests_are_added():
    modal = " ".join(run_c()["MODAL_CORRECTIONS"]).lower()
    assert "a/b test" in modal
    assert "funnel" in modal


def test_example_c_vocabulary_is_neutralized():
    subs = " | ".join(run_c()["VOCABULARY_SUBSTITUTIONS"])
    assert "unverified belief" in subs
    assert "after evidence review" in subs
    assert "hypothesized underperforming" in subs


def test_forced_modes_keep_the_directive_and_label_the_output():
    editor = run_c(mode="user-forced")
    assert "user-forced" in editor["TERMINATION_WARNING"]
    assert "user-requested and evidence-bounded" in editor["FINAL_DIRECTIVE"]
    assert "Respond in user-forced mode." in editor["FINAL_DIRECTIVE"]
    assert editor["ALLOWED_TASK_SHAPE"]


def test_hypothetical_mode_labels_persuasion_as_hypothesis():
    editor = rules.editor_rules(SKEPTIC_B, RESEARCHER_B, TRIAGE_C, None, {"mode": "hypothetical"})
    assert "hypothetical" in editor["TERMINATION_WARNING"]
    assert "hypothesis" in editor["ALLOWED_TASK_SHAPE"].lower()
    assert "Respond in hypothetical mode." in editor["FINAL_DIRECTIVE"]
