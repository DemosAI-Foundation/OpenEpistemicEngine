import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oee import prompts as P


def test_schema_generates_bottom_line_last_and_confidence_after_analysis():
    keys = list(P.PASS2_SCHEMA["properties"])
    assert keys[:3] == ["question_decomposition", "vocabulary_contract", "chat_reply"]
    assert keys.index("confidence_tier") > keys.index("dialectic")
    assert keys.index("basis") > keys.index("confidence_tier")
    assert keys.index("citations") > keys.index("basis")
    assert keys.index("bottom_line") > keys.index("citations")
    assert "bottom_line" in P.PASS2_SCHEMA["required"]


def test_rule_16_structural_disanalogy_check_is_present_and_ordered_after_rule_15():
    s = P.PASS2_SYSTEM
    assert s.index("15. Fluency") < s.index("16. Structural Disanalogy Check")
    for probe in ("Base & Target", "Relational Isomorphism", "category error", "[CONSTRAINT]"):
        assert probe in s


def test_rule_17_presupposition_audit_is_present_and_ordered_after_rule_16():
    s = P.PASS2_SYSTEM
    assert s.index("16. Structural Disanalogy Check") < s.index("17. Presupposition Audit")
    for probe in ("presupposition", "question_decomposition", "[CONSTRAINT]",
                  "[PRESUPPOSITION]"):
        assert probe in s


def test_dialectic_schema_requires_warrant():
    dl = P.PASS2_SCHEMA["properties"]["dialectic"]
    assert "warrant" in dl["properties"]
    assert set(dl["required"]) == {"thesis", "antithesis", "synthesis", "warrant"}


def test_modal_tags_include_provenance_and_presupposition_tags():
    for tag in ("PRESUPPOSITION", "INFERRED ATTRIBUTION", "INFERRED VALUE", "AUTHOR_CLAIM"):
        assert tag in P.MODAL_TAGS


def test_rule_4_requires_verbatim_quote_for_attribution():
    s = P.PASS2_SYSTEM
    for probe in ("verbatim quote", "[INFERRED ATTRIBUTION]", "cannot anchor a counter-argument"):
        assert probe in s


def test_rule_7_steelman_limits_inferred_values():
    s = P.PASS2_SYSTEM
    for probe in ("[INFERRED VALUE]", "would the opponent recognize this as their view"):
        assert probe in s


def test_rule_16_answers_walton_critical_questions_not_stock_label():
    s = P.PASS2_SYSTEM
    for probe in ("Walton's Critical Questions", "relevant dissimilarities",
                  "base domain actually true", "available analogies", "[PRESUPPOSITION]"):
        assert probe in s


def test_rule_17_redirects_audit_to_presupposition_audit():
    s = P.PASS2_SYSTEM
    for probe in ("presupposition_audit", "supported / unsupported / contested",
                  "counterfactual", "Do NOT put these in `question_decomposition`",
                  "load-bearing"):
        assert probe in s


def test_presupposition_audit_field_in_schema():
    pa = P.PASS2_SCHEMA["properties"]["presupposition_audit"]
    assert pa["type"] == "array"
    assert pa["maxItems"] == 4
    props = pa["items"]["properties"]
    assert set(props) == {"presupposition", "verdict", "counterfactual"}
    assert set(pa["items"]["required"]) == {"presupposition", "verdict"}
    assert "presupposition_audit" in P.PASS2_SCHEMA["required"]
    qd = P.PASS2_SCHEMA["properties"]["question_decomposition"]
    assert "Do not put presupposition audits here" in qd["description"]


def test_simplify_bottom_prompt_is_8th_grade_dejargonized_and_not_robotic():
    s = P.SIMPLIFY_BOTTOM_SYSTEM
    for probe in ("8th-grade reading level", "common words and active voice",
                  "academic jargon and abstract nouns", "Keep the exact meaning",
                  "See Spot Run", "vary sentence length"):
        assert probe in s


def test_simplify_whole_prompt_is_8th_grade_non_agreeable_and_not_robotic():
    s = P.SIMPLIFY_WHOLE_SYSTEM
    for probe in ("8th grade reader", "common words and active voice",
                  "academic jargon and abstract nouns",
                  "caveats, uncertainty, citations, and disagreement",
                  "more agreeable", "Do not add new claims", "See Spot Run"):
        assert probe in s
    assert "9th-10th" not in s and "define it briefly" not in s


def test_simplify_prompts_share_the_epistemic_guard():
    for s in (P.SIMPLIFY_BOTTOM_SYSTEM, P.SIMPLIFY_WHOLE_SYSTEM):
        assert "[R#-C#] citations" in s
        assert "[AUTHOR_CLAIM]" in s
        assert "Output only the rewritten text." in s
