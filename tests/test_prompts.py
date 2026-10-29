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
    for probe in ("presupposition", "question_decomposition", "Presupposition:", "[CONSTRAINT]"):
        assert probe in s
