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
