import pytest

from oee.jsonx import coerce, extract_json, normalise_dialectic, parse_and_repair, validate
from oee.prompts import PASS1_SCHEMA, PASS2_SCHEMA


def test_extract_plain_and_fenced():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('prose\n```json\n{"a": 2}\n```\nmore') == {"a": 2}
    assert extract_json('Sure! {"a": 3} hope that helps') == {"a": 3}
    assert extract_json('{"a": {"b": [1, 2]}} tail') == {"a": {"b": [1, 2]}}


def test_extract_rejects_garbage():
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_coerce_fills_required_and_clamps():
    notes = []
    obj = parse_and_repair('{"chat_reply": "hi"}', PASS2_SCHEMA, notes)
    assert obj["chat_reply"] == "hi"
    assert obj["confidence_tier"] in PASS2_SCHEMA["properties"]["confidence_tier"]["enum"]
    assert isinstance(obj["basis"], list)
    assert obj["unknowns"] == []
    assert any("required" in n for n in notes)


def test_coerce_enum_and_maxitems():
    notes = []
    raw = ('{"chat_reply": "x", "confidence_tier": "definitely sure", "basis": "parametric",'
           ' "unknowns": ["a","b","c","d"], "assumptions_used": "one; two",'
           ' "dialectic": {"thesis": "the claim", "antithesis": "however, the field record",'
           ' "synthesis": "the result is lab-bounded"}, "bottom_line": "lab only"}')
    obj = parse_and_repair(raw, PASS2_SCHEMA, notes)
    assert obj["confidence_tier"] == "Conceptually Open"  # x_default, not enum[0]
    assert obj["basis"] == ["parametric"]
    assert len(obj["unknowns"]) == 3
    assert obj["assumptions_used"] == ["one", "two"]
    assert any("enum" in n for n in notes)
    assert not validate(obj, PASS2_SCHEMA)


def test_validate_reports_paths():
    errors = validate({"chat_reply": "x", "confidence_tier": "nope", "basis": []}, PASS2_SCHEMA)
    assert errors
    assert any("confidence_tier" in e for e in errors)


def test_coerce_wraps_non_object():
    notes = []
    out = coerce(["a"], PASS1_SCHEMA, notes)
    assert out["sanitized_request"] == ""
    assert out["intent_preserved"] is True  # absent is not "the intent was lost"
    assert out["bias_removed"] == []
    assert any("missing required field" in n for n in notes)


def test_nested_sidecar_is_flattened_upward():
    notes = []
    raw = ('{"chat_reply": "The reference claims X [AUTHOR_CLAIM] [R1-C1].", '
           '"metadata": {"confidence_tier": "Highly Contested", '
           '"basis": ["reference"], "unknowns": ["adoption friction"]}}')
    obj = parse_and_repair(raw, PASS2_SCHEMA, notes)
    assert obj["confidence_tier"] == "Highly Contested"
    assert obj["basis"] == ["reference"]
    assert obj["unknowns"] == ["adoption friction"]
    assert any("flattened nested object 'metadata'" in n for n in notes)


def test_non_enum_basis_values_are_dropped_not_invented():
    notes = []
    raw = '{"chat_reply": "x", "confidence_tier": "Highly Contested", "basis": "No citations available."}'
    obj = parse_and_repair(raw, PASS2_SCHEMA, notes)
    assert obj["basis"] == []
    assert any("dropped non-enum values" in n for n in notes)


def test_real_model_key_drift_is_aliased():
    notes = []
    raw = ('{"sanitized_prompt": "Does the provided reference support the claim that AI '
           'will cause catastrophic harm?", '
           '"biases_removed": ["leading framing: isn\'t it obvious", "absolutist: everything"], '
           '"preserved_intent": true}')
    obj = parse_and_repair(raw, PASS1_SCHEMA, notes)
    assert obj["sanitized_request"].startswith("Does the provided reference")
    assert obj["bias_removed"] == ["leading framing: isn't it obvious", "absolutist: everything"]
    assert obj["intent_preserved"] is True
    assert any("aliased model key" in n for n in notes)
    assert not validate(obj, PASS1_SCHEMA)


def test_pass2_key_drift_is_aliased():
    notes = []
    raw = ('{"reply": "The reference claims X [AUTHOR_CLAIM] [R1-C1].", '
           '"tier": "Highly Contested", "grounding": ["reference", "parametric"], '
           '"references": [{"id": "R1-C1", "text": "the claim quoted verbatim"}], '
           '"what_would_change_this": ["primary data"], "assumptions": ["single source"],'
           '"dialectic": {"thesis": "the lab result stands", "antithesis": "however the logbook",'
           ' "synthesis": "lab-bounded only"}, "bottom_line": "partly"}')
    obj = parse_and_repair(raw, PASS2_SCHEMA, notes)
    assert obj["chat_reply"].startswith("The reference claims")
    assert obj["confidence_tier"] == "Highly Contested"
    assert obj["basis"] == ["reference", "parametric"]
    assert obj["citations"] == [{"chunk_id": "R1-C1",
                                 "quote": "the claim quoted verbatim", "claim": ""}]
    assert obj["what_would_change"] == ["primary data"]
    assert obj["assumptions_used"] == ["single source"]
    assert not validate(obj, PASS2_SCHEMA)


def test_aliasing_never_invents_values():
    notes = []
    obj = parse_and_repair('{"totally_unrelated": 5}', PASS1_SCHEMA, notes)
    assert obj["sanitized_request"] == ""
    assert obj["bias_removed"] == []
    assert obj.get("totally_unrelated") == 5


def test_citations_emitted_as_id_prefixed_strings_are_recovered():
    notes = []
    raw = ('{"chat_reply": "x [R1-C1]", "confidence_tier": "Highly Contested", '
           '"basis": ["reference"], "provenance": '
           '["R1-C1: the 1998 study measured drift at 85C", "[R1-S1] logbook notes"]} '
           )
    obj = parse_and_repair(raw, PASS2_SCHEMA, notes)
    assert obj["citations"] == [
        {"chunk_id": "R1-C1", "quote": "the 1998 study measured drift at 85C", "claim": ""},
        {"chunk_id": "R1-S1", "quote": "logbook notes", "claim": ""},
    ]
    assert any("aliased model key 'provenance'" in n for n in notes)


def test_open_questions_alias_to_unknowns():
    notes = []
    raw = ('{"chat_reply": "x", "confidence_tier": "Conceptually Open", '
           '"basis": ["parametric"], "open_questions": ["replication?"]} ')
    obj = parse_and_repair(raw, PASS2_SCHEMA, notes)
    assert obj["unknowns"] == ["replication?"]


def test_citation_without_id_is_dropped_with_a_note():
    notes = []
    raw = '{"chat_reply": "x", "confidence_tier": "Conceptually Open", "basis": [], "citations": ["no id here"]}'
    obj = parse_and_repair(raw, PASS2_SCHEMA, notes)
    assert obj["citations"] == []
    assert any("no recoverable chunk id" in n for n in notes)


def test_citation_container_with_id_list_is_recovered():
    notes = []
    raw = ('{"chat_reply": "x [R1-C1]", "confidence_tier": "Highly Contested", '
           '"basis": ["reference"], "provenance": {"reference_used": ["R1-C1", "R1-C2"], '
           '"grounding_note": "all claims tied to the provided chunks"}}')
    obj = parse_and_repair(raw, PASS2_SCHEMA, notes)
    assert [c["chunk_id"] for c in obj["citations"]] == ["R1-C1", "R1-C2"]
    assert obj["citations"][0]["quote"] == ""


def test_normalise_dialectic_defaults_warrant_on_prose():
    notes: list[str] = []
    d = normalise_dialectic("Thesis: A. Antithesis: B. Synthesis: C.", notes)
    assert d["thesis"] and d["antithesis"] and d["synthesis"]
    assert d["warrant"] == ""


def test_normalise_dialectic_defaults_warrant_on_dict():
    d = normalise_dialectic({"thesis": "A", "antithesis": "B", "synthesis": "C"}, [])
    assert d["warrant"] == ""
    d2 = normalise_dialectic({"thesis": "A", "antithesis": "B", "synthesis": "C",
                              "warrant": "W"}, [])
    assert d2["warrant"] == "W"


FULL = ('{"question_decomposition": ["q1"], "vocabulary_contract": [], '
        '"chat_reply": "The analysis text is long and detailed [FACT].", '
        '"dialectic": {"thesis": "T", "antithesis": "A", "synthesis": "S", '
        '"warrant": "W"}, "confidence_tier": "Highly Contested", '
        '"basis": ["inference"], "citations": [], "bottom_line": "Verdict."}')


def test_balanced_json_is_parsed_normally():
    assert extract_json(FULL)["bottom_line"] == "Verdict."


def test_truncated_mid_string_recovers_early_fields():
    frag = FULL[: FULL.index("detailed") + 15]  # cut inside chat_reply string
    rec = extract_json(frag)
    assert rec["question_decomposition"] == ["q1"]
    assert rec["chat_reply"].startswith("The analysis text is long")
    assert "bottom_line" not in rec


def test_truncated_mid_dialectic_keeps_chat_reply_and_thesis():
    frag = FULL[: FULL.index('"synthesis"') - 1]  # cut inside antithesis value
    rec = extract_json(frag)
    assert rec["chat_reply"].startswith("The analysis")
    assert rec["dialectic"]["thesis"] == "T"
    assert "synthesis" not in rec["dialectic"]


def test_truncated_after_dangling_key_is_recoverable():
    frag = FULL[: FULL.index("citations")] + '"what_would_change":'
    rec = extract_json(frag)
    assert rec["chat_reply"].startswith("The analysis")
    assert "what_would_change" not in rec


def test_non_json_still_raises():
    with pytest.raises(ValueError):
        extract_json("no json here at all")


def test_presupposition_audit_normalises_drifted_keys():
    notes: list[str] = []
    raw = '{"presuppositions": [{"premise": "the AI can self-assess", '
    raw += '"status": "load-bearing", "if_false": "the engine must intervene"}]}'
    obj = parse_and_repair(raw, PASS2_SCHEMA, notes)
    pa = obj["presupposition_audit"]
    assert pa[0]["presupposition"] == "the AI can self-assess"
    assert pa[0]["verdict"] == "load-bearing"
    assert pa[0]["counterfactual"] == "the engine must intervene"
