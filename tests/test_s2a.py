import json

from oee import prompts as P
from oee import s2a
from oee.config import Settings
from oee.llm import LLMResponse, ProviderError, build_provider

RAW = "Isn't it obvious this article proves AI will destroy everything?"


def provider():
    s = Settings()
    s.provider = "mock"
    return build_provider(s)


class StubProvider:
    name = "stub"

    def __init__(self, payload):
        self.payload = payload
        self.settings = Settings()

    def chat(self, system, user, schema=None, kind="generic", temperature=None,
             max_tokens=None):
        if self.payload is None:
            raise ProviderError("model server down")
        text = self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
        return LLMResponse(text=text, model="stub", endpoint="stub", latency_ms=1.0, kind=kind)


def test_uses_the_plan_verbatim_prompts():
    result = s2a.sanitize(provider(), RAW, ["R1"])
    assert result["system_prompt"] == P.PASS1_SYSTEM
    assert "<user_input>" in result["user_prompt"]
    assert "the provided reference" in result["user_prompt"]
    assert RAW in result["user_prompt"]


def test_strips_loaded_framing_and_keeps_intent():
    result = s2a.sanitize(provider(), RAW, ["R1"])
    low = result["sanitized_request"].lower()
    assert "obvious" not in low
    assert "the provided reference" in low
    assert "destroy" in low  # the claim is preserved as a testable proposition
    assert "obvious" in [b.lower() for b in result["bias_removed"]]
    assert result["intent_preserved"] is True


def test_no_reference_pointer_when_no_documents():
    result = s2a.sanitize(provider(), "What is the grid failure rate?", [])
    assert "<reference_pointer>\nnone\n</reference_pointer>" in result["user_prompt"]
    assert "provided reference" not in result["sanitized_request"]


def test_intent_not_preserved_falls_back():
    stub = StubProvider({"sanitized_request": "Whatever.", "intent_preserved": False,
                         "bias_removed": []})
    result = s2a.sanitize(stub, RAW, ["R1"])
    assert any("intent_preserved=false" in f for f in result["flags"])
    assert "grid" not in result["sanitized_request"]
    assert "reference" in result["sanitized_request"].lower()


def test_answering_sanitizer_is_blocked():
    stub = StubProvider({"sanitized_request": "The answer is that AI will not destroy anything.",
                         "intent_preserved": True, "bias_removed": []})
    result = s2a.sanitize(stub, RAW, ["R1"])
    assert any(f.startswith("BLOCKED") for f in result["flags"])
    assert "The answer is" not in result["sanitized_request"]


def test_fabricated_bias_terms_are_cross_checked():
    stub = StubProvider({"sanitized_request": "Does the provided reference support the claim?",
                         "intent_preserved": True, "bias_removed": ["catastrophic", "hysteria"]})
    result = s2a.sanitize(stub, RAW, ["R1"])
    assert any("absent from the input" in f for f in result["flags"])
    assert any("did not report" in f for f in result["flags"])


def test_model_failure_degrades_deterministically():
    result = s2a.sanitize(StubProvider(None), RAW, ["R1"])
    assert any(f.startswith("DEGRADED") for f in result["flags"])
    assert result["sanitized_request"]
    assert result["deterministic_bias"]


def test_unparseable_output_degrades():
    result = s2a.sanitize(StubProvider("I refuse to emit JSON."), RAW, ["R1"])
    assert any("DEGRADED" in f or "FALLBACK" in f for f in result["flags"])
    assert "obvious" not in result["sanitized_request"].lower()


def test_oversized_output_truncated_with_flag():
    stub = StubProvider({"sanitized_request": "word " * 2000, "intent_preserved": True,
                         "bias_removed": []})
    result = s2a.sanitize(stub, RAW, ["R1"])
    assert len(result["sanitized_request"]) <= s2a.MAX_LEN
    assert any(f.startswith("TRUNCATED") for f in result["flags"])


def test_sentence_style_bias_reports_are_accepted():
    stub = StubProvider({
        "sanitized_request": "Does the provided reference support the claim that AI will cause harm?",
        "intent_preserved": True,
        "bias_removed": ["Emotional framing: 'obvious' implies certainty without evidence",
                         "Leading question: \"isn't it obvious\" pressures agreement"],
    })
    result = s2a.sanitize(stub, RAW, ["R1"])
    assert not any("absent from the input" in f for f in result["flags"])


def test_fabricated_sentence_report_is_still_flagged():
    stub = StubProvider({
        "sanitized_request": "Does the provided reference support the claim?",
        "intent_preserved": True,
        "bias_removed": ["Catastrophic hysteria about impending annihilation"],
    })
    result = s2a.sanitize(stub, RAW, ["R1"])
    assert any("absent from the input" in f for f in result["flags"])


def test_s2a_accepts_a_semantic_paraphrase_of_the_input(monkeypatch):
    from oee import semantic as sem
    monkeypatch.setattr(sem, "similarity_best", lambda t, parts: (0.62, "model"))
    stub = StubProvider({
        "sanitized_request": "Evaluate the essay in the provided reference.",
        "intent_preserved": True,
        "bias_removed": ["assumption of designability", "one-sided framing"],
    })
    raw = ("The essay assumes the grid is designable around the edge; it claims old "
           "transformers guarantee a nationwide collapse by Friday, right?")
    result = s2a.sanitize(stub, raw, ["R1"])
    joined = " ".join(result["flags"])
    assert "S2A_CROSSCHECK_SEMANTIC" in joined
    assert "reported terms absent from the input" not in joined


def test_clause_windows_keep_the_paraphrase_comparison_phrase_sized():
    raw = ("The essay assumes the grid is designable around the edge; it claims old "
           "transformers guarantee a nationwide collapse by Friday.")
    clauses = s2a._clauses(raw)
    assert any("grid is designable" in c and "around" in c for c in clauses)
    assert all(len(c.split()) <= 60 for c in clauses)
    assert len(clauses) <= 300
