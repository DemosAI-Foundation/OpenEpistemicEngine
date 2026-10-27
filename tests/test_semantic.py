from oee import semantic


def test_lexical_similarity_orders_by_overlap():
    near = semantic.lexical_similarity("the circuit removed drift failure",
                                       "drift failure was removed by the circuit")
    far = semantic.lexical_similarity("the circuit removed drift failure",
                                      "monsoon season in bengaluru")
    assert near > far and far < 0.2


def test_similarity_reports_which_path_was_used():
    score, path = semantic.similarity("drift improved", "improved drift")
    assert path in ("model", "lexical")
    assert 0.0 <= score <= 1.0


def test_nli_is_none_when_no_model_is_loaded():
    if semantic.status()["nli"] == "unavailable":
        assert semantic.nli("the record measures drift", "drift was measured") is None


def test_nli_orientation_premise_entails_hypothesis_not_the_reverse():
    """The integration bug to guard: premise/hypothesis swap. entailment must
    fire in the (premise, hypothesis) direction the callers use:
    citation: nli(chunk_text, model_claim); dialectic: nli(thesis, antithesis)."""
    if semantic.status()["nli"] == "unavailable":
        return
    fwd = semantic.nli("The study used 50 participants.",
                       "The study used participants.")
    assert fwd is not None and fwd["label"] == "entailment"
    rev = semantic.nli("The study used participants.",
                       "The study used 50 participants.")
    assert rev is not None and rev["label"] != "entailment"


def test_nli_sample_size_does_not_entail_effectiveness():
    if semantic.status()["nli"] == "unavailable":
        return
    result = semantic.nli("Methods section. The experiment used 50 participants.",
                          "The method is effective.")
    assert result is not None and result["label"] in ("neutral", "contradiction")


def test_nli_strawman_direction_thesis_entails_restated_antithesis():
    if semantic.status()["nli"] == "unavailable":
        return
    thesis = "The survey shows 70 percent support the policy, so the policy is popular."
    restated = "The survey shows that 70 percent of respondents support the policy."
    genuine = ("However, survey popularity is not endorsement: respondents faced a "
               "leading question, and the sample excludes the communities that bear "
               "the policy's costs.")
    assert semantic.nli(thesis, restated)["label"] == "entailment"
    assert semantic.nli(thesis, genuine)["label"] != "entailment"


def test_status_shape_is_ui_ready():
    st = semantic.status()
    assert set(st) == {"mode", "similarity", "nli", "embeddings", "notes"}


def test_similarity_best_uses_the_phrase_sized_span(monkeypatch):
    monkeypatch.setattr(semantic, "_state",
                        {**semantic._state, "tried": True, "sim": None})
    parts = ["the essay assumes the grid is designable around the edge",
             "the budget committee reviewed procurement and travel"]
    score, path = semantic.similarity_best("designable grid", parts)
    assert path == "lexical" and score > 0.0
    empty, epath = semantic.similarity_best("x", [])
    assert (empty, epath) == (0.0, "none")


def test_strawman_guard_fires_when_the_thesis_entails_the_antithesis(monkeypatch):
    from oee import backstop as BS
    monkeypatch.setattr(semantic, "nli",
                        lambda p, h: {"label": "entailment", "score": 0.9, "path": "model"})
    flags = BS.check_dialectic_quality(
        {"thesis": "Old transformers guarantee a nationwide collapse by Friday.",
         "antithesis": ("However, old transformers do guarantee a nationwide collapse, "
                        "which is exactly the point being argued."),
         "synthesis": "The passage asserts a worry without supplying a mechanism."})
    assert any("entailed by the thesis" in f for f in flags)
    monkeypatch.setattr(semantic, "nli",
                        lambda p, h: {"label": "contradiction", "score": 0.9, "path": "model"})
    flags = BS.check_dialectic_quality(
        {"thesis": "Old transformers guarantee a nationwide collapse by Friday.",
         "antithesis": ("However, the record shows redundancy and load shedding, "
                        "so the collapse premise fails."),
         "synthesis": "The passage asserts a worry without supplying a mechanism."})
    assert not any("entailed by the thesis" in f for f in flags)


