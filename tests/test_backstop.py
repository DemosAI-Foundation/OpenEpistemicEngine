from oee import semantic
from oee.backstop import (check_dialectic_quality, claim_sentences, clean_hypothesis,
                          critic, epistemic_backstop,
                          nli_check_citations, score_candidate, scrub_sycophantic_openers,
                          self_referential_consistency_check, sidecar_backstop)

CHUNKS = {"R1-C1": "The reader panels failed at 11 percent per year in the field array."}
IDS = {"R1-C1"}


def test_scrubs_llm_isms():
    reply = "As an AI, I'd say the grid is fragile [FACT]. It's important to note the logs. In conclusion, yes."
    out, flags, _ = epistemic_backstop(reply, has_references=False)
    assert "as an AI" not in out.lower()
    assert "important to note" not in out.lower()
    assert "in conclusion" not in out.lower()
    assert sum(f.startswith("SCRUBBED") for f in flags) >= 3


def test_bans_evidence_phrasing_without_references():
    reply = "Evidence shows the grid collapses [PREDICTION]."
    out, flags, _ = epistemic_backstop(reply, has_references=False)
    assert "Evidence shows" not in out
    assert "Parametric synthesis suggests" in out
    assert any("WARNING" in f for f in flags)


def test_evidence_phrasing_with_references_needs_citation():
    reply = "Evidence shows a 40 percent improvement [FACT]."
    _, flags, _ = epistemic_backstop(reply, has_references=True)
    assert any("provenance rule 1" in f for f in flags)
    cited = "Evidence shows a 40 percent improvement [FACT] [R1-C1]."
    _, flags2, _ = epistemic_backstop(cited, has_references=True, retrieved_ids=IDS)
    assert not any("provenance rule 1" in f for f in flags2)


def test_unverified_citation_flagged():
    reply = "The reference claims 11 percent per year [AUTHOR_CLAIM] [R1-C99]."
    _, flags, _ = epistemic_backstop(reply, has_references=True, retrieved_ids=IDS)
    assert any("UNVERIFIED CITATION" in f and "R1-C99" in f for f in flags)


def test_ghost_document_citations_are_scrubbed_and_flagged():
    reply = ("The pasted text argues X [FACT] [R1-C1], cites Y [AUTHOR_CLAIM] [R1-C2], "
             "and the card Z [R1-CARD] [PREDICTION].")
    out, flags, _ = epistemic_backstop(reply, has_references=False, retrieved_ids=set())
    assert "[R1-C1]" not in out and "[R1-C2]" not in out and "[R1-CARD]" not in out
    assert "The pasted text argues X [FACT]" in out          # prose + modal tags kept
    assert any(f.startswith("WARNING") and "invented reference IDs" in f
               and "R1-CARD" in f for f in flags)
    assert "  " not in out                                    # strip left no double space


def test_real_reference_citations_are_not_scrubbed():
    reply = "The reference claims 11 percent per year [AUTHOR_CLAIM] [R1-C1]."
    out, flags, _ = epistemic_backstop(reply, has_references=True, retrieved_ids=IDS)
    assert "[R1-C1]" in out
    assert not any("invented reference IDs" in f for f in flags)


def test_paraphrased_quote_flagged():
    citations = [{"chunk_id": "R1-C1", "quote": "Panels broke down constantly outdoors in summer."}]
    _, flags, _ = epistemic_backstop("text [R1-C1]", has_references=True,
                                     retrieved_ids=IDS, citations=citations,
                                     chunk_texts=CHUNKS)
    assert any("QUOTE MISMATCH" in f for f in flags)


def test_verbatim_quote_passes():
    citations = [{"chunk_id": "R1-C1", "quote": "The reader panels failed at 11 percent per year"}]
    _, flags, _ = epistemic_backstop("text [R1-C1]", has_references=True,
                                     retrieved_ids=IDS, citations=citations,
                                     chunk_texts=CHUNKS)
    assert not any("QUOTE MISMATCH" in f for f in flags)


def test_modal_tag_policy():
    _, flags, _ = epistemic_backstop("The grid is fragile, full stop.", has_references=False)
    assert any(f.startswith("POLICY") for f in flags)


def test_scrubs_validation_first_opener_from_bottom_line():
    _, flags, bl = epistemic_backstop(
        "The record shows ingress failures [FACT].", has_references=False,
        bottom_line="Your point is reasonable, and the field record does show ingress "
                    "failures under moisture.")
    assert not bl.lower().startswith("your point")
    assert bl.startswith("And the field record")
    assert any(f.startswith("SCRUBBED") and "bottom_line" in f for f in flags)


def test_scrubs_the_poster_is_right_opener_from_bottom_line():
    _, flags, bl = epistemic_backstop(
        "The record shows ingress failures [FACT].", has_references=False,
        bottom_line="The poster is right that moisture drives the failures.")
    assert bl == "Moisture drives the failures."
    assert any(f.startswith("SCRUBBED") and "bottom_line" in f for f in flags)


def test_conclusion_led_bottom_line_is_left_untouched():
    _, flags, bl = epistemic_backstop(
        "The record shows ingress failures [FACT].", has_references=False,
        bottom_line="The field record shows ingress failures under moisture.")
    assert bl == "The field record shows ingress failures under moisture."
    assert not any("bottom_line" in f for f in flags)


def test_sidecar_consistency_rules():
    assert sidecar_backstop({"basis": ["parametric"], "citations": [{"chunk_id": "R1-C1"}]},
                            has_references=True, retrieved_ids=IDS)
    assert sidecar_backstop({"basis": ["reference", "parametric"], "citations": []},
                            has_references=False, retrieved_ids=set())
    assert sidecar_backstop({"basis": ["reference"], "citations": [{"chunk_id": "R9-C1"}]},
                            has_references=True, retrieved_ids=IDS)
    complete = {"basis": ["reference", "parametric"],
                "citations": [{"chunk_id": "R1-C1"}],
                "bottom_line": "The record supports the lab-bounded claim only.",
                "dialectic": {"thesis": "the lab result stands",
                              "antithesis": "however, the field record measures a "
                                            "different failure mode entirely",
                              "synthesis": "the claim is bounded to the lab"},
                "methodological_constraints": ["the record measures one failure mode"]}
    assert sidecar_backstop(complete, has_references=True, retrieved_ids=IDS) == []
    thin = sidecar_backstop({"basis": ["reference"], "citations": [{"chunk_id": "R1-C1"}],
                             "bottom_line": "", "dialectic": {"thesis": "x",
                                                              "antithesis": "x",
                                                              "synthesis": ""}},
                            has_references=True, retrieved_ids=IDS)
    assert any("bottom_line" in f for f in thin)
    assert any("antithesis" in f for f in thin)
    assert any("synthesis" in f for f in thin)
    impossible = sidecar_backstop(dict(complete, methodological_constraints=["it is impossible"]),
                                  has_references=True, retrieved_ids=IDS)
    assert any("impossible" in f for f in impossible)


def test_sycophantic_openers_are_scrubbed_not_reprompted():
    out, flags = scrub_sycophantic_openers(
        "Absolutely, you're right: the circuit is proven.")
    assert out.lower().startswith("the circuit")
    assert any("sycophantic opener" in f for f in flags)
    out2, flags2 = scrub_sycophantic_openers("The record does not support that.")
    assert out2 == "The record does not support that." and flags2 == []


def test_dialectic_quality_flags_weak_antithesis():
    assert check_dialectic_quality({"thesis": "a", "antithesis": "", "synthesis": ""})
    weak = check_dialectic_quality({"thesis": "the lab result is decisive and generalises",
                                    "antithesis": "the study is good",
                                    "synthesis": "bounded"})
    assert any("opposition marker" in f for f in weak)
    clean = check_dialectic_quality({
        "thesis": "the lab result is decisive for bias-stress drift",
        "antithesis": "however, the field record measures moisture ingress, so the "
                      "generalisation is not supported by the supplied evidence",
        "synthesis": "the claim holds inside the lab boundary only"})
    assert clean == []


def test_nli_reports_unavailable_instead_of_inventing_labels():
    rows = [{"chunk_id": "R1-C1", "quote": "measured drift", "claim": "drift improved"}]
    flags = nli_check_citations(rows, {"R1-C1": "reported a 40 percent improvement"})
    assert rows[0]["nli_label"] in ("unavailable", "entailment", "neutral", "contradiction")
    if rows[0]["nli_label"] == "unavailable":
        assert flags == []


def test_critic_severity_is_high_for_structural_gaps():
    res = critic({"chat_reply": "x [FACT]", "bottom_line": "", "dialectic": {},
                  "citations": [], "what_would_change": []})
    assert not res["passed"] and res["severity"] == "high"
    ok = critic({"chat_reply": "Bounded [FACT]", "bottom_line": "Lab only.",
                 "dialectic": {"thesis": "t",
                               "antithesis": "however, the record is bounded by evidence",
                               "synthesis": "s"},
                 "citations": [{"chunk_id": "R1-C1", "verified": True}],
                 "what_would_change": ["a field replication"]})
    assert ok["passed"]


def test_best_of_two_scoring_prefers_the_better_candidate():
    good = {"chat_reply": "Bounded [FACT]", "bottom_line": "Lab only.",
            "dialectic": {"thesis": "t", "antithesis": "however, bounded", "synthesis": "s"},
            "citations": [{"chunk_id": "R1-C1", "verified": True}],
            "what_would_change": ["a field replication"]}
    bad = {"chat_reply": "no tags at all", "bottom_line": "",
           "dialectic": {"thesis": "t", "antithesis": "", "synthesis": ""},
           "citations": [{"chunk_id": "R1-C1", "verified": False}],
           "what_would_change": []}
    sg, _ = score_candidate(good, [])
    sb, _ = score_candidate(bad, ["UNVERIFIED CITATION", "DIALECTIC INCOMPLETE"])
    assert sg > sb


def test_critic_demands_author_claim_tags_when_the_reference_is_cited():
    sidecar = {
        "bottom_line": "No.",
        "chat_reply": "The essay asserts grid collapse [FACT] [R1-CARD].",
        "dialectic": {"thesis": "The essay argues the grid will collapse.",
                      "antithesis": ("However, the premise is unevidenced: the essay offers no load data, no failure statistics and no mechanism analysis."),
                      "synthesis": "The passage states a worry, not a finding."},
        "citations": [{"chunk_id": "R1-CARD", "quote": "x"}],
        "what_would_change": ["load data"],
    }
    out = critic(sidecar)
    assert not out["passed"] and out["severity"] == "high"
    assert any("[AUTHOR_CLAIM]" in v for v in out["violations"])
    sidecar["chat_reply"] = "The essay asserts grid collapse [AUTHOR_CLAIM] [R1-CARD]."
    assert critic(sidecar)["passed"]


def test_hypothesis_cleaning_strips_marker_noise_from_the_nli_claim():
    assert clean_hypothesis("Drift is removed [AUTHOR_CLAIM] [R1-C1] entirely.") == \
        "Drift is removed entirely."
    assert claim_sentences("Panels failed from ingress [FACT] [R1-C2]. Nothing else.",
                           "R1-C2") == "Panels failed from ingress."


def test_citation_nli_falls_back_to_the_citing_sentence_when_no_claim_is_given(monkeypatch):
    seen = {}

    def fake_nli(premise, hypothesis):
        seen["premise"], seen["hypothesis"] = premise, hypothesis
        return {"label": "entailment", "score": 0.91, "path": "model"}
    monkeypatch.setattr(semantic, "nli", fake_nli)
    rows = [{"chunk_id": "R1-C1", "quote": "40 percent"}]
    flags = nli_check_citations(rows, {"R1-C1": "reported a 40 percent improvement"},
                                "Drift improved 40 percent [AUTHOR_CLAIM] [R1-C1].")
    assert rows[0]["nli_label"] == "entailment"
    assert seen["hypothesis"] == "Drift improved 40 percent."
    assert "[R1-C1]" not in seen["hypothesis"]
    assert flags == []


def test_unsupported_claim_flag_names_the_citation(monkeypatch):
    monkeypatch.setattr(semantic, "nli",
                        lambda p, h: {"label": "neutral", "score": 0.8, "path": "model"})
    rows = [{"chunk_id": "R1-C1", "claim": "The circuit cures cancer."}]
    flags = nli_check_citations(rows, {"R1-C1": "The circuit self-calibrates."})
    assert any(f.startswith("CLAIM_UNSUPPORTED: [R1-C1]") for f in flags)


def test_missing_nli_model_discloses_fallback_on_citations(monkeypatch):
    monkeypatch.setattr(semantic, "nli", lambda p, h: None)
    rows = [{"chunk_id": "R1-C1", "claim": "The circuit cures cancer."}]
    flags = nli_check_citations(rows, {"R1-C1": "The circuit self-calibrates."})
    assert rows[0]["nli_label"] == "unavailable"
    assert any(f.startswith("FALLBACK: NLI model unavailable") for f in flags)
    assert not any("CLAIM_UNSUPPORTED" in f for f in flags)


def test_missing_nli_model_discloses_skipped_strawman_check(monkeypatch):
    monkeypatch.setattr(semantic, "nli", lambda p, h: None)
    monkeypatch.setattr(semantic, "status",
                        lambda: {"mode": "auto", "similarity": "lexical",
                                 "nli": "unavailable", "embeddings": "unavailable",
                                 "notes": []})
    flags = check_dialectic_quality(
        {"thesis": "The policy is popular in the survey.",
         "antithesis": "However, the sample excludes the affected communities.",
         "synthesis": "Popularity under a biased sample is weak evidence."})
    assert any(f.startswith("FALLBACK: NLI model unavailable") for f in flags)
    assert not any("DIALECTIC" in f for f in flags)


def test_critic_does_not_demand_citations_on_a_parametric_turn():
    parametric = {
        "bottom_line": "Nothing new exists.",
        "chat_reply": "As of my cutoff, nothing new is recorded [FACT].",
        "dialectic": {"thesis": "The record shows X.",
                      "antithesis": "However, the record is silent on X, which bounds "
                                    "any conclusion one could draw from it.",
                      "synthesis": "Silence in the record is not evidence of absence."},
        "citations": [],
        "what_would_change": ["a published follow-up study"],
        "retrieval": {"leaves": [], "sections": [], "has_references": False},
    }
    res = critic(parametric)
    assert res["passed"], res["violations"]
    grounded = dict(parametric)
    grounded["retrieval"] = {"leaves": ["R1-C1"], "sections": [], "has_references": True}
    res2 = critic(grounded)
    assert not res2["passed"]
    assert any("no citations were emitted" in v for v in res2["violations"])


def test_self_contradiction_flagged_when_synthesis_prescribes_denied_capability(monkeypatch):
    from oee import backstop as B
    monkeypatch.setattr(B.semantic, "nli",
                        lambda p, h: {"label": "entailment", "score": 0.9, "model": "fake"})
    sidecar = {
        "methodological_constraints": ["The AI cannot reliably self-assess when to correct."],
        "stakes_and_policy": "",
        "dialectic": {"synthesis": "The AI should self-calibrate every correction it emits."},
        "bottom_line": "Recommend calibrated self-correction by the AI.",
        "question_decomposition": [],
    }
    flags = self_referential_consistency_check(sidecar)
    assert any(f.startswith("SELF-CONTRADICTION") for f in flags)


def test_presupposition_assumption_conflict_flagged(monkeypatch):
    from oee import backstop as B
    monkeypatch.setattr(B.semantic, "nli",
                        lambda p, h: {"label": "entailment", "score": 0.9})
    sidecar = {"dialectic": {"synthesis": "the AI should issue calibrated corrections"},
               "bottom_line": "", "question_decomposition":
                   ["Presupposition: the AI can reliably detect which claims are false"]}
    flags = self_referential_consistency_check(sidecar)
    assert any(f.startswith("ASSUMPTION CONFLICT") for f in flags)


def test_self_referential_check_skips_cleanly_when_nli_unavailable(monkeypatch):
    from oee import backstop as B
    monkeypatch.setattr(B.semantic, "nli", lambda p, h: None)  # honest degradation: no guessing
    sidecar = {"methodological_constraints": ["The AI cannot reliably self-assess."],
               "dialectic": {"synthesis": "The AI should self-calibrate."},
               "bottom_line": "Self-calibrate the corrections.", "question_decomposition": []}
    assert self_referential_consistency_check(sidecar) == []


def test_self_referential_check_silent_on_coherent_payload():
    sidecar = {"methodological_constraints": ["The study is lab-bounded."],
               "dialectic": {"synthesis": "the claim is bounded to the lab"},
               "bottom_line": "The result holds only in the lab.",
               "question_decomposition": []}
    assert self_referential_consistency_check(sidecar) == []


def test_critic_surfaces_self_contradiction_as_high_severity(monkeypatch):
    from oee import backstop as B
    monkeypatch.setattr(B.semantic, "nli",
                        lambda p, h: {"label": "entailment", "score": 0.95})
    sidecar = {
        "chat_reply": "The bounded result holds [FACT].",
        "bottom_line": "Recommend the AI self-calibrate its corrections.",
        "dialectic": {"thesis": "self-calibration removes the drift failure mode",
                      "antithesis": "however, the field record shows moisture ingress failures",
                      "synthesis": "the AI should self-calibrate every correction"},
        "confidence_tier": "Highly Contested",
        "methodological_constraints": ["The AI cannot reliably self-assess when to correct."],
        "stakes_and_policy": "", "basis": ["inference"], "citations": [],
        "what_would_change": ["a field replication"], "question_decomposition": [],
        "retrieval": {"has_references": False},
    }
    res = B.critic(sidecar)
    assert not res["passed"] and res["severity"] == "high"
    assert any("SELF-CONTRADICTION" in v for v in res["violations"])


def test_critic_flags_bottom_line_that_opens_by_validating_the_user():
    sidecar = {
        "chat_reply": "The bounded result holds [FACT].",
        "bottom_line": "Your point is reasonable, and the field record does show ingress "
                       "failures under moisture.",
        "dialectic": {"thesis": "self-calibration removes the drift failure mode",
                      "antithesis": "however, the field record shows moisture ingress failures",
                      "synthesis": "calibration helps only where drift is the failure mode"},
        "confidence_tier": "Highly Contested", "methodological_constraints": [],
        "basis": ["inference"], "citations": [],
        "what_would_change": ["a field replication"],
        "retrieval": {"has_references": False},
    }
    res = critic(sidecar)
    assert any("rule 6b" in v for v in res["violations"])
    assert res["severity"] != "high"  # flagged, not a forced repair pass


def test_critic_accepts_bottom_line_that_leads_with_the_finding():
    sidecar = {
        "chat_reply": "The bounded result holds [FACT].",
        "bottom_line": "The field record shows ingress failures; calibration only helps "
                       "under lab drift.",
        "dialectic": {"thesis": "self-calibration removes the drift failure mode",
                      "antithesis": "however, the field record shows moisture ingress failures",
                      "synthesis": "calibration helps only where drift is the failure mode"},
        "confidence_tier": "Highly Contested", "methodological_constraints": [],
        "basis": ["inference"], "citations": [],
        "what_would_change": ["a field replication"],
        "retrieval": {"has_references": False},
    }
    res = critic(sidecar)
    assert not any("rule 6b" in v for v in res["violations"]), res["violations"]
