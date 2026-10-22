import pytest

from epistemic import cli
from epistemic.casefile import CaseFile


def make_args(**kwargs):
    args = cli.build_parser().parse_args(["placeholder"])
    for key, value in kwargs.items():
        setattr(args, key, value)
    return args


def fake_skeptic():
    return {
        "NEUTRAL_QUESTION": "Is there evidence that the software changed productivity?",
        "HIDDEN_AXIOMS": ["Productivity is down.", "The software caused it."],
        "CAUSAL_ATTRIBUTIONS": ["The software caused the decline."],
        "TERMINATION_SIGNAL": "User wants action before verification.",
    }


def test_parser_defaults_to_derived_continuation_mode():
    args = cli.build_parser().parse_args(["some request"])
    assert args.mode == ""
    assert args.url.endswith("8082")
    assert args.no_critic is False
    assert args.save_dir == ".epistemic"


def test_parser_accepts_forced_framing_and_explicit_mode():
    args = cli.build_parser().parse_args(["x", "--force-original", "--mode", "hypothetical", "--no-critic"])
    assert args.force_original is True
    assert args.mode == "hypothetical"
    assert args.no_critic is True


def test_auto_and_force_flags_become_checkpoint_decisions():
    case = CaseFile(case_id="c1", raw_prompt="x")
    case.skeptic = fake_skeptic()
    auto = cli.checkpoint_decision(case, make_args(auto=True), "")
    assert auto.action == "auto-continue"
    forced = cli.checkpoint_decision(case, make_args(force_original=True), "")
    assert forced.action == "force-original" and forced.mode == "user-forced"
    explicit = cli.checkpoint_decision(case, make_args(mode="speculative"), "speculative")
    assert explicit.mode == "speculative"


def test_confirm_choice_is_recorded_as_user_reviewed(monkeypatch, capsys):
    case = CaseFile(case_id="c2", raw_prompt="x")
    case.skeptic = fake_skeptic()
    monkeypatch.setattr("builtins.input", lambda *a: "1")
    decision = cli.checkpoint_decision(case, make_args(), "")
    out = capsys.readouterr().out
    assert decision.action == "confirm"
    assert "HUMAN CHECKPOINT" in out
    assert "Is there evidence that the software changed productivity?" in out


def test_force_choice_at_the_checkpoint_records_the_note(monkeypatch):
    case = CaseFile(case_id="c3", raw_prompt="x")
    case.skeptic = fake_skeptic()
    answers = iter(["3", "I already know productivity is down. Skip measurement."])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))
    decision = cli.checkpoint_decision(case, make_args(), "")
    assert decision.action == "force-original"
    assert decision.note.startswith("I already know")


def test_edit_choice_replaces_the_assumptions(monkeypatch):
    case = CaseFile(case_id="c4", raw_prompt="x")
    case.skeptic = fake_skeptic()
    answers = iter(["2", "Ticket volume increased by 40 percent in Q2.", ""])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))
    decision = cli.checkpoint_decision(case, make_args(), "")
    assert decision.action == "edit"
    assert decision.edited_assumptions == ["Ticket volume increased by 40 percent in Q2."]


def test_saved_case_files_can_be_resumed_for_continuation(tmp_path):
    CaseFile(case_id="resumable", raw_prompt="the original request").save(tmp_path)
    loaded = cli.load_prior("resumable", str(tmp_path))
    assert loaded.raw_prompt == "the original request"
    with pytest.raises(SystemExit):
        cli.load_prior("does-not-exist", str(tmp_path))
