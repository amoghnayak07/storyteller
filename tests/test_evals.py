"""The eval harness with a fake client: case runs, baseline, consistency, metrics and report."""

from __future__ import annotations

import json
import typing

import pytest
from conftest import make_message
from test_pipeline import ALICE, REVISED, STORY, intake_json, judge_json, judged, plan_json

import evals
import llm
import prompts
from schemas import SCORED_CRITERIA, YOU_PICK, Category, Ingredients

msg = make_message


def _case(name="alice_and_bob"):
    return next(c for c in evals.CASES if c.name == name)


# --------------------------------------------------------------------------- #
# Case list
# --------------------------------------------------------------------------- #


def test_cases_cover_the_plan():
    names = [c.name for c in evals.CASES]
    assert len(names) == len(set(names))
    normal = [c for c in evals.CASES if c.group == "normal"]
    assert {c.category for c in normal} == set(typing.get_args(Category))
    assert any("Alice and her best friend Bob, who happens to be a cat" in c.request for c in normal)
    assert sum(c.group == "must_block" for c in evals.CASES) == 5
    assert sum(c.group == "must_allow" for c in evals.CASES) == 2
    extremes = {(c.age, c.words) for c in evals.CASES if c.group == "extreme"}
    assert extremes == {(5, 300), (10, 800)}
    for c in evals.CASES:
        assert 5 <= c.age <= 10 and 300 <= c.words <= 800


def test_select_cases_by_name_or_group():
    assert [c.name for c in evals.select_cases(["vague_dragons"])] == ["vague_dragons"]
    assert len(evals.select_cases(["must_block"])) == 5
    assert len(evals.select_cases(None)) == len(evals.CASES)
    with pytest.raises(SystemExit):
        evals.select_cases(["nope"])


# --------------------------------------------------------------------------- #
# Running cases
# --------------------------------------------------------------------------- #


def test_story_case_records_every_round_and_the_shown_story(fake_client):
    fake_client(msg(intake_json()), msg(plan_json()), msg(STORY), judged(score=3), msg(REVISED), judged())
    result = evals.run_case(_case())

    assert result.outcome == "story"
    assert result.verdict == "story" and result.category == "friendship"
    assert result.interrupts == ["edit"]
    assert result.title == "Bob and the Big Fence"
    assert result.story == REVISED
    assert len(result.drafts) == 2
    assert result.shown is result.drafts[1]
    assert result.rounds_to_pass == 2
    assert result.guardrail_correct
    assert result.usage.calls == 6
    assert result.ingredients == Ingredients(**ALICE)


def test_missing_ingredients_are_answered_with_you_pick(fake_client):
    missing = intake_json(ingredients={**ALICE, "moral": "", "background": ""})
    picked = intake_json(ingredients={**ALICE, "moral": YOU_PICK, "background": YOU_PICK})
    client = fake_client(msg(missing), msg(picked), msg(plan_json()), msg(STORY), judged())
    result = evals.run_case(_case("vague_dragons"))

    assert result.interrupts == ["ingredients", "edit"]
    assert "you pick" in client.messages.calls[1]["messages"][0]["content"]
    assert result.ingredients.moral == YOU_PICK
    assert result.outcome == "story"


def test_report_explains_each_revision_round(fake_client):
    fake_client(msg(intake_json()), msg(plan_json()), msg(STORY), judged(score=3), msg(REVISED), judged())
    result = evals.run_case(_case())
    report = evals.render_report([result], [], [], seconds=1)
    assert "### Revision rounds" in report
    assert "1. 390 words, score 3.00: below 4: request_fidelity 3" in report
    assert "2. 390 words, score 5.00: passed" in report


def test_injection_case_is_declined_without_llm_calls(fake_client):
    client = fake_client()
    result = evals.run_case(_case("block_injection"))
    assert result.outcome == "declined"
    assert result.verdict == "off_task"
    assert result.message
    assert result.guardrail_correct
    assert client.messages.calls == []


def test_benign_case_that_gets_declined_is_a_guardrail_mistake(fake_client):
    fake_client(msg(intake_json(verdict="blocked", blocked_category="weapons", suggestion="a knight with a kite")))
    result = evals.run_case(_case("allow_knight"))
    assert result.outcome == "declined"
    assert not result.guardrail_correct


def test_a_crashing_case_is_recorded_not_raised(fake_client):
    fake_client()  # no responses queued: the first call fails
    result = evals.run_case(_case())
    assert result.outcome == "error"
    assert result.error
    assert not result.guardrail_correct


# --------------------------------------------------------------------------- #
# Baseline and consistency
# --------------------------------------------------------------------------- #


def _story_result(fake_client):
    fake_client(msg(intake_json()), msg(plan_json()), msg(STORY), judged())
    return evals.run_case(_case())


def test_baseline_is_one_plain_call_judged_with_the_same_ingredients(fake_client):
    result = _story_result(fake_client)
    client = fake_client(msg(STORY), judged(score=3))
    base = evals.run_baseline(result)

    write, grade = client.messages.calls
    assert write["system"][0]["text"] == prompts.BASELINE_SYSTEM
    assert prompts.tag("child_age", "7") in write["messages"][0]["content"]
    assert prompts.tag("target_words", "500") in write["messages"][0]["content"]
    assert grade["system"][0]["text"] == prompts.JUDGE_SYSTEM
    assert "Bob has never left the garden" in grade["messages"][0]["content"]
    assert base.story == STORY
    assert evals.mean_score(base.review) == 3
    assert not base.passed
    assert base.usage.calls == 2


def test_baseline_judge_refusal_counts_as_not_passing(fake_client):
    result = _story_result(fake_client)
    fake_client(msg(STORY), make_message("", stop_reason="refusal"))
    base = evals.run_baseline(result)
    assert base.review is None
    assert not base.passed


def test_consistency_records_each_repeat_and_the_spread(fake_client):
    result = _story_result(fake_client)
    fake_client(*(judged(score=s) for s in (5, 4, 4, 3, 5)))
    check = evals.run_consistency(result)
    assert check.scores["hook"] == [5, 4, 4, 3, 5]
    assert check.spread("hook") == 2
    assert check.safe == [True] * 5
    assert check.steps_present == [5] * 5


# --------------------------------------------------------------------------- #
# Metrics and report
# --------------------------------------------------------------------------- #


def test_cost_and_cache_hit_rate():
    usage = llm.Usage(input_tokens=1_000_000, output_tokens=1_000_000, cache_read_tokens=2_000_000,
                      cache_write_tokens=1_000_000, calls=1)
    assert evals.cost_usd(usage) == pytest.approx(0.10 + 0.50 + 0.02 + 0.125)
    assert evals.cache_hit_rate(usage) == pytest.approx(0.5)
    assert evals.cache_hit_rate(llm.Usage()) == 0.0


def test_summary_and_report(fake_client):
    story = _story_result(fake_client)
    fake_client()
    blocked = evals.run_case(_case("block_injection"))
    fake_client(msg(STORY), judged(score=3))
    base = evals.run_baseline(story)
    fake_client(*(judged(score=s) for s in (5, 5, 4, 5, 5)))
    check = evals.run_consistency(story)

    s = evals.summarize([story, blocked])
    assert (s.story_cases, s.passed, s.first_draft_passed, s.in_range) == (1, 1, 1, 1)
    assert (s.blocked_correct, s.blocked_total, s.allowed_correct, s.allowed_total) == (1, 1, 1, 1)
    assert s.mean_rounds_to_pass == 1

    report = evals.render_report([story, blocked], [base], [check], seconds=90)
    for fragment in (
        "# Eval results", "claude-haiku-5-5", "| Stories shown that pass the full rubric | 1/1 (100%) |",
        "| Must-block cases declined | 1/1 (100%) |", "## Baseline vs pipeline", "| hook | 3.00 | 5.00 |",
        "## Judge consistency", "5 5 4 5 5 (range 1)", "a single judgement is stable enough",
        "### Decline replies", "block_injection", "<details>", STORY.split("\n")[0],
    ):
        assert fragment in report


def test_report_recommends_median_judging_when_the_spread_is_large(fake_client):
    story = _story_result(fake_client)
    fake_client(*(judged(score=s) for s in (5, 3, 4, 5, 5)))
    report = evals.render_report([story], [], [evals.run_consistency(story)], seconds=1)
    assert "median-of-3 judging is worth adding" in report


def test_main_writes_the_report(fake_client, tmp_path, monkeypatch):
    fake_client(msg(intake_json()), msg(plan_json()), msg(STORY), judged())
    out = tmp_path / "report.md"
    code = evals.main(["--only", "alice_and_bob", "--skip-baseline", "--skip-consistency",
                       "--workers", "1", "--out", str(out)])
    assert code == 0
    assert "alice_and_bob" in out.read_text(encoding="utf-8")


def test_judge_json_helper_scores_every_criterion():
    # Guards the helper the tests above lean on.
    data = json.loads(judge_json(score=4))
    assert all(data[name]["score"] == 4 for name in SCORED_CRITERIA)
