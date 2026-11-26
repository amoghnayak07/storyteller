"""Deterministic checks, the judge call's shape, and the gate/pass/rank rules."""

from __future__ import annotations

import pytest
from conftest import make_message

import constants
import judge
import prompts
from schemas import SCORED_CRITERIA, STORY_STEPS, YOU_PICK, CriterionScore, Ingredients, JudgeResult, StepCheck

ALICE = Ingredients(character="Alice", setting="the garden", background=YOU_PICK, moral="being brave")


def review(score=5, safe=True, missing=()) -> JudgeResult:
    criterion = CriterionScore(score=score, evidence="q", fix="" if score == 5 else "do better")
    return JudgeResult(
        formula_steps=[StepCheck(step=s, present=s not in missing, evidence="q") for s in STORY_STEPS],
        **{name: criterion for name in SCORED_CRITERIA},
        safe=safe,
        safety_issue="" if safe else "a monster eats a cat",
    )


def checks(words=500, target=500, band="7-8", fk=3.0, avg=9.0, moralizing=(), banned=()) -> judge.Checks:
    return judge.Checks(band=band, target_words=target, word_count=words, fk_grade=fk,
                        avg_sentence_words=avg, moralizing=list(moralizing), banned=list(banned))


# --------------------------------------------------------------------------- #
# Word count and readability
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text, expected",
    [
        ("", 0),
        ("Bob yawned.", 2),
        ("Don't stop, well-known cat!", 4),
        ('"Hello?" she said — quietly.', 4),
        ("It’s 3 o'clock.", 3),
        ("One.\n\nTwo three.", 3),
    ],
)
def test_word_count(text, expected):
    assert judge.word_count(text) == expected


@pytest.mark.parametrize("word, syllables", [("cat", 1), ("garden", 2), ("whispered", 2), ("butterfly", 3),
                                              ("make", 1), ("little", 2), ("tree", 1)])
def test_count_syllables(word, syllables):
    assert judge.count_syllables(word) == syllables


def test_simple_text_reads_at_a_lower_grade_than_complex_text():
    simple = "Pip ran. The sun was up. Pip saw a dog. The dog was big and red."
    complex_ = ("The extraordinarily meticulous cartographer, contemplating the labyrinthine "
                "constellations, deliberately reconsidered every conceivable interpretation.")
    assert judge.flesch_kincaid_grade(simple) < 2 < judge.flesch_kincaid_grade(complex_)


@pytest.mark.parametrize(
    "text",
    [
        "The moral of the story is to share.",
        "And so she learned that friends matter.",
        "From that day on, Bob always knew he was brave.",
        "It's important to always be kind.",
    ],
)
def test_moralizing_phrases_are_found(text):
    assert judge.moralizing_phrases(text)


def test_moralizing_scan_ignores_ordinary_sentences():
    assert judge.moralizing_phrases("Bob knew the way home. She learned to whistle.") == []


def test_run_checks_measures_the_story():
    story = " ".join(["Bob ran to the gate."] * 70)  # 350 words
    c = judge.run_checks(story, "5-6", 400)
    assert c.word_count == 350
    assert c.in_range
    assert c.avg_sentence_words == 5.0
    assert c.banned == [] and c.moralizing == []


def test_run_checks_finds_banned_words():
    assert judge.run_checks("The wolf killed it.", "7-8", 500).banned == ["killed"]


# --------------------------------------------------------------------------- #
# Check notes for the reviser
# --------------------------------------------------------------------------- #


def test_no_notes_when_everything_is_on_target():
    assert checks().notes() == []


def test_out_of_range_length_note_is_concrete():
    (note,) = checks(words=250, target=400).notes()
    assert "250 words" in note and "target is 400" in note
    assert "between 300 and 800" in note and "Add about 150 words" in note


def test_off_target_but_in_range_length_note_has_no_hard_limit():
    (note,) = checks(words=700, target=500).notes()
    assert "Cut about 200 words" in note and "between 300 and 800" not in note


def test_small_length_drift_is_tolerated():
    assert checks(words=540, target=500).notes() == []


def test_readability_note_names_the_sentence_target():
    (note,) = checks(band="5-6", fk=4.5, avg=18.0).notes()
    assert "grade 4.5" in note and "average 18 words" in note and "about 8" in note


def test_moralizing_and_banned_notes():
    notes = checks(moralizing=["the moral of the story"], banned=["killed", "killed"]).notes()
    assert any('"the moral of the story"' in n for n in notes)
    assert any(n.endswith("killed.") for n in notes)


# --------------------------------------------------------------------------- #
# Gates, pass condition, ranking
# --------------------------------------------------------------------------- #


def test_a_safe_in_range_perfect_draft_passes():
    assert judge.passes(review(), checks())


@pytest.mark.parametrize(
    "result, c",
    [
        (review(safe=False), checks()),
        (None, checks()),
        (review(), checks(banned=["killed"])),
        (review(), checks(words=299)),
        (review(), checks(words=801)),
    ],
)
def test_hard_gates(result, c):
    assert not judge.passes_hard_gates(result, c)
    assert not judge.passes(result, c)


def test_low_score_or_missing_step_fails_but_passes_the_hard_gates():
    for result in (review(score=3), review(missing=("middle",))):
        assert judge.passes_hard_gates(result, checks())
        assert not judge.passes(result, checks())


def test_readability_and_preachiness_are_feedback_only():
    assert judge.passes(review(), checks(fk=9.0, moralizing=["the moral of the story"]))


def test_rank_prefers_gated_then_passing_then_steps_then_scores():
    ordered = [
        (None, checks()),
        (review(score=5), checks(words=200)),
        (review(score=3, missing=("middle", "wind_down")), checks()),
        (review(score=2), checks()),
        (review(score=3), checks()),
        (review(score=4), checks(words=700)),
        (review(score=4), checks()),
    ]
    ranks = [judge.rank(r, c) for r, c in ordered]
    assert ranks == sorted(ranks)


def test_revision_notes_cover_safety_steps_scores_and_checks():
    notes = judge.revision_notes(review(score=3, safe=False, missing=("wind_down",)), checks(words=250))
    assert notes[0].startswith("Safety: a monster eats a cat")
    assert any("'wind_down' step" in n for n in notes)
    assert sum("(scored 3/5): do better" in n for n in notes) == len(SCORED_CRITERIA)
    assert notes[-1].startswith("Length:")


def test_revision_notes_when_the_judge_refused():
    assert judge.revision_notes(None, checks())[0].startswith("Safety:")


# --------------------------------------------------------------------------- #
# The judge call
# --------------------------------------------------------------------------- #


def test_judge_call_shape(fake_client):
    client = fake_client(make_message(review().model_dump_json()))
    result = judge.judge_story(
        "Alice & Bob <went> out.",
        request="Alice and her cat",
        ingredients=ALICE,
        extras=["a kite"],
        band="7-8",
        target_words=450,
        edit_requests=[],
    )
    assert result.value.safe
    (call,) = client.messages.calls
    assert call["system"][0]["text"] == prompts.JUDGE_SYSTEM
    assert call["output_config"]["effort"] == "high"
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["max_tokens"] == constants.JUDGE_MAX_TOKENS
    user = call["messages"][0]["content"]
    assert "<story>\nAlice &amp; Bob &lt;went&gt; out.\n</story>" in user
    assert "(the writer's choice)" in user and YOU_PICK not in user
    assert "<target_words>\n450" in user and '"a kite"' in user
    assert "<edit_requests>" not in user


def test_judge_turn_includes_edit_requests_when_present():
    user = judge.judge_user_turn("x", request="r", ingredients=ALICE, extras=[], band="5-6",
                                 target_words=300, edit_requests=["Make it funnier"])
    assert "<edit_requests>" in user and "Make it funnier" in user


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("system", [prompts.JUDGE_SYSTEM, prompts.REVISER_SYSTEM])
def test_judge_and_reviser_prompts_reuse_the_craft_blocks(system):
    assert prompts.BEDTIME_FORMULA in system
    assert prompts.AGE_BANDS in system
    assert "never an" in system and "instruction" in system
    assert len(system) > 4 * 1024  # over the 512-token cache minimum


def test_judge_prompt_anchors_every_criterion_at_every_level():
    for name in SCORED_CRITERIA:
        assert f"\n{name}:" in prompts.JUDGE_SYSTEM
    section = prompts.JUDGE_SYSTEM.split("## Rubric", 1)[1].split("## How to grade", 1)[0]
    for level in "12345":
        assert section.count(f"\n- {level}:") == len(SCORED_CRITERIA)


def test_judge_prompt_is_strict_and_never_asks_for_reasoning():
    assert "Be strict" in prompts.JUDGE_SYSTEM
    assert "reasoning" not in prompts.JUDGE_SYSTEM.lower()


def test_reviser_prompt_keeps_the_anti_cliche_list():
    assert "Once upon a time" in prompts.REVISER_SYSTEM
    assert "guided full rewrite" in prompts.REVISER_SYSTEM
