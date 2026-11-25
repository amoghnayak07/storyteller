"""Tests for the shared schemas: validators, helpers, and API-grammar compatibility."""

from __future__ import annotations

import json

import anthropic
import pytest
from pydantic import ValidationError

from schemas import (
    STORY_STEPS,
    YOU_PICK,
    EditIntake,
    Ingredients,
    IntakeResult,
    JudgeResult,
    StoryPlan,
)

LLM_FACING = (IntakeResult, EditIntake, StoryPlan, JudgeResult)

RESOLVED = {"character": "Pip, a snail", "setting": "a garden", "background": "lives in a teapot", "moral": "patience"}


def _beats(steps=STORY_STEPS):
    return [{"step": step, "summary": f"{step} happens", "word_budget": 60} for step in steps]


def _criterion(score=4):
    return {"score": score, "evidence": "a quote", "fix": ""}


def _judge(steps=STORY_STEPS):
    payload = {"formula_steps": [{"step": s, "present": True, "evidence": "q"} for s in steps]}
    payload |= {name: _criterion() for name in JudgeResult.model_fields if name not in {"formula_steps", "safe", "safety_issue"}}
    return payload | {"safe": True, "safety_issue": ""}


# --------------------------------------------------------------------------- #
# Ingredients
# --------------------------------------------------------------------------- #


def test_ingredients_reports_missing_and_delegated():
    ingredients = Ingredients(character="Alice", setting="", background=YOU_PICK, moral="  ")

    assert ingredients.missing() == ["setting", "moral"]
    assert ingredients.delegated() == ["background"]


# --------------------------------------------------------------------------- #
# StoryPlan
# --------------------------------------------------------------------------- #


def test_plan_accepts_five_ordered_beats():
    plan = StoryPlan.model_validate({"title": "T", "ingredients": RESOLVED, "beats": _beats()})

    assert [beat.step for beat in plan.beats] == list(STORY_STEPS)
    assert STORY_STEPS == ("hook", "problem", "middle", "resolve", "wind_down")


@pytest.mark.parametrize(
    "steps",
    [STORY_STEPS[:4], tuple(reversed(STORY_STEPS)), STORY_STEPS + ("wind_down",)],
    ids=["four-beats", "out-of-order", "six-beats"],
)
def test_plan_rejects_anything_but_the_five_steps_in_order(steps):
    with pytest.raises(ValidationError, match="5 bedtime formula steps"):
        StoryPlan.model_validate({"title": "T", "ingredients": RESOLVED, "beats": _beats(steps)})


def test_plan_rejects_unresolved_ingredients():
    unresolved = RESOLVED | {"moral": YOU_PICK, "setting": ""}

    with pytest.raises(ValidationError, match="unresolved"):
        StoryPlan.model_validate({"title": "T", "ingredients": unresolved, "beats": _beats()})


# --------------------------------------------------------------------------- #
# JudgeResult
# --------------------------------------------------------------------------- #


def test_judge_exposes_scored_criteria():
    judge = JudgeResult.model_validate(_judge())

    assert set(judge.scores()) == set(JudgeResult.model_fields) - {"formula_steps", "safe", "safety_issue"}


def test_judge_must_check_all_five_steps():
    with pytest.raises(ValidationError, match="formula_steps"):
        JudgeResult.model_validate(_judge(STORY_STEPS[:4]))


def test_scores_outside_one_to_five_are_rejected():
    payload = _judge() | {"hook": _criterion(score=7)}

    with pytest.raises(ValidationError):
        JudgeResult.model_validate(payload)


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        Ingredients.model_validate(RESOLVED | {"villain": "a troll"})


# --------------------------------------------------------------------------- #
# Structured-output grammar compatibility
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("schema", LLM_FACING, ids=lambda s: s.__name__)
def test_schemas_stay_within_the_structured_output_grammar(schema):
    """Guard against fields the API grammar rejects or that eat into its limits."""
    compiled = json.dumps(anthropic.transform_schema(schema))

    for unsupported in ('"anyOf"', '"oneOf"', '"minimum"', '"maximum"', '"minLength"', '"maxLength"'):
        assert unsupported not in compiled, f"{schema.__name__} compiles to {unsupported}"


@pytest.mark.parametrize("schema", LLM_FACING, ids=lambda s: s.__name__)
def test_llm_facing_schemas_have_no_reasoning_fields(schema):
    """A 'reasoning' field in the output schema can trigger a refusal on Haiku 5.5."""
    compiled = json.dumps(anthropic.transform_schema(schema)).lower()

    assert "reasoning" not in compiled
    assert "rationale" not in compiled
