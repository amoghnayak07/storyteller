"""Pydantic models shared by the LLM layer, the graph state, and the evals.

Every model that the LLM fills in (via native structured outputs) follows the
constraints of Anthropic's JSON-schema grammar:

- No ``Optional`` fields and no defaults: every field is required, and "absent"
  is expressed as an empty string or empty list. This keeps us well under the
  API's limits on optional and union-typed parameters.
- Bounded choices use ``Literal`` (compiled to ``enum``), never numeric
  ``ge``/``le`` constraints, which the API does not support.
- No "reasoning"/"explain your thinking" fields. The model reasons through
  adaptive thinking; asking it to put reasoning in the output can trigger a
  refusal. The judge records ``evidence`` (a quote) and a ``fix`` instead.

Structural rules the grammar cannot express (e.g. "exactly 8 beats, in order")
are enforced with Pydantic validators; a violation raises ``ValidationError``,
which the LLM layer turns into one retry.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

# --------------------------------------------------------------------------- #
# Shared vocabulary
# --------------------------------------------------------------------------- #

AgeBand = Literal["5-6", "7-8", "9-10"]

Category = Literal[
    "adventure",
    "friendship",
    "animals",
    "silly",
    "magic",
    "everyday",
    "calming",
]

# The eight steps of Dan Harmon's Story Circle, in order.
StoryStep = Literal["you", "need", "go", "search", "find", "take", "return", "change"]
STORY_STEPS: tuple[StoryStep, ...] = (
    "you",
    "need",
    "go",
    "search",
    "find",
    "take",
    "return",
    "change",
)

IngredientName = Literal["character", "setting", "background", "moral"]
INGREDIENT_NAMES: tuple[IngredientName, ...] = ("character", "setting", "background", "moral")

# Sentinel the intake model writes when the user delegates an ingredient
# ("you pick", "surprise me", ...). The planner then chooses it.
YOU_PICK = "YOU_PICK"

# Outcome of screening any piece of user text.
Verdict = Literal["story", "off_task", "blocked", "public_figure"]

BlockedCategory = Literal[
    "none",
    "violence",
    "death",
    "weapons",
    "sexual",
    "horror",
    "substances",
    "profanity",
    "self_harm",
    "hate",
    "abuse",
    "real_tragedy_or_politics",
    "dangerous_imitation",
    "prompt_injection",
    "public_figure",
]

Score = Literal[1, 2, 3, 4, 5]


class _Strict(BaseModel):
    """Base for LLM-facing models.

    Unknown keys are an error rather than silently dropped, and field
    docstrings become JSON-schema descriptions, so the model sees them.
    """

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)


# --------------------------------------------------------------------------- #
# Intake (guardrail + ingredient extraction)
# --------------------------------------------------------------------------- #


class Ingredients(_Strict):
    """The four things a story needs before we write it.

    Each value is the user's description, ``""`` if not provided yet, or
    ``YOU_PICK`` if the user asked us to choose.
    """

    character: str
    setting: str
    background: str
    moral: str

    def missing(self) -> list[IngredientName]:
        """Ingredients that are neither provided nor delegated."""
        return [name for name in INGREDIENT_NAMES if not getattr(self, name).strip()]

    def delegated(self) -> list[IngredientName]:
        return [name for name in INGREDIENT_NAMES if getattr(self, name).strip() == YOU_PICK]


class IntakeResult(_Strict):
    """Classification of a new story request (or an answer to the ingredient prompt)."""

    verdict: Verdict
    blocked_category: BlockedCategory
    category: Category
    ingredients: Ingredients
    extra_elements: list[str]
    """Other things the user asked to include (e.g. "a unicorn", "the beach")."""
    suggestion: str
    """A related, safe alternative request. Empty unless verdict is blocked/public_figure."""


EditKind = Literal["structural", "surface"]


class EditIntake(_Strict):
    """Classification of a change request made after a story was told."""

    verdict: Verdict
    blocked_category: BlockedCategory
    edit_kind: EditKind
    """structural = changes character/setting/background/moral -> re-plan.
    surface = tone, humour, dialogue, length, pacing -> revise directly."""
    ingredients: Ingredients
    """The ingredients after applying the edit (unchanged ones copied through)."""
    instruction: str
    """The edit restated as one clear instruction for the reviser."""
    new_target_words: int
    """Requested new length, or 0 if the edit does not change length."""
    drops_story_step: bool
    """True if the edit asks to cut a part every story needs (e.g. "remove the sad
    part"). The instruction then softens that part instead of removing it."""
    asks_age_change: bool
    """True if the edit asks to retell the story for a different age."""
    suggestion: str
    """A related, safe alternative edit. Empty unless verdict is blocked/public_figure."""


# --------------------------------------------------------------------------- #
# Planner
# --------------------------------------------------------------------------- #


class Beat(_Strict):
    step: StoryStep
    summary: str
    """What happens in this beat, concretely, for this story."""
    word_budget: int


class StoryPlan(_Strict):
    title: str
    ingredients: Ingredients
    """Fully resolved: every YOU_PICK replaced with a concrete choice."""
    beats: list[Beat]

    @field_validator("beats")
    @classmethod
    def _eight_beats_in_order(cls, beats: list[Beat]) -> list[Beat]:
        steps = tuple(beat.step for beat in beats)
        if steps != STORY_STEPS:
            raise ValueError(f"beats must be the 8 Story Circle steps in order {STORY_STEPS}, got {steps}")
        return beats

    @field_validator("ingredients")
    @classmethod
    def _fully_resolved(cls, ingredients: Ingredients) -> Ingredients:
        unresolved = ingredients.missing() + ingredients.delegated()
        if unresolved:
            raise ValueError(f"plan must resolve every ingredient; unresolved: {unresolved}")
        return ingredients


# --------------------------------------------------------------------------- #
# Judge
# --------------------------------------------------------------------------- #


class CriterionScore(_Strict):
    score: Score
    evidence: str
    """A short quote from the story supporting the score."""
    fix: str
    """One concrete instruction to improve this criterion, or "" if none needed."""


class StepCheck(_Strict):
    step: StoryStep
    present: bool
    evidence: str
    """The passage that delivers this step, or "" if missing."""


class JudgeResult(_Strict):
    story_circle: list[StepCheck]
    request_fidelity: CriterionScore
    age_fit: CriterionScore
    hook: CriterionScore
    agency: CriterionScore
    stakes: CriterionScore
    lesson_shown: CriterionScore
    dialogue_imagery: CriterionScore
    bedtime_landing: CriterionScore
    safe: bool
    safety_issue: str
    """What is unsafe, or "" if safe."""

    @field_validator("story_circle")
    @classmethod
    def _all_eight_steps(cls, checks: list[StepCheck]) -> list[StepCheck]:
        steps = tuple(check.step for check in checks)
        if steps != STORY_STEPS:
            raise ValueError(f"story_circle must check the 8 steps in order {STORY_STEPS}, got {steps}")
        return checks

    def scores(self) -> dict[str, CriterionScore]:
        """The scored rubric criteria, keyed by name."""
        return {name: getattr(self, name) for name in SCORED_CRITERIA}


SCORED_CRITERIA: tuple[str, ...] = (
    "request_fidelity",
    "age_fit",
    "hook",
    "agency",
    "stakes",
    "lesson_shown",
    "dialogue_imagery",
    "bedtime_landing",
)
