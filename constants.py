"""Every tunable setting in one place: the model, limits, thresholds and budgets.

Text the user reads lives in ``user_constants.py``. Prompt wording lives in
``prompts.py``, and the guardrail keyword lists stay in ``guardrails.py`` next
to the code that compiles them. The vocabularies that define the schemas
(``STORY_STEPS``, ``SCORED_CRITERIA``, ``YOU_PICK``) stay in ``schemas.py``,
beside the ``Literal`` types they mirror.
"""

from __future__ import annotations

from schemas import AgeBand, StoryStep

# --------------------------------------------------------------------------- #
# Model and API calls
# --------------------------------------------------------------------------- #

MODEL = "claude-haiku-5-5"

# Thinking tokens count toward max_tokens. An 800-word story is ~1.1K tokens,
# so 8K leaves ample headroom. Kept below the SDK's ~21K non-streaming ceiling.
DEFAULT_MAX_TOKENS = 8_000

# The judge thinks at effort "high", which also counts toward max_tokens; keep
# it well under the SDK's ~21K non-streaming ceiling.
JUDGE_MAX_TOKENS = 16_000

# One retry for malformed or truncated output (API-level retries are the SDK's job).
MAX_ATTEMPTS = 2

# --------------------------------------------------------------------------- #
# Session inputs
# --------------------------------------------------------------------------- #

MIN_AGE = 5
MAX_AGE = 10
DEFAULT_AGE = 7

# The hard length gate: a story outside this range is never shown.
MIN_WORDS = 300
MAX_WORDS = 800
DEFAULT_WORDS = 500

# The writers overshoot long targets (an 800-word target gave drafts of 815-926
# words in the evals), and MAX_WORDS is a hard cap with no tolerance. So the
# plan, the writers, the judge and the length feedback all aim at most this
# high, while the user's own number stays in the state's ``target_words``.
WRITE_CAP = 760

# --------------------------------------------------------------------------- #
# Story shape
# --------------------------------------------------------------------------- #

# Share of the word budget (percent) for each bedtime-formula step. The planner
# prompt quotes these, and code falls back to them when the planner's own
# budgets are unusable.
BEAT_WEIGHTS: dict[StoryStep, int] = {
    "hook": 15,
    "problem": 10,
    "middle": 40,
    "resolve": 15,
    "wind_down": 20,
}

# --------------------------------------------------------------------------- #
# Conversation loop
# --------------------------------------------------------------------------- #

# Ingredient answers in a row that fill nothing new before we stop asking and
# let the planner pick the rest.
MAX_FRUITLESS_ANSWERS = 2

# Reviser rounds per story before the best gated draft is shown.
MAX_REVISIONS = 3

# --------------------------------------------------------------------------- #
# Judging
# --------------------------------------------------------------------------- #

# Every scored criterion must reach this for a draft to pass.
PASS_SCORE = 4

# Flesch-Kincaid grade targets (feedback only) and the sentence length to suggest.
READABILITY_TARGETS: dict[AgeBand, float] = {"5-6": 2.0, "7-8": 4.0, "9-10": 6.0}
SENTENCE_TARGETS: dict[AgeBand, int] = {"5-6": 8, "7-8": 10, "9-10": 13}

# How far from the target a story may drift before the reviser hears about it.
# The hard gate is MIN_WORDS-MAX_WORDS; this is feedback only.
TARGET_TOLERANCE = 0.15

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

# Seconds per word for the typewriter replay of an approved story.
TYPEWRITER_DELAY = 0.03

# --------------------------------------------------------------------------- #
# Evals
# --------------------------------------------------------------------------- #

# Haiku 5.5, USD per million tokens (prompts under 100K tokens). Cache writes
# cost 1.25x input and cache reads 0.1x.
PRICE_IN = 0.10
PRICE_OUT = 0.50
PRICE_CACHE_WRITE = PRICE_IN * 1.25
PRICE_CACHE_READ = PRICE_IN * 0.1

CONSISTENCY_STORIES = 3
CONSISTENCY_REPEATS = 5

# A per-criterion range this large means one judgement isn't reliable enough,
# and the pipeline should take the median of 3 judgements instead.
LARGE_SPREAD = 2

# A run needs at most: ingredients (twice), then the edit prompt. Anything
# beyond this means the auto-answers are looping.
MAX_EVAL_INTERRUPTS = 6
