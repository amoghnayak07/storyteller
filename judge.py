"""The judge: an LLM rubric plus deterministic checks, and the pass/gate rules.

Two halves (PLAN.md, "Rubric"):

- **Deterministic checks** (``run_checks``), pure code: word count (the hard
  300-800 gate, because LLMs miscount words), Flesch-Kincaid readability
  (feedback only), a moralizing-phrase scan (feedback only) and the banned-word
  scan (half of the safety gate).
- **The LLM judge** (``judge_story``) at effort ``high``: the Story Circle
  checklist, eight 1-5 criteria with evidence and a fix, and the ``safe`` flag.
  It never sees the plan, so it grades the story the child will hear, not the
  story we meant to write.

``passes_hard_gates`` decides whether a draft may ever be shown; ``passes``
decides whether the revision loop can stop; ``rank`` picks the best draft.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import guardrails
import llm
import prompts
from schemas import SCORED_CRITERIA, STORY_STEPS, YOU_PICK, AgeBand, Ingredients, JudgeResult

PASS_SCORE = 4

# Thinking at effort "high" counts toward max_tokens; keep well under the SDK's
# ~21K non-streaming ceiling.
JUDGE_MAX_TOKENS = 16_000

# Flesch-Kincaid grade targets (feedback only) and the sentence length to suggest.
READABILITY_TARGETS: dict[AgeBand, float] = {"5-6": 2.0, "7-8": 4.0, "9-10": 6.0}
SENTENCE_TARGETS: dict[AgeBand, int] = {"5-6": 8, "7-8": 10, "9-10": 13}

# How far from the target a story may drift before the reviser hears about it.
# The hard gate is the 300-800 range; this is feedback only.
TARGET_TOLERANCE = 0.15

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")
_SENTENCE_END_RE = re.compile(r"[.!?]+[\"'”’)]*(?=\s|$)")

_MORALIZING_PATTERNS: tuple[str, ...] = (
    r"the moral of (the|this) story",
    r"the lesson (is|was|here)",
    r"(learned|learnt|realized|realised|understood|knew) (that|an important lesson|a valuable lesson)",
    r"from that day (on|forward)[^.!?]{0,40}\b(knew|learned|learnt|understood|remembered|realized|realised)\b",
    r"(always|never forget to) remember (that|to)",
    r"(it'?s|it is) important to (always )?(be|share|remember|tell)",
    r"and that'?s why you should",
)
_MORALIZING_RE = re.compile("|".join(f"(?:{p})" for p in _MORALIZING_PATTERNS), re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Deterministic checks
# --------------------------------------------------------------------------- #


def word_count(text: str) -> int:
    """Words as a reader counts them: "don't" and "well-known" are one word each."""
    return len(_WORD_RE.findall(text))


def count_sentences(text: str) -> int:
    return max(1, len(_SENTENCE_END_RE.findall(text.strip())) or 1)


def count_syllables(word: str) -> int:
    """Vowel-group heuristic, good enough for a readability estimate."""
    word = word.lower().strip("'’")
    if not word:
        return 0
    if len(word) <= 3:
        return 1
    if word.endswith("ed") and not word.endswith(("ted", "ded")):
        word = word[:-2]  # silent: "jumped", "whispered"
    elif word.endswith("e") and not word.endswith(("le", "ee", "ye")):
        word = word[:-1]  # silent: "make", "smile"
    groups = re.findall(r"[aeiouy]+", word)
    return max(1, len(groups))


def flesch_kincaid_grade(text: str) -> float:
    words = _WORD_RE.findall(text)
    if not words:
        return 0.0
    sentences = count_sentences(text)
    syllables = sum(count_syllables(w) for w in words)
    return 0.39 * len(words) / sentences + 11.8 * syllables / len(words) - 15.59


def moralizing_phrases(text: str) -> list[str]:
    return [m.group(0) for m in _MORALIZING_RE.finditer(text)]


@dataclass(frozen=True)
class Checks:
    """Results of the code-only checks on one draft."""

    band: AgeBand
    target_words: int
    word_count: int
    fk_grade: float
    avg_sentence_words: float
    moralizing: list[str] = field(default_factory=list)
    banned: list[str] = field(default_factory=list)

    @property
    def in_range(self) -> bool:
        return guardrails.MIN_WORDS <= self.word_count <= guardrails.MAX_WORDS

    @property
    def readable(self) -> bool:
        return self.fk_grade <= READABILITY_TARGETS[self.band]

    @property
    def near_target(self) -> bool:
        return abs(self.word_count - self.target_words) <= TARGET_TOLERANCE * self.target_words

    def notes(self) -> list[str]:
        """Concrete reviser instructions for every check that missed."""
        notes = []
        if not self.in_range or not self.near_target:
            diff = self.target_words - self.word_count
            verb = "Add" if diff > 0 else "Cut"
            hard = "" if self.in_range else (
                f" It must be between {guardrails.MIN_WORDS} and {guardrails.MAX_WORDS} words."
            )
            notes.append(
                f"Length: the story is {self.word_count} words; the target is {self.target_words}.{hard} "
                f"{verb} about {abs(diff)} words, keeping all 8 steps."
            )
        if not self.readable:
            notes.append(
                f"Readability: about grade {self.fk_grade:.1f}; the target for ages {self.band} is "
                f"{READABILITY_TARGETS[self.band]:.0f} or below. Sentences average "
                f"{self.avg_sentence_words:.0f} words; aim for about {SENTENCE_TARGETS[self.band]}, "
                "with shorter, more common words."
            )
        if self.moralizing:
            quoted = "; ".join(f'"{p}"' for p in self.moralizing)
            notes.append(f"Preachy: remove the stated lesson ({quoted}) and show the change through action.")
        if self.banned:
            notes.append(f"Safety: remove these words and what they describe: {', '.join(sorted(set(self.banned)))}.")
        return notes


def run_checks(text: str, band: AgeBand, target_words: int) -> Checks:
    words = word_count(text)
    return Checks(
        band=band,
        target_words=target_words,
        word_count=words,
        fk_grade=round(flesch_kincaid_grade(text), 1),
        avg_sentence_words=round(words / count_sentences(text), 1),
        moralizing=moralizing_phrases(text),
        banned=guardrails.banned_words(text),
    )


# --------------------------------------------------------------------------- #
# LLM judge
# --------------------------------------------------------------------------- #


def judge_user_turn(
    story: str,
    *,
    request: str,
    ingredients: Ingredients,
    extras: list[str],
    band: AgeBand,
    target_words: int,
    edit_requests: list[str],
) -> str:
    shown = {
        name: ("(the writer's choice)" if value == YOU_PICK else value)
        for name, value in ingredients.model_dump().items()
    }
    parts = [
        prompts.tag("user_request", request),
        prompts.tag("ingredients", json.dumps(shown, indent=2)),
        prompts.tag("extra_elements", json.dumps(extras)),
    ]
    if edit_requests:
        parts.append(prompts.tag("edit_requests", json.dumps(edit_requests, indent=2)))
    parts += [
        prompts.tag("age_band", band),
        prompts.tag("target_words", str(target_words)),
        prompts.tag("story", story),
    ]
    return "\n\n".join(parts)


def judge_story(story: str, **context) -> llm.LLMResult[JudgeResult]:
    """Grade one story. ``context`` is the keyword arguments of ``judge_user_turn``.

    Raises ``llm.LLMRefusalError`` if the judge refuses; callers treat that as unsafe.
    """
    user = judge_user_turn(story, **context)
    return llm.generate_structured(
        prompts.JUDGE_SYSTEM, user, JudgeResult, effort="high", max_tokens=JUDGE_MAX_TOKENS
    )


# --------------------------------------------------------------------------- #
# Gates, pass condition, ranking
# --------------------------------------------------------------------------- #


def passes_hard_gates(result: JudgeResult | None, checks: Checks) -> bool:
    """May this draft ever be shown? Safe by both halves of the gate, and in range.

    ``result`` is ``None`` when the judge refused, which counts as unsafe.
    """
    return result is not None and result.safe and not checks.banned and checks.in_range


def missing_steps(result: JudgeResult | None) -> list[str]:
    if result is None:
        return list(STORY_STEPS)
    return [check.step for check in result.story_circle if not check.present]


def low_scores(result: JudgeResult | None) -> list[str]:
    if result is None:
        return list(SCORED_CRITERIA)
    return [name for name, c in result.scores().items() if c.score < PASS_SCORE]


def passes(result: JudgeResult | None, checks: Checks) -> bool:
    """The loop's stop condition: hard gates, all 8 steps, every criterion >= 4."""
    return passes_hard_gates(result, checks) and not missing_steps(result) and not low_scores(result)


def rank(result: JudgeResult | None, checks: Checks) -> tuple:
    """Sort key for choosing the best draft (higher is better)."""
    if result is None:
        return (False, False, 0, 0, -abs(checks.word_count - checks.target_words))
    total = sum(c.score for c in result.scores().values())
    steps = len(STORY_STEPS) - len(missing_steps(result))
    return (
        passes_hard_gates(result, checks),
        passes(result, checks),
        steps,
        total,
        -abs(checks.word_count - checks.target_words),
    )


def revision_notes(result: JudgeResult | None, checks: Checks) -> list[str]:
    """Everything the reviser should fix, most important first."""
    notes: list[str] = []
    if result is None:
        notes.append("Safety: the story could not be approved as safe. Make every part gentle and bedtime-safe.")
    else:
        if not result.safe:
            issue = result.safety_issue.strip() or "something in the story is not safe for ages 5-10"
            notes.append(f"Safety: {issue}. Rewrite that part so it fits the content rules.")
        for check in result.story_circle:
            if not check.present:
                notes.append(f"Story Circle: the '{check.step}' step is missing or too weak. Make it clear.")
        for name, c in result.scores().items():
            if c.score < PASS_SCORE or c.fix.strip():
                fix = c.fix.strip() or "Improve this."
                notes.append(f"{name} (scored {c.score}/5): {fix}")
    notes += checks.notes()
    return notes
