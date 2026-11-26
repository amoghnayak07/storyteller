"""Screening for every piece of user text, and the banned-word scan for stories.

Layers that live here (Submission.md, "Guardrails: four layers"):

1. **Keyword pre-filter** (``prefilter``), pure code, run first.
   - An injection phrase ends screening immediately with the off-task reply;
     the text never reaches the model.
   - A blocked keyword fixes the verdict to ``blocked``. The intake model is
     still called, with a ``<prefilter_flag>``, but only to write the tailored
     suggestion; it cannot overturn the block.
2. **Intake classification** (``screen_request`` / ``screen_edit``) at effort
   ``low``: one of ``story``, ``off_task``, ``blocked``, ``public_figure``,
   plus ingredient extraction (requests) or edit routing (edits).
3. The **banned-word scan** (``banned_words``) is the deterministic half of the
   output gate; the judge's ``safe`` flag is the other half (``judge.py``).

A refusal from the model (``LLMRefusalError``) is treated as a block.

The keyword lists favour precision over recall: they catch only the obvious,
so benign requests like "a brave knight with a sword" or "a haunted house"
pass through to the model, which applies the full content rules.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Literal

import llm
import prompts
import user_constants as ui
from constants import MAX_WORDS, MIN_WORDS
from schemas import (
    INGREDIENT_NAMES,
    YOU_PICK,
    BlockedCategory,
    Category,
    EditIntake,
    IngredientName,
    Ingredients,
    IntakeResult,
    Verdict,
)

# --------------------------------------------------------------------------- #
# Keyword lexicon
# --------------------------------------------------------------------------- #

# Prompt-injection phrases. Matching any of these means "off-task", no LLM call.
_INJECTION_PATTERNS: tuple[str, ...] = (
    r"\b(ignore|disregard|forget|override)\b.{0,30}\b(previous|prior|above|earlier|all|your|the)\b.{0,20}\b(instructions?|prompts?|rules|directions)\b",
    r"\bsystem\s+prompt\b",
    r"\byou\s+are\s+now\b",
    r"\bnew\s+instructions?\b",
    r"\bdeveloper\s+mode\b",
    r"\bjailbreak",
    r"\bDAN\s+mode\b",
    r"</?\s*(system|assistant|user_request|user_edit|known_ingredients|known_category"
    r"|current_ingredients|current_target_words|prefilter_flag|asked_for|draft)\s*>",
)

# Obvious blocked content. Each pattern is matched case-insensitively on word
# boundaries. Hate speech and slurs are left to the model, which judges them
# in context.
_BLOCKED_PATTERNS: dict[BlockedCategory, tuple[str, ...]] = {
    "violence": (
        r"gore", r"gory", r"bloodbath", r"blood-?soaked", r"bloody",
        r"dismember\w*", r"decapitat\w*", r"behead\w*", r"tortur\w*",
        r"stab(s|bed|bing)?", r"massacre\w*", r"slaughter\w*",
    ),
    "death": (
        r"kill(s|ed|ing|er|ers)?", r"murder\w*", r"corpses?", r"dead\s+bod(y|ies)",
    ),
    "weapons": (
        r"rifles?", r"shotguns?", r"pistols?", r"machine\s+guns?", r"grenades?",
        r"bombs?",
    ),
    "sexual": (
        r"sex", r"sexy", r"sexual\w*", r"porn\w*", r"naked", r"nude", r"erotic\w*",
    ),
    "substances": (
        r"cocaine", r"heroin", r"meth", r"marijuana", r"smok(e|es|ed|ing)\s+weed", r"drunk\w*",
        r"alcohol\w*", r"beers?", r"vodka", r"whiske?y", r"booze", r"cigarettes?",
        r"cigars?", r"vap(e|es|ing)",
    ),
    "profanity": (
        r"fuck\w*", r"shit\w*", r"bitch\w*", r"asshole\w*", r"bastards?", r"damn\w*",
        r"cunt\w*",
    ),
    "self_harm": (
        r"suicid\w*", r"self[-\s]?harm\w*",
        r"kill(s|ed|ing)?\s+(myself|yourself|himself|herself|themselves)",
    ),
    "abuse": (r"molest\w*",),
    "real_tragedy_or_politics": (
        r"9/11", r"holocaust", r"school\s+shootings?", r"terroris\w*", r"nazis?",
    ),
    "dangerous_imitation": (
        r"play(s|ed|ing)?\s+with\s+(matches|lighters?)",
        r"drink(s|ing)?\s+bleach",
    ),
}

# Benign phrases that contain a blocked keyword; removed before scanning.
_ALLOWED_PHRASES: tuple[str, ...] = (
    r"killer\s+whales?",
    r"bath\s+bombs?",
    r"photo-?bomb\w*",
)

_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE | re.DOTALL)
_ALLOWED_RE = re.compile("|".join(_ALLOWED_PHRASES), re.IGNORECASE)
_BLOCKED_RES: dict[BlockedCategory, re.Pattern[str]] = {
    category: re.compile(r"\b(" + "|".join(patterns) + r")\b", re.IGNORECASE)
    for category, patterns in _BLOCKED_PATTERNS.items()
}


@dataclass(frozen=True)
class Flag:
    """A pre-filter hit."""

    verdict: Literal["off_task", "blocked"]
    category: BlockedCategory
    match: str


def prefilter(text: str) -> Flag | None:
    """Keyword screen run before any LLM call. ``None`` means nothing obvious found."""
    injection = _INJECTION_RE.search(text)
    if injection:
        return Flag("off_task", "prompt_injection", injection.group(0))
    hits = _blocked_hits(text)
    if hits:
        category, match = hits[0]
        return Flag("blocked", category, match)
    return None


def banned_words(text: str) -> list[str]:
    """Blocked keywords found in a story (empty list = clean). Part of the output gate."""
    return [match for _, match in _blocked_hits(text)]


def _blocked_hits(text: str) -> list[tuple[BlockedCategory, str]]:
    cleaned = _ALLOWED_RE.sub(" ", text)
    hits: list[tuple[BlockedCategory, str]] = []
    for category, pattern in _BLOCKED_RES.items():
        hits.extend((category, m.group(0)) for m in pattern.finditer(cleaned))
    return hits


# --------------------------------------------------------------------------- #
# Decline replies
# --------------------------------------------------------------------------- #

_SCARY: frozenset[str] = frozenset(
    {"violence", "death", "weapons", "horror", "self_harm", "abuse", "real_tragedy_or_politics"}
)


def decline_message(verdict: Verdict, blocked_category: BlockedCategory, suggestion: str) -> str:
    """The warm, never-accusatory reply for a non-story verdict."""
    if verdict == "off_task":
        return ui.OFF_TASK_MESSAGE
    if verdict == "public_figure":
        opener = ui.PUBLIC_FIGURE_OPENER
    elif blocked_category in _SCARY:
        opener = ui.SCARY_OPENER
    else:
        opener = ui.UNSUITABLE_OPENER
    return ui.SUGGESTION_TEMPLATE.format(opener=opener, suggestion=suggestion)


# --------------------------------------------------------------------------- #
# Story requests
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RequestScreening:
    result: IntakeResult
    usage: llm.Usage = field(default_factory=llm.Usage)

    @property
    def message(self) -> str:
        """The reply to show for a non-story verdict ("" for a story)."""
        r = self.result
        return "" if r.verdict == "story" else decline_message(r.verdict, r.blocked_category, r.suggestion)


def screen_request(
    text: str,
    *,
    known_ingredients: Ingredients | None = None,
    known_category: Category | None = None,
    asked_for: list[IngredientName] | None = None,
) -> RequestScreening:
    """Screen a story request, or an answer to the ingredient question.

    Pass ``known_ingredients`` (and ``known_category``) when the user is
    answering the ingredient question; values the answer leaves empty are
    kept from what we already had. ``asked_for`` names the ingredients we just
    asked about, so a bare answer like "friendship" lands in the right slot.
    """
    known = known_ingredients or _empty_ingredients()
    flag = prefilter(text)
    if flag and flag.verdict == "off_task":
        return RequestScreening(
            IntakeResult(
                verdict="off_task",
                blocked_category=flag.category,
                category=known_category or "everyday",
                ingredients=known,
                extra_elements=[],
                suggestion="",
            )
        )

    parts = []
    if known_ingredients is not None:
        parts.append(prompts.tag("known_ingredients", _ingredients_json(known_ingredients)))
    if known_category is not None:
        parts.append(prompts.tag("known_category", known_category))
    if asked_for:
        parts.append(prompts.tag("asked_for", ", ".join(asked_for)))
    if flag:
        parts.append(prompts.tag("prefilter_flag", flag.category))
    parts.append(prompts.tag("user_request", text))

    try:
        call = llm.generate_structured(prompts.INTAKE_SYSTEM, "\n\n".join(parts), IntakeResult, effort="low")
    except llm.LLMRefusalError:
        return RequestScreening(_refused_intake(known, known_category, flag))

    result = _normalize_intake(call.value, known, flag)
    return RequestScreening(result, call.usage)


def _normalize_intake(result: IntakeResult, known: Ingredients, flag: Flag | None) -> IntakeResult:
    verdict, category = _enforce_flag(result.verdict, result.blocked_category, flag)
    return result.model_copy(
        update={
            "verdict": verdict,
            "blocked_category": category,
            "ingredients": _merge_ingredients(known, result.ingredients),
            "suggestion": _clean_suggestion(verdict, result.suggestion),
        }
    )


def _refused_intake(known: Ingredients, known_category: Category | None, flag: Flag | None) -> IntakeResult:
    return IntakeResult(
        verdict="blocked",
        blocked_category=flag.category if flag else "none",
        category=known_category or "everyday",
        ingredients=known,
        extra_elements=[],
        suggestion=ui.FALLBACK_SUGGESTION,
    )


# --------------------------------------------------------------------------- #
# Edit requests
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EditScreening:
    result: EditIntake
    target_words: int
    """The length to write to, already clamped to 300-800."""
    notes: list[str]
    """One-line notes for the user (length clamped, sad part softened, age fixed)."""
    usage: llm.Usage = field(default_factory=llm.Usage)

    @property
    def message(self) -> str:
        """The reply to show for a non-story verdict ("" for an accepted edit)."""
        r = self.result
        return "" if r.verdict == "story" else decline_message(r.verdict, r.blocked_category, r.suggestion)


def screen_edit(text: str, *, ingredients: Ingredients, target_words: int) -> EditScreening:
    """Screen and classify a change request for the story just told."""
    flag = prefilter(text)
    if flag and flag.verdict == "off_task":
        return EditScreening(_declined_edit("off_task", flag.category, ingredients, ""), target_words, [])

    parts = [
        prompts.tag("current_ingredients", _ingredients_json(ingredients)),
        prompts.tag("current_target_words", str(target_words)),
    ]
    if flag:
        parts.append(prompts.tag("prefilter_flag", flag.category))
    parts.append(prompts.tag("user_edit", text))

    try:
        call = llm.generate_structured(prompts.EDIT_SYSTEM, "\n\n".join(parts), EditIntake, effort="low")
    except llm.LLMRefusalError:
        category: BlockedCategory = flag.category if flag else "none"
        return EditScreening(
            _declined_edit("blocked", category, ingredients, ui.FALLBACK_SUGGESTION), target_words, []
        )

    result = call.value
    verdict, category = _enforce_flag(result.verdict, result.blocked_category, flag)
    merged = _merge_ingredients(ingredients, result.ingredients)
    edit_kind = "structural" if merged != ingredients else result.edit_kind
    result = result.model_copy(
        update={
            "verdict": verdict,
            "blocked_category": category,
            "ingredients": merged,
            "edit_kind": edit_kind,
            "suggestion": _clean_suggestion(verdict, result.suggestion),
        }
    )
    if verdict != "story":
        return EditScreening(result, target_words, [], call.usage)

    notes: list[str] = []
    new_target = target_words
    if result.new_target_words > 0:
        new_target = min(max(result.new_target_words, MIN_WORDS), MAX_WORDS)
        if new_target != result.new_target_words:
            bound = "longest" if new_target == MAX_WORDS else "shortest"
            notes.append(ui.LENGTH_CLAMPED_NOTE.format(words=new_target, bound=bound))
    if result.drops_story_step:
        notes.append(ui.STEP_SOFTENED_NOTE)
    if result.asks_age_change:
        notes.append(ui.AGE_FIXED_NOTE)
    return EditScreening(result, new_target, notes, call.usage)


def _declined_edit(
    verdict: Verdict, category: BlockedCategory, ingredients: Ingredients, suggestion: str
) -> EditIntake:
    return EditIntake(
        verdict=verdict,
        blocked_category=category,
        edit_kind="surface",
        ingredients=ingredients,
        instruction="",
        new_target_words=0,
        drops_story_step=False,
        asks_age_change=False,
        suggestion=suggestion,
    )


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

_YOU_PICK_RE = re.compile(
    r"^(you\s*pick|you\s*choose|surprise\s*me|anything|whatever|any|dunno|i\s*don'?t\s*know"
    r"|random|doesn'?t\s*matter|does\s*not\s*matter|no\s*preference|up\s*to\s*you|your\s*choice"
    r"|i\s*don'?t\s*care)[.!]*$",
    re.IGNORECASE,
)


def _enforce_flag(
    verdict: Verdict, category: BlockedCategory, flag: Flag | None
) -> tuple[Verdict, BlockedCategory]:
    """Make the verdict and category consistent; a pre-filter block can't be overturned."""
    if flag and verdict in ("story", "off_task"):
        return "blocked", flag.category
    if verdict == "story":
        return verdict, "none"
    if verdict == "off_task":
        return verdict, "prompt_injection" if category == "prompt_injection" else "none"
    if verdict == "public_figure":
        return verdict, "public_figure"
    if verdict == "blocked" and category in ("none", "public_figure", "prompt_injection"):
        return verdict, flag.category if flag else "none"
    return verdict, category


def _clean_suggestion(verdict: Verdict, suggestion: str) -> str:
    if verdict not in ("blocked", "public_figure"):
        return ""
    cleaned = suggestion.strip().rstrip(".!?").strip()
    return cleaned or ui.FALLBACK_SUGGESTION


def _normalize_value(value: str) -> str:
    value = value.strip()
    return YOU_PICK if _YOU_PICK_RE.match(value) else value


def _merge_ingredients(known: Ingredients, new: Ingredients) -> Ingredients:
    """New non-empty values win; empty ones fall back to what we already knew."""
    merged = {}
    for name in INGREDIENT_NAMES:
        value = _normalize_value(getattr(new, name))
        merged[name] = value or getattr(known, name)
    return Ingredients(**merged)


def _empty_ingredients() -> Ingredients:
    return Ingredients(character="", setting="", background="", moral="")


def _ingredients_json(ingredients: Ingredients) -> str:
    return json.dumps(ingredients.model_dump(), indent=2)
