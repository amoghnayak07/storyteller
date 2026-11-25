"""The story pipeline as a LangGraph ``StateGraph``.

Flow (branch 3; the judge loop and the edit interrupt come next)::

    START -> intake --story, complete--> planner -> storyteller -> finish -> END
               |  \\--story, missing----> ask_ingredients (interrupt) --> intake
               |                               \\--blank answer: "you pick" all --> planner
               \\--off_task/blocked/public_figure--> decline (interrupt)
                                                    |-- "y": the suggestion --> intake
                                                    |-- new text -----------> intake
                                                    \\-- blank/"quit" -------> END

State is the nodes' shared memory. It is never sent to the model whole: each
node builds its own prompt from the few fields it needs.

Interrupts: LangGraph re-runs a node from its start when it resumes, so each
interrupt node calls ``interrupt()`` first and does no LLM work; the LLM calls
live in their own nodes. The interrupt payload is a dict with ``kind``
("ingredients" or "decline") and the ``message`` to show the user; the CLI
resumes with the user's raw reply as a string.
"""

from __future__ import annotations

import json
import operator
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

import guardrails
import llm
import prompts
from schemas import (
    INGREDIENT_NAMES,
    YOU_PICK,
    AgeBand,
    Beat,
    BlockedCategory,
    Category,
    Ingredients,
    IngredientName,
    StoryPlan,
    Verdict,
)

DEFAULT_AGE = 7
DEFAULT_WORDS = 500

# Fallback beat weights (percent) used when the planner's budgets are unusable.
_DEFAULT_WEIGHTS = (8, 8, 10, 18, 16, 16, 10, 14)

_QUIT_WORDS = frozenset({"q", "quit", "exit", "no", "n", "bye"})
_ACCEPT_WORDS = frozenset({"y", "yes", "yeah", "yep", "ok", "okay", "sure"})


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Draft:
    """One written version of the story. The judge result joins it in branch 4."""

    text: str


class StoryState(TypedDict, total=False):
    # Fixed for the session (set by the CLI).
    age: int
    band: AgeBand
    target_words: int

    # The story request and the latest text to screen (the request itself, an
    # ingredient answer, or a new request after a decline).
    request: str
    pending_text: str

    # Intake.
    verdict: Verdict
    blocked_category: BlockedCategory
    suggestion: str
    message: str
    category: Category
    ingredients: Ingredients | None
    extras: list[str]

    # Writing.
    plan: StoryPlan
    drafts: list[Draft]
    final_story: str
    usage: Annotated[llm.Usage, operator.add]


def band_for_age(age: int) -> AgeBand:
    if not 5 <= age <= 10:
        raise ValueError(f"age must be 5-10, got {age}")
    return "5-6" if age <= 6 else "7-8" if age <= 8 else "9-10"


def initial_state(request: str, age: int = DEFAULT_AGE, target_words: int = DEFAULT_WORDS) -> StoryState:
    """The input for a fresh run. Age and length are validated by the CLI."""
    if not guardrails.MIN_WORDS <= target_words <= guardrails.MAX_WORDS:
        raise ValueError(f"target_words must be {guardrails.MIN_WORDS}-{guardrails.MAX_WORDS}, got {target_words}")
    return {
        "age": age,
        "band": band_for_age(age),
        "target_words": target_words,
        "request": request,
        "pending_text": request,
        "ingredients": None,
        "extras": [],
        "drafts": [],
        "usage": llm.Usage(),
    }


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #


def intake(state: StoryState) -> dict[str, Any]:
    """Screen the pending text: a new request, or an answer to the ingredient question."""
    known = state.get("ingredients")
    answering = known is not None
    screening = guardrails.screen_request(
        state["pending_text"],
        known_ingredients=known,
        known_category=state.get("category") if answering else None,
    )
    r = screening.result
    extras = _dedupe((state.get("extras", []) if answering else []) + r.extra_elements)
    return {
        "verdict": r.verdict,
        "blocked_category": r.blocked_category,
        "suggestion": r.suggestion,
        "message": screening.message,
        "category": r.category,
        "ingredients": r.ingredients,
        "extras": extras,
        "usage": screening.usage,
    }


def ask_ingredients(state: StoryState) -> dict[str, Any]:
    """Interrupt: ask for every missing ingredient in one friendly message."""
    ingredients = state["ingredients"]
    missing = ingredients.missing()
    answer = interrupt({"kind": "ingredients", "message": ingredient_question(missing), "missing": missing})
    answer = str(answer).strip()
    if not answer:
        # A blank reply means "you pick" for everything still missing.
        filled = ingredients.model_copy(update={name: YOU_PICK for name in missing})
        return {"ingredients": filled, "pending_text": ""}
    return {"pending_text": answer}


def decline(state: StoryState) -> dict[str, Any]:
    """Interrupt: show the decline reply; the user accepts, retries, or quits."""
    message = state["message"]
    suggestion = state.get("suggestion", "")
    reply = str(interrupt({"kind": "decline", "message": message, "suggestion": suggestion})).strip()

    if not reply or reply.lower().rstrip("!.") in _QUIT_WORDS:
        return {"pending_text": ""}
    if suggestion and reply.lower().rstrip("!.") in _ACCEPT_WORDS:
        reply = suggestion
    # A retry is a fresh request: start intake over.
    return {
        "request": reply,
        "pending_text": reply,
        "ingredients": None,
        "extras": [],
    }


def planner(state: StoryState) -> dict[str, Any]:
    user = "\n\n".join(
        [
            prompts.tag("user_request", state["request"]),
            prompts.tag("ingredients", json.dumps(state["ingredients"].model_dump(), indent=2)),
            prompts.tag("extra_elements", json.dumps(state.get("extras", []))),
            prompts.tag("category", state["category"]),
            prompts.tag("age_band", state["band"]),
            prompts.tag("target_words", str(state["target_words"])),
        ]
    )
    try:
        call = llm.generate_structured(prompts.PLANNER_SYSTEM, user, StoryPlan, effort="medium")
    except llm.LLMRefusalError:
        return _refused()
    plan = call.value.model_copy(update={"beats": rescale_budgets(call.value.beats, state["target_words"])})
    return {"plan": plan, "usage": call.usage}


def storyteller(state: StoryState) -> dict[str, Any]:
    user = "\n\n".join(
        [
            prompts.tag("story_plan", plan_json(state["plan"])),
            prompts.tag("category", state["category"]),
            prompts.tag("age_band", state["band"]),
            prompts.tag("target_words", str(state["target_words"])),
        ]
    )
    try:
        call = llm.generate_text(prompts.STORYTELLER_SYSTEM, user, effort="medium")
    except llm.LLMRefusalError:
        return _refused()
    return {"drafts": [*state.get("drafts", []), Draft(text=call.value)], "usage": call.usage}


def finish(state: StoryState) -> dict[str, Any]:
    """Pick the story to show. Branch 4 puts the judge's hard gates in front of this."""
    return {"final_story": state["drafts"][-1].text}


def _refused() -> dict[str, Any]:
    """A refusal from a writing call is treated as a block (PLAN: guardrail layer 3)."""
    suggestion = guardrails.FALLBACK_SUGGESTION
    return {
        "verdict": "blocked",
        "blocked_category": "none",
        "suggestion": suggestion,
        "message": guardrails.decline_message("blocked", "none", suggestion),
    }


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #


def route_intake(state: StoryState) -> Literal["decline", "ask_ingredients", "planner"]:
    if state["verdict"] != "story":
        return "decline"
    if state["ingredients"].missing():
        return "ask_ingredients"
    return "planner"


def route_ingredient_answer(state: StoryState) -> Literal["intake", "planner"]:
    return "intake" if state["pending_text"] else "planner"


def route_decline(state: StoryState) -> Literal["intake", "__end__"]:
    return "intake" if state["pending_text"] else END


def route_after_write(state: StoryState) -> Literal["decline", "continue"]:
    return "decline" if state.get("verdict") == "blocked" else "continue"


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #

# Our own types that live in checkpointed state, allowed explicitly so the
# checkpointer never deserializes anything else.
_STATE_TYPES = [
    ("llm", "Usage"),
    ("pipeline", "Draft"),
    ("schemas", "Ingredients"),
    ("schemas", "StoryPlan"),
    ("schemas", "Beat"),
]


def build_graph(checkpointer: Any = None) -> Any:
    """Compile the graph. Pass a checkpointer to share one; defaults to in-memory."""
    graph = StateGraph(StoryState)
    graph.add_node("intake", intake)
    graph.add_node("ask_ingredients", ask_ingredients)
    graph.add_node("decline", decline)
    graph.add_node("planner", planner)
    graph.add_node("storyteller", storyteller)
    graph.add_node("finish", finish)

    graph.add_edge(START, "intake")
    graph.add_conditional_edges("intake", route_intake)
    graph.add_conditional_edges("ask_ingredients", route_ingredient_answer)
    graph.add_conditional_edges("decline", route_decline)
    graph.add_conditional_edges("planner", route_after_write, {"decline": "decline", "continue": "storyteller"})
    graph.add_conditional_edges("storyteller", route_after_write, {"decline": "decline", "continue": "finish"})
    graph.add_edge("finish", END)

    if checkpointer is None:
        checkpointer = InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=_STATE_TYPES))
    return graph.compile(checkpointer=checkpointer)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_QUESTIONS: dict[IngredientName, str] = {
    "character": "Who is the story about?",
    "setting": "Where does it happen?",
    "background": "What is their everyday life like when the story starts?",
    "moral": "What should the story be about underneath, like sharing or being brave?",
}


def ingredient_question(missing: list[IngredientName]) -> str:
    """One friendly message asking for every missing ingredient."""
    lines = ["Ooh, I like it! A few quick questions before I start:"]
    lines += [f"  - {_QUESTIONS[name]}" for name in INGREDIENT_NAMES if name in missing]
    lines.append('Answer any or all of them. Say "you pick" for anything you want me to choose, '
                 "or just press Enter and I'll pick them all.")
    return "\n".join(lines)


def rescale_budgets(beats: list[Beat], target: int) -> list[Beat]:
    """Scale the beats' word budgets so they sum exactly to ``target``.

    Uses largest-remainder rounding so the total is exact. If the planner's
    budgets are unusable (all zero or negative), falls back to default weights.
    """
    weights = [max(beat.word_budget, 0) for beat in beats]
    if sum(weights) == 0:
        weights = list(_DEFAULT_WEIGHTS)
    total = sum(weights)
    raw = [w * target / total for w in weights]
    budgets = [int(x) for x in raw]
    by_remainder = sorted(range(len(raw)), key=lambda i: raw[i] - budgets[i], reverse=True)
    for i in by_remainder[: target - sum(budgets)]:
        budgets[i] += 1
    return [beat.model_copy(update={"word_budget": b}) for beat, b in zip(beats, budgets)]


def plan_json(plan: StoryPlan) -> str:
    return json.dumps(plan.model_dump(), indent=2)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out
