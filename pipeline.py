"""The story pipeline as a LangGraph ``StateGraph``.

Flow::

    START -> intake --story, complete--> planner -> storyteller -> judge --pass--> finish
               |  \\--story, missing----> ask_ingredients (interrupt) --> intake
               |                               (two answers in a row that fill nothing new:
               |                                intake picks the rest and goes on to the planner)
               |                               \\--blank answer: "you pick" all --> planner
               \\--off_task/blocked/public_figure--> decline (interrupt)
                                                    |-- "y": the suggestion --> intake
                                                    |-- new text -----------> intake
                                                    \\-- blank/"quit" -------> END

    judge --fail, < 3 revisions--> reviser --> judge
    judge --fail, 3 revisions, no draft passed the hard gates--> regenerate --> storyteller (once)
    judge --fail otherwise--> finish (best draft that passed the hard gates, or an apology)

    finish --story--> ask_edit (interrupt) --blank/"done"--> END
                         \\--change--> edit_intake --structural--> planner
                                                  |--surface----> reviser
                                                  \\--declined----> ask_edit

State is the nodes' shared memory. It is never sent to the model whole: each
node builds its own prompt from the few fields it needs. The judge never sees
the plan.

Interrupts: LangGraph re-runs a node from its start when it resumes, so each
interrupt node calls ``interrupt()`` first and does no LLM work; the LLM calls
live in their own nodes. The interrupt payload is a dict with ``kind``
("ingredients", "decline" or "edit") and the ``message`` to show the user; the
CLI resumes with the user's raw reply as a string.

Progress: each working node emits a playful ``{"status": ...}`` line on the
``custom`` stream as it starts, which the CLI prints. Only a story that passed
the gates ever reaches the user, through the edit interrupt.
"""

from __future__ import annotations

import json
import operator
import random
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

import guardrails
import judge
import llm
import prompts
from schemas import (
    INGREDIENT_NAMES,
    YOU_PICK,
    AgeBand,
    Beat,
    BlockedCategory,
    Category,
    EditKind,
    Ingredients,
    IngredientName,
    JudgeResult,
    StoryPlan,
    Verdict,
)

DEFAULT_AGE = 7
DEFAULT_WORDS = 500

# Fallback beat weights (percent) used when the planner's budgets are unusable.
_DEFAULT_WEIGHTS = (8, 8, 10, 18, 16, 16, 10, 14)

MAX_FRUITLESS_ANSWERS = 2
PICK_THE_REST_NOTE = "No worries, I'll pick the rest!"

MAX_REVISIONS = 3

APOLOGY = (
    "I'm sorry, I couldn't get that story just right for bedtime. "
    "Could you try a different idea?"
)
EDIT_APOLOGY = "Sorry, I couldn't make that change work, so the story stays as it was."
EDIT_PROMPT = (
    'Want to change anything? Tell me (like "make it funnier" or "give Bob a hat"), '
    "or press Enter if you're all done."
)
_EDIT_FOLLOW_UP = "Want a different change? Tell me, or press Enter if you're all done."
_EDIT_ACCEPT_HINT = 'Type "y" for that, tell me a different change, or press Enter if you\'re all done.'

_QUIT_WORDS = frozenset({"q", "quit", "exit", "no", "n", "bye"})
_DONE_WORDS = _QUIT_WORDS | {"done", "nope", "all done", "that's all", "thats all", "nothing"}
_ACCEPT_WORDS = frozenset({"y", "yes", "yeah", "yep", "ok", "okay", "sure"})

_STATUS: dict[str, tuple[str, ...]] = {
    "intake": ("Reading your request...", "Peeking at your idea...", "Unrolling your request..."),
    "planner": ("Sketching the plot...", "Drawing a little story map...", "Plotting the twists and turns..."),
    "storyteller": ("Writing...", "Dipping the quill in moonlight...", "Spinning the story..."),
    "judge": (
        "The judge is peering over its spectacles...",
        "A very picky owl is reading every word...",
        "Checking the story from top to tail...",
    ),
    "reviser": ("Polishing...", "Smoothing out the wrinkles...", "Sprinkling in a little more sparkle..."),
    "edit_intake": ("Reading your change...", "Thinking about your change...", "Sharpening the pencil..."),
}


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Draft:
    """One written version of the story, with the judge's review and the code checks.

    ``review`` is ``None`` when the judge refused, which counts as unsafe.
    """

    text: str
    review: JudgeResult | None
    checks: judge.Checks

    @property
    def gated(self) -> bool:
        """Passed the hard gates: safe and within 300-800 words. Only these may be shown."""
        return judge.passes_hard_gates(self.review, self.checks)

    @property
    def passed(self) -> bool:
        return judge.passes(self.review, self.checks)

    @property
    def rank(self) -> tuple:
        return judge.rank(self.review, self.checks)


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
    asked_for: list[IngredientName]
    fruitless_answers: int
    """Consecutive ingredient answers that filled nothing new."""
    notes: list[str]
    """One-line notes shown to the user with the story."""

    # Writing and judging. ``drafts`` holds the current round only (it resets
    # for each edit), so the best draft is always picked from one round.
    plan: StoryPlan
    candidate: str
    """The newest text from the storyteller or reviser, waiting for the judge."""
    drafts: list[Draft]
    revision_count: int
    regenerated: bool

    # The story the user has seen, and the edit loop.
    final_story: str
    final_plan: StoryPlan
    """The plan behind ``final_story``; restored if an edit fails."""
    story_fresh: bool
    """True from ``finish`` until the edit interrupt has shown the story."""
    apology: str
    edit_kind: EditKind
    edit_instruction: str
    edit_history: list[str]
    edit_message: str
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
        "asked_for": [],
        "fruitless_answers": 0,
        "notes": [],
        "candidate": "",
        "drafts": [],
        "revision_count": 0,
        "regenerated": False,
        "final_story": "",
        "story_fresh": False,
        "apology": "",
        "edit_instruction": "",
        "edit_history": [],
        "edit_message": "",
        "usage": llm.Usage(),
    }


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #


def intake(state: StoryState) -> dict[str, Any]:
    """Screen the pending text: a new request, or an answer to the ingredient question."""
    _status("intake")
    known = state.get("ingredients")
    answering = known is not None
    screening = guardrails.screen_request(
        state["pending_text"],
        known_ingredients=known,
        known_category=state.get("category") if answering else None,
        asked_for=state.get("asked_for") if answering else None,
    )
    r = screening.result
    extras = _dedupe((state.get("extras", []) if answering else []) + r.extra_elements)
    ingredients = r.ingredients
    notes = list(state.get("notes", [])) if answering else []

    # An answer that fills nothing new gets one more ask; after the second,
    # stop asking and let the planner pick the rest.
    fruitless = 0
    if answering and r.verdict == "story":
        made_progress = len(ingredients.missing()) < len(known.missing())
        fruitless = 0 if made_progress else state.get("fruitless_answers", 0) + 1
        if fruitless >= MAX_FRUITLESS_ANSWERS and ingredients.missing():
            ingredients = ingredients.model_copy(update={name: YOU_PICK for name in ingredients.missing()})
            notes.append(PICK_THE_REST_NOTE)

    return {
        "verdict": r.verdict,
        "blocked_category": r.blocked_category,
        "suggestion": r.suggestion,
        "message": screening.message,
        "category": r.category,
        "ingredients": ingredients,
        "extras": extras,
        "fruitless_answers": fruitless,
        "notes": notes,
        "usage": screening.usage,
    }


def ask_ingredients(state: StoryState) -> dict[str, Any]:
    """Interrupt: ask for every missing ingredient in one friendly message."""
    ingredients = state["ingredients"]
    missing = ingredients.missing()
    question = ingredient_question(missing, again=bool(state.get("asked_for")))
    answer = interrupt({"kind": "ingredients", "message": question, "missing": missing})
    answer = str(answer).strip()
    if not answer:
        # A blank reply means "you pick" for everything still missing.
        filled = ingredients.model_copy(update={name: YOU_PICK for name in missing})
        return {"ingredients": filled, "pending_text": ""}
    return {"pending_text": answer, "asked_for": missing}


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
        "asked_for": [],
        "fruitless_answers": 0,
        "candidate": "",
        "drafts": [],
        "revision_count": 0,
        "regenerated": False,
    }


def planner(state: StoryState) -> dict[str, Any]:
    _status("planner")
    parts = [
        prompts.tag("user_request", state["request"]),
        prompts.tag("ingredients", json.dumps(state["ingredients"].model_dump(), indent=2)),
        prompts.tag("extra_elements", json.dumps(state.get("extras", []))),
    ]
    if state.get("edit_history"):
        parts.append(prompts.tag("edit_requests", json.dumps(state["edit_history"], indent=2)))
    parts += [
        prompts.tag("category", state["category"]),
        prompts.tag("age_band", state["band"]),
        prompts.tag("target_words", str(state["target_words"])),
    ]
    try:
        call = llm.generate_structured(prompts.PLANNER_SYSTEM, "\n\n".join(parts), StoryPlan, effort="medium")
    except llm.LLMRefusalError:
        return _refused()
    plan = call.value.model_copy(update={"beats": rescale_budgets(call.value.beats, state["target_words"])})
    return {"plan": plan, "usage": call.usage}


def storyteller(state: StoryState) -> dict[str, Any]:
    _status("storyteller")
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
    return {"candidate": call.value, "usage": call.usage}


def judge_draft(state: StoryState) -> dict[str, Any]:
    """Grade the candidate: code checks plus the LLM judge. A judge refusal counts as unsafe."""
    _status("judge")
    text = state["candidate"]
    checks = judge.run_checks(text, state["band"], state["target_words"])
    review: JudgeResult | None = None
    usage = llm.Usage()
    try:
        call = judge.judge_story(
            text,
            request=state["request"],
            ingredients=state["ingredients"],
            extras=state.get("extras", []),
            band=state["band"],
            target_words=state["target_words"],
            edit_requests=state.get("edit_history", []),
        )
        review, usage = call.value, call.usage
    except llm.LLMRefusalError:
        pass
    draft = Draft(text=text, review=review, checks=checks)
    return {"drafts": [*state.get("drafts", []), draft], "candidate": "", "usage": usage}


def reviser(state: StoryState) -> dict[str, Any]:
    """Guided full rewrite of the latest draft, or of the shown story for a surface edit."""
    _status("reviser")
    drafts = state.get("drafts", [])
    if drafts:
        latest = drafts[-1]
        source, fixes = latest.text, judge.revision_notes(latest.review, latest.checks)
    else:
        source, fixes = state["final_story"], []

    parts = [
        prompts.tag("story_plan", plan_json(state["plan"])),
        prompts.tag("draft", source),
        prompts.tag("fixes", "\n".join(f"- {fix}" for fix in fixes) or "(none: apply the edit instruction)"),
    ]
    if state.get("edit_instruction"):
        parts.append(prompts.tag("edit_instruction", state["edit_instruction"]))
    parts += [
        prompts.tag("category", state["category"]),
        prompts.tag("age_band", state["band"]),
        prompts.tag("target_words", str(state["target_words"])),
    ]
    count = state.get("revision_count", 0) + (1 if drafts else 0)
    try:
        call = llm.generate_text(prompts.REVISER_SYSTEM, "\n\n".join(parts), effort="medium")
    except llm.LLMRefusalError:
        # Stop revising; finish picks the best draft so far (or apologizes).
        return {"candidate": "", "revision_count": count}
    return {"candidate": call.value, "revision_count": count, "usage": call.usage}


def regenerate(state: StoryState) -> dict[str, Any]:
    """No draft passed the hard gates after every revision: write once more from the plan."""
    return {"regenerated": True}


def finish(state: StoryState) -> dict[str, Any]:
    """Show the best draft that passed the hard gates; never an unsafe one."""
    shown = [d for d in state.get("drafts", []) if d.gated]
    if shown:
        best = max(shown, key=lambda d: d.rank)
        return {
            "final_story": best.text,
            "final_plan": state["plan"],
            "story_fresh": True,
            "edit_message": "",
            "suggestion": "",
        }
    if state.get("final_story"):
        # An edit that couldn't be made safely: keep the story the user already has.
        plan = state["final_plan"]
        return {
            "plan": plan,
            "ingredients": plan.ingredients,
            "target_words": sum(beat.word_budget for beat in plan.beats),
            "edit_history": state.get("edit_history", [])[:-1],
            "story_fresh": False,
            "edit_message": EDIT_APOLOGY,
            "suggestion": "",
        }
    return {"apology": APOLOGY}


def ask_edit(state: StoryState) -> dict[str, Any]:
    """Interrupt: show the story (when new) and ask for a change."""
    fresh = state.get("story_fresh", False)
    suggestion = state.get("suggestion", "")
    reason = state.get("edit_message", "")
    if not reason:
        message = EDIT_PROMPT
    else:
        message = f"{reason} {_EDIT_ACCEPT_HINT if suggestion else _EDIT_FOLLOW_UP}"
    reply = str(
        interrupt(
            {
                "kind": "edit",
                "message": message,
                "suggestion": suggestion,
                "new_story": fresh,
                "title": state["final_plan"].title,
                "story": state["final_story"],
                "notes": list(state.get("notes", [])) if fresh else [],
            }
        )
    ).strip()

    cleared = {"story_fresh": False, "edit_message": "", "suggestion": ""}
    word = reply.lower().rstrip("!.")
    if not reply or word in _DONE_WORDS:
        return {**cleared, "pending_text": ""}
    if suggestion and word in _ACCEPT_WORDS:
        reply = suggestion
    return {**cleared, "pending_text": reply}


def edit_intake(state: StoryState) -> dict[str, Any]:
    """Screen and classify a change request, then set up a fresh judge round."""
    _status("edit_intake")
    plan = state["final_plan"]
    screening = guardrails.screen_edit(
        state["pending_text"], ingredients=plan.ingredients, target_words=state["target_words"]
    )
    r = screening.result
    if r.verdict != "story":
        return {
            "verdict": r.verdict,
            "edit_message": screening.message,
            "suggestion": r.suggestion,
            "usage": screening.usage,
        }
    target = screening.target_words
    if target != state["target_words"]:
        plan = plan.model_copy(update={"beats": rescale_budgets(plan.beats, target)})
    return {
        "verdict": "story",
        "edit_kind": r.edit_kind,
        "edit_instruction": r.instruction,
        "edit_history": [*state.get("edit_history", []), r.instruction],
        "ingredients": r.ingredients,
        "target_words": target,
        "plan": plan,
        "notes": screening.notes,
        "candidate": "",
        "drafts": [],
        "revision_count": 0,
        "regenerated": False,
        "suggestion": "",
        "usage": screening.usage,
    }


def _status(stage: str) -> None:
    """Emit a playful progress line on the custom stream (ignored by ``invoke``)."""
    get_stream_writer()({"status": random.choice(_STATUS[stage])})


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


def route_after_write(state: StoryState) -> Literal["decline", "finish", "continue"]:
    """After the planner or storyteller: a refusal declines a new request, or ends an edit round."""
    if state.get("verdict") != "blocked":
        return "continue"
    return "finish" if state.get("final_story") else "decline"


def route_after_revise(state: StoryState) -> Literal["judge", "finish"]:
    return "judge" if state.get("candidate") else "finish"


def route_judge(state: StoryState) -> Literal["finish", "reviser", "regenerate"]:
    drafts = state["drafts"]
    if drafts[-1].passed:
        return "finish"
    if state.get("revision_count", 0) < MAX_REVISIONS and not state.get("regenerated"):
        return "reviser"
    if any(d.gated for d in drafts) or state.get("regenerated"):
        return "finish"
    # A surface edit has no plan-only story to fall back on; keep the shown story.
    if state.get("final_story") and state.get("edit_kind") == "surface":
        return "finish"
    return "regenerate"


def route_finish(state: StoryState) -> Literal["ask_edit", "__end__"]:
    return "ask_edit" if state.get("final_story") else END


def route_ask_edit(state: StoryState) -> Literal["edit_intake", "__end__"]:
    return "edit_intake" if state["pending_text"] else END


def route_edit_intake(state: StoryState) -> Literal["ask_edit", "planner", "reviser"]:
    if state["verdict"] != "story":
        return "ask_edit"
    return "planner" if state["edit_kind"] == "structural" else "reviser"


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #

# Our own types that live in checkpointed state, allowed explicitly so the
# checkpointer never deserializes anything else.
_STATE_TYPES = [
    ("llm", "Usage"),
    ("pipeline", "Draft"),
    ("judge", "Checks"),
    ("schemas", "Ingredients"),
    ("schemas", "StoryPlan"),
    ("schemas", "Beat"),
    ("schemas", "JudgeResult"),
    ("schemas", "CriterionScore"),
    ("schemas", "StepCheck"),
]


def build_graph(checkpointer: Any = None) -> Any:
    """Compile the graph. Pass a checkpointer to share one; defaults to in-memory."""
    graph = StateGraph(StoryState)
    graph.add_node("intake", intake)
    graph.add_node("ask_ingredients", ask_ingredients)
    graph.add_node("decline", decline)
    graph.add_node("planner", planner)
    graph.add_node("storyteller", storyteller)
    graph.add_node("judge", judge_draft)
    graph.add_node("reviser", reviser)
    graph.add_node("regenerate", regenerate)
    graph.add_node("finish", finish)
    graph.add_node("ask_edit", ask_edit)
    graph.add_node("edit_intake", edit_intake)

    graph.add_edge(START, "intake")
    graph.add_conditional_edges("intake", route_intake)
    graph.add_conditional_edges("ask_ingredients", route_ingredient_answer)
    graph.add_conditional_edges("decline", route_decline)
    graph.add_conditional_edges(
        "planner", route_after_write, {"decline": "decline", "finish": "finish", "continue": "storyteller"}
    )
    graph.add_conditional_edges(
        "storyteller", route_after_write, {"decline": "decline", "finish": "finish", "continue": "judge"}
    )
    graph.add_conditional_edges("judge", route_judge)
    graph.add_conditional_edges("reviser", route_after_revise)
    graph.add_edge("regenerate", "storyteller")
    graph.add_conditional_edges("finish", route_finish)
    graph.add_conditional_edges("ask_edit", route_ask_edit)
    graph.add_conditional_edges("edit_intake", route_edit_intake)

    if checkpointer is None:
        checkpointer = InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=_STATE_TYPES))
    return graph.compile(checkpointer=checkpointer)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_QUESTIONS: dict[IngredientName, str] = {
    "character": "Who is the story about?",
    "setting": "Where does it happen?",
    "background": "What's going on when the story begins? (like \"Bob has never left the garden\")",
    "moral": "What should the story be about underneath, like sharing or being brave?",
}


def ingredient_question(missing: list[IngredientName], *, again: bool = False) -> str:
    """One friendly message asking for every missing ingredient."""
    opener = "Almost there! Just tell me:" if again else "Ooh, I like it! A few quick questions before I start:"
    lines = [opener]
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
