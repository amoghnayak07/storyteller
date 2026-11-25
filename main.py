"""
Before submitting the assignment, describe here in a few sentences what you would have built next if you spent 2 more hours on this project:

"""

# The CLI is a thin driver: it collects the request, age and length, runs the
# graph in pipeline.py, prints the progress lines the nodes stream, and answers
# each interrupt with what the user types. --verbose adds a trace of every node;
# --diagram prints the graph's Mermaid topology and exits without an API call.

from __future__ import annotations

import argparse
import sys
import time
import uuid
from typing import Any

from langgraph.types import Command

import guardrails
import judge
import llm
import pipeline

example_requests = "A story about a girl named Alice and her best friend Bob, who happens to be a cat."


def ask(prompt: str) -> str:
    """Read one line. Ctrl+C / Ctrl+D raise ``KeyboardInterrupt`` for a clean exit."""
    try:
        return input(prompt).strip()
    except EOFError:
        raise KeyboardInterrupt from None


def ask_request() -> str:
    while True:
        request = ask("What kind of story do you want to hear? ")
        if request:
            return request
        print(f'Tell me a little about the story, for example: "{example_requests}"')


def ask_number(prompt: str, low: int, high: int, default: int) -> int:
    while True:
        answer = ask(f"{prompt} ({low}-{high}, Enter for {default}): ")
        if not answer:
            return default
        if answer.isdigit() and low <= int(answer) <= high:
            return int(answer)
        print(f"Please type a whole number from {low} to {high}.")


# --------------------------------------------------------------------------- #
# Running the graph
# --------------------------------------------------------------------------- #


def run_story(request: str, age: int, target_words: int, *, verbose: bool = False) -> None:
    graph = pipeline.build_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    command: Any = pipeline.initial_state(request, age, target_words)

    while True:
        for mode, chunk in graph.stream(command, config, stream_mode=["updates", "custom"]):
            if mode == "custom":
                print(chunk["status"])
            elif verbose:
                for node, update in chunk.items():
                    if node not in _UNTRACED and update:
                        print("\n".join(trace_lines(node, update)))
        snapshot = graph.get_state(config)
        if not snapshot.interrupts:
            break
        command = Command(resume=answer_interrupt(snapshot.interrupts[0].value))

    values = snapshot.values
    if values.get("apology"):
        print(f"\n{values['apology']}")
    elif values.get("final_story"):
        print("\nGoodnight, sleep tight!")
    else:
        print("\nOkay, sweet dreams!")


def answer_interrupt(payload: dict[str, Any]) -> str:
    """Show an interrupt's message (and the story, when there's a new one) and read the reply."""
    print()
    if payload["kind"] == "edit" and payload.get("new_story"):
        for note in payload.get("notes", []):
            print(note)
            print()
        print(payload["title"])
        print()
        typewrite(payload["story"])
        print()
    print(payload["message"])
    if payload["kind"] == "decline":
        hint = 'Type "y" to hear that one, ' if payload.get("suggestion") else "Type "
        print(f"{hint}a new idea, or press Enter to stop.")
    return ask("> ")


# Seconds per word for the typewriter replay.
TYPEWRITER_DELAY = 0.03


def typewrite(text: str, *, delay: float = TYPEWRITER_DELAY) -> None:
    """Replay an approved story word by word, like it is being told.

    Only stories that already passed the gates reach here, so nothing unjudged
    is ever streamed. When output is not a terminal (piped, or under tests) the
    story prints at once.
    """
    if not sys.stdout.isatty() or delay <= 0:
        print(text)
        return
    for line in text.split("\n"):
        words = line.split(" ")
        for i, word in enumerate(words):
            sys.stdout.write(word + (" " if i < len(words) - 1 else ""))
            sys.stdout.flush()
            if word:
                time.sleep(delay)
        sys.stdout.write("\n")
    sys.stdout.flush()


def print_diagram() -> None:
    """Print the compiled graph's topology as Mermaid (no API key needed)."""
    print(pipeline.build_graph().get_graph().draw_mermaid())


# --------------------------------------------------------------------------- #
# --verbose trace
# --------------------------------------------------------------------------- #


# Interrupt nodes only record the user's reply, which the user just typed.
_UNTRACED = frozenset({"__interrupt__", "ask_ingredients", "decline", "ask_edit"})


def trace_lines(node: str, update: dict[str, Any]) -> list[str]:
    """Readable trace lines for one node's state update."""
    lines = [f"  [{node}]"]
    if node in ("intake", "edit_intake") and "verdict" in update:
        if node == "intake":
            lines.append(f"    verdict: {update['verdict']}  category: {update['category']}  "
                         f"blocked: {update['blocked_category']}")
        else:
            lines.append(f"    verdict: {update['verdict']}")
        if update.get("ingredients") is not None:
            lines.append(f"    ingredients: {update['ingredients'].model_dump()}")
        if update.get("extras"):
            lines.append(f"    extras: {update['extras']}")
        if update.get("edit_kind"):
            lines.append(f"    edit: {update['edit_kind']}: {update['edit_instruction']} "
                         f"(target {update['target_words']} words)")
        for note in update.get("notes", []):
            lines.append(f"    note: {note}")
    elif node == "planner" and "plan" in update:
        plan = update["plan"]
        lines.append(f"    title: {plan.title}")
        lines.append(f"    ingredients: {plan.ingredients.model_dump()}")
        for beat in plan.beats:
            lines.append(f"    {beat.step:<7} {beat.word_budget:>3}w  {beat.summary}")
    elif node in ("storyteller", "reviser") and "candidate" in update:
        if update["candidate"]:
            round_ = ""
            if node == "reviser":
                count = update["revision_count"]
                round_ = f" (revision {count})" if count else " (the requested change)"
            lines.append(f"    wrote a draft of {judge.word_count(update['candidate'])} words{round_}")
        else:
            lines.append("    declined to write; stopping revisions")
    elif node == "judge" and update.get("drafts"):
        lines += _judge_lines(update["drafts"][-1], len(update["drafts"]))
    elif node == "regenerate":
        lines.append("    no draft passed the hard gates; writing once more from the plan")
    elif node == "finish":
        lines.append("    showing the best draft that passed the hard gates"
                     if update.get("story_fresh") else "    no draft passed the hard gates")

    usage = update.get("usage")
    if usage and usage.calls:
        lines.append(f"    tokens: in {usage.input_tokens}, out {usage.output_tokens}, "
                     f"cache read {usage.cache_read_tokens}, cache write {usage.cache_write_tokens} "
                     f"({usage.calls} call{'s' if usage.calls != 1 else ''})")
    return lines


def _judge_lines(draft: pipeline.Draft, number: int) -> list[str]:
    c = draft.checks
    lines = [f"    draft {number}"]
    if draft.review is None:
        lines.append("    judge refused: counted as unsafe")
    else:
        r = draft.review
        lines.append("    scores: " + ", ".join(f"{name} {s.score}" for name, s in r.scores().items()))
        missing = judge.missing_steps(r)
        lines.append(f"    formula: {'all 5 steps' if not missing else 'missing ' + ', '.join(missing)}")
        lines.append(f"    safe: {r.safe}" + (f" ({r.safety_issue})" if r.safety_issue else ""))
    lines.append(f"    checks: {c.word_count}/{c.target_words} words, FK grade {c.fk_grade} "
                 f"(target <= {judge.READABILITY_TARGETS[c.band]:.0f}), "
                 f"moralizing {c.moralizing or 'none'}, banned {c.banned or 'none'}")
    outcome = "PASS" if draft.passed else "fails rubric (hard gates ok)" if draft.gated else "FAILS hard gates"
    lines.append(f"    result: {outcome}")
    return lines


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tell a bedtime story for ages 5-10.")
    parser.add_argument("--verbose", action="store_true",
                        help="print classification, plan, judge scores, checks and token usage")
    parser.add_argument("--diagram", action="store_true",
                        help="print the LangGraph topology as Mermaid and exit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.diagram:
        print_diagram()
        return 0
    try:
        request = ask_request()
        age = ask_number("How old is the listener?", 5, 10, pipeline.DEFAULT_AGE)
        words = ask_number("How many words?", guardrails.MIN_WORDS, guardrails.MAX_WORDS, pipeline.DEFAULT_WORDS)
        run_story(request, age, words, verbose=args.verbose)
        return 0
    except KeyboardInterrupt:
        print("\nGoodnight!")
        return 0
    except llm.LLMError as exc:
        print(f"\n{exc}")
        return 1
    except Exception:
        print("\nSomething went wrong while telling the story. Please try again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
