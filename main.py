"""
Before submitting the assignment, describe here in a few sentences what you would have built next if you spent 2 more hours on this project:

"""

# The CLI is a thin driver: it collects the request, age and length, runs the
# graph in pipeline.py, and answers each interrupt with what the user types.

from __future__ import annotations

import sys
import uuid

from langgraph.types import Command

import guardrails
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


def run_story(request: str, age: int, target_words: int) -> None:
    graph = pipeline.build_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = graph.invoke(pipeline.initial_state(request, age, target_words), config)

    while result.get("__interrupt__"):
        payload = result["__interrupt__"][0].value
        print()
        print(payload["message"])
        if payload["kind"] == "decline":
            hint = 'Type "y" to hear that one, ' if payload.get("suggestion") else "Type "
            print(f"{hint}a new idea, or press Enter to stop.")
        reply = ask("> ")
        result = graph.invoke(Command(resume=reply), config)

    story = result.get("final_story")
    if not story:
        print("\nOkay, sweet dreams!")
        return
    for note in result.get("notes", []):
        print(f"\n{note}")
    print()
    print(result["plan"].title)
    print()
    print(story)


def main() -> int:
    try:
        request = ask_request()
        age = ask_number("How old is the listener?", 5, 10, pipeline.DEFAULT_AGE)
        words = ask_number("How many words?", guardrails.MIN_WORDS, guardrails.MAX_WORDS, pipeline.DEFAULT_WORDS)
        print("Writing your story...")
        run_story(request, age, words)
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
