"""Every fixed piece of text the user reads, and the short replies we recognise.

The tone is warm and never accusatory. Templates use ``str.format`` fields,
filled in where they are shown. Settings and limits live in ``constants.py``;
prompts for the model live in ``prompts.py``.
"""

from __future__ import annotations

from schemas import IngredientName

# --------------------------------------------------------------------------- #
# CLI questions and sign-offs
# --------------------------------------------------------------------------- #

EXAMPLE_REQUEST = "A story about a girl named Alice and her best friend Bob, who happens to be a cat."

REQUEST_QUESTION = "What kind of story do you want to hear? "
REQUEST_HINT = f'Tell me a little about the story, for example: "{EXAMPLE_REQUEST}"'
AGE_QUESTION = "How old is the listener?"
WORDS_QUESTION = "How many words?"
NUMBER_QUESTION = "{question} ({low}-{high}, Enter for {default}): "
NUMBER_HINT = "Please type a whole number from {low} to {high}."
REPLY_PROMPT = "> "

DECLINE_HINT_WITH_SUGGESTION = 'Type "y" to hear that one, a new idea, or press Enter to stop.'
DECLINE_HINT = "Type a new idea, or press Enter to stop."

GOODBYE_AFTER_STORY = "Goodnight, sleep tight!"
GOODBYE_WITHOUT_STORY = "Okay, sweet dreams!"
GOODBYE_ON_INTERRUPT = "Goodnight!"
UNEXPECTED_ERROR = "Something went wrong while telling the story. Please try again."

# --------------------------------------------------------------------------- #
# Progress lines (one is picked at random as each stage starts)
# --------------------------------------------------------------------------- #

STATUS_LINES: dict[str, tuple[str, ...]] = {
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
# The ingredient question
# --------------------------------------------------------------------------- #

INGREDIENT_OPENER = "Ooh, I like it! A few quick questions before I start:"
INGREDIENT_OPENER_AGAIN = "Almost there! Just tell me:"
INGREDIENT_QUESTIONS: dict[IngredientName, str] = {
    "character": "Who is the story about?",
    "setting": "Where does it happen?",
    "background": "What's going on when the story begins? (like \"Bob has never left the garden\")",
    "moral": "What should the story be about underneath, like sharing or being brave?",
}
INGREDIENT_CLOSER = (
    'Answer any or all of them. Say "you pick" for anything you want me to choose, '
    "or just press Enter and I'll pick them all."
)
PICK_THE_REST_NOTE = "No worries, I'll pick the rest!"

# --------------------------------------------------------------------------- #
# Declines
# --------------------------------------------------------------------------- #

OFF_TASK_MESSAGE = (
    "I only tell bedtime stories, give me a character and a story background. Lemme cook!"
)
PUBLIC_FIGURE_OPENER = "I only make up stories about made-up characters, not real people!"
SCARY_OPENER = "That one's a bit too scary for bedtime!"
UNSUITABLE_OPENER = "That one's not quite right for a bedtime story!"
SUGGESTION_TEMPLATE = "{opener} How about {suggestion}?"

# Used when the model refuses, or returns a decline without a suggestion.
FALLBACK_SUGGESTION = "a sleepy little owl who can't find the moon"

# --------------------------------------------------------------------------- #
# Endings and the edit loop
# --------------------------------------------------------------------------- #

APOLOGY = (
    "I'm sorry, I couldn't get that story just right for bedtime. "
    "Could you try a different idea?"
)
EDIT_APOLOGY = "Sorry, I couldn't make that change work, so the story stays as it was."
EDIT_PROMPT = (
    'Want to change anything? Tell me (like "make it funnier" or "give Bob a hat"), '
    "or press Enter if you're all done."
)
EDIT_FOLLOW_UP = "Want a different change? Tell me, or press Enter if you're all done."
EDIT_ACCEPT_HINT = 'Type "y" for that, tell me a different change, or press Enter if you\'re all done.'

LENGTH_CLAMPED_NOTE = "{words} words is the {bound} story I tell, so I'll make it {words}."
STEP_SOFTENED_NOTE = (
    "Every story needs its little bump in the road, so I'll make that part gentler instead of taking it out."
)
AGE_FIXED_NOTE = "The age stays the same for this session; start a new story to change it."

# --------------------------------------------------------------------------- #
# Short replies we recognise at the interrupts
# --------------------------------------------------------------------------- #

QUIT_WORDS = frozenset({"q", "quit", "exit", "no", "n", "bye"})
DONE_WORDS = QUIT_WORDS | {"done", "nope", "all done", "that's all", "thats all", "nothing"}
ACCEPT_WORDS = frozenset({"y", "yes", "yeah", "yep", "ok", "okay", "sure"})

# --------------------------------------------------------------------------- #
# Errors (shown as one line, without a traceback)
# --------------------------------------------------------------------------- #

MISSING_KEY_ERROR = "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
BAD_KEY_ERROR = "Your Anthropic API key was rejected. Check ANTHROPIC_API_KEY in .env."
CONNECTION_ERROR = "Couldn't reach Anthropic. Check your internet connection and try again."
API_STATUS_ERROR = "Anthropic returned an error ({status}). Please try again in a moment."
REFUSAL_ERROR = "The model declined this request."
CUT_OFF_ERROR = "The storyteller got cut off mid-story. Please try again."
UNREADABLE_ERROR = "Got an unreadable answer twice while building the {schema}. Please try again."
