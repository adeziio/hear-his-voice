"""
Verifies the episode title and summary.

Episode 1 is the regression this file exists for. It produced
"Genealogy of Light (Matthew 1:1-9)" with the summary "The lineage of
Jesus Christ traces back through generations, revealing a divine ancestry
rooted in faith and promise." Three faults: the title invented a symbol the
passage never mentions, the summary asserted theology the genealogy does
not state, and the two said the same thing twice.

Some of that is only fixable in the prompt - no code can tell that "Light"
was invented. What code CAN guarantee is checked here: the reference
appears exactly once and in the title, a summary that merely echoes the
title is replaced, and a missing title or summary falls back to the
passage's own words rather than something made up.

Run with:  python -m pytest tests/test_episode_metadata.py -v
"""

import sys

from pathlib import Path


ROOT = Path(
    __file__
).resolve().parent.parent

sys.path.insert(
    0,
    str(ROOT)
)


from ai.content_generator import (
    DIRECTION_SCHEMA,
    ContentGenerator,
    _first_sentence,
    _restates_title,
    _strip_reference,
)
from core.config_loader import ConfigLoader


# Matthew 1:1-9, the passage episode 1 actually read.
EPISODE_ONE = {
    "text": (
        "The book of the genealogy of Jesus Christ, the son of David, "
        "the son of Abraham. Abraham became the father of Isaac. Isaac "
        "became the father of Jacob. Jacob became the father of Judah and "
        "his brothers."
    ),
    "reference": "Matthew 1:1-9",
    "translation": "World English Bible Catholic (WEBC)",
}

SEGMENTS = [
    "The book of the genealogy of Jesus Christ, the son of David."
]

VISUALS = [
    {
        "search_query": "olive tree",
        "visual_direction": "wide",
    }
]

# Exactly what episode 1 produced.
AS_GENERATED = {
    "title": "Genealogy of Light",
    "summary": (
        "The lineage of Jesus Christ traces back through generations, "
        "revealing a divine ancestry rooted in faith and promise."
    ),
    "mood": "solemn",
    "visuals": VISUALS,
}

COMPLIANT = {
    "title": "The Genealogy of Jesus Christ",
    "summary": (
        "Matthew lists the ancestors of Jesus from Abraham down to David."
    ),
    "mood": "solemn",
    "visuals": VISUALS,
}


def _generator():

    generator = ContentGenerator(
        ConfigLoader().load_all()
    )

    generator.log = lambda *a, **k: None

    return generator


def _assemble(direction):

    return _generator().assemble_content(
        EPISODE_ONE,
        SEGMENTS,
        dict(direction),
    )


def _direction(title, summary):

    return {
        "title": title,
        "summary": summary,
        "mood": "solemn",
        "visuals": VISUALS,
    }
# --------------------------------------------------------------------------
# The prompt carries the rules
# --------------------------------------------------------------------------

def test_the_prompt_forbids_invented_concepts():
    """
    The model's only defence against "Genealogy of Light" is being told
    not to do it, so the instruction has to actually be there.
    """

    prompt = _generator().build_prompt(
        EPISODE_ONE,
        SEGMENTS,
    )

    assert "TITLE RULES" in prompt

    assert (
        "NEVER invent a poetic, symbolic, or literary concept"
    ) in prompt

    # The real failure, named outright.
    assert "Genealogy of Light" in prompt

    assert (
        "NEVER add a theological claim the passage does not state"
    ) in prompt


def test_the_prompt_stops_the_summary_repeating_the_title():
    prompt = _generator().build_prompt(
        EPISODE_ONE,
        SEGMENTS,
    )

    assert "SUMMARY RULES" in prompt

    assert (
        "Do NOT repeat or restate the title"
    ) in prompt

    assert (
        "Do NOT include the Scripture reference"
    ) in prompt

    assert "Do NOT interpret" in prompt


def test_the_schema_bounds_both_fields():
    """
    A length cap in the schema is a grammar-level brake, not advice.
    """

    assert DIRECTION_SCHEMA["properties"]["title"][
        "maxLength"
    ] <= 90

    assert DIRECTION_SCHEMA["properties"]["summary"][
        "maxLength"
    ] <= 180


# --------------------------------------------------------------------------
# The reference appears exactly once, in the title
# --------------------------------------------------------------------------

def test_the_reference_is_added_to_the_title_exactly_once():
    content = _assemble(COMPLIANT)

    assert content["title"] == (
        "The Genealogy of Jesus Christ — Matthew 1:1-9"
    )

    assert content["title"].count("Matthew 1:1-9") == 1

    # No parentheses around the reference.
    assert "(" not in content["title"]
    assert ")" not in content["title"]

    # The reference is not also stored as its own field.
    assert "reference" not in content


def test_a_reference_the_model_supplied_is_removed():
    """
    The model is told not to include the reference, but one that does
    must not produce a doubled title.
    """

    content = _assemble(_direction(
        "The Genealogy of Jesus Christ (Matthew 1:1-9)",
        "Matthew 1:1-9 lists the ancestors of Jesus.",
    ))

    assert content["title"] == (
        "The Genealogy of Jesus Christ — Matthew 1:1-9"
    )

    assert (
        content["title"].count("Matthew 1:1-9")
        == 1
    )

    # The summary must not carry it either - it is already in the title.
    assert "Matthew 1:1-9" not in content["summary"]


def test_stripping_a_reference_keeps_real_punctuation():
    """
    Removing the reference must not eat the sentence's full stop or
    leave a fragment starting in lower case.
    """

    assert _strip_reference(
        "The genealogy of Jesus.",
        "Matthew 1:1-9",
    ) == "The genealogy of Jesus."

    assert _strip_reference(
        "Matthew 1:1-9 lists the ancestors of Jesus.",
        "Matthew 1:1-9",
    ) == "Lists the ancestors of Jesus."
# --------------------------------------------------------------------------
# The summary must add something the title does not say
# --------------------------------------------------------------------------

def test_a_summary_that_echoes_the_title_is_replaced():
    """
    A listing where the title and the description say the same thing
    wastes the space, so the echo is caught.
    """

    content = _assemble(_direction(
        "The Genealogy of Jesus Christ",
        "The Genealogy of Jesus Christ",
    ))

    # Replaced with the passage's own opening words, which are factual.
    assert content["summary"] == _first_sentence(
        EPISODE_ONE["text"]
    )


def test_a_summary_that_adds_detail_is_kept():
    """
    Sharing vocabulary with the title is not the same as repeating it.
    """

    summary = (
        "The genealogy of Jesus Christ lists the ancestors from Abraham "
        "down to David."
    )

    assert not _restates_title(
        summary,
        "The Genealogy of Jesus Christ — Matthew 1:1-9",
        "Matthew 1:1-9",
    )

    content = _assemble(_direction(
        "The Genealogy of Jesus Christ",
        summary,
    ))

    assert content["summary"] == summary


# --------------------------------------------------------------------------
# Fallbacks come from the passage, never from invention
# --------------------------------------------------------------------------

def test_a_missing_title_and_summary_fall_back_to_the_passage():
    content = _assemble(_direction("", ""))

    opening = _first_sentence(EPISODE_ONE["text"])

    assert content["summary"] == opening

    assert content["title"] == (
        opening.rstrip(" .")
        + " — Matthew 1:1-9"
    )

    assert (
        content["title"].count("Matthew 1:1-9")
        == 1
    )


def test_the_title_never_ends_with_a_full_stop():
    """
    The fallback produced "... the son of Abraham. — Matthew 1:1-9".
    A title carries no full stop.
    """

    content = _assemble(_direction(
        "The Genealogy of Jesus Christ.",
        COMPLIANT["summary"],
    ))

    assert content["title"] == (
        "The Genealogy of Jesus Christ — Matthew 1:1-9"
    )


# --------------------------------------------------------------------------
# Episode 1, done properly
# --------------------------------------------------------------------------

def test_the_compliant_episode_one_metadata_is_clean():
    content = _assemble(COMPLIANT)

    title = content["title"]
    summary = content["summary"]

    # Factual title, plainly about what the passage contains.
    assert title == (
        "The Genealogy of Jesus Christ — Matthew 1:1-9"
    )

    # No invented concept.
    assert "Light" not in title

    # The summary says what happens, and does not restate the title.
    assert (
        "lists the ancestors of Jesus from Abraham down to David"
        in summary
    )

    # No interpretation, and no reference - that is in the title.
    assert "faith" not in summary.lower()
    assert "promise" not in summary.lower()
    assert "Matthew 1:1-9" not in summary

    # Short and useful.
    assert len(summary.split()) <= 30