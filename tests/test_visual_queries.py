"""
The Pexels search-query rules.

The narration, the captions, the segmentation and the rendering are all fixed
and out of scope here. This file covers only one thing: the rules that steer
the AI toward Pexels queries that support the Scripture being narrated,
instead of arbitrary stock that happens to match a noun in the sentence.

The regression is concrete. A genealogy passage once produced:

    wooden table with bread
    hands raising a cup
    still water at dawn

Those are real, filmable, Pexels-searchable queries - and they mean nothing
next to a genealogy. The rules must make that outcome unlikely by anchoring
every query in the passage's meaning.

Run with:  python -m pytest tests/test_visual_queries.py -v
"""

import json
import sys

from pathlib import Path


ROOT = Path(
    __file__
).resolve().parent.parent

sys.path.insert(
    0,
    str(ROOT)
)


from ai.content_generator import ContentGenerator
from core.config_loader import ConfigLoader


# The three queries that prompted this work, and three of the replacements
# the project prefers.
ARBITRARY_QUERIES = {
    "wooden table with bread",
    "hands raising a cup",
    "still water at dawn",
}

PREFERRED_QUERIES = (
    "ancient genealogy scroll",
    "ancient parchment writing",
    "Middle Eastern desert landscape",
    "ancient Jerusalem",
    "ancient royal city",
    "family lineage historical",
)

SCRIPTURE = {
    "text": (
        "The book of the genealogy of Jesus Christ, the son of David, "
        "the son of Abraham. Abraham became the father of Isaac."
    ),
    "reference": "Matthew 1:1-9",
    "translation": "World English Bible Catholic (WEBC)",
}

SEGMENTS = [
    "The book of the genealogy of Jesus Christ, the son of David.",
]


def _generator():
    generator = ContentGenerator(
        ConfigLoader().load_all()
    )

    generator.log = lambda *a, **k: None

    return generator


def _prompt():
    return _generator().build_prompt(
        SCRIPTURE,
        SEGMENTS,
    )


def _visual_rules():
    config = json.loads(
        (ROOT / "config" / "content.json").read_text(
            encoding="utf-8"
        )
    )

    return config["content_generation"]["visual_rules"]


# --------------------------------------------------------------------------
# The rules anchor every query in the passage's meaning
# --------------------------------------------------------------------------

def test_the_rules_anchor_the_query_in_the_passage_meaning():
    rules = " ".join(_visual_rules())

    assert "MEANING, THEME, SETTING, MOOD" in rules

    assert "not for a prop" in rules


def test_the_rules_prefer_cinematic_and_historically_appropriate_footage():
    rules = " ".join(_visual_rules())

    assert "cinematic" in rules.lower()
    assert "historically appropriate" in rules.lower()
    assert "atmospheric" in rules.lower()


def test_the_rules_require_queries_pelexels_can_actually_serve():
    rules = " ".join(_visual_rules())

    assert "genuinely exist on Pexels" in rules

    assert "not a poetic invention" in rules


def test_the_rules_forbid_literal_biblical_people_and_events():
    rules = " ".join(_visual_rules())

    assert "Do NOT try to literally depict" in rules

    assert "genealogical relationships" in rules

    assert "named events or miracles" in rules


def test_the_rules_offer_a_thematic_fallback():
    """
    Literal footage often does not exist, so the rules have to say what to do
    instead of leaving the model to invent a prop.
    """

    rules = " ".join(_visual_rules())

    assert "thematic or symbolic visual" in rules

    assert "When literal footage does not exist" in rules


def test_the_rules_explicitly_reject_the_arbitrary_queries():
    """
    The regression is named outright. A model told only to "avoid generic
    footage" will still reach for a plausible prop.
    """

# --------------------------------------------------------------------------
# Exactly one query per existing segment, unchanged
# --------------------------------------------------------------------------

def test_exactly_one_query_is_requested_per_segment():
    prompt = _prompt()

    assert "exactly one" in prompt.lower()

    assert (
        "one per segment, in the same "
        "order as the segments above"
    ) in prompt


def test_the_prompt_asks_for_exactly_the_number_of_segments():
    prompt = _prompt()

    assert (
        "Return exactly {} objects".format(len(SEGMENTS))
        in prompt
    )

    assert (
        "SPLIT INTO {} SPOKEN SEGMENTS".format(len(SEGMENTS))
        in prompt
    )


def test_the_search_query_field_is_told_to_support_the_meaning():
    prompt = _prompt()

    assert "SUPPORTS THE MEANING" in prompt

    assert "theme, setting, era or mood" in prompt


def test_the_rules_prefer_concrete_cinematic_search_phrases():
    rules = " ".join(_visual_rules())

    assert "concrete, searchable ideas" in rules

    assert "stormy lake night" in rules

    assert "theological claims" in rules


def test_the_rules_replace_awkward_literal_scenes_with_filmable_context():
    rules = " ".join(_visual_rules())

    assert "awkward literal translations" in rules

    assert "ancient cushion on wooden bench" in rules

    assert "filmable setting" in rules


# --------------------------------------------------------------------------
# Everything out of scope stays out of scope
# --------------------------------------------------------------------------

def test_the_visual_prompt_still_forbids_writing_scripture():
    """
    The telling has its own call and its own schema; this prompt is
    still only for what the viewer sees, and it still may not touch a
    word of Scripture or of the narration spoken over it.
    """
    prompt = _prompt()

    assert (
        "CRITICAL - YOU DO NOT WRITE THE SCRIPTURE"
        in prompt
    )

    assert "must NEVER generate" in prompt


def test_the_prompt_quotes_the_passage_it_directs_visuals_for():
    """
    The visual director now sees both the passage and the telling cut
    into segments, so every query can be chosen for what the segment
    actually means.
    """
    prompt = _prompt()

    assert SCRIPTURE["text"] in prompt

    assert SEGMENTS[0] in prompt


def test_the_title_and_summary_rules_are_untouched():
    """
    The visual change must not have disturbed the metadata work.
    """

    prompt = _prompt()

    assert "TITLE RULES" in prompt
    assert "SUMMARY RULES" in prompt

    assert (
        "NEVER invent a poetic, symbolic, or literary concept"
        in prompt
    )

    assert "Do NOT repeat or restate the title" in prompt


def test_the_visual_direction_and_safety_wording_is_unchanged():
    """
    Guarding the boundaries of this change: the prompt still describes a
    single reverent visual direction sentence per segment, and still forbids
    disturbing imagery and cheap iconography.
    """

    prompt = _prompt()

    assert (
        "\"visual_direction\": one sentence describing the shot"
        in prompt
    )

    assert "horror, gore, wreckage, or disaster" in prompt

    assert "no glowing halos" in prompt
    rules = " ".join(_visual_rules())

    for query in ARBITRARY_QUERIES:

        assert query in rules

    assert "generic or arbitrary stock" in rules


def test_the_rules_offer_the_preferred_queries_as_a_worked_example():
    rules = " ".join(_visual_rules())

    assert "Worked example" in rules

    for query in PREFERRED_QUERIES:

        assert query in rules