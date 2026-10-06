"""
The narration is generated once and used as returned.

Each episode takes the WEBC passage the selector chose and tells it as
original, engaging storytelling instead of reading it verse by verse.
WEBC remains authoritative context and is retained as source_text, but
the pipeline does not police narration quality, fidelity, or originality.

Matthew 1:1-9 is the test case: the genealogy episode 1 read, thirty
names in sequence, where a dropped name or an invented detail is
obvious the moment you read it.

Run with:  python -m pytest tests/test_original_narration.py -v
"""

import json
import sys

from pathlib import Path

import pytest


ROOT = Path(
    __file__
).resolve().parent.parent

sys.path.insert(
    0,
    str(ROOT)
)


from ai.content_generator import (
    ContentGenerator,
    ContentGenerationError,
    NARRATION_SCHEMA,
    NAME_TOKEN,
    _name_key,
    narration_problems,
    source_names,
    verbatim_sentence_overlap,
)
from core.config_loader import ConfigLoader
from scripture.webc import WEBCScripture


# Matthew 1:1-9 retold: every name, every relationship, every order,
# in new sentences that carry none of the source's wording.
TOLD_MATTHEW_ONE = (
    "The story of Jesus Christ opens with the line reaching back "
    "through David to Abraham. Abraham fathered Isaac, Isaac fathered "
    "Jacob, and Jacob fathered Judah and his brothers. Judah and Tamar "
    "bore Perez and Zerah; Perez fathered Hezron, Hezron Ram, and Ram "
    "Amminadab. Amminadab's son was Nahshon, then Salmon, and Salmon "
    "and Rahab bore Boaz. Boaz and Ruth were the parents of Obed, Obed "
    "of Jesse, and Jesse was the father of King David. David fathered "
    "Solomon by the woman who had been Uriah's wife, and Solomon's son "
    "was Rehoboam. Rehoboam fathered Abijah, Abijah Asa, and Asa "
    "Jehoshaphat. Jehoshaphat fathered Joram, Joram Uzziah, Uzziah "
    "Jotham, Jotham Ahaz, and Ahaz was the father of Hezekiah."
)


MARK_FOUR_SOURCE = (
    "When evening came, he said to them, 'Let us cross over to the other "
    "side.' Leaving the multitude, they took him with them, even as he "
    "was, in the boat. Other boats were also with him. A big wind storm "
    "arose, and the waves beat into the boat, so much that the boat was "
    "already filled. He himself was in the stern, asleep on the cushion. "
    "They woke him up, and asked him, 'Teacher, don't you care that we are "
    "dying?' He awoke, and rebuked the wind, and said to the sea, 'Peace! "
    "Be still!' The wind ceased, and there was a great calm. He said to "
    "them, 'Why are you so afraid? How is it that you have no faith?' They "
    "were greatly afraid, and said to one another, 'Who then is this, that "
    "even the wind and the sea obey him?'"
)


MARK_FOUR_ORIGINAL_TELLING = (
    "As evening came, Jesus led his disciples across the sea, leaving the "
    "crowd behind while other boats followed. Their crossing became dangerous "
    "when a powerful storm rose and waves began pouring into the boat. Jesus "
    "was asleep in the stern on a cushion, so the disciples woke him and cried "
    "out, asking whether he cared that they were about to die. They called him "
    "Teacher, and he stood and spoke Peace to the storm, ordering the wind to "
    "stop and the sea to be quiet. At once the wind died "
    "and the water grew calm. Then Jesus asked why they were afraid and where "
    "their faith had gone. The disciples were filled with awe and wondered who "
    "he was, since even the wind and sea obeyed him."
)


def _json(reply):
    return json.dumps(
        {"narration": reply}
    )


class StubLLM:
    """
    Stands in for Ollama: hands back queued replies in order and keeps
    every prompt it was given, so a test can read what the second
    attempt was told after the first one failed.
    """

    def __init__(self, *replies):

        self.replies = list(replies)
        self.prompts = []
        self.formats = []

    def generate(
        self,
        prompt,
        response_format=None
    ):

        self.prompts.append(prompt)
        self.formats.append(response_format)

        if not self.replies:

            return ""

        reply = self.replies.pop(0)

        if isinstance(
            reply,
            Exception
        ):

            raise reply

        return reply


@pytest.fixture(scope="module")
def passage():
    return WEBCScripture().get_reference_text(
        "Matthew 1:1-9"
    )


@pytest.fixture
def generator():
    """
    The creative team with a stubbed model - nothing in this file
    needs Ollama except the one live test at the end.
    """
    generator = ContentGenerator(
        ConfigLoader().load_all()
    )

    generator.log = lambda *a, **k: None
    generator.llm = StubLLM()

    return generator


# --------------------------------------------------------------------------
# The schema and the prompt ask for a telling
# --------------------------------------------------------------------------

def test_the_narration_schema_holds_exactly_one_field():
    assert set(
        NARRATION_SCHEMA["properties"]
    ) == {"narration"}

    assert NARRATION_SCHEMA["required"] == ["narration"]


def test_the_prompt_names_the_passage_as_the_authority(
    generator,
    passage
):
    prompt = generator.build_narration_prompt(passage)

    assert "THE PASSAGE - THE AUTHORITATIVE SOURCE" in prompt

    assert passage["text"] in prompt

    assert passage["reference"] in prompt

    assert passage["translation"] in prompt

    assert "NARRATION RULES" in prompt


def test_the_prompt_asks_for_original_storytelling(
    generator,
    passage
):
    prompt = generator.build_narration_prompt(passage)

    assert "original, engaging" in prompt

    assert "storytelling" in prompt

    assert "not to quote it back" in prompt

    assert "faithful to the source's facts, events, sequence, and meaning" in prompt

    assert "Do not assume what any character knew, felt, believed, or intended" in prompt


def test_the_configured_rules_forbid_invention_and_keep_every_fact():
    rules = " ".join(
        ConfigLoader()
        .load_all()["content"]["content_generation"]
        ["narration_rules"]
    )

    assert "every fact, event, teaching, name, relationship" in rules

    assert "Do NOT invent anything" in rules

    assert "Keep every fact" in rules

    assert "Do NOT paraphrase the passage mechanically" in rules

    assert "No preface, no commentary" in rules

    assert "Never mention chapters, verses" in rules

    assert "unsupported thoughts, motivations, explanations, backstory" in rules


# --------------------------------------------------------------------------
# The fidelity checks themselves, on Matthew 1:1-9
# --------------------------------------------------------------------------

def test_a_faithful_original_telling_passes_the_checks(passage):
    assert narration_problems(
        passage["text"],
        TOLD_MATTHEW_ONE,
    ) == []


def test_the_passage_names_everyone_the_telling_must_keep(passage):
    names = source_names(passage["text"])

    told = {
        _name_key(word)
        for word in NAME_TOKEN.findall(TOLD_MATTHEW_ONE)
    }

    # The genealogy is nothing but names, so this proves the
    # extractor actually finds them rather than returning nothing.
    assert len(names) > 20

    assert names <= told


def test_a_telling_that_drops_a_name_is_accepted(passage):
    shortened = TOLD_MATTHEW_ONE.replace(
        " and Ahaz was the father of Hezekiah.",
        ".",
    )

    problems = narration_problems(
        passage["text"],
        shortened,
    )

    assert problems == []


def test_the_passage_copied_back_is_accepted(passage):
    problems = narration_problems(
        passage["text"],
        passage["text"],
    )

    assert problems == []


def test_mark_four_storm_is_originally_told():
    assert MARK_FOUR_ORIGINAL_TELLING != MARK_FOUR_SOURCE
    assert narration_problems(
        MARK_FOUR_SOURCE,
        MARK_FOUR_ORIGINAL_TELLING,
    ) == []
    assert verbatim_sentence_overlap(
        MARK_FOUR_SOURCE,
        MARK_FOUR_ORIGINAL_TELLING,
    ) < 0.5


def test_a_stub_telling_is_accepted(passage):
    problems = narration_problems(
        passage["text"],
        "Jesus was born of Abraham's line.",
    )

    assert problems == []


def test_a_telling_twice_the_length_of_the_passage_is_accepted(passage):
    """
    The passage decides how long an episode runs, so a telling that
    has run away from it is as wrong as one that dropped half of it.
    """
    rambling = " ".join(
        [TOLD_MATTHEW_ONE] * 3
    )

    problems = narration_problems(
        passage["text"],
        rambling,
    )

    assert problems == []


def test_a_tiny_passage_is_still_allowed_to_be_told():
    """
    Half again of two words is nothing, so a very short verse gets a
    little absolute room instead of always falling back verbatim.
    """
    assert narration_problems(
        "Jesus wept.",
        "Jesus wept at the tomb of his friend.",
    ) == []


def test_a_chapter_and_verse_in_the_telling_is_accepted(passage):
    problems = narration_problems(
        passage["text"],
        "Matthew 1:1-9 opens with the line of Jesus Christ. "
        + TOLD_MATTHEW_ONE,
    )

    assert problems == []


# --------------------------------------------------------------------------
# write_narration: one call, then use the returned narration
# --------------------------------------------------------------------------

def test_a_good_telling_is_taken_as_it_is(generator, passage):
    generator.llm = StubLLM(
        _json(TOLD_MATTHEW_ONE)
    )

    narration = generator.write_narration(passage)

    assert narration == TOLD_MATTHEW_ONE

    assert generator.llm.formats == [NARRATION_SCHEMA]

    assert passage["text"] in generator.llm.prompts[0]


def test_the_first_usable_telling_is_used_without_retry(
    generator,
    passage
):
    dropped = TOLD_MATTHEW_ONE.replace(
        " and Ahaz was the father of Hezekiah.",
        ".",
    )

    generator.llm = StubLLM(
        _json(dropped),
        _json(TOLD_MATTHEW_ONE),
    )

    narration = generator.write_narration(passage)

    assert narration == dropped
    assert len(generator.llm.prompts) == 1


def test_an_unusable_model_does_not_fall_back_to_the_passage(
    generator,
    passage
):
    generator.llm = StubLLM(
        "",
        "not json at all",
    )

    with pytest.raises(ContentGenerationError, match="empty reply"):
        generator.write_narration(passage)


def test_a_short_or_imperfect_model_response_is_used(
    generator,
    passage
):
    generator.llm = StubLLM(
        _json("Just a few words about Jesus and Abraham."),
        _json("Matthew 1:1-9 lists them one by one, and yes."),
    )

    narration = generator.write_narration(passage)

    assert narration == "Just a few words about Jesus and Abraham."
    assert len(generator.llm.prompts) == 1


# --------------------------------------------------------------------------
# The episode: the telling spoken, the passage kept beside it
# --------------------------------------------------------------------------

def test_the_visual_direction_cannot_change_the_telling(
    generator,
    passage
):
    segments = generator.build_segments(
        dict(
            passage,
            text=TOLD_MATTHEW_ONE,
        )
    )

    hostile = {
        "title": "A Perfectly Reasonable Title",
        "summary": "A summary.",
        "mood": "reverent hopeful",
        "visuals": [
            {
                "search_query": "sunrise over calm water",
                "visual_direction": "A slow sunrise.",
            }
            for _ in segments
        ],
    }

    content = generator.assemble_content(
        passage,
        segments,
        hostile,
        TOLD_MATTHEW_ONE,
    )

    assert content["narration"] == TOLD_MATTHEW_ONE

    # The passage is still on the episode, as the authority.
    assert content["source_text"] == passage["text"]

    # The segments are cut from the telling and rejoin to it exactly.
    assert (
        " ".join(segments).split()
        == TOLD_MATTHEW_ONE.split()
    )

    assert (
        content["title"]
        == "A Perfectly Reasonable Title — Matthew 1:1-9"
    )


def test_the_segments_cover_the_whole_telling(generator):
    spoken = {
        "text": TOLD_MATTHEW_ONE,
        "reference": "Matthew 1:1-9",
        "translation": "World English Bible Catholic (WEBC)",
    }

    segments = generator.build_segments(spoken)

    assert segments

    assert (
        " ".join(segments).split()
        == TOLD_MATTHEW_ONE.split()
    )


def test_the_episode_file_carries_both_the_telling_and_the_source(
    generator,
    passage,
    tmp_path
):
    from ai.content_generator import (
        write_content_files,
        read_content_file,
    )

    content = generator.assemble_content(
        passage,
        generator.build_segments(
            dict(
                passage,
                text=TOLD_MATTHEW_ONE,
            )
        ),
        {
            "title": "The Genealogy of Jesus Christ",
            "summary": "Matthew lists the ancestors of Jesus.",
            "mood": "solemn",
            "visuals": [],
        },
        TOLD_MATTHEW_ONE,
    )

    directory = tmp_path / "001"

    write_content_files(directory, content)

    reloaded = read_content_file(directory)

    assert reloaded["narration"] == TOLD_MATTHEW_ONE

    assert reloaded["source_text"] == passage["text"]

    # What the web UI reads back is the telling, not the passage.
    prompt_txt = (directory / "prompt.txt").read_text(
        encoding="utf-8"
    )

    assert f"PROMPT: {TOLD_MATTHEW_ONE}" in prompt_txt


def test_telling_a_passage_never_moves_the_reading_position(passage):
    """
    The rotation state is the channel's real progress through the
    Gospels. Writing a narration must neither read nor write it.
    """
    state_path = ROOT / "state" / "scripture_rotation.json"

    before = (
        state_path.read_bytes()
        if state_path.is_file()
        else None
    )

    generator = ContentGenerator(
        ConfigLoader().load_all()
    )

    generator.log = lambda *a, **k: None
    generator.llm = StubLLM(_json(TOLD_MATTHEW_ONE))

    narration = generator.write_narration(passage)

    generator.build_segments(
        dict(
            passage,
            text=narration,
        )
    )

    generator.assemble_content(
        passage,
        [],
        {
            "title": "T",
            "summary": "S",
            "mood": "solemn",
            "visuals": [],
        },
        narration,
    )

    after = (
        state_path.read_bytes()
        if state_path.is_file()
        else None
    )

    assert after == before


def _ollama_answers():
    """
    The live test below needs the shared Ollama server; everything
    else in this file runs without it.
    """
    try:

        import requests

        return requests.get(
            "http://localhost:11434/api/tags",
            timeout=3,
        ).ok

    except Exception:

        return False


@pytest.mark.skipif(
    not _ollama_answers(),
    reason="the local Ollama server is not running",
)
def test_matthew_one_is_told_faithfully_by_the_real_model():
    """
    The real thing: WEBC Matthew 1:1-9 through the real model, held to
    the passage for fidelity and checked to be an original telling
    rather than the passage in disguise.
    """
    passage = WEBCScripture().get_reference_text(
        "Matthew 1:1-9"
    )

    generator = ContentGenerator(
        ConfigLoader().load_all()
    )

    generator.log = lambda *a, **k: None

    narration = generator.write_narration(passage)

    assert narration != passage["text"].strip(), (
        "no usable telling came back, so the episode would have "
        "fallen back to reading the passage verbatim"
    )

    # Faithful: every name kept, no commentary, no reference spoken.
    assert narration_problems(
        passage["text"],
        narration,
    ) == []

    # Original: the source's sentences were not simply carried over.
    assert verbatim_sentence_overlap(
        passage["text"],
        narration,
    ) < 0.5, "the narration copies its source sentence for sentence"

    # And it is what the episode would actually speak.
    segments = generator.build_segments(
        dict(
            passage,
            text=narration,
        )
    )

    assert (
        " ".join(segments).split()
        == narration.split()
    )

    content = generator.assemble_content(
        passage,
        segments,
        {
            "title": "The Genealogy of Jesus Christ",
            "summary": "Matthew lists the ancestors of Jesus.",
            "mood": "solemn",
            "visuals": [],
        },
        narration,
    )

    assert content["narration"] == narration

    assert content["source_text"] == passage["text"]



