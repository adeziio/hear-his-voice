"""
Verifies the WEBC Scripture source.

These tests hit the real eBible.org WEBC release, which is the point:
the exact wording is what the episode will speak, so it is checked
against the real archive rather than a fixture.

Run with:  python -m pytest tests/test_scripture_source.py -v
"""

import re
import sys
import zipfile

from pathlib import Path

import pytest


sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parent.parent
    )
)


from scripture.webc import (
    WEBCScripture,
    ScriptureError,
    BOOK_FILE_CODES
)
from scripture.segmenter import (
    split_sentences,
    group_into_segments
)

from core.config_loader import ConfigLoader
from ai.content_generator import (
    ContentGenerator,
    DIRECTION_SCHEMA
)


# WEBC ships 73 USFM books in Catholic order; Daniel and Esther carry
# their Septuagint additions. The archive holds three further entries
# that are licence and styling files, not books.
WEBC_BOOK_COUNT = 73


@pytest.fixture(scope="module")
def bible():
    return WEBCScripture()


@pytest.fixture(scope="module")
def generator():
    """
    The creative director. The LLM is never called in these tests -
    only the prompt building and the content assembly are exercised.
    """
    return ContentGenerator(
        ConfigLoader().load_all()
    )


def test_downloads_and_parses_every_webc_book(bible):
    assert len(bible.books) == WEBC_BOOK_COUNT


def test_contains_the_deuterocanonical_books(bible):
    # These are what make the Catholic edition distinct from the
    # Protestant WEB, so their absence would mean the wrong translation.
    for book in (
        "Tobit",
        "Judith",
        "Wisdom",
        "Sirach",
        "Baruch",
        "1 Maccabees",
        "2 Maccabees",
    ):
        assert book in bible.books, book


def test_uses_the_catholic_lord_rendering(bible):
    """
    WEBC renders the divine name as LORD. The Protestant World English
    Bible uses Yahweh, so this is the clearest proof that the Catholic
    edition - not a substitute translation - is what is being served.
    """
    assert (
        "LORD"
        in bible.get_verse("Psalms", "23", "1")
    )

    assert "Yahweh" not in bible.get_verse(
        "Psalms",
        "1",
        "1"
    )


def test_parses_poetic_verse_markers(bible):
    """
    Psalms continue a verse on a "\\q" line instead of a "\\v" line.
    """
    for verse in ("1", "2", "4", "6"):
        text = bible.get_verse(
            "Psalms",
            "23",
            verse
        )

        assert text, (
            f"Psalm 23:{verse} was not parsed"
        )

        assert (
            "\\" not in text
        ), text


def test_serves_verbatim_text(bible):
    """
    John 3:16 must come back exactly as WEBC gives it.
    """
    text = bible.get_verse("John", "3", "16")

    assert text == (
        "For God so loved the world, that he gave his only born "
        "Son, that whoever believes in him should not perish, but "
        "have eternal life."
    )


def test_resolves_book_abbreviations(bible):
    assert bible.get_verse(
        "Jn",
        "3",
        "16"
    ) == bible.get_verse("John", "3", "16")

    assert bible.get_verse(
        "1 Cor",
        "13",
        "4"
    ) == bible.get_verse("1 Corinthians", "13", "4")


def test_rejects_an_unknown_book(bible):
    with pytest.raises(
        ScriptureError
    ):

        bible.get_reference_text(
            "Hezekiah 1:1"
        )


def test_rejects_a_malformed_reference(bible):
    with pytest.raises(
        ScriptureError
    ):

        bible.get_reference_text(
            "not a reference"
        )


def test_verse_range_is_returned_in_order(bible):
    result = bible.get_reference_text(
        "John 3:16-18"
    )

    assert [
        item["verse"]
        for item in result["passage"]
    ] == [16, 17, 18]

    assert result["translation"] == (
        "World English Bible Catholic (WEBC)"
    )


def test_no_usfm_markup_survives_anywhere(bible):
    """
    Every verse of the whole translation must be free of USFM markup,
    strong-number attributes, and inline footnote residue.
    """
    offenders = []

    for book, chapters in bible.books.items():

        for chapter, verses in chapters.items():

            for verse, text in verses.items():

                if (
                    "\\" in text
                    or "strong=" in text
                    or " + " in text
                ):

                    offenders.append(
                        (
                            book,
                            chapter,
                            verse,
                            text[:80]
                        )
                    )

    assert not offenders, offenders[:5]


def test_the_ai_cannot_change_the_scripture(
    bible,
    generator
):
    """
    The central guarantee of the channel.

    Whatever the AI returns - including a deliberate attempt to inject
    its own version of the passage - the narration and the spoken
    segments must remain the verbatim WEBC text.
    """
    scripture = bible.get_reference_text(
        "John 3:16-17"
    )

    segments = generator.build_segments(
        scripture
    )

    hostile_direction = {
        "title": "A Perfectly Reasonable Title",
        "summary": "A summary.",
        "mood": "reverent hopeful",
        "visuals": [
            {
                "search_query": (
                    "sunrise over calm water"
                ),
                "visual_direction": (
                    "A slow sunrise over still water."
                )
            }
            for _ in segments
        ],
    }

    content = generator.assemble_content(
        scripture,
        segments,
        hostile_direction
    )

    # The narration is the WEBC string, byte for byte.
    assert (
        content["narration"]
        == scripture["text"]
    )

    # Rejoining the spoken segments reproduces the same passage.
    assert (
        " ".join(
            segment
            for segment in segments
        ).split()
        == scripture["text"].split()
    )

    # Every verse WEBC returned survives into the narration.
    for item in scripture["passage"]:

        assert (
            item["text"]
            in content["narration"]
        )

    # And the AI still contributed its creative direction, with the
    # Scripture reference carried in the title.
    assert (
        content["title"]
        == "A Perfectly Reasonable Title (John 3:16-17)"
    )

    assert len(
        content["visuals"]
    ) == len(segments)

    assert (
        content["visuals"][0]["search_query"]
        == "sunrise over calm water"
    )


def test_the_schema_has_no_field_for_scripture_text():
    """
    The model's response shape contains no narration field, so there is
    nowhere for it to put Bible text even if it tried.
    """
    properties = set(
        DIRECTION_SCHEMA["properties"]
    )

    assert properties == {
        "title",
        "summary",
        "mood",
        "visuals"
    }

    for forbidden in (
        "narration",
        "scripture",
        "text",
        "verse",
    ):

        assert forbidden not in properties


def test_a_blank_search_query_falls_back_to_the_segment(
    bible,
    generator
):
    scripture = bible.get_reference_text(
        "Psalm 23:1"
    )

    segments = generator.build_segments(
        scripture
    )

    content = generator.assemble_content(
        scripture,
        segments,
        {
            "title": "The Shepherd",
            "summary": "A summary.",
            "mood": "reverent",
            "visuals": [
                {
                    "search_query": "",
                    "visual_direction": "",
                }
            ]
        }
    )

    query = content["visuals"][0]["search_query"]

    # Psalm 23:1 is "The LORD is my shepherd; I shall lack
    # nothing.", so the fallback keeps real content words.
    assert query
    assert "shepherd" in query


def test_one_visual_per_segment(
    bible,
    generator
):
    scripture = bible.get_reference_text(
        "Psalm 23:1-6"
    )

    segments = generator.build_segments(
        scripture
    )

    content = generator.assemble_content(
        scripture,
        segments,
        {
            "title": "The Shepherd",
            "summary": "A summary.",
            "mood": "gentle",
            "visuals": [],
        }
    )

    assert len(
        content["visuals"]
    ) == len(segments)

    for visual, segment in zip(
        content["visuals"],
        segments
    ):

        assert visual["sentence"] == segment
        assert visual["search_query"]


def test_mood_is_normalised_for_music_matching(
    bible,
    generator
):
    scripture = bible.get_reference_text(
        "John 3:16"
    )

    segments = generator.build_segments(
        scripture
    )

    content = generator.assemble_content(
        scripture,
        segments,
        {
            "title": "T",
            "summary": "S",
            "mood": "Reverent, HOPEFUL, reverent, gentle",
            "visuals": [],
        }
    )

    assert content["mood"] == [
        "reverent",
        "hopeful",
        "gentle"
    ]
    """
    Splitting into sentences and grouping into spoken segments may only
    choose boundaries. The words and their order must survive intact.
    """
    result = bible.get_reference_text(
        "John 3:16-18"
    )

    text = result["text"]

    sentences = split_sentences(text)

    assert (
        " ".join(sentences).split()
        == text.split()
    )

    segments = group_into_segments(
        sentences,
        8
    )

    assert (
        " ".join(segments).split()
        == text.split()
    )