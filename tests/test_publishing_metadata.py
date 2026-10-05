"""
Verifies the publishing metadata for both YouTube and Instagram.

Both platforms must carry the same blocks, in the same order:

    1. the episode title
    2. a short, factual summary of the passage
    3. the WEBC attribution line
    4. the hashtags

The title leads, matching how Curious About Things and Your Next Location
publish. The passage reference lives in that title and is never restated as
a line of its own.

Run with:  python -m pytest tests/test_publishing_metadata.py -v
"""

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


from core.publishing import (
    SCRIPTURE_ATTRIBUTION,
    build_caption,
    scripture_credit_lines,
)
from youtube.metadata_generator import (
    generate_metadata_from_prompt,
)


EPISODE = {
    "title": "The Genealogy of Jesus — Matthew 1:1-9",
    "summary": "Matthew lists the ancestors of Jesus from Abraham down to David.",
    "narration": (
        "The book of the genealogy of Jesus Christ, "
        "the son of David, the son of Abraham."
    ),
}


HASHTAG_LINE = "#jesus #bible #gospel #scripture #christian"


def _metadata():
    return generate_metadata_from_prompt(
        dict(EPISODE),
        "001",
    )


def _description():
    return _metadata()["description"]


def _caption():
    return build_caption(
        EPISODE["title"],
        EPISODE["summary"],
        HASHTAG_LINE.split(),
    )


def _blocks(text):
    return [
        part
        for part in text.split("\n\n")
        if part.strip()
    ]


def test_the_credit_block_is_the_attribution_only():
    """
    The passage reference is already in the episode title, so the credit
    block must not repeat it. Attribution only, and one line of it.
    """

    lines = scripture_credit_lines()

    assert lines == [
        "Scripture: World English Bible Catholic (WEBC), "
        "public domain. Text provided by eBible.org."
    ]

    assert lines[0] == SCRIPTURE_ATTRIBUTION

    assert len(lines) == 1

    assert (
        "Matthew 1:1-9"
        not in "\n".join(lines)
    )


def test_no_attribution_means_no_block():
    assert scripture_credit_lines("") == []
    assert scripture_credit_lines(None) == []


# --------------------------------------------------------------------------
# The title, and the em dash that carries the reference
# --------------------------------------------------------------------------

def test_the_title_uses_an_em_dash_and_no_parentheses():
    """
    The published title format is "Title - Book Chapter:Verse", with a real
    em dash and never parentheses.
    """

    title = _metadata()["title"]

    assert title == (
        "The Genealogy of Jesus — Matthew 1:1-9"
    )

    assert " — " in title

    assert "(" not in title
    assert ")" not in title


# --------------------------------------------------------------------------
# The same blocks, in the same order, on both platforms
# --------------------------------------------------------------------------

def test_the_youtube_description_is_exactly_the_expected_blocks():
    assert _blocks(_description()) == [
        EPISODE["title"],
        EPISODE["summary"],
        SCRIPTURE_ATTRIBUTION,
        HASHTAG_LINE,
    ]


def test_the_instagram_caption_is_exactly_the_expected_blocks():
    assert _blocks(_caption()) == [
        EPISODE["title"],
        EPISODE["summary"],
        SCRIPTURE_ATTRIBUTION,
        HASHTAG_LINE,
    ]


def test_both_platforms_carry_the_same_blocks():
    """
    The description and the caption are the same blocks in the same order.
    Any drift here means one platform quietly stops looking like the other.
    """

    assert _blocks(_description()) == _blocks(_caption())


def test_the_title_leads_on_both_platforms():
    """
    Every channel in this workspace leads with the title, so the two
    platforms stay structurally alike.
    """

    assert _blocks(_description())[0] == EPISODE["title"]
    assert _blocks(_caption())[0] == EPISODE["title"]

# --------------------------------------------------------------------------
# The reference is not restated as a line of its own
# --------------------------------------------------------------------------

def test_the_reference_is_not_restated_as_its_own_line():
    """
    The reference travels in the title, which leads both platforms. What must
    not happen is the old behaviour: a separate "Scripture: Matthew 1:1-9"
    line repeating it away from the title.
    """

    for text in (_description(), _caption()):

        blocks = _blocks(text)

        # It appears in the leading title line, and nowhere else.
        assert (
            blocks[0].count("Matthew 1:1-9") == 1
        )

        for block in blocks[1:]:

            assert "Matthew 1:1-9" not in block

        # No block restates it on its own.
        for block in blocks:

            if "Matthew 1:1-9" in block:

                assert block == EPISODE["title"]


def test_the_passage_text_itself_is_not_published():
    """
    The passage is the video. Repeating it in the description adds nothing
    and dilutes the summary.
    """

    for text in (_description(), _caption()):

        assert "the son of Abraham" not in text


def test_the_attribution_does_not_name_a_passage():
    """
    The attribution is about the translation, not the passage, so it must not
    smuggle the reference back in.
    """

    assert "Matthew" not in SCRIPTURE_ATTRIBUTION


# --------------------------------------------------------------------------
# The hashtags stay put
# --------------------------------------------------------------------------

def test_the_hashtags_are_last_and_unchanged():
    for text in (_description(), _caption()):

        assert _blocks(text)[-1] == HASHTAG_LINE

        assert text.strip().endswith("#christian")


def test_the_configured_hashtags_are_not_altered():
    assert _metadata()["tags"] == [
        "#jesus",
        "#bible",
        "#gospel",
        "#scripture",
        "#christian",
    ]


def test_a_missing_summary_still_yields_the_attribution_and_hashtags():
    """
    The summary is the only optional block. Losing it must not cost the
    attribution - the WEBC name is a trademark and is always required.
    """

    metadata = generate_metadata_from_prompt(
        {"title": "Some Title — Matthew 1:1"},
        "001",
    )

    assert _blocks(metadata["description"]) == [
        "Some Title — Matthew 1:1",
        SCRIPTURE_ATTRIBUTION,
        HASHTAG_LINE,
    ]


def test_a_caption_with_no_summary_still_carries_the_credit():
    caption = build_caption(
        "Some Title — Matthew 1:1",
        "",
        HASHTAG_LINE.split(),
    )

    assert _blocks(caption) == [
        "Some Title — Matthew 1:1",
        SCRIPTURE_ATTRIBUTION,
        HASHTAG_LINE,
    ]
