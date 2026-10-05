"""
Verifies the publishing metadata for both YouTube and Instagram.

The Scripture credit block - the passage that was read, plus the WEBC
attribution - must appear in both descriptions, positioned after the
summary and immediately before the hashtags. It is taken from the
episode's stored reference, never re-typed.

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
    scripture_credit_lines,
)
from youtube.metadata_generator import (
    generate_metadata_from_prompt,
)


EPISODE = {
    "title": "The Genealogy of Jesus (Matthew 1:1-9)",
    "reference": "Matthew 1:1-9",
    "summary": "Jesus is introduced as the son of David.",
    "narration": (
        "The book of the genealogy of Jesus Christ, "
        "the son of David, the son of Abraham."
    ),
}


def _description():
    return generate_metadata_from_prompt(
        dict(EPISODE),
        "001",
    )["description"]


def test_the_credit_block_names_the_passage():
    lines = scripture_credit_lines(
        "Matthew 1:1-9"
    )

    assert lines[0] == (
        "\U0001F4D6 Scripture: Matthew 1:1-9"
    )


def test_the_credit_block_carries_the_webc_attribution():
    lines = scripture_credit_lines(
        "Matthew 1:1-9"
    )

    assert lines[1] == (
        "Scripture: World English Bible Catholic (WEBC), "
        "public domain. Text provided by eBible.org."
    )

    assert lines[1] == SCRIPTURE_ATTRIBUTION


def test_no_reference_means_no_block():
    """
    Older episodes may have no stored reference. The block is then
    omitted rather than rendered with an empty reference.
    """

    assert scripture_credit_lines("") == []
    assert scripture_credit_lines(None) == []


def test_youtube_description_carries_the_block():
    description = _description()

    assert (
        "\U0001F4D6 Scripture: Matthew 1:1-9"
    ) in description

    assert SCRIPTURE_ATTRIBUTION in description


def test_youtube_block_sits_between_the_summary_and_the_hashtags():
    """
    The credit belongs with the closing lines: after the summary, and
    immediately before the hashtag line.
    """

    parts = _description().split("\n\n")

    credit_index = next(
        index
        for index, part in enumerate(parts)
        if part.startswith("\U0001F4D6 Scripture:")
    )

    summary_index = parts.index(
        EPISODE["summary"]
    )

    hashtag_index = next(
        index
        for index, part in enumerate(parts)
        if part.strip().startswith("#")
    )

    assert summary_index < credit_index < hashtag_index

    # The attribution sits directly under the reference line.
    assert (
        parts[credit_index + 1]
        == SCRIPTURE_ATTRIBUTION
    )


def test_instagram_caption_sits_between_the_summary_and_the_hashtags():
    """
    The Instagram caption is built from title, summary and hashtags, so
    the same block is spliced in the same place.
    """

    tags = "#HearHisVoice #Scripture"

    caption = "\n\n".join(
        [EPISODE["title"], EPISODE["summary"]]
        + scripture_credit_lines(EPISODE["reference"])
        + [tags]
    )

    parts = caption.split("\n\n")

    assert parts[0] == EPISODE["title"]
    assert parts[1] == EPISODE["summary"]
    assert parts[2].startswith("\U0001F4D6 Scripture:")
    assert parts[3] == SCRIPTURE_ATTRIBUTION
    assert parts[4] == tags


def test_the_reference_comes_from_the_episode_not_from_the_title():
    """
    The block must not be scraped back out of the title - the title also
    carries the reference, but the stored field is the source of truth.
    """

    without_reference = dict(EPISODE)
    without_reference.pop("reference")

    description = generate_metadata_from_prompt(
        without_reference,
        "001",
    )["description"]

    assert (
        "\U0001F4D6 Scripture:"
        not in description
    )