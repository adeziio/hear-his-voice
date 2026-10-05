"""
Checks that the episode files the web UI reads are in the exact shape
Curious About Things uses.

The UI parses prompt.txt by splitting on the divider and looking for
TITLE / PROMPT / SUMMARY. If those markers are not in the same section
the listing silently comes back empty, so this is verified directly
against the real parser from web.server.

Run with:  python -m pytest tests/test_episode_files.py -v
"""

import shutil
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
    write_content_files,
    read_content_file
)


CONTENT = {
    "title": "The Light of Love — John 3:16-17",
    "summary": "A reflection on divine love.",
    "narration": (
        "For God so loved the world, that he gave his only born Son."
    ),
    "mood": ["reverent", "hopeful"],
    "visuals": [
        {
            "search_query": "still water at dawn",
            "sentence": "For God so loved the world.",
        }
    ],
}


@pytest.fixture
def episode_directory(tmp_path):
    return tmp_path / "001"


def test_content_json_has_only_the_expected_keys(episode_directory):
    """
    content.json must carry exactly the keys the pipeline and UI rely
    on - no copy of the Scripture text itself.

    The passage reference is not stored. It lives in the
    title, and neither the YouTube description nor the Instagram caption
    repeats it, so a separate field would have no reader. The other
    channels publish without one.
    """
    write_content_files(
        episode_directory,
        CONTENT
    )

    reloaded = read_content_file(
        episode_directory
    )

    assert reloaded is not None

    assert sorted(
        reloaded.keys()
    ) == [
        "mood",
        "narration",
        "summary",
        "title",
        "visuals",
    ]

    # No reference field, under any name.
    assert "reference" not in reloaded


def test_visual_entries_match_the_expected_shape(
    episode_directory
):
    write_content_files(
        episode_directory,
        CONTENT
    )

    reloaded = read_content_file(
        episode_directory
    )

    for visual in reloaded["visuals"]:

        assert sorted(
            visual.keys()
        ) == ["search_query", "sentence"]


def test_title_carries_the_scripture_reference(
    episode_directory
):
    write_content_files(
        episode_directory,
        CONTENT
    )

    reloaded = read_content_file(
        episode_directory
    )

    assert "John 3:16-17" in reloaded["title"]


def test_web_ui_can_parse_prompt_txt(episode_directory):
    """
    Runs the real UI parser over the real prompt.txt. This is the
    regression that left the prompt screen blank.
    """
    from web.server import (
        parse_prompt_file
    )

    write_content_files(
        episode_directory,
        CONTENT
    )

    prompts = parse_prompt_file(
        episode_directory / "prompt.txt"
    )

    assert prompts, (
        "The UI parser returned nothing, so the prompt "
        "screen will be blank."
    )

    item = prompts[0]

    assert item["title"] == CONTENT["title"]
    assert item["prompt"] == CONTENT["narration"]
    assert item["summary"] == CONTENT["summary"]


def test_prompt_txt_uses_the_expected_markers(
    episode_directory
):
    write_content_files(
        episode_directory,
        CONTENT
    )

    body = (
        episode_directory / "prompt.txt"
    ).read_text(encoding="utf-8")

    assert "TITLE: " in body
    assert "PROMPT: " in body
    assert "SUMMARY: " in body

    # No REFERENCE line. The reference is already in TITLE, and a third
    # copy would only be able to drift out of step with it.
    assert "REFERENCE: " not in body

    # Nothing that would break the section-based parser.
    assert "TRANSLATION:" not in body