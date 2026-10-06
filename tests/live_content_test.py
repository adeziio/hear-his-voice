"""
Runs the Scripture and Prompt stages against the real WEBC source and
the real local LLM, and writes a real episode directory.

Usage:
    python tests/live_content_test.py [Reference]
    python tests/live_content_test.py "John 3:16-17"

This stops before the video stage, because Pexels needs an interactive
browser session. It verifies that the passage really came from WEBC,
that the narration is an original telling held faithful to that
passage, and that the saved content carries both of them.
"""

import json
import sys

from pathlib import Path


sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parent.parent
    )
)


from core.pipeline import HearHisVoicePipeline
from scripture.webc import WEBCScripture
from ai.content_generator import (
    narration_problems,
    verbatim_sentence_overlap,
)


def main():

    reference = (
        sys.argv[1]
        if len(sys.argv) > 1
        else None
    )

    pipeline = HearHisVoicePipeline()

    def progress(percent, message, stage):
        print(
            f"[{stage.upper()} {percent}%] {message}",
            flush=True
        )

    pipeline.set_progress_callback(
        progress
    )

    episode_directory = (
        pipeline._next_episode_directory()
    )

    episode_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    content = (
        pipeline._generate_content(
            episode_directory,
            reference
        )
    )

    # The episode carries both the telling and the passage it was
    # written from. Re-read the passage straight from WEBC to prove
    # the stored source really is the source.
    bible = WEBCScripture()

    title = content["title"]

    # The reference lives in the title as "Title — Book 1:2-3", and it
    # is the reference of the passage actually spoken - a window given
    # on the command line may have been sized down to it.
    if " — " in title:

        reference = title.rsplit(
            " — ",
            1
        )[-1]

    expected = (
        bible.get_reference_text(reference)
    )

    assert content["source_text"] == expected["text"], (
        "The stored source does not match the WEBC text."
    )

    # The telling is faithful to that source: every name kept, no
    # commentary, no reference spoken, nothing invented.
    problems = narration_problems(
        content["source_text"],
        content["narration"],
    )

    assert problems == [], problems

    # And it is a telling rather than the passage in disguise.
    assert content["narration"] != content["source_text"], (
        "The narration is the passage verbatim - no telling was "
        "produced."
    )

    overlap = verbatim_sentence_overlap(
        content["source_text"],
        content["narration"],
    )

    assert overlap < 0.5, (
        f"The narration repeats {overlap:.0%} of the passage's "
        "sentences word for word."
    )

    print()
    print("=" * 68)
    print("EPISODE", episode_directory.name)
    print("=" * 68)
    print("Title      :", content["title"])
    print("Summary    :", content["summary"])
    print("Mood       :", content["mood"])
    print("Visuals    :", len(content["visuals"]))
    print("Reference  :", reference)
    print("Overlap    :", f"{overlap:.0%}")
    print()
    print("Keys       :", sorted(content.keys()))
    print()

    saved = episode_directory / "content.json"

    with open(
        saved,
        "r",
        encoding="utf-8"
    ) as file:

        reloaded = json.load(file)

    assert reloaded["title"] == content["title"]
    assert reloaded["narration"] == content["narration"]
    assert reloaded["source_text"] == content["source_text"]

    print("THE PASSAGE (WEBC - THE SOURCE OF TRUTH)")
    print("-" * 68)
    print(content["source_text"])
    print()

    print("THE NARRATION (WHAT WILL BE SPOKEN)")
    print("-" * 68)
    print(content["narration"])
    print()

    print("VISUAL DIRECTION FROM THE AI")
    print("-" * 68)
    for index, visual in enumerate(
        content["visuals"],
        start=1
    ):
        print(f"{index:2d}. [{visual['search_query']}]")
        print(f"    {visual['sentence']}")

    print()
    print("Saved:", saved)
    print("content.json re-read and verified.")
    print()

    # The publishing metadata still carries the reference in the title
    # and the WEBC attribution in the description and the caption -
    # and never the passage itself.
    from core.publishing import (
        SCRIPTURE_ATTRIBUTION,
        build_caption,
    )
    from youtube.metadata_generator import (
        generate_metadata_from_prompt,
    )

    metadata = generate_metadata_from_prompt(
        {
            "title": content["title"],
            "summary": content["summary"],
        },
        episode_directory.name,
    )

    caption = build_caption(
        content["title"],
        content["summary"],
        metadata["tags"],
    )

    assert reference in metadata["title"], metadata["title"]

    for published in (metadata["description"], caption):

        assert SCRIPTURE_ATTRIBUTION in published

        assert published.count(reference) == 1, (
            "the reference must appear exactly once, in the title "
            f"line, not {published.count(reference)} times"
        )

        assert (
            content["source_text"][:60]
            not in published
        ), "the passage itself was published"

    print("PUBLISHING METADATA (YOUTUBE DESCRIPTION)")
    print("-" * 68)
    print(metadata["description"])
    print()

    print("LIVE CONTENT TEST PASSED")


if __name__ == "__main__":

    main()