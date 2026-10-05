"""
Runs the Scripture and Prompt stages against the real WEBC source and
the real local LLM, and writes a real episode directory.

Usage:
    python tests/live_content_test.py [Reference]
    python tests/live_content_test.py "John 3:16-17"

This stops before the video stage, because Pexels needs an interactive
browser session. It verifies that the passage really came from WEBC,
that the AI only produced visual direction, and that the saved content
carries the verbatim text.
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

    # Prove the narration is the exact WEBC text by re-reading the
    # passage straight from the source and comparing.
    bible = WEBCScripture()

    # The reference now lives in the title, so pull it back out and
    # re-read the passage from WEBC to prove the narration is verbatim.
    title = content["title"]

    reference = (
        title[
            title.rfind("(") + 1:
            title.rfind(")")
        ]
        if "(" in title and title.endswith(")")
        else reference
    )

    bible = WEBCScripture()

    expected = (
        bible.get_reference_text(reference)
    )

    assert content["narration"] == expected["text"], (
        "The narration does not match the WEBC source."
    )

    print()
    print("=" * 68)
    print("EPISODE", episode_directory.name)
    print("=" * 68)
    print("Title      :", content["title"])
    print("Summary    :", content["summary"])
    print("Mood       :", content["mood"])
    print("Visuals    :", len(content["visuals"]))
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

    print("SCRIPTURE AS SPOKEN (verbatim WEBC)")
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
    print("LIVE CONTENT TEST PASSED")


if __name__ == "__main__":

    main()