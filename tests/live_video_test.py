"""
Runs the video stage for real, using the real WEBC passage, the real
telling written from it, the real edge-tts narration, the real caption
builder, and the real composer.

The only thing stubbed is the Pexels footage provider: downloading
stock footage needs an interactive browser session, so synthetic colour
clips stand in for it. Everything that touches the episode - the
narration and its source, captions, segment timing, composition and
rendering - is the real code running on the real passage.

Usage:
    python tests/live_video_test.py [Reference]
    python tests/live_video_test.py "John 3:16-18"
"""

import shutil
import sys

from pathlib import Path


sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parent.parent
    )
)


from moviepy import ColorClip

from core.pipeline import HearHisVoicePipeline
from production.footage.base import VideoProvider
from production import video as production_video
from scripture.webc import WEBCScripture
from ai.content_generator import narration_problems


class StubFootageProvider(VideoProvider):
    """
    Stands in for Pexels. Produces one real, decodable clip per query so
    the composer, the segment timing and the renderer all run for real.
    """

    name = "stub"

    def __init__(self, config, notify=None):

        self.config = config
        self.notify = notify

    def fetch(
        self,
        query,
        destination_dir,
        max_videos=2,
        downloaded_ids=None,
        downloaded_hashes=None
    ):
        destination_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        clips = []

        for index in range(max_videos):
            clip_path = destination_dir / f"clip_{index + 1}.mp4"
            ColorClip(
                (1080, 1920),
                color=(30, 60 + index * 40, 90),
                duration=6
            ).with_fps(30).write_videofile(
                str(clip_path),
                logger=None
            )
            clips.append(str(clip_path))

        return clips


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

    # The passage is stored exactly as WEBC gives it...
    assert content["source_text"] == expected["text"]

    # ...and the narration is a telling of it that passes every
    # fidelity check, rather than the passage itself.
    assert narration_problems(
        content["source_text"],
        content["narration"],
    ) == []

    assert content["narration"] != content["source_text"]

    print()
    print(f"Episode   : {episode_directory.name}")
    print(f"Title     : {content['title']}")
    print()

    # Swap in the stub footage provider for this run only.
    original_factory = (
        production_video.create_video_provider
    )

    production_video.create_video_provider = (
        lambda config, notify=None: StubFootageProvider(
            config,
            notify=notify
        )
    )

    try:

        result = (
            pipeline.production.run(
                episode_directory,
                content
            )
        )

    finally:

        production_video.create_video_provider = (
            original_factory
        )

    video_path = Path(result["video_path"])
    captions_path = Path(result["captions_path"])

    print()
    print("=" * 68)
    print("RENDERED")
    print("=" * 68)
    print("Video    :", video_path)
    print("Size     :", f"{video_path.stat().st_size:,} bytes")
    print("Captions :", captions_path)

    srt = captions_path.read_text(encoding="utf-8")
    blocks = [
        block
        for block in srt.strip().split("\n\n")
        if block.strip()
    ]

    print("Caption cues:", len(blocks))

    # No caption may be a lone word, and every word of the narration
    # must be captioned.
    captioned = 0
    for block in blocks:
        lines = block.splitlines()
        text = lines[2]
        assert text.strip(), block
        assert len(text.split()) >= 1, block
        captioned += len(text.split())

    narration_words = len(
        content["narration"].split()
    )

    # Captions are built from the narration engine's own word
    # timings, and that tokenizer splits at a few places where the
    # plain-text split does not - a closing quote followed by the next
    # word comes back as two tokens. So the counts track each other
    # closely rather than matching exactly on a long passage.
    tolerance = max(
        2,
        int(narration_words * 0.01)
    )

    print(
        f"Words captioned: {captioned} of {narration_words} "
        f"(tolerance {tolerance})"
    )

    assert abs(captioned - narration_words) <= tolerance, (
        "captions do not cover the narration"
    )

    singles = [
        block
        for block in blocks
        if len(block.splitlines()[2].split()) == 1
    ]
    assert not singles, singles

    print()
    print("FIRST CAPTION CUES")
    print("-" * 68)
    for block in blocks[:8]:
        print(block)
        print()

    print("VIDEO EXISTS:", video_path.is_file())
    print("LIVE VIDEO TEST PASSED")


if __name__ == "__main__":

    main()