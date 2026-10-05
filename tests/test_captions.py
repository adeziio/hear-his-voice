"""
Verifies the caption builder used on screen.

Captions must be meaningful phrases synchronised to the narration of
the exact WEBC text. Word-by-word or arbitrary single-word cues would
make Scripture unreadable, so the phrase grouping is checked here.

Run with:  python -m pytest tests/test_captions.py -v
"""

import sys

from pathlib import Path

import pytest


sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parent.parent
    )
)


from production.captions import (
    build_caption_cues,
    build_word_cues,
    write_srt
)

from scripture.webc import WEBCScripture


NARRATION = (
    "For God so loved the world, that he gave his only born Son, "
    "that whoever believes in him should not perish, but have "
    "eternal life. For God didn’t send his Son into the world to "
    "judge the world, but that the world should be saved through him."
)


def _word_timings(text):
    """
    Builds evenly spaced word timings, which is all the caption
    builder needs to be exercised.
    """
    words = text.split()

    timings = []
    start = 0.0

    for index, word in enumerate(words):

        timings.append({
            "word": word,
            "start": round(start, 3),
            "end": round(start + 0.38, 3),
        })

        start += 0.45

    return timings


@pytest.fixture(scope="module")
def words():
    return _word_timings(NARRATION)


def test_cues_are_phrases_not_single_words(words):
    """
    Every cue is a readable phrase, so a viewer can follow the text
    rather than watching one word blink at a time.
    """
    cues = build_caption_cues(
        words,
        max_words_per_line=4
    )

    assert cues

    single_word = [
        cue
        for cue in cues
        if len(cue["text"].split()) == 1
    ]

    assert not single_word, single_word


def test_cues_carry_readable_phrases_not_loose_words(words):
    """
    The channel is devotional, so captions show whole phrases of the
    passage rather than one large word popping up at a time. Several words
    must share each cue, and none may be a lone word.
    """
    cues = build_caption_cues(
        words,
        max_words_per_line=7
    )

    assert cues

    single_word = [
        cue
        for cue in cues
        if len(cue["text"].split()) == 1
    ]

    assert not single_word, single_word

    # Phrases, not pairs: the configured width should actually be used.
    assert max(
        len(cue["text"].split())
        for cue in cues
    ) >= 4


def test_cues_respect_the_configured_line_length(words):
    max_words = 7

    cues = build_caption_cues(
        words,
        max_words_per_line=max_words
    )

    for cue in cues:

        assert (
            len(cue["text"].split())
            <= max_words
        ), cue["text"]


def test_the_configured_captions_fit_the_frame(words):
    """
    Longer phrases must never run off the edge of the video. The composer
    wraps each cue to the safe area and shrinks the font only if a single
    word still will not fit, so check both outcomes with real font
    measurements at the configured size.
    """
    import json

    from PIL import (
        Image,
        ImageDraw,
        ImageFont
    )

    from production.composer import Composer

    config = json.loads(
        (
            Path(__file__).resolve().parent.parent
            / "config"
            / "app.json"
        ).read_text(encoding="utf-8")
    )

    captions = config["captions"]

    # This channel reads as phrases, not single words.
    assert captions["max_words_per_line"] >= 5

    font_size = int(captions["font_size"])
    stroke_width = int(captions["stroke_width"])

    # A devotional register needs a calmer, smaller type size than a
    # high-energy one-word-at-a-time channel.
    assert font_size <= 52

    frame_width = int(config["shorts"]["resolution"]["width"])

    safe_width = (
        int(frame_width * 0.92)
        - 2 * stroke_width
    )

    font_path = "C:/Windows/Fonts/arialbd.ttf"

    if not Path(font_path).exists():
        pytest.skip("Arial is not available on this machine")

    composer = Composer.__new__(Composer)

    cues = build_caption_cues(
        words,
        max_words_per_line=int(captions["max_words_per_line"])
    )

    draw = ImageDraw.Draw(
        Image.new("RGB", (1, 1))
    )

    for cue in cues:

        wrapped, resolved_size = composer._wrap_caption_text(
            cue["text"],
            font_path,
            font_size,
            stroke_width,
            safe_width
        )

        font_pil = ImageFont.truetype(
            font_path,
            resolved_size
        )

        for line in wrapped.split("\n"):

            box = draw.textbbox(
                (0, 0),
                line,
                font=font_pil
            )

            width = box[2] - box[0]

            assert width <= safe_width, (
                f"{cue['text']!r} still overflows at "
                f"{resolved_size}px: {width} > {safe_width}"
            )


def test_cues_last_long_enough_to_read(words):
    """
    A cue that flashes for a fraction of a second is unreadable, so
    the short-cue merge must leave every cue on screen long enough.
    """
    cues = build_caption_cues(
        words,
        max_words_per_line=4
    )

    for cue in cues:

        duration = cue["end"] - cue["start"]

        assert (
            duration >= 0.6
        ), (cue["text"], duration)


def test_cues_cover_the_whole_narration(words):
    """
    No word may be dropped between the first and last cue.
    """
    cues = build_caption_cues(
        words,
        max_words_per_line=4
    )

    assert (
        cues[0]["start"]
        == words[0]["start"]
    )

    assert (
        cues[-1]["end"]
        == words[-1]["end"]
    )

    spoken = sum(
        len(cue["text"].split())
        for cue in cues
    )

    assert spoken == len(words)


def test_word_cues_remain_available_for_karaoke(words):
    """
    The per-word cues are still produced, but they are used for the
    word-level highlight only - never as the on-screen caption text.
    """
    word_cues = build_word_cues(words)

    assert len(word_cues) == len(words)

    assert all(
        len(cue["text"].split()) == 1
        for cue in word_cues
    )


def test_srt_is_written_from_the_phrase_cues(
    words,
    tmp_path
):
    cues = build_caption_cues(
        words,
        max_words_per_line=4
    )

    path = write_srt(
        cues,
        tmp_path / "captions" / "captions.srt"
    )

    body = path.read_text(
        encoding="utf-8"
    )

    assert "1\n" in body
    assert "-->" in body

    # Timestamps are well formed for a standard subtitle file.
    assert "00:00:00," in body