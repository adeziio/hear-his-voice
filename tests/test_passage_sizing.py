"""
Verifies that a passage is sized to the target duration by taking more
or fewer WHOLE verses from a configured window.

The target is a guide, not a limit: the result may sit either side of
it when the verse boundaries do not line up, and a verse is never cut
in half.

Run with:  python -m pytest tests/test_passage_sizing.py -v
"""

import copy
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


from core.config_loader import ConfigLoader
from scripture.webc import WEBCScripture
from scripture.selector import (
    PassageSelector,
    ROTATION_ORDER,
    fit_passage_to_duration,
    estimate_seconds,
)


# The single duration setting, read from config rather than repeated here,
# so editing app.json can never leave this test asserting a stale number.
TARGET_SECONDS = float(
    ConfigLoader().load_all()["app"]["shorts"]["target_duration_seconds"]
)


@pytest.fixture(scope="module")
def selector():
    return PassageSelector(
        ConfigLoader().load_all()
    )


@pytest.fixture(scope="module")
def bible():
    return WEBCScripture()


def sample_positions(selector, step=211):
    """
    Start positions sampled across all four Gospels, so a test can cover
    the whole pool without walking every verse of it.
    """

    positions = []

    for book in ROTATION_ORDER:

        for chapter, number, _text in selector.verses_by_book[book][::step]:

            positions.append((book, chapter, number))

    return positions


def _verses(*word_counts):
    return [
        {
            "verse": index + 1,
            "text": " ".join(
                ["word"] * count
            ),
        }
        for index, count in enumerate(word_counts)
    ]


# --------------------------------------------------------------------------
# The fitting rule itself
# --------------------------------------------------------------------------

def test_it_takes_more_verses_when_the_passage_is_short():
    # Ten verses of 30 words. The target is 60s = 174 words: five
    # verses is 150 words (51.7s), six is 180 words (62.1s). Six is
    # nearer, so the passage grows until it reaches the target.
    verses = _verses(*([30] * 10))

    assert (
        fit_passage_to_duration(
            verses,
            TARGET_SECONDS,
            2.9
        )
        == 6
    )


def test_it_takes_fewer_verses_when_the_verse_is_long():
    # A single 200-word verse is already past the target, and it must
    # be kept whole rather than truncated.
    verses = _verses(200, 30)

    assert (
        fit_passage_to_duration(
            verses,
            TARGET_SECONDS,
            2.9
        )
        == 1
    )


def test_it_keeps_whichever_boundary_is_closer():
    # Target is 50s = 145 words. 20+20+90 = 130 words (44.8s); adding
    # the next 20-word verse gives 150 words (51.7s). Including it is
    # nearer, so four verses are kept.
    verses = _verses(20, 20, 90, 20)

    assert (
        fit_passage_to_duration(
            verses,
            TARGET_SECONDS,
            2.9
        )
        == 4
    )


def test_it_uses_every_verse_when_the_passage_cannot_reach_the_target():
    # Target is 60s = 174 words. 20+20+100 = 140 words (48.3s); adding
    # the next verse gives 160 words (55.2s), still short of the target,
    # and the verses run out, so all four are kept.
    verses = _verses(20, 20, 100, 20)

    assert (
        fit_passage_to_duration(
            verses,
            TARGET_SECONDS,
            2.9
        )
        == 4
    )


def test_it_uses_the_whole_passage_when_it_is_too_short():
    verses = _verses(10, 10, 10)
# --------------------------------------------------------------------------
# Applied to the real rotation
# --------------------------------------------------------------------------

def test_the_target_comes_from_the_config(selector):
    assert selector.target_seconds == float(
        TARGET_SECONDS
    )


# --------------------------------------------------------------------------
# One place to change it
# --------------------------------------------------------------------------

def test_the_target_is_defined_in_app_config_only():
    """
    The target duration lives in exactly one place:
    app.json -> shorts.target_duration_seconds, which is where the
    other project keeps it too. The Scripture settings inside
    content.json must not repeat it, or the two would drift and editing
    one would silently do nothing.
    """
    config = ConfigLoader().load_all()

    assert "target_seconds" not in config["content"]["scripture"]

    assert config["app"]["shorts"][
        "target_duration_seconds"
    ] == TARGET_SECONDS


def test_changing_the_app_target_changes_the_sizing():
    """
    Proves the app.json value is actually the one in use, rather than
    being shadowed by a copy somewhere else.
    """
    config = ConfigLoader().load_all()

    config["app"]["shorts"][
        "target_duration_seconds"
    ] = 20

    selector = PassageSelector(config)

    assert selector.target_seconds == 20.0

    passage = selector.get_reference(
        "John 3:16-30"
    )

    short = len(
        passage["text"].split()
    )

    config["app"]["shorts"][
        "target_duration_seconds"
    ] = 80

    longer = PassageSelector(
        config
    ).get_reference("John 3:16-30")

    assert len(
        longer["text"].split()
    ) > short


# --------------------------------------------------------------------------
# No duplicated numbers, and everything reacts to the one target
# --------------------------------------------------------------------------

def test_the_target_is_not_repeated_anywhere_else():
    """
    app.json owns the target. No other config may repeat it, or the
    copies would drift apart.
    """
    config = ConfigLoader().load_all()

    scripture_config = config["content"]["scripture"]

    for forbidden in (
        "target_seconds",
        "target_duration_seconds",
        "target",
    ):

        assert forbidden not in scripture_config, (
            f"content.json scripture settings repeat '{forbidden}'"
        )


def test_visual_settings_live_with_the_footage_provider():
    """
    The visual cadence is not configured at all. It is derived from the
    episode target the same way the other project derives its
    per-sentence word budget, so pexels.json stays exactly as it is
    there - provider mechanics only.
    """
    config = ConfigLoader().load_all()

    assert "visuals" not in config["pexels"]
    assert "visuals" not in config["content"]["scripture"]

    # No seconds-per-visual or per-segment settings anywhere.
    for name in ("app", "content", "pexels"):
        for forbidden in (
            "target_seconds_per_visual",
            "min_seconds_per_visual",
            "max_seconds_per_visual",
            "clause_seconds",
            "max_segment_seconds",
            "clause_words",
            "max_words_per_visual",
        ):
            assert forbidden not in config[name], (
                f"{name} still defines {forbidden}"
            )


def test_the_segment_count_is_derived_from_the_target_duration():
    """
    The target duration is the only duration setting. The segment count -
    and so the visual count - follows from it, so the two numbers can
    never be edited out of step with each other.
    """
    from ai.content_generator import (
        segments_per_episode,
        SECONDS_PER_VISUAL,
        DEFAULT_TARGET_SECONDS
    )

    config = ConfigLoader().load_all()

    target = config["app"]["shorts"]["target_duration_seconds"]

    # At the configured target the derived count is what the pipeline was
    # tuned to: one visual every SECONDS_PER_VISUAL.
    assert segments_per_episode(config) == int(
        round(target / SECONDS_PER_VISUAL)
    )

    # The key that used to hold the count is gone entirely.
    assert "segments_per_episode" not in config["app"]["shorts"]

    # The duration alone drives it: halving the target roughly halves the
    # segments, doubling it roughly doubles them.
    config["app"]["shorts"]["target_duration_seconds"] = 30
    short = segments_per_episode(config)

    config["app"]["shorts"]["target_duration_seconds"] = 90
    long = segments_per_episode(config)

    assert short < segments_per_episode(
        ConfigLoader().load_all()
    ) < long

    # Missing or unusable config falls back to the default target rather
    # than raising.
    assert segments_per_episode({}) == int(
        round(DEFAULT_TARGET_SECONDS / SECONDS_PER_VISUAL)
    )
    assert segments_per_episode(
        {"app": {"shorts": {"target_duration_seconds": "nonsense"}}}
    ) == int(round(DEFAULT_TARGET_SECONDS / SECONDS_PER_VISUAL))


def test_raising_the_target_scales_words_and_visuals():
    """
    Changing only app.json must lengthen the passage and add visuals,
    with no other setting touched.
    """
    from ai.content_generator import (
        ContentGenerator
    )

    base = ConfigLoader().load_all()

    def run(target):
        config = copy.deepcopy(base)
        config["app"]["shorts"][
            "target_duration_seconds"
        ] = target

        selector = PassageSelector(config)
        generator = ContentGenerator(config)

        passage = selector.get_reference(
            "Luke 15:11-32"
        )

        segments = generator.build_segments(
            passage
        )

        return (
            len(passage["text"].split()),
            len(segments),
        )

    short_words, short_visuals = run(50)
    long_words, long_visuals = run(150)

    assert long_words > short_words
    assert long_visuals > short_visuals


def test_each_passage_keeps_whole_verses(selector):
    """
    Every position must resolve to a run of whole verses that starts
    exactly where it was asked to start, with no verse cut in half.

    Verse NUMBERS can skip: Luke 17:36 exists in the WEBC only as a
    manuscript footnote, so it has no body text and is correctly absent.
    A passage may also cross a chapter boundary, so the check is against
    the whole book read in order rather than one chapter.
    """
    for book, chapter, number in sample_positions(selector):

        passage, _last = selector._build_episode(
            book,
            chapter,
            number,
        )

        window = passage["reference"]

        verses = passage["passage"]

        assert verses, window

        # It opens exactly at the requested verse.
        assert (
            verses[0]["chapter"],
            verses[0]["verse"],
        ) == (chapter, number), window

        # And it is a contiguous run of the book's verses, in order.
        canonical = [
            (entry_chapter, entry_verse)
            for entry_chapter, entry_verse, _text
            in selector.verses_by_book[book]
        ]

        start = canonical.index((chapter, number))

        taken = [
            (item["chapter"], item["verse"])
            for item in verses
        ]

        assert taken == canonical[start:start + len(taken)], window

        # Every verse is the source's own wording, untouched.
        for item in verses:

            expected = selector.scripture.get_verse(
                book,
                item["chapter"],
                item["verse"],
            )

            assert item["text"] == expected, window


def test_the_passage_stays_inside_its_window(selector):
    """
    Sizing may only stop early. It must never invent verses that are not
    there, nor read past the end of the book.
    """
    for book, chapter, number in sample_positions(selector):

        passage, _last = selector._build_episode(
            book,
            chapter,
            number,
        )

        window = passage["reference"]

        available = selector._verses_from(
            book,
            chapter,
            number,
        )

        taken = passage["passage"]

        assert len(taken) <= len(available), window

        # What was taken is exactly the start of what was available.
        assert [
            (item["chapter"], item["verse"])
            for item in taken
        ] == [
            (item["chapter"], item["verse"])
            for item in available[:len(taken)]
        ], window


def test_sized_text_is_verbatim_webc(bible, selector):
    """
    Sizing only chooses where to stop; the wording is untouched.
    """
    for book, chapter, number in sample_positions(selector):

        passage, _last = selector._build_episode(
            book,
            chapter,
            number,
        )

        window = passage["reference"]

        expected = " ".join(
            bible.get_verse(
                passage["book"],
                item["chapter"],
                item["verse"],
            )
            for item in passage["passage"]
        )

        assert (
            passage["text"]
            == expected
        ), window


def test_sized_passages_land_near_the_target(selector):
    """
    Episodes come out around the target. A passage may sit either side
    of it - that is the point of using whole verses - but none should
    run away from it.
    """
    for book, chapter, number in sample_positions(selector):

        passage, _last = selector._build_episode(
            book,
            chapter,
            number,
        )

        window = passage["reference"]

        seconds = estimate_seconds(
            len(
                passage["text"].split()
            ),
            selector.words_per_second,
        )

        assert seconds <= 95.0, (
            f"{window} -> {seconds:.0f}s"
        )


def test_no_window_produces_a_stub_episode(selector):
    """
    A window must always have enough runway to reach a real episode.

    This is the failure that produced a two-verse, twelve-second
    episode: a window like "John 10:10-11" had only two verses to work
    with, so no amount of sizing could get it near the target. The pool
    therefore only opens windows with enough chapter left to speak for a
    usable episode, and every one is checked to land in range.
    """
    for book, chapter, number in sample_positions(selector):

        passage, _last = selector._build_episode(
            book,
            chapter,
            number,
        )

        window = passage["reference"]

        seconds = estimate_seconds(
            len(
                passage["text"].split()
            ),
            selector.words_per_second,
        )

        assert seconds >= 30.0, (
            f"{window} -> only {seconds:.0f}s"
        )

        assert seconds <= 70.0, (
            f"{window} -> {seconds:.0f}s is too long"
        )


def test_a_window_shrinks_when_it_is_too_long(selector):
    """
    Windows must actually be trimmed, otherwise the sizing is inert.
    """
    shrunk = []

    for book, chapter, number in sample_positions(selector):

        passage, _last = selector._build_episode(
            book,
            chapter,
            number,
        )

        window = passage["reference"]

        available = selector._verses_from(
            passage["book"],
            chapter,
            number,
        )

        if len(passage["passage"]) < len(available):

            shrunk.append(window)

    assert shrunk, (
        "no window was trimmed; the sizing is inert"
    )


def test_a_window_takes_more_than_one_verse(selector):
    """
    Short passages grow by taking further verses, which is how they
    reach the target.
    """
    grew = []

    for book, chapter, number in sample_positions(selector):

        passage, _last = selector._build_episode(
            book,
            chapter,
            number,
        )

        window = passage["reference"]

        if len(
            passage["passage"]
        ) > 1:

            grew.append(window)

    assert grew, (
        "no window took more than one verse"
    )


def test_it_never_returns_nothing():
    assert (
        fit_passage_to_duration(
            [],
            TARGET_SECONDS,
            2.9
        )
        == 0
    )


def test_estimate_seconds_is_sane():
    assert estimate_seconds(145, 2.9) == pytest.approx(
        50.0
    )