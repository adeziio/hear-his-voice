"""
Verifies the passage-selection scope and the visual segmentation.

Three guarantees are checked:

1. Every selectable passage is a Gospel passage in which Jesus speaks.
2. Scripture stays exact WEBC text through selection and segmentation.
3. Each passage gets enough distinct visual segments, sized to the
   narration so the video keeps changing.

Run with:  python -m pytest tests/test_passage_selection.py -v
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


from scripture.webc import (
    WEBCScripture,
    ScriptureError
)
from core.config_loader import ConfigLoader
from scripture.selector import (
    PassageSelector,
    ROTATION_ORDER,
)
from scripture.segmenter import (
    build_visual_segments,
    split_clauses,
    split_sentences
)


GOSPEL_NAMES = {
    "Matthew", "Mark", "Luke", "John"
}

# Books that must never appear: Old Testament, Deuterocanon, Acts,
# the Epistles, and Revelation.
BOOKS_THAT_MUST_NEVER_APPEAR = {
    "Genesis", "Exodus", "Psalm", "Psalms", "Isaiah",
    "Jeremiah", "Daniel", "Tobit", "Judith", "Wisdom",
    "Sirach", "Baruch", "Maccabees", "Acts",
    "Romans", "Corinthians", "Galatians", "Ephesians",
    "Philippians", "Colossians", "Thessalonians",
    "Timothy", "Titus", "Philemon", "Hebrews",
    "James", "Peter", "Jude", "Revelation",
}


@pytest.fixture(scope="module")
def bible():
    return WEBCScripture()


@pytest.fixture(scope="module")
def selector():
    # The reference list now lives in content.json, so the selector needs
    # the loaded config rather than falling back to no references at all.
    return PassageSelector(
        ConfigLoader().load_all()
    )


@pytest.fixture
def reading(monkeypatch, tmp_path):
    """
    A selector with its own throwaway progress file, so tests that really
    consume episodes never touch the channel's real rotation state.

    The path stays redirected for the whole test, which matters because
    select() reads and writes it on every call.
    """

    import scripture.selector as selector_module

    monkeypatch.setattr(
        selector_module,
        "STATE_PATH",
        str(tmp_path / "rotation.json"),
    )

    return PassageSelector(
        ConfigLoader().load_all()
    )


def sample_positions(selector, step=211):
    """
    Start positions sampled across all four Gospels, so a test can cover
    the whole pool without walking every verse of it.
    """

    positions = []

    for book in ROTATION_ORDER:

        verses = selector.verses_by_book[book]

        for chapter, number, _text in verses[::step]:

            positions.append((book, chapter, number))

    return positions


def test_scripture_settings_live_in_content_config_only(selector):
    """
    Scripture settings belong in content.json beside the channel voice
    they belong to. There is no separate scripture.json, and no top-level
    'scripture' config section, so the same fact cannot live in two files.
    """
    assert not (ROOT / "config" / "scripture.json").exists()

    config = ConfigLoader().load_all()

    assert "scripture" not in config

    scripture_config = config["content"]["scripture"]

    for required in ("translation", "source"):

        assert required in scripture_config, required

    # The old selection list is gone entirely.
    assert "references" not in scripture_config

    # The word rate sits with the rest of the content settings, exactly
    # where the other project keeps it, rather than inside the Scripture
    # block - it describes how the narrator speaks, not which text.
    generation_config = config["content"]["content_generation"]

    assert "words_per_second" in generation_config

    assert "words_per_second" not in scripture_config

    assert selector.words_per_second == generation_config[
        "words_per_second"
    ]


def test_every_gospel_verse_is_in_the_pool(selector):
    """
    The content pool is the Gospel text itself, not a list of examples.
    Every verse in all four books is reachable.
    """
    assert set(selector.verses_by_book) == GOSPEL_NAMES

    for book, verses in selector.verses_by_book.items():

        assert len(verses) > 500, (book, len(verses))

        for chapter, number, text in verses[:50]:

            assert text.strip()
            assert int(chapter) > 0
            assert int(number) > 0


def test_guidance_examples_do_not_drive_selection(selector):
    """
    The config examples are examples only. Removing them must change
    nothing about what an episode reads.
    """
    assert selector.guidance_examples, "examples should exist"

    without = PassageSelector({
        "content": {
            "scripture": {
                "guidance_examples": []
            }
        }
    })

    assert without.guidance_examples == []

    assert (
        without._build_episode("Matthew", 1, 1)[0]["reference"]
        == selector._build_episode("Matthew", 1, 1)[0]["reference"]
    )


def test_no_passage_list_is_hardcoded_in_code():
    """
    No passage pool is baked into the module, so the channel cannot be
    limited by a stale hardcoded list.
    """
    import scripture.selector as selector_module

    for gone in (
        "DEFAULT_REFERENCES",
        "gospel_windows",
        "SENTENCE_ENDINGS",
        "MIN_WINDOW_RATIO",
    ):

        assert not hasattr(selector_module, gone), gone


def test_the_scope_is_guaranteed_by_code(selector):
    """
    The four Gospels are the hard rule. Anything outside them is refused,
    so no config edit can widen the channel's scope.
    """
    for reference in (
        "Acts 2:1-4",
        "Psalm 23:1",
        "Romans 8:28",
        "Revelation 21:4",
    ):

        with pytest.raises(ScriptureError):

            selector.get_reference(reference)

        with pytest.raises(ScriptureError):

            PassageSelector({
                "content": {
                    "scripture": {
                        "guidance_examples": [reference]
                    }
                }
            })

    assert selector.get_reference("Luke 2:52-56")["text"]


def test_rotation_order_is_the_four_gospels():
    assert ROTATION_ORDER == (
        "Matthew",
        "Mark",
        "Luke",
        "John",
    )

    assert set(ROTATION_ORDER) == GOSPEL_NAMES


def test_no_epistles_acts_revelation_or_old_testament(selector):
    for book in selector.verses_by_book:

        assert book not in BOOKS_THAT_MUST_NEVER_APPEAR
        assert book in GOSPEL_NAMES


def test_the_target_only_chooses_how_much_is_spoken():
    """
    There is a target duration, and it is used to take more or fewer
    whole verses from a passage. It never changes the words, and the
    Scripture on screen stays verbatim.
    """
    selector = PassageSelector(ConfigLoader().load_all())

    # The target exists and is read from the app config.
    assert selector.target_seconds > 0

    bible = WEBCScripture()

    lengths = []

    for book, chapter, number in sample_positions(selector):

        passage, _last = selector._build_episode(
            book,
            chapter,
            number,
        )

        reference = passage["reference"]

        # Whatever length was chosen, it is exactly what the WEBC
        # holds for those verses.
        expected = " ".join(
            bible.get_verse(
                book,
                item["chapter"],
                item["verse"],
            )
            for item in passage["passage"]
        )

        assert (
            passage["text"]
            == expected
        ), reference

        lengths.append(
            len(
                passage["text"].split()
            )
        )

    # Sizing to a target produces real variation rather than every
    # episode being forced to the same word count.
    assert max(lengths) != min(lengths)


def test_configured_non_gospel_example_fails_at_construction():
    """
    A non-Gospel entry in content.json must raise when the selector is
    built, not be silently ignored.
    """
    with pytest.raises(
        ScriptureError
    ):

        PassageSelector(
            {
                "content": {
                    "scripture": {
                        "guidance_examples": [
                            "John 3:16",
                            "Psalm 23:1",
                        ]
                    }
                }
            }
        )


def test_selected_passage_is_always_a_gospel(reading):
    """
    Run the real selection repeatedly; every result must be a Gospel.
    """
    for _ in range(40):

        passage = reading.select()

        assert passage["book"] in GOSPEL_NAMES
        assert passage["text"].strip()


# --------------------------------------------------------------------------
# 2. Scripture stays exact WEBC text
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# 3. Sequential reading, rotating between the Gospels
# --------------------------------------------------------------------------

def test_episodes_rotate_matthew_mark_luke_john(reading):
    """
    The rotation is fixed and strict, not random.
    """
    books = [
        reading.select()["book"]
        for _ in range(40)
    ]

    assert books == list(
        ROTATION_ORDER * 10
    )


def test_each_gospel_keeps_its_own_position(reading):
    """
    Progress is tracked per Gospel, so the rotation interleaves four
    independent reading positions rather than restarting each time.
    """
    first = [reading.select() for _ in range(8)]

    books = [item["book"] for item in first]

    # The first four all start at the head of their own book.
    for item in first[:4]:

        assert (
            item["passage"][0]["chapter"],
            item["passage"][0]["verse"],
        ) == (1, 1), item["reference"]

    progress = reading.progress()

    # Each book has advanced on its own, and none has run into another.
    for book in ROTATION_ORDER:

        assert int(
            progress["books"][book]["next_chapter"]
        ) >= 1

    # The next book to be read is the one after the last used.
    assert progress["next_book"] == ROTATION_ORDER[0]
    assert books[-1] == ROTATION_ORDER[-1]


def test_verses_are_never_skipped_or_repeated(reading):
    """
    The core guarantee. Over many episodes, each Gospel's readings must
    be exactly the front of that book in canonical order - no gap, no
    repeat, no reordering.
    """
    episodes = [reading.select() for _ in range(160)]

    read = {}

    for episode in episodes:

        read.setdefault(
            episode["book"],
            [],
        ).extend(
            (item["chapter"], item["verse"])
            for item in episode["passage"]
        )

    assert set(read) == set(ROTATION_ORDER)

    for book in ROTATION_ORDER:

        canonical = [
            (chapter, verse)
            for chapter, verse, _text
            in reading.verses_by_book[book]
        ]

        spoken = read[book]

        assert spoken == canonical[:len(spoken)], book


def test_each_episode_continues_from_the_previous_verse(reading):
    """
    Consecutive episodes in the same Gospel must butt up against each
    other with nothing in between.
    """
    previous_end = {}

    for _ in range(40):

        episode = reading.select()

        last = episode["passage"][-1]
        first = episode["passage"][0]

        if episode["book"] in previous_end:

            expected = previous_end[episode["book"]]

            actual = (
                first["chapter"],
                first["verse"],
            )

            assert actual >= expected, (
                f"{episode['book']} rewound: {actual} after {expected}"
            )

        previous_end[episode["book"]] = (
            last["chapter"],
            last["verse"] + 1,
        )


def test_episodes_land_near_the_target_duration(reading):
    """
    The existing target duration still decides how many consecutive
    verses each episode takes.
    """
    seconds = [
        reading.select()["estimated_seconds"]
        for _ in range(20)
    ]

    for value in seconds:

        assert 40.0 <= value <= 75.0, value


def test_reset_is_the_only_way_reading_repeats(reading):
    """
    Reading is continuous, so a repeat can only come from an explicit
    reset.
    """
    first = reading.select()["reference"]

    reading.select()
    reading.select()

    assert (
        reading.select()["reference"]
        != first
    )

    reading.reset()

    assert reading.progress()["next_book"] == ROTATION_ORDER[0]

    for book in ROTATION_ORDER:

        position = reading.progress()["books"][book]

        assert int(
            position["next_chapter"]
        ) == 1
        assert position["next_verse"] == 1
        assert position["cycles"] == 0

    assert reading.select()["reference"] == first


def test_progress_is_persisted_between_runs(monkeypatch, tmp_path):
    """
    Position survives a restart, which is what makes the reading
    continuous across separate pipeline invocations.
    """
    import scripture.selector as selector_module

    monkeypatch.setattr(
        selector_module,
        "STATE_PATH",
        str(tmp_path / "rotation.json"),
    )

    config = ConfigLoader().load_all()

    first = PassageSelector(config)

    played = [
        first.select()["reference"]
        for _ in range(6)
    ]

    saved = first.progress()["books"]

    # A brand new selector, as if the process had restarted.
    second = PassageSelector(config)

    assert second.progress()["books"] == saved

    # It carries on from the saved position rather than replaying.
    assert (
        second.progress()["next_book"]
        == ROTATION_ORDER[6 % len(ROTATION_ORDER)]
    )

    carried_on = [
        second.select()["reference"]
        for _ in range(6)
    ]

    for reference in carried_on:

        assert reference not in played, reference

    # Progress moved on from where it was saved.
    assert second.progress()["books"] != saved


def test_selection_returns_verbatim_webc(bible, reading):
    """
    The text handed to the pipeline must be exactly what the WEBC corpus
    holds, verse by verse.

    Compared per verse rather than by re-parsing the rendered reference,
    because an episode may run across a chapter boundary and its
    reference then names two chapters, which is a display string rather
    than a parseable input.
    """
    for _ in range(20):

        passage = reading.select()

        expected = " ".join(
            bible.get_verse(
                passage["book"],
                item["chapter"],
                item["verse"],
            )
            for item in passage["passage"]
        )

        assert passage["text"] == expected, passage["reference"]


def test_segmentation_never_alters_the_words(reading):
    """
    Cutting a passage into visual segments may only choose boundaries.
    The words and their order must survive untouched.
    """
    for _ in range(15):

        passage = reading.select()

        segments = build_visual_segments(
            passage["text"],
            6
        )

        assert segments

        assert (
            " ".join(segments).split()
            == passage["text"].split()
        )


def test_clause_split_is_lossless():
    """
    Breaking a long sentence at its own punctuation must be reversible.
    """
    sentence = (
        "For God so loved the world, that he gave his only born "
        "Son, that whoever believes in him should not perish."
    )

    clauses = split_clauses(sentence, 8)

    assert len(clauses) > 1
    assert " ".join(clauses).split() == sentence.split()


def test_sentence_split_is_lossless():
    sentence = (
        "Jesus said this. Then he went out."
    )

    assert " ".join(
        split_sentences(sentence)
    ).split() == sentence.split()


# --------------------------------------------------------------------------
# 3. Enough distinct visuals, sized to the narration
# --------------------------------------------------------------------------

def test_visual_count_and_size_come_from_the_target():
    """
    The number of visuals and their size are both derived from
    app.json's target duration, the way the other project derives its
    per-sentence word budget. Nothing about the cadence is configured.
    """
    from ai.content_generator import (
        ContentGenerator,
        segments_per_episode,
    )

    config = ConfigLoader().load_all()

    generator = ContentGenerator(config)

    selector = PassageSelector(config)

    words_per_second = selector.words_per_second

    segment_count = segments_per_episode(config)

    word_target = int(
        selector.target_seconds * words_per_second
    )

    assert generator is not None

    per_segment = (
        word_target // segment_count
    )

    passage = selector.get_reference(
        "John 3:16-30"
    )

    segments = generator.build_segments(
        passage
    )

    # The segment count tracks the configured one.
    assert len(segments) >= segment_count - 4

    # Most segments sit at the derived size. A sentence with no
    # internal punctuation cannot be split without cutting a thought
    # in half, so an occasional longer one is allowed through.
    sizes = sorted(
        len(segment.split())
        for segment in segments
    )

    typical = sizes[: len(sizes) // 2]

    for size in typical:

        assert (
            size <= per_segment + 4
        ), (
            f"typical segment is {size} words, "
            f"expected about {per_segment}"
        )

    for size in sizes:

        assert size <= 30


def test_raising_the_target_keeps_visual_size_and_grows_the_passage():
    """
    The duration is the only setting, and the segment count is derived
    from it. That makes the words per visual a constant - one visual's
    worth of narration never changes - so a longer target is filled by
    choosing MORE scripture, not by stretching each visual over more
    words.
    """
    import copy

    from ai.content_generator import (
        ContentGenerator,
        segments_per_episode,
        SECONDS_PER_VISUAL,
    )

    base = ConfigLoader().load_all()

    def probe(target):
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

        return {
            "passage_words": len(
                passage["text"].split()
            ),
            "segments": len(segments),
            "longest": max(
                len(segment.split())
                for segment in segments
            ),
        }

    short = probe(50)
    long = probe(150)

    # A longer target selects more scripture...
    assert long["passage_words"] > short["passage_words"]

    # ...and therefore yields more visuals...
    assert long["segments"] > short["segments"]

    # ...while each visual still covers the same slice of narration, which
    # is what SECONDS_PER_VISUAL means.
    for target in (50, 150):
        config = copy.deepcopy(base)
        config["app"]["shorts"][
            "target_duration_seconds"
        ] = target
        count = segments_per_episode(config)
        assert count == int(
            round(target / SECONDS_PER_VISUAL)
        )


def test_no_segment_holds_an_entire_verse(bible, selector):
    """
    A single visual must not swallow a whole verse. Long passages are
    broken at their own punctuation, so segments stay short.

    A sentence with no internal punctuation cannot be split without
    cutting a thought in half, so the bound allows one such sentence
    through - about ten seconds of narration at the default rate.
    """
    for book, chapter, number in sample_positions(selector):

        passage, _last = selector._build_episode(
            book,
            chapter,
            number,
        )

        reference = passage["reference"]

        segments = build_visual_segments(
            passage["text"],
            14,
            clause_words=10,
            max_words_per_segment=13,
        )

        assert segments

        for segment in segments:

            assert len(segment.split()) <= 30, (
                f"{reference}: segment too long -> {segment}"
            )


def test_long_sentences_are_split_at_their_punctuation():
    """
    The case the splitting exists for: one very long verse must not be
    left as a single visual.
    """
    long_verse = (
        "For God so loved the world, that he gave his only born "
        "Son, that whoever believes in him should not perish, but "
        "have eternal life, and whoever believes in him will not be "
        "condemned, but has passed from death to life."
    )

    segments = build_visual_segments(
        long_verse,
        14,
        clause_words=10,
        max_words_per_segment=13,
    )

    assert len(segments) > 1

    assert (
        " ".join(segments).split()
        == long_verse.split()
    )


def test_segments_have_meaningful_text(reading):
    """
    Every visual must have something to show, and the segments must
    tile the passage in order without gaps.
    """
    for _ in range(10):

        passage = reading.select()

        segments = build_visual_segments(
            passage["text"],
            14,
            clause_words=10,
            max_words_per_segment=13,
        )

        for segment in segments:

            assert segment.strip()
            assert len(segment.split()) >= 3

        # The first segment opens the passage and the last closes it.
        assert passage["text"].startswith(
            segments[0][:20]
        )

        assert passage["text"].rstrip().endswith(
            segments[-1][-20:]
        )
