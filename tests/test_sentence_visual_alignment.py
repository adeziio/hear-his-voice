"""
Sentence-locked narration and visual generation.

Hear His Voice speaks one narration sentence per SnapGenAI clip, so the
narration is cut only at completed sentence boundaries and every sentence
gets exactly one visual prompt.

`duration_target` in config/ai_models.json is a CEILING, not a quota: a
sentence shorter than the clip is kept whole rather than padded, and only a
sentence that would run longer than the clip is divided - at its own
punctuation, into shorter source-faithful sentences.

These tests cover that timing, the preservation of the narration text, the
ordering, the one-to-one narration/visual mapping, and the visual accuracy
rules the prompts are held to.
"""

import json
import re

from pathlib import Path

import pytest

import ai.content_generator as content_generator_module

from core.config_loader import ConfigLoader

from ai.content_generator import (
    ContentGenerator,
    ContentGenerationError,
    _duration_target_seconds,
    _max_words_per_segment,
    _words_per_second,
    METADATA_SCHEMA,
    NARRATION_SCHEMA,
    NARRATION_VERIFICATION_SCHEMA,
    SEGMENT_GROUPING_SCHEMA,
    VISUAL_SCHEMA,
    VISUAL_VERIFICATION_SCHEMA,
)

from scripture.segmenter import (
    _protected_spans,
    build_word_segments,
    split_sentences,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


STORM_NARRATION = (
    "A violent storm rose on the sea until the waves covered the boat, "
    "yet he slept on a cushion at the stern. They woke him, crying, "
    "'Save us, Lord! We are dying!' He rose and rebuked the wind, "
    "and the sea became calm."
)


# The narration of the generated episode media/output/shorts/001, kept
# verbatim as the regression case for story-beat grouping: that episode
# shipped one visual per narration unit (15 visuals, no merges) even
# though several consecutive pairs formed one beat and fit the clip.
EPISODE_001_NARRATION = (
    "Jesus sent the crowds away and told his disciples to get into a "
    "boat and go ahead of him. He went up onto a mountain by himself "
    "to pray. As evening came, he was alone there. But the boat was "
    "now in the middle of the sea, struggling against the waves as "
    "the wind blew strongly against them. In the fourth watch of the "
    "night, Jesus came walking on the water toward them. When the "
    "disciples saw him, they were terrified and cried out, thinking "
    "it was a ghost. Immediately, Jesus spoke to them, saying, "
    "“Cheer up! It is I! Don’t be afraid.” Peter "
    "answered, “Lord, if it is you, command me to come to you "
    "on the water.” He said, “Come!” Peter stepped "
    "down from the boat and walked on the water toward Jesus. But "
    "when he saw how strong the wind was, he became afraid and began "
    "to sink. He cried out, “Lord, save me!” Immediately "
    "Jesus reached out his hand, took hold of him, and said, "
    "“You of little faith, why did you doubt?” When they "
    "got into the boat, the wind ceased. Those in the boat came and "
    "worshiped him, saying, “You are truly the Son of God!”"
)


# Episode 001's REGENERATED content (the story-beat grouping run),
# kept verbatim as the regression case for visual accuracy: the
# grouping worked - 12 visuals from 15 units - but three shipped
# prompts put the scene at odds with the passage: the crowd shown
# boarding the boat for the sentence that sends the crowd away, the
# disciples shown climbing aboard from shore when the passage has
# them already at sea, and Jesus reduced to "a solitary figure".
EPISODE_001_REGENERATED_NARRATION = (
    "Jesus sent the crowds away and told his disciples to get into "
    "the boat and go ahead of him. He went up onto a mountain by "
    "himself to pray. When evening came, he was alone there. But "
    "the boat was now in the middle of the sea, struggling against "
    "the waves as the wind blew strongly against them. In the "
    "fourth watch of the night, Jesus came walking on the water "
    "toward them. The "
    "disciples saw him and were terrified, crying out in fear, "
    "thinking it was a ghost. Immediately Jesus spoke to them, "
    "saying, 'Cheer up! It is I! Don’t be afraid.' Peter answered, "
    "'Lord, if it is you, command me to come to you on the waters.' "
    "He said, 'Come!' Peter stepped down from the boat and walked on "
    "the water toward Jesus. But when he saw how strong the wind "
    "was, he became afraid and began to sink. He cried out, 'Lord, "
    "save me!' Immediately Jesus reached out his hand, took hold of "
    "him, and said, 'You of little faith, why did you doubt?' When "
    "they got into the boat, the wind ceased. Those in the boat came "
    "and worshiped him, saying, 'You are truly the Son of God!'"
)


# The 12 shipped sentence-visual pairs, as episode 001's content.json
# holds them.
EPISODE_001_SHIPPED_SEGMENTS = (
    "Jesus sent the crowds away and told his disciples to get into "
    "the boat and go ahead of him.",
    "He went up onto a mountain by himself to pray. When evening "
    "came, he was alone there.",
    "But the boat was now in the middle of the sea, struggling "
    "against the waves as the wind blew strongly against them.",
    "In the fourth watch of the night, Jesus came walking on the "
    "water toward them.",
    "The disciples saw him and were terrified, crying out in fear, "
    "thinking it was a ghost.",
    "Immediately Jesus spoke to them, saying, 'Cheer up! It is I! "
    "Don’t be afraid.'",
    "Peter answered, 'Lord, if it is you, command me to come to you "
    "on the waters.' He said, 'Come!'",
    "Peter stepped down from the boat and walked on the water toward "
    "Jesus.",
    "But when he saw how strong the wind was, he became afraid and "
    "began to sink. He cried out, 'Lord, save me!'",
    "Immediately Jesus reached out his hand, took hold of him, and "
    "said, 'You of little faith, why did you doubt?'",
    "When they got into the boat, the wind ceased.",
    "Those in the boat came and worshiped him, saying, 'You are "
    "truly the Son of God!'",
)


# The two prompts episode 001 shipped against the passage, verbatim.
EPISODE_001_BAD_CROWD_BOARDING = (
    "Style: realistic. A group of disciples are gathering on a rocky "
    "shoreline at dusk, helping people from a crowd to climb into a "
    "wooden boat as Jesus stands nearby, watching them with calm "
    "authority. The sky is painted in deep indigo and gold, the sea "
    "gently lapping against the shore. Medium shot, focusing on the "
    "disciples' hands assisting others while Jesus remains slightly "
    "apart, facing them but not yet moving toward the boat."
)

EPISODE_001_BAD_UNNAMED_JESUS = (
    "Style: realistic. A wide shot of a solitary figure standing "
    "atop a rocky hillside at dusk, facing away from the viewer with "
    "arms raised in prayer, the sky transitioning to deep indigo as "
    "the first stars appear above, soft light casting long shadows "
    "across the landscape."
)

EPISODE_001_BAD_SHORE_BOARDING = (
    "Style: realistic. A medium shot of disciples climbing into a "
    "small wooden boat on the shore of a stormy lake at night, their "
    "faces illuminated by the faint light of distant torches, as the "
    "wind begins to die down and the water grows calm behind them."
)


class StubLLM:
    """
    Stands in for the Ollama provider so generation runs offline.

    It honours the per-sentence contract: one VISUAL_SCHEMA call returns
    a single visual object, one METADATA_SCHEMA call returns the episode
    metadata, and a narration call returns the narration. Every prompt is
    captured so tests can assert how many calls were made and what each
    was asked to direct.
    """

    def __init__(
        self,
        narration="Jesus stilled the storm.",
        queries=None,
        verifications=None,
        narration_verifications=None,
        verification_reply=None,
        grouping_replies=None
    ):

        self.narration = narration
        self.queries = queries
        # Scripted verdicts, consumed in check order. When the script
        # runs out the checker approves, so a test that is not about
        # verification sees every check pass on the first attempt.
        self.verifications = list(verifications or [])
        self.narration_verifications = list(
            narration_verifications or []
        )
        # When set, returned verbatim instead of a verdict object -
        # used to cover the fail-closed path on an unparseable reply.
        self.verification_reply = verification_reply
        # Scripted story-beat partitions, consumed in call order.
        self.grouping_replies = list(grouping_replies or [])
        self.prompts = []
        self.visual_prompts = []
        self.metadata_prompts = []
        self.narration_prompts = []
        self.verification_prompts = []
        self.narration_verification_prompts = []
        self.grouping_prompts = []
        self.visual_calls = 0

    def generate(
        self,
        prompt,
        response_format=None
    ):

        self.prompts.append(prompt)

        # Grouping calls come first of all: a story-beat call must
        # never consume a verdict or a visual reply. When the scripted
        # replies run out, every unit gets its own visual - the safe
        # default a test that is not about grouping expects.
        if response_format is SEGMENT_GROUPING_SCHEMA:

            self.grouping_prompts.append(prompt)

            if self.grouping_replies:

                return self.grouping_replies.pop(0)

            numbers = re.findall(r"(?m)^(\d+)\. ", prompt)
            count = max(
                (int(number) for number in numbers),
                default=1,
            )

            return {
                "groups": [
                    [unit]
                    for unit in range(1, count + 1)
                ]
            }

        # Verification calls share the transport but not the contract:
        # they must be routed before any generation call, or a check
        # would silently consume a visual reply.
        if response_format is VISUAL_VERIFICATION_SCHEMA:

            self.verification_prompts.append(prompt)

            if self.verification_reply is not None:

                return self.verification_reply

            aligned = (
                self.verifications.pop(0)
                if self.verifications
                else True
            )

            return {
                "aligned": aligned,
                "reason": (
                    ""
                    if aligned
                    else "The query shows the wrong beat."
                ),
            }

        if response_format is NARRATION_VERIFICATION_SCHEMA:

            self.narration_verification_prompts.append(prompt)

            accurate = (
                self.narration_verifications.pop(0)
                if self.narration_verifications
                else True
            )

            return {
                "accurate": accurate,
                "issues": (
                    ""
                    if accurate
                    else
                    "The narration gives the action to the wrong "
                    "person."
                ),
            }

        if response_format is NARRATION_SCHEMA:

            self.narration_prompts.append(prompt)

            return {
                "narration": self.narration
            }

        if response_format is METADATA_SCHEMA:

            self.metadata_prompts.append(prompt)

            return {
                "title": "Jesus Stills the Storm",
                "summary": "Jesus rebukes the wind and the sea becomes calm.",
                "mood": "reverent",
            }

        # A per-sentence visual call: return one visual object for this
        # segment. The next query in the list maps to the next segment.
        self.visual_prompts.append(prompt)

        index = self.visual_calls
        self.visual_calls += 1

        if self.queries is not None and index < len(self.queries):

            query = self.queries[index]

        else:

            query = f"Style: realistic. Shot {index + 1}."

        return {
            "search_query": query,
            "visual_direction": query,
        }


def make_config(
    duration_target="8s",
    words_per_second=2.8,
    verify_visuals=True,
    verify_narration=True,
    story_beat_grouping=True,
    grouping_attempts=3
):
    """
    A configuration holding only the keys the generator reads, so each test
    sets its own clip length, speaking rate, and verification behaviour.
    """
    return {
        "ai_models": {
            "models": {
                "video_model": {
                    "snapgenai": {
                        "duration_target": duration_target
                    }
                }
            }
        },
        "content": {
            "content_generation": {
                "words_per_second": words_per_second,
                "verify_visuals": verify_visuals,
                "verify_narration": verify_narration,
                "story_beat_grouping": story_beat_grouping,
                "grouping_attempts": grouping_attempts
            },
            "visual_rules": [
                "Generate exactly one distinct SnapGenAI visual prompt "
                "for each narration segment."
            ],
            "narration_rules": [
                "Keep every fact from the passage."
            ],
        },
        "channel": {
            "name": "Hear His Voice",
            "description": "Scripture retold.",
        },
    }


def make_generator(
    llm=None,
    **config_overrides
):

    config = make_config(**config_overrides)

    # __new__ skips the real __init__ so no OllamaProvider is built for an
    # offline test; every attribute the code under test touches is set here
    # explicitly. log_prefix is what BaseAIService.log() prints, so it has to
    # be present or build_segments() raises on its logging line.
    generator = ContentGenerator.__new__(ContentGenerator)
    generator.config = config
    generator.generation_config = config["content"]["content_generation"]
    generator.channel_config = config
    generator.log_prefix = "TEST"

    stub = llm or StubLLM()
    generator.llm = stub

    return generator, stub


@pytest.fixture
def project_config():
    """
    The project's real configuration, so the committed duration_target and
    speaking rate are covered rather than a test-only value.
    """
    return ConfigLoader(
        str(PROJECT_ROOT / "config")
    ).load_all()


# --------------------------------------------------------------------
# duration_target drives the segment word budget
# --------------------------------------------------------------------

def test_the_segment_budget_comes_from_duration_target():

    config = make_config(
        duration_target="8s",
        words_per_second=2.8
    )

    seconds = _duration_target_seconds(config)
    rate = _words_per_second(config)

    assert seconds == 8.0

    # The budget is the duration-derived product, never a constant:
    # a longer clip allows a longer chunk; a shorter one does not.
    assert _max_words_per_segment(config) == round(rate * seconds)

    assert _max_words_per_segment(
        make_config(duration_target="12s")
    ) == round(2.8 * 12.0)

    assert _max_words_per_segment(
        make_config(duration_target="4s")
    ) == round(2.8 * 4.0)

    assert _max_words_per_segment(
        make_config(duration_target="4s")
    ) < _max_words_per_segment(config)


def test_a_missing_or_unparseable_duration_target_falls_back():

    assert _duration_target_seconds(make_config(duration_target="")) == 8.0
    assert _duration_target_seconds(
        make_config(duration_target="long")
    ) == 8.0

    assert _duration_target_seconds(
        make_config(duration_target="-4s")
    ) == 8.0

    # No snapgenai settings at all.
    assert _duration_target_seconds({}) == 8.0

    # A bare number and a decimal are both accepted.
    assert _duration_target_seconds(
        make_config(duration_target="10")
    ) == 10.0

    assert _duration_target_seconds(
        make_config(duration_target="6.5s")
    ) == 6.5


def test_the_committed_project_config_supplies_the_budget(project_config):

    seconds = _duration_target_seconds(project_config)

    assert seconds > 0

    assert _max_words_per_segment(project_config) == max(
        1,
        round(_words_per_second(project_config) * seconds)
    )


# --------------------------------------------------------------------
# strict word-count segmentation
# --------------------------------------------------------------------

def test_a_shorter_duration_target_yields_more_chunks():

    # duration_target sets the word budget: the same narration is cut
    # into more chunks when the clip is short than when it is long.
    short, _ = make_generator(duration_target="4s")
    long, _ = make_generator(duration_target="30s")

    short_chunks = short.build_segments(
        {"text": STORM_NARRATION}
    )
    long_chunks = long.build_segments(
        {"text": STORM_NARRATION}
    )

    assert len(short_chunks) > len(long_chunks)

    # Every chunking covers the narration exactly, whatever the clip.
    assert " ".join(short_chunks) == STORM_NARRATION
    assert " ".join(long_chunks) == STORM_NARRATION


def test_text_longer_than_the_clip_is_cut_at_word_boundaries(capsys):

    # Strict word-count grouping: words beyond the budget are cut
    # BETWEEN words - mid-sentence is intentional - and the count
    # alone decides where. Nothing is padded and nothing is flagged.
    generator, _ = make_generator(duration_target="4s")

    ceiling = _max_words_per_segment(generator.config)

    narration = (
        "The voice of one crying in the wilderness prepares the way "
        "of the Lord before him and makes his paths straight"
    )
    total = len(narration.split())
    assert total > ceiling

    segments = generator.build_segments({"text": narration})

    # Full ceiling-sized chunks plus the shorter remainder: the tell
    # is ONE sentence, so the cut lands inside it by design.
    expected = [ceiling] * (total // ceiling)
    if total % ceiling:
        expected.append(total % ceiling)

    assert [len(s.split()) for s in segments] == expected
    assert len(segments) > 1

    assert " ".join(segments) == narration
    assert [
        word
        for segment in segments
        for word in segment.split()
    ] == narration.split()

    # Word chunks never overrun, so the sentence-review flag has no
    # reason to fire - the old overlong-sentence failure is gone.
    captured = capsys.readouterr().out
    assert "Narration review needed" not in captured


def test_a_short_telling_is_kept_whole_not_padded():

    # duration_target is a ceiling, never a duration to fill: a
    # telling that already fits one clip ships as a single chunk
    # holding exactly its own words - nothing split, nothing padded.
    generator, _ = make_generator(duration_target="8s")

    narration = "Jesus rose. The wind stopped."
    segments = generator.build_segments({"text": narration})

    assert segments == [narration]
    assert len(segments[0].split()) == len(narration.split())


def test_chunks_always_cut_between_whole_words():

    generator, _ = make_generator(duration_target="4s")

    segments = generator.build_segments({"text": STORM_NARRATION})

    # A chunk is exactly its words re-spaced: non-empty, stripped,
    # never a half-word, never a dangling separator at the cut.
    for segment in segments:

        assert segment
        assert segment == segment.strip()
        assert segment == " ".join(segment.split())

    # Rejoined, the text is intact - the cut only removed a space.
    assert " ".join(segments) == STORM_NARRATION
    assert [
        word
        for segment in segments
        for word in segment.split()
    ] == STORM_NARRATION.split()

# --------------------------------------------------------------------
# the narration survives segmentation unchanged
# --------------------------------------------------------------------

def test_segmentation_preserves_every_word_in_order():

    for duration in ("4s", "8s", "12s", "30s"):

        generator, _ = make_generator(duration_target=duration)

        segments = generator.build_segments({"text": STORM_NARRATION})

        # Joined back together the narration is byte for byte the input:
        # nothing omitted, duplicated, or rewritten.
        assert " ".join(segments) == STORM_NARRATION

        # And the order is the original order.
        assert [
            word
            for segment in segments
            for word in segment.split()
        ] == STORM_NARRATION.split()


def test_segmentation_preserves_dialogue_and_punctuation():

    narration = (
        "They said to him, 'Who is this, that even the wind and the "
        "sea obey him?'"
    )

    generator, _ = make_generator(duration_target="4s")

    segments = generator.build_segments({"text": narration})

    assert " ".join(segments) == narration

    # The quoted question survives intact somewhere in the telling.
    assert "'Who is this, that even the wind and the sea obey him?'" in (
        " ".join(segments)
    )


def test_segmentation_handles_empty_and_whitespace_narration():

    generator, _ = make_generator()

    assert generator.build_segments({"text": ""}) == []

    assert generator.build_segments({"text": "   \n  "}) == []


def scripture(reference="Matthew 8:23-27"):
    """
    The scripture dict assemble_content() is handed: the reference it
    files the title under and the WEBC source text the telling came from.
    """
    return {
        "reference": reference,
        "translation": "WEBC",
        "text": (
            "Then he entered into the boat, and his disciples followed "
            "him. Behold, a violent storm came up on the lake. He was "
            "asleep. They woke him, saying, 'Save us, Lord! We are "
            "dying!' He got up and rebuked the wind and the sea, and "
            "there was a great calm."
        ),
    }


def direction_for(queries):
    """
    A model reply carrying one search_query per given string, shaped the
    way assemble_content() expects the visual direction.
    """
    return {
        "title": "Jesus Stills the Storm",
        "summary": "Jesus rebukes the wind and the sea becomes calm.",
        "mood": "reverent",
        "visuals": [
            {"search_query": query, "visual_direction": query}
            for query in queries
        ],
    }


def test_the_visual_schema_asks_for_exactly_one_visual_per_call():

    # Each focused call returns a single visual object - a search_query
    # and its visual_direction - never an array, so one call can only
    # ever direct one sentence's beat. That is what keeps the mapping
    # one-to-one; the count comes from the narration, never hardcoded.
    assert VISUAL_SCHEMA["type"] == "object"
    assert "visuals" not in VISUAL_SCHEMA["properties"]
    assert set(VISUAL_SCHEMA["required"]) == {
        "search_query",
        "visual_direction",
    }

    # The title, summary, and mood come from a separate metadata call
    # that carries no visuals at all.
    assert METADATA_SCHEMA["type"] == "object"
    assert "visuals" not in METADATA_SCHEMA["properties"]
    assert set(METADATA_SCHEMA["required"]) == {
        "title",
        "summary",
        "mood",
    }


def test_each_visual_pairs_with_its_exact_narration_sentence():

    # A punctuation- and dialogue-bearing narration: the pairing must carry
    # each sentence through byte for byte, quote and all.
    narration = STORM_NARRATION

    generator, _ = make_generator()

    segments = generator.build_segments({"text": narration})

    queries = [
        f"Style: realistic. Shot {index + 1}."
        for index in range(len(segments))
    ]

    content = generator.assemble_content(
        scripture(),
        segments,
        direction_for(queries),
        narration=narration,
    )

    # One visual per sentence, and the sentence field is that sentence
    # exactly - the text that is spoken and captioned for this clip.
    assert len(content["visuals"]) == len(segments)

    for visual, segment in zip(content["visuals"], segments):

        assert visual["sentence"] == segment

    # The quoted cry is carried through intact in its paired sentence.
    paired = " ".join(visual["sentence"] for visual in content["visuals"])

    assert "'Save us, Lord! We are dying!'" in paired


def test_narration_and_visual_counts_match_and_keep_their_order():

    narration = STORM_NARRATION

    generator, _ = make_generator()

    segments = generator.build_segments({"text": narration})

    queries = [
        f"Style: realistic. Shot {index + 1}."
        for index in range(len(segments))
    ]

    content = generator.assemble_content(
        scripture(),
        segments,
        direction_for(queries),
        narration=narration,
    )

    # The spoken telling and the visuals are two views of the same
    # sequence: same length, same order. There is no separate top-level
    # narration_sentences list - each visual carries its own sentence.
    assert content["narration"] == narration
    assert "narration_sentences" not in content
    assert len(content["visuals"]) == len(segments)

    # Order is preserved: visual N pairs with sentence N, never shuffled.
    assert [
        visual["sentence"] for visual in content["visuals"]
    ] == segments


def test_a_short_model_answer_still_yields_one_visual_per_sentence():

    # The model returned too few prompts. The mapping must not break: a
    # short answer is padded to exactly one visual per sentence, each still
    # paired with its own sentence, rather than losing clips or pairing a
    # prompt with the wrong line.
    narration = STORM_NARRATION

    generator, _ = make_generator()

    segments = generator.build_segments({"text": narration})

    content = generator.assemble_content(
        scripture(),
        segments,
        direction_for(["Style: realistic. The boat on the water."]),
        narration=narration,
    )

    assert len(content["visuals"]) == len(segments)
    assert [
        visual["sentence"] for visual in content["visuals"]
    ] == segments

    for visual in content["visuals"]:

        assert visual["search_query"]


def test_a_long_model_answer_is_trimmed_to_one_visual_per_sentence():

    # The model returned more prompts than there are sentences. Only the
    # first sentence's worth is kept, so the count can never drift above
    # the narration.
    narration = STORM_NARRATION

    generator, _ = make_generator()

    segments = generator.build_segments({"text": narration})

    too_many = [
        f"Style: realistic. Shot {index + 1}."
        for index in range(len(segments) + 3)
    ]

    content = generator.assemble_content(
        scripture(),
        segments,
        direction_for(too_many),
        narration=narration,
    )

    assert len(content["visuals"]) == len(segments)
    assert [
        visual["sentence"] for visual in content["visuals"]
    ] == segments


def test_assemble_refuses_to_fall_back_to_the_source_passage():

    # The narration is what gets spoken; the WEBC passage is only the
    # authority it was checked against. With no narration there is nothing
    # to speak, so assembly fails rather than silently reading the passage
    # aloud as if it were the telling.
    generator, _ = make_generator()

    with pytest.raises(ContentGenerationError):

        generator.assemble_content(
            scripture(),
            ["Jesus rose.", "The wind stopped."],
            direction_for(["Style: realistic. A calm sea."] * 2),
            narration="",
        )


def test_the_segment_prompt_directs_exactly_one_sentence_with_context():

    generator, _ = make_generator()

    segments = generator.build_segments({"text": STORM_NARRATION})

    # One focused prompt per segment: each marks its own sentence as the
    # one to direct, shows the narration on either side for context, and
    # never marks another sentence as the target.
    for index, segment in enumerate(segments):

        prompt = generator.build_segment_prompt(scripture(), segments, index)

        assert f"SEGMENT {index + 1}: {segment}" in prompt

        # Exactly one SEGMENT marker - the target sentence - so no other
        # sentence can be mistaken for the one to direct.
        assert prompt.count("SEGMENT ") == 1

        if index > 0:

            assert f"JUST BEFORE: {segments[index - 1]}" in prompt

        if index + 1 < len(segments):

            assert f"JUST AFTER: {segments[index + 1]}" in prompt

    assert "Return one visual object for this ONE sentence" in prompt


def test_the_prompt_forbids_combining_two_segments_into_one_visual():

    # Single-event: a prompt must depict only its own sentence's beat, and
    # must never reach into the next sentence before the narration does.
    generator, _ = make_generator()

    prompt = generator.build_segment_prompt(
        scripture(),
        generator.build_segments({"text": STORM_NARRATION}),
        0,
    )

    assert "Never combine the" in prompt
    assert "events of two segments" in prompt


def test_the_prompt_requires_pronouns_to_be_resolved():

    # Self-contained prompts: "they climb into the boat" must become the
    # characters the passage names, and identities are never guessed.
    generator, _ = make_generator()

    prompt = generator.build_segment_prompt(
        scripture(),
        generator.build_segments({"text": STORM_NARRATION}),
        0,
    )

    assert "resolve every pronoun" in prompt
    assert "never guess" in prompt.lower()


def test_the_prompt_requires_historical_setting_and_character_context():

    # The era, the people, and Jesus's treatment are all spelled out so
    # SnapGenAI does not drift into a generic modern scene.
    generator, _ = make_generator()

    prompt = generator.build_segment_prompt(
        scripture(),
        generator.build_segments({"text": STORM_NARRATION}),
        0,
    )

    # Jesus is named, never described.
    assert "Name Jesus explicitly whenever he appears" in prompt
    assert "never an unnamed stand-in" in prompt
    assert "Never describe his physical appearance" in prompt

    # When he is absent, the other people are described historically.
    assert "describe the other visible" in prompt
    assert "period-appropriate appearance, clothing" in prompt

    # The scene is pinned to the biblical era, not a modern one.
    assert "Set every prompt in the biblical era" in prompt
    assert "Never show modern clothing" in prompt


def test_the_prompt_pins_the_visual_to_this_moment_in_the_chronology():

    # A sentence about one beat must never be shown as a later or earlier
    # event from the passage: "Lord, save me!" is Peter sinking and crying
    # out, not the climbing into the boat that comes sentences later, and
    # a sentence about departing is not the walking on the water ahead.
    generator, _ = make_generator()

    prompt = generator.build_segment_prompt(
        scripture(),
        generator.build_segments({"text": STORM_NARRATION}),
        0,
    )

    assert "this exact point in the passage" in prompt
    assert "flash-forward" in prompt
    assert "flashback" in prompt
    assert "not the action it leads to" in prompt

    # Placed directly beside the task itself: an ask is the asking, never
    # the doing that follows it.
    assert "an ask is shown as the asking" in prompt
    assert "never the doing that" in prompt


def test_the_prompt_lets_the_passage_decide_unstated_locations():

    # Where the narration is silent, the passage decides: people stay
    # where its own account puts them at that point in the story. Jesus
    # is never moved to the shore when the passage keeps the people
    # together in the boat.
    generator, _ = make_generator()

    prompt = generator.build_segment_prompt(
        scripture(),
        generator.build_segments({"text": STORM_NARRATION}),
        0,
    )

    assert "the passage above decides it" in prompt
    assert "do not move Jesus to the shore" in prompt

    # Jesus stays where the passage keeps him, and the scene never moves
    # ashore or from night to dawn on its own initiative.
    assert "keep him apart for as long as the passage does" in prompt
    assert "Never move the scene onto a shore" in prompt


def test_the_narration_prompt_preserves_causal_request_response_exchanges():

    # The telling may summarise freely, but a request and the reply that
    # explains why someone then acts - "command me to come", "Come" -
    # must survive as an exchange, so the listener hears the ask before
    # the doing.
    generator, _ = make_generator()

    prompt = generator.build_narration_prompt(scripture())

    assert "asking and another answering" in prompt
    assert "hears the ask before the doing" in prompt

    # The condition and the agents survive with the exchange: the ask's
    # "if it is you" travels with the request, and the passage's own
    # doer of an action is never handed to another character.
    assert "travels with the request" in prompt
    assert "doer of it" in prompt


def test_write_narration_routes_through_the_narration_schema():
    """
    write_narration calls the LLM with NARRATION_SCHEMA; the stub must
    route that call to its narration reply rather than treat it as a
    visual call - a misroute would silently return a search_query as
    narration.
    """
    generator, stub = make_generator(
        llm=StubLLM(narration="Jesus walked on the water.")
    )

    narration = generator.write_narration(scripture())

    assert "Jesus walked on the water" in narration
    assert not getattr(stub, "visual_calls", 0)


def test_generate_makes_one_focused_call_per_sentence():
    """
    generate() must ask the model once per narration sentence, and each
    call's prompt must carry the correct segment plus its before/after
    neighbours - never a single all-at-once call over the whole story.
    """
    generator, stub = make_generator()

    segments = generator.build_segments({"text": STORM_NARRATION})

    content = generator.generate(
        scripture(),
        segments,
        narration=STORM_NARRATION,
    )

    # One focused visual call per sentence, plus one metadata call for
    # title/summary/mood.
    assert len(stub.visual_prompts) == len(segments)
    assert len(stub.metadata_prompts) == 1

    assert len(segments) > 1

    for index, prompt in enumerate(stub.visual_prompts):
        # This call is directed at this segment specifically.
        assert f"SEGMENT {index + 1}: {segments[index]}" in prompt
        assert prompt.count("SEGMENT ") == 1

        # Its immediate neighbours are present for context.
        if index > 0:
            assert f"JUST BEFORE: {segments[index - 1]}" in prompt
        if index + 1 < len(segments):
            assert f"JUST AFTER: {segments[index + 1]}" in prompt

    # The assembled content pairs each visual with its own sentence, in
    # order, and carries no redundant narration_sentences field.
    assert [visual["sentence"] for visual in content["visuals"]] == segments
    assert "narration_sentences" not in content


# --------------------------------------------------------------------
# generate() and write_narration() check their own work and re-roll
# what fails, so a fresh regeneration ships checked output rather
# than whatever the first sample happened to say.
# --------------------------------------------------------------------

def test_every_visual_is_checked_against_its_sentence():
    """
    One verification call per sentence, each judging that sentence
    against the passage - the check is part of generation, not an
    optional extra step.
    """
    generator, stub = make_generator()

    segments = generator.build_segments({"text": STORM_NARRATION})

    generator.generate(
        scripture(),
        segments,
        narration=STORM_NARRATION,
    )

    assert len(stub.verification_prompts) == len(segments)
    assert len(stub.verification_prompts) == len(stub.visual_prompts)

    for index, prompt in enumerate(stub.verification_prompts):
        # Each check judges its own sentence, with the passage as the
        # authority and a structured verdict demanded.
        assert segments[index] in prompt
        assert scripture()["text"] in prompt
        assert '"aligned"' in prompt


def test_the_visual_check_prompt_carries_query_and_neighbours():
    # A short clip -> more chunks, so the middle segment has both
    # neighbours to carry.
    generator, _ = make_generator(duration_target="4s")

    segments = generator.build_segments({"text": STORM_NARRATION})

    prompt = generator.build_visual_verification_prompt(
        scripture(),
        segments,
        1,
        "Style: realistic. Wide shot of the boat in the storm.",
    )

    assert segments[1] in prompt
    assert segments[0] in prompt
    assert segments[2] in prompt
    assert (
        "Style: realistic. Wide shot of the boat in the storm."
        in prompt
    )

    # The judge is calibrated to the generator's own contract:
    # passage-supported context is legitimate context, not invention,
    # and style is never grounds for rejection - otherwise every good
    # query would be re-rolled for a nitpick.
    assert "legitimate context" in prompt
    assert "WHAT IS NOT A FAILURE" in prompt


def test_a_rejected_visual_is_re_rolled_with_the_rejection_reason():
    """
    Fail once, pass on the second attempt: the segment must be
    generated twice, and the second prompt must carry the checker's
    rejection so the re-roll fixes that specific problem. The kept
    visual is the re-roll, not the first sample.
    """
    narration = "Jesus stilled the storm."
    generator, stub = make_generator(
        llm=StubLLM(verifications=[False, True])
    )

    segments = generator.build_segments({"text": narration})

    content = generator.generate(
        scripture(),
        segments,
        narration=narration,
    )

    assert len(segments) == 1
    assert stub.visual_calls == 2
    assert len(stub.verification_prompts) == 2

    assert "YOUR PREVIOUS ANSWER WAS REJECTED" in stub.visual_prompts[1]
    assert "The query shows the wrong beat." in stub.visual_prompts[1]

    # The second attempt's query is what shipped.
    assert "Shot 2." in content["visuals"][0]["search_query"]


def test_the_re_roll_loop_runs_until_the_checker_approves():
    """
    A rejected visual is re-rolled up to verification_attempts times;
    when the checker never approves, the last draft ships instead of
    looping forever. Seven pre-configured rejections are far more than
    the cap of three, so only three calls are made and the third draft
    ships.
    """
    narration = "Jesus stilled the storm."
    generator, stub = make_generator(
        llm=StubLLM(verifications=[False] * 7 + [True])
    )

    segments = generator.build_segments({"text": narration})

    content = generator.generate(
        scripture(),
        segments,
        narration=narration,
    )

    # Cap stops the loop at three attempts; the third (rejected) draft
    # ships instead of an eighth approved call.
    assert stub.visual_calls == 3
    assert len(stub.verification_prompts) == 3

    # Every re-roll carried the rejection forward.
    assert all(
        "YOUR PREVIOUS ANSWER WAS REJECTED" in prompt
        for prompt in stub.visual_prompts[1:]
    )
    assert len(content["visuals"]) == 1
    assert "Shot 3." in content["visuals"][0]["search_query"]


def test_verification_can_be_switched_off():
    generator, stub = make_generator(
        llm=StubLLM(),
        verify_visuals=False,
        verify_narration=False,
    )

    segments = generator.build_segments({"text": STORM_NARRATION})

    generator.generate(
        scripture(),
        segments,
        narration=STORM_NARRATION,
    )
    generator.write_narration(scripture())

    assert not stub.verification_prompts
    assert not stub.narration_verification_prompts
    assert stub.visual_calls == len(segments)
    assert len(stub.narration_prompts) == 1


def test_a_rejected_narration_is_re_written_with_the_issues():
    """
    The narration gets the same treatment as a visual: rejected once,
    re-written with the checker's issues, accepted on the second
    attempt - and the re-write prompt names the problems to fix.
    """
    generator, stub = make_generator(
        llm=StubLLM(
            narration="Peter answered, calling out to Jesus.",
            narration_verifications=[False, True],
        )
    )

    narration = generator.write_narration(scripture())

    assert len(stub.narration_prompts) == 2
    assert len(stub.narration_verification_prompts) == 2

    rejected = stub.narration_prompts[1]
    assert "YOUR PREVIOUS NARRATION WAS REJECTED" in rejected
    assert "wrong person" in rejected

    assert "Peter answered" in narration


def test_the_narration_check_judges_against_the_passage():
    generator, _ = make_generator()

    prompt = generator.build_narration_verification_prompt(
        scripture(),
        "Jesus stilled the storm.",
    )

    assert scripture()["text"] in prompt
    assert "Jesus stilled the storm." in prompt
    assert '"accurate"' in prompt
    # The two failure classes this channel actually hit are judged.
    assert "Agents" in prompt
    assert "Conditions" in prompt


def test_a_verdict_we_cannot_parse_counts_as_a_rejection():
    """
    Fail closed: a reply that is not the verdict object can never read
    as approval, no matter how approving it sounds.
    """
    generator, _ = make_generator(
        llm=StubLLM(verification_reply="Looks good to me.")
    )

    aligned, reason = generator.verify_visual(
        scripture(),
        ["Jesus stilled the storm."],
        0,
        "Style: realistic. Jesus calms the sea.",
    )

    assert aligned is False
    assert reason


def test_chunks_fill_the_ceiling_before_starting_the_next():
    """
    Strict deterministic word counting: every chunk except the last
    holds exactly the ceiling; only the final chunk may be shorter.
    No sentence structure decides anything - the word count alone
    does - and coverage stays exact.
    """
    generator, _ = make_generator()

    ceiling = _max_words_per_segment(generator.config)
    total = len(STORM_NARRATION.split())
    assert total > ceiling

    segments = generator.build_segments({"text": STORM_NARRATION})

    expected = [ceiling] * (total // ceiling)
    if total % ceiling:
        expected.append(total % ceiling)

    assert [len(s.split()) for s in segments] == expected

    # Complete coverage: the chunks ARE the narration, with nothing
    # omitted, duplicated, or rewritten.
    assert " ".join(segments) == STORM_NARRATION


def test_no_chunk_exceeds_the_clip_ceiling():
    """
    No chunk may hold more words than the clip's duration_target
    ceiling. Every chunk stays within it and every word survives in
    order.
    """
    generator, _ = make_generator()

    segments = generator.build_segments({"text": STORM_NARRATION})

    ceiling = _max_words_per_segment(generator.config)

    assert all(
        len(segment.split()) <= ceiling
        for segment in segments
    )
    assert " ".join(segments) == STORM_NARRATION


def test_chunks_keep_every_word_in_order():
    """
    Joining the chunks with single spaces reproduces the narration
    exactly - no word omitted, duplicated, or reordered.
    """
    generator, _ = make_generator()

    for narration in (
        STORM_NARRATION,
        EPISODE_001_NARRATION,
        MARK_002_NARRATION,
        "Jesus went up onto the mountain by himself to pray and stayed "
        "there all night. His disciples crossed the sea to meet him at "
        "dawn.",
    ):

        segments = generator.build_segments({"text": narration})

        assert " ".join(segments) == narration
        assert [
            word
            for segment in segments
            for word in segment.split()
        ] == narration.split()


def test_cuts_can_fall_inside_sentences_and_quotations():
    """
    Mid-sentence and mid-quotation cuts are intentional: grouping is
    strictly by word count. A cut crosses a real sentence boundary,
    and a quotation longer than the ceiling necessarily spans chunks
    - neither may be protected from the cut.
    """
    generator, _ = make_generator(duration_target="4s")

    segments = generator.build_segments({"text": STORM_NARRATION})

    # STORM's opening sentence is longer than this clip's budget, so
    # at least one chunk is NOT a whole number of complete sentences -
    # proof the sentence boundary does not shield text from a cut.
    sentences = split_sentences(
        STORM_NARRATION,
        _protected_spans(STORM_NARRATION),
    )
    sentence_runs = {
        " ".join(sentences[start:end])
        for start in range(len(sentences))
        for end in range(start + 1, len(sentences) + 1)
    }

    assert any(
        segment not in sentence_runs
        for segment in segments
    )

    # A quotation longer than the ceiling spans chunk boundaries: no
    # single chunk can hold it whole, and coverage still holds.
    generator, _ = make_generator()
    ceiling = _max_words_per_segment(generator.config)

    long_quotation = (
        "'After me comes he who is mightier than I, the strap of whose "
        "sandals I am not worthy to stoop down and loosen. I baptized "
        "you in water, but he will baptize you in the Holy Spirit.'"
    )
    assert len(long_quotation.split()) > ceiling

    segments = generator.build_segments({"text": MARK_002_NARRATION})

    assert all(
        long_quotation not in segment
        for segment in segments
    )
    assert long_quotation in MARK_002_NARRATION
    assert " ".join(segments) == MARK_002_NARRATION


# --------------------------------------------------------------------
# build_word_segments: strict word-count chunking (unit tests)
# --------------------------------------------------------------------


def test_build_word_segments_fills_the_ceiling_then_continues():
    """
    Strict word counting: each chunk holds exactly max_words words
    until the remainder, which may be shorter.
    """
    chunks = build_word_segments("One two three four five six seven.", 3)

    assert chunks == [
        "One two three",
        "four five six",
        "seven.",
    ]


def test_build_word_segments_never_exceeds_the_ceiling():
    """
    No chunk's word count may exceed max_words - the count is the
    only rule.
    """
    text = "Alpha beta gamma delta epsilon zeta eta theta."

    chunks = build_word_segments(text, 5)

    assert all(
        len(chunk.split()) <= 5
        for chunk in chunks
    )
    assert " ".join(chunks) == text


def test_build_word_segments_short_text_is_one_chunk():
    """
    The ceiling is a maximum, never a quota: a text that fits ships
    as a single chunk, unpadded.
    """
    assert build_word_segments("Short and whole.", 100) == [
        "Short and whole."
    ]


def test_build_word_segments_preserves_all_text_in_order():
    """
    Joining every chunk with single spaces reproduces the input
    exactly - no omission, duplication, or reordering - even when a
    cut lands inside a sentence or quotation.
    """
    text = (
        "'Save us, Lord! We are dying!' he cried, and the sea "
        "grew calm before them."
    )

    chunks = build_word_segments(text, 4)

    assert " ".join(chunks) == text
    assert [
        word
        for chunk in chunks
        for word in chunk.split()
    ] == text.split()


def test_build_word_segments_empty_input():
    assert build_word_segments("", 10) == []
    assert build_word_segments("   \n  ", 10) == []


def test_build_word_segments_single_chunk_at_the_ceiling():
    assert build_word_segments("One two three four.", 4) == [
        "One two three four."
    ]


def test_the_grouping_prompt_names_the_units_the_ceiling_and_the_feedback():
    generator, _ = make_generator()
    units = split_sentences(
        STORM_NARRATION,
        _protected_spans(STORM_NARRATION),
    )

    prompt = generator.build_grouping_prompt(
        {
            "text": STORM_NARRATION,
            "reference": "Mark 4:35-41",
            "translation": "WEBC",
        },
        units,
        22,
        feedback="missing unit(s) [2].",
    )

    assert "YOUR PREVIOUS GROUPING WAS REJECTED" in prompt
    assert "missing unit(s) [2]" in prompt
    assert f"1. {units[0]}" in prompt
    assert f"1 to {len(units)} must appear EXACTLY ONCE" in prompt
    assert "MERGE FIRST" in prompt
    assert "22 words or fewer" in prompt
    assert STORM_NARRATION in prompt


def test_the_real_episode_units_and_their_fitting_pairs():
    """
    Regression for the shipped episode: its 15 narration units and the
    exact set of consecutive pairs whose combined spoken length fits
    the duration-derived ceiling - the mountain prayer + evening
    (17 words), Peter's request + "Come!" (19 words), the short
    "Come!" + stepping out (16 words), and the sinking + "Lord, save
    me!" (22 words). Every other boundary legally exceeds the clip.
    """
    config = make_config()
    max_words = _max_words_per_segment(config)
    units = split_sentences(
        EPISODE_001_NARRATION,
        _protected_spans(EPISODE_001_NARRATION),
    )

    assert len(units) == 15

    fitting = {
        (position, position + 1): (
            len(units[position - 1].split())
            + len(units[position].split())
        )
        for position in range(1, len(units))
        if (
            len(units[position - 1].split())
            + len(units[position].split())
            <= max_words
        )
    }

    assert fitting == {
        (2, 3): 17,
        (8, 9): 19,
        (9, 10): 16,
        (11, 12): 22,
    }


def test_the_grouping_prompt_lists_every_pair_that_fits_one_clip():
    """
    The prompt hands the model the code-computed candidates: every
    pair that fits one clip is listed for an explicit decision, and a
    pair that would overrun the ceiling is never offered.
    """
    generator, _ = make_generator()
    config = make_config()
    max_words = _max_words_per_segment(config)
    units = split_sentences(
        EPISODE_001_NARRATION,
        _protected_spans(EPISODE_001_NARRATION),
    )

    prompt = generator.build_grouping_prompt(
        {"text": EPISODE_001_NARRATION},
        units,
        max_words,
    )

    assert "PAIRS THAT FIT ONE CLIP" in prompt
    assert "units 2 + 3 (17 words total)" in prompt
    assert "units 8 + 9 (19 words total)" in prompt
    assert "units 11 + 12 (22 words total)" in prompt
    # Unit 4 (22 words) + unit 5 (15 words) overrun the clip, so the
    # ceiling - not the model - rules that pair out.
    assert "units 4 + 5" not in prompt


def test_matthew_14_chunks_respect_the_ceiling():
    """
    Matthew 14:22-33 corpus (episode 001's narration): strict word
    chunks respect the duration-derived ceiling, fill it exactly
    before starting the next chunk, and cover every word of the
    narration once, in order.
    """
    generator, _ = make_generator()

    ceiling = _max_words_per_segment(generator.config)
    total = len(EPISODE_001_NARRATION.split())
    assert total > ceiling

    segments = generator.build_segments(
        {"text": EPISODE_001_NARRATION}
    )

    expected = [ceiling] * (total // ceiling)
    if total % ceiling:
        expected.append(total % ceiling)

    assert [len(s.split()) for s in segments] == expected
    assert all(
        len(segment.split()) <= ceiling
        for segment in segments
    )
    assert " ".join(segments) == EPISODE_001_NARRATION
    assert [
        word
        for segment in segments
        for word in segment.split()
    ] == EPISODE_001_NARRATION.split()


def test_the_visual_checker_covers_the_shipped_crowd_boarding_failure():
    """
    Episode 001 shipped a visual of the crowd climbing into the boat
    for the sentence that sends the crowd away. The judge prompt for
    that exact sentence must carry the shipped query and the
    sharpened criteria: an action assigned to present bystanders is
    still an invention, and entering is resolved against the
    passage's own positions.
    """
    generator, _ = make_generator()
    segments = list(EPISODE_001_SHIPPED_SEGMENTS)

    prompt = generator.build_visual_verification_prompt(
        scripture(reference="Matthew 14:22-33"),
        segments,
        0,
        EPISODE_001_BAD_CROWD_BOARDING,
    )

    assert EPISODE_001_BAD_CROWD_BOARDING in prompt
    assert "sends that crowd away" in prompt
    assert "never the same people shown climbing in" in prompt


def test_the_visual_checker_treats_an_unnamed_jesus_as_a_failure():
    """
    Episode 001 shipped "a solitary figure" where the passage has
    Jesus praying on the mountain. The judge must treat a named
    person reduced to an unnamed stand-in as a failure, and the
    director prompt must ban the stand-in wording so a re-roll can
    comply.
    """
    generator, _ = make_generator()
    segments = list(EPISODE_001_SHIPPED_SEGMENTS)

    checker = generator.build_visual_verification_prompt(
        scripture(reference="Matthew 14:22-33"),
        segments,
        1,
        EPISODE_001_BAD_UNNAMED_JESUS,
    )
    director = generator.build_segment_prompt(
        scripture(reference="Matthew 14:22-33"),
        segments,
        1,
    )

    assert "'a lone figure', 'a solitary figure'" in checker
    assert "is a failure, even when the scene makes the identity" in (
        checker
    )
    assert "'a lone figure', 'a solitary figure'" in director
    assert "Naming him is not describing him" in director


@pytest.mark.parametrize(
    ("index", "bad_query", "fixed_query"),
    [
        (
            0,
            EPISODE_001_BAD_CROWD_BOARDING,
            "Style: realistic. The crowd disperses along the shore "
            "while the disciples ready the boat to leave without "
            "him. Medium shot, the path behind them filling with "
            "people walking away as the disciples turn to the sea.",
        ),
        (
            10,
            EPISODE_001_BAD_SHORE_BOARDING,
            "Style: realistic. Jesus and Peter climb up into the "
            "small wooden boat in the middle of the dark sea as the "
            "waves settle around the disciples already aboard. Wide "
            "shot, no shore anywhere in frame.",
        ),
    ],
)
def test_a_shipped_bad_visual_is_re_rolled_with_the_checkers_reason(
    index,
    bad_query,
    fixed_query
):
    """
    Both visuals episode 001 shipped against the passage, replayed
    through the live loop on the shipped 12 segments: the checker
    rejects the shipped query, the rejection rides the re-roll
    prompt, and the corrected query is what ships for that sentence.
    """
    # The verdicts pop in call order, so the script must approve every
    # segment before the bad one before rejecting that one's first try.
    generator, stub = make_generator(
        llm=StubLLM(
            queries=[
                f"Style: realistic. Shot {position + 1}."
                for position in range(
                    len(EPISODE_001_SHIPPED_SEGMENTS) + 1
                )
            ],
            verifications=[True] * index + [False, True],
        )
    )

    stub.queries[index] = bad_query
    stub.queries[index + 1] = fixed_query

    segments = list(EPISODE_001_SHIPPED_SEGMENTS)
    content = generator.generate(
        scripture(reference="Matthew 14:22-33"),
        segments,
        narration=EPISODE_001_REGENERATED_NARRATION,
    )

    assert len(content["visuals"]) == len(segments)
    assert stub.visual_calls == len(segments) + 1
    assert len(stub.verification_prompts) == len(segments) + 1

    # The re-roll carries the checker's rejection forward.
    re_roll_prompt = stub.visual_prompts[index + 1]
    assert "YOUR PREVIOUS ANSWER WAS REJECTED" in re_roll_prompt
    assert "The query shows the wrong beat." in re_roll_prompt

    # The corrected query is what shipped for that sentence.
    assert content["visuals"][index]["search_query"] == fixed_query
    assert bad_query not in content["visuals"][index]["search_query"]


def test_cap_group_words_rejects_oversized_group_never_truncates():
    """
    _cap_group_words is fail-closed: a group whose units would speak
    longer than one clip is rejected outright - never truncated or cut
    to fit. A compliant partition keeps every segment within the
    ceiling while covering every word.
    """
    narration = (
        "Jesus went up onto the mountain by himself to pray and stayed "
        "there all night. His disciples crossed the sea to meet him at "
        "dawn."
    )
    generator, _ = make_generator()

    ceiling = _max_words_per_segment(generator.config)
    units = split_sentences(
        narration,
        _protected_spans(narration),
    )

    # 15 + 10 = 25 words, over the 22-word ceiling: rejected outright.
    capped = ContentGenerator._cap_group_words(
        [[1, 2]],
        units,
        ceiling,
    )
    assert capped is None

    # Re-split respects the ceiling: 15 words and 10 words.
    capped = ContentGenerator._cap_group_words(
        [[1], [2]],
        units,
        ceiling,
    )
    assert capped == [[1], [2]]

    segments = [" ".join(units[i - 1] for i in group) for group in capped]
    assert all(len(segment.split()) <= ceiling for segment in segments)
    assert " ".join(segments) == narration


def test_cap_group_words_merges_small_fragments():
    """
    _cap_group_words merges adjacent small groups whose combined word
    count fits the ceiling, so a tiny leftover fragment is absorbed by
    its neighbor instead of shipping as its own visual.
    """
    generator, _ = make_generator()

    units = [
        "One short sentence here.",
        "Another short one.",
        "A third short sentence to round things out nicely.",
    ]
    ceiling = 20

    # [1] + [2] = 9 words, fits; then [1,2] + [3] = 19 words also fits,
    # so greedy merging absorbs all three into one group.
    capped = ContentGenerator._cap_group_words(
        [[1], [2], [3]],
        units,
        ceiling,
    )

    assert capped == [[1, 2, 3]]
    segments = [" ".join(units[i - 1] for i in group) for group in capped]
    assert " ".join(segments) == " ".join(units)


def test_small_fragment_groups_are_merged_when_they_fit():
    """
    _cap_group_words merges adjacent small groups whose combined word
    count fits the ceiling, so a tiny leftover fragment is absorbed by
    its neighbor instead of shipping as its own visual. A group that
    overruns the ceiling is rejected outright - never truncated.
    Coverage stays exact: every unit appears once, in order.
    """
    generator, _ = make_generator()
    max_words = _max_words_per_segment(generator.config)

    u_long = " ".join(["word"] * 20)
    u_tiny = "word"
    units = [u_long, u_tiny, " ".join(["word"] * 20)]

    merged = generator._cap_group_words([[1], [2], [3]], units, max_words)

    # The tiny fragment merges with its neighbor; the long unit stays
    # separate because 20 + 20 would overrun the ceiling.
    assert merged == [[1, 2], [3]]

    # An over-ceiling group is rejected outright, never truncated.
    assert generator._cap_group_words([[1, 2, 3]], units, max_words) is None

    # Exact coverage: every unit appears once, in ascending order.
    flat = [item for group in merged for item in group]
    assert flat == [1, 2, 3]


def test_a_long_narration_splits_evenly_across_chunks(capsys):
    """
    A narration longer than one clip is cut at the word boundary the
    budget allows - even when that leaves a tiny final chunk. The
    narration stays word-for-word intact and no sentence-review flag
    fires: the count can never overrun the ceiling.
    """
    narration = (
        "Jesus went up onto the mountain by himself to pray and he "
        "remained there alone all through the night while the disciples "
        "struggled"
    )
    generator, _ = make_generator()

    ceiling = _max_words_per_segment(generator.config)
    total = len(narration.split())
    assert total > ceiling

    segments = generator.build_segments({"text": narration})

    assert [len(s.split()) for s in segments] == [
        ceiling,
        total - ceiling,
    ]
    assert " ".join(segments) == narration
    assert [
        word
        for segment in segments
        for word in segment.split()
    ] == narration.split()

    captured = capsys.readouterr().out
    assert "Narration review needed" not in captured


def test_a_cut_can_fall_inside_a_quotation():
    """
    Mid-quotation cuts are intentional: the word count decides, not
    the quotation marks. With a five-word budget the first chunk ends
    inside the saying - the saying spans two chunks - while coverage
    and order stay exactly intact.
    """
    narration = 'Jesus said, "Come to me now." Peter stepped out.'
    generator, _ = make_generator(
        words_per_second=1,
        duration_target="5s",
    )

    ceiling = _max_words_per_segment(generator.config)

    segments = generator.build_segments({"text": narration})

    assert [len(s.split()) for s in segments] == (
        [ceiling] * (len(narration.split()) // ceiling)
        + (
            [len(narration.split()) % ceiling]
            if len(narration.split()) % ceiling
            else []
        )
    )

    # The saying is not whole in any single chunk - it was cut.
    assert all(
        '"Come to me."' not in segment
        for segment in segments
    )

    assert " ".join(segments) == narration
    assert all(
        len(segment.split()) <= ceiling
        for segment in segments
    )


def test_no_chunk_exceeds_the_ceiling_with_exact_coverage():
    """
    Every chunk stays within the clip's spoken ceiling, fills it
    before the next chunk begins, and the chunks cover the whole
    narration exactly once in order - nothing omitted, duplicated,
    or rewritten.
    """
    narration = (
        "The beginning of the gospel of Jesus Christ. John came "
        "baptizing in the wilderness. Jesus came from Nazareth and was "
        "baptized by John in the Jordan. The heavens opened and the "
        "Spirit descended like a dove. A voice came from the sky "
        'saying, "You are my beloved Son."'
    )
    generator, _ = make_generator()

    segments = generator.build_segments({"text": narration})

    ceiling = _max_words_per_segment(generator.config)
    total = len(narration.split())

    assert all(
        len(segment.split()) <= ceiling for segment in segments
    )

    expected = [ceiling] * (total // ceiling)
    if total % ceiling:
        expected.append(total % ceiling)

    assert [len(s.split()) for s in segments] == expected
    assert " ".join(segments) == narration


def test_genealogy_and_ordinary_narration_stay_within_the_ceiling():
    """
    Sanity for a list-like (genealogy) narration and an ordinary
    narrative: every segment fits the ceiling and coverage is exact.
    """
    generator, _ = make_generator()

    genealogy = (
        "Abraham was the father of Isaac. Isaac was the father of "
        "Jacob. Jacob was the father of Judah and his brothers. Judah "
        "was the father of Perez and Zerah."
    )
    ordinary = (
        "Jesus sent the crowds away and told his disciples to get into "
        "a boat and go ahead of him. He went up onto a mountain by "
        "himself to pray."
    )

    for narration in (genealogy, ordinary):

        segments = generator.build_segments(
            {"text": narration
            }
        )

        ceiling = _max_words_per_segment(generator.config)

        assert all(
            len(segment.split()) <= ceiling for segment in segments
        )
        assert " ".join(segments) == narration


def test_matthew_14_ordinary_narrative_alignment():
    """
    Regression for Matthew 14:22-33: ordinary narrative events align
    one visual per duration-grouped segment, with Jesus named
    explicitly and no action displaced to the wrong moment.
    """
    generator, _ = make_generator()

    narration = (
        "Jesus sent the crowds away and told his disciples to get into "
        "a boat and go ahead of him. He went up onto a mountain by "
        "himself to pray. When evening came, he was alone there. But "
        "the boat was now in the middle of the sea, struggling against "
        "the waves as the wind blew strongly against them. In the "
        "fourth watch of the night, Jesus came walking on the water "
        "toward them. When the disciples saw him, they were terrified "
        "and cried out, thinking it was a ghost. Immediately Jesus "
        "spoke to them, saying, 'Cheer up! It is I! Don't be afraid.' "
        "Peter answered, 'Lord, if it is you, command me to come to "
        "you on the water.' He said, 'Come!' Peter stepped down from "
        "the boat and walked on the water toward Jesus. But when he "
        "saw how strong the wind was, he became afraid and began to "
        "sink. He cried out, 'Lord, save me!' Immediately Jesus "
        "reached out his hand, took hold of him, and said, 'You of "
        "little faith, why did you doubt?' When they got into the "
        "boat, the wind ceased. Those in the boat came and worshiped "
        "him, saying, 'You are truly the Son of God!'"
    )

    segments = generator.build_segments(
        {"text": narration}
    )

    ceiling = _max_words_per_segment(generator.config)

    assert all(
        len(segment.split()) <= ceiling for segment in segments
    )
    assert " ".join(segments) == narration

    # No segment reduces Jesus to an unnamed stand-in such as "a lone
    # figure" or "a solitary figure" wherever the passage names him.
    for segment in segments:

        assert "a lone figure" not in segment, (
            f"Unnamed stand-in for Jesus: {segment!r}"
        )
        assert "a solitary figure" not in segment, (
            f"Unnamed stand-in for Jesus: {segment!r}"
        )


def test_one_visual_per_duration_group():
    """
    Exactly one visual prompt is generated per narration group: the
    number of visuals equals the number of segments, and each visual's
    sentence matches its segment exactly, in order.
    """
    generator, stub = make_generator()

    narration = (
        "Jesus sent the crowds away and told his disciples to get into "
        "a boat and go ahead of him. He went up onto a mountain by "
        "himself to pray."
    )

    segments = generator.build_segments(
        {"text": narration}
    )

    content = generator.generate(
        scripture(reference="Matthew 14:22-33"),
        segments,
        narration=narration,
    )

    assert len(content["visuals"]) == len(segments)

    for index, segment in enumerate(segments):

        assert content["visuals"][index]["sentence"] == segment


# --------------------------------------------------------------------
# word-count segmentation: duration limits, coverage, reroll safety
# --------------------------------------------------------------------

# A mix of short sentences and one sentence longer than a small clip,
# so both sides of the rule are exercised: ceiling-filling word cuts,
# and a long sentence divided at word boundaries rather than kept
# whole.
MIXED_NARRATION = (
    "Jesus rose. The wind stopped. The voice of one crying in the "
    "wilderness prepares the way of the Lord before him and makes his "
    "paths straight. The disciples were amazed."
)


def test_chunks_cover_exact_words_at_any_duration():
    """
    At every configured clip length, chunks reproduce the narration
    word for word in order, hold at most the duration-derived budget,
    and fill it exactly before the next chunk begins - sentence and
    quotation boundaries change nothing.
    """
    for duration in ("4s", "8s", "12s", "30s"):

        generator, _ = make_generator(duration_target=duration)

        ceiling = _max_words_per_segment(generator.config)
        total = len(MIXED_NARRATION.split())

        segments = generator.build_segments({"text": MIXED_NARRATION})

        # Exact coverage and ordering, word for word.
        assert " ".join(segments) == MIXED_NARRATION
        assert [
            word
            for segment in segments
            for word in segment.split()
        ] == MIXED_NARRATION.split()

        # Strict ceiling: within budget, filling it exactly.
        expected = [ceiling] * (total // ceiling)
        if total % ceiling:
            expected.append(total % ceiling)

        assert [len(s.split()) for s in segments] == expected
        assert all(
            len(segment.split()) <= ceiling
            for segment in segments
        )


def test_duration_limits_fill_exactly_without_padding(capsys):
    """
    The ceiling (duration_target x words_per_second) is what ends
    every chunk: each chunk fills the ceiling before the next begins,
    the last chunk holds only the remainder, and the chunks hold
    exactly the narration's own words - nothing padded, nothing
    flagged - even though the fixture contains a sentence longer than
    one clip.
    """
    generator, _ = make_generator(
        words_per_second=2,
        duration_target="8s",
    )

    ceiling = _max_words_per_segment(generator.config)
    total = len(MIXED_NARRATION.split())

    segments = generator.build_segments({"text": MIXED_NARRATION})

    # The fixture really does contain a sentence longer than one clip.
    assert any(
        len(sentence.split()) > ceiling
        for sentence in split_sentences(
            MIXED_NARRATION,
            _protected_spans(MIXED_NARRATION),
        )
    )

    # Strict filling: full ceiling-sized chunks plus the remainder.
    expected = [ceiling] * (total // ceiling)
    if total % ceiling:
        expected.append(total % ceiling)

    assert [len(s.split()) for s in segments] == expected

    # No padding and no over-ceiling chunk: the long sentence was
    # divided at word boundaries, not kept whole or flagged.
    assert sum(
        len(segment.split())
        for segment in segments
    ) == total
    assert " ".join(segments) == MIXED_NARRATION
    assert all(
        len(segment.split()) <= ceiling
        for segment in segments
    )

    captured = capsys.readouterr().out
    assert "Narration review needed" not in captured


def test_visual_rerolls_preserve_finalized_chunks():
    """
    Segmentation is finalized before visual generation: a verification
    reroll repairs only the visual prompt. The finalized chunks come
    back byte for byte, the narration is untouched, every chunk still
    gets exactly one visual, and each reroll prompt still carries the
    finalized chunks' exact text - never a regrouped one.
    """
    generator, stub = make_generator(
        llm=StubLLM(
            # One rejection then approval per segment: rerolls happen
            # for every chunk.
            verifications=[False, True, False, True],
        )
    )

    segments = generator.build_segments(
        {"text": STORM_NARRATION}
    )

    assert len(segments) >= 2

    finalized = list(segments)

    content = generator.generate(
        scripture(reference="Mark 4:35-41"),
        segments,
        narration=STORM_NARRATION,
    )

    # The finalized chunks are unchanged - same text, same order, same
    # count - and still exactly the deterministic word-count result.
    ceiling = _max_words_per_segment(generator.config)
    assert segments == finalized
    assert segments == build_word_segments(STORM_NARRATION, ceiling)

    # The narration shipped verbatim; nothing was rewritten by the
    # visual checks.
    assert content["narration"] == STORM_NARRATION

    # Exactly one visual per finalized group, each paired with its own
    # segment text, in order.
    assert len(content["visuals"]) == len(finalized)

    for index, segment in enumerate(finalized):

        assert content["visuals"][index]["sentence"] == segment

    # Two calls per segment: a rejection followed by the re-roll.
    assert len(stub.visual_prompts) == 2 * len(finalized)
    assert len(stub.verification_prompts) == 2 * len(finalized)

    for first_attempt in range(0, len(stub.visual_prompts), 2):

        reroll = stub.visual_prompts[first_attempt + 1]

        assert "YOUR PREVIOUS ANSWER WAS REJECTED" in reroll

        # The reroll still directs the SAME finalized segments, with
        # their neighbors from the finalized grouping - never a
        # different or regrouped unit of narration.
        for segment in finalized:

            assert segment in reroll


# --------------------------------------------------------------------
# regression: episode 002 (Mark 1:1-11) quote-boundary collapse
# --------------------------------------------------------------------

# Verbatim from media/output/shorts/002/content.json, kept as the
# regression case: this narration quotes two levels with the same
# straight mark ('Behold ... 'Make ready ... straight!'), which the
# old quote scanner misread - the inner opener closed the outer saying
# and the real closer opened a runaway span that swallowed six
# complete sentences, shipping blobs of 37/91/69 words where the
# ceiling allows about 22.
MARK_002_NARRATION = (
    "The beginning of the Good News of Jesus Christ, the Son of God. As "
    "it is written in the prophets, 'Behold, I send my messenger before "
    "your face, who will prepare your way before you: the voice of one "
    "crying in the wilderness, 'Make ready the way of the Lord! Make his "
    "paths straight!' John came baptizing in the wilderness and preaching "
    "the baptism of repentance for forgiveness of sins. All the country "
    "of Judea and all those of Jerusalem went out to him. They were "
    "baptized by him in the Jordan river, confessing their sins. John was "
    "clothed with camel\u2019s hair and a leather belt around his waist. He "
    "ate locusts and wild honey. He preached, saying, 'After me comes he "
    "who is mightier than I, the strap of whose sandals I am not worthy "
    "to stoop down and loosen. I baptized you in water, but he will "
    "baptize you in the Holy Spirit.' In those days, Jesus came from "
    "Nazareth of Galilee, and was baptized by John in the Jordan. "
    "Immediately coming up from the water, he saw the heavens parting and "
    "the Spirit descending on him like a dove. A voice came out of the "
    "sky, 'You are my beloved Son, in whom I am well pleased.'"
)


def test_nested_quotation_marks_keep_sentence_boundaries():
    """
    Two quotation levels quoted with the same straight mark (the Mark 1
    pattern): the inner opener must not close the outer saying, and the
    outer closer must not open a runaway span over the sentences that
    follow. The saying stays whole as one sentence; the next sentence
    splits out at its own period.
    """
    narration = (
        "As it is written, 'Behold, I send my messenger before your "
        "face; the voice of one crying in the wilderness, 'Make ready "
        "the way of the Lord! Make his paths straight!' John came "
        "baptizing in the wilderness."
    )

    sentences = split_sentences(
        narration,
        _protected_spans(narration),
    )

    assert sentences == [
        "As it is written, 'Behold, I send my messenger before your "
        "face; the voice of one crying in the wilderness, 'Make ready "
        "the way of the Lord! Make his paths straight!'",
        "John came baptizing in the wilderness.",
    ]
    assert " ".join(sentences) == narration


def test_a_stray_quote_closer_does_not_swallow_following_sentences():
    """
    A close-looking quote with nothing open - the possessive in "the
    disciples' boat" - must open no protected span. Otherwise every
    sentence after it collapses into one giant blob that no ceiling can
    group.
    """
    narration = (
        "They followed behind the boat's wake all morning. "
        "He walked beside the disciples' boat in the shallows. "
        "The wind stopped at noon."
    )

    sentences = split_sentences(
        narration,
        _protected_spans(narration),
    )

    assert len(sentences) == 3
    assert " ".join(sentences) == narration


def test_mark_002_chunks_respect_the_ceiling(capsys):
    """
    Regression for the shipped Mark 1:1-11 episode (002): blobs of
    37/91/69 words against a ~22-word ceiling. With word-count
    chunking, every chunk is at most the ceiling, every word appears
    exactly once in order, and nothing is flagged - chunking cuts at
    word boundaries regardless of quotation structure.
    """
    generator, _ = make_generator()

    ceiling = _max_words_per_segment(generator.config)

    segments = generator.build_segments(
        {"text": MARK_002_NARRATION}
    )

    # Exact coverage and ordering: every word once, in order.
    assert " ".join(segments) == MARK_002_NARRATION
    assert [
        word
        for segment in segments
        for word in segment.split()
    ] == MARK_002_NARRATION.split()

    # Strict ceiling: full ceiling-sized chunks plus the remainder.
    total = len(MARK_002_NARRATION.split())
    expected = [ceiling] * (total // ceiling)
    if total % ceiling:
        expected.append(total % ceiling)

    assert [len(s.split()) for s in segments] == expected
    assert all(
        len(segment.split()) <= ceiling
        for segment in segments
    )

    # No flagging: word-count chunking cuts at word boundaries, it
    # does not flag over-ceiling text for narration review.
    captured = capsys.readouterr().out
    assert "Narration review needed" not in captured


def test_an_unclosed_quotation_chunks_by_word_count(capsys):
    """
    An unclosed quotation is chunked by word count like any other
    text - cuts may fall mid-quotation, which is intentional for
    word-count-based segmentation. Nothing is flagged.
    """
    generator, _ = make_generator()

    narration = (
        "Jesus said, 'The hour has come, and the night walks. "
        "Whoever walks in the night does not see where he goes. "
        "The light is with you only a little while longer."
    )

    ceiling = _max_words_per_segment(generator.config)
    assert len(narration.split()) > ceiling

    segments = generator.build_segments({"text": narration})

    # Exact coverage and ordering: every word once, in order.
    assert " ".join(segments) == narration
    assert [
        word
        for segment in segments
        for word in segment.split()
    ] == narration.split()

    # Strict ceiling: full ceiling-sized chunks plus the remainder.
    total = len(narration.split())
    expected = [ceiling] * (total // ceiling)
    if total % ceiling:
        expected.append(total % ceiling)

    assert [len(s.split()) for s in segments] == expected
    assert all(
        len(segment.split()) <= ceiling
        for segment in segments
    )

    # No flagging: word-count chunking does not flag for review.
    captured = capsys.readouterr().out
    assert "Narration review needed" not in captured


