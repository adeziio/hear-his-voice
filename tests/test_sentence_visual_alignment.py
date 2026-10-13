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
    _max_words_per_sentence,
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
    build_sentence_segments,
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


def narration_sentence_count(text):
    """
    The number of sentences the narration is really made of, counted the
    way the segmenter counts them: a quotation is protected, so a spoken
    saying - "Save us, Lord! We are dying!" - is one sentence, never split
    at the "!" inside it. This is the baseline a dividing test compares a
    segment count against, so it must agree with build_sentence_segments.
    """
    return len(split_sentences(text, _protected_spans(text)))


# --------------------------------------------------------------------
# duration_target drives the sentence budget
# --------------------------------------------------------------------

def test_the_sentence_budget_comes_from_duration_target():

    config = make_config(
        duration_target="8s",
        words_per_second=2.8
    )

    assert _duration_target_seconds(config) == 8.0
    assert _max_words_per_sentence(config) == 22

    # A longer clip allows a longer sentence; a shorter one does not.
    assert _max_words_per_sentence(
        make_config(duration_target="12s")
    ) == 34

    assert _max_words_per_sentence(
        make_config(duration_target="4s")
    ) == 11


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

    assert _max_words_per_sentence(project_config) == max(
        1,
        round(_words_per_second(project_config) * seconds)
    )


# --------------------------------------------------------------------
# duration-based splitting
# --------------------------------------------------------------------

def test_a_shorter_duration_target_yields_more_segments():

    generator, _ = make_generator(duration_target="4s")

    short = generator.build_segments({"text": STORM_NARRATION})

    longer = make_generator(duration_target="30s")[0].build_segments(
        {"text": STORM_NARRATION}
    )

    assert len(short) > len(longer)


def test_a_long_sentence_is_divided_to_stay_inside_the_clip():

    generator, _ = make_generator(duration_target="4s")

    segments = generator.build_segments({"text": STORM_NARRATION})

    # The opening sentence is one long thought; it is divided at its own
    # comma so no segment runs past its clip. Compared against the true
    # sentence count (quotation-protected), the long sentence has been
    # broken up into shorter, natural pieces.
    sentences = split_sentences(
        STORM_NARRATION,
        _protected_spans(STORM_NARRATION),
    )

    assert len(segments) > len(sentences)

    # Every resulting segment is shorter than the sentence it was divided
    # from: the cut reduced the length. (A clause with no punctuation of
    # its own is left whole rather than chopped mid-phrase, so a segment
    # may still exceed the budget when the text gives no natural break -
    # but it never exceeds the sentence it came from.)
    longest_sentence_words = max(
        len(sentence.split())
        for sentence in sentences
    )

    for segment in segments:

        assert len(segment.split()) <= longest_sentence_words


def test_a_short_sentence_is_kept_whole_not_padded():

    # duration_target is a ceiling, never a duration to fill: a sentence
    # that already fits one clip is not split, merged, or padded.
    generator, _ = make_generator(duration_target="8s")

    segments = generator.build_segments(
        {"text": "Jesus rose. The wind stopped."}
    )

    assert segments == ["Jesus rose.", "The wind stopped."]


def test_segments_are_cut_only_at_punctuation():

    generator, _ = make_generator(duration_target="4s")

    segments = generator.build_segments({"text": STORM_NARRATION})

    # Only punctuation already in the text is used as a boundary, so a
    # segment never begins or ends on a stray fragment.
    for segment in segments:

        assert segment == segment.strip()

        assert segment[-1] not in " ([{"

    # Rejoined, the text is intact - proof the cuts fell at punctuation.
    assert " ".join(segments) == STORM_NARRATION

    # The quoted cry is never opened in one segment and closed in the next.
    for segment in segments:

        assert segment.count("'") != 1

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
    generator, _ = make_generator()

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
    A rejected visual never ships. There is no attempt cap: the loop
    re-rolls for as long as the checker keeps rejecting - seven
    rejections in a row (well past any former cap) must produce
    eight visual calls, and only the approved eighth query ships.
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

    assert stub.visual_calls == 8
    assert len(stub.verification_prompts) == 8

    # Every re-roll carried the rejection forward, and only the
    # checker-approved attempt is what shipped.
    assert all(
        "YOUR PREVIOUS ANSWER WAS REJECTED" in prompt
        for prompt in stub.visual_prompts[1:]
    )
    assert len(content["visuals"]) == 1
    assert "Shot 8." in content["visuals"][0]["search_query"]


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


def test_story_beat_grouping_merges_one_moment_into_one_visual():
    """
    The storm and the cry it draws belong on one screen: consecutive
    narration units that form one moment share a single visual, and
    the narration itself is carried over word for word.
    """
    generator, stub = make_generator(
        llm=StubLLM(
            grouping_replies=[{"groups": [[1, 2], [3]]}]
        ),
        # Ceiling of 32 words, so unit 1 (22 words) may share a clip
        # with unit 2 - the merge is a real choice, not one the
        # duration cap would split straight back apart.
        words_per_second=4,
    )

    segments = generator.build_story_beat_segments(
        {"text": STORM_NARRATION}
    )

    assert len(stub.grouping_prompts) == 1
    assert len(segments) == 2

    # Compare against the quote-protected units, which are what the
    # grouping actually operates on: the cry stays whole in unit 2.
    ceiling = _max_words_per_sentence(generator.config)
    units = build_sentence_segments(STORM_NARRATION, ceiling)

    assert segments[0] == f"{units[0]} {units[1]}"
    assert segments[1] == units[2]

    # Complete coverage: the segments ARE the narration, with nothing
    # omitted, duplicated, or rewritten.
    assert " ".join(segments) == STORM_NARRATION


def test_a_broken_grouping_is_rejected_and_re_rolled_with_the_reason():
    """
    A partition that drops a unit can never be shipped: it is
    rejected, the reason goes back in the next prompt, and the fixed
    partition takes its place.
    """
    generator, stub = make_generator(
        llm=StubLLM(
            grouping_replies=[
                # Unit 2 is missing - an unusable partition.
                {"groups": [[1], [3]]},
                {"groups": [[1, 2], [3]]},
            ]
        ),
        words_per_second=4,
    )

    segments = generator.build_story_beat_segments(
        {"text": STORM_NARRATION}
    )

    assert len(stub.grouping_prompts) == 2

    rejected = stub.grouping_prompts[1]
    assert "YOUR PREVIOUS GROUPING WAS REJECTED" in rejected
    assert "missing unit(s) [2]" in rejected
    assert len(segments) == 2


def test_unusable_grouping_falls_back_to_one_visual_per_sentence():
    """
    Fail-safe, not fail-silent: when no attempt returns a usable
    partition, the old one-visual-per-sentence cut is used - always
    correct, even if less clever.
    """
    generator, stub = make_generator(
        llm=StubLLM(
            grouping_replies=[
                "not json",
                {"groups": [[1, 9]]},
                {"groups": "everything"},
            ]
        )
    )

    segments = generator.build_story_beat_segments(
        {"text": STORM_NARRATION}
    )

    assert len(stub.grouping_prompts) == 3
    assert segments == generator.build_segments(
        {"text": STORM_NARRATION}
    )
    assert len(segments) == 3


def test_grouping_never_exceeds_the_clip_spoken_ceiling():
    """
    duration_target is the MAXIMUM spoken length of a group, never a
    duration to fill: a group built beyond it is cut back at unit
    boundaries - never inside a unit - and every word survives.
    """
    generator, stub = make_generator(
        llm=StubLLM(
            grouping_replies=[{"groups": [[1, 2, 3]]}]
        )
    )

    segments = generator.build_story_beat_segments(
        {"text": STORM_NARRATION}
    )

    ceiling = _max_words_per_sentence(generator.config)

    assert len(stub.grouping_prompts) == 1
    assert len(segments) == 2
    assert all(
        len(segment.split()) <= ceiling
        for segment in segments
    )
    assert " ".join(segments) == STORM_NARRATION


def test_story_beat_grouping_can_be_switched_off():
    generator, stub = make_generator(
        llm=StubLLM(),
        story_beat_grouping=False,
    )

    segments = generator.build_story_beat_segments(
        {"text": STORM_NARRATION}
    )

    assert not stub.grouping_prompts
    assert len(segments) == 3


def test_the_grouping_prompt_names_the_units_the_ceiling_and_the_feedback():
    generator, _ = make_generator()
    units = build_sentence_segments(STORM_NARRATION, 22)

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


def test_an_all_single_sentence_grouping_is_re_rolled_when_merging_was_possible():
    """
    The exact failure this feature exists for: a partition with one
    unit per visual is rejected while some pair of units would have
    fit one clip, and the re-roll carries the reason.
    """
    generator, stub = make_generator(
        llm=StubLLM(
            grouping_replies=[
                {"groups": [[1], [2], [3]]},
                {"groups": [[1, 2], [3]]},
            ]
        ),
        # Ceiling of 32 words: units 1 and 2 (22 + 10) could share.
        words_per_second=4,
    )

    segments = generator.build_story_beat_segments(
        {"text": STORM_NARRATION}
    )

    assert len(stub.grouping_prompts) == 2
    assert "Every group held a single unit" in stub.grouping_prompts[1]
    assert len(segments) == 2


def test_single_sentence_groups_are_kept_when_no_pair_fits_one_clip():
    """
    The merge rule never overrules the duration ceiling: when every
    consecutive pair would overrun the clip, one visual per unit is
    the correct cut and stands on the first attempt.
    """
    narration = (
        "The disciples gathered the fragments that remained and "
        "filled twelve baskets with the pieces from the meal that "
        "day. Then he sent the crowds away and went up the mountain "
        "alone to pray while the evening came upon the shore. "
        "The boat was out in the middle of the sea by then, beaten "
        "by the waves as the wind drove against it in the night."
    )

    generator, stub = make_generator(
        llm=StubLLM(
            grouping_replies=[
                {"groups": [[1], [2], [3], [4]]}
            ]
        )
    )

    segments = generator.build_story_beat_segments(
        {"text": narration}
    )

    assert len(stub.grouping_prompts) == 1
    assert len(segments) == 4
    assert " ".join(segments) == narration


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
    max_words = _max_words_per_sentence(config)
    units = build_sentence_segments(EPISODE_001_NARRATION, max_words)

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
    max_words = _max_words_per_sentence(config)
    units = build_sentence_segments(EPISODE_001_NARRATION, max_words)

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


def test_the_episode_regression_merges_beats_after_an_all_single_sentence_reply():
    """
    The shipped failure as a regression: the episode's visuals were
    cut one per unit. The model's default all-singleton reply is
    rejected; on the re-roll it merges exactly the fitting beat pairs
    - the mountain prayer with the evening, Peter's request with
    "Come!", and the sinking with the cry for help - while coverage
    stays exact and no clip overruns the ceiling.
    """
    generator, stub = make_generator(
        llm=StubLLM(
            grouping_replies=[
                {"groups": [[i] for i in range(1, 16)]},
                {
                    "groups": [
                        [1],
                        [2, 3],
                        [4],
                        [5],
                        [6],
                        [7],
                        [8, 9],
                        [10],
                        [11, 12],
                        [13],
                        [14],
                        [15],
                    ]
                },
            ]
        )
    )

    segments = generator.build_story_beat_segments(
        {"text": EPISODE_001_NARRATION}
    )

    assert len(stub.grouping_prompts) == 2
    assert "Every group held a single unit" in stub.grouping_prompts[1]

    max_words = _max_words_per_sentence(make_config())
    units = build_sentence_segments(EPISODE_001_NARRATION, max_words)

    assert len(segments) == 12
    assert " ".join(segments) == EPISODE_001_NARRATION
    assert all(
        len(segment.split()) <= max_words
        for segment in segments
    )
    # The three pairs the shipped episode cut incorrectly.
    assert segments[1] == units[1] + " " + units[2]
    assert segments[6] == units[7] + " " + units[8]
    assert segments[8] == units[10] + " " + units[11]


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

