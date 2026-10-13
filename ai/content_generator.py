import json
import re

from pathlib import Path

from ai.base_ai_service import BaseAIService
from ai.providers.ollama_provider import OllamaProvider

from scripture.segmenter import (
    split_sentences,
    build_sentence_segments
)


# `target_duration_seconds` is only a target for narration generation. Visual
# timing is derived from the narration that was actually generated.
#
# Each SnapGenAI clip carries one visual segment, and its length is the
# `duration_target` in config/ai_models.json (for example "8s"). The
# spoken-length ceiling used when cutting the narration is read from that
# setting by _duration_target_seconds() rather than hardcoded: consecutive
# narration units that form one story beat share one segment (the model
# chooses the beats, build_story_beat_segments enforces the coverage and
# the ceiling), while a segment that would speak for longer than the clip
# is divided at its own punctuation. A shorter segment is fine as it is -
# the duration is a maximum, never a duration to fill. The default below
# is only the fallback used when that setting is missing or cannot be
# parsed.
DEFAULT_SECONDS_PER_VISUAL = 8
DEFAULT_WORDS_PER_SECOND = 2.9


def _duration_target_seconds(config):
    """
    Returns the SnapGenAI clip length in seconds, read from
    config/ai_models.json -> models.video_model.snapgenai.duration_target
    (for example "8s").

    One clip carries one narration sentence, so this is the maximum
    spoken length any one segment should reach before it is divided. The
    value comes from configuration, never from a hardcoded constant;
    DEFAULT_SECONDS_PER_VISUAL is only the fallback used when the setting
    is missing or unparseable.
    """
    snapgenai = (
        config.get("ai_models", {})
        .get("models", {})
        .get("video_model", {})
        .get("snapgenai", {})
    )

    match = re.fullmatch(
        r"(\d+(?:\.\d+)?)\s*s?",
        str(
            snapgenai.get(
                "duration_target",
                ""
            )
        ).strip(),
        flags=re.IGNORECASE,
    )

    if match is None:

        return DEFAULT_SECONDS_PER_VISUAL

    seconds = float(
        match.group(1)
    )

    if seconds <= 0:

        return DEFAULT_SECONDS_PER_VISUAL

    return seconds


def _words_per_second(config):
    """
    Returns the narrator's speaking rate, read from config/content.json ->
    content.content_generation.words_per_second.

    It lives in content.json with the rest of the narration settings, so
    there is one content config; DEFAULT_WORDS_PER_SECOND is only the
    fallback used when the setting is missing or unusable.
    """
    value = float(
        config.get("content", {})
        .get("content_generation", {})
        .get("words_per_second", DEFAULT_WORDS_PER_SECOND)
    )

    return max(
        value,
        0.1
    )


def _max_words_per_sentence(config):
    """
    Returns the most words one narration sentence may hold.

    A sentence is spoken over exactly one SnapGenAI clip, so its spoken
    length may not exceed that clip's duration_target. This is the same
    arithmetic build_segments() uses to cut the narration, and it is
    reused to tell the writer the budget up front, so the telling is
    written to fit the clips rather than cut to fit them afterwards.

    Both inputs come from configuration, so the limit changes with the
    configured clip length or speaking rate instead of a hardcoded number.
    """
    return max(
        1,
        int(
            round(
                _words_per_second(config)
                * _duration_target_seconds(config)
            )
        ),
    )


class ContentGenerationError(
    RuntimeError
):

    pass


# The AI directs the episode; it does not author the Scripture.
#
# Visuals are directed ONE narration sentence at a time. Each focused
# call returns exactly one visual object for the single sentence it was
# given, so the model is never asked to hold the whole story in one
# response - the failure mode that collapsed an episode into a handful
# of repeated establishing shots that contradicted the narration around
# them. The narration itself is written by write_narration() under
# NARRATION_SCHEMA below, from the WEBC passage as its only source, so
# neither call gives the model anywhere to put invented Scripture.
VISUAL_SCHEMA = {
    "type": "object",
    "properties": {
        "search_query": {"type": "string"},
        "visual_direction": {"type": "string"},
    },
    "required": ["search_query", "visual_direction"],
}


# The metadata schema: the episode title, summary, and mood, asked for
# in one separate call after the visuals are written. maxLength is not
# decoration - it is a grammar-level brake on the two failure modes this
# channel actually hit: a poetic title ("Genealogy of Light") and an
# interpretive summary ("revealing a divine ancestry rooted in faith and
# promise" for a passage that says no such thing).
METADATA_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "maxLength": 90},
        "summary": {"type": "string", "maxLength": 180},
        "mood": {"type": "string"},
    },
    "required": ["title", "summary", "mood"],
}


# The narration schema. One field, and it holds the finished telling -
# the model can return nothing else from this call, so there is no
# second field for stray Bible text or commentary to hide in.
NARRATION_SCHEMA = {
    "type": "object",
    "properties": {
        "narration": {
            "type": "string"
        }
    },
    "required": [
        "narration"
    ]
}


# The verification schemas. Each correctness pass answers ONE boolean
# question in structured form - "does this visual depict this exact
# sentence?" and "is this narration accurate to the passage?" - so
# generate() and write_narration() can judge and re-roll their own
# output instead of shipping whatever the first sample said. The
# reason/issues field carries WHY it failed, which becomes the feedback
# the re-roll is written against. A reply that cannot be parsed counts
# as a failure, never as a pass.
VISUAL_VERIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "aligned": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["aligned", "reason"],
}


NARRATION_VERIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "accurate": {"type": "boolean"},
        "issues": {"type": "string"},
    },
    "required": ["accurate", "issues"],
}


# The narration is cut into visual segments BY STORY BEAT: consecutive
# narration units that form one coherent moment share one visual. The
# model returns which unit numbers (1..N) belong to each group, and the
# validation in _validate_grouping rejects any partition that omits,
# repeats, or reorders a unit - so narration coverage is guaranteed in
# code even though the choice of beats is the model's judgement.
SEGMENT_GROUPING_SCHEMA = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "integer"},
            },
        },
    },
    "required": ["groups"],
    "additionalProperties": False,
}


# Commentary that has no place in spoken narration. Each phrase is
# something a model reaches for when it stops telling the passage and
# starts talking about it.
COMMENTARY_MARKERS = (
    "in this passage",
    "in this verse",
    "this passage tells",
    "this verse tells",
    "the bible tells us",
    "scripture tells us",
    "as we read in",
)

# Capitalised words that are only names sometimes. English opens a
# sentence with a capital, so "JESUS said" capitalises nothing useful;
# a word in this list therefore only counts as a name when the source
# also uses it mid-sentence, where the capital cannot be the reason.
SENTENCE_START_WORDS = frozenset(
    """
    a about after again against all also am an and another any anyone
    anything are as at away back be because been before being both but
    by came come could did do does doing down each either else even ever
    every everyone everything few for from further get gets go goes
    going got had has have he her here hers herself him himself his how
    however i if in into is it its itself just let like made make makes
    may me might more most much must my myself neither no nor not
    nothing now of off often on once one only or other others our ours
    ourselves out over own perhaps said same saw say says see seen
    shall she should since so some someone something still such sure
    than that the their theirs them themselves then there therefore
    these they thing things this those though through thus till to
    together toward under until up upon us very was we were what when
    where whether which while who whom whose why will with within
    without would yet you your yours yourself yourselves
    behold verily wherefore moreover yea hence indeed truly finally
    """.split()
)

# A name is compared with its possessive and hyphen tail removed, so
# "Isaac", "Isaac's" and "ISAAC" are all the same name.
NAME_TOKEN = re.compile(
    r"[A-Za-z][A-Za-z'’\-]*"
)

# Sentence punctuation: a capitalised word directly after one of these
# opens a sentence or a quotation, so its capital is not a name's.
SENTENCE_BOUNDARY = tuple(
    ".?!:;\"”'’()[]{}"
)


def _normalize_words(text):
    """
    Lower-cased words with the punctuation gone, for comparing a
    telling against its source without caring about case or stops.
    """
    return re.findall(
        r"[a-z0-9]+",
        str(text or "").lower()
    )


def _name_key(word):
    """
    The comparable form of a name: lower-cased, with any possessive or
    hyphenated tail dropped, so "Isaac's", "Isaac" and "ISAAC" all give
    "isaac".
    """
    head = re.split(
        r"[’'\-]",
        str(word or "")
    )[0]

    return re.sub(
        r"[^a-z]",
        "",
        head.lower()
    )


def source_names(text):
    """
    Every person, place, and title the passage names.

    A capitalised word that is not opening a sentence can only be a
    name ("...the father of Isaac", "...by Tamar"), and a word that
    keeps its capital everywhere it appears is a name even when it
    does open a sentence ("JESUS said" is still Jesus). Those names
    ARE much of the passage's content - in a genealogy they are the
    whole of it - so they are what the faithfulness check counts.
    """
    value = str(text or "")

    mid_sentence = set()

    for match in NAME_TOKEN.finditer(value):

        word = match.group(0)

        if len(word) < 3 or not word[:1].isupper():

            continue

        key = _name_key(word)

        if not key:

            continue

        before = value[:match.start()].rstrip()

        if before and before[-1] not in SENTENCE_BOUNDARY:

            mid_sentence.add(key)

    names = set(mid_sentence)

    # Seen only at the head of a sentence: still a name unless it is
    # ordinary English that merely starts sentences ("Behold").
    for match in NAME_TOKEN.finditer(value):

        word = match.group(0)

        if len(word) < 3 or not word[:1].isupper():

            continue

        key = _name_key(word)

        if key and key not in SENTENCE_START_WORDS:

            names.add(key)

    return names


def verbatim_sentence_overlap(source_text, narration):
    """
    Retained as a compatibility helper for older callers.

    Narration generation does not call this function and does not reject
    output based on source similarity.
    """
    source_sentences = [
        sentence
        for sentence in split_sentences(str(source_text or ""))
        if sentence
    ]

    if not source_sentences:

        return 0.0

    source_words = _normalize_words(narration)
    matched = 0

    for sentence in source_sentences:

        words = _normalize_words(sentence)

        if words and tuple(words) in {
            tuple(source_words[index:index + len(words)])
            for index in range(
                max(1, len(source_words) - len(words) + 1)
            )
        }:

            matched += 1

    return matched / len(source_sentences)


def narration_problems(source_text, narration):
    """
    Compatibility helper that only reports an empty narration.

    The pipeline deliberately does not judge similarity, names, facts,
    length, wording, creativity, or faithfulness to the source.
    """
    if not str(narration or "").strip():

        return ["the narration is empty"]

    return []


class ContentGenerator(
    BaseAIService
):

    """
    The creative team for a Hear His Voice episode.

    Given the exact WEBC passage, this:

        1. tells the passage as original narration, checked against
           the passage itself for fidelity (write_narration),
        2. cuts that narration into story-beat visual segments - the
           model groups the beats, code guarantees complete
           narration coverage and the duration ceiling
           (build_story_beat_segments),
        3. asks the AI for one SnapGenAI prompt per narration segment,
           each shown the narration immediately before and after it so
           the prompt fits that exact moment of the story,
        4. asks for a title, summary, and mood.

    The passage remains the authority throughout: the narration may
    only restate what the WEBC text says, and content["source_text"]
    keeps the exact passage beside the telling that came from it.
    """

    def __init__(
        self,
        config
    ):

        super().__init__(
            config,
            "CONTENT"
        )

        self.llm = OllamaProvider(
            config
        )

        self.channel_config = config["content"]

        self.generation_config = (
            self.channel_config.get(
                "content_generation",
                {}
            )
        )

    def format_bullets(
        self,
        values
    ):

        if not isinstance(
            values,
            list
        ):

            return ""

        return "\n".join(
            f"- {str(value).strip()}"
            for value in values
            if str(value).strip()
        )

    def build_segments(
        self,
        scripture
    ):
        """
        Splits the generated narration into visual segments: one segment
        per completed narration sentence, so each sentence gets exactly
        one SnapGenAI prompt and one clip.

        The configured duration_target is the maximum spoken length a
        segment may reach - only a sentence that would speak for longer
        than one clip is divided at its own punctuation into shorter
        sentences. A shorter sentence is kept whole; the duration is a
        ceiling, never a duration to fill.
        """
        text = scripture["text"]

        # How fast the narrator reads, and how long one clip holds - both
        # from configuration, through the same helpers the narration prompt
        # uses to tell the writer the budget up front. One sentence is
        # spoken over one clip, so this is the ceiling for one segment.
        words_per_second = _words_per_second(self.config)
        seconds_per_visual = _duration_target_seconds(self.config)
        max_words_per_segment = _max_words_per_sentence(self.config)

        segments = build_sentence_segments(
            text,
            max_words_per_segment,
        )

        self.log(
            f"{len(segments)} sentence visuals for "
            f"{len(split_sentences(text))} narration sentences "
            f"({len(text.split())} words, "
            f"~{len(text.split()) / max(words_per_second, 0.1):.0f}s, "
            f"at most {seconds_per_visual:g}s per segment)."
        )

        return segments

    def build_grouping_prompt(
        self,
        scripture,
        units,
        max_words,
        feedback=None
    ):
        """
        Asks which consecutive narration units form one visual.

        The model judges the story beats - a request and its immediate
        reply are one moment; a person then acting begins a new visual
        - while the prompt states the two rules the code re-checks
        anyway: every unit used exactly once in order (complete
        narration coverage) and at most max_words per group (the
        clip's spoken ceiling from duration_target - a maximum to
        respect, never a duration to fill).
        """
        channel = self.channel_config.get("channel", {})
        name = str(channel.get("name", "Hear His Voice"))

        numbered = "\n".join(
            f"{position + 1}. {text}"
            for position, text in enumerate(units)
        )

        # The ceiling arithmetic is code's job, the beat judgment is
        # the model's: list every consecutive pair whose combined
        # spoken length fits one clip, so the model rules on each
        # pair explicitly instead of defaulting to one cut per unit.
        eligible = []

        for position in range(1, len(units)):

            pair_words = (
                len(units[position - 1].split())
                + len(units[position].split())
            )

            if pair_words <= max_words:

                eligible.append(
                    (
                        position,
                        pair_words,
                        units[position - 1],
                        units[position],
                    )
                )

        pair_lines = "".join(
            f"  units {position} + {position + 1} "
            f"({pair_words} words total):\n"
            f'    "{first}"\n'
            f'    "{second}"\n'
            for position, pair_words, first, second in eligible
        )

        pairs_block = ""

        if eligible:

            pairs_block = (
                "PAIRS THAT FIT ONE CLIP - DECIDE EACH ONE\n"
                "Every pair listed below is consecutive and its "
                "combined spoken length is within the clip ceiling, "
                "so the ceiling never forces a cut between them. For "
                "each pair decide explicitly: if the two units are "
                "ONE continuous moment - a request and its reply, an "
                "action and its immediate effect, speech and the "
                "reaction it draws - they share one visual; only a "
                "genuine story beat change justifies the cut. Never "
                "cut a pair merely because it holds two "
                "sentences.\n"
                + pair_lines
                + "\n"
            )

        rejection = ""

        if feedback:

            rejection = (
                "YOUR PREVIOUS GROUPING WAS REJECTED\n"
                "Fix this problem in your next answer:\n"
                f"{feedback}\n\n"
            )

        return (
            "You cut the narration of \""
            + name
            + "\" into visual segments - ONE VISUAL PER STORY BEAT, "
            "not one visual per sentence.\n\n"
            + rejection
            + "Context: "
            + str(scripture.get("translation", ""))
            + " "
            + str(scripture.get("reference", ""))
            + ". The narration below was already written from that "
            "passage and is FIXED: the narrator speaks it exactly as "
            "written. You group its units into visuals; you never "
            "edit, reorder, add, or remove a single word.\n\n"
            "FULL NARRATION:\n"
            + " ".join(units)
            + "\n\n"
            "NARRATION UNITS TO GROUP (numbered):\n"
            + numbered
            + "\n\n"
            + pairs_block
            + "GROUP THE UNITS INTO VISUALS\n"
            "- Return groups: a list of groups, each holding the unit "
            "numbers of consecutive units that share one visual.\n"
            "- MERGE FIRST: consecutive units that belong to ONE "
            "coherent moment - a request and its immediate reply, an "
            "action and its immediate effect, speech and the "
            "reaction it draws - must share one visual whenever "
            "their combined length fits the ceiling. For instance, "
            "a disciple asking to come and the reply \"Come!\" are "
            "one visual; the disciple then stepping out of the boat "
            "begins the next one.\n"
            "- Cut at a genuine story beat only: a new action "
            "begins, the place or time changes, or the depicted "
            "moment is over. Short groups are fine, but cut because "
            "the beat changed - never just to keep groups small.\n"
            "- One visual per unit - one per sentence - is a FAILED "
            "answer whenever any two consecutive units could have "
            "shared one clip.\n"
            "- Every number from 1 to "
            + str(len(units))
            + " must appear EXACTLY ONCE, in ascending order across "
            "the groups. No unit may be omitted, repeated, or "
            "reordered: together the groups cover the whole "
            "narration, in order, word for word.\n"
            "- Start a new group when a new action, place, or beat "
            "begins. A group may hold a single unit; it just needs "
            "a reason from the story, not the sentence "
            "boundary.\n"
            "- Each group's total must be "
            + str(max_words)
            + " words or fewer: that is the spoken ceiling of one "
            "clip (its duration_target). The ceiling is a maximum, "
            "never a duration to fill - if the story beats would "
            "overrun it, cut at the next natural beat instead.\n"
            "- Never split a unit in half; never rewrite any text.\n\n"
            'Return JSON: {"groups": [[1, 2], [3], [4, 5]]}'
        )

    def _validate_grouping(
        self,
        reply,
        count
    ):
        """
        Returns (groups, error): a usable partition of 1..count, or the
        reason the reply cannot be trusted.

        Only an explicit, complete, in-order partition passes: a reply
        with no JSON, a gap, a repeat, or an out-of-range index is
        rejected with this reason, which becomes the feedback on the
        re-roll.
        """
        try:

            payload = self.parse_direction(
                reply
            )

        except ContentGenerationError:

            return (
                None,
                "The reply contained no parseable JSON.",
            )

        groups = (
            payload.get("groups")
            if isinstance(payload, dict)
            else None
        )

        if not isinstance(groups, list) or not groups:

            return (
                None,
                'The reply must be {"groups": [[...]]} with at '
                "least one group.",
            )

        flattened = []

        for group in groups:

            if not isinstance(group, list) or not group:

                return (
                    None,
                    "Every group must be a non-empty list of unit "
                    "numbers.",
                )

            for item in group:

                if not isinstance(item, int) or isinstance(item, bool):

                    return (
                        None,
                        "Unit numbers must be integers.",
                    )

                if not 1 <= item <= count:

                    return (
                        None,
                        f"Unit {item} does not exist; units run 1 to "
                        f"{count}.",
                    )

                flattened.append(item)

        expected = list(
            range(1, count + 1)
        )

        if flattened != expected:

            missing = sorted(
                set(expected) - set(flattened)
            )
            repeated = sorted(
                {
                    number
                    for number in flattened
                    if flattened.count(number) > 1
                }
            )
            detail = []

            if missing:

                detail.append(f"missing unit(s) {missing}")

            if repeated:

                detail.append(f"repeated unit(s) {repeated}")

            if not detail:

                detail.append(
                    "the units must run in ascending order across "
                    "the groups"
                )

            return (
                None,
                "The groups must cover every unit exactly once, in "
                "order; " + "; ".join(detail) + ".",
            )

        return (
            groups,
            None,
        )

    @staticmethod
    def _cap_group_words(
        groups,
        units,
        max_words
    ):
        """
        Enforces the spoken ceiling on top of the model's beat choice.

        The story beats come from the model; the clip length does not.
        A group whose units would speak longer than the clip is cut
        into consecutive pieces at unit boundaries - never inside a
        unit - and a group already within the ceiling is left exactly
        as the model chose it.
        """
        capped = []

        for group in groups:

            piece = []
            words = 0

            for item in group:

                unit_words = len(
                    units[item - 1].split()
                )

                if (
                    piece
                    and words + unit_words > max_words
                ):

                    capped.append(piece)
                    piece = []
                    words = 0

                piece.append(item)
                words += unit_words

            if piece:

                capped.append(piece)

        return capped

    def build_story_beat_segments(
        self,
        scripture
    ):
        """
        Cuts the narration into one visual per STORY BEAT instead of
        one visual per sentence.

        The model decides which consecutive narration units form one
        coherent moment; this method guarantees the rest: every unit
        appears in exactly one segment, in order, joined word for word
        from the units themselves (no narration is omitted, repeated,
        or rewritten), and no segment may speak longer than the
        clip's duration_target ceiling - a maximum, never a duration
        to fill. When grouping is switched off or the model never
        returns a usable partition, it falls back to build_segments():
        one visual per sentence - always correct, if less clever.
        """
        if not self.generation_config.get(
            "story_beat_grouping",
            True
        ):

            return self.build_segments(
                scripture
            )

        text = scripture["text"]
        max_words = _max_words_per_sentence(self.config)
        units = build_sentence_segments(
            text,
            max_words,
        )

        if len(units) < 2:

            return self.build_segments(
                scripture
            )

        attempts = max(
            1,
            int(
                self.generation_config.get(
                    "grouping_attempts",
                    3
                )
            ),
        )

        groups = None
        feedback = None

        for _ in range(attempts):

            prompt = self.build_grouping_prompt(
                scripture,
                units,
                max_words,
                feedback,
            )

            try:

                reply = self.llm.generate(
                    prompt,
                    response_format=SEGMENT_GROUPING_SCHEMA,
                )

            except Exception as error:

                feedback = (
                    f"The grouping request itself failed: {error}"
                )
                continue

            groups, feedback = self._validate_grouping(
                reply,
                len(units),
            )

            if groups is None:

                continue

            # One visual per sentence is exactly what this feature
            # exists to stop: reject an all-singleton partition when
            # at least one pair of consecutive units would have fit
            # one clip together. Singleton stays legal when no pair
            # could ever share the clip - the ceiling wins over the
            # merge, never the other way round.
            if (
                len(groups) == len(units)
                and any(
                    len(units[position - 1].split())
                    + len(units[position].split())
                    <= max_words
                    for position in range(1, len(units))
                )
            ):

                groups = None
                feedback = (
                    "Every group held a single unit, so not one of "
                    "the fitting pairs listed in this prompt was "
                    "merged. Rule on each listed pair: pairs that "
                    "form one continuous moment share one visual; "
                    "only a genuine story beat change justifies a "
                    "cut."
                )
                continue

            break

        if groups is None:

            self.log(
                "Story-beat grouping never returned a usable "
                f"partition after {attempts} attempts; falling back "
                "to one visual per sentence."
            )

            return self.build_segments(
                scripture
            )

        groups = self._cap_group_words(
            groups,
            units,
            max_words,
        )

        segments = [
            " ".join(
                units[item - 1]
                for item in group
            )
            for group in groups
        ]

        words_per_second = _words_per_second(self.config)
        seconds_per_visual = _duration_target_seconds(self.config)

        self.log(
            f"{len(segments)} story-beat visuals from "
            f"{len(units)} narration units "
            f"({len(text.split())} words, "
            f"~{len(text.split()) / max(words_per_second, 0.1):.0f}s, "
            f"at most {seconds_per_visual:g}s per segment)."
        )

        return segments

    def build_segment_prompt(
        self,
        scripture,
        segments,
        index,
        feedback=None
    ):
        """
        Builds the visual-direction prompt for ONE narration sentence.

        A single sentence needs a single focused call: handing the model
        every sentence at once invites it to collapse the whole story
        into a few generic establishing shots. This prompt gives the
        model the full passage and the full telling for context, but
        marks exactly one sentence - with the narration immediately
        before and after it - as the one to direct. The output is one
        visual object, so the model cannot merge or drop beats.
        """
        channel = self.channel_config.get("channel", {})
        name = str(channel.get("name", "Hear His Voice"))
        description = str(channel.get("description", ""))

        visual_rules = self.format_bullets(
            self.generation_config.get("visual_rules", [])
        )
        creative_directions = self.format_bullets(
            self.generation_config.get("creative_directions", [])
        )

        segment = segments[index]
        before = (
            segments[index - 1] if index > 0
            else "(nothing - this sentence opens the narration)"
        )
        after = (
            segments[index + 1] if index + 1 < len(segments)
            else "(nothing - this sentence closes the narration)"
        )

        # The whole telling, for context only. A plain numbered list,
        # not SEGMENT markers, so the one marked sentence below stays
        # the only thing the model is asked to direct.
        full_telling = "\n".join(
            f"{position + 1}. {text}"
            for position, text in enumerate(segments)
        )

        prompt = (
            "You are the visual director for \""
            + name
            + "\", a short-form channel that lets Scripture speak "
            "for itself.\n\n"
            "CRITICAL - YOU DO NOT WRITE THE SCRIPTURE\n"
            "The passage below is the exact "
            + scripture["translation"]
            + " text, and it is the authority for this episode. The "
            "narration was already written from that passage and is "
            "fixed: the narrator speaks it exactly as written and it is "
            "captioned exactly as spoken. You must NEVER generate, "
            "paraphrase, rewrite, summarise, expand, correct, or invent "
            "any part of the passage or the narration, and you must not "
            "echo either back in your answer. Your only job is to decide "
            "what the viewer should SEE while ONE sentence of that "
            "narration is spoken.\n\n"
            "CHANNEL DESCRIPTION\n"
            + description
            + "\n\nSCRIPTURE REFERENCE\n"
            + scripture["reference"]
            + " ("
            + scripture["translation"]
            + ")\n\n"
            "THE PASSAGE - THE SOURCE OF EVERYTHING SPOKEN\n"
            + str(scripture.get("text") or "").strip()
            + "\n\n"
            "THE FULL NARRATION - CONTEXT ONLY\n"
            "This is the whole telling, in order. Read it to know who "
            "the people are and what has already happened. You are NOT "
            "directing all of it - only the one marked sentence below.\n"
            + full_telling
            + "\n\n"
            "THE SENTENCE YOU ARE DIRECTING\n"
            "Write exactly one visual for this sentence and nothing "
            "else. Depict only what this sentence has happening right "
            "now: an ask is shown as the asking, a cry as the crying "
            "out, a decision as the deciding - never the doing that "
            "follows it. The two indented lines are context for this "
            "moment - they are not part of what you depict.\n"
            f"SEGMENT {index + 1}: {segment}\n"
            f"  JUST BEFORE: {before}\n"
            f"  JUST AFTER: {after}\n\n"

            "VISUAL PROMPT RULES\n"
            "Return one visual object for this ONE sentence. Its "
            "search_query is a single, complete SnapGenAI visual prompt "
            "for this sentence's clip. It must depict the one clear "
            "event or action this sentence narrates - its setting, its "
            "people, and its emotional moment - and nothing else. "
            "Anchor the shot at this exact point in the passage's "
            "chronology: read the passage above to find where this "
            "sentence falls in the story, and show only what is "
            "happening at that point. Never depict an event from a "
            "later or an earlier part of the passage - no "
            "flash-forward to what happens next and no flashback to "
            "what already happened. If the sentence is a cry, a "
            "question, a statement, or a decision, depict exactly that "
            "moment, not the action it leads to. Never combine the "
            "events of two segments into one prompt, and "
            "never reach into the next sentence's beat before the "
            "narration gets there. Do not reuse a generic "
            "overall-passage establishing shot. Make the prompt "
            "straightforward and descriptive: identify the main scene, "
            "subject, setting, action, and atmosphere, and give the "
            "camera framing that fits the moment (for example "
            "close-up, medium shot, wide shot, or establishing shot). "
            "Begin with 'Style: realistic.' Do not write a "
            "stock-footage search query, narration, explanation, or "
            "list.\n"
            "The prompt must stand alone: SnapGenAI sees only the "
            "prompt, so resolve every pronoun and vague reference to "
            "the actual people the passage establishes. Write 'Jesus "
            "and Peter climb into the boat', not 'they climb into the "
            "boat', whenever the passage names who 'they' are. Only "
            "name people the passage itself establishes - never guess "
            "an identity.\n"
            "Name Jesus explicitly whenever he appears in a scene: "
            "always write 'Jesus', never an unnamed stand-in such as "
            "'a lone figure', 'a solitary figure', or 'a distant man'. "
            "Naming him is not describing him: write 'Jesus stands on "
            "the hillside in prayer', never 'a solitary figure "
            "prays'. Never describe his physical appearance - no "
            "face, hair, build, or clothing details for him. Show him "
            "by his actions, his position, and the people around "
            "him.\n"
            "When Jesus is not in the scene, describe the other visible "
            "people plainly and historically: who they are in the "
            "story, and their period-appropriate appearance, clothing, "
            "and role. For first-century Jewish people, use plausible "
            "Levantine features and simple wool or linen tunics, "
            "cloaks, head coverings, and sandals. Represent every other "
            "group according to the identity the passage gives them.\n"
            "Set every prompt in the biblical era: stone and mudbrick "
            "buildings, ancient streets, fields, hillsides, shores, "
            "boats, and period objects. Never show modern clothing, "
            "hairstyles, buildings, roads, vehicles, technology, or any "
            "modern environment.\n"
            "Keep the prompt focused and concise - include only the "
            "details that support the narrated event. Never invent "
            "actions, people, objects, or outcomes the passage does not "
            "give.\n"
            "Carry continuity from the JUST BEFORE line: if the "
            "previous sentence is in the same place and situation, keep "
            "the same location, positions, and conditions here too, and "
            "change them only where the narration itself changes them - "
            "a move, an entrance, a departure, or a change of time or "
            "weather. Never move the scene onto a shore, onto land, or "
            "into a building, and never turn night into dawn or "
            "daylight, unless the narration itself says so.\n"
            "Place people exactly where the narration places them. "
            "Where the narration does not say where someone is, the "
            "passage above decides it: put each person where the "
            "passage has them at this point in the story, and never "
            "place a character where the passage contradicts - when "
            "the passage keeps the people together in the boat, do "
            "not move Jesus to the shore. If "
            "someone is outside a place - outside the tomb, on the "
            "shore, at a distance, in the boat, before a closed door - "
            "keep them outside it unless the narration says they "
            "entered. Never move a scene indoors, into a city, or onto "
            "a road the narration never mentions.\n"
            "Where the passage keeps Jesus apart from the group - "
            "walking on the water, standing apart to pray, no longer "
            "in the tomb - keep him apart for as long as the passage "
            "does, and never move him into the boat, the house, or the "
            "crowd unless the narration says he entered.\n"
            "Depict each action in a physically plausible position: "
            "someone asleep lies down, someone praying kneels or stands "
            "with head bowed, someone travelling is on their feet on "
            "the way, someone addressing a crowd faces the crowd.\n"
            "Never introduce a condition that contradicts one the "
            "narration still holds true: no calm water in a violent "
            "storm, no bright daylight in a narrated night, no empty "
            "street in a narrated crowd. Change a condition only when "
            "the narration says it changed.\n"
            "Do not add details, people, objects, or events that change "
            "the context of the scene the narration describes.\n\n"
            "VISUAL STYLE\n"
            + creative_directions
            + "\n\n"
            "CHANNEL VISUAL RULES\n"
            + visual_rules
            + "\n\n"
            "FINAL OUTPUT\n"
            "Return exactly ONE object with two fields, and nothing "
            "else:\n"
            "- \"search_query\": one distinct, complete SnapGenAI "
            "visual prompt for THIS sentence only, including its "
            "specific scene, subject, action, setting, atmosphere, and "
            "camera framing.\n"
            "- \"visual_direction\": that same shot in one concise "
            "sentence.\n"
        )

        # A re-roll carries the checker's rejection forward, so the
        # second attempt fixes the specific problem instead of
        # resampling the same mistake.
        if feedback:
            prompt += (
                "\n\nYOUR PREVIOUS ANSWER WAS REJECTED\n"
                "An accuracy check read your earlier visual for this "
                "sentence and rejected it for this reason:\n"
                + feedback
                + "\nWrite a NEW search_query that fixes exactly this "
                "problem while still depicting only the marked "
                "sentence. Do not repeat the rejected wording.\n"
            )

        return prompt


    def build_visual_verification_prompt(
        self,
        scripture,
        segments,
        index,
        query
    ):
        """
        Builds the judge prompt for ONE visual against ONE sentence.

        This is the accuracy check that lets generate() reject and
        re-roll its own output: an independent second read that gets
        the passage, the full telling, the marked sentence with its
        neighbours, and the query written for it, and answers one
        structured question - aligned, yes or no, and why not. The
        criteria are the failure modes this channel has actually hit:
        the wrong beat in the chronology, wrong agents, invented
        locations, contradicted conditions, invented events, named
        people left unnamed, and prompts that cannot stand alone.
        """
        channel = self.channel_config.get("channel", {})
        name = str(channel.get("name", "Hear His Voice"))

        segment = segments[index]
        before = (
            segments[index - 1] if index > 0
            else "(nothing - this sentence opens the narration)"
        )
        after = (
            segments[index + 1] if index + 1 < len(segments)
            else "(nothing - this sentence closes the narration)"
        )

        full_telling = "\n".join(
            f"{position + 1}. {text}"
            for position, text in enumerate(segments)
        )

        return (
            "You are the accuracy checker for \""
            + name
            + "\". You read ONE visual prompt that was written for ONE "
            "sentence of the narration, and you decide one thing: does "
            "this visual depict exactly that sentence?\n\n"
            "THE PASSAGE - THE AUTHORITY\n"
            "The passage below is the exact "
            + scripture["translation"]
            + " text and the source of everything. The narration was "
            "written from it and is fixed.\n\n"
            "SCRIPTURE REFERENCE\n"
            + scripture["reference"]
            + " ("
            + scripture["translation"]
            + ")\n\n"
            "THE PASSAGE\n"
            + str(scripture.get("text") or "").strip()
            + "\n\n"
            "THE FULL NARRATION - CONTEXT\n"
            + full_telling
            + "\n\n"
            "THIS SENTENCE - WHAT THE CLIP MUST SHOW\n"
            f"THIS SENTENCE: {segment}\n"
            f"  JUST BEFORE: {before}\n"
            f"  JUST AFTER: {after}\n\n"
            "THE VISUAL PROMPT IN QUESTION\n"
            + str(query or "")
            + "\n\n"
            "JUDGE IT ON THESE POINTS ONLY\n"
            "- Chronology: does it show THIS sentence's moment - not an "
            "earlier beat, and not what happens later in the story?\n"
            "- Agents: does the action THIS sentence narrates happen, "
            "done by the people the passage has doing it? Never accept "
            "one person's action credited to another. Other people the "
            "passage has present in this scene may appear as "
            "bystanders or witnesses - their presence is not a "
            "failure, but an action given to them that this sentence "
            "does not narrate is: showing people from the crowd "
            "climbing into the boat while the sentence sends that "
            "crowd away and tells only the disciples to board fails "
            "here, even though the passage has the crowd present.\n"
            "- Location: are the people where the passage and the "
            "narration put them at this point? Never show a location "
            "the narration does not put them in - no invented shore, "
            "land, or house, and no boarding shown when the narration "
            "has someone already aboard. Resolve every entering and "
            "leaving against the passage's own positions: when the "
            "narration reports a group entering a place the passage "
            "already has them in, the ones who enter are the people "
            "the story has outside it at this point - never the same "
            "people shown climbing in, and never from a shore the "
            "narration never mentions.\n"
            "- Conditions: does it contradict a condition still in "
            "effect - storm, night, rain, crowd, being apart? A "
            "contradiction is a failure.\n"
            "- Invention: does it add people, actions, objects, or "
            "outcomes the PASSAGE does not support? Details the "
            "passage supports even without naming them in this exact "
            "sentence - the setting, the other people present in this "
            "scene, the weather - are legitimate context, not "
            "invention.\n"
            "- Naming: is every person the passage names who appears "
            "in the shot called by that name? A named person reduced "
            "to an unnamed stand-in - 'a lone figure', 'a solitary "
            "figure', or 'a distant man' where the passage has Jesus "
            "- is a failure, even when the scene makes the identity "
            "guessable.\n"
            "- Self-containedness: does it name its people from the "
            "passage instead of leaving an unresolved word like "
            "'they'?\n\n"
            "WHAT IS NOT A FAILURE\n"
            "Do not reject for style, wording, atmosphere, camera "
            "framing, or for carrying context the passage supports. "
            "Reject only for one of the points above. If the visual "
            "faithfully depicts this sentence and nothing on that "
            "list fails, aligned must be true.\n\n"
            "RETURN\n"
            "Return exactly ONE object with two fields, and nothing "
            "else:\n"
            "- \"aligned\": true only if the visual faithfully depicts "
            "THIS sentence on every point above; otherwise false.\n"
            "- \"reason\": when aligned is false, one short sentence "
            "naming exactly what is wrong with the visual; when "
            "aligned is true, an empty string.\n"
            "A reply that is not the JSON object above cannot be "
            "parsed, and an unparsable reply counts as a rejection.\n"
        )


    def verify_visual(
        self,
        scripture,
        segments,
        index,
        query
    ):
        """
        Returns (aligned, reason) for one visual against its sentence.

        Every failure path - a transport error, a reply with no JSON,
        a verdict that is not plainly true - returns aligned=False, so
        the only way a visual passes is an explicit parsed verdict of
        true. Silence or noise can never read as approval.
        """
        prompt = self.build_visual_verification_prompt(
            scripture,
            segments,
            index,
            query
        )

        try:

            response = self.llm.generate(
                prompt,
                response_format=VISUAL_VERIFICATION_SCHEMA
            )

        except Exception as error:

            return (
                False,
                f"The accuracy check itself failed: {error}"
            )

        try:

            verdict = self.parse_direction(response)

        except ContentGenerationError:

            return (
                False,
                "The accuracy check returned no parseable verdict."
            )

        if not isinstance(verdict, dict):

            return (
                False,
                "The accuracy check returned no parseable verdict."
            )

        if verdict.get("aligned") is True:

            return True, ""

        reason = str(
            verdict.get("reason") or ""
        ).strip()

        if not reason:
            reason = (
                "The visual does not faithfully depict the marked "
                "sentence."
            )

        return False, reason


    def build_metadata_prompt(
        self,
        scripture,
        segments
    ):
        """
        Builds the prompt for the episode's title, summary, and mood.

        This is a separate, final call after the visuals are written, so
        it never competes with visual direction and has no way to touch
        a single spoken word or visual. It reads the passage and the
        finished narration and returns only metadata.
        """
        channel = self.channel_config.get("channel", {})
        name = str(channel.get("name", "Hear His Voice"))
        description = str(channel.get("description", ""))

        full_telling = "\n".join(
            f"{position + 1}. {text}"
            for position, text in enumerate(segments)
        )

        return (
            "You are the show-runner for \""
            + name
            + "\", a short-form channel that lets Scripture speak "
            "for itself.\n\n"
            "CRITICAL - YOU DO NOT WRITE THE SCRIPTURE\n"
            "The passage below is the exact "
            + scripture["translation"]
            + " text, and it is the authority for this episode. Never "
            "generate, paraphrase, rewrite, summarise, expand, correct, "
            "or invent any part of the passage or the narration, and do "
            "not echo them back. Return only the title, summary, and "
            "mood described below.\n\n"
            "CHANNEL DESCRIPTION\n"
            + description
            + "\n\nSCRIPTURE REFERENCE\n"
            + scripture["reference"]
            + " ("
            + scripture["translation"]
            + ")\n\n"
            "THE PASSAGE\n"
            + str(scripture.get("text") or "").strip()
            + "\n\n"
            "THE FIXED NARRATION\n"
            + full_telling
            + "\n\n"
            "TITLE RULES\n"
            "The title must be clear, factual, and about what this "
            "passage actually says or does. Write it in plain words.\n"
            "- Name the event, teaching, saying, or story the passage "
            "contains, using the passage's own language.\n"
            "- NEVER invent a poetic, symbolic, or literary concept and "
            "attach it to the passage. No metaphors, no invented 'of ...' "
            "pairings, no made-up themes.\n"
            "- NEVER add a theological claim the passage does not state. "
            "Do not assert meaning, promise, or doctrine that is not in "
            "the quoted text.\n"
            "- Keep it short: a few words, not a sentence.\n"
            "- Do NOT include the Scripture reference. It is added to the "
            "title automatically; do not repeat it yourself.\n"
            "Example - the passage below opens Matthew's genealogy of "
            "Jesus. A correct title is \"The Genealogy of Jesus Christ\". "
            "A WRONG title is \"Genealogy of Light\", which invents a "
            "symbol the passage never mentions.\n\n"
            "SUMMARY RULES\n"
            "The summary is one short sentence for the episode listing.\n"
            "- State plainly what happens or is said in the passage.\n"
            "- Do NOT repeat or restate the title, and do not reuse the "
            "title's distinctive wording.\n"
            "- Do NOT include the Scripture reference. It is already in "
            "the title.\n"
            "- Do NOT interpret, and do not add any claim the passage does "
            "not state. Report the content; do not explain its theology.\n"
            "- Keep it short and useful: one sentence, under 25 words.\n\n"
            "FINAL OUTPUT\n"
            "Return exactly one object with three fields, and nothing "
            "else:\n"
            "- \"title\": a short, clear, factual title, following the "
            "TITLE RULES above. No Scripture reference.\n"
            "- \"summary\": one short factual sentence, following the "
            "SUMMARY RULES above. No reference, no interpretation.\n"
            "- \"mood\": 1-3 lowercase English words for the emotional "
            "tone, used to choose the background music. Choose the style "
            "from the passage: reverent, contemplative, peaceful, solemn, "
            "hopeful, gentle, uplifting, or tender for reflective passages; "
            "suspenseful, tense, storm, fearful, or uncertain when the "
            "passage involves danger, storms, fear, conflict, or uncertainty. "
            "Suspense must remain soft and atmospheric, never upbeat or "
            "aggressive.\n"
        )


    def parse_direction(
        self,
        response
    ):
        """
        Pulls the JSON object out of the model's reply.
        """
        if isinstance(
            response,
            dict
        ):

            return response

        text = str(
            response or ""
        ).strip()

        match = re.search(
            r"\{.*\}",
            text,
            re.DOTALL
        )

        if match is None:

            raise ContentGenerationError(
                "The AI response contained no JSON object."
            )

        try:

            return json.loads(
                match.group(0)
            )

        except Exception as error:

            raise ContentGenerationError(
                f"The AI response was not valid JSON: {error}"
            ) from error

    def assemble_content(
        self,
        scripture,
        segments,
        direction,
        narration=None
    ):
        """
        Joins the AI's visual direction to the telling of the passage.

        The narration is what write_narration() produced - never what
        this call's model returned - so nothing in the visual
        direction can change a single spoken word. The exact WEBC
        passage is kept alongside it as source_text, the authority the
        telling was measured against.
        """
        spoken = str(
            narration or ""
        ).strip()

        if not spoken:

            raise ContentGenerationError(
                "Original narration is required; the WEBC source "
                "cannot be used as a narration fallback."
            )

        if not isinstance(
            direction,
            dict
        ):

            raise ContentGenerationError(
                "The AI returned no usable visual direction."
            )

        raw_visuals = direction.get(
            "visuals",
            []
        )

        if not isinstance(
            raw_visuals,
            list
        ):

            raw_visuals = []

        visuals = []
        for index, segment in enumerate(segments):
            entry = raw_visuals[index] if index < len(raw_visuals) else {}
            if not isinstance(entry, dict):
                entry = {}

            query = str(entry.get("search_query", "")).strip()
            if not query:
                # The fallback still has to belong to this moment of the
                # story: the narration on either side of the segment is
                # handed over with it, so a very short segment borrows
                # its setting from what surrounds it instead of
                # falling back to an unrelated default.
                context = " ".join(
                    neighbour
                    for neighbour in (
                        segments[index - 1] if index > 0 else "",
                        segments[index + 1] if index + 1 < len(segments) else "",
                    )
                    if neighbour
                )
                query = self._fallback_query(segment, context)

            visuals.append({
                "search_query": query,
                "sentence": segment,
            })

        title = str(
            direction.get(
                "title",
                ""
            )
        ).strip()

        reference = scripture["reference"]

        # The model is told not to include the reference, but a model
        # can still append one. Strip it so the reference is added
        # exactly once, by the code below, in one consistent format.
        title = _strip_reference(
            title,
            reference,
        )

        if not title:

            # Nothing usable came back. The passage's own opening words
            # are factual and on-topic, so use them rather than
            # inventing something.
            title = _strip_reference(
                _first_sentence(scripture["text"]),
                reference,
            ) or reference

        # The reference belongs in the title, so the episode is
        # identifiable from the listing without a separate field.
        #
        # The separator is an em dash, and the reference is never
        # wrapped in parentheses: "Title — Book Chapter:Verse" is the
        # format this channel publishes with. A title carries no full
        # stop, because the reference supplies the ending.
        title = "{} — {}".format(
            title.rstrip(" ."),
            reference,
        )

        summary = str(
            direction.get(
                "summary",
                ""
            )
        ).strip()

        # The reference already appears in the title, so a summary that
        # repeats it is redundant. Strip any that leaked in.
        summary = _strip_reference(
            summary,
            reference,
        )

        if not summary or _restates_title(
            summary,
            title,
            reference,
        ):

            # Fall back to the passage's own opening sentence: factual,
            # short, and impossible to invent theology with.
            summary = _first_sentence(
                scripture["text"]
            )

        # No "reference" field. The reference is already in the title,
        # and neither the description nor the caption repeats it, so a
        # second copy on disk would be redundancy with no reader. The
        # other channels publish without one.
        return {
            "title": title,
            "summary": summary,
            "narration": spoken,
            # The exact WEBC wording the telling was written from.
            # It is not spoken or captioned - it is the passage the
            # narration is held to, kept with the episode so what was
            # said can always be checked against the source.
            "source_text": str(
                scripture["text"]
            ).strip(),
            "mood": self._normalize_mood(
                direction.get(
                    "mood",
                    ""
                )
            ),
            "visuals": visuals,
        }

    def _fallback_query(
        self,
        segment,
        context=""
    ):
        """
        A last-resort visual prompt for a segment the AI left blank.
        SnapGenAI produces better results from concrete subjects, so the
        longest content words in the segment are used.

        `context` is the narration immediately before and after the
        segment. A segment of its own can be too short to name its
        setting ("Amen.", "he said."), and the surrounding narration is
        the same moment of the same story - so its place and its
        conditions are borrowed to fill the prompt out rather than
        falling back to an unrelated default.
        """
        stop_words = {
            "the", "a", "an", "and", "or", "but", "is",
            "are", "was", "were", "be", "been", "of",
            "in", "on", "at", "to", "for", "with", "that",
            "this", "these", "those", "it", "he", "she",
            "they", "we", "you", "his", "her", "their",
            "as", "not", "so", "if", "then", "than",
        }

        def content_words(text):

            return [
                word
                for word in re.findall(
                    r"[A-Za-z]{4,}",
                    str(text or "")
                )
                if word.lower()
                not in stop_words
            ]

        words = content_words(segment)

        if len(words) < 3:

            seen = {
                word.lower()
                for word in words
            }

            for word in content_words(context):

                if word.lower() in seen:

                    continue

                words.append(word)
                seen.add(word.lower())

                if len(words) >= 3:

                    break

        return " ".join(
            words[:3]
        ) or "gentle morning light"

    def _normalize_mood(
        self,
        value
    ):
        """
        Music matching wants 1-3 lowercase plain words.
        """
        words = re.findall(
            r"[a-z]+",
            str(value or "").lower()
        )

        unique = []

        for word in words:

            if word not in unique:

                unique.append(word)

        return unique[:3]

    def build_narration_prompt(
        self,
        scripture,
        feedback=None
    ):
        """
        Builds the prompt that tells the passage.

        The passage is quoted in full and named as the authoritative
        source context. The model is asked once for the narration.
        """
        channel = self.channel_config.get(
            "channel",
            {}
        )

        name = str(
            channel.get(
                "name",
                "Hear His Voice"
            )
        )

        description = str(
            channel.get(
                "description",
                ""
            )
        )

        rules = self.format_bullets(
            self.generation_config.get(
                "narration_rules",
                []
            )
        )

        prompt = (
            "You are the storyteller for \""
            + name
            + "\", a short-form channel that tells one passage of "
            + "Scripture per episode as original, engaging storytelling "
            + "through original narration and selected "
            + "verbatim WEBC quotations.\n\n"
            "YOUR TASK, IN TWO PHASES\n"
            "Phase 1 - UNDERSTAND. Read the whole passage first and take "
            "in its events, its meaning, its context, and how its parts "
            "relate to each other: who acts, what happens, what is taught, "
            "and why it matters. Hold that understanding - not the wording "
            "- in mind.\n"
            "Phase 2 - TELL. Tell what you understood as one continuous, "
            "compelling narration that a narrator can read aloud and a "
            "listener can follow by ear. Use original narration for context, "
            "transitions, scene-setting, and less-important details. Preserve "
            "iconic, memorable, emotionally important, or theologically "
            "important sayings verbatim from source_text, especially direct "
            "words of Jesus. Preserve the complete relevant saying from "
            "source_text; never shorten it or paraphrase part of it. "
            "Preserve the exchanges that make someone act: when the "
            "passage shows one character asking and another answering - "
            "a request, a question, a command and its reply - keep "
            "both sides of that exchange whenever the next action "
            "depends on it, so the listener hears the ask before the "
            "doing. Keep the condition a request carries: when someone "
            "asks 'if it is you' or 'if you will', that condition is the "
            "reason the answer matters, so it travels with the request "
            "instead of being trimmed away. Keep the passage's own "
            "agents: whoever the passage says does an action stays the "
            "doer of it - never hand the action to another character "
            "because that order of events seems tidier. "
            "The result must be a purposeful hybrid, "
            "not a complete read-through and not a paraphrase of every line; "
            "your job is not to quote it back wholesale.\n"
            "When you use a direct quote, copy the complete relevant saying "
            "exactly from source_text. "
            "Never invent, modernize, combine, or alter Scripture wording. "
            "Do not return the entire passage by default: select quotations "
            "for their importance and surround them with faithful original "
            "narration.\n\n"
            "CHANNEL DESCRIPTION\n"
            + description
            + "\n\nSCRIPTURE REFERENCE\n"
            + scripture["reference"]
            + " ("
            + scripture["translation"]
            + ")\n\n"
            "THE PASSAGE - THE AUTHORITATIVE SOURCE\n"
            + str(
                scripture.get("text") or ""
            ).strip()
            + "\n\n"
            "NARRATION RULES\n"
            + rules
            + "\n\nSPOKEN LENGTH\n"
            f"Each sentence is spoken over one SnapGenAI clip of "
            f"{_duration_target_seconds(self.config):g} seconds, and the "
            f"cut between clips happens only at a completed sentence "
            f"boundary. Keep every sentence to about "
            f"{_max_words_per_sentence(self.config)} words or fewer so it "
            f"fits that clip: when one thought would run longer, divide it "
            f"into several shorter, natural, source-faithful sentences. A "
            f"short sentence is never padded."
        )

        prompt += (
            "\n\nWORKED EXAMPLES - OF STYLE, FROM OTHER PASSAGES\n"
            "A list carries its names; a story carries its events. The "
            "passage above may be either - tell it as what it is.\n"
            "List: source: \"Abraham became the father of Isaac. "
            "Isaac became the father of Jacob.\"\n"
            "Telling: \"From Abraham came Isaac, and from Isaac "
            "came Jacob.\"\n"
            "The telling keeps every name and relationship and builds "
            "the sentences differently, without forcing a story onto "
            "a list.\n"
            "Story: source: \"Behold, a violent storm came up on the sea, "
            "so much that the boat was covered with the waves, but he "
            "was asleep. They came to him and woke him up, saying, "
            "'Save us, Lord! We are dying!'\"\n"
            "Telling: \"A violent storm rose on the sea until waves "
            "covered the boat, yet he slept. They came and woke him, "
            "crying, 'Save us, Lord! We are dying!'\"\n"
            "The telling keeps every event and preserves the important cry "
            "verbatim, told as one unfolding moment - not clause by clause, "
            "and with nothing added.\n\n"
            "ACCURACY AND PROOFREADING\n"
            "Write clean, correctly spelled content with no typos, accidental punctuation, or malformed words. "
            "Before returning the final output, carefully proofread the entire generated content for spelling and punctuation errors. "
            "Do not introduce accidental changes to quoted or source-derived text.\n\n"
            "RETURN\n"
            "Return JSON with one field:\n"
            "- \"narration\": the finished telling, ready to be spoken "
            "exactly as written - no heading, no label, no notes.\n"
        )

        # A re-write carries the checker's issues forward, so the
        # second telling fixes the specific accuracy problems instead
        # of resampling the same ones.
        if feedback:
            prompt += (
                "\n\nYOUR PREVIOUS NARRATION WAS REJECTED\n"
                "An accuracy check compared your earlier narration "
                "with the passage and found these problems:\n"
                + feedback
                + "\nWrite the telling again, fixing exactly these "
                "problems while following every rule above.\n"
            )

        return prompt

    def build_narration_verification_prompt(
        self,
        scripture,
        narration
    ):
        """
        Builds the judge prompt that reads the finished telling against
        the passage.

        write_narration() uses it to catch the failure modes the
        passage rules exist for - an action handed to the wrong agent,
        a request's condition trimmed away, an exchange whose answer
        erases its question - and re-writes with the issues as
        feedback. The passage is quoted in full; the judge never
        rewrites, only returns a verdict.
        """
        channel = self.channel_config.get("channel", {})
        name = str(channel.get("name", "Hear His Voice"))

        return (
            "You are the accuracy checker for \""
            + name
            + "\". You compare a written narration with the passage "
            "it was told from, and you decide one thing: is the "
            "narration accurate to the passage?\n\n"
            "CRITICAL - YOU DO NOT REWRITE ANYTHING\n"
            "You return only a verdict. The passage below is the "
            "exact "
            + scripture["translation"]
            + " text and the authority.\n\n"
            "SCRIPTURE REFERENCE\n"
            + scripture["reference"]
            + " ("
            + scripture["translation"]
            + ")\n\n"
            "THE PASSAGE - THE AUTHORITY\n"
            + str(scripture.get("text") or "").strip()
            + "\n\n"
            "THE NARRATION IN QUESTION\n"
            + str(narration or "").strip()
            + "\n\n"
            "JUDGE IT ON THESE POINTS ONLY\n"
            "- Agents: does every person in the narration act only as "
            "the passage makes them act? Whoever the passage says does "
            "an action stays the doer of it - an action handed to "
            "another character is a failure, even when the events "
            "themselves are all present.\n"
            "- Conditions: is every condition a request or statement "
            "carries kept with it - 'if it is you', 'if you will' - "
            "instead of being trimmed away? A dropped condition that "
            "changes what the answer means is a failure.\n"
            "- Exchanges: when the passage shows one person asking and "
            "another answering, and the action that follows depends on "
            "it, does the narration keep both sides? An answer that "
            "erases its question is a failure.\n"
            "- Events and order: are the passage's events present and "
            "in the passage's order? Anything the passage states but "
            "the narration drops, and anything the narration has that "
            "the passage never states - invented dialogue, motives, "
            "feelings, events - is a failure.\n"
            "- Quotations: is every verbatim quotation a complete, "
            "faithful saying from the passage, attributed to the "
            "passage's own speaker?\n"
            "- Names and references: does the narration use the "
            "passage's own names and titles, and never speak a "
            "chapter, verse, or reference? An invented title is a "
            "failure.\n\n"
            "RETURN\n"
            "Return exactly ONE object with two fields, and nothing "
            "else:\n"
            "- \"accurate\": true only if the narration passes every "
            "point above; otherwise false.\n"
            "- \"issues\": when accurate is false, a short list of the "
            "exact problems, each one saying what is wrong; when "
            "accurate is true, an empty string.\n"
            "A reply that is not the JSON object above cannot be "
            "parsed, and an unparsable reply counts as a rejection.\n"
        )


    def verify_narration(
        self,
        scripture,
        narration
    ):
        """
        Returns (accurate, issues) for one narration against the
        passage. Same fail-closed contract as verify_visual: only an
        explicit parsed true passes; anything else - transport error,
        no JSON, missing field - is a rejection with a reason.
        """
        prompt = self.build_narration_verification_prompt(
            scripture,
            narration
        )

        try:

            response = self.llm.generate(
                prompt,
                response_format=NARRATION_VERIFICATION_SCHEMA
            )

        except Exception as error:

            return (
                False,
                f"The accuracy check itself failed: {error}"
            )

        try:

            verdict = self.parse_direction(response)

        except ContentGenerationError:

            return (
                False,
                "The accuracy check returned no parseable verdict."
            )

        if not isinstance(verdict, dict):

            return (
                False,
                "The accuracy check returned no parseable verdict."
            )

        if verdict.get("accurate") is True:

            return True, ""

        issues = str(
            verdict.get("issues") or ""
        ).strip()

        if not issues:
            issues = (
                "The narration does not faithfully tell the passage."
            )

        return False, issues


    def parse_narration(
        self,
        response
    ):
        """
        Pulls the telling out of the model's reply.

        The schema asks for JSON, and a reply that is plainly prose is
        still usable. Only an empty or structurally unusable reply is
        rejected.
        """
        if isinstance(
            response,
            dict
        ):

            narration = _clean_narration(
                response.get(
                    "narration",
                    ""
                )
            )

            if not narration:

                raise ContentGenerationError(
                    "The AI returned no narration."
                )

            return narration

        text = str(
            response or ""
        ).strip()

        if not text:

            raise ContentGenerationError(
                "The AI returned an empty reply."
            )

        if text.startswith("{"):

            match = re.search(
                r"\{.*\}",
                text,
                re.DOTALL
            )

            if match is None:

                raise ContentGenerationError(
                    "The AI reply contained no JSON object."
                )

            try:

                data = json.loads(
                    match.group(0)
                )

            except Exception as error:

                raise ContentGenerationError(
                    f"The AI reply was not valid JSON: {error}"
                ) from error

            if not isinstance(
                data,
                dict
            ):

                raise ContentGenerationError(
                    "The AI reply was not a JSON object."
                )

            narration = _clean_narration(
                data.get(
                    "narration",
                    ""
                )
            )

            if not narration:

                raise ContentGenerationError(
                    "The AI reply had no \"narration\" field."
                )

            return narration

        # Plain prose rather than JSON - usable as it stands.
        return _clean_narration(text)

    def write_narration(
        self,
        scripture
    ):
        """
        Calls the LLM for narration, then checks it against the
        passage. The WEBC passage is context and authority only; it
        is never a narration fallback. The returned narration is used
        as-is after transport parsing. A narration the accuracy check
        rejects is re-written with the issues as feedback and checked
        again, repeating until the check passes - a rejected
        narration never ships (set content_generation
        .verify_narration to false to skip the check).
        """
        verify = bool(
            self.generation_config.get("verify_narration", True)
        )

        self.log("Writing the narration...")

        feedback = None
        narration = ""

        while True:

            prompt = self.build_narration_prompt(
                scripture,
                feedback=feedback
            )

            narration = self.parse_narration(
                self.llm.generate(
                    prompt,
                    response_format=NARRATION_SCHEMA
                )
            )

            narration = repair_source_word_punctuation(
                scripture.get("text", ""),
                narration,
            )

            if not verify:
                break

            accurate, issues = self.verify_narration(
                scripture,
                narration
            )

            if accurate:
                break

            self.log(
                f"Narration rejected ({issues}) - writing it "
                "again..."
            )
            feedback = issues

        self.log(
            f"Narration ready ({len(narration.split())} words)."
        )

        return narration

    def generate(
        self,
        scripture,
        segments,
        narration=None
    ):
        """
        Returns the episode content dict.

        The narration is the telling this call is handed, never this
        call's own model output: the model here directs the visuals,
        the title, the summary, and the mood, and nothing else.

        Visuals are directed one narration sentence at a time. Each
        segment gets its own focused call - showing the full passage,
        the full narration, and that sentence's immediate neighbours -
        so every prompt depicts exactly its own beat instead of the
        model collapsing the whole episode into a few repeated
        establishing shots. Every finished visual is then checked
        against its own sentence by a separate verification call; a
        rejected visual is re-rolled with the checker's reason as
        feedback and checked again, repeating until the check passes
        - a rejected visual never ships (set
        content_generation.verify_visuals to false to skip the check).
        A final separate call writes the title, summary, and mood.
        assemble_content() then pairs each collected visual with its
        own sentence, in order.
        """
        self.log(
            f"Generating {len(segments)} visuals, one call per "
            "sentence..."
        )

        verify = bool(
            self.generation_config.get("verify_visuals", True)
        )

        visuals = []
        for index in range(len(segments)):
            feedback = None
            parsed = {}

            while True:

                prompt = self.build_segment_prompt(
                    scripture,
                    segments,
                    index,
                    feedback=feedback
                )

                response = self.llm.generate(
                    prompt,
                    response_format=VISUAL_SCHEMA
                )

                parsed = self.parse_direction(response)
                if not isinstance(parsed, dict):
                    parsed = {}

                if not verify:
                    break

                aligned, reason = self.verify_visual(
                    scripture,
                    segments,
                    index,
                    str(parsed.get("search_query", ""))
                )

                if aligned:
                    break

                self.log(
                    f"Sentence {index + 1}: rejected ({reason}) - "
                    "re-rolling..."
                )
                feedback = reason

            visuals.append(parsed)

        self.log("Generating metadata (title, summary, mood)...")

        metadata_prompt = self.build_metadata_prompt(
            scripture,
            segments
        )

        metadata_response = self.llm.generate(
            metadata_prompt,
            response_format=METADATA_SCHEMA
        )

        metadata = self.parse_direction(metadata_response)
        if not isinstance(metadata, dict):
            metadata = {}

        direction = {
            "title": metadata.get("title", ""),
            "summary": metadata.get("summary", ""),
            "mood": metadata.get("mood", ""),
            "visuals": visuals,
        }

        return self.assemble_content(
            scripture,
            segments,
            direction,
            narration
        )


def _clean_narration(text):
    """
    Tidies what the model returned into plain spoken prose: strips the
    code fences, labels, and wrapping quotes a model likes to add, and
    flattens paragraph breaks into spaces so the segments, the TTS and
    the captions all see one continuous piece of narration.
    """
    value = str(
        text or ""
    ).strip()

    value = re.sub(
        r"^```[a-zA-Z]*\s*",
        "",
        value,
    )

    value = re.sub(
        r"\s*```$",
        "",
        value,
    ).strip()

    value = re.sub(
        r"^\s*(?:narration|story|telling)\s*:\s*",
        "",
        value,
        flags=re.IGNORECASE,
    ).strip()

    if (
        len(value) > 1
        and value[0] == value[-1]
        and value[0] in "\"'"
    ):

        value = value[1:-1].strip()

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def repair_source_word_punctuation(source_text, narration):
    """
    Repairs punctuation inserted inside source-derived words.

    This is deliberately narrower than spellchecking. A narration token is
    changed only when its letters, ignoring punctuation, exactly match one
    source token's letters, and the narration token contains internal
    punctuation that the source token does not. The narration's surrounding
    punctuation and all unrelated wording remain unchanged.

    For example, ``bapt,ize`` is restored to ``baptize`` because the exact
    source text contains ``baptize``. A word that is not present in the
    source is never changed.
    """
    source_tokens = str(source_text or "").split()
    narration_tokens = str(narration or "").split()

    source_forms = {}
    for token in source_tokens:
        match = _word_token_parts(token)
        if match is None:
            continue

        _leading, core, _trailing = match
        key = _source_word_key(core)
        if not key:
            continue

        source_forms.setdefault(key, set()).add(core)

    repaired = []
    for token in narration_tokens:
        match = _word_token_parts(token)
        if match is None:
            repaired.append(token)
            continue

        leading, core, trailing = match
        key = _source_word_key(core)
        candidates = source_forms.get(key, set())

        if (
            len(candidates) == 1
            and core != next(iter(candidates))
            and any(not char.isalnum() for char in core)
        ):
            source_core = next(iter(candidates))
            repaired.append(leading + source_core + trailing)
        else:
            repaired.append(token)

    return " ".join(repaired)


def _word_token_parts(token):
    """Returns leading punctuation, word core, and trailing punctuation."""
    match = re.match(
        r"^([^A-Za-z0-9]*)([A-Za-z0-9][A-Za-z0-9,'’\-]*[A-Za-z0-9]|[A-Za-z0-9]+)([^A-Za-z0-9]*)$",
        str(token),
    )
    return match.groups() if match else None


def _source_word_key(value):
    """Normalizes a word only for source-token comparison."""
    return re.sub(r"[^A-Za-z0-9]", "", str(value)).casefold()


def _first_sentence(text):
    """
    The opening sentence of a passage, trimmed to something short enough
    to stand as a title or a summary on its own.

    Used only as a fallback when the model returns nothing usable, so
    the fallback is taken verbatim from the WEBC text rather than
    invented.
    """

    cleaned = str(
        text or ""
    ).strip()

    if not cleaned:

        return ""

    for mark in (". ", "? ", "! "):

        head, _sep, _tail = cleaned.partition(mark)

        if head.strip():

            return head.strip() + mark.strip()

    return cleaned


def _strip_reference(text, reference):
    """
    Removes a Scripture reference from a piece of text, wherever the
    model chose to put it.

    The reference is added to the title by the code, in one place and one
    format. Anything the model supplied is duplication, so it is removed
    to keep the title from reading "Foo (Matthew 1:1) (Matthew 1:1-9)".

    Only the punctuation disturbed by the removal is tidied. Sentence
    punctuation belonging to the text is left alone - a summary must
    keep its full stop.
    """

    value = str(
        text or ""
    ).strip()

    if not value or not reference:

        return value

    # A bracketed or trailing form, e.g. "Foo (Matthew 1:1-9)".
    for opening, closing in (("(", ")"), ("[", "]")):

        value = value.replace(
            f"{opening}{reference}{closing}",
            "",
        )

    # A bare occurrence, e.g. "Matthew 1:1-9 - Foo", or
    # "Matthew 1:1-9 lists ...".
    value = value.replace(
        reference,
        "",
    )

    # Tidy only what the removal disturbed.
    value = re.sub(r"\s{2,}", " ", value)
    value = re.sub(r"^[\s\-–—,;:]+", "", value)
    value = re.sub(r"[\s\-–—,:]+$", "", value)
    value = value.strip()

    # Removing the reference can leave a clause that starts mid-sentence
    # ("lists the ancestors of Jesus"). Restore the capital.
    if value[:1].islower():

        value = value[0].upper() + value[1:]

    return value


def _restates_title(summary, title, reference):
    """
    True when the summary is really just the title again.

    The prompt asks for a summary that adds something the title does not
    say, but a model asked twice will sometimes echo it back verbatim.
    That is worth catching here: shipping a listing where the title and
    the description say the same thing wastes the space.

    The test is deliberately narrow - high word overlap - because the
    opposite mistake is worse. A summary that legitimately shares the
    title's vocabulary and then adds detail ("The genealogy of Jesus
    Christ lists the ancestors from Abraham down to David") must be
    kept. Only an almost-verbatim echo counts.

    It cannot catch a paraphrase that swaps the words around ("lineage"
    for "genealogy"); that is left to the prompt.
    """

    def content_words(value):

        return {
            word
            for word in re.findall(
                r"[a-z0-9]+",
                str(value or "").lower(),
            )
            if len(word) > 3
        }

    summary_words = content_words(summary)

    if not summary_words:

        return True

    title_words = content_words(
        _strip_reference(
            title,
            reference,
        )
    )

    if not title_words:

        return False

    shared = summary_words & title_words
    every = summary_words | title_words

    return (len(shared) / len(every)) >= 0.8


def write_content_files(
    episode_directory,
    content
):
    """
    Saves the episode content. `content.json` is the source of truth
    for the video stage; `prompt.txt` is kept for reference.
    """
    episode_directory = Path(episode_directory)
    episode_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    content_path = episode_directory / "content.json"

    with open(
        content_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            content,
            file,
            indent=2,
            ensure_ascii=False
        )

    divider = "=" * 72

    # TITLE, PROMPT, and SUMMARY must all be in the same section
    # between the dividers, because the web UI parses this file by
    # splitting on the divider and looking for those markers.
    #
    # There is deliberately no REFERENCE line. The reference already
    # sits in the TITLE line, and neither published caption repeats it,
    # so a third copy here would only be able to drift out of step.
    lines = [
        divider,
        f"TITLE: {content['title']}",
        f"PROMPT: {content['narration']}",
        f"SUMMARY: {content['summary']}",
        divider,
        ""
    ]

    prompt_path = episode_directory / "prompt.txt"

    prompt_path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    return content_path


def read_content_file(
    episode_directory
):
    """
    Reads back the saved episode content, or None when the episode has
    not reached the content stage.
    """
    content_path = (
        Path(episode_directory)
        / "content.json"
    )

    if not content_path.is_file():

        return None

    with open(
        content_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)