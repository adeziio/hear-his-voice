import json
import math
import re

from pathlib import Path

from ai.base_ai_service import BaseAIService
from ai.providers.ollama_provider import OllamaProvider

from scripture.segmenter import (
    split_sentences,
    build_visual_segments
)


# `target_duration_seconds` is only a target for narration generation. Visual
# timing is derived from the narration that was actually generated.
SECONDS_PER_VISUAL = 8
DEFAULT_WORDS_PER_SECOND = 2.9

# A segment is never smaller than one, however short the target.
MIN_SEGMENTS_PER_EPISODE = 1


def segments_per_episode(config, narration=None):
    """
    Returns the approximate number of visual segments for the generated
    narration, using one visual for about eight seconds of speech.

    The configured target duration is deliberately not used here: it is only
    an instruction for the narration writer, while the generated narration is
    the source of truth for the episode's actual length.
    """
    words_per_second = float(
        config.get("content", {})
        .get("content_generation", {})
        .get("words_per_second", DEFAULT_WORDS_PER_SECOND)
    )
    words_per_second = max(words_per_second, 0.1)
    word_count = len(str(narration or "").split())
    expected_duration = word_count / words_per_second

    return max(
        MIN_SEGMENTS_PER_EPISODE,
        int(math.ceil(expected_duration / SECONDS_PER_VISUAL)),
    )


def build_direction_schema(segment_count):
    """Build the visual schema for the exact number of narration segments."""
    schema = json.loads(json.dumps(DIRECTION_SCHEMA))
    schema["properties"]["visuals"]["minItems"] = segment_count
    schema["properties"]["visuals"]["maxItems"] = segment_count
    return schema


class ContentGenerationError(
    RuntimeError
):

    pass


def format_segments_with_context(segments):
    """
    Lists each narration segment together with the narration that comes
    immediately before and after it.

    A visual has to be directed at one exact moment of the story, and a
    segment on its own rarely says where that moment sits: "they came to
    him and woke him" looks completely different depending on whether the
    storm has already risen and the disciples are already afraid. The
    neighbouring narration is what tells the director what has just
    happened and what happens next, so the shot written for the segment
    belongs to that moment instead of to the passage in general, and so
    place, position, and ongoing conditions can be carried from one
    visual into the next.

    The context lines are indented labels, never SEGMENT headers: the
    number of "SEGMENT n:" lines is still exactly the number of
    segments, so the exact-count direction schema and the one-visual-
    per-segment rule are untouched.
    """
    blocks = []

    total = len(segments)

    for index, segment in enumerate(segments):

        if index == 0:
            before = (
                "(nothing - this segment opens the narration)"
            )
        else:
            before = segments[index - 1]

        if index + 1 >= total:
            after = (
                "(nothing - this segment closes the narration)"
            )
        else:
            after = segments[index + 1]

        blocks.append(
            f"SEGMENT {index + 1}: {segment}\n"
            f"  JUST BEFORE: {before}\n"
            f"  JUST AFTER: {after}"
        )

    return "\n".join(blocks)


# The AI directs the episode; it does not author the Scripture.
#
# This is the visual-direction schema, and it has no narration field at
# all: the model returns one overall-passage search_query plus a
# visual_direction, together with the title, summary, and mood. The
# narration itself is written by write_narration() under the separate
# NARRATION_SCHEMA below, from the WEBC passage as its only source, so
# neither call gives the model anywhere to put invented Scripture.
DIRECTION_SCHEMA = {
    "type": "object",
    "properties": {
        # maxLength is not decoration. It is a grammar-level brake on
        # the two failure modes this channel actually hit: a poetic
        # title ("Genealogy of Light") and an interpretive summary
        # ("revealing a divine ancestry rooted in faith and promise"
        # for a passage that says no such thing).
        "title": {
            "type": "string",
            "maxLength": 90
        },
        "summary": {
            "type": "string",
            "maxLength": 180
        },
        "mood": {
            "type": "string"
        },
        "visuals": {
            "type": "array",
            "minItems": 1,
            "maxItems": 100,
            "items": {
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string"
                    },
                    "visual_direction": {
                        "type": "string"
                    }
                },
                "required": [
                    "search_query",
                    "visual_direction"
                ]
            }
        }
    },
    "required": [
        "title",
        "summary",
        "mood",
        "visuals"
    ]
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
        2. splits that narration into spoken segments
           (deterministic),
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
        Splits the generated narration into visual segments. The narration
        itself determines the expected episode duration and therefore the
        approximate number of visuals; the configured target is not used as
        the final-duration assumption.
        """
        text = scripture["text"]

        # How fast the narrator reads. It lives in content.json with the
        # rest of the Scripture settings, so there is one content config.
        words_per_second = float(
            self.config.get(
                "content",
                {}
            ).get(
                "content_generation",
                {}
            )
            .get(
                "words_per_second",
                2.9
            )
        )

        segment_count = segments_per_episode(self.config, text)
        words_per_segment = max(
            1,
            int(round(words_per_second * SECONDS_PER_VISUAL)),
        )

        segments = build_visual_segments(
            text,
            segment_count,
            clause_words=words_per_segment,
            max_words_per_segment=words_per_segment + 6,
        )

        self.log(
            f"{len(segments)} visual segments for "
            f"{len(text.split())} words "
            f"(~{len(text.split()) / max(words_per_second, 0.1):.0f}s "
            f"of narration, target {SECONDS_PER_VISUAL}s per segment)."
        )

        return segments

    def build_prompt(
        self,
        scripture,
        segments
    ):
        """
        Builds the visual-direction prompt.

        The Scripture is quoted so the AI understands what each segment
        is about, but it is told explicitly that the wording is fixed
        and that it must only answer with direction.
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

        visual_rules = self.format_bullets(
            self.generation_config.get(
                "visual_rules",
                []
            )
        )

        creative_directions = self.format_bullets(
            self.generation_config.get(
                "creative_directions",
                []
            )
        )

        return (
            "You are the visual director for \""
            + name
            + "\", a short-form channel that lets Scripture speak "
            "for itself.\n\n"
            "CRITICAL - YOU DO NOT WRITE THE SCRIPTURE\n"
            "The passage below is the exact "
            + scripture["translation"]
            + " text, and it is the authority for this episode. The "
            "narration was already written from that passage and is "
            "fixed for this call: the narrator speaks that telling "
            "exactly as written and it is captioned exactly as spoken. You must NEVER "
            "generate, paraphrase, rewrite, summarise, expand, "
            "correct, or invent any part of the passage, and you must "
            "not echo it back in your answer. Your only job is to "
            "decide what the viewer should SEE while those words are "
            "spoken.\n\n"
            "CHANNEL DESCRIPTION\n"
            + description
            + "\n\nSCRIPTURE REFERENCE\n"
            + scripture["reference"]
            + " ("
            + scripture["translation"]
            + ")\n\n"
            "THE PASSAGE - THE SOURCE OF EVERYTHING SPOKEN\n"
            + str(
                scripture.get("text") or ""
            ).strip()
            + "\n\n"
            "THE FIXED NARRATION, IN VISUAL SEGMENTS\n"
            "Each segment is shown with the narration immediately around "
            "it, so you can see what has just happened and what happens "
            "next at that point in the story. The segment line is what "
            "you direct; the two indented lines are context for that "
            "moment only - they are not separate visuals.\n"
            + format_segments_with_context(segments)
            + "\n\nVISUAL PROMPT RULES\n"
            "Return exactly one visual object for each numbered narration "
            "segment above, in the same order. Each search_query will be "
            "sent to SnapGenAI for that segment's clip. The prompt must "
            "depict the specific sentence, action, event, setting, and "
            "emotional moment narrated in that segment. Keep related "
            "sentences and events together, and make the visuals progress "
            "naturally from the beginning to the end of the story. Do not "
            "reuse one generic overall-passage prompt. Make each prompt "
            "straightforward and descriptive: identify the main scene, "
            "subject, setting, action, and atmosphere. Begin with 'Style: "
            "realistic.' Do not write a stock-footage search query, "
            "narration, explanation, or list.\n"
            "Read the whole narration and the context lines before "
            "writing a prompt. Direct each prompt at that segment's exact "
            "moment in the story - what the viewer should see while these "
            "words are spoken, given what has JUST happened and what "
            "happens NEXT - rather than at the passage in general.\n"
            "Keep physical and spatial continuity between related "
            "visuals: when consecutive segments happen in the same place "
            "and situation, keep the same location, the same positions, "
            "and the same circumstances in each prompt, and change them "
            "only where the narration itself changes them - a move, an "
            "entrance, a departure, or a change of time or weather.\n"
            "Place people exactly where the narration places them. If "
            "someone is outside a place - outside the tomb, on the shore, "
            "at a distance, in the boat, before a closed door - keep them "
            "outside it unless the narration says they entered. Never "
            "move a scene indoors, into a city, or onto a road the "
            "narration never mentions.\n"
            "Depict each action in a physically plausible position for "
            "that action: someone asleep lies down, someone praying "
            "kneels or stands with head bowed, someone travelling is on "
            "their feet on the way, someone addressing a crowd faces the "
            "crowd.\n"
            "Carry ongoing conditions across the related visuals - storm, "
            "darkness, night, rain, wind, waves, crowds, water, fire, "
            "heat, cold - and never introduce a condition that "
            "contradicts one the narration still holds true: no calm "
            "water in a violent storm, no bright daylight in a narrated "
            "night, no empty street in a narrated crowd. Change a "
            "condition only when the narration says it changed.\n"
            "Do not add details, people, objects, or events that change "
            "the context of the scene the narration describes.\n"
            + visual_rules
            + "\n\nVISUAL STYLE\n"
            + creative_directions
            + "\n\nFINAL OUTPUT\n"
            "Return exactly one object in \"visuals\" for every numbered "
            "segment, in order. Each object's \"search_query\" must be a "
            "distinct, complete SnapGenAI visual prompt for that segment, "
            "including its specific scene, subject, action, setting, and "
            "atmosphere. Consecutive prompts must also read as one "
            "continuous story - same place, same people, same ongoing "
            "conditions - wherever the narration keeps them there. The "
            "object's \"visual_direction\" should repeat "
            "that same segment shot in one concise sentence.\n"
            + "\n\nTITLE RULES\n"
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
        "symbol the passage never mentions.\n"
        + "\n\nSUMMARY RULES\n"
        "The summary is one short sentence for the episode listing.\n"
        "- State plainly what happens or is said in the passage.\n"
        "- Do NOT repeat or restate the title, and do not reuse the "
        "title's distinctive wording.\n"
        "- Do NOT include the Scripture reference. It is already in "
        "the title.\n"
        "- Do NOT interpret, and do not add any claim the passage does "
        "not state. Report the content; do not explain its theology.\n"
        "- Keep it short and useful: one sentence, under 25 words.\n"
        "\nAlso return:\n"
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
        scripture
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
            "source_text; never shorten it or paraphrase part of it. The "
            "result must be a purposeful hybrid, "
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

        return prompt

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
        Calls the LLM once for narration. The WEBC passage is context
        and authority only; it is never a narration fallback. The
        returned narration is used as-is after transport parsing.
        """
        prompt = self.build_narration_prompt(scripture)

        self.log("Writing the narration...")

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
        """
        prompt = self.build_prompt(
            scripture,
            segments
        )

        self.log(
            "Generating visual direction..."
        )

        response = self.llm.generate(
            prompt,
            response_format=build_direction_schema(len(segments))
        )

        return self.assemble_content(
            scripture,
            segments,
            self.parse_direction(response),
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