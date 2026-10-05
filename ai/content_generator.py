import json
import re

from pathlib import Path

from ai.base_ai_service import BaseAIService
from ai.providers.ollama_provider import OllamaProvider

from scripture.segmenter import (
    split_sentences,
    build_visual_segments
)


# The only duration setting is app.json -> shorts.target_duration_seconds.
# How long each visual should cover is the one pacing constant: 60 seconds
# divided by this gives the 14 segments all three projects were tuned to, so
# changing the target duration moves the segment count with it and the two
# numbers can never disagree.
SECONDS_PER_VISUAL = 60 / 14

DEFAULT_TARGET_SECONDS = 60

# A segment is never smaller than one, however short the target.
MIN_SEGMENTS_PER_EPISODE = 1


def segments_per_episode(config):
    """
    How many spoken segments - and therefore how many visuals - an episode
    asks for, derived from the target duration alone: one visual per
    SECONDS_PER_VISUAL of narration.

    This is a guide, not a hard limit. The narration is the verbatim WEBC
    text, so it cannot be padded or trimmed to hit an exact count - the
    passage's own punctuation decides the final number of segments. All
    this sets is how finely that passage is cut.
    """
    try:

        target_seconds = float(
            config.get(
                "app",
                {}
            ).get(
                "shorts",
                {}
            ).get(
                "target_duration_seconds",
                DEFAULT_TARGET_SECONDS
            )
        )

    except (TypeError, ValueError):

        target_seconds = DEFAULT_TARGET_SECONDS

    return max(
        MIN_SEGMENTS_PER_EPISODE,
        int(round(target_seconds / SECONDS_PER_VISUAL))
    )


class ContentGenerationError(
    RuntimeError
):

    pass


# The AI directs the episode; it does not author the Scripture.
#
# The schema has no narration field at all. The narration is the exact
# WEBC text supplied by scripture/, and the model only returns one
# search_query plus a visual_direction per spoken segment, together
# with the title, summary, and mood. There is therefore no field the
# model could use to rewrite, paraphrase, or invent Bible text.
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


class ContentGenerator(
    BaseAIService
):

    """
    The creative director for a Hear His Voice episode.

    Given the exact WEBC passage, this:

        1. splits the passage into spoken segments (deterministic),
        2. asks the AI for a Pexels search query and a visual
           direction for each segment,
        3. asks for a title, summary, and mood.

    The AI never writes the narration. The narration is always the
    verbatim WEBC text that the Scripture source returned.
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
        Splits the exact WEBC text into spoken segments, each of
        which gets its own Pexels query and visual direction.

        The count follows the passage's own length, aiming at one
        visual every few seconds and never letting a single visual
        cover more than about ten seconds of narration. Long sentences
        are broken at their own punctuation, so no visual is stuck
        holding an entire verse.
        """
        text = scripture["text"]

        # How fast the narrator reads. It lives in content.json with the
        # rest of the Scripture settings, so there is one content config.
        words_per_second = float(
            self.config.get(
                "content",
                {}
            ).get(
                "scripture",
                {}
            )
            .get(
                "words_per_second",
                2.9
            )
        )

        # The episode's target length is the only input, exactly as in
        # the other project: total spoken words come from it, and the
        # per-segment size is that total divided by the number of
        # segments. There is no separate "seconds per visual" setting.
        shorts_config = (
            self.config.get(
                "app",
                {}
            ).get(
                "shorts",
                {}
            )
        )

        target_seconds = float(
            shorts_config.get(
                "target_duration_seconds",
                50
            )
        )

        segment_count = max(
            1,
            segments_per_episode(self.config)
        )

        word_target = int(
            target_seconds
            * words_per_second
        )

        words_per_segment = max(
            1,
            word_target // segment_count
        )

        segments = build_visual_segments(
            text,
            segment_count,
            clause_words=words_per_segment,
            max_words_per_segment=words_per_segment + 3,
        )

        self.log(
            f"{len(segments)} visual segments for "
            f"{len(text.split())} words "
            f"(~{len(text.split()) / max(words_per_second, 0.1):.0f}s "
            f"of narration, {words_per_segment} words per segment)."
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

        segment_lines = "\n".join(
            f"{index}. SEGMENT: {segment}\n"
            f"   Return exactly one visuals entry for this segment."
            for index, segment in enumerate(
                segments,
                start=1
            )
        )

        return (
            "You are the visual director for \""
            + name
            + "\", a short-form channel that lets Scripture speak "
            "for itself.\n\n"
            "CRITICAL - YOU DO NOT WRITE THE SCRIPTURE\n"
            "The narration is already fixed. It is the exact text of "
            "the "
            + scripture["translation"]
            + ", quoted below, and it will be read verbatim by a "
            "narrator and captioned verbatim on screen. You must NEVER "
            "generate, paraphrase, rewrite, summarise, expand, "
            "correct, or invent any part of it, and you must not echo "
            "it back in your answer. Your only job is to decide what "
            "the viewer should SEE while those exact words are "
            "spoken.\n\n"
            "CHANNEL DESCRIPTION\n"
            + description
            + "\n\nSCRIPTURE REFERENCE\n"
            + scripture["reference"]
            + " ("
            + scripture["translation"]
            + ")\n\n"
            "THE FIXED NARRATION, SPLIT INTO "
            + str(len(segments))
            + " SPOKEN SEGMENTS\n"
            + segment_lines
            + "\n\nVISUAL SEARCH QUERY RULES\n"
            "Each search_query is a Pexels search phrase. The viewer "
            "hears this segment's words while watching the footage it "
            "returns, so every query must support the meaning of its "
            "own segment.\n"
            + visual_rules
            + "\n\nVISUAL DIRECTION\n"
            + creative_directions
            + "\n\nFINAL OUTPUT\n"
            "Return exactly "
            + str(len(segments))
            + " objects in \"visuals\", one per segment, in the same "
            "order as the segments above. Each object has exactly two "
            "fields:\n"
            "- \"search_query\": a Pexels stock footage search phrase "
            "(2-5 words) chosen because it SUPPORTS THE MEANING of that "
            "segment - its theme, setting, era or mood. It must be a "
            "phrase that genuinely returns usable reverent stock "
            "footage, and it must never be a literal attempt to film a "
            "biblical person, a family relationship, or an event.\n"
            "- \"visual_direction\": one sentence describing the shot "
            "the viewer should see, in the channel's reverent "
            "cinematic style.\n"
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
            "tone, used to choose the background music (for example "
            "reverent, hopeful, solemn, gentle, uplifting, tender).\n"
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
        direction
    ):
        """
        Joins the AI's visual direction to the exact WEBC text.

        The narration and its sentences come straight from the
        Scripture source. Nothing the model returned can change them.
        """
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

        for index, segment in enumerate(
            segments
        ):

            entry = (
                raw_visuals[index]
                if index < len(raw_visuals)
                else {}
            )

            if not isinstance(
                entry,
                dict
            ):

                entry = {}

            query = str(
                entry.get(
                    "search_query",
                    ""
                )
            ).strip()

            # Every segment needs a usable query for the footage
            # stage, so a blank one falls back to content words from
            # the segment rather than aborting the episode.
            if not query:

                query = self._fallback_query(
                    segment
                )

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
            "narration": scripture["text"],
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
        segment
    ):
        """
        A last-resort search query for a segment the AI left blank.
        Pexels matches on concrete nouns, so the longest content words
        in the segment are used.
        """
        stop_words = {
            "the", "a", "an", "and", "or", "but", "is",
            "are", "was", "were", "be", "been", "of",
            "in", "on", "at", "to", "for", "with", "that",
            "this", "these", "those", "it", "he", "she",
            "they", "we", "you", "his", "her", "their",
            "as", "not", "so", "if", "then", "than",
        }

        words = [
            word
            for word in re.findall(
                r"[A-Za-z]{4,}",
                str(segment)
            )
            if word.lower()
            not in stop_words
        ]

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

    def generate(
        self,
        scripture,
        segments
    ):
        """
        Returns the episode content dict. The narration and its
        sentences are the verbatim WEBC text, never model output.
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
            response_format=DIRECTION_SCHEMA
        )

        return self.assemble_content(
            scripture,
            segments,
            self.parse_direction(response)
        )


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