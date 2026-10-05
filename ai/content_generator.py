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
        "title": {
            "type": "string"
        },
        "summary": {
            "type": "string"
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
            + visual_rules
            + "\n\nVISUAL DIRECTION\n"
            + creative_directions
            + "\n\nFINAL OUTPUT\n"
            "Return exactly "
            + str(len(segments))
            + " objects in \"visuals\", one per segment, in the same "
            "order as the segments above. Each object has exactly two "
            "fields:\n"
            "- \"search_query\": a short, practical Pexels stock "
            "footage search phrase (2-5 words) for that segment.\n"
            "- \"visual_direction\": one sentence describing the shot "
            "the viewer should see, in the channel's reverent "
            "cinematic style.\n"
            "Also return:\n"
            "- \"title\": a short, reverent, properly punctuated "
            "episode title.\n"
            "- \"summary\": one complete sentence describing the "
            "passage for the episode listing.\n"
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

        if not title:

            title = reference

        # The reference belongs in the title, so the episode is
        # identifiable from the listing without a separate field.
        if reference not in title:

            title = f"{title} ({reference})"

        summary = str(
            direction.get(
                "summary",
                ""
            )
        ).strip()

        if not summary:

            summary = f"{reference}."

        return {
            "title": title,
            "reference": reference,
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

# IMPORTANT: TITLE, REFERENCE, PROMPT, and SUMMARY must all be in
    # the same section between the dividers, because the web UI parses
    # this file by splitting on the divider and looking for those
    # markers. REFERENCE carries the passage that was read, and is reused
    # verbatim by the YouTube and Instagram descriptions.
    lines = [
        divider,
        f"TITLE: {content['title']}",
        f"REFERENCE: {content.get('reference', '')}",
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