import json
import math
import random
import re

from ai.base_ai_service import BaseAIService
from ai.providers.ollama_provider import OllamaProvider


class PromptGenerator(
    BaseAIService
):

    def __init__(
        self,
        config
    ):

        super().__init__(
            config,
            "PROMPT"
        )

        self.llm = OllamaProvider(
            config
        )

        content_config = config["content"]

        self.channel_config = (
            content_config
        )

        self.prompt_config = (
            content_config.get(
                "prompt_generation",
                {}
            )
        )

        self.recent_concepts = []

    def pick_episode_style(self):

        styles = [
            str(value).strip()
            for value in self.prompt_config.get(
                "style_rotation",
                []
            )
            if str(value).strip()
        ]

        if not styles:

            return ""

        return random.choice(styles)

    def get_generation_attempts(self):

        # A concept only survives validation when the selected Scripture
        # text appears verbatim inside the prompt. The local model is
        # sampled at a high temperature, so a single attempt can come
        # back paraphrased; retry a few times before failing the job.

        attempts = self.prompt_config.get(
            "generation_attempts",
            3
        )

        try:

            attempts = int(
                attempts
            )

        except (TypeError, ValueError):

            attempts = 3

        return max(
            1,
            attempts
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

    def build_shared_context_sections(self):

        genre = self.channel_config.get(
            "genre",
            "visually striking short-form entertainment"
        )

        tone = self.channel_config.get(
            "tone",
            []
        )

        world = self.channel_config.get(
            "world",
            []
        )

        protagonists = self.channel_config.get(
            "protagonists",
            []
        )

        creative_engines = self.channel_config.get(
            "creative_engines",
            []
        )

        instructions = self.prompt_config.get(
            "instructions",
            []
        )

        creative_directions = self.prompt_config.get(
            "creative_directions",
            []
        )

        priorities = self.prompt_config.get(
            "creative_priorities",
            []
        )

        diversity = self.prompt_config.get(
            "diversity_guidance",
            []
        )

        return f"""GENRE
{genre}

MOOD AND TONE (rotate naturally between concepts)
{self.format_bullets(tone)}

ENVIRONMENTS (critical - the setting must add visual appeal)
{self.format_bullets(world)}

POSSIBLE SUBJECTS
{self.format_bullets(protagonists)}

CREATIVE ENGINES
{self.format_bullets(creative_engines)}

RECENT CONCEPTS TO AVOID REPEATING
{self.get_recent_concepts_text()}

RULES
{self.format_bullets(instructions)}

CREATIVE FREEDOM
{self.format_bullets(creative_directions)}

PRIORITIES
{self.format_bullets(priorities)}

VARIETY
{self.format_bullets(diversity)}

MAIN SUBJECT RULES
{self.format_bullets(self.prompt_config.get("living_subject_rules", []))}

WOW FACTOR
{self.format_bullets(self.prompt_config.get("wow_factor_guidance", []))}
"""

    def get_recent_concepts_text(self):

        recent_concepts = getattr(
            self,
            "recent_concepts",
            []
        )

        if not recent_concepts:

            return "- No previous concepts are available."

        return "\n".join(
            f"- {concept['title']}: {concept['prompt']}"
            for concept in recent_concepts[-8:]
        )

    def remember_concepts(self, concepts):

        self.recent_concepts.extend(
            concepts
        )

        self.recent_concepts = self.recent_concepts[
            -20:
        ]

    def get_prompt_word_range(self):

        video_config = (
            self.channel_config.get(
                "video",
                {}
            )
        )

        prompt_format = (
            self.prompt_config.get(
                "prompt_format",
                {}
            )
        )

        try:

            duration_seconds = float(
                video_config.get(
                    "duration_seconds",
                    8
                )
            )

        except (
            TypeError,
            ValueError
        ):

            duration_seconds = 8.0

        if duration_seconds <= 0:

            duration_seconds = 8.0

        try:

            minimum_rate = float(
                prompt_format.get(
                    "minimum_words_per_second",
                    2.75
                )
            )

            maximum_rate = float(
                prompt_format.get(
                    "maximum_words_per_second",
                    4.75
                )
            )

            minimum_floor = int(
                prompt_format.get(
                    "minimum_words_floor",
                    12
                )
            )

            maximum_cap = int(
                prompt_format.get(
                    "maximum_words_cap",
                    80
                )
            )

        except (
            TypeError,
            ValueError
        ):

            minimum_rate = 2.75
            maximum_rate = 4.75
            minimum_floor = 12
            maximum_cap = 80

        minimum_words = max(
            minimum_floor,
            math.ceil(
                duration_seconds * minimum_rate
            )
        )

        maximum_words = min(
            maximum_cap,
            math.floor(
                duration_seconds * maximum_rate
            )
        )

        maximum_words = max(
            minimum_words,
            maximum_words
        )

        return {
            "duration_seconds": duration_seconds,
            "minimum_words": minimum_words,
            "maximum_words": maximum_words
        }

    def build_prompt(
        self,
        count
    ):

        word_range = self.get_prompt_word_range()

        duration_seconds = word_range[
            "duration_seconds"
        ]

        minimum_words = word_range[
            "minimum_words"
        ]

        maximum_words = word_range[
            "maximum_words"
        ]

        ltx2_rules = self.prompt_config.get(
            "ltx2_prompt_rules",
            []
        )

        audio_guidance = self.prompt_config.get(
            "audio_guidance",
            []
        )

        episode_style = self.pick_episode_style()

        style_section = ""

        if episode_style:

            style_section = f"""
THIS EPISODE'S VISUAL STYLE

Every concept in this batch must be rendered as: {episode_style}.
The paragraph MUST begin exactly with "Style: {episode_style}," and
all visuals (characters, textures, motion, lighting) must match that
style throughout. Do not use any other visual style.
"""

        return f"""Generate exactly {count} original Scripture moment concepts.

For each concept, write a single FULL generation prompt that can be passed
directly to the video model. Each concept presents ONE powerful cinematic
moment from the life and teachings of Jesus, based on an actual passage from
Matthew, Mark, Luke, or John, in which Jesus speaks the exact Scripture text
of that passage. The Bible passage itself is the content: Scripture first,
visual second, cinematic execution third. The full prompt must contain every
visual and every audio element in one continuous paragraph.

{self.build_shared_context_sections()}
{style_section}
MUSIC AND SOUND FX (derive dynamically from the concept)
{self.format_bullets(audio_guidance)}

LTX2 PROMPT STRUCTURE (the "prompt" field, follow strictly)
{self.format_bullets(ltx2_rules)}

PROMPT COMPOSITION

While writing the paragraph keep these priorities in mind (arrange them in
your own creative order, not a rigid template):
- a self-contained visual moment: Jesus and the people around Him fully
  described, in an authentic first-century setting with real light,
  materials, and atmosphere;
- Jesus speaking the EXACT Scripture text in quotation marks - word for
  word, nothing added before or after - followed by His voice description;
- simple, physically coherent action that fits one continuous shot;
- a complete, cohesive soundtrack (restrained music + subtle ambient +
  clean effects) woven with the scene;
- one final image that lets the Scripture land.

Keep the moment short and uncluttered so it fits one continuous shot.

MOTION AND PACING

{duration_seconds:g} seconds holds ONE moment. Prioritize believable, subtle
natural movement over action: Jesus speaking naturally with small hand
gestures, gentle movement of hair and clothing, wind moving grass or dust
through sunlight, people shifting naturally, a few unhurried steps, water
moving, animals moving naturally. Do not cram multiple major events, plot
beats, or an entire story into the shot. Stillness and restraint are
appropriate when the moment is quiet - but the frame must never look
frozen: something small and real should always be in motion.

PROMPT LENGTH

Write the full "prompt" paragraph in approximately
{minimum_words}-{maximum_words} words. This range is derived from the
configured video duration of approximately
{duration_seconds:g} seconds. The paragraph must be long enough to establish
the style, Jesus' full appearance, the first-century setting, the exact
Scripture spoken in quotes with voice direction, a cohesive soundtrack, and
a final image - and short enough that a single continuous shot of this
duration can show it all without clutter.

OUTPUT

Return exactly {count} concepts. Use exactly this JSON structure:

[
    {{
        "title": "Short evocative title for the video",
        "reference": "Bible reference, e.g. John 3:16 or Luke 23:34",
        "scripture": "The EXACT Scripture text, word for word",
        "prompt": "THE COMPLETE single-paragraph video+audio prompt",
        "summary": "1-3 short sentences describing the visual moment"
    }}
]

Field rules:
- "reference": the Book Chapter:Verse citation of the passage used.
- "scripture": the exact Scripture text of the selected passage, copied
  word for word - no paraphrase, no added words, no commentary, no verse
  numbers. Only use wording you know with certainty; if you are not sure of
  the exact wording of a passage, choose a different well-known passage
  instead of guessing.
- "prompt": must contain that same Scripture in quotation marks as Jesus'
  spoken words, exactly as given in "scripture".
- "summary": a plain-language description of the visual moment for a
  viewer-facing caption: no camera or style jargon, no mention of prompts or
  AI, no motivational text, no commentary - at most 3 short sentences.
- "title": a short title that works as a video title; including the
  reference is fine (e.g. "John 3:16 - For God So Loved the World").

Return ONLY the JSON array.
No markdown.
No code block.
No explanation.
No commentary.
"""

    def generate(
        self,
        count
    ):

        self.log(
            f"Generating {count} video prompts"
        )

        return self.generate_single_stage(
            count
        )

    def generate_single_stage(
        self,
        count
    ):

        attempts = (
            self.get_generation_attempts()
        )

        response = ""

        for attempt in range(
            1,
            attempts + 1
        ):

            prompt = self.build_prompt(
                count
            )

            response = self.llm.generate(
                prompt,
                response_format=self.get_response_schema()
            )

            prompts = self.parse_response(
                response,
                count
            )

            if prompts:

                return prompts

            if attempt < attempts:

                self.log(
                    "No concept survived Scripture validation "
                    f"(attempt {attempt}/{attempts}); "
                    "retrying prompt generation."
                )

        raise RuntimeError(
            "Prompt generator returned no usable prompts after "
            f"{attempts} attempts. "
            f"Raw response: {str(response)[:500]}"
        )

    def get_response_schema(self):

        return {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string"
                    },
                    "reference": {
                        "type": "string"
                    },
                    "scripture": {
                        "type": "string"
                    },
                    "prompt": {
                        "type": "string"
                    },
                    "summary": {
                        "type": "string"
                    }
                },
                "required": [
                    "title",
                    "reference",
                    "scripture",
                    "prompt",
                    "summary"
                ],
                "additionalProperties": False
            }
        }

    def parse_response(
        self,
        response,
        count
    ):

        if not response:

            self.log(
                "Prompt generator returned an empty response."
            )

            return []

        response = (
            self.clean_response(
                response
            )
        )

        if not response:

            self.log(
                "Prompt generator returned no usable content."
            )

            return []

        data = None

        try:

            data = json.loads(
                response
            )

        except (
            json.JSONDecodeError,
            TypeError
        ):

            self.log(
                "Prompt generator returned non-direct JSON. "
                "Attempting JSON array extraction."
            )

            data = (
                self.extract_json_array(
                    response
                )
            )

        if not isinstance(
            data,
            list
        ):

            self.log(
                "Prompt generator did not return a JSON list."
            )

            return []

        requested_count = max(
            int(count),
            1
        )

        prompts = []

        for item in data:

            if not isinstance(
                item,
                dict
            ):

                continue

            title = str(
                item.get(
                    "title",
                    ""
                )
            ).strip()

            prompt = str(
                item.get(
                    "prompt",
                    ""
                )
            ).strip()

            if not title:

                title = " ".join(
                    prompt.split()[:6]
                ).strip(" ,.-") or "Untitled Concept"

                self.log(
                    "Concept had no title; "
                    "derived one from the prompt."
                )

            if not prompt:

                self.log(
                    f"Skipping '{title}' because "
                    "the prompt is empty."
                )

                continue

            reference = str(
                item.get(
                    "reference",
                    ""
                )
            ).strip()

            scripture = str(
                item.get(
                    "scripture",
                    ""
                )
            ).strip()

            if not reference or not scripture:

                self.log(
                    f"Skipping '{title}' because it is missing the "
                    "Bible reference or the exact Scripture text."
                )

                continue

            if not self.scripture_in_prompt(
                scripture,
                prompt
            ):

                self.log(
                    f"Skipping '{title}' because the exact Scripture "
                    "text is not present verbatim in the prompt."
                )

                continue

            summary = str(
                item.get(
                    "summary",
                    ""
                )
            ).strip()

            if not summary:

                # Fallback for responses without a summary: derive a
                # short one from the opening of the full prompt,
                # dropping the Style clause.

                body = re.sub(
                    r"^Style:[^,]*,\s*",
                    "",
                    prompt,
                    flags=re.IGNORECASE
                )

                sentences = re.split(
                    r"(?<=[.!?])\s+",
                    body
                )

                summary = " ".join(
                    sentences[:2]
                ).strip()

            spoken = scripture

            caption = (
                f"{scripture}\n\n\u2014 {reference}"
            )

            prompts.append(
                {
                    "title": title,
                    "reference": reference,
                    "scripture": scripture,
                    "prompt": prompt,
                    "summary": summary,
                    "spoken": spoken,
                    "caption": caption
                }
            )

        if not prompts:

            self.log(
                "Prompt generator parsed zero usable prompts."
            )

            return []

        if len(prompts) < requested_count:

            self.log(
                f"Model returned {len(prompts)} usable prompts "
                f"out of {requested_count} requested."
            )

        if len(prompts) > requested_count:

            self.log(
                f"Model returned {len(prompts)} prompts. "
                f"Using the requested {requested_count}."
            )

            prompts = prompts[
                :requested_count
            ]

        self.log(
            f"Generated {len(prompts)} usable video prompts"
        )

        self.remember_concepts(
            prompts
        )

        return prompts

    def normalize_scripture_text(
        self,
        text
    ):

        return re.sub(
            r"[^a-z0-9]",
            "",
            str(text).lower()
        )

    def scripture_in_prompt(
        self,
        scripture,
        prompt
    ):

        # The spoken Scripture inside the prompt must contain the exact
        # selected Scripture text. Comparison ignores case, whitespace,
        # and punctuation (straight vs. curly quotes) but nothing else.

        normalized_scripture = (
            self.normalize_scripture_text(
                scripture
            )
        )

        if not normalized_scripture:

            return False

        return (
            normalized_scripture
            in self.normalize_scripture_text(
                prompt
            )
        )

    def validate_prompt(
        self,
        prompt
    ):

        if not isinstance(
            prompt,
            str
        ):

            return False

        return bool(
            prompt.strip()
        )

    def clean_response(
        self,
        response
    ):

        if not isinstance(
            response,
            str
        ):

            return ""

        cleaned = response.strip()

        if not cleaned:

            return ""

        if "```" in cleaned:

            cleaned = re.sub(
                r"```(?:json)?",
                "",
                cleaned,
                flags=re.IGNORECASE
            )

            cleaned = cleaned.replace(
                "```",
                ""
            )

            cleaned = cleaned.strip()

        return cleaned

    def extract_json_array(
        self,
        response
    ):

        if not response:

            return []

        start = response.find(
            "["
        )

        if start == -1:

            return []

        depth = 0
        in_string = False
        escaped = False

        for index in range(
            start,
            len(response)
        ):

            character = response[index]

            if escaped:

                escaped = False
                continue

            if character == "\\" and in_string:

                escaped = True
                continue

            if character == '"':

                in_string = not in_string
                continue

            if in_string:

                continue

            if character == "[":

                depth += 1

            elif character == "]":

                depth -= 1

                if depth == 0:

                    candidate = response[
                        start:index + 1
                    ]

                    try:

                        return json.loads(
                            candidate
                        )

                    except (
                        json.JSONDecodeError,
                        TypeError
                    ):

                        repaired = (
                            self.repair_json_array(
                                candidate
                            )
                        )

                        try:

                            return json.loads(
                                repaired
                            )

                        except (
                            json.JSONDecodeError,
                            TypeError
                        ):

                            return []

        return []

    def repair_json_array(
        self,
        value
    ):

        if not value:

            return value

        repaired = value.strip()

        repaired = repaired.replace(
            "\\,",
            ","
        )

        repaired = repaired.replace(
            "\\/",
            "/"
        )

        repaired = re.sub(
            r'"\s*,\s*\\\s*,',
            '",',
            repaired
        )

        repaired = re.sub(
            r'"\s*,\s*\\\s*"',
            '", "',
            repaired
        )

        repaired = re.sub(
            r'\\\s*,\s*\\',
            ',',
            repaired
        )

        repaired = re.sub(
            r'\\\s*$',
            '',
            repaired
        )

        return repaired
