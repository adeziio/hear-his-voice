import difflib
import json
import re

from pathlib import Path

from ai.base_ai_service import BaseAIService
from ai.providers.ollama_provider import OllamaProvider

from scripture.segmenter import (
    split_sentences,
    build_word_segments
)


# `target_duration_seconds` is only a target for narration generation. Visual
# timing is derived from the narration that was actually generated.
#
# Each SnapGenAI clip carries one visual segment, and its length is the
# `duration_target` in config/ai_models.json (for example "10s"). The
# spoken-length ceiling used when cutting the narration is read from that
# setting by _duration_target_seconds() rather than hardcoded: the narration
# is cut into consecutive word chunks of up to duration_target x
# words_per_second words each (build_word_segments enforces the count), and
# a cut may fall mid-sentence or mid-quotation - always between words, so
# every word of the narration appears exactly once, in order. A shorter
# final chunk is fine - the duration is a maximum, never a duration to
# fill. The default below is only the fallback used when that setting is
# missing or cannot be parsed.
DEFAULT_SECONDS_PER_VISUAL = 10
DEFAULT_WORDS_PER_SECOND = 2.9


def _duration_target_seconds(config):
    """
    Returns the SnapGenAI clip length in seconds, read from
    config/ai_models.json -> models.video_model.snapgenai (shared
    settings, e.g. "10s").

    The configured clip duration sets the maximum spoken length assigned
    to one visual segment. The narration is divided into word chunks later,
    so a cut may fall mid-sentence. The value comes from configuration,
    never from a hardcoded constant;
    DEFAULT_SECONDS_PER_VISUAL is only the fallback used when the setting
    is missing or unparseable.
    """
    snapgenai = (
        config.get("ai_models", {})
        .get("models", {})
        .get("video_model", {})
        .get("snapgenai", {})
    )

    value = str(snapgenai.get("duration", "")).strip()

    match = re.fullmatch(
        r"(\d+(?:\.\d+)?)\s*s?",
        value,
        flags=re.IGNORECASE,
    )

    if match is None:

        return DEFAULT_SECONDS_PER_VISUAL

    seconds = float(match.group(1))

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


def _max_words_per_segment(config):
    """
    Returns the most words one narration segment may hold.

    The spoken-length ceiling is duration_target x words_per_second:
    one clip speaks at most this much, so it is the word budget of one
    narration segment - the narration is cut into chunks of exactly
    this many words (except the shorter final chunk) by
    build_word_segments - and it is the budget the narration prompt
    tells the writer up front.

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
# Visuals are directed ONE narration word-chunk at a time. Each focused
# call returns exactly one visual object for the assigned chunk, while
# adjacent narration and the source passage clarify its story context.
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


# The metadata schema: the episode title, summary, and mood, generated
# after the visuals are written. The title and summary then receive one
# independent review call with separate verdicts, followed by targeted
# repair and re-review when either field fails. A deterministic summary
# completeness guard prevents visibly truncated summaries without imposing
# a word, character, or sentence-count limit.
METADATA_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "maxLength": 90},
        "summary": {"type": "string"},
        "mood": {"type": "string"},
    },
    "required": ["title", "summary", "mood"],
}


METADATA_VERIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "title_passes": {"type": "boolean"},
        "title_issues": {"type": "string"},
        "summary_passes": {"type": "boolean"},
        "summary_issues": {"type": "string"},
        "summary_attribution_passes": {"type": "boolean"},
        "summary_attribution_issues": {"type": "string"},
    },
    "required": [
        "title_passes",
        "title_issues",
        "summary_passes",
        "summary_issues",
        "summary_attribution_passes",
        "summary_attribution_issues",
    ],
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
        "has_concrete_action": {"type": "boolean"},
        "depicts_concrete_action": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": [
        "aligned",
        "has_concrete_action",
        "depicts_concrete_action",
        "reason",
    ],
}


VISUAL_PAIRWISE_VERIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "pair_reviews": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "left_segment_index": {"type": "integer"},
                    "right_segment_index": {"type": "integer"},
                    "left_visual_action": {"type": "string"},
                    "right_visual_action": {"type": "string"},
                    "left_visual_composition": {"type": "string"},
                    "right_visual_composition": {"type": "string"},
                    "narrated_beats_are_distinct": {"type": "boolean"},
                    "same_primary_action": {"type": "boolean"},
                    "same_primary_composition": {"type": "boolean"},
                    "adds_meaningful_difference": {"type": "boolean"},
                    "materially_repetitive": {"type": "boolean"},
                    "issue": {"type": "string"},
                    "suggested_focus": {"type": "string"},
                },
                "required": [
                    "left_segment_index",
                    "right_segment_index",
                    "left_visual_action",
                    "right_visual_action",
                    "left_visual_composition",
                    "right_visual_composition",
                    "narrated_beats_are_distinct",
                    "same_primary_action",
                    "same_primary_composition",
                    "adds_meaningful_difference",
                    "materially_repetitive",
                    "issue",
                    "suggested_focus",
                ],
            },
        },
    },
    "required": ["pair_reviews"],
}


VISUAL_SEQUENCE_VERIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "passes": {"type": "boolean"},
        "segment_reviews": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "segment_index": {"type": "integer"},
                    "narrated_beat": {"type": "string"},
                    "depicted_beat": {"type": "string"},
                    "beat_is_represented": {"type": "boolean"},
                    "materially_repetitive": {"type": "boolean"},
                    "issue": {"type": "string"},
                    "suggested_focus": {"type": "string"},
                },
                "required": [
                    "segment_index",
                    "narrated_beat",
                    "depicted_beat",
                    "beat_is_represented",
                    "materially_repetitive",
                    "issue",
                    "suggested_focus",
                ],
            },
        },
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "segment_index": {"type": "integer"},
                    "issue": {"type": "string"},
                    "suggested_focus": {"type": "string"},
                },
                "required": ["segment_index", "issue", "suggested_focus"],
            },
        },
    },
    "required": ["passes", "segment_reviews", "findings"],
}


NARRATION_VERIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "accurate": {"type": "boolean"},
        "issues": {"type": "string"},
    },
    "required": ["accurate", "issues"],
}


NARRATION_STORYTELLING_SCHEMA = {
    "type": "object",
    "properties": {
        "passes": {"type": "boolean"},
        "issues": {"type": "string"},
    },
    "required": ["passes", "issues"],
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


# These checks are deliberately deterministic. The language-model verifier
# remains responsible for biblical meaning and completeness, but it must not
# be the only gate for errors that code can identify exactly.
_NON_PERSON_SOURCE_NAMES = frozenset({
    "behold", "christ", "come", "god", "holy", "israel", "lamb",
    "messiah", "rabbi", "son", "spirit", "teacher",
})

_SPEECH_PRONOUN_RE = re.compile(
    r"\b(?P<subject>he|she)\s+"
    r"(?:(?:quietly|softly|solemnly|immediately|then|again)\s+)?"
    r"(?P<verb>said|asked|replied|answered|declared|testified|spoke|"
    r"called|cried|shouted|whispered|told)\b",
    re.IGNORECASE,
)

_VISUAL_TEXT_STOP_WORDS = frozenset({
    "style", "realistic", "the", "and", "but", "with", "from", "into",
    "onto", "over", "under", "near", "beside", "while", "where", "when",
    "his", "her", "their", "they", "them", "him", "she", "he", "it",
    "this", "that", "these", "those", "was", "were", "are", "is", "be",
    "been", "being", "has", "have", "had", "for", "one", "two", "both",
    "simple", "modest", "ancient", "first", "century", "biblical", "period",
    "setting", "scene", "shot", "camera", "framing", "light", "soft", "warm",
    "golden", "morning", "early", "dawn", "afternoon", "quiet", "calm",
    "gentle", "reverent", "atmosphere", "shadow", "shadows", "around",
})


def _scan_quoted_spans(text):
    """Return nested quoted spans and report genuinely unbalanced quotes.

    Quote style is intentionally flexible: straight, curly, single, and
    double marks are accepted at any nesting level. The checker enforces
    balanced, correctly paired marks, not typographic preferences. Apostrophes
    inside contractions and possessives are treated as ordinary characters.
    """
    value = str(text or "")
    stack = []
    spans = []
    problems = []

    def add_closed_span(item, end_index):
        quoted = value[item["start"]:end_index].strip()
        if quoted:
            spans.append((quoted, item["start"], end_index))

    for index, char in enumerate(value):
        previous = value[index - 1] if index else ""
        following = value[index + 1] if index + 1 < len(value) else ""

        # Apostrophes inside words (didn't, Jesus's) are not delimiters.
        if char in ("'", "’") and previous.isalnum() and following.isalnum():
            continue

        # A trailing apostrophe after s can mark a possessive (the disciples'
        # boat, Moses' words). Outside a quoted span it is unambiguous when a
        # following word is present. Inside a single-quoted span, look for a
        # later matching closer: this distinguishes a possessive inside a
        # phrase such as ‘Jesus’ mother’ from a quote around one word ('Jesus').
        if (
            char in ("'", "’")
            and previous.lower() == "s"
            and previous.isalnum()
            and following.isspace()
            and re.match(r"\s+[A-Za-z0-9]", value[index + 1:])
        ):
            closes_active_single_quote = bool(
                stack and stack[-1]["closing"] == char
            )
            if not closes_active_single_quote:
                continue
            later_closer = value.find(char, index + 1)
            if later_closer >= 0 and re.search(
                r"[A-Za-z0-9]", value[index + 1:later_closer]
            ):
                continue

        if char in ("“", "‘"):
            stack.append({
                "opening": char,
                "closing": {"“": "”", "‘": "’"}[char],
                "start": index + 1,
                "position": index,
            })
            continue

        if char == '"':
            if stack and stack[-1]["closing"] == '"':
                add_closed_span(stack.pop(), index)
            else:
                stack.append({
                    "opening": char,
                    "closing": char,
                    "start": index + 1,
                    "position": index,
                })
            continue

        if char == "'":
            if stack and stack[-1]["closing"] == "'":
                add_closed_span(stack.pop(), index)
            elif (not previous or previous.isspace() or previous in "([{—–:;,") and following and not following.isspace():
                stack.append({
                    "opening": char,
                    "closing": char,
                    "start": index + 1,
                    "position": index,
                })
            else:
                problems.append(
                    f"Unmatched or mis-nested apostrophe quotation mark at character {index}."
                )
            continue

        if char in ("”", "’"):
            if stack and stack[-1]["closing"] == char:
                add_closed_span(stack.pop(), index)
            else:
                problems.append(
                    f"Unmatched or mis-nested closing quotation mark {char!r} at character {index}."
                )

    for item in stack:
        problems.append(
            f"Unclosed quotation mark {item['opening']!r} at character {item['position']}."
        )

    spans.sort(key=lambda item: (item[1], -item[2]))
    return spans, problems


def _closest_source_quote_excerpt(source_text, quoted_words):
    """Return the closest same-length source word window for actionable feedback."""
    source = str(source_text or "")
    token_matches = list(re.finditer(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*", source))
    flat_words = []
    flat_to_token = []
    for token_index, match in enumerate(token_matches):
        normalized = _normalize_words(match.group(0))
        for word in normalized:
            flat_words.append(word)
            flat_to_token.append(token_index)

    width = len(quoted_words)
    if not width or len(flat_words) < width:
        return ""

    best_start = None
    best_score = -1.0
    for start in range(len(flat_words) - width + 1):
        window = flat_words[start:start + width]
        score = difflib.SequenceMatcher(
            None, quoted_words, window, autojunk=False
        ).ratio()
        if score > best_score:
            best_score = score
            best_start = start
            if score == 1.0:
                break

    if best_start is None or best_score < 0.55:
        return ""

    first_token = flat_to_token[best_start]
    last_token = flat_to_token[best_start + width - 1]
    excerpt = source[
        token_matches[first_token].start():token_matches[last_token].end()
    ]
    return " ".join(excerpt.split())


def _quoted_source_sentence_boundaries(quoted_source_text):
    """Return word-index boundaries for complete sentences in a source quote."""
    value = str(quoted_source_text or "")
    tokens = list(re.finditer(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*", value))
    if not tokens:
        return set(), set(), []

    words = []
    token_word_ends = []
    for token in tokens:
        words.extend(_normalize_words(token.group(0)))
        token_word_ends.append(len(words))

    sentence_starts = {0}
    sentence_ends = set()
    for index in range(len(tokens) - 1):
        gap = value[tokens[index].end():tokens[index + 1].start()]
        # A sentence boundary must be terminal punctuation followed only by
        # optional closing punctuation/whitespace before the next source word.
        if re.search(r"[.!?](?:[\"'”’)]*)\s*$", gap):
            boundary = token_word_ends[index]
            sentence_ends.add(boundary)
            sentence_starts.add(boundary)

    sentence_ends.add(len(words))
    return sentence_starts, sentence_ends, words


def narration_quote_completeness_problems(source_text, narration):
    """Reject quoted fragments that cut through a sentence in source speech.

    A narration may select one or more complete sentences from a longer source
    speech, but it must not put a clipped phrase or half-sentence in quotation
    marks. The separate accuracy reviewer still decides whether the selected
    complete sentence(s) preserve the whole relevant saying. This conservative
    deterministic check only applies when the source itself marks the speech
    with quotation marks; otherwise it leaves the decision to the LLM reviewer.
    """
    source = str(source_text or "")
    source_spans, _source_structure_problems = _scan_quoted_spans(source)
    narration_spans, _narration_structure_problems = _scan_quoted_spans(narration)
    if not source_spans or not narration_spans:
        return []

    source_quotes = []
    for source_quote, _start, _end in source_spans:
        variants = [source_quote]
        without_editorial_gloss = re.sub(
            r"\(\s*which is\b[^)]*\)",
            " ",
            source_quote,
            flags=re.IGNORECASE,
        )
        if without_editorial_gloss != source_quote:
            variants.append(without_editorial_gloss)
        for variant in variants:
            sentence_starts, sentence_ends, words = _quoted_source_sentence_boundaries(variant)
            if words:
                source_quotes.append((sentence_starts, sentence_ends, words))

    problems = []
    for quoted_text, _start, _end in narration_spans:
        quoted_words = _normalize_words(quoted_text)
        if not quoted_words:
            continue

        found_inside_source_speech = False
        complete_source_sentence = False
        width = len(quoted_words)

        for sentence_starts, sentence_ends, source_words in source_quotes:
            if len(source_words) < width:
                continue
            for start in range(len(source_words) - width + 1):
                if source_words[start:start + width] != quoted_words:
                    continue

                found_inside_source_speech = True
                end = start + width
                if start in sentence_starts and end in sentence_ends:
                    complete_source_sentence = True
                    break
            if complete_source_sentence:
                break

        if found_inside_source_speech and not complete_source_sentence:
            excerpt = " ".join(quoted_text.split())
            if len(excerpt) > 120:
                excerpt = excerpt[:117] + "..."
            problems.append(
                f'Direct quotation "{excerpt}" is only part of a sentence in the source quotation. '
                "Quote the complete source sentence (or complete selected sentence(s)) exactly, "
                "or paraphrase the fragment outside quotation marks. Do not shorten a quoted saying."
            )

    return problems


def narration_quote_problems(source_text, narration):
    """Reject unbalanced quotes and quoted words that are not source-exact.

    WEBC translator explanations in parentheses are editorial glosses. The
    source is checked both as printed and with those glosses omitted, allowing
    narration either to retain the exact parenthetical text or keep it outside
    the spoken quotation. Actual quoted words must still match consecutively.
    When the source marks a direct-speech span, quotations must also begin and
    end on complete source-sentence boundaries rather than clipping a saying.
    """
    spans, problems = _scan_quoted_spans(narration)
    source = str(source_text or "")
    source_variants = [
        _normalize_words(source),
        _normalize_words(re.sub(
            r"\(\s*which is\b[^)]*\)",
            " ",
            source,
            flags=re.IGNORECASE,
        )),
    ]

    for quoted_text, _start, _end in spans:
        quoted_words = _normalize_words(quoted_text)
        if not quoted_words:
            continue

        width = len(quoted_words)
        found = any(
            any(
                words[index:index + width] == quoted_words
                for index in range(len(words) - width + 1)
            )
            for words in source_variants
        )
        if not found:
            excerpt = " ".join(quoted_text.split())
            if len(excerpt) > 120:
                excerpt = excerpt[:117] + "..."
            source_excerpt = _closest_source_quote_excerpt(source, quoted_words)
            detail = (
                f' Direct source wording nearby: "{source_excerpt}".'
                if source_excerpt else ""
            )
            problems.append(
                f'Direct quotation "{excerpt}" does not match consecutive source words exactly.'
                f'{detail} Restore the exact WEBC wording; do not paraphrase words inside quotation marks.'
            )

    problems.extend(
        narration_quote_completeness_problems(source, narration)
    )
    return problems


def narration_speaker_problems(source_text, narration):
    """Reject speech attributed with he/she when two named people are local candidates.

    This intentionally conservative rule requires the narrator to name the
    speaker when two source-named people occur across the current and previous
    sentence. It avoids guessing which person a pronoun refers to.
    """
    person_names = (
        source_names(source_text)
        - SENTENCE_START_WORDS
        - _NON_PERSON_SOURCE_NAMES
    )
    if len(person_names) < 2:
        return []

    spans, _structure_problems = _scan_quoted_spans(narration)
    problems = []
    value = str(narration or "")

    for quoted_text, quote_start, _quote_end in spans:
        prefix = value[:quote_start]
        sentence_parts = re.split(r"[.!?][\"”’']?\s+", prefix)
        local_context = " ".join(sentence_parts[-2:])
        matches = list(_SPEECH_PRONOUN_RE.finditer(local_context))
        if not matches:
            continue

        match = matches[-1]
        # The speaker cue must be close to the quotation it introduces.
        if match.end() < max(0, len(local_context) - 120):
            continue

        candidate_names = {
            name for name in person_names
            if re.search(rf"\b{re.escape(name)}\b", local_context[:match.start()], re.IGNORECASE)
        }
        if len(candidate_names) >= 2:
            problems.append(
                f'Potentially ambiguous speaker: "{match.group("subject")} {match.group("verb")}" '
                f'before "{quoted_text[:70]}" follows references to '
                f'{", ".join(sorted(candidate_names))}. Name the speaker explicitly, '
                "following the source passage."
            )

    return problems


def _visual_similarity_tokens(value):
    return [
        token for token in re.findall(r"[a-z0-9]+", str(value or "").lower())
        if len(token) > 2 and token not in _VISUAL_TEXT_STOP_WORDS
    ]


def visual_repetition_problem(query, prior_visuals):
    """Reject an identical or near-identical scene already used in this episode."""
    current_tokens = _visual_similarity_tokens(query)
    if len(current_tokens) < 7:
        return None

    current_set = set(current_tokens)
    for index, item in enumerate(prior_visuals or []):
        if not isinstance(item, dict):
            continue
        previous = str(item.get("search_query") or "").strip()
        previous_tokens = _visual_similarity_tokens(previous)
        if len(previous_tokens) < 7:
            continue

        previous_set = set(previous_tokens)
        union = current_set | previous_set
        jaccard = len(current_set & previous_set) / max(1, len(union))
        sequence_ratio = difflib.SequenceMatcher(
            None,
            current_tokens,
            previous_tokens,
            autojunk=False,
        ).ratio()

        if sequence_ratio >= 0.84 and jaccard >= 0.76:
            return (
                f"This scene substantially repeats the scene already approved for segment {index + 1}. "
                "Create a genuinely different shot showing the assigned segment's own action or story beat. "
                "Keep character appearance consistent, but do not reuse the same action, composition, and scene."
            )

    return None


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
        2. cuts that narration into word-count visual segments - the
           ceiling comes from duration_target x words_per_second, and
           code guarantees exact narration coverage and the word count
           (build_segments),
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
        Cuts the generated narration into visual segments: consecutive
        word chunks of up to the configured word ceiling
        (duration_target x words_per_second), so each chunk gets
        exactly one SnapGenAI prompt and one clip.

        The cut falls between words - it may land mid-sentence or
        mid-quotation, which is intentional: grouping is strictly by
        spoken length, never by punctuation. Every word of the
        narration appears exactly once, in order; a shorter final
        chunk is fine and is never padded - the duration is a
        ceiling, never a duration to fill.

        The visual prompts receive the full passage and the full
        narration as context, so a chunk that starts or ends
        mid-sentence still depicts the correct characters, action,
        setting, and story moment.
        """
        text = scripture["text"]

        # How fast the narrator reads, and how long one clip holds -
        # both from configuration, through the same helpers the
        # narration prompt uses to tell the writer the budget up
        # front.
        words_per_second = _words_per_second(self.config)
        seconds_per_visual = _duration_target_seconds(self.config)
        max_words_per_segment = _max_words_per_segment(self.config)

        segments = build_word_segments(
            text,
            max_words_per_segment,
        )

        self.log(
            f"{len(segments)} word-chunk visuals for "
            f"{len(text.split())} narration words "
            f"(~{len(text.split()) / max(words_per_second, 0.1):.0f}s, "
            f"at most {max_words_per_segment} words / "
            f"{seconds_per_visual:g}s per segment)."
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

        Returns a usable partition (groups of unit numbers) in which
        every group is within the clip's spoken ceiling and adjacent
        small groups or fragments have been merged where they fit, or
        None when the model's partition cannot be made compliant
        without truncation - in which case the caller rejects the
        partition and either re-rolls or falls back to one visual per
        sentence.

        The story beats come from the model; the clip length does not.

        Rules:
        - Any group whose units would speak longer than the clip is
          rejected outright (fail-closed). Groups are never truncated
          or cut to fit - an over-ceiling group is a model judgment
          error, and the whole partition is rejected.
        - Adjacent groups whose combined word count fits the ceiling
          are merged, so a tiny leftover fragment is absorbed by its
          neighbor instead of shipping as its own visual.
        - Units are never split: a speaker's quoted utterance stays
          whole.
        - The narration text, order, and coverage are preserved
          exactly - every unit appears in exactly one group, in order.
        """
        unit_words = [0] + [
            len(text.split())
            for text in units
        ]

        # Fail closed: no group may exceed the clip's spoken ceiling,
        # and groups are never truncated or cut to fit. An over-ceiling
        # group means the model misjudged the beats, so reject the
        # whole partition and let the caller re-roll or fall back.
        for group in groups:

            if sum(
                unit_words[item]
                for item in group
            ) > max_words:

                return None

        # Merge adjacent small groups and fragments whenever they fit
        # within one clip. Units are never split, so dialogue stays
        # intact; the combined group stays within the ceiling by
        # construction.
        merged = []

        for group in groups:

            if merged:

                combined = (
                    sum(
                        unit_words[item]
                        for item in merged[-1]
                    )
                    + sum(
                        unit_words[item]
                        for item in group
                    )
                )

                if combined <= max_words:

                    merged[-1].extend(group)

                    continue

            merged.append(list(group))

        return merged

    def build_segment_prompt(
        self,
        scripture,
        segments,
        index,
        feedback=None,
        prior_visuals=None
    ):
        """Build one self-contained visual prompt for a narration word chunk."""
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
            else "(this is the opening segment)"
        )
        after = (
            segments[index + 1] if index + 1 < len(segments)
            else "(this is the final segment)"
        )
        full_telling = "\n".join(
            f"{position + 1}. {part}"
            for position, part in enumerate(segments)
        )
        prior_visual_context = "\n".join(
            f"- Segment {position + 1}: {str(item.get('search_query') or '').strip()}"
            for position, item in enumerate(prior_visuals or [])
            if isinstance(item, dict) and str(item.get("search_query") or "").strip()
        )

        prompt = (
            "You are the visual director for \""
            + name
            + "\", creating one realistic biblical-era AI video clip "
            "for a narrated Scripture episode.\n\n"
            "CHANNEL DESCRIPTION\n"
            + description
            + "\n\nSCRIPTURE REFERENCE\n"
            + scripture["reference"]
            + " ("
            + scripture["translation"]
            + ")\n\n"
            "THE SOURCE PASSAGE\n"
            + str(scripture.get("text") or "").strip()
            + "\n\n"
            "FULL NARRATION IN ORDER - CONTEXT ONLY\n"
            + full_telling
            + "\n\n"
            "THE ASSIGNED NARRATION SEGMENT - VISUAL FOCUS\n"
            + segment
            + "\n\n"
            "IMMEDIATELY BEFORE\n"
            + before
            + "\nIMMEDIATELY AFTER\n"
            + after
            + ("\n\nPREVIOUSLY APPROVED VISUALS - CHARACTER CONTINUITY ONLY\n"
               "Reuse the same established character appearance and clothing "
               "for recurring people. These earlier prompts are references "
               "for character consistency, not instructions to repeat their "
               "actions or scenes:\n" + prior_visual_context
               if prior_visual_context else "")
            + "\n\n"
            "Create one clear, self-contained visual scene that illustrates "
            "at least one important moment from the ASSIGNED NARRATION SEGMENT. "
            "This segment is a word-count chunk and may contain more than one "
            "action or story beat, or may begin or end in the middle of a sentence. "
            "Use adjacent narration to understand its grammar and meaning. "
            "Because this is one short clip, choose the single clearest meaningful "
            "moment; do NOT try to depict every action or every sentence in the "
            "segment, and do NOT combine separate chronological events into a "
            "multi-scene montage. Showing one accurate, relevant part of the "
            "segment is sufficient.\n\n"
            "The assigned segment determines what the clip is about. "
            "Use the full narration and Scripture passage only to clarify "
            "who is involved, the biblical setting, and the order of events. "
            "Do not replace the assigned moment with a different, more "
            "familiar event from this passage or from elsewhere in Scripture. "
            "Do not show a later or earlier event instead of the moment "
            "being narrated.\n\n"
            "Make the chosen moment concrete and recognizable. If the segment "
            "contains several actions, select ONE important, visually clear "
            "action or moment and depict that well; the clip does not need to "
            "cover the other actions in the same segment. Include the people, "
            "objects, or setting needed to make the selected moment understandable, "
            "but do not force every named person into the shot. A relevant symbolic "
            "or contextual image is also acceptable when it meaningfully represents "
            "one part of the assigned segment. Do not use an unrelated biblical "
            "scene as a substitute. Keep the prompt to one coherent continuous shot. "
            "Do not add a separate later or earlier event as though it is happening "
            "simultaneously; subtle atmosphere or background context is fine when it "
            "does not confuse the chronology or distract from the selected moment.\n\n"
            "Only include characters who belong in this particular moment. "
            "Name the relevant biblical characters explicitly, using their "
            "full name and title when the passage establishes one. Do not "
            "add unrelated characters or make a different person perform "
            "the narrated action. Include Jesus by name when he is actually "
            "part of this moment; do not insert him into scenes where he is "
            "not present.\n\n"
            "CHARACTER APPEARANCE AND CONTINUITY\n"
            "When a named biblical character is central to the scene, "
            "describe their recognizable appearance and clothing using "
            "distinctive details established in Scripture, including relevant "
            "details from elsewhere in Scripture when appropriate. Prefer a "
            "distinctive scriptural garment or identifying detail over generic "
            "robes, tunics, cloaks, or head coverings when Scripture establishes "
            "something more specific. Keep the same character's description "
            "consistent across scenes. Do not invent physical features or "
            "clothing that Scripture does not establish.\n\n"
            "Keep the setting and actions consistent with what the passage "
            "establishes. Do not invent a specific location, movement, "
            "interaction, or outcome that changes the meaning of the scene. "
            "When the text does not specify a person's exact position, do "
            "not invent a restrictive location rule.\n\n"
            "Use a historically plausible biblical-era setting: ancient "
            "stone or mudbrick buildings, streets, homes, landscapes, and "
            "period-appropriate clothing and objects. No modern clothing, "
            "architecture, roads, vehicles, or technology. Never describe "
            "Jesus's physical appearance; identify him by name, actions, "
            "and place in the scene.\n\n"
            "Write a concise prompt for one continuous short shot. Identify "
            "the main subject, action, setting, atmosphere, and suitable "
            "camera framing. Begin exactly with 'Style: realistic.' "
            "Do not write narration, a stock-footage search query, or an "
            "explanation.\n\n"
            "VISUAL STYLE\n"
            + creative_directions
            + "\n\nCHANNEL VISUAL RULES\n"
            + visual_rules
            + "\n\nRETURN exactly one JSON object with:\n"
            "- search_query: the complete standalone SnapGenAI visual prompt.\n"
            "- visual_direction: one concise sentence describing the same shot.\n"
        )

        if feedback:
            prompt += (
                "\n\nPREVIOUS VISUAL ATTEMPTS WERE REJECTED\n"
                "Use the feedback below to correct the mismatch. Do not "
                "repeat a rejected scene or merely reword it; choose a "
                "clearer visual that directly represents the assigned "
                "narration segment. Address each rejection reason.\n"
                + feedback
                + "\n"
            )

        return prompt


    def build_visual_verification_prompt(
        self,
        scripture,
        segments,
        index,
        query,
        prior_visuals=None
    ):
        """Build a focused check of one visual prompt against its narration chunk."""
        channel = self.channel_config.get("channel", {})
        name = str(channel.get("name", "Hear His Voice"))
        segment = segments[index]
        before = (
            segments[index - 1] if index > 0
            else "(this is the opening segment)"
        )
        after = (
            segments[index + 1] if index + 1 < len(segments)
            else "(this is the final segment)"
        )
        full_telling = "\n".join(
            f"{position + 1}. {part}"
            for position, part in enumerate(segments)
        )
        prior_visual_context = "\n".join(
            f"- Segment {position + 1}: {str(item.get('search_query') or '').strip()}"
            for position, item in enumerate(prior_visuals or [])
            if isinstance(item, dict) and str(item.get("search_query") or "").strip()
        )

        return (
            "You are the visual accuracy checker for \""
            + name
            + "\". Decide whether the proposed visual is a faithful, "
            "recognizable illustration of the assigned narration segment.\n\n"
            "SCRIPTURE REFERENCE\n"
            + scripture["reference"]
            + " ("
            + scripture["translation"]
            + ")\n\nSOURCE PASSAGE\n"
            + str(scripture.get("text") or "").strip()
            + "\n\nFULL NARRATION IN ORDER - CONTEXT\n"
            + full_telling
            + "\n\nASSIGNED NARRATION SEGMENT - THE VISUAL MUST SUPPORT THIS\n"
            + segment
            + "\n\nIMMEDIATELY BEFORE\n"
            + before
            + "\nIMMEDIATELY AFTER\n"
            + after
            + "\n\nPROPOSED VISUAL\n"
            + str(query or "")
            + ("\n\nPREVIOUSLY APPROVED VISUALS - SCENE DIVERSITY AND CHARACTER CONTINUITY\n"
               "Compare recurring characters' appearance and clothing with "
               "these earlier prompts. Also compare the central action and shot: "
               "reject an identical or near-identical scene or a repeated action "
               "when this segment has a different story beat. The same character "
               "or setting may recur when the action genuinely changes.\n"
               + prior_visual_context
               if prior_visual_context else "")
            + "\n\nACCEPTANCE STANDARD FOR A SHORT CLIP\n"
            "A narration segment may contain several actions, clauses, or story "
            "beats because it is cut by word count, not by scene. The visual is "
            "NOT required to illustrate the whole segment. Accept it when it clearly "
            "and meaningfully depicts AT LEAST ONE important action, moment, person, "
            "or idea that genuinely belongs to this assigned segment. One coherent "
            "10-second scene is preferred over trying to fit several events into "
            "one clip. Do not reject a good scene merely because another action or "
            "detail from the same segment is not shown. Use neighboring narration "
            "to resolve fragments and identify possible moments; it does not mean "
            "every neighboring or later event must appear in this visual.\n\n"
            "Reject only a material mismatch: an unrelated scene, a wrong central "
            "actor or recipient for the moment chosen, a clear contradiction of "
            "Scripture, or a chronology error that materially misleads the viewer. "
            "An extra background detail or subtle foreshadowing is not automatically "
            "a failure; reject it only if it makes separate events appear to happen "
            "at the same time, changes who did what, or distracts from / contradicts "
            "the selected relevant moment.\n\n"
            "For the particular moment the prompt chooses to show, verify that "
            "its actors and relationships are faithful. Do not require every actor, "
            "recipient, action, or outcome mentioned anywhere in a multi-beat "
            "segment to appear in this one shot. A wrong actor or recipient is a "
            "failure only when the chosen scene actually assigns that moment to "
            "the wrong person.\n"
            "When the chosen scene depicts speech, verify the speaker against "
            "the source passage's grammar, not merely the nearest person named "
            "in the sentence. For a chosen physical action, include enough of "
            "the correct actor and relevant recipient or object to make THAT "
            "selected moment understandable; do not require participants or "
            "targets belonging only to other actions in the same segment.\n"
            "CONCRETE-ACTION CHECK - REQUIRED SEPARATE DECISION\n"
            "Identify the meaningful story beats present in this segment, using "
            "the surrounding words to resolve sentence fragments. Then identify "
            "which one of those beats the proposed visual actually chooses to show. "
            "The segment may contain multiple events; one accurate, important beat "
            "is sufficient for this short clip.\n"
            "Set has_concrete_action=true when at least one beat in the segment is "
            "a visibly depictable physical action or stage of an event, such as "
            "taking or carrying someone, departing, traveling, arriving, entering, "
            "giving, receiving, or gathering. Set it false when the central beat "
            "is a state, condition, or passage of time rather than a distinct "
            "physical action. This includes remaining, staying, living, or "
            "continuing to reside in a stated place until a later event, when no "
            "separate physical action is narrated.\n"
            "IMPORTANT - CONTEXTUAL STATES ARE NOT PHYSICAL ACTIONS: If the "
            "narration says people remained or stayed somewhere until a later "
            "event, a faithful scene showing those people in that place is enough "
            "to represent the beat. A scene showing people living in the stated "
            "place can represent that they remained there. Do NOT "
            "require the image to demonstrate duration, waiting, watchfulness, "
            "anticipation, the later event, or the reason their stay ended. Do NOT "
            "require time-lapse imagery, aging, news of a death, or symbolic proof "
            "of time passing unless the narration actually describes those things.\n"
            "Set depicts_concrete_action=true when the visual clearly shows at "
            "least ONE meaningful beat from the segment. If that chosen beat is a "
            "physical action, it should be recognizable as that action and be "
            "consistent with the source. It does not have to show the other actions "
            "in the segment, every participant, or the full outcome. If the chosen "
            "beat is a contextual state, line of dialogue, prophecy, or abstract idea, "
            "a meaningful contextual or symbolic scene can also satisfy this field. "
            "Do not fail a relevant scene just because it illustrates one part rather "
            "than the whole chunk. Fail only when none of the meaningful beats in the "
            "assigned segment is represented, or when the selected scene materially "
            "misrepresents the passage.\n\n"
            "CHARACTER DEPICTION CHECK\n"
            "Use recognizable appearance, clothing, and regional details "
            "established in Scripture or clearly supported by the biblical setting "
            "when they help identify a central character. However, do not reject "
            "an otherwise relevant, period-appropriate scene merely because the "
            "prompt does not spell out every known physical or clothing detail. "
            "Reject only a clear contradiction, anachronism, or materially "
            "inappropriate depiction; accept reasonable artistic variation and "
            "details Scripture does not specify.\n\n"
            "A relevant representative or symbolic visual may be acceptable "
            "even when the segment also mentions physical actions, provided the "
            "chosen image meaningfully illustrates one important part of that "
            "segment. Do not require every person or detail from the full passage "
            "to appear in this one clip. Do not reject simply because the visual "
            "is not a word-for-word reenactment, omits secondary actions, or uses "
            "reasonable atmosphere, framing, or background detail.\n\n"
            "CHRONOLOGY AND LOCATION CHECK\n"
            "Read the narration in its actual event order. Distinguish what has "
            "already happened at the assigned moment from what happens later. "
            "Never project a later departure, arrival, destination, or outcome "
            "backward onto an earlier warning, dream, conversation, or decision. "
            "A warning to leave a place occurs before the departure unless the "
            "source explicitly says otherwise. Do not infer that a character is "
            "already at a later destination merely because the full passage says "
            "they travel there afterward. If the source does not establish the "
            "exact location for this moment, accept a plausible location that "
            "does not contradict the passage; do not invent a location conflict.\n\n"
            "The passage and surrounding narration provide context; they "
            "do not authorize replacing the assigned segment with another "
            "scene. Do not infer exact character locations from sentence "
            "counts or estimated position timelines. Only reject a location "
            "when the proposed scene clearly contradicts what the passage "
            "actually establishes for this moment.\n\n"
            "RETURN exactly one JSON object:\n"
            "- aligned: true when the visual faithfully and meaningfully shows "
            "at least one important beat from the assigned segment, without a "
            "material contradiction. Do not require complete segment coverage.\n"
            "- has_concrete_action: true if the segment contains at least one "
            "physical action that could be shown; otherwise false.\n"
            "- depicts_concrete_action: true when the visual meaningfully depicts "
            "at least one relevant beat from the segment (a physical action, "
            "contextual state, speech, or symbolic idea); it need not depict all "
            "actions in the chunk.\n"
            "- reason: if any check fails, briefly identify the specific mismatch "
            "and what the visual should represent instead; if all pass, an empty string.\n"
        )


    def verify_visual(
        self,
        scripture,
        segments,
        index,
        query,
        prior_visuals=None
    ):
        """Return the checker's explicit alignment verdict for one visual."""
        prompt = self.build_visual_verification_prompt(
            scripture,
            segments,
            index,
            query,
            prior_visuals=prior_visuals
        )

        try:
            response = self.llm.generate(
                prompt,
                response_format=VISUAL_VERIFICATION_SCHEMA
            )
        except Exception as error:
            raise ContentGenerationError(
                f"Visual verification could not run: {error}"
            ) from error

        try:
            verdict = self.parse_direction(response)
        except ContentGenerationError:
            return False, "Visual verification returned no parseable verdict; generate and verify a new prompt."

        if (
            not isinstance(verdict, dict)
            or not isinstance(verdict.get("aligned"), bool)
            or not isinstance(verdict.get("has_concrete_action"), bool)
            or not isinstance(verdict.get("depicts_concrete_action"), bool)
        ):
            return False, (
                "Visual verification did not return explicit relevance and "
                "concrete-action verdicts; generate and verify a new prompt."
            )

        reason = str(verdict.get("reason") or "").strip()
        if not verdict["depicts_concrete_action"]:
            if verdict["has_concrete_action"]:
                action_issue = (
                    "The assigned segment contains a depictable physical action, "
                    "but the proposed visual does not actually show that action. "
                    "Show the correct actor performing the narrated event, rather "
                    "than a related prayer, reaction, listening pose, distant "
                    "silhouette, or general setting."
                )
            else:
                action_issue = (
                    "The proposed visual does not meaningfully represent the "
                    "assigned segment's speech, idea, or abstract beat."
                )
            reason = f"{reason} {action_issue}".strip()
            return False, reason

        if verdict["aligned"]:
            return True, ""

        if not reason:
            reason = "The visual does not clearly represent the assigned narration segment."

        return False, reason


    def build_visual_sequence_review_prompt(
        self,
        scripture,
        segments,
        visuals,
    ):
        """Build an episode-wide review of story-beat coverage and visual progression."""
        channel = self.channel_config.get("channel", {})
        name = str(channel.get("name", "Hear His Voice"))

        sequence = []
        for index, (segment, visual) in enumerate(zip(segments, visuals), start=1):
            if isinstance(visual, dict):
                prompt = str(visual.get("search_query") or "").strip()
                direction = str(visual.get("visual_direction") or "").strip()
            else:
                prompt = str(visual or "").strip()
                direction = ""

            entry = (
                f"SEGMENT {index} NARRATION\n{segment}\n\n"
                f"SEGMENT {index} VISUAL PROMPT\n{prompt}"
            )
            if direction:
                entry += f"\n\nSEGMENT {index} VISUAL DIRECTION\n{direction}"
            sequence.append(entry)

        return (
            "You are the independent episode-wide visual storytelling and "
            "coverage reviewer for \""
            + name
            + "\". Every visual below has passed an initial individual review, "
            "but you must independently check whether each prompt actually "
            "depicts its assigned story beat and whether the sequence works "
            "as a coherent set. Do not assume an individual approval proves "
            "the concrete action is shown.\n\n"
            "SCRIPTURE REFERENCE\n"
            + scripture["reference"]
            + " ("
            + scripture["translation"]
            + ")\n\nSOURCE PASSAGE - CONTEXT FOR FEASIBLE ALTERNATIVES\n"
            + str(scripture.get("text") or "").strip()
            + "\n\nCOMPLETE VISUAL SEQUENCE IN NARRATION ORDER\n"
            + "\n\n------------------------------\n\n".join(sequence)
            + "\n\n"
            "MANDATORY PER-SEGMENT STORY-BEAT AUDIT\n"
            "For EVERY segment, independently do the following before deciding "
            "whether the episode passes:\n"
            "1. Identify the important story beats present in that narration "
            "segment, resolving sentence fragments with neighboring narration. "
            "A single word-count segment may contain multiple actions or events.\n"
            "2. State what the visual prompt actually depicts as its main subject "
            "and action. Do not infer an action that the prompt does not describe.\n"
            "3. Decide whether the visual clearly represents AT LEAST ONE meaningful "
            "beat from that segment. One coherent scene is sufficient even when the "
            "segment mentions several actions; do not require the visual to cover "
            "every action, participant, or detail in the chunk. The correct character "
            "or general setting alone is not enough if the chosen scene is unrelated, "
            "but a clearly relevant partial illustration is enough. A distant person "
            "need not perform every action mentioned elsewhere in the segment.\n"
            "IMPORTANT - STATES AND DURATION: A narration beat such as people "
            "remaining, staying, living, or continuing to reside in a place until "
            "a later event is a contextual state, not a physical action that must "
            "be performed on camera. A faithful scene showing the relevant people "
            "in the stated place is sufficient. A scene showing people living in "
            "the stated place can represent that they remained there. Do NOT "
            "demand visible waiting, watchfulness, anticipation, time passing, "
            "aging, news of the later event, or a depiction of why the stay ended. "
            "Do not reject a suitable contextual scene merely because it cannot "
            "visually prove how long the people stayed or what later ended that stay.\n"
            "4. Compare the shot with earlier and later clips. Decide whether it "
            "adds meaningful visual information or substantially repeats the "
            "same action, pose, composition, and focal subject.\n\n"
            "Do not give a visual credit for an action that appears only in the "
            "narration or is merely implied by an unrelated setting. Judge what the "
            "shot actually describes. However, because each clip is short, showing "
            "one meaningful action or moment from a multi-beat segment is sufficient; "
            "the other actions in that segment may be left for narration or neighboring "
            "clips. A contextual state such as remaining, staying, or living in a place "
            "does NOT require a physical action or a visual demonstration of duration. "
            "A meaningful scene showing the relevant people in the stated place is "
            "sufficient. For speech, prophecy, or abstract ideas, a meaningful symbolic "
            "or contextual image is acceptable even if the same segment also mentions "
            "a physical event that the clip does not show.\n\n"
            "CHRONOLOGY AUDIT\n"
            "Read every segment in narration order. Do not project a later event "
            "or destination backward into an earlier segment. A later arrival "
            "does not establish that the character was already at that destination "
            "during an earlier warning or decision. If the passage leaves a precise "
            "location unstated, do not reject a plausible location as a contradiction. "
            "Only identify a chronology or location error when the source or narration "
            "clearly establishes that the proposed scene is impossible at that moment.\n\n"
            "SEQUENCE STORYTELLING REVIEW\n"
            "- Does the ordered set of clips move through the passage's major "
            "events and stages in a way a viewer can follow?\n"
            "- Is the overall story progression understandable from the sequence "
            "as a whole? Individual clips may show only one selected moment, so do "
            "not require every action or transition mentioned inside each chunk "
            "to appear visually. Flag an omitted transition only when its absence "
            "across the complete sequence would make the story materially confusing "
            "or misleading. Use only events that actually occur in this passage.\n"
            "- Do consecutive clips offer meaningfully different visual "
            "information when the story provides room for it, rather than merely "
            "rewording the same pose or shot?\n"
            "- Are any major segment beats missing because multiple prompts "
            "reuse a general scene that does not show the assigned event?\n\n"
            "IMPORTANT ACCEPTANCE STANDARD\n"
            "Be practical, not perfectionistic. The episode may contain one or "
            "two less compelling or imperfect clips if they are still relevant "
            "and do not materially misrepresent the narration. Do not regenerate "
            "a clip merely to make every shot exceptional. However, do not pass "
            "a sequence where no meaningful part of a segment is represented, "
            "where the visuals materially misrepresent an essential story turn, "
            "or where repeated shots substantially replace distinct narrated "
            "events. Do not fail the sequence merely because some actions or "
            "transitions are conveyed by narration rather than shown on screen.\n"
            "Do not require every clip to depict a different event, location, "
            "character, or action. The same people or setting may recur when "
            "the action or stage genuinely changes. Do not manufacture variety "
            "by inventing actions, emotions, props, locations, or events that "
            "the passage does not support. Accept reasonable symbolic imagery "
            "for abstract or prophetic narration.\n\n"
            "TARGETED FINDINGS\n"
            "When a problem exists, identify the smallest set of segments that "
            "should be regenerated. For a repeated pair, normally flag only the "
            "later or weaker segment. Give a specific reason and a feasible "
            "alternative focus grounded in that segment and the passage. If no "
            "clearly better, faithful alternative exists for a minor visual "
            "weakness, do not force a change.\n\n"
            "RETURN EXACTLY ONE JSON OBJECT\n"
            "- passes: true when every segment has at least one meaningful beat "
            "represented and the overall sequence has no material contradiction "
            "or repetition that obscures the story. Do NOT require complete coverage "
            "of every segment's actions or details. Minor imperfections in one or "
            "two otherwise relevant clips do not prevent a pass.\n"
            "- segment_reviews: return EXACTLY one assessment for EVERY segment, "
            "in segment order. Each assessment must include segment_index "
            "(1-based), narrated_beat (the meaningful beat selected for review "
            "from the segment, not an exhaustive list), depicted_beat "
            "(what the prompt actually shows), beat_is_represented (boolean), "
            "materially_repetitive (boolean), issue (empty only when no material "
            "problem), and suggested_focus (empty when no replacement is needed).\n"
            "- findings: include only segments that need replacement, using "
            "segment_index (1-based), issue, and suggested_focus. Findings must "
            "include segments where NONE of the meaningful beats in the chunk is "
            "represented, a material contradiction exists, or a later shot "
            "materially repeats an earlier shot for a distinct narrated beat. Do "
            "not flag a segment merely because some secondary narration detail "
            "is not shown.\n"
        )


    def build_visual_pairwise_review_prompt(
        self,
        scripture,
        segments,
        visuals,
    ):
        """Build a focused comparison of every adjacent pair of visual prompts."""
        pairs = []
        for index in range(len(segments) - 1):
            left_visual = visuals[index]
            right_visual = visuals[index + 1]
            left_prompt = (
                str(left_visual.get("search_query") or "").strip()
                if isinstance(left_visual, dict)
                else str(left_visual or "").strip()
            )
            right_prompt = (
                str(right_visual.get("search_query") or "").strip()
                if isinstance(right_visual, dict)
                else str(right_visual or "").strip()
            )
            pairs.append(
                f"ADJACENT PAIR {index + 1} AND {index + 2}\n"
                f"LEFT NARRATION (segment {index + 1})\n{segments[index]}\n\n"
                f"LEFT VISUAL PROMPT\n{left_prompt}\n\n"
                f"RIGHT NARRATION (segment {index + 2})\n{segments[index + 1]}\n\n"
                f"RIGHT VISUAL PROMPT\n{right_prompt}"
            )

        return (
            "You are an independent visual repetition auditor for the Hear His Voice episode. "
            "Your only task is to compare each adjacent pair of proposed visual prompts. "
            "The previous episode-wide reviewer may have missed near-duplicate shots, so "
            "inspect the actual scene descriptions side by side rather than relying on its verdict.\n\n"
            "SCRIPTURE REFERENCE\n"
            + scripture["reference"]
            + " ("
            + scripture["translation"]
            + ")\n\nSOURCE PASSAGE\n"
            + str(scripture.get("text") or "").strip()
            + "\n\nADJACENT VISUAL PAIRS IN EPISODE ORDER\n"
            + "\n\n========================================\n\n".join(pairs)
            + "\n\nFOR EACH PAIR, COMPARE THESE SEPARATELY\n"
            "1. Are the two narration chunks expressing the same continuous story beat, "
            "or are they different stages/actions/events? A word-count boundary may split "
            "one continuous sentence, quotation, or action; do not call that a problem by itself.\n"
            "2. What is the main action and composition actually described in the LEFT visual?\n"
            "3. What is the main action and composition actually described in the RIGHT visual?\n"
            "4. Are the primary physical action and composition substantially the same? "
            "Compare the actor's pose/action, focal subject, camera framing, and scene setup. "
            "Changing adjectives, lighting words, or small background details does not make "
            "a duplicate shot meaningfully different.\n"
            "5. Does the later prompt add a meaningful visual difference or show the next "
            "stage of the story? A different action, actual change of stage/location, "
            "different focal event, or substantially different useful composition can count.\n\n"
            "MATERIAL REPETITION STANDARD\n"
            "Mark materially_repetitive=true only when the two narration beats are distinct "
            "and the visuals substantially reuse the same primary action AND composition "
            "without adding meaningful story information. Also mark it true when the later "
            "visual clearly substitutes the earlier shot for a different concrete action "
            "that its narration segment describes. Do NOT flag a continuous speech/dream/event "
            "merely because it spans adjacent word chunks, provided the same scene remains "
            "a faithful depiction of that continuous beat. Repeated characters, locations, "
            "clothing, or general atmosphere alone are not repetition. Do not demand every "
            "pair be visually different.\n\n"
            "Be practical: one or two less varied clips can be tolerated when they remain "
            "relevant. Flag clear duplicate shots or a repeated pose that replaces a new "
            "story action; do not flag a minor imperfection or a reasonable artistic choice. "
            "Do not invent a different event simply to create variety. When the later pair "
            "member needs replacement, normally target the RIGHT segment only and propose "
            "a concrete alternative supported by its narration and the source.\n\n"
            "RETURN EXACTLY ONE JSON OBJECT containing pair_reviews, with EXACTLY one entry "
            "for every adjacent pair, in order. For pair N and N+1 include: left_segment_index "
            "(N), right_segment_index (N+1), left_visual_action (a short description of the "
            "left prompt's actual main action), right_visual_action (the right prompt's actual "
            "main action), left_visual_composition (the left prompt's main framing/focal "
            "subject/pose), right_visual_composition (the right prompt's main framing/focal "
            "subject/pose), narrated_beats_are_distinct (boolean), "
            "same_primary_action (boolean), same_primary_composition (boolean), "
            "adds_meaningful_difference (boolean), materially_repetitive (boolean), "
            "issue (string), and suggested_focus (string). If not materially repetitive, "
            "issue and suggested_focus should be empty. Never omit a pair. "
            "When adjacent segments describe the same contextual state (for "
            "example, a family continuing to live in the same place), suitable "
            "depictions of that state may remain similar across clips; do not force "
            "a new action or invent a time-passing device merely to create variety.\n"
        )

    def verify_visual_pairs(
        self,
        scripture,
        segments,
        visuals,
    ):
        """Independently audit every adjacent pair for substantial visual repetition."""
        expected_pairs = max(0, len(segments) - 1)
        if expected_pairs == 0:
            return []
        if len(segments) != len(visuals):
            raise ContentGenerationError(
                "Visual pair review requires exactly one visual per narration segment."
            )

        prompt = self.build_visual_pairwise_review_prompt(
            scripture,
            segments,
            visuals,
        )
        try:
            response = self.llm.generate(
                prompt,
                response_format=VISUAL_PAIRWISE_VERIFICATION_SCHEMA,
            )
        except Exception as error:
            raise ContentGenerationError(
                f"Adjacent visual-pair review could not run: {error}"
            ) from error

        try:
            verdict = self.parse_direction(response)
        except ContentGenerationError as error:
            raise ContentGenerationError(
                "Adjacent visual-pair review returned no parseable verdict."
            ) from error

        if not isinstance(verdict, dict) or not isinstance(verdict.get("pair_reviews"), list):
            raise ContentGenerationError(
                "Adjacent visual-pair review did not return pair_reviews."
            )

        raw_pairs = verdict["pair_reviews"]
        if len(raw_pairs) != expected_pairs:
            raise ContentGenerationError(
                "Adjacent visual-pair review must assess every neighboring pair exactly once."
            )

        findings_by_segment = {}
        seen_pairs = set()

        def _normalize_review_description(value):
            return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()

        for item in raw_pairs:
            if not isinstance(item, dict):
                raise ContentGenerationError(
                    "Adjacent visual-pair review returned a malformed pair assessment."
                )

            left_index = item.get("left_segment_index")
            right_index = item.get("right_segment_index")
            if (
                isinstance(left_index, bool)
                or isinstance(right_index, bool)
                or not isinstance(left_index, int)
                or not isinstance(right_index, int)
                or right_index != left_index + 1
                or left_index < 1
                or right_index > len(segments)
                or (left_index, right_index) in seen_pairs
            ):
                raise ContentGenerationError(
                    "Adjacent visual-pair review returned invalid or duplicate segment indices."
                )
            seen_pairs.add((left_index, right_index))

            description_fields = (
                "left_visual_action",
                "right_visual_action",
                "left_visual_composition",
                "right_visual_composition",
            )
            if any(
                not isinstance(item.get(field), str) or not item[field].strip()
                for field in description_fields
            ):
                raise ContentGenerationError(
                    f"Adjacent visual-pair review omitted a non-empty action or composition description for pair {left_index}-{right_index}."
                )

            boolean_fields = (
                "narrated_beats_are_distinct",
                "same_primary_action",
                "same_primary_composition",
                "adds_meaningful_difference",
                "materially_repetitive",
            )
            if any(not isinstance(item.get(field), bool) for field in boolean_fields):
                raise ContentGenerationError(
                    f"Adjacent visual-pair review omitted a required boolean for pair {left_index}-{right_index}."
                )

            same_action_and_composition_without_delta = (
                item["narrated_beats_are_distinct"]
                and item["same_primary_action"]
                and item["same_primary_composition"]
                and not item["adds_meaningful_difference"]
            )
            identical_action_description = (
                _normalize_review_description(item["left_visual_action"])
                and _normalize_review_description(item["left_visual_action"])
                == _normalize_review_description(item["right_visual_action"])
            )
            identical_composition_description = (
                _normalize_review_description(item["left_visual_composition"])
                and _normalize_review_description(item["left_visual_composition"])
                == _normalize_review_description(item["right_visual_composition"])
            )
            repeated_descriptions_without_delta = (
                item["narrated_beats_are_distinct"]
                and identical_action_description
                and identical_composition_description
                and not item["adds_meaningful_difference"]
            )
            # Repetition is meaningful only when the narration moves to a
            # distinct beat. Do not reject similar imagery for adjacent chunks
            # that describe the same continuous action or contextual state,
            # even if the model's raw materially_repetitive flag conflicts.
            materially_repetitive = (
                item["narrated_beats_are_distinct"]
                and (
                    item["materially_repetitive"]
                    or same_action_and_composition_without_delta
                    or repeated_descriptions_without_delta
                )
            )

            self.log(
                f"Visual pair {left_index}-{right_index} audit: "
                f"left action='{item['left_visual_action']}', "
                f"right action='{item['right_visual_action']}', "
                f"left composition='{item['left_visual_composition']}', "
                f"right composition='{item['right_visual_composition']}', "
                f"distinct beats={item['narrated_beats_are_distinct']}, "
                f"same action={item['same_primary_action']}, "
                f"same composition={item['same_primary_composition']}, "
                f"meaningful difference={item['adds_meaningful_difference']}, "
                f"material repetition={materially_repetitive}."
            )

            if materially_repetitive:
                issue = str(item.get("issue") or "").strip()
                if not issue:
                    issue = (
                        f"Segment {right_index} substantially repeats the visual action and "
                        f"composition of segment {left_index}, although the narration moves "
                        "to a distinct story beat. Show the later segment's own event or stage."
                    )
                focus = str(item.get("suggested_focus") or "").strip()
                if not focus:
                    focus = (
                        f"Depict the specific action or stage narrated in segment {right_index}, "
                        f"visibly distinct from segment {left_index}, without inventing details."
                    )
                findings_by_segment[right_index] = {
                    "segment_index": right_index,
                    "issue": issue,
                    "suggested_focus": focus,
                }

        expected_set = {
            (index, index + 1)
            for index in range(1, len(segments))
        }
        if seen_pairs != expected_set:
            raise ContentGenerationError(
                "Adjacent visual-pair review did not cover every neighboring pair exactly once."
            )

        return [findings_by_segment[index] for index in sorted(findings_by_segment)]

    def verify_visual_sequence(
        self,
        scripture,
        segments,
        visuals,
    ):
        """Verify every segment's story-beat coverage and the complete sequence."""
        if len(segments) != len(visuals):
            raise ContentGenerationError(
                "Visual sequence review requires exactly one visual per narration segment."
            )

        prompt = self.build_visual_sequence_review_prompt(
            scripture,
            segments,
            visuals,
        )

        try:
            response = self.llm.generate(
                prompt,
                response_format=VISUAL_SEQUENCE_VERIFICATION_SCHEMA,
            )
        except Exception as error:
            raise ContentGenerationError(
                f"Episode-wide visual storytelling review could not run: {error}"
            ) from error

        try:
            verdict = self.parse_direction(response)
        except ContentGenerationError as error:
            raise ContentGenerationError(
                "Episode-wide visual storytelling review returned no parseable verdict."
            ) from error

        if not isinstance(verdict, dict) or not isinstance(verdict.get("passes"), bool):
            raise ContentGenerationError(
                "Episode-wide visual storytelling review did not return an explicit boolean verdict."
            )

        raw_reviews = verdict.get("segment_reviews")
        if not isinstance(raw_reviews, list) or len(raw_reviews) != len(segments):
            raise ContentGenerationError(
                "Episode-wide visual review must assess every segment exactly once."
            )

        reviews_by_segment = {}
        for item in raw_reviews:
            if not isinstance(item, dict):
                raise ContentGenerationError(
                    "Episode-wide visual review returned a malformed segment assessment."
                )
            segment_number = item.get("segment_index")
            if (
                isinstance(segment_number, bool)
                or not isinstance(segment_number, int)
                or segment_number < 1
                or segment_number > len(segments)
                or segment_number in reviews_by_segment
            ):
                raise ContentGenerationError(
                    "Episode-wide visual review returned a missing, duplicate, or invalid segment index."
                )
            if (
                not isinstance(item.get("beat_is_represented"), bool)
                or not isinstance(item.get("materially_repetitive"), bool)
            ):
                raise ContentGenerationError(
                    "Episode-wide visual review omitted an explicit beat-coverage or repetition verdict."
                )
            if any(
                not isinstance(item.get(field), str) or not item[field].strip()
                for field in ("narrated_beat", "depicted_beat")
            ):
                raise ContentGenerationError(
                    f"Episode-wide visual review omitted a meaningful narrated/depicted beat description for segment {segment_number}."
                )
            reviews_by_segment[segment_number] = item
            self.log(
                f"Episode-wide segment {segment_number} audit: "
                f"narrated beat='{str(item.get('narrated_beat') or '').strip()}', "
                f"depicted beat='{str(item.get('depicted_beat') or '').strip()}', "
                f"represented={item['beat_is_represented']}, "
                f"materially repetitive={item['materially_repetitive']}."
            )

        if set(reviews_by_segment) != set(range(1, len(segments) + 1)):
            raise ContentGenerationError(
                "Episode-wide visual review did not provide one assessment for every segment."
            )

        findings_by_segment = {}

        def add_finding(segment_number, issue, suggested_focus=""):
            normalized_issue = str(issue or "").strip()
            if not normalized_issue:
                normalized_issue = (
                    "The visual does not clearly depict the assigned story beat. "
                    "Replace it with a scene showing that segment's actual event."
                )
            entry = findings_by_segment.setdefault(
                segment_number,
                {"issue": [], "suggested_focus": []},
            )
            entry["issue"].append(normalized_issue)
            focus = str(suggested_focus or "").strip()
            if focus:
                entry["suggested_focus"].append(focus)

        # The per-segment audit is authoritative for missing beat coverage or
        # material repetition, even if the top-level reviewer verdict conflicts.
        for segment_number in range(1, len(segments) + 1):
            item = reviews_by_segment[segment_number]
            if not item["beat_is_represented"]:
                add_finding(
                    segment_number,
                    item.get("issue") or (
                        "The prompt does not actually depict the segment's central narrated event. "
                        "Show the correct actor performing that event rather than a related pose, "
                        "prayer, reaction, or distant implication."
                    ),
                    item.get("suggested_focus"),
                )
            if item["materially_repetitive"]:
                add_finding(
                    segment_number,
                    item.get("issue") or (
                        "This shot materially repeats an earlier visual without adding a distinct "
                        "stage or meaningful visual information."
                    ),
                    item.get("suggested_focus"),
                )

        raw_findings = verdict.get("findings")
        if raw_findings is not None and not isinstance(raw_findings, list):
            raise ContentGenerationError(
                "Episode-wide visual review returned malformed findings."
            )
        for item in raw_findings or []:
            if not isinstance(item, dict):
                raise ContentGenerationError(
                    "Episode-wide visual review returned a malformed finding."
                )
            segment_number = item.get("segment_index")
            if (
                isinstance(segment_number, bool)
                or not isinstance(segment_number, int)
                or segment_number < 1
                or segment_number > len(segments)
            ):
                raise ContentGenerationError(
                    "Episode-wide visual review returned an invalid segment index."
                )
            add_finding(
                segment_number,
                item.get("issue"),
                item.get("suggested_focus"),
            )

        # A separate adjacent-pair review is deliberately run even when the
        # episode-wide reviewer reports a pass. The focused comparison catches
        # near-duplicate prompts that can be overlooked in a broad sequence audit.
        pairwise_findings = self.verify_visual_pairs(
            scripture,
            segments,
            visuals,
        )
        for item in pairwise_findings:
            add_finding(
                item["segment_index"],
                item["issue"],
                item.get("suggested_focus", ""),
            )

        findings = []
        for segment_number in sorted(findings_by_segment):
            item = findings_by_segment[segment_number]
            findings.append({
                "segment_index": segment_number,
                "issue": " ".join(dict.fromkeys(item["issue"])),
                "suggested_focus": " ".join(dict.fromkeys(item["suggested_focus"])),
            })

        # Concrete per-segment and pairwise findings are actionable and must
        # trigger targeted regeneration, regardless of the reviewer's overall
        # flag. Conversely, an overall rejection with no segment-level or
        # pairwise finding is internally inconsistent: there is nothing the
        # regeneration loop can target. Do not abort a valid generation for
        # this contradictory summary flag; rely on the exhaustive detailed
        # assessments and record the inconsistency for diagnosis.
        if not findings:
            if not verdict["passes"]:
                self.log(
                    "WARNING: Episode-wide visual reviewer returned passes=false "
                    "but all per-segment assessments and adjacent-pair checks "
                    "identified no actionable issue. Accepting the sequence "
                    "based on the detailed assessments rather than aborting "
                    "the generation."
                )
            return True, []

        return False, findings


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
            "The title must be clear, factual, recognizable, and about what "
            "this passage actually says or does. Write it in plain words.\n"
            "- Consider the entire passage before choosing the title. Identify "
            "its most distinctive event, person or group, and story arc.\n"
            "- Prefer concrete identifying words: name the relevant person or "
            "group, the specific action, or the recognizable biblical event "
            "when the passage makes those details clear. A viewer should be "
            "able to tell what this episode is about from the title alone.\n"
            "- Avoid vague titles made only from broad or abstract topic words. "
            "Do not compress distinct events into generic labels that hide who "
            "acted or what happened. If two connected story beats are both "
            "central, name them plainly when that makes the title more useful.\n"
            "- For narrative passages, prefer a concrete event title that makes "
            "clear who does what. Do not reduce a passage's central events to "
            "abstract noun pairs such as a bare combination of themes or actions; "
            "identify the people or group and the recognizable event instead.\n"
            "- A title does not need to list every event. Choose the most "
            "distinctive story event or connected events, but make the wording "
            "specific enough that a viewer can recognize the passage.\n"
            "- Keep the title concise, but clarity and recognizability matter "
            "more than an artificially short length. Use natural, idiomatic "
            "wording that sounds like a clear description of the episode. When "
            "the people and their actions are clear, prefer naming them and what "
            "they do rather than making an incidental location the focus. Avoid "
            "awkward constructions that make an incidental home or setting sound "
            "like the main event when the passage is about the people's actions.\n"
            "- Use the passage's own language where helpful. Common or "
            "traditional biblical terms for clearly identifiable people, groups, "
            "and events are acceptable even when the term is not repeated in "
            "this exact verse range; do not invent a specific identity that "
            "the passage and its context do not support.\n"
            "- NEVER invent a poetic, symbolic, or literary concept and "
            "attach it to the passage. No metaphors, no invented 'of ...' "
            "pairings, no made-up themes.\n"
            "- NEVER add a theological claim the passage does not state. "
            "Do not assert meaning, promise, or doctrine that is not in "
            "the text.\n"
            "- Do NOT include the Scripture reference. It is added to the "
            "title automatically; do not repeat it yourself.\n"
            "Example - the passage below opens Matthew's genealogy of "
            "Jesus. A correct title is \"The Genealogy of Jesus Christ\". "
            "A WRONG title is \"Genealogy of Light\", which invents a "
            "symbol the passage never mentions.\n\n"
            "SUMMARY RULES\n"
            "Write a short, natural, high-level overview for the episode "
            "description, based on the entire passage. A summary is not a second "
            "narration: capture the central point in a few plain words rather "
            "than retelling the events. The ideal is usually ONE concise sentence "
            "that tells viewers what the passage is mainly about.\n"
            "- Identify the central event or connected storyline and its key "
            "meaningful outcome. Mention another storyline only when it is "
            "essential to understanding the passage; combine connected events "
            "into the main idea instead of listing actions in sequence.\n"
            "- Prefer a simple high-level statement over a detailed account. "
            "Leave secondary actions, transitions, explanations, and supporting "
            "details to the narration. Do not try to mention every important "
            "moment just because it appears in the passage.\n"
            "- Use a second sentence only when one sentence would make genuinely "
            "distinct central storylines unclear or misleading. This is a natural "
            "writing preference, NOT a word, character, or sentence-count limit. "
            "Do not pad or expand the summary to cover the narration's details.\n"
            "- When the passage shifts between different people or groups, "
            "identify the key action of each group when needed to keep the "
            "overview accurate. Keep every subject, action, and result "
            "correctly attributed; never assign one group's actions to "
            "another.\n"
            "- Use names, descriptions, and commonly accepted biblical "
            "terms when the passage and its context clearly identify the "
            "person or group. Do not invent a specific identity that the "
            "text and context do not support.\n"
            "- Do NOT repeat or restate the title, and do not reuse the "
            "title's distinctive wording.\n"
            "- Do NOT include the Scripture reference. It is already in "
            "the title.\n"
            "- Do NOT interpret, and do not add any claim the passage does "
            "not state. Report the content; do not explain its theology.\n"
            "- Include enough context to make the central story understandable, "
            "but leave secondary details, repeated actions, and step-by-step "
            "transitions to the narration. Do not reproduce the narration in "
            "compressed form or restate each sentence. Use as many complete "
            "sentences as the story genuinely needs, with a clear finished "
            "ending. There is no fixed length limit, but concision is preferred "
            "over exhaustive retelling.\n\n"
            "FINAL OUTPUT\n"
            "Return exactly one object with three fields, and nothing "
            "else:\n"
            "- \"title\": a short, clear, factual title, following the "
            "TITLE RULES above. No Scripture reference.\n"
            "- \"summary\": a short, high-level, factual overview of the "
            "passage, following the SUMMARY RULES above. Prefer one concise "
            "sentence. No reference, no unsupported interpretation.\n"
            "- \"mood\": 1-3 lowercase English words for the emotional "
            "tone, used to choose the background music. Choose the style "
            "from the passage: reverent, contemplative, peaceful, solemn, "
            "hopeful, gentle, uplifting, or tender for reflective passages; "
            "suspenseful, tense, storm, fearful, or uncertain when the "
            "passage involves danger, storms, fear, conflict, or uncertainty. "
            "Suspense must remain soft and atmospheric, never upbeat or "
            "aggressive.\n"
        )


    def build_metadata_verification_prompt(
        self,
        scripture,
        segments,
        title,
        summary,
    ):
        """Build one independent review of the title and summary as a pair."""
        channel = self.channel_config.get("channel", {})
        name = str(channel.get("name", "Hear His Voice"))
        full_telling = "\n".join(
            f"{position + 1}. {text}"
            for position, text in enumerate(segments)
        )

        return (
            "You are the independent metadata quality reviewer for \""
            + name
            + "\". Review the episode title and summary against the complete "
            "Scripture passage and finished narration. Evaluate the two fields "
            "separately, while also checking that they complement rather than "
            "repeat one another. Mood is not part of this review.\n\n"
            "SCRIPTURE REFERENCE\n"
            + scripture["reference"]
            + " ("
            + scripture["translation"]
            + ")\n\nSOURCE PASSAGE\n"
            + str(scripture.get("text") or "").strip()
            + "\n\nFINISHED NARRATION\n"
            + full_telling
            + "\n\nPROPOSED TITLE (the code adds the Scripture reference separately)\n"
            + str(title or "").strip()
            + "\n\nPROPOSED SUMMARY\n"
            + str(summary or "").strip()
            + "\n\nTITLE REVIEW\n"
            "Set title_passes=true only when the title is accurate to the passage, "
            "natural and idiomatic in its wording, recognizable and specific enough to tell "
            "a viewer what the episode is about, and appropriately concise. "
            "For narrative passages, prefer a concrete event, identifiable "
            "person/group, or recognizable biblical event over vague abstract "
            "labels. A title made only of broad topic words is too generic "
            "when it hides who acts or what specifically happens. The title "
            "does not need to list every event; it should identify the most "
            "distinctive central event or connected story. Accept traditional "
            "biblical terminology supported by the passage and context, "
            "including terms such as Magi where appropriate. Do not require "
            "one specific wording or title formula. Also reject clearly awkward "
            "wording that foregrounds an incidental place instead of the people "
            "and event, when a more natural subject-action title is available. "
            "Reject only a material inaccuracy, unsupported claim, or a title "
            "that is genuinely too vague or unnatural to identify the episode.\n\n"
            "SUMMARY REVIEW\n"
            "Set summary_passes=true when the summary is a clear, natural, "
            "factually supported HIGH-LEVEL overview of the passage's central "
            "event or connected storyline. It should add useful information "
            "beyond the title, not merely restate it. The desired style is short "
            "and sweet: usually ONE concise sentence that communicates the main "
            "point without walking through the events. Prefer a simple central "
            "idea and meaningful outcome over a list of actions, secondary "
            "details, or chronological transitions. Use a second sentence only "
            "when genuinely distinct central storylines cannot be conveyed "
            "clearly in one. If the summary could be substantially simplified "
            "without losing the passage's central point, mark it for revision. "
            "This is a qualitative writing standard, NOT a word, character, or "
            "sentence-count limit; do not enforce a numeric cutoff. A concise "
            "summary may omit secondary actions, details, and long quotations. "
            "Do not require every event or line of the passage, and do not treat "
            "ordinary high-level compression as a material omission. Accept "
            "valid biblical terminology, including Magi where appropriate. "
            "Reject unsupported details, theological interpretation not stated "
            "in the text, or a summary so generic that it is not useful. It must "
            "be a complete, coherent description with a finished ending; reject "
            "visibly truncated output or an unfinished final clause. "
            "Subject/action attribution is evaluated separately in "
            "summary_attribution_passes.\n\n"
            "SUMMARY ATTRIBUTION REVIEW (CHECK ONLY CLAIMS THE SUMMARY ACTUALLY MAKES)\n"
            "Set summary_attribution_passes=false ONLY when the summary clearly "
            "assigns an action, warning, spoken message, journey, destination, "
            "or outcome to the wrong person or group. Check the meaning of each "
            "clause in context, not just its nearest grammatical subject. "
            "Distinguish the person who receives a message from the person who "
            "delivers or speaks it: for example, 'Joseph received a message "
            "from an angel' correctly makes Joseph the recipient and the angel "
            "the messenger; it does NOT claim Joseph spoke the message. Likewise, "
            "'the angel warned Joseph to flee' correctly makes the angel the "
            "speaker and Joseph the recipient of the instruction. Do not treat "
            "the words inside an embedded quotation as an action performed by "
            "the grammatical subject of the surrounding clause. For example, "
            "'Joseph received the angel's message: [quoted instruction]' does "
            "not attribute the instruction to Joseph. Treat a prophecy or "
            "fulfillment clause as an outcome/context, not as words spoken by "
            "the people experiencing that fulfillment. If the summary says a "
            "family remained in a place, that does not mean the family spoke "
            "the prophecy associated with the event. Attribute quoted prophetic "
            "words to the speaker identified by the source only when the summary "
            "itself makes a claim about who spoke them.\n"
            "Evaluate only actions and claims actually stated or clearly implied "
            "by the summary. Do NOT require the summary to list all source "
            "actions, all source actors, every outcome, or the full wording of "
            "any quotation; coverage belongs to summary_passes and concise "
            "summaries necessarily omit details. Do not fail because a source "
            "action or quotation is absent. If a summary's wording is compressed "
            "but its intended actor/recipient is clear and supported, pass it. "
            "Fail only for a clear, material misattribution or unsupported claim. "
            "When failing, identify the exact phrase, who it appears to assign "
            "the action to, and who actually performs or receives it according "
            "to the source.\n\n"
            "SHARED RULES\n"
            "Use the whole passage, not only its opening or ending. The source "
            "passage is authoritative; the narration is a faithful retelling "
            "that may naturally paraphrase it. Do not demand that metadata "
            "repeat the source wording. Do not reject reasonable stylistic "
            "choices, synonymous wording, or accepted biblical terms. Be "
            "practical rather than perfectionistic: pass a good, accurate, "
            "useful title or summary. Fail only when there is a clear, "
            "meaningful problem. The Scripture reference is appended to the "
            "title by code and should not be included in the proposed title.\n\n"
            "RETURN exactly one JSON object with six fields: title_passes "
            "(boolean), title_issues (string), summary_passes (boolean), "
            "summary_issues (string), summary_attribution_passes (boolean), "
            "and summary_attribution_issues (string). The attribution verdict "
            "is a separate mandatory check and must be false whenever any major "
            "action, actor, group, warning, destination, or outcome is assigned "
            "to the wrong subject. A summary passes overall only when BOTH "
            "summary_passes and summary_attribution_passes are true. For a "
            "passing check, its issues string must be empty. For a failing "
            "check, explain the specific subject/action error and what a "
            "correction should accomplish. Judge title and summary separately; "
            "one may pass while the other fails.\n"
        )

    def verify_metadata(
        self,
        scripture,
        segments,
        title,
        summary,
    ):
        """Return independent title and summary verdicts from one review call."""
        prompt = self.build_metadata_verification_prompt(
            scripture,
            segments,
            title,
            summary,
        )

        try:
            response = self.llm.generate(
                prompt,
                response_format=METADATA_VERIFICATION_SCHEMA,
            )
        except Exception as error:
            raise ContentGenerationError(
                f"Title and summary verification could not run: {error}"
            ) from error

        try:
            verdict = self.parse_direction(response)
        except ContentGenerationError as error:
            raise ContentGenerationError(
                f"Title and summary verification returned no parseable verdict: {error}"
            ) from error

        if (
            not isinstance(verdict, dict)
            or not isinstance(verdict.get("title_passes"), bool)
            or not isinstance(verdict.get("summary_passes"), bool)
            or not isinstance(verdict.get("summary_attribution_passes"), bool)
        ):
            raise ContentGenerationError(
                "Title and summary verification did not return explicit "
                "title, summary, and summary-attribution verdicts."
            )

        title_issues = str(verdict.get("title_issues") or "").strip()
        summary_issues = str(verdict.get("summary_issues") or "").strip()
        summary_attribution_issues = str(
            verdict.get("summary_attribution_issues") or ""
        ).strip()
        summary_shape_issue = _summary_completeness_issue(summary)

        if summary_shape_issue:
            summary_issues = " ".join(
                part for part in (summary_issues, summary_shape_issue) if part
            )

        if not verdict["title_passes"] and not title_issues:
            title_issues = (
                "The title does not adequately identify the episode. "
                "Make it more specific and recognizable without adding unsupported details."
            )
        if not verdict["summary_passes"] and not summary_issues:
            summary_issues = (
                "The summary has a meaningful accuracy, coverage, attribution, "
                "or usefulness problem; correct it using the complete passage."
            )
        if not verdict["summary_attribution_passes"] and not summary_attribution_issues:
            summary_attribution_issues = (
                "The summary assigns a major action, warning, destination, or "
                "outcome to the wrong person or group. Map each action to its "
                "actual subject in the source and correct the sentence."
            )

        if not verdict["summary_attribution_passes"]:
            summary_issues = " ".join(
                part for part in (summary_issues, summary_attribution_issues) if part
            )

        return {
            "title_passes": verdict["title_passes"],
            "title_issues": title_issues,
            "summary_passes": (
                verdict["summary_passes"]
                and verdict["summary_attribution_passes"]
                and not summary_shape_issue
            ),
            "summary_issues": summary_issues,
            "summary_attribution_passes": verdict["summary_attribution_passes"],
            "summary_attribution_issues": summary_attribution_issues,
        }

    def build_metadata_repair_prompt(
        self,
        scripture,
        segments,
        title,
        summary,
        failed_fields,
        feedback,
    ):
        """Request replacements only for metadata fields that failed review."""
        channel = self.channel_config.get("channel", {})
        name = str(channel.get("name", "Hear His Voice"))
        full_telling = "\n".join(
            f"{position + 1}. {text}"
            for position, text in enumerate(segments)
        )
        failed_fields = set(failed_fields)
        requested = []
        if "title" in failed_fields:
            requested.append(
                '"title": replace the rejected title with a concise, specific, '
                "recognizable title grounded in the central event or connected "
                "story of the entire passage. Identify the people/group or "
                "recognizable biblical event when useful. Use natural, idiomatic "
                "wording and avoid making an incidental location the focus when "
                "the main point is what the people do. Avoid abstract two-topic "
                "labels that conceal the people or events being described."
            )
        if "summary" in failed_fields:
            requested.append(
                '"summary": replace the rejected summary with a short, natural, '
                "high-level overview of the passage's central event or connected "
                "storyline. Aim for ONE concise sentence that tells viewers what "
                "the passage is mainly about. Distill the central point instead "
                "of retelling events beat by beat or paraphrasing the narration. "
                "Use a second sentence only when genuinely distinct central "
                "storylines need it for clarity. There is no fixed word, "
                "character, or sentence-count limit. Do "
                "not add filler or reproduce long quotations unnecessarily. "
                "When multiple people or groups take different actions, keep "
                "each action, warning, speaker, recipient, destination, and outcome "
                "attached to the correct person or group. Distinguish who delivers "
                "a message from who receives it. Make the summary add useful "
                "information beyond the title without repeating it. Ensure the "
                "final thought is complete and ends cleanly; do not stop mid-clause "
                "or leave a dangling final word."
            )

        locked = []
        if "title" not in failed_fields:
            locked.append(f"Keep this approved title exactly unchanged: {title}")
        if "summary" not in failed_fields:
            locked.append(f"Keep this approved summary exactly unchanged: {summary}")

        return (
            "You are correcting episode metadata for \""
            + name
            + "\" after an independent review found specific problems. "
            "Correct only the rejected field(s); preserve every approved "
            "field exactly as instructed.\n\n"
            "SCRIPTURE REFERENCE\n"
            + scripture["reference"]
            + " ("
            + scripture["translation"]
            + ")\n\nSOURCE PASSAGE\n"
            + str(scripture.get("text") or "").strip()
            + "\n\nFINISHED NARRATION\n"
            + full_telling
            + "\n\nCURRENT TITLE\n"
            + str(title or "").strip()
            + "\n\nCURRENT SUMMARY\n"
            + str(summary or "").strip()
            + "\n\nREVIEW FEEDBACK\n"
            + str(feedback or "").strip()
            + "\n\nAPPROVED FIELDS - DO NOT CHANGE\n"
            + ("\n".join(locked) if locked else "No field is currently approved.")
            + "\n\nFIELDS TO REPAIR\n"
            + "\n".join(f"- {item}" for item in requested)
            + "\n\nRULES\n"
            "Use the entire passage and narration. Be accurate, concrete, and "
            "natural. Do not invent details, add theological interpretation, "
            "or include the Scripture reference in the title. Commonly accepted "
            "biblical terminology, including Magi when appropriate, is valid. "
            "Do not force the title to list every event. The summary should "
            "complement the title rather than repeat it, and should be a short, "
            "high-level overview rather than a compressed retelling of the "
            "narration. Aim for one concise sentence capturing the central point; "
            "use a second only when distinct central storylines genuinely require "
            "it. Leave secondary details and chronological beats to the narration. "
            "There is no fixed word, character, or sentence-count limit. Ensure it is coherent "
            "and that its final sentence ends cleanly rather than stopping "
            "mid-clause. Return only the "
            "requested field(s) as one JSON object.\n"
        )

    @staticmethod
    def build_metadata_repair_schema(failed_fields):
        """Build a strict response schema containing only fields needing repair."""
        properties = {}
        if "title" in failed_fields:
            properties["title"] = {"type": "string", "maxLength": 90}
        if "summary" in failed_fields:
            properties["summary"] = {"type": "string"}
        return {
            "type": "object",
            "properties": properties,
            "required": list(properties),
        }

    def generate_metadata(
        self,
        scripture,
        segments,
    ):
        """Generate title/summary, independently verify each, and repair failures."""
        self.log("Generating metadata (title, summary, mood)...")
        metadata_prompt = self.build_metadata_prompt(
            scripture,
            segments,
        )
        metadata_response = self.llm.generate(
            metadata_prompt,
            response_format=METADATA_SCHEMA,
        )
        metadata = self.parse_direction(metadata_response)
        if not isinstance(metadata, dict):
            raise ContentGenerationError("Metadata generation returned no usable object.")

        reference = scripture["reference"]
        title = _strip_reference(
            str(metadata.get("title") or "").strip(),
            reference,
        )
        summary = _strip_reference(
            str(metadata.get("summary") or "").strip(),
            reference,
        )
        mood = metadata.get("mood", "")

        review_round = 0
        while True:
            review_round += 1

            # Apply the same narrow title/summary de-duplication check before
            # verification that assemble_content uses when assembling output.
            # This ensures the reviewer sees the exact summary that can ship;
            # an almost-verbatim title echo is sent back for repair instead of
            # being silently replaced after it has passed review.
            if summary and _restates_title(summary, title, reference):
                summary = ""

            verdict = self.verify_metadata(
                scripture,
                segments,
                title,
                summary,
            )

            title_passes = bool(title) and verdict["title_passes"]
            summary_passes = bool(summary) and verdict["summary_passes"]
            title_issues = verdict["title_issues"]
            summary_issues = verdict["summary_issues"]

            if not title and not title_issues:
                title_issues = "The generated title was empty. Provide a specific, recognizable title."
            if not summary and not summary_issues:
                summary_issues = "The generated summary was empty. Provide a useful factual description of the passage."

            if title_passes and summary_passes:
                self.log(
                    f"Title and summary passed metadata verification on round {review_round}."
                )
                break

            failed_fields = []
            feedback = []
            if not title_passes:
                failed_fields.append("title")
                feedback.append("TITLE FAILED: " + title_issues)
            if not summary_passes:
                failed_fields.append("summary")
                feedback.append("SUMMARY FAILED: " + summary_issues)

            self.log(
                "Metadata verification rejected "
                + " and ".join(failed_fields)
                + f" on round {review_round}; repairing only those field(s)..."
            )

            repair_prompt = self.build_metadata_repair_prompt(
                scripture,
                segments,
                title,
                summary,
                failed_fields,
                "\n".join(feedback),
            )
            repair_response = self.llm.generate(
                repair_prompt,
                response_format=self.build_metadata_repair_schema(failed_fields),
            )
            repaired = self.parse_direction(repair_response)
            if not isinstance(repaired, dict):
                raise ContentGenerationError(
                    "Metadata repair returned no usable JSON object."
                )

            for field in failed_fields:
                replacement = _strip_reference(
                    str(repaired.get(field) or "").strip(),
                    reference,
                )
                if replacement:
                    if field == "title":
                        title = replacement
                    else:
                        summary = replacement
                else:
                    self.log(
                        f"Metadata repair returned an empty {field}; "
                        "the field will be reviewed again."
                    )

        return {
            "title": title,
            "summary": summary,
            "mood": mood,
        }


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

        # The reference comes first so the episode is identifiable from
        # the listing without a separate field. The separator is an
        # em dash, and the reference uses an en dash (–) between the
        # verse ranges. A title carries no full stop, because the
        # reference supplies the ending.
        title = "{} — {}".format(
            reference,
            title.rstrip(" ."),
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

        if (
            not summary
            or _restates_title(summary, title, reference)
            or _summary_completeness_issue(summary)
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

        # Put the source's actual direct-speech spans in a separate copy
        # reference. The full passage remains authoritative; this simply makes
        # exact quotations easier for the writer to reproduce without
        # accidentally paraphrasing them while rewriting surrounding prose.
        source_text = str(scripture.get("text") or "").strip()
        source_quote_spans, _source_quote_problems = _scan_quoted_spans(source_text)
        outer_source_quotes = [
            (quoted, start, end)
            for quoted, start, end in source_quote_spans
            if not any(
                other_start < start and other_end > end
                for _other, other_start, other_end in source_quote_spans
            )
        ]
        source_quotes_section = ""
        if outer_source_quotes:
            source_quotes_section = (
                "VERBATIM SOURCE QUOTATIONS - EXACT COPY REFERENCES\n"
                "These are the source's quoted speech spans, copied directly "
                "from the authoritative passage. They are optional to use, "
                "not a requirement to quote every speech. If you use one as "
                "direct speech, copy the complete selected sentence(s) exactly "
                "as shown, preserving every word and its order. Never shorten, "
                "modernize, or paraphrase wording inside quotation marks. If "
                "you want to retell only part of a saying, paraphrase that part "
                "outside quotation marks instead.\n\n"
                + "\n\n".join(
                    f"SOURCE QUOTATION {index}:\n{quoted}"
                    for index, (quoted, _start, _end) in enumerate(outer_source_quotes, 1)
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
            "because that order of events seems tidier. Make the speaker "
            "and actor explicit whenever a pronoun could refer to more "
            "than one person. Account for the important claims and events "
            "from the opening of the passage through its ending; do not "
            "skip an opening statement, identity claim, or stated purpose "
            "just to begin with a later or more familiar event. "
            "The result must be a purposeful hybrid, "
            "not a complete read-through and not a paraphrase of every line; "
            "your job is not to quote it back wholesale. Except for selected "
            "verbatim quotations, do not follow the source sentence by sentence "
            "or preserve its wording and syntax mechanically. Build natural "
            "spoken sentences that connect related actions and carry the listener "
            "through the events. Keep necessary names, biblical terms, and key "
            "phrases, but express the surrounding narration in your own clear "
            "storytelling voice. Do not add drama or details that the passage "
            "does not establish.\n"
            "When you use a direct quote, copy the complete relevant saying "
            "exactly from source_text. Never omit, replace, add, or paraphrase "
            "any word inside quotation marks. Start and end selected quotations "
            "at complete source-sentence boundaries; never clip a sentence or "
            "leave out its continuing clause. Apply this with special care to "
            "Jesus's words. If you cannot reproduce the complete relevant saying "
            "exactly, narrate it faithfully outside quotation marks. "
            "Keep quotation marks balanced, and "
            "keep WEBC translator explanations in parentheses outside spoken "
            "quotations. Do not expand contractions or normalize source wording. "
            "Preserve contractions, spelling, and every source word exactly; for "
            "example, do not expand a contraction such as 'didn’t' into 'did not'. "
            "If you cannot reproduce a saying exactly, narrate it outside quotation marks. "
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
            + source_text
            + "\n\n"
            + source_quotes_section
            + "\n\n"
            "NARRATION RULES\n"
            + rules
            + "\n\nSPOKEN LENGTH\n"
            f"Each clip of {_duration_target_seconds(self.config):g} "
            f"seconds carries at most "
            f"{_max_words_per_segment(self.config)} words of narration "
            f"when the words are spoken at the configured rate, and the "
            f"cut between clips falls between words - it may land inside "
            f"a sentence. Write the telling naturally; the segmentation "
            f"code groups the words into clips afterwards. A short "
            f"ending is never padded."
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
                "\n\nYOUR PREVIOUS NARRATION WAS REJECTED BY A QUALITY CHECK\n"
                "Previous versions failed one or more requirements. The issues "
                "below are cumulative: keep every listed requirement satisfied "
                "together, even if a later attempt fixed only one of them.\n"
                "Known problems to avoid reintroducing:\n"
                + feedback
                + "\nWrite the complete narration again. Correct the identified "
                "problems while preserving every material event and meaning, "
                "keeping direct quotations exact, and telling the passage in "
                "natural original spoken prose rather than copying its unquoted "
                "sentences. Do not solve a style issue by omitting important "
                "content or inventing details.\n"
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
            "- Coverage from beginning to end: make an internal ordered "
            "inventory of the passage's material events, claims, testimony, "
            "identities, relationships, and stated purposes. Check the opening, "
            "middle, and ending separately. A missing opening claim or stated "
            "purpose is a failure even when the later narration is accurate. "
            "Do not require repetitive or minor incidental wording, but do "
            "require every defining claim, event, and causal/purpose detail.\n"
            "- Events and order: are the passage's events present and "
            "in the passage's order? Anything material the passage states "
            "but the narration drops, and anything the narration has that "
            "the passage never states - invented dialogue, motives, "
            "feelings, events - is a failure.\n"
            "- Speaker attribution: is it unmistakably clear who speaks, "
            "looks, asks, answers, and acts? Reject a pronoun when it could "
            "reasonably assign the words or action to the wrong person.\n"
            "- Quotations: is every direct quotation the exact source wording, "
            "complete for the relevant saying, and attributed to the passage's "
            "own speaker? Check the actual words, not just the meaning. An exact "
            "substring is not sufficient if it clips the beginning, continuing "
            "clause, condition, reason, or conclusion of the selected saying. "
            "A complete sentence or a coherent set of complete consecutive "
            "sentences may be selected from a longer speech when it preserves a "
            "self-contained saying; do not require unrelated surrounding speech "
            "or the entire passage to be quoted. Apply this especially carefully "
            "to Jesus's words: never approve a paraphrase, modernization, or "
            "shortened fragment presented as His direct words.\n"
            "- Punctuation: are quotation marks balanced and correctly nested? "
            "Accept any consistent straight or curly quote style; do not "
            "reject solely for single-versus-double style.\n"
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


    def build_narration_storytelling_prompt(
        self,
        scripture,
        narration
    ):
        """Build a contextual check for natural, original spoken storytelling."""
        channel = self.channel_config.get("channel", {})
        name = str(channel.get("name", "Hear His Voice"))

        return (
            "You are the storytelling-quality reviewer for \""
            + name
            + "\". The narration has already been checked for biblical "
            "accuracy, completeness, speaker attribution, and exact quotations. "
            "Your task is to decide whether it tells the passage naturally and "
            "engagingly in original spoken prose, rather than reading the source "
            "back almost word for word.\n\n"
            "SOURCE PASSAGE\n"
            + str(scripture.get("text") or "").strip()
            + "\n\nNARRATION TO REVIEW\n"
            + str(narration or "").strip()
            + "\n\n"
            "WHAT GOOD STORYTELLING MEANS\n"
            "- The listener should hear a connected telling of what happens, "
            "not a sentence-by-sentence reading of the source with only small "
            "word substitutions.\n"
            "- For narrative passages, the unquoted narration should use its own "
            "natural spoken sentence structure, connect related actions and "
            "transitions, and guide the listener through the events.\n"
            "- Preserve the passage's actual sequence, meaning, important "
            "participants, relationships, causes, and outcomes. Original wording "
            "must never come at the expense of accuracy or completeness.\n"
            "- Direct quotations are intentionally verbatim. Do not count quoted "
            "words against originality and do not ask for valid quotations to "
            "be paraphrased. The separate quotation check handles their exactness.\n"
            "- Necessary biblical terms, names, titles, short familiar phrases, "
            "and wording for which there is no natural alternative are acceptable. "
            "Some overlap with the source is expected. A small number of ordinary "
            "event descriptions may naturally resemble the source and are not, by "
            "themselves, a reason to reject.\n"
            "- Judge the kind of passage fairly. A genealogy, list, teaching, "
            "prayer, poem, or series of sayings should not be forced into a "
            "dramatic story. Preserve names, relationships, teaching, and the "
            "character of the material while still avoiding an unnecessary "
            "word-for-word read-through of unquoted prose.\n"
            "- The delivery should sound clear and natural when read aloud. It "
            "may be vivid and engaging, but must not manufacture suspense, "
            "emotions, motives, dialogue, or details that the passage does not "
            "establish. Do not demand theatrical or sensational language.\n\n"
            "WHEN TO REJECT\n"
            "Reject only when the unquoted narration is clearly and substantially "
            "a source-text read-through: for example, when substantial portions "
            "of the non-quoted prose repeatedly preserve the source's sentence "
            "construction and clause-by-clause phrasing with only minor substitutions, "
            "so the narration as a whole sounds like the source being read aloud "
            "rather than retold. Also reject when the unquoted writing is so stiff "
            "or clause-by-clause that it does not function as natural spoken narration.\n"
            "Do NOT reject based on one isolated sentence, a small number of short "
            "phrases, or ordinary descriptions of necessary events that naturally "
            "share wording with the source. To reject, you must identify a sustained "
            "pattern across substantial unquoted portions of the narration, not just "
            "cite one or two overlapping phrases. Events may remain in their original "
            "order, biblical terminology may be retained, and a short passage may "
            "leave limited room for rewording. Do not use a numerical overlap score, "
            "percentage, word-count threshold, or requirement that every sentence "
            "be rewritten. Borderline cases pass.\n\n"
            "RETURN EXACTLY ONE JSON OBJECT\n"
            "- passes: true when the narration is a natural, sufficiently original "
            "telling for this kind of passage; otherwise false.\n"
            "- issues: when false, identify the sustained pattern across the "
            "unquoted narration and cite representative substantial portions. "
            "Do not fail based only on isolated phrases. Ask for more natural spoken "
            "retelling without dropping content. When true, return an empty string.\n"
        )

    def verify_narration_storytelling(
        self,
        scripture,
        narration
    ):
        """Return (passes, issues) for the independent storytelling-style check."""
        prompt = self.build_narration_storytelling_prompt(
            scripture,
            narration,
        )

        try:
            response = self.llm.generate(
                prompt,
                response_format=NARRATION_STORYTELLING_SCHEMA,
            )
        except Exception as error:
            raise ContentGenerationError(
                f"Narration storytelling verification could not run: {error}"
            ) from error

        try:
            verdict = self.parse_direction(response)
        except ContentGenerationError:
            return False, "The storytelling check returned no parseable verdict."

        if not isinstance(verdict, dict) or not isinstance(verdict.get("passes"), bool):
            return False, "The storytelling check did not return an explicit boolean verdict."

        if verdict["passes"]:
            return True, ""

        issues = str(verdict.get("issues") or "").strip()
        return False, issues or "The unquoted narration reads too much like the source instead of natural storytelling."

    def verify_narration_coverage(
        self,
        scripture,
        narration
    ):
        """Check that the retelling preserves every material part of the source."""
        channel = self.channel_config.get("channel", {})
        name = str(channel.get("name", "Hear His Voice"))
        prompt = (
            "You are the completeness checker for \""
            + name
            + "\". Compare the narration with the entire source passage. "
            "Your only task is to find material omissions.\n\n"
            "SOURCE PASSAGE - READ FROM FIRST WORD TO LAST\n"
            + str(scripture.get("text") or "").strip()
            + "\n\nNARRATION\n"
            + str(narration or "").strip()
            + "\n\n"
            "Make an internal ordered inventory of the passage's essential "
            "events, statements, testimony, identities, relationships, and "
            "stated purposes. Check the beginning, middle, and ending of the "
            "source separately. A narration that starts partway through the "
            "passage or skips an opening claim or purpose is incomplete, even "
            "if everything it does include is accurate. Preserve claims that "
            "establish who someone is, why an action is done, or what a "
            "testimony declares.\n\n"
            "This is a purposeful retelling, not a full read-through: do not "
            "reject the omission of repetitive wording, minor incidental "
            "details, or stylistic phrasing that does not change the story. "
            "Reject any missing event, defining claim, theological statement, "
            "relationship, or causal/purpose detail that materially contributes "
            "to the passage. Do not confuse paraphrase with omission.\n\n"
            "RETURN exactly one JSON object with: accurate (boolean) and "
            "issues (string). Set accurate=true only when all material parts "
            "are represented in the correct order. When false, list the exact "
            "omitted part(s) and what must be restored.\n"
        )

        try:
            response = self.llm.generate(
                prompt,
                response_format=NARRATION_VERIFICATION_SCHEMA
            )
        except Exception as error:
            raise ContentGenerationError(
                f"Narration completeness verification could not run: {error}"
            ) from error

        try:
            verdict = self.parse_direction(response)
        except ContentGenerationError:
            return False, "The completeness check returned no parseable verdict."

        if not isinstance(verdict, dict) or not isinstance(verdict.get("accurate"), bool):
            return False, "The completeness check did not return an explicit boolean verdict."

        if verdict["accurate"]:
            return True, ""

        issues = str(verdict.get("issues") or "").strip()
        return False, issues or "The narration omits material content from the source passage."


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
        deterministic_issues = (
            narration_quote_problems(
                scripture.get("text", ""),
                narration,
            )
            + narration_speaker_problems(
                scripture.get("text", ""),
                narration,
            )
        )
        if deterministic_issues:
            return False, "\n".join(deterministic_issues)

        complete, coverage_issues = self.verify_narration_coverage(
            scripture,
            narration
        )
        if not complete:
            return False, coverage_issues

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
            raise ContentGenerationError(
                f"Narration accuracy verification could not run: {error}"
            ) from error

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

        if verdict.get("accurate") is not True:
            issues = str(
                verdict.get("issues") or ""
            ).strip()

            if not issues:
                issues = (
                    "The narration does not faithfully tell the passage."
                )

            return False, issues

        storytelling_passes, storytelling_issues = self.verify_narration_storytelling(
            scripture,
            narration,
        )
        if not storytelling_passes:
            return False, (
                "STORYTELLING QUALITY: " + storytelling_issues
            )

        return True, ""


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
        """Generate narration and keep rerolling until every enabled check passes."""
        verify = bool(
            self.generation_config.get("verify_narration", True)
        )

        self.log("Writing the narration...")
        # Keep the latest actionable feedback for each failure category across
        # retries. A new attempt can fix one category while regressing another;
        # replacing the previous feedback with only the latest rejection caused
        # the quote/storytelling failures to alternate without converging.
        feedback_by_category = {}

        def remember_feedback(category, message):
            cleaned = str(message or "").strip()
            if cleaned:
                feedback_by_category[category] = cleaned

        def render_feedback():
            if not feedback_by_category:
                return None
            sections = [
                "The following requirements have failed in earlier attempts. "
                "Keep ALL of them satisfied in the next narration; correcting "
                "one issue must not reintroduce another:"
            ]
            for category, message in feedback_by_category.items():
                sections.append(f"{category}:\n- {message}")
            return "\n\n".join(sections)

        attempt = 0

        while True:
            attempt += 1
            prompt = self.build_narration_prompt(
                scripture,
                feedback=render_feedback()
            )

            try:
                response = self.llm.generate(
                    prompt,
                    response_format=NARRATION_SCHEMA
                )
            except Exception as error:
                raise ContentGenerationError(
                    f"Narration generation could not run: {error}"
                ) from error

            try:
                narration = self.parse_narration(response)
            except ContentGenerationError as error:
                remember_feedback(
                    "Response format",
                    f"The previous narration response was unusable: {error}. "
                    "Return a complete JSON object with a non-empty narration field.",
                )
                self.log(
                    f"Narration attempt {attempt} rejected ({error}) - writing again..."
                )
                continue

            narration = repair_source_word_punctuation(
                scripture.get("text", ""),
                narration,
            )

            deterministic_issues = (
                narration_quote_problems(
                    scripture.get("text", ""),
                    narration,
                )
                + narration_speaker_problems(
                    scripture.get("text", ""),
                    narration,
                )
            )
            if deterministic_issues:
                for issue in deterministic_issues:
                    lowered = issue.lower()
                    if "only part of a sentence" in lowered or "complete source sentence" in lowered:
                        category = "Quotation completeness"
                    elif "direct quotation" in lowered or "quoted" in lowered:
                        category = "Quotation exactness"
                    elif "speaker" in lowered or "attribution" in lowered:
                        category = "Speaker attribution"
                    else:
                        category = "Deterministic narration fidelity"
                    remember_feedback(category, issue)
                self.log(
                    f"Narration attempt {attempt} rejected ({'; '.join(deterministic_issues)}) - writing again..."
                )
                continue

            if not verify:
                self.log(
                    f"Narration ready ({len(narration.split())} words; verification disabled)."
                )
                return narration

            accurate, issues = self.verify_narration(
                scripture,
                narration
            )

            if accurate:
                self.log(
                    f"Narration passed all quality checks on attempt {attempt} "
                    f"({len(narration.split())} words)."
                )
                return narration

            if str(issues).startswith("STORYTELLING QUALITY:"):
                remember_feedback(
                    "Storytelling originality",
                    str(issues).split(":", 1)[1].strip(),
                )
            elif any(
                marker in str(issues).lower()
                for marker in ("omits", "omitted", "missing", "incomplete", "coverage")
            ):
                remember_feedback("Content completeness", issues)
            else:
                remember_feedback("Biblical accuracy", issues)

            self.log(
                f"Narration attempt {attempt} rejected ({issues}) - writing again..."
            )

    def _generate_visual_for_segment(
        self,
        scripture,
        segments,
        index,
        prior_visuals=None,
        verify=True,
        sequence_feedback=None,
    ):
        """Generate one prompt and keep rerolling until its existing checks pass."""
        rejected_attempts = []

        while True:
            feedback_parts = []
            if rejected_attempts:
                unique_reasons = list(dict.fromkeys(
                    item["reason"] for item in rejected_attempts
                ))
                feedback_parts.append(
                    "Rejection reasons from previous attempts:\n"
                    + "\n".join(f"- {reason}" for reason in unique_reasons)
                    + "\n\nMost recent rejected visual:\n"
                    + rejected_attempts[-1]["visual"]
                )

            if sequence_feedback:
                feedback_parts.append(sequence_feedback)

            feedback = "\n\n".join(feedback_parts)
            prompt = self.build_segment_prompt(
                scripture,
                segments,
                index,
                feedback=feedback or None,
                prior_visuals=prior_visuals or [],
            )

            response = self.llm.generate(
                prompt,
                response_format=VISUAL_SCHEMA
            )

            try:
                parsed = self.parse_direction(response)
            except ContentGenerationError as error:
                reason = f"The visual response was not valid JSON: {error}"
                rejected_attempts.append({
                    "visual": "(unparseable response)",
                    "reason": reason,
                })
                self.log(
                    f"Segment {index + 1}: rejected ({reason}) - re-rolling..."
                )
                continue

            if not isinstance(parsed, dict):
                parsed = {}

            query = str(parsed.get("search_query") or "").strip()
            if not query:
                reason = (
                    "The visual prompt was empty. Return a complete, specific "
                    "scene for the assigned narration segment."
                )
                rejected_attempts.append({
                    "visual": "(empty visual prompt)",
                    "reason": reason,
                })
                self.log(
                    f"Segment {index + 1}: rejected ({reason}) - re-rolling..."
                )
                continue

            repetition_issue = visual_repetition_problem(
                query,
                prior_visuals or [],
            )
            if repetition_issue:
                rejected_attempts.append({
                    "visual": query,
                    "reason": repetition_issue,
                })
                self.log(
                    f"Segment {index + 1}: rejected ({repetition_issue}) - re-rolling..."
                )
                continue

            if verify:
                aligned, reason = self.verify_visual(
                    scripture,
                    segments,
                    index,
                    query,
                    prior_visuals=prior_visuals or [],
                )
                if not aligned:
                    rejected_attempts.append({
                        "visual": query,
                        "reason": reason,
                    })
                    self.log(
                        f"Segment {index + 1}: rejected ({reason}) - re-rolling..."
                    )
                    continue

            self.log(
                f"Segment {index + 1}: visual approved."
                if verify else
                f"Segment {index + 1}: visual generated (verification disabled)."
            )
            return parsed


    def _build_visual_sequence_revision_feedback(
        self,
        target_index,
        finding,
        visuals,
    ):
        """Give a targeted regeneration the entire current sequence as context."""
        sequence = []
        for index, visual in enumerate(visuals):
            query = (
                str(visual.get("search_query") or "").strip()
                if isinstance(visual, dict)
                else str(visual or "").strip()
            )
            label = "REPLACE THIS SEGMENT" if index == target_index else "KEEP THIS SEGMENT"
            sequence.append(
                f"Segment {index + 1} [{label}]: {query}"
            )

        suggested_focus = str(finding.get("suggested_focus") or "").strip()
        prompt = (
            "EPISODE-WIDE VISUAL STORYTELLING REVIEW - TARGETED REVISION\n"
            f"The episode-level reviewer identified a sequence problem in segment {target_index + 1}:\n"
            f"{finding['issue']}\n"
        )
        if suggested_focus:
            prompt += f"Suggested scripture-consistent alternative focus: {suggested_focus}\n"

        prompt += (
            "\nCURRENT EPISODE VISUAL SEQUENCE\n"
            + "\n".join(sequence)
            + "\n\nCreate a replacement only for the assigned narration segment. "
            "Make it meaningfully different from the other clips while remaining "
            "faithful to this segment and the source passage. Do not change the "
            "other segments. If the narration describes the same continuous "
            "event as a neighboring clip, use a genuinely different useful "
            "framing, perspective, focal detail, or stage of that same supported "
            "event rather than inventing a new action. If no distinct alternative "
            "is possible, prioritize a faithful, coherent visual over forced variety."
        )
        return prompt


    def generate(
        self,
        scripture,
        segments,
        narration=None
    ):
        """Generate, individually verify, and review the complete visual sequence."""
        self.log(
            f"Generating {len(segments)} visuals, one per narration segment..."
        )

        verify = bool(
            self.generation_config.get("verify_visuals", True)
        )

        visuals = []
        for index, segment in enumerate(segments):
            parsed = self._generate_visual_for_segment(
                scripture,
                segments,
                index,
                prior_visuals=visuals,
                verify=verify,
            )
            visuals.append(parsed)

        if verify and visuals:
            review_round = 0
            while True:
                review_round += 1
                self.log(
                    f"Reviewing the complete visual sequence (round {review_round})..."
                )
                sequence_passes, findings = self.verify_visual_sequence(
                    scripture,
                    segments,
                    visuals,
                )

                if sequence_passes:
                    self.log(
                        f"Episode-wide visual storytelling review passed on round {review_round}."
                    )
                    break

                finding_details = "; ".join(
                    f"segment {item['segment_index']}: {item['issue']}"
                    for item in findings
                )
                self.log(
                    "Episode-wide visual storytelling review flagged "
                    f"{len(findings)} segment(s). "
                    + finding_details
                    + ". Regenerating only those segments..."
                )

                for finding in findings:
                    index = finding["segment_index"] - 1
                    sequence_feedback = self._build_visual_sequence_revision_feedback(
                        index,
                        finding,
                        visuals,
                    )
                    visuals[index] = self._generate_visual_for_segment(
                        scripture,
                        segments,
                        index,
                        prior_visuals=visuals[:index],
                        verify=verify,
                        sequence_feedback=sequence_feedback,
                    )

        metadata = self.generate_metadata(
            scripture,
            segments,
        )

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


def _summary_completeness_issue(summary):
    """Return a repair reason when a summary is visibly unfinished.

    This is intentionally a conservative structural guard, not a grammar
    checker. It catches empty/truncated output, missing final sentence-ending
    punctuation, unbalanced quotation marks, and obvious dangling final words.
    It does not limit summary length or sentence count. The metadata LLM
    reviewer remains responsible for meaning and attribution.
    """
    value = str(summary or "").strip()
    if not value:
        return "The summary is empty. Provide a complete, useful factual summary."

    _spans, quote_problems = _scan_quoted_spans(value)
    if quote_problems:
        return (
            "The summary contains an unfinished or unbalanced quotation. "
            "Close it properly or remove the incomplete quotation."
        )

    without_closers = value.rstrip("'\"”’)]}")
    if not without_closers or without_closers[-1] not in ".!?":
        return (
            "The summary appears truncated or unfinished because it does not "
            "end with sentence punctuation. Finish the summary with a complete "
            "final sentence or thought."
        )

    final_word = re.search(
        r"([A-Za-z]+(?:['’][A-Za-z]+)?)\s*[.!?…]*[\"”’')\]]*$",
        value,
    )
    dangling_words = {
        "a", "an", "the", "this", "these", "those", "his", "its", "their",
        "our", "your", "my", "and", "or", "but", "nor", "yet", "so",
        "while", "although", "because", "if", "when", "unless", "whereas",
        "to", "of", "for", "with", "from", "by", "into", "onto", "at",
        "in", "on", "under", "over", "through",
        "without", "within", "between", "among", "who", "whom", "whose",
        "which", "that",
    }
    if final_word and final_word.group(1).casefold() in dangling_words:
        return (
            "The summary ends with the dangling word '"
            + final_word.group(1)
            + "'. Complete the unfinished phrase and finish the summary naturally."
        )

    return ""


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