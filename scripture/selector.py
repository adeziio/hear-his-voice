import json
import random
import re

from pathlib import Path

from ai.providers.ollama_provider import OllamaProvider


from scripture.webc import (
    WEBCScripture,
    ScriptureError
)


# Where the reading position is kept. This is deliberately NOT under
# media/, which is git-ignored for generated output: the reading position
# is meaningful, long-lived project state, and needs to be committed so
# progress through the Gospels is visible in the repository and shared
# with anyone who clones it.
STATE_PATH = "state/scripture_rotation.json"


# The channel stays inside the four Gospels. That is the one hard scope
# rule, and it lives in code so no config edit - a typo, a pasted Epistle
# reference, a whole book - can quietly widen it.
GOSPEL_BOOKS = frozenset(
    ("Matthew", "Mark", "Luke", "John")
)


# The channel reads straight through the Gospels, one book at a time,
# rotating between the four. Episodes do not jump around: each one
# continues from the verse after the last verse the previous episode in
# that same Gospel finished, so no verse is ever skipped or spoken twice.
#
# This is the one hard scope rule, and it lives in code so no config edit
# can quietly widen it.
GOSPEL_BOOKS = frozenset(
    ("Matthew", "Mark", "Luke", "John")
)


# The rotation order. Episodes alternate through the Gospels in this
# sequence rather than being drawn at random, so the channel visits every
# book regularly instead of favouring whichever one was picked first.
ROTATION_ORDER = (
    "Matthew",
    "Mark",
    "Luke",
    "John",
)


# The progress file format. Version 1 stored a flat list of used
# reference strings, which cannot express a position inside a book.
STATE_VERSION = 2


# The model only sees a small number of consecutive ending choices around
# the soft duration target. This keeps selection sequential and bounded while
# allowing a story or teaching to finish naturally instead of stopping at an
# arbitrary word count.
NATURAL_ENDING_EXTRA_VERSES = 4
NATURAL_ENDING_MAX_CANDIDATES = 8


NATURAL_ENDING_SCHEMA = {
    "type": "object",
    "properties": {
        "ending_index": {
            "type": "integer"
        }
    },
    "required": [
        "ending_index"
    ]
}


def book_verses(scripture, book):
    """
    Every verse in a Gospel as (chapter, verse, text), in canonical
    reading order.

    This is the whole of the book - the actual content pool, taken from
    the WEBC text rather than from any config list. Chapters are walked in
    order so reading can cross a chapter boundary when an episode needs
    more text than the rest of one chapter holds.
    """

    chapters = (
        scripture.books.get(book) or {}
    )

    verses = []

    for chapter in sorted(chapters, key=lambda value: int(value)):

        entries = chapters[chapter]

        for number in sorted(entries, key=lambda value: int(value)):

            text = str(entries[number]).strip()

            if text:

                # Chapters are stored as integers deliberately. The WEBC
                # keys are strings, and comparing those as text puts
                # chapter "10" before chapter "9", which silently rewinds
                # the reading at every chapter boundary.
                verses.append(
                    (int(chapter), int(number), text)
                )

    return verses


def next_verse_position(scripture, book, chapter, verse):
    """
    The next verse at or after (chapter, verse) in reading order, or None
    when the book has nothing left.

    This is what guarantees no verse is skipped: progress is simply "the
    verse after the last one spoken", and this finds it.
    """

    for entry_chapter, entry_verse, _text in book_verses(scripture, book):

        if (entry_chapter, entry_verse) >= (chapter, verse):

            return entry_chapter, entry_verse

    return None


def format_reference(book, first, last):
    """
    Renders a passage reference, crossing a chapter boundary honestly
    when the episode ran past the end of one.
    """

    first_chapter, first_verse = first
    last_chapter, last_verse = last

    if (first_chapter, first_verse) == (last_chapter, last_verse):

        return f"{book} {first_chapter}:{first_verse}"

    if first_chapter == last_chapter:

        return f"{book} {first_chapter}:{first_verse}-{last_verse}"

    return (
        f"{book} {first_chapter}:{first_verse}"
        f"-{last_chapter}:{last_verse}"
    )


def estimate_seconds(
    word_count,
    words_per_second=2.9
):
    """
    Rough spoken length of a passage, used to decide how many verses to
    include. Real narration runs a little faster than this, so the
    estimate errs on the short side.
    """
    return float(
        word_count
    ) / max(
        float(words_per_second),
        0.1
    )


def fit_passage_to_duration(
    verses,
    target_seconds,
    words_per_second=2.9,
):
    """
    Chooses how much of a passage to speak.

    `verses` is an ordered list of the passage's verses, each as
    {"verse": number, "text": text}. Verses are added one at a time
    until the passage reaches the target length, then whichever
    boundary lands nearer the target is kept - so the result may sit
    either side of it.

    Only whole verses are used. Cutting one in half would break its
    meaning, which matters more than landing exactly on the number, and
    a verse that is already too long on its own is kept whole rather
    than truncated.

    Returns the number of leading verses to use.
    """
    if not verses:

        return 0

    target_words = float(
        target_seconds
    ) * float(words_per_second)

    running = 0

    for index, verse in enumerate(
        verses
    ):

        words = len(
            str(
                verse.get("text", "")
            ).split()
        )

        previous = running

        running += words

        # The target is reached on this verse. Stopping just before
        # it and including it are both legitimate; keep the nearer.
        if running >= target_words:

            if index == 0:

                # The first verse is already the whole answer.
                return 1

            under = abs(
                target_words - previous
            )

            over = abs(
                running - target_words
            )

            # `index` stops just before this verse, `index + 1`
            # includes it. Keep whichever boundary is nearer the
            # target.
            return (
                index + 1
                if over <= under
                else index
            )

    # The passage is shorter than the target. Use all of it rather
    # than reading past what Jesus actually said.
    return len(verses)


class PassageSelector:
    """
    Chooses which WEBC passage the next episode will speak.

    The selector only deals in references and in verse counts read back
    from the WEBC. It never carries or assembles Scripture wording.
    """

    def __init__(
        self,
        config=None,
        cache_directory=None
    ):

        self.config = config or {}

        # Scripture settings live inside content.json, beside the channel
        # voice they belong to, so there is one content config rather than
        # a separate scripture file that could drift from it.
        scripture_config = (
            self.config.get(
                "content",
                {}
            ).get(
                "scripture",
                {}
            )
        )

        self.translation_name = str(
            scripture_config.get(
                "translation",
                "World English Bible Catholic (WEBC)"
            )
        )

        # How long an episode should run. The passage is sized to sit near
        # this by taking more or fewer whole verses, so it is a guide
        # rather than a limit.
        #
        # It is read from app.json -> shorts.target_duration_seconds,
        # the same single place the other project keeps it. This is the
        # only definition of the target, and the Scripture settings in
        # content.json do not repeat it.
        self.target_seconds = float(
            self.config.get(
                "app",
                {}
            ).get(
                "shorts",
                {}
            ).get(
                "target_duration_seconds",
                60
            )
        )

        # Examples only. These show the KIND of passage this channel
        # wants - teachings, parables, miracles, the arc of his life and
        # death and resurrection - for whoever is writing prompts or
        # reviewing tone.
        #
        # They are NOT the content-selection mechanism and are never
        # consulted when choosing what an episode reads. Selection is
        # sequential: the next unread verse of the next Gospel in the
        # rotation. Deleting this list changes nothing about what plays.
        self.guidance_examples = [
            str(item).strip()
            for item in scripture_config.get(
                "guidance_examples",
                []
            )
            if str(item).strip()
        ]

        # Even examples should stay inside the Gospels, so a pasted Psalm
        # is still an error rather than a silent no-op.
        non_gospel = [
            reference
            for reference in self.guidance_examples
            if not self._is_gospel(reference)
        ]

        if non_gospel:

            raise ScriptureError(
                "Only passages from Matthew, Mark, Luke, and John belong "
                "in the guidance examples, but these are not in the "
                f"Gospels: {', '.join(non_gospel)}"
            )

        self.words_per_second = float(
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

        # The natural-ending model is created lazily so selector construction
        # remains usable for deterministic tooling and offline fallbacks.
        self.llm = None

        self.scripture = WEBCScripture(
            cache_directory
        ) if cache_directory else WEBCScripture()

        # Every verse in each Gospel, in reading order. This is the
        # content pool, taken from the WEBC text itself, so no config
        # list can narrow or reorder what gets read.
        self.verses_by_book = {
            book: book_verses(
                self.scripture,
                book,
            )
            for book in ROTATION_ORDER
        }

        empty = [
            book
            for book, verses in self.verses_by_book.items()
            if not verses
        ]

        if empty:

            raise ScriptureError(
                "The WEBC source has no readable text for: "
                f"{', '.join(empty)}"
            )

    def _is_gospel(
        self,
        reference
    ):

        """
        True when the reference is in Matthew, Mark, Luke, or John.
        """

        text = str(reference or "").strip()

        if not text:

            return False

        return text.split()[0].lower() in {
            book.lower()
            for book in GOSPEL_BOOKS
        }

    def _verses_from(self, book, chapter, verse):
        """
        The consecutive verses from (chapter, verse) onward, in reading
        order, running across chapter boundaries when needed.

        Crossing a chapter is what stops a passage that opens near the end
        of a chapter from turning into a stub: the episode simply carries
        on into the next chapter rather than stopping short.
        """

        start = next_verse_position(
            self.scripture,
            book,
            chapter,
            verse,
        )

        if start is None:

            return []

        return [
            {
                "chapter": entry_chapter,
                "verse": entry_verse,
                "text": text,
            }
            for entry_chapter, entry_verse, text
            in self.verses_by_book.get(book) or []
            if (entry_chapter, entry_verse) >= start
        ]

    def _build_episode(self, book, chapter, verse):
        """
        The passage dict for one episode, sized to the target duration
        from whole consecutive verses.
        """

        available = self._verses_from(book, chapter, verse)

        if not available:

            raise ScriptureError(
                f"{book} {chapter}:{verse} is past the end of the book."
            )

        target_count = fit_passage_to_duration(
            available,
            self.target_seconds,
            self.words_per_second,
        )

        count = self._choose_natural_ending(
            available,
            target_count,
        )

        chosen = available[:max(1, int(count))]

        first = (chosen[0]["chapter"], chosen[0]["verse"])
        last = (chosen[-1]["chapter"], chosen[-1]["verse"])

        text = " ".join(
            item["text"]
            for item in chosen
        )

        return {
            "reference": format_reference(book, first, last),
            "book": book,
            "chapter": str(first[0]),
            "translation": self.translation_name,
            "passage": chosen,
            "text": text,
            "estimated_seconds": round(
                estimate_seconds(
                    len(text.split()),
                    self.words_per_second,
                ),
                1,
            ),
        }, last

    def _natural_ending_candidates(self, available, target_count):
        """
        Returns a bounded set of leading consecutive verse choices.

        The first choice is the short-side duration boundary when there is
        one; the remaining choices extend beyond the target so the model can
        finish a complete thought without being offered an unbounded window.
        """
        target_count = max(
            1,
            min(
                int(target_count),
                len(available),
            )
        )

        first_count = max(
            1,
            target_count - 2,
        )

        last_count = min(
            len(available),
            target_count + NATURAL_ENDING_EXTRA_VERSES,
        )

        counts = list(
            range(
                first_count,
                last_count + 1,
            )
        )

        if len(counts) > NATURAL_ENDING_MAX_CANDIDATES:

            counts = counts[:NATURAL_ENDING_MAX_CANDIDATES]

        return [
            {
                "count": count,
                "verses": available[:count],
            }
            for count in counts
        ]

    @staticmethod
    def _candidate_text(candidate):
        return " ".join(
            f"{item['chapter']}:{item['verse']} {item['text']}"
            for item in candidate["verses"]
        )

    def _choose_natural_ending(self, available, target_count):
        """
        Uses simple LLM judgment to choose a natural ending from consecutive
        verse boundaries, with the duration target treated only as a guide.

        Any unavailable or unusable model response falls back to the existing
        whole-verse duration choice. No verse can be skipped because every
        candidate starts at the first available verse and ends at a prefix of
        that same ordered list.
        """
        candidates = self._natural_ending_candidates(
            available,
            target_count,
        )

        if len(candidates) <= 1:

            return candidates[0]["count"] if candidates else 1

        candidate_lines = "\n\n".join(
            f"OPTION {index}: ends after "
            f"{candidate['verses'][-1]['chapter']}:{candidate['verses'][-1]['verse']}\n"
            f"{self._candidate_text(candidate)}"
            for index, candidate in enumerate(
                candidates,
                start=1,
            )
        )

        prompt = (
            "You choose the ending of a sequential Gospel passage for a "
            "short-form episode. The passage MUST begin with the first "
            "unread verse shown and MUST use consecutive verses only.\n\n"
            "The target duration is a soft guide, not a hard limit. Prefer "
            "the option that forms a coherent, intentional interval with a "
            "natural beginning and a finished ending. End after a natural "
            "narrative, teaching, parable, miracle, or complete thought. "
            "Do not stop in the middle of an event, saying, explanation, or "
            "idea merely to match the target. A few extra verses are better "
            "when they complete the passage; a shorter option is better when "
            "it already reaches a satisfying conclusion.\n\n"
            f"Soft target: about {self.target_seconds:g} seconds at "
            f"{self.words_per_second:g} words per second.\n"
            "Choose exactly one option. Return only JSON with the 1-based "
            "option number in `ending_index`.\n\n"
            + candidate_lines
        )

        try:

            if self.llm is None:

                self.llm = OllamaProvider(
                    self.config
                )

            response = self.llm.generate(
                prompt,
                response_format=NATURAL_ENDING_SCHEMA,
            )

            if isinstance(response, dict):

                data = response

            else:

                match = re.search(
                    r"\{.*\}",
                    str(response or ""),
                    re.DOTALL,
                )

                if match is None:

                    return target_count

                data = json.loads(match.group(0))

            option_number = int(
                data.get("ending_index", 0)
            )

            if not 1 <= option_number <= len(candidates):

                return target_count

            return candidates[option_number - 1]["count"]

        except Exception:

            return target_count

    def _size_passage(self, reference):
        """
        Resolves an explicit reference - used by tooling and tests - and
        sizes it to the target from whole consecutive verses.
        """

        text = str(reference or "").strip()

        if not self._is_gospel(text):

            raise ScriptureError(
                f"'{reference}' is not in Matthew, Mark, Luke, or "
                "John. This channel only uses the four Gospels."
            )

        window = (
            self.scripture.get_reference_text(text)
        )

        episode, _last = self._build_episode(
            window["book"],
            int(window["chapter"]),
            int(window["passage"][0]["verse"]),
        )

        return episode

    def get_reference(self, reference):
        """
        Resolves one specific reference, refusing anything outside the
        four Gospels so the channel's scope stays where it belongs.
        """

        return self._size_passage(reference)

    def select(self):
        """
        Returns the WEBC passage dict for the next episode.

        The channel reads sequentially and rotates. It visits Matthew ->
        Mark -> Luke -> John and repeats, and the book it lands on
        continues from the first verse after the previous episode in THAT
        book finished. Progress is therefore kept separately per Gospel, so
        each book advances on its own schedule while the rotation
        interleaves them.

        Verses are never skipped and never spoken twice. Only an explicit
        reset (delete the progress file, or call reset()) starts a book
        over from its beginning.
        """

        state = self._load_state()

        book = ROTATION_ORDER[
            state["rotation_index"] % len(ROTATION_ORDER)
        ]

        progress = state["books"][book]

        start = next_verse_position(
            self.scripture,
            book,
            progress["chapter"],
            progress["verse"],
        )

        if start is None:

            # Read all the way through. Begin a new pass and record it, so
            # the repeat is visible in the progress rather than hidden.
            first = self.verses_by_book[book][0]

            start = (first[0], first[1])

            progress["chapter"] = start[0]
            progress["verse"] = start[1]
            progress["cycles"] = int(
                progress.get("cycles", 0)
            ) + 1

        episode, last = self._build_episode(
            book,
            start[0],
            start[1],
        )

        # Resume from the verse after the one just finished. This is what
        # makes reading continuous: no verse falls between episodes.
        following = next_verse_position(
            self.scripture,
            book,
            last[0],
            last[1] + 1,
        )

        if following is None:

            # The book ended exactly on this episode. Next time this
            # Gospel comes round, start a fresh pass.
            progress["chapter"] = start[0]
            progress["verse"] = start[1]
            progress["cycles"] = int(
                progress.get("cycles", 0)
            ) + 1

        else:

            progress["chapter"] = following[0]
            progress["verse"] = following[1]

        state["rotation_index"] = (
            int(state["rotation_index"]) + 1
        ) % len(ROTATION_ORDER)

        self._save_state(state)

        return episode

    def progress(self):
        """
        Where each Gospel has reached, and which book comes next. Lets the
        UI or a quick check see progress without reading anything.
        """

        state = self._load_state()

        index = int(
            state["rotation_index"]
        ) % len(ROTATION_ORDER)

        return {
            "next_book": ROTATION_ORDER[index],
            "rotation_order": list(ROTATION_ORDER),
            "books": {
                book: {
                    "next_chapter": str(
                        state["books"][book]["chapter"]
                    ),
                    "next_verse": int(
                        state["books"][book]["verse"]
                    ),
                    "cycles": int(
                        state["books"][book].get("cycles", 0)
                    ),
                    "verses": len(
                        self.verses_by_book.get(book) or []
                    ),
                }
                for book in ROTATION_ORDER
            },
        }

    def reset(self):
        """
        Starts every Gospel again from its first verse and returns the
        rotation to Matthew. The only way reading repeats.
        """

        self._save_state(
            self._fresh_state()
        )

    def _fresh_state(self):

        state = {
            "version": STATE_VERSION,
            "translation": self.translation_name,
            "rotation_index": 0,
            "books": {},
        }

        for book in ROTATION_ORDER:

            first = self.verses_by_book[book][0]

            state["books"][book] = {
                "chapter": first[0],
                "verse": first[1],
                "cycles": 0,
            }

        return state

    def _load_state(self):

        path = Path(STATE_PATH)

        if path.is_file():

            try:

                with open(
                    path,
                    "r",
                    encoding="utf-8"
                ) as file:

                    state = json.load(
                        file
                    )

                if (
                    isinstance(state, dict)
                    and int(
                        state.get("version", 0)
                    ) == STATE_VERSION
                    and isinstance(
                        state.get("books"),
                        dict,
                    )
                    and all(
                        book in state["books"]
                        for book in ROTATION_ORDER
                    )
                ):

                    return state

            except Exception:

                pass

        # Nothing usable on disk - including the old flat "used
        # references" format, which cannot express a position in a book.
        # Start from the top of each Gospel.
        return self._fresh_state()

    def _save_state(self, state):

        path = Path(STATE_PATH)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                state,
                file,
                indent=2,
                ensure_ascii=False
            )
