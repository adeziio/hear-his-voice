import re


# A spoken segment ends at sentence punctuation. The wording inside a
# segment is never touched: the segmenter only chooses boundaries, so
# whatever text it is given comes back word for word, ready to speak
# and caption exactly as it was written.
#
# The boundary is the whitespace that closes one sentence and opens the
# next. Splitting there - rather than before the punctuation - keeps the
# terminal mark with the sentence it belongs to.
SENTENCE_SPLIT = re.compile(
    r"(?:(?<=[.!?…])|(?<=[.!?…][\"'”’)\]]))"
    r"\s+"
    r"(?=[\"'“”(\[]*\s*[A-Z0-9])"
)


# A sentence that would overrun one clip is divided at its own
# punctuation - commas, semicolons, colons, and dashes - into the
# shorter natural sentences the task asks for. Splitting after the
# punctuation keeps the clause lossless: rejoining the clauses with a
# single space reproduces the sentence.
CLAUSE_SPLIT = re.compile(
    r"(?<=[,;:])\s+|(?<=\s[-–—])\s+"
)


def _protected_spans(text):
    """
    Character-index (start, end) ranges that fall inside a quotation.

    Straight and curly double quotes and straight single quotes open and
    close a quotation, while an apostrophe inside a word (Don't,
    Jesus's) never does. The segmenter keeps its cuts out of these
    ranges so a line of spoken dialogue is not broken in half between
    two visuals - "Save us, Lord! We are dying!" is one saying, not
    three.
    """
    spans = []
    quote = None
    start = None

    for index, char in enumerate(text):

        if char in ('"', "“", "”"):

            if char == "”" or quote == char:

                if start is not None:

                    spans.append((start, index + 1))

                quote = None
                start = None

            elif quote is None:

                quote = char
                start = index

            continue

        if char == "'":

            previous = text[index - 1] if index else ""
            following = (
                text[index + 1]
                if index + 1 < len(text)
                else ""
            )
            is_apostrophe = previous.isalnum() and following.isalnum()

            if not is_apostrophe:

                if quote == "'":

                    if start is not None:

                        spans.append((start, index + 1))

                    quote = None
                    start = None

                elif quote is None:

                    quote = "'"
                    start = index

    if quote is not None and start is not None:

        spans.append((start, len(text)))

    return spans


def _split_outside_quotes(text, pattern, protected):
    """
    Splits text at every match of pattern whose cut sits outside the
    protected (quoted) ranges, leaving the words inside a quotation
    intact.

    Matches are consumed exactly as re.split would, so the pieces still
    rejoin into the original text.
    """
    pieces = []
    last = 0

    for match in pattern.finditer(text):

        cut = match.start()

        if any(
            begin <= cut < end
            for begin, end in protected
        ):
            continue

        pieces.append(text[last:cut])
        last = match.end()

    pieces.append(text[last:])

    return pieces


def split_sentences(text, protected=None):
    """
    Splits narration text into sentence-sized pieces at existing
    punctuation. No word is added, removed, or reordered - the
    concatenation of the result is the original text.

    When ``protected`` is given - the (start, end) ranges of a
    quotation - no cut is made inside those ranges, so a quoted saying
    stays whole. With no ``protected`` the behaviour is unchanged.
    """
    value = str(text).strip()

    if protected:

        pieces = _split_outside_quotes(
            value,
            SENTENCE_SPLIT,
            protected
        )

    else:

        pieces = SENTENCE_SPLIT.split(value)

    return [
        piece.strip()
        for piece in pieces
        if piece.strip()
    ]


def split_clauses(sentence, max_words, protected=None):
    """
    Breaks one sentence into shorter clauses when it is too long to sit
    on a single clip.

    Only punctuation that is already in the text is used as a break, and
    the break happens after it, so no word or mark is added, removed,
    or moved. When ``protected`` is given, no break is made inside those
    quoted ranges.

    A clause that is still longer than ``max_words`` on its own cannot
    be divided without inventing a boundary the text does not have, so
    it is returned whole rather than rewritten.
    """
    text = str(sentence).strip()

    if (
        max_words is None
        or max_words <= 0
        or len(text.split()) <= max_words
    ):

        return [text]

    if protected:

        clauses = _split_outside_quotes(
            text,
            CLAUSE_SPLIT,
            protected
        )

    else:

        clauses = CLAUSE_SPLIT.split(text)

    clauses = [
        clause.strip()
        for clause in clauses
        if clause.strip()
    ]

    if not clauses:

        return [text]

    # Regroup the raw clauses into chunks of at most max_words, so a
    # long run of short clauses still reads as balanced phrases.
    chunks = []
    current = []

    for clause in clauses:

        candidate = " ".join(
            current + [clause]
        )

        if (
            current
            and len(candidate.split()) > max_words
        ):

            chunks.append(" ".join(current))
            current = [clause]

        else:

            current.append(clause)

    if current:

        chunks.append(" ".join(current))

    return chunks


def build_sentence_segments(text, max_words=None):
    """
    Breaks the narration into the spoken segments that each get their
    own SnapGenAI visual prompt and clip: one segment per completed
    narration sentence, in the original order.

    `max_words` is the most words one sentence may hold, because a
    sentence is spoken over exactly one clip and that clip is only
    `max_words` long. It is a ceiling, never a quota: a sentence shorter
    than the ceiling is kept whole and never padded, and only a sentence
    that would overrun the clip is divided - at its own punctuation, into
    the shorter natural sentences the task asks for, each of which then
    gets its own visual.

    Only punctuation already in the text is used as a boundary, and the
    break happens after it, so joining the result with single spaces
    reproduces the input exactly: the narration and captions still speak
    every word they were given, in order.

    Cuts are kept out of quotations, so a spoken saying - a cry and the
    reply it draws - stays whole inside one segment.
    """
    value = str(text).strip()

    # Quoted ranges are located in the string actually being split: a
    # range found in the whole narration would point at the wrong
    # characters once that narration is cut into sentences.
    protected = _protected_spans(value)

    segments = []

    for sentence in split_sentences(value, protected):

        # Each sentence is its own string, so its quotations are found
        # in it - not in the narration it came from.
        sentence_spans = _protected_spans(sentence)

        segments.extend(
            split_clauses(
                sentence,
                max_words,
                sentence_spans
            )
        )

    return segments


def group_into_segments(
    sentences,
    target_segments
):
    """
    Groups sentences into roughly `target_segments` segments so the
    passage reads at a natural pace for a short-form video.

    Grouping only changes where a visual cut happens; the spoken words
    and their order stay exactly as the input gives them.
    """
    sentences = [
        sentence
        for sentence in sentences
        if sentence
    ]

    if not sentences:

        return []

    target_segments = max(
        1,
        int(target_segments)
    )

    if target_segments >= len(sentences):

        # One segment per sentence keeps every visual distinct.
        return list(sentences)

    # Distribute sentences into target_segments groups of nearly equal
    # size, always keeping at least one sentence per group.
    per_segment = len(sentences) / target_segments

    segments = []
    start = 0

    for index in range(target_segments):

        end = int(
            round(
                (index + 1) * per_segment
            )
        )

        if index == target_segments - 1:

            end = len(sentences)

        end = max(
            end,
            start + 1
        )

        end = min(
            end,
            len(sentences)
        )

        segments.append(
            " ".join(
                sentences[start:end]
            )
        )

        start = end

    return segments