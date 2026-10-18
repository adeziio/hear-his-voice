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


def _protected_spans(text):
    """
    Character-index (start, end) ranges that fall inside a quotation.

    Straight and curly double quotes and straight single quotes open and
    close a quotation, while an apostrophe inside a word (Don't,
    Jesus's) never does. The segmenter keeps its cuts out of these
    ranges so a line of spoken dialogue is not broken in half between
    two visuals - "Save us, Lord! We are dying!" is one saying, not
    three.

    A straight quote is classified by its edges: one followed by a
    letter is a NESTED OPENING, one followed by a space, punctuation,
    or the end of the text is a close. An opener found inside an open
    quotation never closes it - narration that quotes two levels with
    the same mark ('Behold ... 'Make ready ... Lord! Make his paths
    straight!') keeps ONE span covering the whole saying - and a closer
    with nothing open (the possessive in "the disciples' boat") opens
    nothing, so a stray mark cannot swallow every sentence after it.
    Both failures previously collapsed a dozen complete sentences into
    one giant "sentence" that no ceiling could group.

    An open quotation that never closes still runs to the end of the
    text: a cut inside a real saying is the one error this scanner may
    never make. The ranges here guide sentence-aware readers
    (split_sentences); the spoken segmentation itself cuts by word
    count (build_word_segments) and needs no quote handling.
    """
    spans = []
    quote = None
    start = None

    for index, char in enumerate(text):

        following = (
            text[index + 1]
            if index + 1 < len(text)
            else ""
        )

        if char in ('"', "“", "”"):

            # A curly close is unambiguous: it closes whatever
            # quotation is open, as it always has.
            if char == "”":

                if start is not None:

                    spans.append((start, index + 1))

                quote = None
                start = None

                continue

            if char == "“":

                if quote is None:

                    quote = char
                    start = index

                # Already inside a quotation: the mark is content,
                # not a boundary - never a close, never a re-open.
                continue

            # char == '"': classified by its edge, like straight '.
            if quote == '"':

                if following.isalnum():

                    # Nested opening inside a straight-quoted span
                    # ("I saw "them" leave."): keep the outer span.
                    continue

                if start is not None:

                    spans.append((start, index + 1))

                quote = None
                start = None

                continue

            if quote is None:

                if following.isalnum():

                    quote = '"'
                    start = index

                # A stray close-looking quote opens nothing.

            # Some other quotation is open: this mark is content.
            continue

        if char == "'":

            previous = text[index - 1] if index else ""
            is_apostrophe = previous.isalnum() and following.isalnum()

            if is_apostrophe:

                continue

            if quote == "'":

                if following.isalnum():

                    # Nested opening inside a single-quoted span: not
                    # the close of the outer saying.
                    continue

                if start is not None:

                    spans.append((start, index + 1))

                quote = None
                start = None

                continue

            if quote is None:

                if following.isalnum():

                    quote = "'"
                    start = index

                # A stray close-looking quote - the possessive in
                # "the disciples' boat" - opens no span, so the
                # sentences after it are still split normally.

            # Some other quotation is open: this mark is content.

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


def build_word_segments(text, max_words):
    """
    Splits the narration into consecutive word chunks of at most
    `max_words` words each - the spoken segments that each get their
    own SnapGenAI visual prompt and clip.

    The ceiling comes from duration_target x words_per_second and is
    enforced by counting words alone: every chunk except the last
    holds exactly max_words words, the final chunk holds the remainder
    (a shorter final chunk is fine), and no chunk is padded to fill a
    clip.

    A cut may fall mid-sentence and mid-quotation - this is
    intentional, so any telling can be grouped strictly by spoken
    length rather than by punctuation. The cut always falls BETWEEN
    words: runs of whitespace collapse to single spaces, and every
    word and mark of the narration appears exactly once, in the
    original order, with nothing omitted, duplicated, or rewritten.
    """
    words = str(text or "").split()

    if not words:

        return []

    max_words = max(
        1,
        int(max_words)
    )

    return [
        " ".join(words[start:start + max_words])
        for start in range(0, len(words), max_words)
    ]


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