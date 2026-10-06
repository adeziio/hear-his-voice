import re


# A spoken segment ends at sentence punctuation. The wording inside a
# segment is never touched: the segmenter only chooses boundaries, so
# whatever text it is given comes back word for word, ready to speak
# and caption exactly as it was written.
SENTENCE_SPLIT = re.compile(
    r"(?<=[.!?…”’\"])\s+"
)


# A long sentence can still be broken at its own punctuation - commas,
# semicolons, colons, and dashes - so one visual never covers an entire
# verse. Splitting after the punctuation keeps the clause lossless:
# rejoining the clauses with a single space reproduces the sentence.
CLAUSE_SPLIT = re.compile(
    r"(?<=[,;:])\s+|(?<=\s[-–—])\s+"
)


def split_sentences(text):
    """
    Splits narration text into sentence-sized pieces at existing
    punctuation. No word is added, removed, or reordered - the
    concatenation of the result is the original text.
    """
    pieces = SENTENCE_SPLIT.split(
        str(text).strip()
    )

    return [
        piece.strip()
        for piece in pieces
        if piece.strip()
    ]


def split_clauses(sentence, max_words):
    """
    Breaks one sentence into shorter clauses when it is too long to sit
    on a single visual.

    Only punctuation that is already in the text is used as a break, and
    the break happens after it, so no word or mark is added, removed,
    or moved.
    """
    text = str(sentence).strip()

    if max_words <= 0:

        return [text]

    if len(text.split()) <= max_words:

        return [text]

    clauses = [
        clause.strip()
        for clause in CLAUSE_SPLIT.split(text)
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


def build_visual_segments(
    text,
    target_segments,
    clause_words=12,
    max_words_per_segment=26,
):
    """
    Breaks the narration into the spoken segments that each get their
    own Pexels query and visual direction.

    The text is only ever cut at punctuation that is already in it.
    Joining the result with single spaces reproduces the input exactly,
    so the narration and captions still speak what they were given,
    word for word.
    """
    atoms = []

    for sentence in split_sentences(text):

        atoms.extend(
            split_clauses(
                sentence,
                clause_words
            )
        )

    if not atoms:

        return []

    target_segments = max(
        1,
        int(target_segments)
    )

    segments = []

    if len(atoms) <= target_segments:

        # Already one visual per clause: nothing to group.
        segments = list(atoms)

    else:

        # Otherwise pack clauses together, keeping every segment
        # inside the word budget so a single visual never covers a
        # whole verse.
        total_words = sum(
            len(atom.split())
            for atom in atoms
        )

        budget = max(
            1,
            total_words // target_segments
        )

        current = []
        current_words = 0

        for atom in atoms:

            atom_words = len(atom.split())

            candidate_words = (
                current_words + atom_words
            )

            starts_new = bool(
                current
                and (
                    current_words >= budget
                    or candidate_words > max_words_per_segment
                )
            )

            if starts_new:

                segments.append(" ".join(current))
                current = [atom]
                current_words = atom_words

            else:

                current.append(atom)
                current_words = candidate_words

        if current:

            segments.append(" ".join(current))


    # A clause can still be very short - a bare "Amen." closing a
    # prayer, or a one-word reply. Alone that would give the viewer a
    # single-word visual, so any such fragment is folded into the
    # segment beside it. Joining with one space keeps the text itself
    # unchanged - only the boundaries move.
    minimum_words = 3

    while len(segments) > 1:

        shortest = min(
            range(len(segments)),
            key=lambda index: len(
                segments[index].split()
            )
        )

        if (
            len(segments[shortest].split())
            >= minimum_words
        ):

            break

        if shortest > 0:

            segments[shortest - 1] = (
                f"{segments[shortest - 1]} "
                f"{segments[shortest]}"
            )

        else:

            segments[1] = (
                f"{segments[0]} {segments[1]}"
            )

        del segments[shortest]

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