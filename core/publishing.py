"""
Shared publishing metadata for Hear His Voice.

The published narration is original wording written from the Gospel
passage, not the WEBC text, so no translation attribution is published.
WEBC stays internal: it is the authoritative source the narration is
written from, kept beside the episode as ``source_text``.

Everywhere the episode is published, the description and the caption
carry the same blocks in the same order:

    1. the episode title (which carries the Scripture reference)
    2. a short, factual summary of the passage
    3. the hashtags

The title leads, matching how the other channels publish. It is the
only place the passage reference appears - the reference is never
restated as a separate line of its own.
"""


def build_caption(
    title,
    summary="",
    hashtags=(),
):
    """
    Builds the Instagram caption: title, summary, hashtags.

    Same blocks, same order, as the YouTube description and the same
    order the other channels use: the title leads, the summary follows.

    The caption is assembled here and nowhere else. It used to be
    possible to store a ready-made caption beside the episode and post
    that instead, but a second place to edit is a second place to
    drift, and that path also dropped the hashtags, which every post
    needs.
    """

    blocks = []

    for value in (title, summary):

        text = str(
            value or ""
        ).strip()

        if text:

            blocks.append(text)

    tags = " ".join(
        str(tag).strip()
        for tag in (hashtags or ())
        if str(tag).strip()
    )

    if tags:

        blocks.append(tags)

    return "\n\n".join(blocks)