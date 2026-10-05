"""
Shared publishing metadata for Hear His Voice.

The WEBC text is public domain, but the "World English Bible" name is a
trademark of eBible.org, so the edition is always identified wherever the
text is published.

Everywhere the text is published, the description and the caption carry the
same blocks in the same order:

    1. the episode title
    2. a short, factual summary of the passage
    3. the attribution line
    4. the hashtags

The title leads, matching how the other channels publish. It is the only
place the passage reference appears - the reference is never restated as a
separate line of its own.
"""


SCRIPTURE_ATTRIBUTION = (
    "Scripture: World English Bible Catholic (WEBC), public domain. "
    "Text provided by eBible.org."
)


def scripture_credit_lines(
    attribution=SCRIPTURE_ATTRIBUTION,
):
    """
    The Scripture credit block for an episode: the attribution line only.

    Returns an empty list when there is no attribution to add, so callers
    can splice the result in unconditionally without checking first.
    """

    text = str(
        attribution or ""
    ).strip()

    if not text:

        return []

    return [text]


def build_caption(
    title,
    summary="",
    hashtags=(),
    attribution=SCRIPTURE_ATTRIBUTION,
):
    """
    Builds the Instagram caption: title, summary, attribution, hashtags.

    Same blocks, same order, as the YouTube description and the same order
    the other channels use: the title leads, the summary follows.

    The attribution takes the place of the "extra lines" the other channels
    read from config, so it sits between the summary and the hashtags on both
    platforms.

    The caption is assembled here and nowhere else. It used to be possible to
    store a ready-made caption beside the episode and post that instead, but a
    second place to edit is a second place to drift, and that path also
    dropped the hashtags, which every post needs.
    """

    blocks = []

    for value in (title, summary):

        text = str(
            value or ""
        ).strip()

        if text:

            blocks.append(text)

    blocks.extend(
        scripture_credit_lines(attribution)
    )

    tags = " ".join(
        str(tag).strip()
        for tag in (hashtags or ())
        if str(tag).strip()
    )

    if tags:

        blocks.append(tags)

    return "\n\n".join(blocks)