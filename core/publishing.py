"""
Shared publishing metadata for Hear His Voice.

Both YouTube and Instagram carry the same credit block: which passage was
read, and where the text came from. The WEBC text is public domain, but
the "World English Bible" name is a trademark of eBible.org, so the
edition is always identified alongside the passage it came from.
"""


SCRIPTURE_ATTRIBUTION = (
    "Scripture: World English Bible Catholic (WEBC), public domain. "
    "Text provided by eBible.org."
)


def scripture_credit_lines(
    reference,
    attribution=SCRIPTURE_ATTRIBUTION,
):
    """
    The credit block for an episode: the passage reference, then the
    attribution line.

    The reference is the passage the pipeline actually read, so it is
    never re-typed by hand. Returns an empty list when there is no
    reference, so callers can splice the result in unconditionally
    without having to check first.
    """

    text = str(
        reference or ""
    ).strip()

    if not text:

        return []

    return [
        f"📖 Scripture: {text}",
        attribution,
    ]