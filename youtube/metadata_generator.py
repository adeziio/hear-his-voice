"""
Builds the prefill metadata for an episode.

The Scripture is public domain, but the World English Bible name is a
trademark of eBible.org and the edition is identified as WEBC, so the
reference and translation are always shown with the text. The passage is
posted verbatim and is never edited.
"""

from youtube.config import (
    get_metadata_defaults
)
from core.publishing import (
    scripture_credit_lines
)


DEFAULT_CATEGORY_ID = "22"  # People & Blogs

DEFAULT_PRIVACY_STATUS = "private"

DEFAULT_TAGS = ["Jesus", "Bible", "Gospel", "Scripture"]


def generate_metadata_from_prompt(
    prompt_item,
    episode_number="",
    config=None
):

    """
    Builds the prefill metadata for an episode from the episode's
    content dict (title / reference / translation / narration /
    summary), the episode number directory, and the config/youtube.json
    defaults.

    The result is only a starting point - every field is editable
    in the upload form before the user submits.
    """

    prompt_item = (
        prompt_item
        if isinstance(prompt_item, dict)
        else {}
    )

    defaults = get_metadata_defaults(config)

    base_title = str(
        prompt_item.get("title") or ""
    ).strip()

    title_suffix = str(
        defaults.get("title_suffix") or ""
    ).strip()

    title = base_title

    if title_suffix and title_suffix not in title:

        title = (
            f"{title} {title_suffix}"
        ).strip()

    # The exact WEBC wording that was spoken, straight from the
    # episode content. It is never re-typed or reworded here.
    scripture_text = str(
        prompt_item.get("narration") or ""
    ).strip()

    summary = str(
        prompt_item.get("summary") or ""
    ).strip()

    tags = list(
        defaults.get("tags") or DEFAULT_TAGS
    )

    # The passage that was read, plus the WEBC credit. Taken from the
    # episode's stored reference rather than re-typed, so it always
    # matches the narration.
    reference = str(
        prompt_item.get("reference") or ""
    ).strip()

    credit = scripture_credit_lines(
        reference
    )

    lines = []

    if base_title:

        lines.append(base_title)

    if summary:

        lines.append(summary)

    if scripture_text:

        lines.append(scripture_text)

    lines.extend(
        list(
            defaults.get("description_extra_lines") or []
        )
    )

    # The Scripture credit sits with the closing lines, after the summary
    # and immediately before the hashtags.
    lines.extend(credit)

    lines.append(" ".join(tags))

    description = "\n\n".join(
        line
        for line in lines
        if line.strip()
    )

    return {
        "title": title,
        "description": description,
        "tags": tags,
        "category_id": str(
            defaults.get(
                "category_id",
                DEFAULT_CATEGORY_ID
            )
        ),
        "privacy_status": str(
            defaults.get(
                "privacy_status",
                DEFAULT_PRIVACY_STATUS
            )
        ),
        "made_for_kids": bool(
            defaults.get(
                "made_for_kids",
                False
            )
        )
    }


def normalize_upload_metadata(
    metadata
):

    """
    Validates and normalizes the submitted form metadata so it can be
    sent straight to the YouTube Data API v3 videos.insert payload.
    """

    metadata = dict(metadata or {})

    title = str(
        metadata.get("title") or ""
    ).strip()

    description = str(
        metadata.get("description") or ""
    ).strip()

    if not title:

        raise ValueError(
            "Title is required."
        )

    if not description:

        raise ValueError(
            "Description is required."
        )

    tags = metadata.get("tags") or []

    if isinstance(tags, str):

        tags = [
            tag.strip()
            for tag in tags.split(",")
            if tag.strip()
        ]

    elif isinstance(tags, (list, tuple)):

        tags = [
            str(tag).strip()
            for tag in tags
            if str(tag).strip()
        ]

    else:

        tags = []

    unique_tags = []

    seen = set()

    for tag in tags:

        if tag in seen:

            continue

        seen.add(tag)

        unique_tags.append(tag)

    privacy_status = str(
        metadata.get("privacy_status") or DEFAULT_PRIVACY_STATUS
    ).strip().lower()

    if privacy_status not in ("private", "unlisted", "public"):

        raise ValueError(
            "Privacy status must be 'private', 'unlisted' or 'public'."
        )

    made_for_kids_value = (
        metadata.get("made_for_kids", False)
    )

    if isinstance(made_for_kids_value, str):

        made_for_kids = (
            made_for_kids_value
            .strip()
            .lower()
            in ("true", "1", "yes", "on")
        )

    else:

        made_for_kids = bool(
            made_for_kids_value
        )

    return {
        "title": title,
        "description": description,
        "tags": unique_tags,
        "category_id": str(
            metadata.get("category_id") or DEFAULT_CATEGORY_ID
        ).strip(),
        "privacy_status": privacy_status,
        "made_for_kids": made_for_kids
    }
