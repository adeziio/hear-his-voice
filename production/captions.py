from pathlib import Path
import re


def build_caption_cues(
    words,
    max_words_per_line=7,
    max_gap=1.0,
    min_cue_duration=0.65,
    narration_text=None,
):
    """Groups timed narration words into complete-sentence SRT cues."""
    words = restore_narration_word_text(words, narration_text)
    sentence_end_indices = (
        set(_sentence_end_indices(narration_text))
        if narration_text
        else None
    )
    cues = []
    current_words = []

    for word in words:
        text = str(word.get("word", "")).strip()
        if not text:
            continue

        current_words.append({
            "word": text,
            "start": float(word["start"]),
            "end": float(word["end"])
        })

        source_index = word.get("_source_index")
        ends_sentence = (
            source_index in sentence_end_indices
            if sentence_end_indices is not None
            else _ends_sentence(text)
        )

        if ends_sentence:
            cues.append(_make_cue(current_words))
            current_words = []

    if current_words:
        cues.append(_make_cue(current_words))

    return cues


def build_word_cues(words, narration_text=None):
    """Returns word timings for timeline alignment and internal composition."""
    words = restore_narration_word_text(words, narration_text)
    cues = []
    for word in words:
        text = str(word.get("word", "")).strip()
        if not text:
            continue
        cues.append({
            "start": float(word["start"]),
            "end": float(word["end"]),
            "text": text
        })
    return cues


def restore_narration_word_text(words, narration_text=None):
    """Restores punctuation and quotation marks lost by word-boundary events."""
    if not narration_text:
        return list(words)

    source_tokens = str(narration_text).split()
    restored = []
    source_index = 0

    for word in words:
        timed_text = str(word.get("word", "")).strip()
        if not timed_text:
            continue

        timed_key = _word_key(timed_text)
        match_index = None

        for index in range(source_index, len(source_tokens)):
            if _word_key(source_tokens[index]) == timed_key:
                match_index = index
                break

        if match_index is None:
            restored.append(dict(word))
            continue

        restored_word = dict(word)
        restored_word["word"] = source_tokens[match_index]
        restored_word["_source_index"] = match_index
        restored.append(restored_word)
        source_index = match_index + 1

    return restored


def _word_key(value):
    value = str(value).replace("’", "'").replace("‘", "'")
    value = re.sub(r"[^\w']", "", value, flags=re.UNICODE)
    return value.strip("'").casefold()


def _ends_sentence(value):
    value = str(value).rstrip()
    return value.rstrip('"\'”’)]}').endswith((".", "!", "?"))


def _sentence_end_indices(source_text):
    """Returns source-token indexes ending outer sentences."""
    source_tokens = str(source_text).split()
    end_indices = []
    quote = None

    for index, token in enumerate(source_tokens):
        sentence_punctuation = False

        for char_index, char in enumerate(token):
            if char in ('"', "“", "”"):
                if char == "”" or quote == char:
                    quote = None
                elif quote is None:
                    quote = char
                continue

            if char == "'":
                previous = token[char_index - 1] if char_index else ""
                following = token[char_index + 1] if char_index + 1 < len(token) else ""
                is_apostrophe = previous.isalnum() and following.isalnum()
                if not is_apostrophe:
                    quote = None if quote == "'" else ("'" if quote is None else quote)
                continue

            if char in (".", "!", "?"):
                sentence_punctuation = True

        if sentence_punctuation and quote is None:
            next_token = source_tokens[index + 1] if index + 1 < len(source_tokens) else ""
            next_letter = next((char for char in next_token if char.isalpha()), "")
            if not next_letter or next_letter.isupper() or next_token.startswith(("'", '"', "“")):
                end_indices.append(index)

    return end_indices


def _make_cue(words):
    text = " ".join(word["word"] for word in words)
    return {
        "start": round(words[0]["start"], 3),
        "end": round(words[-1]["end"], 3),
        "text": text,
        "words": [{
            "word": word["word"],
            "start": float(word["start"]),
            "end": float(word["end"])
        } for word in words]
    }


def write_srt(cues, output_path):
    """Writes cues as a standard .srt subtitle file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    blocks = []
    for index, cue in enumerate(cues, start=1):
        blocks.append(
            f"{index}\n"
            f"{_srt_time(cue['start'])} --> {_srt_time(cue['end'])}\n"
            f"{cue['text']}\n"
        )

    output_path.write_text("\n".join(blocks), encoding="utf-8")
    return output_path


def _srt_time(seconds):
    milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
