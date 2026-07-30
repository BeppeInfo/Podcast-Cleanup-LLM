"""Whisper output parsing.

whisper-cli's ``--output-json-full`` gives per-segment token lists, each token
carrying millisecond offsets. Words are reassembled from those tokens so that
the LLM stage can address speech by word index rather than by timestamp.

Token timings are not always usable — depending on build and flags the offsets
can arrive all-zero or non-monotonic. When that happens the segment's words are
spread across its span in proportion to their length, which is coarse but never
wrong enough to matter for a disfluency cut.
"""

from __future__ import annotations

import json
import re

# whisper's non-textual tokens: [_BEG_], [_TT_120], [_SOT_], and friends.
_SPECIAL = re.compile(r"^\[_.*_?\]$|^<\|.*\|>$")


def _is_special(text: str) -> bool:
    return bool(_SPECIAL.match(text.strip()))


def _segment_words_from_tokens(segment) -> list[dict] | None:
    """Reassemble words from token offsets, or None if the timings are unusable."""
    tokens = segment.get("tokens") or []
    seg_start = segment["offsets"]["from"] / 1000.0
    seg_end = segment["offsets"]["to"] / 1000.0

    usable = []
    for token in tokens:
        text = token.get("text", "")
        if _is_special(text) or not text.strip():
            continue
        offsets = token.get("offsets") or {}
        if "from" not in offsets or "to" not in offsets:
            return None
        usable.append((text, offsets["from"] / 1000.0, offsets["to"] / 1000.0))

    if not usable:
        return None
    # All-zero or collapsed offsets mean the build did not emit token timings.
    if all(end <= start for _, start, end in usable):
        return None

    words: list[dict] = []
    for text, start, end in usable:
        starts_word = text.startswith(" ") or not words
        piece = text.strip()
        if not piece:
            continue
        if starts_word:
            words.append({"text": piece, "start": start, "end": end})
        else:
            words[-1]["text"] += piece
            words[-1]["end"] = max(words[-1]["end"], end)

    # Enforce monotonicity and keep everything inside the segment's own span.
    cursor = seg_start
    for word in words:
        word["start"] = min(max(word["start"], cursor), seg_end)
        word["end"] = min(max(word["end"], word["start"]), seg_end)
        cursor = word["start"]
    if any(w["end"] <= w["start"] for w in words):
        return None
    return words


def _segment_words_by_proportion(segment) -> list[dict]:
    """Fallback: distribute the segment's words across its span by length."""
    seg_start = segment["offsets"]["from"] / 1000.0
    seg_end = segment["offsets"]["to"] / 1000.0
    pieces = segment.get("text", "").split()
    if not pieces:
        return []
    span = max(seg_end - seg_start, 1e-3)
    weights = [len(p) + 1 for p in pieces]
    scale = span / sum(weights)
    words = []
    cursor = seg_start
    for piece, weight in zip(pieces, weights):
        end = min(cursor + weight * scale, seg_end)
        words.append({"text": piece, "start": cursor, "end": end})
        cursor = end
    return words


def parse_whisper_json(path: str, participant: str) -> dict:
    """Read a whisper JSON file into {participant, language, segments, words}."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)

    raw_segments = data.get("transcription") or []
    language = (data.get("result") or {}).get("language", "")

    segments: list[dict] = []
    words: list[dict] = []
    approximated = 0

    for index, segment in enumerate(raw_segments):
        text = (segment.get("text") or "").strip()
        offsets = segment.get("offsets") or {}
        if "from" not in offsets or "to" not in offsets:
            continue
        start = offsets["from"] / 1000.0
        end = max(offsets["to"] / 1000.0, start)
        if not text:
            continue

        segment_words = _segment_words_from_tokens(segment)
        if segment_words is None:
            segment_words = _segment_words_by_proportion(segment)
            approximated += 1

        first_index = len(words)
        for word in segment_words:
            words.append(
                {
                    "i": len(words),
                    "text": word["text"],
                    "start": round(word["start"], 3),
                    "end": round(word["end"], 3),
                    "segment": index,
                }
            )

        segments.append(
            {
                "i": index,
                "start": round(start, 3),
                "end": round(end, 3),
                "text": text,
                "first_word": first_index,
                "last_word": len(words) - 1,
            }
        )

    return {
        "participant": participant,
        "language": language,
        "segments": segments,
        "words": words,
        "approximated_segments": approximated,
    }


def word_span(words, first: int, last: int) -> tuple[float, float]:
    return words[first]["start"], words[last]["end"]


def word_text(words, first: int, last: int) -> str:
    return " ".join(w["text"] for w in words[first : last + 1])


def neighbour_gaps(words, first: int, last: int) -> tuple[float, float]:
    """Silent room before the first word and after the last, in seconds.

    Used to decide how much padding a cut may claim without eating into the
    surrounding words.
    """
    before = words[first]["start"] - words[first - 1]["end"] if first > 0 else 0.0
    after = (
        words[last + 1]["start"] - words[last]["end"]
        if last + 1 < len(words)
        else 0.0
    )
    return max(before, 0.0), max(after, 0.0)
