"""Whisper output parsing.

whisper-cli's ``--output-json-full`` gives per-segment token lists, each token
carrying millisecond offsets. Words are reassembled from those tokens so that
the LLM stage can address speech by word index rather than by timestamp.

Token timings are not always usable — depending on build and flags the offsets
can arrive all-zero or non-monotonic. When that happens the segment's words are
spread across its span in proportion to their length, which is coarse but never
wrong enough to matter for a disfluency cut.

This module also answers where a track has speech at all (`speech_from_words`).
That used to be a separate stage running its own detector over the audio; it is
here now because Whisper's own VAD pass decides what gets transcribed, so the
words *are* the speech map. An interpolated segment tiles its whole span
continuously, so silence inside it is invisible — which loses a cut rather than
inventing one, and is the direction to err in.
"""

from __future__ import annotations

import json
import re

from . import intervals

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
        # Some servers report tokens as bare ids rather than objects; there is
        # no timing to recover from those.
        if not isinstance(token, dict):
            return None
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
    """Read a whisper-cli JSON file into {participant, language, segments, words}."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)

    return build_from_segments(
        data.get("transcription") or [],
        participant,
        (data.get("result") or {}).get("language", ""),
    )


def build_from_segments(raw_segments, participant: str, language: str = "") -> dict:
    """Assemble words and segments from raw whisper segments.

    Segments carry millisecond `offsets` and optionally a `tokens` list. This is
    the shape whisper-cli writes; the remote client in `asr.py` converts a
    server's response into it, so both paths reassemble words identically.
    """
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


# Every word's span is widened by this much before the union that makes the
# speech map. Two things ride on it.
#
# Whisper's word timings are approximate — a boundary sits within a couple of
# hundred milliseconds of the real one, and further than that where a segment's
# token timings were unusable and `_segment_words_by_proportion` had to spread
# words evenly across it. The padding is how much of that error a cut is not
# allowed to eat.
#
# It also prices every cut. A gap between words has to exceed
# SILENCE_MIN_DURATION plus twice this before it becomes one, so raising it
# makes cuts rarer and more conservative and lowering it makes them tighter and
# more numerous. That is the whole knob: there is no threshold to tune any more.
SPEECH_PAD = 0.25


def speech_from_words(words, duration: float, pad: float = SPEECH_PAD, loud=None):
    """Where this track has speech, according to its own transcript.

    Whisper does its own Silero VAD pass and only transcribes what that pass
    calls speech, so the words coming back already carry that judgement — which
    is why nothing here runs a second detector over the audio.

    The consequence worth stating plainly: this map cannot disagree with the
    transcript, because it is made of it. Audible material Whisper wrote nothing
    for — a laugh, a cough, a mumble, a filler it dropped — reads as silence
    here and is eligible for a silence cut. That is a deliberate trade, and
    `plan.looping_words` is what catches the failure it can no longer see.

    `loud` bounds *when* a word was said, never whether it was. Whisper's word
    timings are stretched across the silence between phrases — a single word
    routinely spans seconds of it, and one arrived spanning nineteen — and a map
    built from those spans reads as wall-to-wall speech, which has two effects
    and both are wrong. No gap is ever long enough to shorten, and one
    participant's stretched word makes every disfluency the other says during it
    look like crosstalk, to be muted rather than cut.

    So a word claims only the parts of its span the level scan measured sound in.
    This trims, never extends: it cannot make speech out of silence, and what
    Whisper never transcribed stays invisible here exactly as before. A word with
    no measured sound anywhere in it claims nothing — there is no audio there to
    protect, and after a filler-biased decode (see WHISPER_PROMPT) that is the
    shape an invented word takes.
    """
    spans: list[tuple[float, float]] = []
    for word in words or []:
        try:
            start = float(word["start"])
            end = float(word["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end < start:
            continue
        # Clipped before padding, so the margin is measured from the audible
        # part rather than from a boundary the level scan just disowned.
        pieces = (
            intervals.intersect([(start, end)], loud)
            if loud is not None
            else [(start, end)]
        )
        for piece_start, piece_end in pieces:
            spans.append(
                (max(0.0, piece_start - pad), min(duration, piece_end + pad))
            )
    return intervals.normalize(spans)


# Kinds whose span is defined against a survivor: something was said more than
# once, and one of them is the copy the sentence keeps. A filler has no
# survivor — both halves of "um, um" go — so the guard below must leave it be.
SURVIVOR_KINDS = ("stutter", "repetition")


def same_word(one: str, other: str) -> bool:
    """Whether two transcript tokens are the same word, punctuation aside."""
    def bare(text):
        return "".join(c for c in str(text).lower() if c.isalnum())
    reduced = bare(one)
    return bool(reduced) and reduced == bare(other)


def spare_the_survivor(words, first: int, last: int) -> int:
    """The last index a repetition span may take, keeping one copy of the word.

    The LLM is told to keep the completed attempt and does not reliably do it:
    asked about "pancakes, pancakes" it returned both indices, and about
    "would, would, would" all three. Rendered, that is a sentence missing the
    word altogether — a worse outcome than not editing, and the reason this is
    enforced rather than requested.

    Only fires where the whole span is one word repeated, so it cannot shorten a
    span that was doing something else — a repeated *phrase* is left alone,
    since which copy survives cannot be read off as safely. If the word after
    the span is that word again, a survivor already stands outside it and the
    span is returned unchanged.

    A single-word span is never trimmed. One word is not a repetition of itself,
    and the model asking for it is asking to delete that word — which is what
    removing a filler *is*, and is legitimate whether or not a copy stands next
    to it. Only a span holding two or more copies can swallow the last one, so
    the returned index is always >= `first`.
    """
    if last <= first:
        return last
    tokens = [words[index]["text"] for index in range(first, last + 1)]
    # same_word is false for a token with no letters in it, so punctuation-only
    # entries fall out here rather than counting as every word at once.
    if not all(same_word(tokens[0], token) for token in tokens):
        return last
    following = words[last + 1]["text"] if last + 1 < len(words) else ""
    if same_word(tokens[0], following):
        return last
    return last - 1


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
