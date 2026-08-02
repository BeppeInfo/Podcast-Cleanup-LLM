"""Level-based silence detection, used only to place chunk boundaries.

This is deliberately *not* the pipeline's speech map. Whisper runs its own
Silero VAD pass before transcribing, so the map every edit decision rests on is
derived from the words that come back — see `transcript.speech_from_words`.

What survives here is the one question a long track raises before it has been
transcribed at all: it is split into several requests, and a split landing
mid-word costs that word. Finding a quiet spot to split on does not need to tell
speech from a breath, so ffmpeg's silencedetect is enough, and ffmpeg is already
required to render anything.
"""

from __future__ import annotations

import re
import wave

from . import intervals

_SILENCE_START = re.compile(r"silence_start:\s*(-?[0-9.]+)")
_SILENCE_END = re.compile(r"silence_end:\s*(-?[0-9.]+)")


def parse_silencedetect(log_text: str, duration: float) -> list[tuple[float, float]]:
    """Turn a silencedetect log into non-silent intervals.

    Events are read in order rather than paired positionally, so a recording
    that ends mid-silence (a silence_start with no matching silence_end) does
    not shift every subsequent interval.
    """
    silence: list[tuple[float, float]] = []
    pending: float | None = None

    for line in log_text.splitlines():
        start_match = _SILENCE_START.search(line)
        if start_match:
            pending = max(0.0, float(start_match.group(1)))
            continue
        end_match = _SILENCE_END.search(line)
        if end_match:
            end = min(float(end_match.group(1)), duration)
            start = pending if pending is not None else 0.0
            if end > start:
                silence.append((start, end))
            pending = None

    if pending is not None and duration - pending > intervals.EPS:
        silence.append((pending, duration))

    return intervals.complement(intervals.normalize(silence), 0.0, duration)


def wav_duration(path: str) -> float:
    with wave.open(path, "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())
