"""Speech detection, in two interchangeable flavours.

Both backends answer the same question — which stretches of a single track
contain speech — and emit the same structure, so the rest of the pipeline never
learns which one ran.

``ffmpeg``  parses silencedetect output. Level based, no extra dependencies,
            and it will happily call a breath or room tone "speech".
``silero``  runs Silero VAD over the prepared 16 kHz mono track. Needs torch,
            markedly better at telling speech from noise.
"""

from __future__ import annotations

import re
import wave

from . import intervals

_SILENCE_START = re.compile(r"silence_start:\s*(-?[0-9.]+)")
_SILENCE_END = re.compile(r"silence_end:\s*(-?[0-9.]+)")

SILERO_WINDOW_SECONDS = 600


def parse_silencedetect(log_text: str, duration: float) -> list[tuple[float, float]]:
    """Turn a silencedetect log into speech intervals.

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


def silero_speech(
    path: str,
    threshold: float = 0.5,
    min_silence_ms: int = 200,
    min_speech_ms: int = 120,
) -> list[tuple[float, float]]:
    """Run Silero VAD over a 16 kHz mono WAV in windows.

    A two-hour track is processed in chunks so peak memory stays bounded;
    speech crossing a window boundary is stitched back together by the final
    normalise, since the two halves come out adjacent.
    """
    import numpy as np
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    model = load_silero_vad()

    with wave.open(path, "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise SystemExit(
                f"{path}: silero backend expects 16-bit mono, got "
                f"{handle.getnchannels()}ch/{handle.getsampwidth() * 8}-bit"
            )
        rate = handle.getframerate()
        if rate != 16000:
            raise SystemExit(f"{path}: silero backend expects 16 kHz, got {rate}")

        window_frames = SILERO_WINDOW_SECONDS * rate
        speech: list[tuple[float, float]] = []
        offset_frames = 0

        while True:
            raw = handle.readframes(window_frames)
            if not raw:
                break
            samples = np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0
            stamps = get_speech_timestamps(
                torch.from_numpy(samples),
                model,
                sampling_rate=rate,
                threshold=threshold,
                min_silence_duration_ms=min_silence_ms,
                min_speech_duration_ms=min_speech_ms,
                return_seconds=True,
            )
            base = offset_frames / float(rate)
            speech.extend((base + s["start"], base + s["end"]) for s in stamps)
            offset_frames += len(samples)

    # Fuse intervals split only by a window boundary.
    return intervals.normalize(speech, gap=0.02)
