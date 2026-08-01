"""Speech detection, in two interchangeable flavours.

Both backends answer the same question — which stretches of a single track
contain speech — and emit the same structure, so the rest of the pipeline never
learns which one ran.

``ffmpeg``  parses silencedetect output. Level based, no extra dependencies,
            and it will happily call a breath or room tone "speech".
``silero``  runs Silero VAD over the prepared 16 kHz mono track. Markedly
            better at telling speech from noise.

"silero" names a model and an algorithm, not a package. Two implementations of
it exist and either will do:

``silero-vad``    the reference package, and the one preferred when present:
                  asking for Silero usually means this. ~970 MB of torch and
                  torchaudio, and it brings its own ``get_speech_timestamps``.
``pysilero_vad``  a 2.3 MB ggml build needing nothing else, and about twice as
                  fast. Used when the reference is not installed. It exposes
                  only a speech probability per 32 ms chunk, so the segmentation
                  below is ours rather than upstream's.

They agree: on the sample recording both find the same five spans with every
boundary inside one 32 ms chunk of the other. Which one ran is recorded in the
VAD output, since they are not bit-identical and an edit should be reproducible.
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


# Silero ends a speech run at a lower probability than it starts one, so a brief
# dip mid-word does not split it. Same margin the reference implementation uses.
SILERO_HYSTERESIS = 0.15
# Speech is widened by this much at both ends, again matching the reference.
SILERO_PAD_MS = 30


def _check_wav(handle, path: str) -> int:
    if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
        raise SystemExit(
            f"{path}: silero backend expects 16-bit mono, got "
            f"{handle.getnchannels()}ch/{handle.getsampwidth() * 8}-bit"
        )
    rate = handle.getframerate()
    if rate != 16000:
        raise SystemExit(f"{path}: silero backend expects 16 kHz, got {rate}")
    return rate


def speech_from_probabilities(
    probabilities,
    step: float,
    threshold: float,
    min_silence_ms: int,
    min_speech_ms: int,
    duration: float | None = None,
) -> list[tuple[float, float]]:
    """Turn per-chunk speech probabilities into speech intervals.

    This is the part `silero-vad` performs in `get_speech_timestamps` and
    `pysilero_vad` leaves to the caller: start on `threshold`, end only after
    `min_silence_ms` below `threshold - SILERO_HYSTERESIS`, drop anything
    shorter than `min_speech_ms`, then pad both ends.
    """
    spans: list[tuple[float, float]] = []
    release = threshold - SILERO_HYSTERESIS
    start: float | None = None
    silence_run = 0.0
    silence_began = 0.0
    min_silence = min_silence_ms / 1000.0

    for index, probability in enumerate(probabilities):
        moment = index * step
        if start is None:
            if probability >= threshold:
                start, silence_run = moment, 0.0
            continue
        if probability < release:
            if silence_run == 0.0:
                silence_began = moment
            silence_run += step
            if silence_run >= min_silence:
                spans.append((start, silence_began))
                start, silence_run = None, 0.0
        else:
            silence_run = 0.0

    total = len(probabilities) * step if duration is None else duration
    if start is not None:
        spans.append((start, total))

    pad = SILERO_PAD_MS / 1000.0
    kept = [
        (max(0.0, s - pad), min(total, e + pad))
        for s, e in spans
        if e - s >= min_speech_ms / 1000.0
    ]
    return intervals.normalize(kept)


def _pysilero_speech(path, threshold, min_silence_ms, min_speech_ms):
    """Preferred implementation: no dependencies, and about twice as fast."""
    from pysilero_vad import SileroVoiceActivityDetector

    detector = SileroVoiceActivityDetector()
    chunk_bytes = SileroVoiceActivityDetector.chunk_bytes()
    with wave.open(path, "rb") as handle:
        rate = _check_wav(handle, path)
        duration = handle.getnframes() / float(rate)
        step = SileroVoiceActivityDetector.chunk_samples() / float(rate)
        probabilities: list[float] = []
        while True:
            # Read in whole chunks; a trailing partial chunk is not accepted.
            raw = handle.readframes(SILERO_WINDOW_SECONDS * rate)
            if not raw:
                break
            usable = len(raw) - (len(raw) % chunk_bytes)
            if usable:
                probabilities.extend(detector.process_chunks(raw[:usable]))

    return speech_from_probabilities(
        probabilities, step, threshold, min_silence_ms, min_speech_ms, duration
    )


def _torch_silero_speech(path, threshold, min_silence_ms, min_speech_ms):
    """Reference implementation, which brings its own segmentation."""
    import numpy as np
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    model = load_silero_vad()

    with wave.open(path, "rb") as handle:
        rate = _check_wav(handle, path)
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


# Tried in order; the first that imports wins. `silero-vad` leads because it is
# the reference implementation — when someone asks for Silero, that is the thing
# they mean, and its segmentation is upstream's rather than our port of it.
# `pysilero_vad` is the lighter, faster stand-in for when torch is not wanted.
SILERO_IMPLEMENTATIONS = (
    ("silero-vad", _torch_silero_speech),
    ("pysilero_vad", _pysilero_speech),
)


def silero_speech(
    path: str,
    threshold: float = 0.5,
    min_silence_ms: int = 200,
    min_speech_ms: int = 120,
) -> tuple[list[tuple[float, float]], str]:
    """Speech intervals for a 16 kHz mono WAV, and the implementation used.

    A two-hour track is read in windows so peak memory stays bounded, whichever
    implementation runs.
    """
    tried = []
    for name, run in SILERO_IMPLEMENTATIONS:
        try:
            return run(path, threshold, min_silence_ms, min_speech_ms), name
        except ImportError:
            tried.append(name)
    raise SystemExit(
        "VAD_BACKEND=silero needs one of: pysilero_vad (2 MB, no dependencies) "
        "or silero-vad (needs torch). Neither imported — tried "
        f"{', '.join(tried)}. Install one, or use VAD_BACKEND=ffmpeg."
    )
