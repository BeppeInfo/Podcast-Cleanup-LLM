"""Transcription against a remote whisper.cpp server.

The local path runs `whisper-cli` per track and reads the JSON it writes. This
is the alternative: POST the prepared audio to a `whisper-server` `/inference`
endpoint and convert whatever comes back into the same segment shape, so
`transcript.build_from_segments` reassembles words identically either way.

Two things make this more than a plain upload.

**Chunking.** A two-hour track is ~230 MB of 16 kHz mono PCM, and sending it as
one request means no progress for however long the server takes. It is split
instead — and the split points are chosen inside silence the VAD stage already
found, so a chunk boundary does not land in the middle of a word.

**Tolerant parsing.** whisper-server's response shape has varied between
versions and `response_format` values. Rather than pin one, several shapes are
accepted, and the one thing that is checked properly is whether usable timings
came back at all.
"""

from __future__ import annotations

import json
import os
import secrets
import tempfile
import urllib.error
import urllib.request
import wave

from . import intervals as iv
from .llm import AuthRejected

# Below this, a trailing chunk is merged into its predecessor rather than sent
# on its own — whisper's accuracy suffers on very short fragments.
MIN_CHUNK_SECONDS = 20.0


# --- chunk planning -----------------------------------------------------------


def plan_audio_chunks(duration: float, target: float, speech=None):
    """Split [0, duration) into chunks of roughly `target` seconds.

    Where the VAD's speech map is available, each boundary is nudged onto the
    middle of a nearby silence, so no word is cut in half. Without it, or when
    no silence is close enough, the boundary falls where it falls.
    """
    if target <= 0 or duration <= target:
        return [(0.0, duration)]

    silence = iv.complement(iv.normalize(speech or []), 0.0, duration)
    chunks: list[tuple[float, float]] = []
    cursor = 0.0

    while duration - cursor > target + MIN_CHUNK_SECONDS:
        ideal = cursor + target
        # Only silences that leave a workable chunk behind them are candidates.
        best = None
        for gap_start, gap_end in silence:
            middle = (gap_start + gap_end) / 2.0
            if middle <= cursor + MIN_CHUNK_SECONDS:
                continue
            if middle >= duration - MIN_CHUNK_SECONDS:
                break
            distance = abs(middle - ideal)
            if best is None or distance < best[0]:
                best = (distance, middle)

        # A boundary is only worth moving if the silence is genuinely nearby.
        if best is not None and best[0] <= target * 0.25:
            split = best[1]
        else:
            split = ideal
        chunks.append((cursor, split))
        cursor = split

    chunks.append((cursor, duration))
    return chunks


# Pauses shorter than this stay inside a speech region rather than splitting it.
# Cutting on every breath would multiply the requests and strip the surrounding
# context Whisper uses to punctuate, and it buys nothing: a two-second pause is
# not what it hallucinates over.
SPEECH_MERGE_GAP = 2.0

# Margin kept either side of every speech region. The VAD's boundaries are its
# own judgement, and a word's opening consonant is usually quieter than the
# vowel behind it, so the edges are exactly where it errs.
SPEECH_PAD = 0.5


def plan_speech_chunks(duration: float, target: float, speech=None):
    """Chunks covering only what this track's VAD calls speech.

    Whisper invents text when handed silence, and loops a phrase while doing it:
    on the recording this was written for it produced one sentence 198 times
    across 4.6 minutes of a muted mic — 73% of that track — and every stage
    downstream then worked faithfully on fiction. The cheapest defence is to
    stop sending it the silence.

    Regions are padded and fused before use, so the cost of the VAD being
    slightly wrong at a boundary is a little extra silence rather than a
    clipped word.

    Falls back to covering everything when there is no speech map, or when the
    map is empty. An empty map is far more likely to be a misconfigured VAD than
    a genuinely silent track, and transcribing nothing would hide that behind an
    empty transcript that looks perfectly well-formed.
    """
    spans = iv.normalize(speech or [])
    if not spans:
        return plan_audio_chunks(duration, target, speech)

    regions = iv.normalize(
        [(max(0.0, s - SPEECH_PAD), min(duration, e + SPEECH_PAD)) for s, e in spans],
        gap=SPEECH_MERGE_GAP,
    )

    chunks: list[tuple[float, float]] = []
    for start, end in regions:
        span = end - start
        if target <= 0 or span <= target:
            chunks.append((start, end))
            continue
        # Too long for one request: split it with the usual boundary logic,
        # working in region-local time and shifting back onto the timeline.
        local = [
            (max(0.0, s - start), min(span, e - start))
            for s, e in spans if e > start and s < end
        ]
        for lo, hi in plan_audio_chunks(span, target, local):
            chunks.append((start + lo, start + hi))
    return chunks


# --- audio slicing ------------------------------------------------------------


def slice_wav(source: str, start: float, end: float, target: str) -> None:
    """Copy [start, end) of a PCM WAV into a new WAV, sample-accurately."""
    with wave.open(source, "rb") as reader:
        rate = reader.getframerate()
        first = max(0, int(round(start * rate)))
        last = min(reader.getnframes(), int(round(end * rate)))
        reader.setpos(first)
        frames = reader.readframes(max(0, last - first))
        with wave.open(target, "wb") as writer:
            writer.setnchannels(reader.getnchannels())
            writer.setsampwidth(reader.getsampwidth())
            writer.setframerate(rate)
            writer.writeframes(frames)


def wav_info(path: str) -> tuple[float, int]:
    with wave.open(path, "rb") as handle:
        return handle.getnframes() / float(handle.getframerate()), handle.getframerate()


# --- response parsing ---------------------------------------------------------


def to_seconds(value) -> float | None:
    """Accept a number, a numeric string, or HH:MM:SS(.|,)mmm."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace(",", ".")
    try:
        return float(text)
    except ValueError:
        pass
    parts = text.split(":")
    if not 2 <= len(parts) <= 3:
        return None
    try:
        numbers = [float(part) for part in parts]
    except ValueError:
        return None
    while len(numbers) < 3:
        numbers.insert(0, 0.0)
    return numbers[0] * 3600 + numbers[1] * 60 + numbers[2]


def _clean_tokens(tokens):
    """Keep tokens only when they carry the timings we would use."""
    if not isinstance(tokens, list):
        return None
    usable = []
    for token in tokens:
        if not isinstance(token, dict):
            return None
        offsets = token.get("offsets")
        if not isinstance(offsets, dict) or "from" not in offsets or "to" not in offsets:
            return None
        usable.append(token)
    return usable or None


def normalize_response(payload, offset: float = 0.0) -> list[dict]:
    """Convert a server response into whisper-cli-shaped segments.

    Returns segments with millisecond `offsets`, shifted by `offset` seconds so
    a chunk's timings sit on the episode's timeline. Raises ValueError when the
    response carries no usable timing at all, since silently inventing one would
    put cuts in the wrong place.
    """
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object, got {type(payload).__name__}")

    raw = payload.get("transcription")
    if not isinstance(raw, list):
        raw = payload.get("segments")
    if not isinstance(raw, list):
        if payload.get("text"):
            raise ValueError(
                "the response has text but no timings — the endpoint needs to be "
                "asked for verbose_json, since word positions cannot be guessed"
            )
        raise ValueError("the response contains neither 'transcription' nor 'segments'")

    shift_ms = int(round(offset * 1000))
    segments: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = (item.get("text") or "").strip()
        if not text:
            continue

        offsets = item.get("offsets")
        if isinstance(offsets, dict) and "from" in offsets and "to" in offsets:
            start_ms = to_seconds(offsets["from"])
            end_ms = to_seconds(offsets["to"])
            if start_ms is None or end_ms is None:
                continue
            start_ms, end_ms = int(start_ms), int(end_ms)
        else:
            start = to_seconds(item.get("start", item.get("from")))
            end = to_seconds(item.get("end", item.get("to")))
            if start is None or end is None:
                continue
            start_ms = int(round(start * 1000))
            end_ms = int(round(end * 1000))

        segment = {
            "text": text,
            "offsets": {
                "from": start_ms + shift_ms,
                "to": max(end_ms, start_ms) + shift_ms,
            },
        }
        tokens = _clean_tokens(item.get("tokens"))
        if tokens:
            segment["tokens"] = [
                {
                    **token,
                    "offsets": {
                        "from": int(to_seconds(token["offsets"]["from"]) or 0) + shift_ms,
                        "to": int(to_seconds(token["offsets"]["to"]) or 0) + shift_ms,
                    },
                }
                for token in tokens
            ]
        segments.append(segment)

    if not segments:
        raise ValueError("no segments with usable timings in the response")
    return segments


# --- client -------------------------------------------------------------------


def _multipart(fields: dict, filename: str, content: bytes) -> tuple[bytes, str]:
    boundary = f"----podcastcleanup{secrets.token_hex(16)}"
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n".encode()
        )
    parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: audio/wav\r\n\r\n".encode()
    )
    parts.append(content)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


class WhisperClient:
    def __init__(
        self,
        endpoint: str,
        timeout: float = 1800.0,
        path: str = "/inference",
        api_key: str | None = None,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.path = path
        self.api_key = api_key or None

    def _headers(self, **extra) -> dict:
        headers = dict(extra)
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _auth_rejected(self, code: int) -> AuthRejected:
        return AuthRejected(
            f"the whisper endpoint rejected our credentials (HTTP {code}). "
            + (
                "Check WHISPER_API_KEY against whatever fronts the server."
                if self.api_key
                else "It requires an API key; set WHISPER_API_KEY or "
                "WHISPER_API_KEY_FILE."
            )
        )

    def wait_until_ready(self, timeout: float, poll: float = 2.0) -> bool:
        """Any HTTP answer means a server is there; only a refused connection
        counts as absent. `/inference` rejects GET, and that rejection is proof
        enough that it exists.

        A 401 is the exception: something is listening, but it will refuse the
        real request too, so it is reported now rather than mid-episode.
        """
        import time

        deadline = time.monotonic() + timeout
        last = "no response"
        while time.monotonic() < deadline:
            for probe in (self.path, "/"):
                request = urllib.request.Request(
                    f"{self.endpoint}{probe}", headers=self._headers()
                )
                try:
                    with urllib.request.urlopen(request, timeout=10):
                        return True
                except urllib.error.HTTPError as exc:
                    if exc.code in (401, 403):
                        print(self._auth_rejected(exc.code))
                        return False
                    return True  # it spoke HTTP, so it is listening
                except Exception as exc:
                    last = f"{type(exc).__name__} on {probe}"
            time.sleep(poll)
        print(f"whisper endpoint not reachable: {last}")
        return False

    def transcribe_file(self, path: str, fields: dict) -> dict:
        with open(path, "rb") as handle:
            content = handle.read()
        body, content_type = _multipart(fields, os.path.basename(path), content)
        request = urllib.request.Request(
            f"{self.endpoint}{self.path}",
            data=body,
            headers=self._headers(
                **{"Content-Type": content_type, "Accept": "application/json"}
            ),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                exc.close()  # or its buffered body leaks a ResourceWarning
                raise self._auth_rejected(exc.code) from exc
            raise


def transcribe(
    client: WhisperClient,
    wav_path: str,
    participant: str,
    *,
    language: str = "auto",
    chunk_seconds: float = 600.0,
    speech=None,
    temperature: float = 0.0,
    retries: int = 1,
    on_progress=None,
    skip_silence: bool = True,
) -> dict:
    """Transcribe one prepared track, returning the parsed words structure.

    `skip_silence` sends only the stretches this track's own VAD calls speech.
    It is on by default because the alternative is inviting hallucination, but
    it does mean speech the VAD missed is never transcribed at all — so how much
    was skipped comes back in the result, and the caller says so out loud.
    """
    from . import transcript as tr

    duration, _ = wav_info(wav_path)
    chunks = (
        plan_speech_chunks(duration, chunk_seconds, speech)
        if skip_silence
        else plan_audio_chunks(duration, chunk_seconds, speech)
    )

    fields = {
        "response_format": "verbose_json",
        "temperature": str(temperature),
        # Best effort: on builds that honour these, every segment comes back as
        # a single word and the timings are word-accurate rather than
        # interpolated. Builds that ignore them are unaffected.
        "max_len": "1",
        "split_on_word": "true",
    }
    if language and language != "auto":
        fields["language"] = language

    segments: list[dict] = []
    workdir = tempfile.mkdtemp(prefix="podcast-asr-")
    try:
        for index, (start, end) in enumerate(chunks):
            if on_progress:
                on_progress(index + 1, len(chunks))
            piece = os.path.join(workdir, f"chunk{index:04d}.wav")
            if len(chunks) == 1:
                piece = wav_path
            else:
                slice_wav(wav_path, start, end, piece)

            last_error = None
            for attempt in range(retries + 1):
                try:
                    payload = client.transcribe_file(piece, fields)
                    segments.extend(normalize_response(payload, offset=start))
                    last_error = None
                    break
                except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as exc:
                    last_error = exc
                    if attempt < retries:
                        import time

                        time.sleep(2.0 * (attempt + 1))
            if last_error is not None:
                raise SystemExit(
                    f"transcription failed for {participant} "
                    f"chunk {index + 1}/{len(chunks)} ({start:.1f}-{end:.1f}s): {last_error}"
                )
            if piece != wav_path:
                os.unlink(piece)
    finally:
        try:
            os.rmdir(workdir)
        except OSError:
            pass

    segments.sort(key=lambda segment: segment["offsets"]["from"])
    parsed = tr.build_from_segments(segments, participant)
    parsed["chunks"] = len(chunks)
    # What was actually put in front of the model. Worth recording because the
    # difference between this and the track length is audio nothing will ever
    # transcribe: if the VAD was wrong, the words are simply missing, and there
    # is nothing in the transcript itself to show for it.
    covered = sum(end - start for start, end in chunks)
    parsed["audio_seconds"] = round(duration, 3)
    parsed["transcribed_seconds"] = round(min(covered, duration), 3)
    parsed["skipped_seconds"] = round(max(0.0, duration - covered), 3)
    return parsed
