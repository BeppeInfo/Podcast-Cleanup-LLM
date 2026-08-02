"""Transcription against a remote whisper.cpp server.

The local path runs `whisper-cli` per track and reads the JSON it writes. This
is the alternative: POST the prepared audio to a `whisper-server` `/inference`
endpoint and convert whatever comes back into the same segment shape, so
`transcript.build_from_segments` reassembles words identically either way.

Three things make this more than a plain upload.

**Server-side VAD.** whisper.cpp runs Silero over the audio and transcribes only
what it calls speech. Asking for that is the whole reason this pipeline no longer
detects speech itself: handed silence, Whisper invents text and loops a phrase
while doing it. The parameters travel per request, so a shared server needs
nothing at launch but `-vm` pointing at the Silero model — `vad_model` is the one
setting the request cannot carry.

**Chunking.** A two-hour track is ~230 MB of 16 kHz mono PCM, and sending it as
one request means no progress for however long the server takes. It is split
instead, on quiet spots ffmpeg found, so a boundary does not land mid-word. That
hint is only ever used to choose a split point — never as a speech map.

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


def plan_audio_chunks(duration: float, target: float, loud=None):
    """Split [0, duration) into chunks of roughly `target` seconds.

    `loud` is the non-silent stretches ffmpeg reported, and each boundary is
    nudged onto the middle of a nearby quiet spot so no word is cut in half.
    Without it, or when no quiet spot is close enough, the boundary falls where
    it falls: one damaged word every `target` seconds, at a known position.
    """
    if target <= 0 or duration <= target:
        return [(0.0, duration)]

    silence = iv.complement(iv.normalize(loud or []), 0.0, duration)
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
    a chunk's timings sit on the episode's timeline.

    An empty segment list is a *valid* answer and comes back as `[]`. With the
    server running Silero over the audio first, a chunk that holds no speech is
    answered `{"text": "", "segments": []}` with a 200, and that is the truth
    about that chunk. Treating it as a failure cost a whole track on the first
    run where chunking met a genuinely silent stretch — every track here has
    minutes of it, because one participant is silent while the other talks.

    What still raises is a response that had something to say and no way to
    place it: text without timings, or no recognisable structure at all. Guessing
    a position would put cuts in the wrong place.
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
    # Items that said something, whether or not we could place them. Silence
    # comes back either as no items at all or as items with empty text, and both
    # are answers; an item with words and no timing is not.
    with_text = 0
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = (item.get("text") or "").strip()
        if not text:
            continue
        with_text += 1

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

    if not segments and with_text:
        raise ValueError(
            f"{with_text} segment(s) carried text but none a usable timing"
        )
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
    loud=None,
    temperature: float = 0.0,
    retries: int = 1,
    on_progress=None,
    vad: bool = True,
    vad_options: dict | None = None,
) -> dict:
    """Transcribe one prepared track, returning the parsed words structure.

    `vad` asks the server to run Silero first and transcribe only speech. It is
    on by default because the alternative is inviting hallucination — and
    because everything downstream reads the returned words as this track's
    speech map, so silence reaching the model would become speech in the plan.

    `loud` is ffmpeg's non-silent stretches, used only to place chunk
    boundaries.
    """
    from . import transcript as tr

    duration, _ = wav_info(wav_path)
    chunks = plan_audio_chunks(duration, chunk_seconds, loud)

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
    if vad:
        # Older server builds parse none of these and httplib ignores what it
        # does not know, so an unsupported build looks exactly like a working
        # one until the transcript arrives full of invented speech. Nothing here
        # can tell the difference; the readiness check reports what it can.
        fields["vad"] = "true"
        fields.update({key: str(value) for key, value in (vad_options or {}).items()})

    segments: list[dict] = []
    silent_chunks = 0
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
                    found = normalize_response(payload, offset=start)
                    if not found:
                        silent_chunks += 1
                    segments.extend(found)
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
    # The whole track went to the server; what it chose to transcribe is the
    # server's VAD decision, and the plan stage measures it by deriving the
    # speech map from these words.
    parsed["audio_seconds"] = round(duration, 3)
    parsed["server_vad"] = bool(vad)
    # Chunks the server found no speech in at all. Expected on a two-mic
    # recording — one participant is silent for minutes while the other talks —
    # but every chunk coming back empty is not silence, it is a misconfiguration.
    parsed["chunks_without_speech"] = silent_chunks
    return parsed
