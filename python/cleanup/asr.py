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


# --- recovering a skipped decode window ---------------------------------------
#
# Whisper works in 30-second windows and throws away a window whose decode ends
# on a lone timestamp token ("single timestamp ending - skip entire chunk").
# Nothing in the response says so. Which window a passage lands in depends on how
# much speech precedes it, so no request length prevents this — but *re-asking*
# about the missing span on its own does, because the alignment is then different
# and the span starts the request rather than sitting mid-stream. On the episode
# this was written for, the 33 seconds lost from a 600s request came back intact
# when the same audio was sent as a short request of its own.
#
# The level scan says where to look: loud audio the first pass returned no words
# for. That is the same signal `plan.untranscribed_audio` refuses over, and this
# runs first so the refusal is about what could not be recovered.

# Worth re-asking about. Kept equal to plan.UNTRANSCRIBED_MIN_SPAN: chasing
# anything the plan stage would not have mentioned only spends requests.
RECOVERY_MIN_SPAN = 3.0

# Context added either side, so the retry does not begin mid-word and Whisper has
# something to condition on. Trimmed back out afterwards.
RECOVERY_PAD = 1.0

# One retry is sent per span, each no longer than this. Short is the whole point:
# a long retry would land in the same trap it is escaping.
RECOVERY_CHUNK_SECONDS = 60.0

# A retry is only worth accepting where the first pass had nothing. Recovered
# words overlapping what is already known by more than this are the padding
# transcribing a neighbour again, and are dropped rather than duplicated.
RECOVERY_OVERLAP_FRACTION = 0.5

# Bound on how much chasing one track can provoke. A track needing more than this
# is not suffering the occasional skipped window; something else is wrong, and
# spending a hundred requests to confirm it helps nobody.
RECOVERY_MAX_SPANS = 20

# Missing stretches this close together are asked about as one. A brief dip below
# the level threshold splits what is really a single skipped window into
# neighbours, and asking separately then loses the words on the seam between them:
# each retry re-transcribes the other's edge, and the overlap filter drops it as
# already known. Observed doing exactly that — "This time, as" fell between spans
# ending and starting at 47s.
RECOVERY_MERGE_GAP = 3.0


# --- re-asking where one word swallowed the audio -----------------------------
#
# The recovery above answers "loud audio, no words at all". This answers the
# graded version of the same question: loud audio with implausibly *few* words on
# it.
#
# Whisper returns fluent prose rather than what was said. Fillers, stutters and
# false starts are quietly dropped, and the time they occupied is absorbed into a
# neighbouring word, so one word ends up spanning seconds of continuous speech.
# Nothing in the response marks it, and both stages downstream are then blind in
# the same place: the LLM stage cannot cut a disfluency that is not in the
# transcript, and the plan stage sees one long word where there was a pause, so
# the silence never becomes a gap either.
#
# Asked again in a short window the same audio comes back verbatim, because the
# model has no fluent context to smooth into. Measured on the sample fixture:
# "Yeah, I wonder what Fairpunk would talk about this" from the whole-file pass
# became "Yeah, I wonder what Fairpunk, uh, would, would, would talk about this"
# from a five-second window. Sending the same span as a nineteen-second request
# was not enough — the length of the window is the mechanism.

# A word carrying more loud audio than this had something else inside it. No
# ordinary word does: even a long one, said slowly, is well under a second of
# speech, and the level scan does not count the pause around it.
COLLAPSED_WORD_LOUD = 1.2

# Short is the point, not an optimisation. A long window brings back the fluent
# reading that hid the disfluency in the first place.
COLLAPSED_WINDOW = 5.0

# Context either side so a window does not begin mid-word. Never accepted back:
# the padding re-transcribes a neighbour that is already known.
COLLAPSED_PAD = 1.0

# Replaced only where the second pass found at least this many more words than
# the first. Equal counts are the same reading spelled differently — proper nouns
# come back unstable from a short window ("Shwereponk", "SharePunk", "Fairpunk")
# — and swapping one spelling for another is churn, not recovery.
COLLAPSED_MIN_GAIN = 1

# Same reasoning as RECOVERY_MAX_SPANS: a track needing more than this is not
# suffering the occasional collapsed word, and spending the requests to prove it
# helps nobody.
COLLAPSED_MAX_SPANS = 30


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


def missing_spans(segments, loud, duration: float, pad: float) -> list[tuple]:
    """Loud stretches no segment accounts for, longest first.

    `pad` is the same margin the plan stage will widen words by, so this asks the
    question the plan stage would ask rather than a stricter one — a span that
    would not have been reported is not worth a request.
    """
    covered = iv.normalize(
        [
            (
                max(0.0, segment["offsets"]["from"] / 1000.0 - pad),
                min(duration, segment["offsets"]["to"] / 1000.0 + pad),
            )
            for segment in segments
        ]
    )
    # The fusing happens on the *loud* spans, before anything is subtracted. A
    # brief dip below the level threshold splits one skipped window into
    # neighbours, and asking about those separately loses the words where they
    # meet. Fusing the gaps afterwards instead would be wrong in a way that is
    # easy to miss: it would also bridge stretches separated by audio that *was*
    # transcribed, and a track with ordinary inter-word gaps would then look
    # missing from end to end.
    gaps = [
        (start, end)
        for start, end in iv.subtract(
            iv.normalize(loud or [], gap=RECOVERY_MERGE_GAP), covered
        )
        if end - start >= RECOVERY_MIN_SPAN
    ]
    return sorted(gaps, key=lambda span: span[0] - span[1])


def recover_missing(
    client: WhisperClient,
    wav_path: str,
    duration: float,
    fields: dict,
    segments: list,
    loud,
    pad: float,
    workdir: str,
    on_note=None,
) -> tuple[list, dict]:
    """Re-ask about each loud stretch the first pass returned no words for.

    Returns the recovered segments and a summary. Failures here are not fatal:
    the first pass already succeeded, and a span that cannot be recovered is
    exactly what `plan.untranscribed_audio` exists to refuse over.
    """
    spans = missing_spans(segments, loud, duration, pad)
    summary = {
        "spans": len(spans),
        "attempted": 0,
        "recovered_spans": 0,
        "recovered_segments": 0,
        "skipped": max(0, len(spans) - RECOVERY_MAX_SPANS),
    }
    if not spans:
        return [], summary

    known = iv.normalize(
        [
            (segment["offsets"]["from"] / 1000.0, segment["offsets"]["to"] / 1000.0)
            for segment in segments
        ]
    )
    recovered: list[dict] = []

    for index, (start, end) in enumerate(spans[:RECOVERY_MAX_SPANS]):
        # Padded outwards for context, then split if the span is long enough that
        # one request would be back in the situation this is escaping.
        lo = max(0.0, start - RECOVERY_PAD)
        hi = min(duration, end + RECOVERY_PAD)
        pieces = [
            (lo + a, lo + b)
            for a, b in plan_audio_chunks(hi - lo, RECOVERY_CHUNK_SECONDS)
        ]
        summary["attempted"] += 1
        found: list[dict] = []
        for number, (piece_start, piece_end) in enumerate(pieces):
            target = os.path.join(workdir, f"recover{index:03d}_{number:02d}.wav")
            try:
                slice_wav(wav_path, piece_start, piece_end, target)
                payload = client.transcribe_file(target, fields)
                found.extend(normalize_response(payload, offset=piece_start))
            except AuthRejected:
                raise
            except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
                # The first pass stands; this span simply stays missing.
                found = []
                break
            finally:
                if os.path.exists(target):
                    os.unlink(target)

        # Only what the first pass genuinely lacked. The padding re-transcribes
        # whatever borders the span, and that is already known.
        fresh = []
        for segment in found:
            span = (
                segment["offsets"]["from"] / 1000.0,
                segment["offsets"]["to"] / 1000.0,
            )
            length = span[1] - span[0]
            if length <= iv.EPS:
                continue
            if iv.overlap_amount(span, known) >= RECOVERY_OVERLAP_FRACTION * length:
                continue
            fresh.append(segment)

        if fresh:
            summary["recovered_spans"] += 1
            summary["recovered_segments"] += len(fresh)
            recovered.extend(fresh)
            known = iv.normalize(
                known
                + [
                    (s["offsets"]["from"] / 1000.0, s["offsets"]["to"] / 1000.0)
                    for s in fresh
                ]
            )
            if on_note:
                on_note(
                    f"recovered {len(fresh)} word(s) from {start:.0f}-{end:.0f}s, "
                    "which the first pass returned nothing for"
                )
        elif on_note:
            on_note(
                f"nothing came back for {start:.0f}-{end:.0f}s on a second ask "
                "either; if that is speech, the plan stage will refuse to cut it"
            )

    return recovered, summary


def _split_window(start: float, end: float, target: float):
    """Even windows of at most `target`, covering start..end."""
    span = end - start
    count = 1
    while span / count > target:
        count += 1
    step = span / count
    return [(start + i * step, start + (i + 1) * step) for i in range(count)]


def _centre_inside(segment: dict, start: float, end: float) -> bool:
    """Whether a segment belongs to a span, judged by its midpoint.

    Midpoints rather than containment, so the same test decides what leaves and
    what arrives: a word straddling the edge must not be dropped from the first
    pass and then rejected from the second as well.
    """
    lo = segment["offsets"]["from"] / 1000.0
    hi = segment["offsets"]["to"] / 1000.0
    return start <= (lo + hi) / 2.0 < end


def collapsed_spans(segments: list, loud, word_loud: float = COLLAPSED_WORD_LOUD):
    """Words sitting on more loud audio than a word can account for."""
    loud_spans = iv.normalize(loud or [])
    suspicious = []
    for segment in segments:
        start = segment["offsets"]["from"] / 1000.0
        end = segment["offsets"]["to"] / 1000.0
        if end - start <= word_loud:
            continue
        if iv.overlap_amount((start, end), loud_spans) >= word_loud:
            suspicious.append((start, end))
    # Neighbouring collapsed words are one passage read fluently, not two
    # independent findings; asking once keeps the window's context intact.
    return iv.normalize(suspicious, gap=0.5)


def reask_collapsed(
    client: WhisperClient,
    wav_path: str,
    duration: float,
    fields: dict,
    segments: list,
    loud,
    workdir: str,
    word_loud: float = COLLAPSED_WORD_LOUD,
    window: float = COLLAPSED_WINDOW,
    on_note=None,
) -> tuple[list, dict]:
    """Re-ask in short windows wherever one word swallowed seconds of speech.

    Returns the segment list with those passages replaced, and a summary. Like
    the recovery above, failure is not fatal: the first pass stands.
    """
    spans = collapsed_spans(segments, loud, word_loud)
    summary = {
        "spans": len(spans),
        "attempted": 0,
        "replaced_spans": 0,
        "words_before": 0,
        "words_after": 0,
        "skipped": max(0, len(spans) - COLLAPSED_MAX_SPANS),
    }
    if not spans:
        return segments, summary

    result = list(segments)
    for index, (start, end) in enumerate(spans[:COLLAPSED_MAX_SPANS]):
        summary["attempted"] += 1
        found: list[dict] = []
        failed = False
        for number, (lo, hi) in enumerate(_split_window(start, end, window)):
            piece_start = max(0.0, lo - COLLAPSED_PAD)
            piece_end = min(duration, hi + COLLAPSED_PAD)
            target = os.path.join(workdir, f"verbatim{index:03d}_{number:02d}.wav")
            try:
                slice_wav(wav_path, piece_start, piece_end, target)
                payload = client.transcribe_file(target, fields)
                found.extend(normalize_response(payload, offset=piece_start))
            except AuthRejected:
                raise
            except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
                failed = True
                break
            finally:
                if os.path.exists(target):
                    os.unlink(target)
        if failed:
            continue

        fresh = [s for s in found if _centre_inside(s, start, end)]
        stale = [s for s in result if _centre_inside(s, start, end)]
        summary["words_before"] += len(stale)
        # More words for the same audio is the evidence: the first pass was
        # smoothing something over. Fewer or the same is a worse reading of audio
        # already transcribed, and is discarded.
        if len(fresh) - len(stale) < COLLAPSED_MIN_GAIN:
            summary["words_after"] += len(stale)
            continue

        summary["words_after"] += len(fresh)
        summary["replaced_spans"] += 1
        result = [s for s in result if not _centre_inside(s, start, end)]
        result.extend(fresh)
        if on_note:
            said = " ".join(str(s.get("text", "")).strip() for s in fresh)
            on_note(
                f"{start:.1f}-{end:.1f}s came back as {len(fresh)} words rather "
                f"than {len(stale)} when asked in short windows: {said.strip()!r}"
            )

    result.sort(key=lambda segment: segment["offsets"]["from"])
    return result, summary


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
    on_note=None,
    vad: bool = True,
    vad_options: dict | None = None,
    recover: bool = True,
    speech_pad: float = 0.25,
    prompt: str = "",
    reask: bool = True,
    reask_word_seconds: float = COLLAPSED_WORD_LOUD,
    reask_window: float = COLLAPSED_WINDOW,
) -> dict:
    """Transcribe one prepared track, returning the parsed words structure.

    `vad` asks the server to run Silero first and transcribe only speech. It is
    on by default because the alternative is inviting hallucination — and
    because everything downstream reads the returned words as this track's
    speech map, so silence reaching the model would become speech in the plan.

    `loud` is ffmpeg's non-silent stretches. It places chunk boundaries in quiet
    spots, and — when `recover` is on — says where to look for a decode window
    Whisper threw away, which is the one thing that can find those. `speech_pad`
    should match the plan stage's, so this asks the same question of the coverage
    that the plan stage will.

    `prompt` is whisper's initial prompt: conditioning text, not an instruction.
    A filler-laden one biases the decode towards what was actually said, which is
    the only setting that stops fillers being dropped from the transcript.
    `reask` then chases what a prompt alone does not reach — see the notes above
    COLLAPSED_WORD_LOUD. Both need `loud` to have anything to work from.
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
    if prompt:
        fields["prompt"] = prompt
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
    recovery = {"spans": 0, "attempted": 0, "recovered_spans": 0,
                "recovered_segments": 0, "skipped": 0}
    collapsed = {"spans": 0, "attempted": 0, "replaced_spans": 0,
                 "words_before": 0, "words_after": 0, "skipped": 0}
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

        # Whatever the first pass missed, asked again one span at a time. Done
        # here rather than in the plan stage because that stage has no endpoint
        # and touches no audio; done before the words are written so nothing
        # downstream ever sees the damaged transcript.
        if recover and loud:
            extra, recovery = recover_missing(
                client, wav_path, duration, fields, segments, loud,
                speech_pad, workdir, on_note=on_note,
            )
            segments.extend(extra)

        # After the recovery, not before: a span the first pass returned nothing
        # for has no word on it to look collapsed, and re-asking about it here
        # would duplicate what the recovery just brought back.
        if reask and loud:
            segments, collapsed = reask_collapsed(
                client, wav_path, duration, fields, segments, loud, workdir,
                word_loud=reask_word_seconds, window=reask_window,
                on_note=on_note,
            )
    finally:
        try:
            os.rmdir(workdir)
        except OSError:
            pass

    segments.sort(key=lambda segment: segment["offsets"]["from"])
    parsed = tr.build_from_segments(segments, participant)
    parsed["chunks"] = len(chunks)
    parsed["recovery"] = recovery
    parsed["collapsed"] = collapsed
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
