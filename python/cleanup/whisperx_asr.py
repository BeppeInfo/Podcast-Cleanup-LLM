"""Transcription in this process, with word timings from forced alignment.

This replaces the whisper-server client. The reason is not that HTTP was in the
way — it is that whisper's own word timings are interpolated from token
positions, and everything here is decided in seconds: which stretches are
speech, where a cut may start, how long the result should be. WhisperX aligns
the transcript against the audio with wav2vec2 and returns a real start and end
per word, which is the number this pipeline has always wanted.

**Not a server.** The old design kept both models behind HTTP so neither was
this program's problem. That still holds for the LLM. It stopped being worth it
for transcription once the answer was "run it on the CPU of whichever machine
this is" — WhisperX has no server mode, both available machines are Radeon and
so CPU-only anyway, and the web interface is already the remote thing. A second
hop bought nothing.

**Diarization is not used and never will be.** Every episode arrives as one
track per participant, so who is speaking is a fact about the filename. That is
half of what WhisperX is famous for and none of what it is here for.

The import is deferred: `python/cleanup/` is standard library so a checkout can
run the CLI without installing anything, and only transcription breaks that.
"""

from __future__ import annotations

DEFAULT_MODEL = "small"
DEFAULT_COMPUTE_TYPE = "int8"
DEFAULT_BATCH_SIZE = 8


class WhisperXMissing(Exception):
    """whisperx is not installed in this interpreter."""


def _load_whisperx():
    """Import whisperx, and say something useful when it is not there.

    Kept as a function so the tests can replace it: everything below works
    against whatever this returns, and none of it needs the real package.
    """
    try:
        import whisperx
    except ImportError as exc:                       # pragma: no cover - trivial
        raise WhisperXMissing(
            "whisperx is not installed in this interpreter. The pipeline is "
            "standard library and runs from a checkout, but transcription is "
            "not — use the container image, which carries it, or install "
            "whisperx here."
        ) from exc
    return whisperx


def fill_missing_timings(words, start: float, end: float) -> list[dict]:
    """Give every word a start and an end, borrowing from its neighbours.

    Alignment leaves some words untimed — numerals and symbols it cannot map to
    audio frames come back with `start` and `end` absent. Dropping those would
    be the worst thing this code could do: the speech map is *derived from these
    words*, silence is defined as their absence, and cuts happen where every
    track is silent. A dropped word is therefore not a missing label, it is
    audio that has stopped defending itself and can be cut out from under the
    speaker.

    So an untimed run is spread evenly across the gap between the timings that
    do exist on either side of it, which is what `transcript._segment_words_by_
    proportion` does for a segment with no token positions. The timing is a
    guess; the protection is not.
    """
    timed = []
    for word in words:
        text = str(word.get("word", word.get("text", ""))).strip()
        if not text:
            continue
        begin, finish = word.get("start"), word.get("end")
        timed.append({
            "text": text,
            "start": None if begin is None else float(begin),
            "end": None if finish is None else float(finish),
        })
    if not timed:
        return []

    # Walk each run of untimed words and share out the space around it.
    index = 0
    while index < len(timed):
        if timed[index]["start"] is not None and timed[index]["end"] is not None:
            index += 1
            continue
        run_start = index
        while index < len(timed) and (
                timed[index]["start"] is None or timed[index]["end"] is None):
            index += 1
        before = timed[run_start - 1]["end"] if run_start > 0 else None
        after = timed[index]["start"] if index < len(timed) else None
        left = start if before is None else before
        right = end if after is None else after
        if right < left:
            right = left
        step = (right - left) / (index - run_start)
        for offset, word in enumerate(timed[run_start:index]):
            word["start"] = left + step * offset
            word["end"] = left + step * (offset + 1)

    for word in timed:
        if word["end"] < word["start"]:
            word["end"] = word["start"]
    return timed


def segments_from_alignment(aligned) -> list[dict]:
    """Aligned output as whisper.cpp-shaped segments, one word each.

    One word per segment is not a compromise, it is the shape the rest of this
    pipeline already reads: the old client asked whisper-server for `max_len=1`
    and `split_on_word` precisely so that segments arrived one word at a time
    with timings that were measured rather than interpolated. Alignment produces
    that natively, so `transcript.build_from_segments` needs no changes and
    neither does anything downstream of it.
    """
    out: list[dict] = []
    for segment in aligned.get("segments") or []:
        start = float(segment.get("start") or 0.0)
        end = float(segment.get("end") or start)
        for word in fill_missing_timings(segment.get("words") or [], start, end):
            begin_ms = int(round(word["start"] * 1000))
            end_ms = int(round(word["end"] * 1000))
            out.append({
                "text": word["text"],
                "offsets": {"from": begin_ms, "to": max(end_ms, begin_ms)},
            })
    out.sort(key=lambda segment: segment["offsets"]["from"])
    return out


class Transcriber:
    """One loaded model, reused for every track in the episode.

    Loading costs tens of seconds on a CPU and an episode is several tracks, so
    it happens once per run rather than once per track. The aligner is loaded
    separately and per language, because the language is not known until the
    first track has been transcribed.
    """

    def __init__(self, model: str = DEFAULT_MODEL, device: str = "cpu",
                 compute_type: str = DEFAULT_COMPUTE_TYPE, language: str = "",
                 prompt: str = "", vad_method: str = "", vad_options=None,
                 threads: int = 4, temperature_fallback: bool = False,
                 loader=None):
        self.device = device
        self.language = language if language and language != "auto" else ""
        self._whisperx = (loader or _load_whisperx)()
        self._align = {}
        # The prompt belongs to the model, not to the call: WhisperX builds its
        # decode options once at load time and its pipeline's transcribe() takes
        # no initial_prompt at all. Passing it per call raises TypeError, which
        # is a kinder failure than the one it would otherwise be — a prompt
        # silently ignored means fluent prose, no fillers, and nothing for the
        # detector to find. See DESIGN.md §6.
        asr_options = {}
        if prompt:
            asr_options["initial_prompt"] = prompt
        if not temperature_fallback:
            # One decode, at temperature 0, kept whatever it says. WhisperX
            # otherwise re-decodes at rising temperatures whenever a pass looks
            # too repetitive or too improbable — and a stutter looks exactly
            # like that, so the retry removes the thing this pipeline exists to
            # find. This does not make a run reproducible; threads do, and the
            # ladder was innocent of that. See WHISPER_THREADS.
            asr_options["temperatures"] = [0.0]
        # vad_method is passed explicitly rather than left to default. WhisperX
        # picks pyannote when it is not told, and whisper-server ran Silero —
        # so leaving this out does not keep the old behaviour, it quietly
        # changes which detector decides what counts as speech. That decision
        # is upstream of everything: the speech map comes from the words, and
        # the words come from whatever the VAD passed through.
        self._model = self._whisperx.load_model(
            model, device=device, compute_type=compute_type,
            language=self.language or None, threads=threads,
            vad_method=vad_method or "pyannote",
            **({"asr_options": asr_options} if asr_options else {}),
            **({"vad_options": vad_options} if vad_options else {}),
        )

    def _aligner(self, language: str):
        if language not in self._align:
            self._align[language] = self._whisperx.load_align_model(
                language_code=language, device=self.device)
        return self._align[language]

    def transcribe(self, wav_path: str, *,
                   batch_size: int = DEFAULT_BATCH_SIZE) -> tuple[list[dict], str]:
        """One track's words, as segments, plus the language that was used."""
        audio = self._whisperx.load_audio(wav_path)
        options = {"batch_size": batch_size}
        if self.language:
            options["language"] = self.language
        result = self._model.transcribe(audio, **options)

        language = result.get("language") or self.language or "en"
        segments = result.get("segments") or []
        if not segments:
            return [], language

        model, meta = self._aligner(language)
        aligned = self._whisperx.align(
            segments, model, meta, audio, self.device,
            return_char_alignments=False)
        return segments_from_alignment(aligned), language
