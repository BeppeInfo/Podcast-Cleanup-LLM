"""The stages, as the shell used to run them.

One function per stage, each reading what it needs from the episode work
directory and writing its results back there. That is what makes `--from`,
`--only` and a resumed run work without special cases, and it is what will let
the web app drive the same stages the CLI does.

A stage takes the resolved settings and a `runlog.Log`, does its work, and
raises `StageError` with something worth reading if it cannot. It does not
decide whether the run continues — that belongs to whatever is sequencing them.
"""

from __future__ import annotations

import concurrent.futures
import glob
import json
import os

from . import intervals as iv
from . import plan as planner
from . import asr, proc, render, silence, transcript as tr


class StageError(Exception):
    """The stage could not finish. The message is for the operator."""


# Settings that reach the plan as numbers, and the names it knows them by.
# Written to params.json as well, which is a record of what a run used and is
# what tools/sweep_params.py re-plans against.
PLAN_PARAMS = {
    "silence_min_duration": "SILENCE_MIN_DURATION",
    "silence_keep": "SILENCE_KEEP",
    "edge_keep": "EDGE_KEEP",
    "cut_padding": "CUT_PADDING",
    "min_cut": "MIN_CUT",
    "mute_fade": "MUTE_FADE",
    "speech_pad": "SPEECH_PAD",
    "max_cut_fraction": "MAX_CUT_FRACTION",
}


def plan_params(settings) -> dict[str, float]:
    return {key: float(settings[name]) for key, name in PLAN_PARAMS.items()}


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def collect(directory: str, suffix: str) -> dict:
    """{participant: parsed} for every `<participant><suffix>` in a directory."""
    found = {}
    for path in sorted(glob.glob(os.path.join(directory, f"*{suffix}"))):
        participant = os.path.basename(path)[: -len(suffix)]
        found[participant] = read_json(path)
    return found


def build_speech_map(participants, words, durations, loud, pad, clip, log):
    """The speech map, and a word about it when the level scan shortened it."""
    speech = {
        name: tr.speech_from_words(
            words[name], durations[name], pad=pad,
            loud=loud.get(name) if clip else None,
        )
        for name in participants
    }
    if clip and loud:
        trimmed = {
            name: round(
                iv.total(tr.speech_from_words(words[name], durations[name], pad=pad))
                - iv.total(speech[name]), 1)
            for name in loud
        }
        busy = {name: amount for name, amount in trimmed.items() if amount >= 1.0}
        if busy:
            log.warn(
                "word timings ran past the audio by "
                + ", ".join(f"{name} {amount}s" for name, amount in sorted(busy.items()))
                + "; the speech map is the overlap with the level scan")
    return speech


def refresh_durations(work: str, log) -> dict:
    """Replace container durations with the length of the decoded audio.

    A container's header is not always right: AAC decodes longer than it claims,
    Opus shorter, and a truncated file of any format can claim anything. The
    prepare stage has just decoded every track, so its output is the authority —
    and the frame-exact render prediction is only exact if this is.
    """
    path = os.path.join(work, "meta.json")
    meta = read_json(path)
    changes = []
    for track in meta["tracks"]:
        prepared = os.path.join(work, "prep", f"{track['participant']}.wav")
        if not os.path.isfile(prepared):
            raise StageError(f"prepared track missing, cannot measure: {prepared}")
        measured = round(silence.wav_duration(prepared), 4)
        before = track["duration"]
        track["duration"] = measured
        if abs(measured - before) > 0.002:
            changes.append((track["participant"], before, measured))

    meta["duration"] = max(t["duration"] for t in meta["tracks"])
    durations = [t["duration"] for t in meta["tracks"]]
    meta["duration_spread"] = round(max(durations) - min(durations), 3)
    meta["durations_measured"] = True
    render.write_json(path, meta)

    log.report(
        f"measured length of {len(meta['tracks'])} tracks: {meta['duration']:.3f}s")
    for participant, before, after in changes:
        log.report(f"  {participant}: container said {before:.3f}s, decodes to "
                   f"{after:.3f}s ({(after - before) * 1000:+.0f} ms)")
    return meta


def decode_command(ffmpeg: str, source: str, target: str):
    """16 kHz mono PCM — the one format Whisper and Silero both want."""
    return [
        ffmpeg, "-nostdin", "-y", "-v", "warning", "-progress", "pipe:1",
        "-nostats", "-i", source,
        "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        "-f", "wav", target,
    ]


def stage_prepare(work: str, settings, log, ffmpeg: str = "ffmpeg") -> None:
    """Decode every track to what the transcriber wants, then measure it.

    Already-decoded tracks are skipped on their marker, which is what lets a
    resumed run pick up mid-episode.
    """
    meta = read_json(os.path.join(work, "meta.json"))
    state = os.path.join(work, "state")
    jobs = max(1, int(settings["FFMPEG_JOBS"]))

    pending = []
    for track in meta["tracks"]:
        participant = track["participant"]
        target = os.path.join(work, "prep", f"{participant}.wav")
        marker = os.path.join(state, f"prep-{participant}.ok")
        done = (os.path.isfile(marker) and os.path.isfile(target)
                and os.path.getsize(target) > 0)
        if done:
            log.debug(f"{participant} already prepared")
            continue
        if os.path.exists(marker):
            os.remove(marker)
        pending.append((participant, track, target, marker))

    def decode(item) -> tuple[str, int]:
        participant, track, target, marker = item
        argv = decode_command(ffmpeg, track["source"], target)
        # Live progress only when one job owns the console; concurrent writers
        # on one in-place counter garble each other.
        on_line = None
        if jobs <= 1:
            on_line = proc.ffmpeg_progress(
                log, float(track["duration"]) * 1_000_000, f"decoding {participant}")
        status = proc.run(argv, log, on_line=on_line)
        if status == 0:
            open(marker, "w").close()
            log.ok(f"decoded {participant}")
        return participant, status

    if jobs <= 1 or len(pending) <= 1:
        results = [decode(item) for item in pending]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(decode, pending))

    failed = [name for name, status in results if status != 0]
    if failed:
        raise StageError(f"decoding failed for: {', '.join(sorted(failed))}")

    # Unconditionally, even when everything was already decoded: `discover`
    # rewrites meta.json with container durations and clears durations_measured,
    # so a run resumed at this stage would otherwise carry those forward into a
    # render prediction that is supposed to be frame-exact.
    refresh_durations(work, log)


VAD_REQUEST_FIELDS = (
    ("vad_threshold", "WHISPER_VAD_THRESHOLD"),
    ("vad_min_speech_duration_ms", "WHISPER_VAD_MIN_SPEECH_MS"),
    ("vad_min_silence_duration_ms", "WHISPER_VAD_MIN_SILENCE_MS"),
    ("vad_speech_pad_ms", "WHISPER_VAD_SPEECH_PAD_MS"),
    ("vad_samples_overlap", "WHISPER_VAD_SAMPLES_OVERLAP"),
)


def scan_levels(work: str, participant: str, wav: str, duration: float,
                settings, log, ffmpeg: str):
    """ffmpeg's silencedetect for one track, parsed into non-silent stretches.

    Every track, not only the ones long enough to split into chunks: the plan
    stage uses this as the one opinion about the audio Whisper had no hand in.
    Returns None when it could not be done, which is a warning rather than a
    failure — the run continues with less to cross-check against.
    """
    scan_log = os.path.join(work, "asr", f"{participant}.silence.log")
    target = os.path.join(work, "asr", f"{participant}.loud.json")
    threshold = settings["SPLIT_SILENCE_THRESHOLD"]
    argv = [
        ffmpeg, "-nostdin", "-v", "info", "-i", wav,
        "-af", f"silencedetect=noise={threshold}:d={settings['SPLIT_MIN_SILENCE']}",
        "-f", "null", "-",
    ]
    if proc.run_to_file(argv, log, scan_log) != 0:
        log.warn(f"could not scan {participant} for quiet spots: chunk boundaries "
                 "may land mid-word, and nothing will cross-check its transcript "
                 "against its audio")
        return None

    with open(scan_log, encoding="utf-8", errors="replace") as handle:
        loud = silence.parse_silencedetect(handle.read(), duration)
    render.write_json(target, {
        "participant": participant,
        "duration": round(duration, 3),
        "threshold": threshold,
        "loud": [[round(s, 4), round(e, 4)] for s, e in loud],
    })
    log.raw(f"{len(loud)} non-silent stretches, {iv.total(loud):.1f}s of "
            f"{duration:.1f}s above the threshold")
    return [tuple(span) for span in loud]


def describe_transcript(parsed, settings, log) -> None:
    """What the transcription found, and what it had to work around.

    The `note:` lines go to the run log only, which is where they went when the
    shell was reading this off a pipe — its parser matched PROGRESS and WARN and
    let everything else fall through to the log. Worth knowing that the comment
    in the old code calling one of these "worth saying out loud" was not actually
    achieved; changing that is a decision, not part of a port.
    """
    log.raw(f"{len(parsed['words'])} words in {len(parsed['segments'])} segments "
            f"over {parsed['chunks']} chunk(s)")

    recovery = parsed.get("recovery") or {}
    if recovery.get("spans"):
        detail = (f"{recovery['spans']} stretch(es) of loud audio came back with "
                  f"no words; re-asked about {recovery['attempted']} and recovered "
                  f"{recovery['recovered_segments']} word(s) from "
                  f"{recovery['recovered_spans']}")
        log.raw("note: " + detail
                + ("" if recovery["recovered_spans"] else " — probably not speech, then"))
        if recovery.get("skipped"):
            log.warn(f"{recovery['skipped']} further stretch(es) were left unasked; "
                     "that many is not the occasional skipped window")

    collapsed = parsed.get("collapsed") or {}
    if collapsed.get("spans"):
        log.raw(f"note: {collapsed['spans']} word(s) sat on more than "
                f"{settings['WHISPER_REASK_WORD_SECONDS']}s of speech; asked again "
                f"in {settings['WHISPER_REASK_WINDOW']}s windows and replaced "
                f"{collapsed['replaced_spans']}, turning "
                f"{collapsed['words_before']} word(s) into "
                f"{collapsed['words_after']}")
        if collapsed.get("skipped"):
            log.warn(f"{collapsed['skipped']} further collapsed word(s) were left "
                     "unasked; that many is not the occasional fluent reading")

    empty = parsed.get("chunks_without_speech") or 0
    if empty:
        # Normal on a two-mic recording; total silence is not.
        detail = f"{empty}/{parsed['chunks']} chunk(s) held no speech at all"
        if empty == parsed["chunks"]:
            log.warn(f"{detail}, so this track has no transcript. If it really was "
                     "talking, the server is dropping it — check that vad_* is "
                     "honoured and lower WHISPER_CHUNK_SECONDS")
        else:
            log.raw(f"note: {detail}")


def stage_transcribe(work: str, settings, log, ffmpeg: str = "ffmpeg",
                     api_key=None, ready_timeout: float = 60.0) -> None:
    """Transcribe every track, and scan each one's levels while we are here."""
    meta = read_json(os.path.join(work, "meta.json"))
    state = os.path.join(work, "state")
    endpoint = settings["WHISPER_ENDPOINT"]

    client = asr.WhisperClient(
        endpoint, timeout=float(settings["WHISPER_REQUEST_TIMEOUT"]),
        path=settings["WHISPER_ENDPOINT_PATH"], api_key=api_key,
    )
    if not client.wait_until_ready(ready_timeout):
        raise StageError(f"the whisper endpoint at {endpoint} is not usable")

    vad = settings["WHISPER_VAD"] == "1"
    vad_options = {field: settings[name] for field, name in VAD_REQUEST_FIELDS}

    total = len(meta["tracks"])
    for index, track in enumerate(meta["tracks"], start=1):
        participant = track["participant"]
        target = os.path.join(work, "words", f"{participant}.words.json")
        marker = os.path.join(state, f"asr-{participant}.ok")
        wav = os.path.join(work, "prep", f"{participant}.wav")

        if (os.path.isfile(marker) and os.path.isfile(target)
                and os.path.getsize(target) > 0):
            log.debug(f"{participant} already transcribed")
            continue
        if not (os.path.isfile(wav) and os.path.getsize(wav) > 0):
            raise StageError(f"missing prepared track: {wav}")

        log.info(f"whisper (remote): {participant} ({index}/{total})")
        duration = float(track["duration"])
        loud = scan_levels(work, participant, wav, duration, settings, log, ffmpeg)

        parsed = asr.transcribe(
            client, wav, participant,
            language=settings["WHISPER_LANG"],
            chunk_seconds=float(settings["WHISPER_CHUNK_SECONDS"]),
            loud=loud,
            temperature=0.0,
            on_progress=lambda done, count, name=participant: log.progress(
                done, count, name),
            on_note=lambda message: log.raw(f"note: {message}"),
            vad=vad,
            vad_options=vad_options,
            recover=settings["WHISPER_RECOVER"] == "1",
            speech_pad=float(settings["SPEECH_PAD"]),
            prompt=settings["WHISPER_PROMPT"],
            reask=settings["WHISPER_REASK"] == "1",
            reask_word_seconds=float(settings["WHISPER_REASK_WORD_SECONDS"]),
            reask_window=float(settings["WHISPER_REASK_WINDOW"]),
        )
        log.progress_done()
        render.write_json(target, parsed)
        describe_transcript(parsed, settings, log)

        if not vad:
            log.warn("server-side VAD is off, so silence was transcribed as well; "
                     "anything invented there becomes speech in the plan")
        open(marker, "w").close()


def stage_plan(work: str, settings, log, force: bool = False) -> dict:
    """Unify the transcripts and the LLM's findings into one plan.

    Writes params.json, plan.json, edit-report.txt, one filtergraph per track
    and expected.json — the frame-exact prediction the render stage is checked
    against. Returns the plan.
    """
    meta = read_json(os.path.join(work, "meta.json"))
    params = plan_params(settings)
    render.write_json(os.path.join(work, "params.json"), params)

    participants = [t["participant"] for t in meta["tracks"]]
    words_by = collect(os.path.join(work, "words"), ".words.json")

    # A missing transcript is fatal rather than "that track was silent". Silence
    # is the absence of words, so a track without any would read as silent
    # everywhere — and since cuts only happen where every track is silent, it
    # would stop protecting its own audio from everyone else's cuts.
    missing = [p for p in participants if p not in words_by]
    if missing:
        raise StageError(
            f"no transcript for: {', '.join(missing)} — the speech map is derived "
            "from the words, so a track without one cannot be planned")

    words = {p: words_by[p]["words"] for p in participants}
    durations = {t["participant"]: float(t["duration"]) for t in meta["tracks"]}

    edits = {}
    if settings["LLM_ENABLE"] == "1":
        edits = {
            name: found["edits"]
            for name, found in collect(os.path.join(work, "llm"), ".edits.json").items()
        }

    # Absent when the level scan could not be run. The cross-check then cannot
    # happen, which is worth saying rather than passing over in silence.
    loud_by = collect(os.path.join(work, "asr"), ".loud.json")
    loud = {p: loud_by[p]["loud"] for p in participants if p in loud_by}
    unscanned = [p for p in participants if p not in loud]
    if unscanned:
        log.warn(f"no level scan for {', '.join(unscanned)}, so nothing checks "
                 "their transcripts against their own audio")

    speech = build_speech_map(
        participants, words, durations, loud, params["speech_pad"],
        settings["SPEECH_MAP_CLIP"] == "1", log)

    result = planner.build_plan(meta, speech, edits, words, params, loud=loud)
    render.write_json(os.path.join(work, "plan.json"), result)

    report = planner.format_report(result)
    with open(os.path.join(work, "edit-report.txt"), "w", encoding="utf-8") as handle:
        handle.write(report)
    log.report(report)

    blocking = result.get("blocking") or []
    if blocking and not force:
        raise StageError(
            "refusing to continue: " + "; ".join(blocking) + " (override with --force)")

    describe_filters(write_filters(work, meta, result, settings), log)
    return result


def write_filters(work: str, meta, current, settings) -> dict:
    """One ffmpeg filtergraph per track, and the length each should come out."""
    directory = os.path.join(work, "render")
    os.makedirs(directory, exist_ok=True)
    frame_samples = int(settings["RENDER_FRAME_SAMPLES"])
    fade = float(settings["MUTE_FADE"])

    expectations = {}
    for track in meta["tracks"]:
        participant = track["participant"]
        mutes = current["mutes"].get(participant, [])
        # The rate the filter graph works at, which is the output rate too.
        render_rate = int(track.get("render_rate") or track["sample_rate"])
        resample = render_rate if render_rate != track["sample_rate"] else None

        graph = render.build_filter(
            current["cuts"], mutes, frame_samples, fade, resample=resample)
        path = os.path.join(directory, f"{participant}.filter")
        if graph is None:
            if os.path.exists(path):
                os.remove(path)
        else:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(graph)

        # The track may be shorter than the episode; cuts past its end are moot.
        samples = render.expected_output_samples(
            min(track["duration"], current["duration"]),
            render_rate, frame_samples, current["cuts"])
        expectations[participant] = {
            "filter": path if graph is not None else None,
            "passthrough": graph is None,
            "mutes": len(mutes),
            "resampled_from": track["sample_rate"] if resample else None,
            "expected_samples": samples,
            "expected_duration": round(samples / float(render_rate), 3),
            "sample_rate": render_rate,
            "sample_fmt": track["sample_fmt"],
            "source_lossless": track.get("lossless", True),
        }

    render.write_json(os.path.join(work, "expected.json"),
                      {"frame_samples": frame_samples, "tracks": expectations})
    return expectations


def describe_filters(expectations, log) -> None:
    """What each track will come out as, in the same words the shell used."""
    for participant, values in expectations.items():
        if values["passthrough"]:
            notes = ["nothing to change"]
        else:
            notes = [f"{values['mutes']} mutes"]
        if values["resampled_from"]:
            notes.append(f"resampled from {values['resampled_from']} Hz")
        log.debug(f"{participant}: {values['expected_duration']:.3f}s expected "
                  f"({', '.join(notes)})")
