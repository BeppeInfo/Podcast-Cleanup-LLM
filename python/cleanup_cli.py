#!/usr/bin/env python3
"""Data-side entry point for the podcast cleanup pipeline.

Every subcommand reads and writes files in the episode work directory, so any
stage can be re-run by hand. Audio is never touched here: this side of the
pipeline decides *what* to do, the shell side runs ffmpeg and the models.

Progress is reported as "PROGRESS <done> <total>" lines on stdout, which the
shell turns into a live counter.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cleanup import intervals as iv  # noqa: E402
from cleanup import asr, llm, plan as planner, render, transcript as tr, vad  # noqa: E402


def _read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path, payload):
    render.write_json(path, payload)


def _progress(done, total):
    print(f"PROGRESS {done} {total}", flush=True)


# API keys arrive through the environment, never through argv: a command line is
# readable by any process on the machine, and the shell logs every command it
# runs into a file that outlives the episode.
WHISPER_KEY_ENV = "PODCAST_WHISPER_API_KEY"
LLAMA_KEY_ENV = "PODCAST_LLAMA_API_KEY"


def _api_key(name):
    return os.environ.get(name) or None


def _fail(message):
    raise SystemExit(f"error: {message}")


# --- meta ---------------------------------------------------------------------


# Codecs that reproduce their input exactly. Anything else is assumed lossy,
# which is worth saying out loud before it gets re-encoded.
LOSSLESS_CODECS = {
    "flac", "alac", "wavpack", "tta", "ape", "monkeysaudio", "shorten",
    "mlp", "truehd", "als", "pcm_s16le", "pcm_s16be", "pcm_s24le", "pcm_s24be",
    "pcm_s32le", "pcm_s32be", "pcm_f32le", "pcm_f64le", "pcm_u8", "pcm_s8",
}


def _probe(ffprobe, path):
    result = subprocess.run(
        [
            ffprobe, "-v", "error", "-select_streams", "a:0",
            "-show_entries",
            "stream=codec_name,sample_rate,channels,sample_fmt,bits_per_raw_sample",
            "-show_entries", "format=duration,format_name",
            "-of", "json", path,
        ],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        _fail(f"ffprobe failed on {path}: {result.stderr.strip()}")
    data = json.loads(result.stdout)
    streams = data.get("streams") or []
    if not streams:
        _fail(f"{path}: no audio stream ffmpeg can see")
    stream = streams[0]
    duration = float((data.get("format") or {}).get("duration") or 0.0)
    if duration <= 0:
        _fail(f"{path}: could not determine duration")
    codec = stream.get("codec_name") or ""
    return {
        # Renamed once the prepare stage has measured the real thing.
        "container_duration": round(duration, 3),
        "duration": round(duration, 3),
        "codec": codec,
        "lossless": codec in LOSSLESS_CODECS,
        "container": (data.get("format") or {}).get("format_name", ""),
        "sample_rate": int(stream["sample_rate"]),
        "channels": int(stream.get("channels") or 1),
        "sample_fmt": stream.get("sample_fmt") or "",
        "bits_per_raw_sample": int(stream.get("bits_per_raw_sample") or 0),
    }


def cmd_meta(args):
    tracks = []
    for item in args.track:
        if "=" not in item:
            _fail(f"track argument must be participant=path, got '{item}'")
        participant, path = item.split("=", 1)
        if not os.path.isfile(path):
            _fail(f"track file not found: {path}")
        probe = _probe(args.ffprobe, path)
        tracks.append({"participant": participant, "source": os.path.abspath(path), **probe})

    tracks.sort(key=lambda t: t["participant"])
    rates = sorted({t["sample_rate"] for t in tracks})

    # Every track must be frame-aligned at the same rate, or identical cuts
    # would remove different amounts of time from each and they would drift.
    resample_to = None
    if args.resample_to == "auto":
        resample_to = max(rates)
    elif args.resample_to:
        resample_to = int(args.resample_to)

    if resample_to is None and len(rates) > 1:
        detail = ", ".join(f"{t['participant']}={t['sample_rate']}Hz" for t in tracks)
        _fail(
            "all tracks of an episode must share a sample rate, so that one cut "
            f"removes the same span from every track ({detail}). Either resample "
            "beforehand, or set RESAMPLE_TO (or --resample-to auto) to have this "
            "run do it."
        )

    target_rate = resample_to if resample_to is not None else rates[0]
    for track in tracks:
        track["render_rate"] = target_rate
        track["resampled"] = track["sample_rate"] != target_rate

    durations = [t["duration"] for t in tracks]
    spread = max(durations) - min(durations)
    meta = {
        "episode_id": args.episode,
        "duration": max(durations),
        "duration_spread": round(spread, 3),
        "sample_rate": target_rate,
        "resample_to": resample_to,
        "durations_measured": False,
        "tracks": tracks,
    }
    _write_json(args.out, meta)

    formats = ", ".join(sorted({f"{t['codec']}" for t in tracks}))
    print(
        f"{len(tracks)} tracks, {formats}, {target_rate} Hz, "
        f"{meta['duration']:.1f}s (spread {spread:.3f}s)"
    )
    changed = [t for t in tracks if t["resampled"]]
    if changed:
        print(
            "resampling to "
            f"{target_rate} Hz: "
            + ", ".join(f"{t['participant']} from {t['sample_rate']}" for t in changed)
        )
    lossy = [t for t in tracks if not t["lossless"]]
    if lossy:
        print(
            "note: lossy input ("
            + ", ".join(f"{t['participant']}={t['codec']}" for t in lossy)
            + "). Cutting and re-encoding cannot recover what the codec already "
            "discarded, and a lossless output will be larger for no gain."
        )
    if spread > 1.0:
        print(
            f"warning: track lengths differ by {spread:.1f}s — if they are not "
            "aligned at sample 0 the cleanup will drift"
        )


def cmd_meta_refresh(args):
    """Replace container durations with the length of the decoded audio.

    A container's header is not always right: AAC decodes longer than it claims,
    Opus shorter, and a truncated file of any format can claim anything. The
    prepare stage has already decoded every track, so its output is the
    authority — and the frame-exact render prediction depends on it.
    """
    meta = _read_json(args.meta)
    changes = []
    for track in meta["tracks"]:
        prepared = os.path.join(args.prep_dir, f"{track['participant']}.wav")
        if not os.path.isfile(prepared):
            _fail(f"prepared track missing, cannot measure: {prepared}")
        measured = round(vad.wav_duration(prepared), 4)
        before = track["duration"]
        track["duration"] = measured
        if abs(measured - before) > 0.002:
            changes.append((track["participant"], before, measured))

    meta["duration"] = max(t["duration"] for t in meta["tracks"])
    durations = [t["duration"] for t in meta["tracks"]]
    meta["duration_spread"] = round(max(durations) - min(durations), 3)
    meta["durations_measured"] = True
    _write_json(args.meta, meta)

    print(f"measured length of {len(meta['tracks'])} tracks: {meta['duration']:.3f}s")
    for participant, before, after in changes:
        print(
            f"  {participant}: container said {before:.3f}s, decodes to "
            f"{after:.3f}s ({(after - before) * 1000:+.0f} ms)"
        )


# --- vad ----------------------------------------------------------------------


def cmd_meta_shell(args):
    """Emit meta.json as shell assignments, for `eval` by a resumed stage.

    Values go through shlex.quote, so a participant name with a space in it
    cannot turn into two words.
    """
    import shlex

    meta = _read_json(args.meta)
    tracks = meta["tracks"]
    print(f"EPISODE_ID={shlex.quote(meta['episode_id'])}")
    print(f"EPISODE_DURATION={shlex.quote(str(meta['duration']))}")
    print(f"EPISODE_SAMPLE_RATE={shlex.quote(str(meta['sample_rate']))}")
    names = " ".join(shlex.quote(t["participant"]) for t in tracks)
    print(f"PARTICIPANTS=({names})")
    for key, field in (
        ("TRACK_SOURCE", "source"),
        ("TRACK_DURATION", "duration"),
        ("TRACK_SAMPLE_FMT", "sample_fmt"),
        ("TRACK_CODEC", "codec"),
    ):
        pairs = " ".join(
            f"[{shlex.quote(t['participant'])}]={shlex.quote(str(t.get(field, '')))}"
            for t in tracks
        )
        print(f"{key}=({pairs})")


def cmd_vad_ffmpeg(args):
    with open(args.log, encoding="utf-8", errors="replace") as handle:
        speech = vad.parse_silencedetect(handle.read(), args.duration)
    _write_json(
        args.out,
        {
            "participant": args.participant,
            "backend": "ffmpeg",
            "duration": args.duration,
            "speech": [[round(s, 4), round(e, 4)] for s, e in speech],
        },
    )
    print(f"{len(speech)} speech regions, {iv.total(speech):.1f}s of speech")


def cmd_vad_silero(args):
    speech, implementation = vad.silero_speech(args.wav, threshold=args.threshold)
    duration = args.duration or vad.wav_duration(args.wav)
    _write_json(
        args.out,
        {
            "participant": args.participant,
            "backend": "silero",
            # Which package answered. The two agree to within one 32 ms chunk
            # but are not bit-identical, so an edit that needs reproducing needs
            # to know which one produced this map.
            "implementation": implementation,
            "duration": round(duration, 3),
            "speech": [[round(s, 4), round(e, 4)] for s, e in speech],
        },
    )
    print(
        f"{len(speech)} speech regions, {iv.total(speech):.1f}s of speech "
        f"(via {implementation})"
    )


# --- transcript ---------------------------------------------------------------


def cmd_words(args):
    parsed = tr.parse_whisper_json(args.whisper_json, args.participant)
    _write_json(args.out, parsed)
    note = ""
    if parsed["approximated_segments"]:
        note = (
            f", {parsed['approximated_segments']} segments had unusable token "
            "timings and were interpolated"
        )
    print(f"{len(parsed['words'])} words in {len(parsed['segments'])} segments{note}")


# --- detection ----------------------------------------------------------------


def cmd_transcribe_remote(args):
    speech = None
    if args.vad and os.path.isfile(args.vad):
        speech = [tuple(span) for span in _read_json(args.vad)["speech"]]

    client = asr.WhisperClient(
        args.endpoint, timeout=args.request_timeout, path=args.path,
        api_key=_api_key(WHISPER_KEY_ENV),
    )
    parsed = asr.transcribe(
        client,
        args.wav,
        args.participant,
        language=args.language,
        chunk_seconds=args.chunk_seconds,
        speech=speech,
        temperature=args.temperature,
        on_progress=_progress,
        skip_silence=args.skip_silence,
    )
    _write_json(args.out, parsed)

    words, segments = len(parsed["words"]), len(parsed["segments"])
    print(
        f"{words} words in {segments} segments over {parsed['chunks']} chunk(s)"
    )
    skipped = parsed.get("skipped_seconds") or 0.0
    if skipped > 0.5:
        share = skipped / max(parsed.get("audio_seconds") or 1.0, 1e-6)
        detail = (
            f"{skipped:.0f}s of {parsed['audio_seconds']:.0f}s ({share * 100:.0f}%) "
            "was not sent to Whisper: the VAD heard no speech there"
        )
        # Skipping silence is the point, but skipping most of a track is also
        # what a wrong VAD threshold looks like, and the words it drops leave no
        # trace in the transcript to notice later.
        if share > 0.5:
            print(f"WARN {detail}. If that track really was talking, lower the "
                  "VAD threshold or set WHISPER_SKIP_SILENCE=0", flush=True)
        else:
            print(f"note: {detail}")
    if parsed["approximated_segments"]:
        # Worth saying plainly: it decides how tightly a stutter can be cut.
        share = parsed["approximated_segments"] / max(segments, 1)
        detail = (
            f"{parsed['approximated_segments']}/{segments} segments arrived without "
            "per-token timings"
        )
        if words and words / max(segments, 1) < 1.5:
            print(f"note: {detail}, but they are one word each, so timings are exact")
        else:
            print(
                f"warning: {detail} ({share * 100:.0f}%), so word positions inside "
                "them are interpolated and cuts will be less tight. A local "
                "whisper-cli run, or a server build that honours max_len, gives "
                "better boundaries."
            )


def cmd_whisper_wait(args):
    client = asr.WhisperClient(
        args.endpoint, path=args.path, api_key=_api_key(WHISPER_KEY_ENV)
    )
    if not client.wait_until_ready(args.timeout):
        raise SystemExit(1)
    print("whisper endpoint reachable")


def cmd_llm_wait(args):
    client = llm.LlamaClient(
        args.endpoint, api_key=_api_key(LLAMA_KEY_ENV), api=args.api,
        model=args.model_name,
    )
    if not client.wait_until_ready(args.timeout):
        raise SystemExit(1)
    # One tiny constrained request before the episode starts. A server that
    # answers /health but ignores the schema would otherwise be discovered only
    # after every chunk of every track had been dropped for unparseable output.
    if args.check_schema:
        client.check_schema_support()
    print("llama endpoint ready")


def cmd_detect(args):
    parsed = _read_json(args.words)
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    unknown = [k for k in kinds if k not in llm.KINDS]
    if unknown:
        _fail(f"unknown edit kinds: {', '.join(unknown)} (known: {', '.join(llm.KINDS)})")

    client = llm.LlamaClient(
        args.endpoint, timeout=args.request_timeout, temperature=args.temperature,
        api_key=_api_key(LLAMA_KEY_ENV),
        api=args.api, max_reply_tokens=args.max_reply_tokens,
        model=args.model_name,
    )
    result = llm.detect(
        client,
        parsed,
        chunk_words=args.chunk_words,
        overlap=args.overlap,
        limits={
            "max_words": args.max_words,
            "max_seconds": args.max_seconds,
            "min_confidence": args.min_confidence,
        },
        accepted=kinds,
        audit_path=args.audit,
        on_progress=_progress,
        concurrency=args.concurrency,
    )
    _write_json(args.out, result)
    print(
        f"{len(result['edits'])} edits accepted, {result['rejected_count']} rejected "
        f"over {result['chunks']} chunks"
    )
    if result.get("chunks_truncated"):
        print(
            f"WARN {result['chunks_truncated']} of {result['chunks']} chunks ran "
            "past LLM_MAX_REPLY_TOKENS; the edits they had already emitted were "
            "kept, but the tail of those windows went unjudged. Raise it, or "
            "lower LLM_CHUNK_WORDS so a window yields fewer edits.",
            flush=True,
        )
    if result["chunk_failures"]:
        failed, total = result["chunk_failures"], result["chunks"]
        print(
            f"WARN {failed} of {total} chunks produced no usable response; "
            "those stretches were left untouched",
            flush=True,
        )
        # Every chunk failing is not per-item tolerance doing its job, it is the
        # whole track silently keeping its disfluencies while the report shows a
        # clean-looking zero. DESIGN.md §9 invariant 10: a fault that hits every
        # chunk identically must not be able to pass for the survivable kind.
        # Exit 4 so the shell marks the track failed rather than done — the run
        # still carries on, but it will not claim this track was analysed, and
        # --from detect will retry it instead of skipping it as complete.
        if failed == total and total > 0:
            print(
                f"WARN every chunk failed for {result['participant']}: this "
                "track was not analysed at all. If they timed out, the endpoint "
                "is too slow for LLM_CONCURRENCY windows at once — lower it, or "
                "raise LLAMA_REQUEST_TIMEOUT.",
                flush=True,
            )
            raise SystemExit(4)


# --- plan ---------------------------------------------------------------------


def _collect(directory, suffix):
    found = {}
    for path in sorted(glob.glob(os.path.join(directory, f"*{suffix}"))):
        data = _read_json(path)
        found[data["participant"]] = data
    return found


def cmd_plan(args):
    meta = _read_json(args.meta)
    params = _read_json(args.params)
    speech_files = _collect(args.vad_dir, ".json")
    word_files = _collect(args.words_dir, ".words.json")
    edit_files = _collect(args.edits_dir, ".edits.json") if args.edits_dir else {}

    participants = [t["participant"] for t in meta["tracks"]]
    missing = [p for p in participants if p not in speech_files]
    if missing:
        _fail(f"no VAD result for: {', '.join(missing)}")

    result = planner.build_plan(
        meta,
        {p: speech_files[p]["speech"] for p in participants},
        {p: edit_files[p]["edits"] for p in edit_files},
        {p: word_files[p]["words"] for p in word_files},
        params,
    )
    _write_json(args.out, result)
    report = planner.format_report(result)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            handle.write(report)
    print(report, end="")

    if result["warnings"] and not args.force:
        hard = [w for w in result["warnings"] if "safety limit" in w]
        if hard:
            raise SystemExit(
                "refusing to continue: " + "; ".join(hard) + " (override with --force)"
            )


# --- filters ------------------------------------------------------------------


def cmd_filters(args):
    meta = _read_json(args.meta)
    current = _read_json(args.plan)
    os.makedirs(args.dir, exist_ok=True)

    expectations = {}
    for track in meta["tracks"]:
        participant = track["participant"]
        mutes = current["mutes"].get(participant, [])
        # The rate the filter graph works at, which is the output rate too.
        render_rate = int(track.get("render_rate") or track["sample_rate"])
        resample = render_rate if render_rate != track["sample_rate"] else None

        graph = render.build_filter(
            current["cuts"], mutes, args.frame_samples, args.fade, resample=resample
        )
        path = os.path.join(args.dir, f"{participant}.filter")
        if graph is None:
            if os.path.exists(path):
                os.remove(path)
        else:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(graph)

        # The track may be shorter than the episode; cuts past its end are moot.
        samples = render.expected_output_samples(
            min(track["duration"], current["duration"]),
            render_rate,
            args.frame_samples,
            current["cuts"],
        )
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

    _write_json(args.out, {"frame_samples": args.frame_samples, "tracks": expectations})
    for participant, values in expectations.items():
        notes = []
        if values["passthrough"]:
            notes.append("nothing to change")
        else:
            notes.append(f"{values['mutes']} mutes")
        if values["resampled_from"]:
            notes.append(f"resampled from {values['resampled_from']} Hz")
        print(
            f"{participant}: {values['expected_duration']:.3f}s expected "
            f"({', '.join(notes)})"
        )


# --- final transcript ---------------------------------------------------------


def cmd_transcript(args):
    current = _read_json(args.plan)
    words = _collect(args.words_dir, ".words.json")
    result = render.build_transcript(current, words)
    _write_json(args.out_json, result)
    if args.out_srt:
        with open(args.out_srt, "w", encoding="utf-8") as handle:
            handle.write(render.transcript_to_srt(result))
    if args.out_txt:
        with open(args.out_txt, "w", encoding="utf-8") as handle:
            handle.write(render.transcript_to_text(result))
    print(
        f"{len(result['segments'])} segments, {result['removed_words']} words removed "
        f"by the edit"
    )


# --- verification -------------------------------------------------------------


def cmd_verify(args):
    expectations = _read_json(args.expected)["tracks"]
    problems = []
    for item in args.actual:
        if "=" not in item:
            _fail(f"actual argument must be participant=duration, got '{item}'")
        participant, value = item.split("=", 1)
        if participant not in expectations:
            problems.append(f"{participant}: rendered but not in the plan")
            continue
        actual = float(value)
        expected = expectations[participant]["expected_duration"]
        allowed = max(args.tolerance * expected, 0.05)
        delta = abs(actual - expected)
        status = "ok" if delta <= allowed else "MISMATCH"
        print(
            f"{participant:<16} expected {expected:9.3f}s  actual {actual:9.3f}s  "
            f"delta {delta:6.3f}s  {status}"
        )
        if delta > allowed:
            problems.append(
                f"{participant}: expected {expected:.3f}s but rendered {actual:.3f}s "
                f"(delta {delta:.3f}s, allowed {allowed:.3f}s)"
            )

    rendered = {item.split("=", 1)[0] for item in args.actual}
    for participant in expectations:
        if participant not in rendered:
            problems.append(f"{participant}: planned but never rendered")

    if problems:
        for problem in problems:
            print(f"error: {problem}")
        raise SystemExit(1)
    print("all tracks verified")


# --- argument parsing ---------------------------------------------------------


def build_parser():
    parser = argparse.ArgumentParser(prog="cleanup_cli", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("meta", help="probe tracks and write meta.json")
    p.add_argument("--episode", required=True)
    p.add_argument("--ffprobe", default="ffprobe")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--resample-to", default="",
        help="'auto' for the highest rate present, or an explicit rate; "
             "empty makes a rate mismatch an error",
    )
    p.add_argument("track", nargs="+", help="participant=path")
    p.set_defaults(func=cmd_meta)

    p = sub.add_parser(
        "meta-refresh", help="replace container durations with measured ones"
    )
    p.add_argument("--meta", required=True)
    p.add_argument("--prep-dir", required=True)
    p.set_defaults(func=cmd_meta_refresh)

    p = sub.add_parser("meta-shell", help="meta.json as shell assignments")
    p.add_argument("--meta", required=True)
    p.set_defaults(func=cmd_meta_shell)

    p = sub.add_parser("vad-ffmpeg", help="parse a silencedetect log")
    p.add_argument("--log", required=True)
    p.add_argument("--duration", type=float, required=True)
    p.add_argument("--participant", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_vad_ffmpeg)

    p = sub.add_parser("vad-silero", help="run Silero VAD over a prepared track")
    p.add_argument("--wav", required=True)
    p.add_argument("--participant", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--duration", type=float, default=0.0)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_vad_silero)

    p = sub.add_parser("words", help="whisper json -> words + segments")
    p.add_argument("--whisper-json", required=True)
    p.add_argument("--participant", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_words)

    p = sub.add_parser(
        "transcribe-remote", help="transcribe a track via a whisper-server endpoint"
    )
    p.add_argument("--wav", required=True)
    p.add_argument("--participant", required=True)
    p.add_argument("--endpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--vad", help="VAD result, used to place chunk boundaries in silence")
    p.add_argument("--chunk-seconds", type=float, default=600.0)
    p.add_argument(
        "--no-skip-silence", dest="skip_silence", action="store_false",
        help="send the whole track, including stretches the VAD calls silence",
    )
    p.add_argument("--language", default="auto")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--request-timeout", type=float, default=1800.0)
    p.add_argument("--path", default="/inference")
    p.set_defaults(func=cmd_transcribe_remote)

    p = sub.add_parser("whisper-wait", help="check a whisper endpoint is reachable")
    p.add_argument("--endpoint", required=True)
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--path", default="/inference")
    p.set_defaults(func=cmd_whisper_wait)

    p = sub.add_parser("llm-wait", help="block until the llama endpoint is ready")
    p.add_argument("--endpoint", required=True)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--api", choices=("chat", "completion"), default="chat")
    p.add_argument("--model-name", default="")
    p.add_argument(
        "--check-schema", action="store_true",
        help="also verify the server constrains replies to a JSON schema",
    )
    p.set_defaults(func=cmd_llm_wait)

    p = sub.add_parser("detect", help="find disfluencies in one participant's words")
    p.add_argument("--words", required=True)
    p.add_argument("--endpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--audit")
    p.add_argument("--chunk-words", type=int, default=350)
    p.add_argument("--overlap", type=int, default=40)
    p.add_argument("--max-words", type=int, default=12)
    p.add_argument("--max-seconds", type=float, default=4.0)
    p.add_argument("--min-confidence", type=float, default=0.6)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--request-timeout", type=float, default=600.0)
    p.add_argument("--api", choices=("chat", "completion"), default="chat")
    p.add_argument("--max-reply-tokens", type=int, default=2048)
    p.add_argument("--model-name", default="")
    p.add_argument("--kinds", default="stutter,repetition,false_start")
    p.add_argument(
        "--concurrency", type=int, default=1,
        help="windows in flight at once; match the server's --parallel slots",
    )
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser("plan", help="unify silence and edits into an edit plan")
    p.add_argument("--meta", required=True)
    p.add_argument("--params", required=True)
    p.add_argument("--vad-dir", required=True)
    p.add_argument("--words-dir", required=True)
    p.add_argument("--edits-dir")
    p.add_argument("--out", required=True)
    p.add_argument("--report")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("filters", help="write per-track ffmpeg filter graphs")
    p.add_argument("--meta", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--frame-samples", type=int, default=512)
    p.add_argument("--fade", type=float, default=0.030)
    p.set_defaults(func=cmd_filters)

    p = sub.add_parser("transcript", help="build the final speaker transcript")
    p.add_argument("--plan", required=True)
    p.add_argument("--words-dir", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-srt")
    p.add_argument("--out-txt")
    p.set_defaults(func=cmd_transcript)

    p = sub.add_parser("verify", help="compare rendered durations against the plan")
    p.add_argument("--expected", required=True)
    p.add_argument("--tolerance", type=float, default=0.02)
    p.add_argument("actual", nargs="+", help="participant=duration")
    p.set_defaults(func=cmd_verify)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except llm.AuthRejected as exc:
        # A clear sentence beats a traceback for something this mundane.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except llm.SchemaIgnored as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except llm.ModelUnavailable as exc:
        # Every remaining window would fail the same way, so this ends the run
        # rather than the track. Exit 5 keeps it distinct from 4 ("this track
        # could not be analysed"), which is survivable and lets the rest finish.
        print(f"error: {exc}", file=sys.stderr)
        return 5
    except BrokenPipeError:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
