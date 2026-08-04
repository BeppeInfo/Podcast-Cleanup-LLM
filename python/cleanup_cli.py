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
from cleanup import asr, config as cfg, discover, llm, plan as planner  # noqa: E402
from cleanup import pipeline, render, runlog, silence, transcript as tr  # noqa: E402


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


def build_meta(track_pairs, episode, ffprobe, resample_to, say=print):
    """Probe each track and assemble meta.json's contents.

    Shared by `meta` and `discover`; the checks it makes — one sample rate per
    episode, lossy inputs, a suspicious duration spread — belong to the episode,
    not to whichever subcommand asked.

    `say` is where the human-readable findings go. `discover` sends them to
    stderr, because its stdout is shell assignments that get eval'd.
    """
    tracks = []
    for item in track_pairs:
        if "=" not in item:
            _fail(f"track argument must be participant=path, got '{item}'")
        participant, path = item.split("=", 1)
        if not os.path.isfile(path):
            _fail(f"track file not found: {path}")
        probe = _probe(ffprobe, path)
        tracks.append({"participant": participant, "source": os.path.abspath(path), **probe})

    tracks.sort(key=lambda t: t["participant"])
    rates = sorted({t["sample_rate"] for t in tracks})

    # Every track must be frame-aligned at the same rate, or identical cuts
    # would remove different amounts of time from each and they would drift.
    if resample_to == "auto":
        resample_to = max(rates)
    elif resample_to:
        resample_to = int(resample_to)
    else:
        resample_to = None

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
        "episode_id": episode,
        "duration": max(durations),
        "duration_spread": round(spread, 3),
        "sample_rate": target_rate,
        "resample_to": resample_to,
        "durations_measured": False,
        "tracks": tracks,
    }

    formats = ", ".join(sorted({f"{t['codec']}" for t in tracks}))
    say(
        f"{len(tracks)} tracks, {formats}, {target_rate} Hz, "
        f"{meta['duration']:.1f}s (spread {spread:.3f}s)"
    )
    changed = [t for t in tracks if t["resampled"]]
    if changed:
        say(
            "resampling to "
            f"{target_rate} Hz: "
            + ", ".join(f"{t['participant']} from {t['sample_rate']}" for t in changed)
        )
    lossy = [t for t in tracks if not t["lossless"]]
    if lossy:
        say(
            "note: lossy input ("
            + ", ".join(f"{t['participant']}={t['codec']}" for t in lossy)
            + "). Cutting and re-encoding cannot recover what the codec already "
            "discarded, and a lossless output will be larger for no gain."
        )
    if spread > 1.0:
        say(
            f"warning: track lengths differ by {spread:.1f}s — if they are not "
            "aligned at sample 0 the cleanup will drift"
        )
    return meta
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
def cmd_config(args):
    """Resolve every setting and emit it as shell assignments.

    The launcher `eval`s this, so the Python side is the single authority for
    defaults, precedence and validation, and the shell keeps only what it needs
    to run ffmpeg. Warnings go to stderr so `eval` never swallows them.
    """
    overrides = {}
    for item in args.set or []:
        if "=" not in item:
            _fail(f"--set expects NAME=VALUE, got '{item}'")
        name, _, value = item.partition("=")
        overrides[name] = value

    def warn(message):
        print(f"WARN {message}", file=sys.stderr, flush=True)

    try:
        settings = cfg.load(args.config or "", overrides)
        cfg.resolve_paths(settings, args.script_root)
        cfg.validate(settings, warn)
        if not args.no_keys:
            cfg.resolve_api_keys(settings, warn)
    except cfg.ConfigError as exc:
        _fail(str(exc))
    print(cfg.to_shell(settings))


def cmd_discover(args):
    """Identify the episode, build its work tree, and write meta.json.

    Emits shell assignments on stdout so the launcher can `eval` them, in the
    same way `meta-shell` does for a resumed run. Exit 2 means an empty inbox,
    which is not a failure — the launcher reports "nothing to do" and stops.
    """
    import shlex

    # The same log the launcher is writing, so these lines land in order and in
    # the same format. See cleanup/runlog.py.
    log = runlog.Log.from_env()
    try:
        paths = (list(args.file) if args.file
                 else discover.find_tracks(args.input_dir, args.exts.split()))
        episode_id, tracks = discover.parse_tracks(
            paths, args.separator, args.episode)
    except discover.NothingToDo as exc:
        # Not a failure: an empty inbox. Exit 2 tells the launcher to stop
        # quietly rather than report a fault.
        log.warn(f"{exc} — nothing to do")
        raise SystemExit(2) from None
    except discover.DiscoverError as exc:
        log.error(str(exc))
        raise SystemExit(1) from None

    where = discover.episode_paths(episode_id, args.work_root, args.output_dir)
    if not args.dry_run:
        discover.make_work_tree(where)
        meta = build_meta(
            [f"{name}={path}" for name, path in sorted(tracks.items())],
            episode_id, args.ffprobe, args.resample_to,
            say=lambda line: (log.warn(line.split(": ", 1)[-1])
                              if line.startswith(("warning:", "note:"))
                              else log.info(line)),
        )
        _write_json(os.path.join(where["work"], "meta.json"), meta)

    print(f"EPISODE_ID={shlex.quote(episode_id)}")
    print(f"WORK={shlex.quote(where['work'])}")
    print(f"STAGE_DIR={shlex.quote(where['state'])}")
    print(f"OUT_DIR={shlex.quote(where['output'])}")
    print(f"STAGING_DIR={shlex.quote(where['staging'])}")
    names = " ".join(shlex.quote(n) for n in sorted(tracks))
    print(f"PARTICIPANTS=({names})")
    for name, path in sorted(tracks.items()):
        print(f"TRACK_SOURCE[{shlex.quote(name)}]={shlex.quote(path)}")


def cmd_config_names(args):
    """Every setting name, one per line, for the launcher to export."""
    for name in cfg.SETTINGS:
        print(name)


# --- detection ----------------------------------------------------------------


VAD_REQUEST_FIELDS = (
    ("vad_threshold", "vad_threshold"),
    ("vad_min_speech_duration_ms", "vad_min_speech_ms"),
    ("vad_min_silence_duration_ms", "vad_min_silence_ms"),
    ("vad_speech_pad_ms", "vad_speech_pad_ms"),
    ("vad_samples_overlap", "vad_samples_overlap"),
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
    word_files = _collect(args.words_dir, ".words.json")
    edit_files = _collect(args.edits_dir, ".edits.json") if args.edits_dir else {}

    participants = [t["participant"] for t in meta["tracks"]]
    # A missing transcript is fatal rather than "that track was silent". Silence
    # is now the absence of words, so a track with no words at all would read as
    # silent everywhere — and since cuts only happen where every track is silent,
    # it would stop protecting its own audio from everyone else's cuts.
    missing = [p for p in participants if p not in word_files]
    if missing:
        _fail(
            f"no transcript for: {', '.join(missing)} — the speech map is derived "
            "from the words, so a track without one cannot be planned"
        )

    words = {p: word_files[p]["words"] for p in participants}
    durations = {t["participant"]: float(t["duration"]) for t in meta["tracks"]}

    # Absent when the level scan could not be run. The cross-check then cannot
    # happen, which is worth saying rather than passing over in silence.
    loud_files = _collect(args.loud_dir, ".loud.json") if args.loud_dir else {}
    loud = {p: loud_files[p]["loud"] for p in participants if p in loud_files}
    unscanned = [p for p in participants if p not in loud]
    if unscanned:
        print(
            f"note: no level scan for {', '.join(unscanned)}, so nothing checks "
            "their transcripts against their own audio"
        )

    # The same construction the stage uses, so the two cannot drift.
    speech = pipeline.build_speech_map(
        participants, words, durations, loud, params["speech_pad"],
        args.clip_speech, runlog.Log.from_env())

    result = planner.build_plan(
        meta,
        speech,
        {p: edit_files[p]["edits"] for p in edit_files},
        words,
        params,
        loud=loud,
    )
    _write_json(args.out, result)
    report = planner.format_report(result)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            handle.write(report)
    print(report, end="")

    hard = result.get("blocking") or []
    if hard and not args.force:
        raise SystemExit(
            "refusing to continue: " + "; ".join(hard) + " (override with --force)"
        )


def cmd_stage_plan(args):
    """The plan stage, end to end: params, plan, report, filtergraphs."""
    log = runlog.Log.from_env()
    settings = cfg.from_environment()
    try:
        pipeline.stage_plan(args.work, settings, log, force=args.force)
    except pipeline.StageError as exc:
        log.error(str(exc))
        raise SystemExit(1) from None


def cmd_stage_prepare(args):
    """The prepare stage: decode every track, then measure what came out."""
    log = runlog.Log.from_env()
    settings = cfg.from_environment()
    try:
        pipeline.stage_prepare(args.work, settings, log, ffmpeg=args.ffmpeg)
    except pipeline.StageError as exc:
        log.error(str(exc))
        raise SystemExit(1) from None


def cmd_stage_transcribe(args):
    """The transcribe stage: level scan and transcription, per track."""
    log = runlog.Log.from_env()
    settings = cfg.from_environment()
    try:
        pipeline.stage_transcribe(
            args.work, settings, log, ffmpeg=args.ffmpeg,
            api_key=_api_key(WHISPER_KEY_ENV),
        )
    except pipeline.StageError as exc:
        log.error(str(exc))
        raise SystemExit(1) from None


# --- filters ------------------------------------------------------------------
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

    p = sub.add_parser("meta-shell", help="meta.json as shell assignments")
    p.add_argument("--meta", required=True)
    p.set_defaults(func=cmd_meta_shell)

    p = sub.add_parser(
        "config", help="resolve settings and print them as shell assignments")
    p.add_argument("--config", default="", help="config file; searched if absent")
    p.add_argument("--script-root", required=True,
                   help="fallback PODCAST_ROOT: the directory holding the script")
    p.add_argument("--set", action="append", metavar="NAME=VALUE",
                   help="override one setting; repeatable, beats env and file")
    p.add_argument("--no-keys", action="store_true",
                   help="skip reading the API key files")
    p.set_defaults(func=cmd_config)

    p = sub.add_parser(
        "config-names", help="the setting names, for the launcher to export")
    p.set_defaults(func=cmd_config_names)

    p = sub.add_parser("discover", help="identify one episode's tracks")
    p.add_argument("--input-dir", default="")
    p.add_argument("--file", action="append", default=[],
                   help="explicit track path; repeatable, skips the scan")
    p.add_argument("--exts", default="flac")
    p.add_argument("--separator", default="_")
    p.add_argument("--episode", default="", help="override the parsed episode id")
    p.add_argument("--work-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--ffprobe", default="ffprobe")
    p.add_argument("--resample-to", default="")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_discover)

    p = sub.add_parser("stage-prepare", help="run the prepare stage")
    p.add_argument("--work", required=True)
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.set_defaults(func=cmd_stage_prepare)

    p = sub.add_parser("stage-transcribe", help="run the transcribe stage")
    p.add_argument("--work", required=True)
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.set_defaults(func=cmd_stage_transcribe)

    p = sub.add_parser("stage-plan", help="run the plan stage")
    p.add_argument("--work", required=True, help="the episode work directory")
    p.add_argument("--force", action="store_true",
                   help="proceed even when the plan trips a safety limit")
    p.set_defaults(func=cmd_stage_plan)

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
    p.add_argument("--words-dir", required=True)
    p.add_argument(
        "--loud-dir",
        help="loud-spans output, to cross-check each transcript against its audio",
    )
    p.add_argument("--edits-dir")
    p.add_argument(
        "--no-clip-speech", dest="clip_speech", action="store_false",
        help="build the speech map from the word timings alone, however far "
             "past the audio they run",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--report")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_plan)

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
