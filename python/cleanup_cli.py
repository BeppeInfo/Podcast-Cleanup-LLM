#!/usr/bin/env python3
"""Entry point for the podcast cleanup pipeline.

`run` is the whole pipeline; the rest are single steps that read and write files
in the episode work directory, so any stage can be re-run by hand.

stdout belongs to the subcommands that emit data for the launcher to `eval` or
read — `config`, `config-names`, `meta-shell`. Nothing logs to it.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cleanup import intervals as iv  # noqa: E402
from cleanup import config as cfg, discover, llm, plan as planner  # noqa: E402
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
LLAMA_KEY_ENV = "PODCAST_LLAMA_API_KEY"


def _api_key(name):
    return os.environ.get(name) or None


def _fail(message):
    raise SystemExit(f"error: {message}")


def cmd_list_stages(args):
    for stage in pipeline.ALL_STAGES:
        print(f"  {stage}")


def cmd_run(args):
    """Run the pipeline. The one place that sequences the stages."""
    log = runlog.Log.from_env()
    if not log.path:
        # Nobody has named a log, so this run opens one somewhere temporary.
        # discover adopts it into the work directory once the episode is known,
        # which is why it cannot simply be written there to begin with.
        handle, path = tempfile.mkstemp(prefix="podcast-cleanup-", suffix=".log")
        os.close(handle)
        log.init(path, path)
    log.debug(f"python: {sys.executable}")
    settings = cfg.from_environment()
    try:
        stages = pipeline.select_stages(args.from_stage, args.to_stage, args.stages)
    except pipeline.StageError as exc:
        log.error(str(exc))
        raise SystemExit(1) from None

    log.line("")
    log.line(f"{log.c.bold}Podcast cleanup{log.c.reset}"
             f"{log.c.dim} — stages: {' '.join(stages)}{log.c.reset}")
    log.info(f"Whisper: whisperx {settings['WHISPER_MODEL']} on "
             f"{settings['WHISPER_DEVICE']}   LLM: {settings['LLAMA_ENDPOINT']}")
    if args.dry_run:
        log.warn("dry run: no files will be written or removed")

    raise SystemExit(pipeline.run_episode(
        settings, log, stages,
        episode_override=args.episode,
        input_files=args.file or (),
        dry_run=args.dry_run,
        force=args.force,
        program=args.program,
        api_keys={"llama": _api_key(LLAMA_KEY_ENV)},
    ))


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


def cmd_config_names(args):
    """Every setting name, one per line, for the launcher to export."""
    for name in cfg.SETTINGS:
        print(name)


# --- detection ----------------------------------------------------------------


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


def cmd_verify(args):
    """Check a finished directory's durations against the plan, by hand.

    Shares the comparison with the render stage rather than repeating it; what
    is different here is only where the durations come from.
    """
    log = runlog.Log.from_env()
    actual = {}
    for item in args.actual:
        if "=" not in item:
            _fail(f"actual argument must be participant=duration, got '{item}'")
        participant, value = item.split("=", 1)
        actual[participant] = float(value)
    try:
        pipeline.verify_durations(
            _read_json(args.expected)["tracks"], actual, args.tolerance, log)
    except pipeline.StageError as exc:
        _fail(str(exc))


# --- argument parsing ---------------------------------------------------------


def build_parser():
    parser = argparse.ArgumentParser(prog="cleanup_cli", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list-stages", help="the stage names, in order")
    p.set_defaults(func=cmd_list_stages)

    p = sub.add_parser("run", help="run the pipeline")
    p.add_argument("--from", dest="from_stage", default="")
    p.add_argument("--to", dest="to_stage", default="")
    p.add_argument("--stages", default="", help="exactly these, comma separated")
    p.add_argument("--episode", default="")
    p.add_argument("--file", action="append", default=[],
                   help="explicit input track; repeatable, skips the scan")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--program", default="",
                   help="how this run was invoked, for the resume hint")
    p.set_defaults(func=cmd_run)

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
