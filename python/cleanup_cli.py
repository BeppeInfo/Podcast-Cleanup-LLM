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
from cleanup import llm, plan as planner, render, transcript as tr, vad  # noqa: E402


def _read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path, payload):
    render.write_json(path, payload)


def _progress(done, total):
    print(f"PROGRESS {done} {total}", flush=True)


def _fail(message):
    raise SystemExit(f"error: {message}")


# --- meta ---------------------------------------------------------------------


def _probe(ffprobe, path):
    result = subprocess.run(
        [
            ffprobe, "-v", "error", "-select_streams", "a:0",
            "-show_entries",
            "stream=sample_rate,channels,sample_fmt,bits_per_raw_sample",
            "-show_entries", "format=duration",
            "-of", "json", path,
        ],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        _fail(f"ffprobe failed on {path}: {result.stderr.strip()}")
    data = json.loads(result.stdout)
    streams = data.get("streams") or []
    if not streams:
        _fail(f"{path}: no audio stream")
    stream = streams[0]
    duration = float((data.get("format") or {}).get("duration") or 0.0)
    if duration <= 0:
        _fail(f"{path}: could not determine duration")
    return {
        "duration": round(duration, 3),
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
    rates = {t["sample_rate"] for t in tracks}
    if len(rates) > 1:
        detail = ", ".join(f"{t['participant']}={t['sample_rate']}Hz" for t in tracks)
        _fail(
            "all tracks of an episode must share a sample rate so that cuts land "
            f"on identical samples on every track ({detail}). Resample the odd one "
            "out before running."
        )

    durations = [t["duration"] for t in tracks]
    spread = max(durations) - min(durations)
    meta = {
        "episode_id": args.episode,
        "duration": max(durations),
        "duration_spread": round(spread, 3),
        "sample_rate": rates.pop(),
        "tracks": tracks,
    }
    _write_json(args.out, meta)
    print(
        f"{len(tracks)} tracks, {meta['sample_rate']} Hz, "
        f"{meta['duration']:.1f}s (spread {spread:.3f}s)"
    )
    if spread > 1.0:
        print(
            f"warning: track lengths differ by {spread:.1f}s — if they are not "
            "aligned at sample 0 the cleanup will drift"
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
    ):
        pairs = " ".join(
            f"[{shlex.quote(t['participant'])}]={shlex.quote(str(t[field]))}"
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
    speech = vad.silero_speech(args.wav, threshold=args.threshold)
    duration = args.duration or vad.wav_duration(args.wav)
    _write_json(
        args.out,
        {
            "participant": args.participant,
            "backend": "silero",
            "duration": round(duration, 3),
            "speech": [[round(s, 4), round(e, 4)] for s, e in speech],
        },
    )
    print(f"{len(speech)} speech regions, {iv.total(speech):.1f}s of speech")


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


def cmd_llm_wait(args):
    client = llm.LlamaClient(args.endpoint)
    if not client.wait_until_ready(args.timeout):
        raise SystemExit(1)
    print("llama endpoint ready")


def cmd_detect(args):
    parsed = _read_json(args.words)
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    unknown = [k for k in kinds if k not in llm.KINDS]
    if unknown:
        _fail(f"unknown edit kinds: {', '.join(unknown)} (known: {', '.join(llm.KINDS)})")

    client = llm.LlamaClient(
        args.endpoint, timeout=args.request_timeout, temperature=args.temperature
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
    )
    _write_json(args.out, result)
    print(
        f"{len(result['edits'])} edits accepted, {result['rejected_count']} rejected "
        f"over {result['chunks']} chunks"
    )
    if result["chunk_failures"]:
        print(
            f"warning: {result['chunk_failures']} of {result['chunks']} chunks "
            "produced no usable response; those stretches were left untouched"
        )


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
        graph = render.build_filter(
            current["cuts"], mutes, args.frame_samples, args.fade
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
            track["sample_rate"],
            args.frame_samples,
            current["cuts"],
        )
        expectations[participant] = {
            "filter": path if graph is not None else None,
            "passthrough": graph is None,
            "mutes": len(mutes),
            "expected_samples": samples,
            "expected_duration": round(samples / float(track["sample_rate"]), 3),
            "sample_rate": track["sample_rate"],
            "sample_fmt": track["sample_fmt"],
        }

    _write_json(args.out, {"frame_samples": args.frame_samples, "tracks": expectations})
    for participant, values in expectations.items():
        kind = "passthrough" if values["passthrough"] else f"{values['mutes']} mutes"
        print(
            f"{participant}: {values['expected_duration']:.3f}s expected ({kind})"
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
    p.add_argument("track", nargs="+", help="participant=path")
    p.set_defaults(func=cmd_meta)

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

    p = sub.add_parser("llm-wait", help="block until the llama endpoint is ready")
    p.add_argument("--endpoint", required=True)
    p.add_argument("--timeout", type=float, default=600.0)
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
    p.add_argument("--kinds", default="stutter,repetition,false_start")
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
    except BrokenPipeError:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
