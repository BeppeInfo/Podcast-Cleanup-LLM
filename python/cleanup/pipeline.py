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
import shutil
import subprocess
import time

from . import intervals as iv
from . import plan as planner
from . import asr, discover as disco, llm, proc, render
from . import silence, transcript as tr


class StageError(Exception):
    """The stage could not finish. The message is for the operator."""


def _probe_fail(message):
    raise StageError(message)


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
        _probe_fail(f"ffprobe failed on {path}: {result.stderr.strip()}")
    data = json.loads(result.stdout)
    streams = data.get("streams") or []
    if not streams:
        _probe_fail(f"{path}: no audio stream ffmpeg can see")
    stream = streams[0]
    duration = float((data.get("format") or {}).get("duration") or 0.0)
    if duration <= 0:
        _probe_fail(f"{path}: could not determine duration")
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
            _probe_fail(f"track argument must be participant=path, got '{item}'")
        participant, path = item.split("=", 1)
        if not os.path.isfile(path):
            _probe_fail(f"track file not found: {path}")
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
        _probe_fail(
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
ALL_STAGES = ("discover", "prepare", "transcribe", "detect", "plan",
              "render", "finalize")


class NothingToDo(Exception):
    """An empty inbox. Not a failure; the run reports it and stops."""


class Episode:
    """The paths and tracks every stage needs, in one place.

    Built by `discover`, or rebuilt from meta.json by a resumed run. Nothing
    here is derived twice: `resume` reads the same meta.json `discover` wrote,
    which is what makes `--from` work without re-scanning the input directory.
    """

    def __init__(self, episode_id: str, settings):
        self.id = episode_id
        where = disco.episode_paths(
            episode_id, settings["WORK_ROOT"], settings["OUTPUT_DIR"])
        self.work = where["work"]
        self.state = where["state"]
        self.output = where["output"]
        self.staging = where["staging"]
        self.tracks: dict[str, str] = {}

    @property
    def log_path(self) -> str:
        return os.path.join(self.work, "logs", "run.log")

    def load_tracks(self) -> dict:
        meta = read_json(os.path.join(self.work, "meta.json"))
        self.tracks = {t["participant"]: t["source"] for t in meta["tracks"]}
        return meta


def select_stages(from_stage: str = "", to_stage: str = "",
                  explicit: str = "") -> list[str]:
    """Which stages this run executes, in order."""
    if explicit:
        chosen = [name.strip() for name in explicit.split(",") if name.strip()]
        unknown = [name for name in chosen if name not in ALL_STAGES]
        if unknown:
            raise StageError(f"unknown stage '{unknown[0]}' "
                             f"(known: {' '.join(ALL_STAGES)})")
        return chosen

    for name in (from_stage, to_stage):
        if name and name not in ALL_STAGES:
            raise StageError(f"unknown stage '{name}' "
                             f"(known: {' '.join(ALL_STAGES)})")
    first = ALL_STAGES.index(from_stage) if from_stage else 0
    last = ALL_STAGES.index(to_stage) if to_stage else len(ALL_STAGES) - 1
    if first > last:
        raise StageError(f"--from {from_stage} comes after --to {to_stage}")
    return list(ALL_STAGES[first:last + 1])


def state_done(episode, stage: str) -> bool:
    return os.path.isfile(os.path.join(episode.state, f"{stage}.done"))


def state_mark(episode, stage: str, dry_run: bool) -> None:
    if not dry_run:
        open(os.path.join(episode.state, f"{stage}.done"), "w").close()


def resume_episode(settings, log, episode_id: str) -> Episode:
    """Re-establish what a run needs without re-scanning the input directory."""
    if not episode_id:
        raise StageError("--from requires --episode (or an input dir to scan)")
    episode = Episode(episode_id, settings)
    if not os.path.isdir(episode.work):
        raise StageError(f"no work directory for episode '{episode_id}' "
                         f"at {episode.work}")
    disco.make_work_tree({"work": episode.work, "output": episode.output})
    log.path = episode.log_path
    episode.load_tracks()
    return episode


def resume_command(program: str, episode_id: str) -> str:
    """What to type to pick this run up again. Composed here because the
    episode id is not known until discovery, and the launcher runs before it."""
    if not program:
        return ""
    return f"{program} --episode {episode_id} --from <stage>"


def report_failure(episode, settings, log, program: str, dry_run: bool) -> None:
    """Keep everything, say where it is, and leave a marker saying why.

    A failed run must be resumable, so nothing is cleaned up here. The inputs
    can optionally be parked instead, so an unattended run does not retry the
    same broken episode forever.
    """
    if episode is None or dry_run or not os.path.isdir(episode.work):
        return
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    try:
        with open(os.path.join(episode.work, "FAILED"), "w",
                  encoding="utf-8") as handle:
            handle.write(f"failed at {stamp}\n")
    except OSError:
        pass

    log.line(f"{log.c.dim}      work directory kept: {episode.work}{log.c.reset}")
    hint = resume_command(program, episode.id)
    if hint:
        log.line(f"{log.c.dim}      resume with: {hint}{log.c.reset}")

    failed_dir = settings["FAILED_DIR"]
    os.makedirs(failed_dir, exist_ok=True)
    if log.path and os.path.isfile(log.path):
        copied = os.path.join(failed_dir, f"{episode.id}.log")
        try:
            shutil.copyfile(log.path, copied)
            log.line(f"{log.c.dim}      log copied to {copied}{log.c.reset}")
        except OSError:
            pass

    if settings.get("FAILED_ACTION", "log") == "move" and episode.tracks:
        target = os.path.join(failed_dir, episode.id)
        os.makedirs(target, exist_ok=True)
        for source in episode.tracks.values():
            if os.path.isfile(source):
                try:
                    shutil.move(source, target)
                except OSError:
                    pass
        log.line(f"{log.c.dim}      inputs moved to {target}{log.c.reset}")
    else:
        log.line(f"{log.c.dim}      inputs left untouched{log.c.reset}")


def run_episode(settings, log, stages, *, episode_override: str = "",
                input_files=(), dry_run: bool = False, force: bool = False,
                ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe",
                program: str = "", api_keys=None) -> int:
    """Run the selected stages in order. Returns a process exit status.

    The one place that knows the sequence, so both the CLI and anything else
    driving the pipeline get the same ordering, the same resume behaviour and the
    same failure handling.
    """
    keys = api_keys or {}
    log.stage_total(len(stages))
    episode = None

    try:
        if stages[0] != "discover":
            episode = resume_episode(settings, log, episode_override)

        for stage in stages:
            if stage == "discover":
                log.stage_begin("discover", "locating the episode and its tracks")
                episode = discover_episode(
                    settings, log, input_files, episode_override, dry_run, ffprobe)
                state_mark(episode, "discover", dry_run)
                log.stage_end(f"{len(episode.tracks)} tracks")
                continue

            if stage == "prepare":
                log.stage_begin("prepare", "decoding tracks to 16 kHz mono")
                if dry_run:
                    log.info(f"would decode {len(episode.tracks)} tracks")
                else:
                    stage_prepare(episode.work, settings, log, ffmpeg=ffmpeg)
                    episode.load_tracks()
                log.stage_end(f"{len(episode.tracks)} tracks decoded")

            elif stage == "transcribe":
                log.stage_begin(
                    "transcribe", f"transcribing via {settings['WHISPER_ENDPOINT']}")
                if dry_run:
                    log.info(f"would transcribe {len(episode.tracks)} tracks")
                else:
                    stage_transcribe(episode.work, settings, log, ffmpeg=ffmpeg,
                                     api_key=keys.get("whisper"))
                log.stage_end(f"{len(episode.tracks)} tracks transcribed remotely")

            elif stage == "detect":
                if settings["LLM_ENABLE"] != "1":
                    log.stage_skip("detect", "LLM_ENABLE=0")
                    state_mark(episode, "detect", dry_run)
                    continue
                log.stage_begin("detect", "finding stutters and false starts")
                if not state_done(episode, "transcribe"):
                    log.warn("transcribe stage has not completed in this work dir")
                if dry_run:
                    log.info(f"would analyse {len(episode.tracks)} transcripts")
                else:
                    stage_detect(episode.work, settings, log,
                                 api_key=keys.get("llama"),
                                 resume_hint=resume_command(program, episode.id))
                log.stage_end(f"{len(episode.tracks)} tracks analysed")

            elif stage == "plan":
                log.stage_begin("plan", "deciding what to cut and what to mute")
                if dry_run:
                    log.info(f"would unify {episode.work}/words and llm into a plan")
                else:
                    stage_plan(episode.work, settings, log, force=force)
                log.stage_end("plan written")

            elif stage == "render":
                log.stage_begin("render", "rendering cleaned tracks")
                if dry_run:
                    log.info(f"would render into {episode.staging}")
                else:
                    stage_render(episode.work, episode.staging, settings, log,
                                 ffmpeg=ffmpeg, ffprobe=ffprobe)
                log.stage_end(f"{len(episode.tracks)} tracks rendered and verified")

            elif stage == "finalize":
                log.stage_begin("finalize", "publishing outputs and cleaning up")
                if dry_run:
                    log.info(f"would publish into {episode.output}")
                else:
                    stage_finalize(episode.work, episode.output, episode.staging,
                                   settings, log, episode.id, episode.tracks)
                log.stage_end(f"outputs in {episode.output}")

            # Not finalize: it has just removed the work directory that the
            # marker would live in, and a stage that deletes its own state has
            # nothing to record. Nothing resumes past it either.
            if stage not in ("discover", "finalize"):
                state_mark(episode, stage, dry_run)

    except NothingToDo as exc:
        log.warn(f"{exc} — nothing to do")
        log.line("")
        log.line("Nothing to do.")
        return 0
    except StageError as exc:
        log.line("")
        log.error(str(exc))
        report_failure(episode, settings, log, program, dry_run)
        log.line("")
        return 1

    log.line("")
    log.line(f"{log.c.green}✓{log.c.reset} {log.c.bold}finished {episode.id} "
             f"in {log.elapsed()}{log.c.reset}")
    if os.path.isdir(episode.output):
        log.line(f"  {log.c.dim}outputs: {episode.output}{log.c.reset}")
    log.line("")
    return 0


def discover_episode(settings, log, input_files, episode_override: str,
                     dry_run: bool, ffprobe: str) -> Episode:
    """Identify the episode, build its work tree, and adopt the run log."""
    try:
        paths = (list(input_files) if input_files
                 else disco.find_tracks(settings["INPUT_DIR"],
                                        settings["INPUT_EXTS"].split()))
        episode_id, tracks = disco.parse_tracks(
            paths, settings["TRACK_SEPARATOR"], episode_override)
    except disco.NothingToDo as exc:
        raise NothingToDo(str(exc)) from None
    except disco.DiscoverError as exc:
        raise StageError(str(exc)) from None

    episode = Episode(episode_id, settings)
    episode.tracks = tracks
    if not dry_run:
        disco.make_work_tree({"work": episode.work, "output": episode.output})
        adopt_log(log, episode)
        meta = build_meta(
            [f"{name}={path}" for name, path in sorted(tracks.items())],
            episode_id, ffprobe, settings["RESAMPLE_TO"],
            say=lambda line: (log.warn(line.split(": ", 1)[-1])
                              if line.startswith(("warning:", "note:"))
                              else log.info(line)))
        render.write_json(os.path.join(episode.work, "meta.json"), meta)

    log.line("")
    log.line(f"  {log.c.bold}Episode{log.c.reset} {episode_id}  "
             f"{log.c.dim}({len(tracks)} tracks){log.c.reset}")
    for participant in sorted(tracks):
        log.line(f"    {log.c.cyan}{participant}{log.c.reset} "
                 f"{log.c.dim}← {os.path.basename(tracks[participant])}"
                 f"{log.c.reset}")
    log.line("")

    if not dry_run:
        episode.load_tracks()
    elif os.path.isfile(os.path.join(episode.work, "meta.json")):
        episode.load_tracks()
        log.info("reusing the meta.json from a previous run")
    else:
        log.warn("no meta.json yet, so later stages can only be listed")
    return episode


def adopt_log(log, episode) -> None:
    """Move the run log into the episode directory, carrying over what it holds.

    Until the episode is known the log is somewhere temporary. Starting a fresh
    one here would lose the config dump that precedes discovery.
    """
    target = episode.log_path
    if log.path == target:
        return
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if log.path and os.path.isfile(log.path):
        with open(log.path, encoding="utf-8", errors="replace") as source, \
                open(target, "a", encoding="utf-8") as sink:
            sink.write(source.read())
        os.remove(log.path)
    log.path = target
    log.raw(f"=== log adopted into {episode.work} ===")


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


def describe_edits(result, settings, log) -> None:
    """What the detection accepted, and the two ways it can fail quietly."""
    log.raw(f"{len(result['edits'])} edits accepted, {result['rejected_count']} "
            f"rejected over {result['chunks']} chunks")

    if result.get("chunks_truncated"):
        log.warn(f"{result['chunks_truncated']} of {result['chunks']} chunks ran "
                 "past LLM_MAX_REPLY_TOKENS; the edits they had already emitted "
                 "were kept, but the tail of those windows went unjudged. Raise "
                 "it, or lower LLM_CHUNK_WORDS so a window yields fewer edits.")

    if result["chunk_failures"]:
        failed, total = result["chunk_failures"], result["chunks"]
        log.warn(f"{failed} of {total} chunks produced no usable response; "
                 "those stretches were left untouched")


def track_was_analysed(result, log) -> bool:
    """Whether this track was analysed at all, rather than analysed and clean.

    Every chunk failing is not per-item tolerance doing its job: it is the whole
    track keeping its disfluencies while the report shows a clean-looking zero.
    DESIGN.md §9 invariant 10 — a fault that hits every chunk identically must
    not be able to pass for the survivable kind. Saying no here is what stops the
    marker being written, so a resumed run retries the track instead of skipping
    it as complete.
    """
    failed, total = result["chunk_failures"], result["chunks"]
    if total > 0 and failed == total:
        log.warn(f"every chunk failed for {result['participant']}: this track was "
                 "not analysed at all. If they timed out, the endpoint is too "
                 "slow for LLM_CONCURRENCY windows at once — lower it, or raise "
                 "LLAMA_REQUEST_TIMEOUT.")
        return False
    return True


def stage_detect(work: str, settings, log, api_key=None, resume_hint: str = "",
                 ready_timeout: float = 60.0) -> None:
    """Ask the model which stretches of each transcript are disfluencies.

    Three faults end the run rather than the track, because each is a property
    of the server: a refused key, a server that ignores the JSON schema, and a
    server with no model loaded. Every remaining track would fail identically,
    and an episode that quietly found no edits at all looks like clean speech.
    """
    meta = read_json(os.path.join(work, "meta.json"))
    state = os.path.join(work, "state")
    endpoint = settings["LLAMA_ENDPOINT"]

    kinds = [k.strip() for k in settings["LLM_ACCEPT_KINDS"].split(",") if k.strip()]
    unknown = [k for k in kinds if k not in llm.KINDS]
    if unknown:
        raise StageError(f"unknown edit kinds: {', '.join(unknown)} "
                         f"(known: {', '.join(llm.KINDS)})")

    client = llm.LlamaClient(
        endpoint, timeout=float(settings["LLAMA_REQUEST_TIMEOUT"]),
        temperature=float(settings["LLM_TEMP"]), api_key=api_key,
        api=settings["LLM_API"],
        max_reply_tokens=int(settings["LLM_MAX_REPLY_TOKENS"]),
        model=settings["LLAMA_MODEL_NAME"],
    )
    if not client.wait_until_ready(ready_timeout):
        raise StageError(f"the llama endpoint at {endpoint} is not usable")

    resume = f" Then resume with: {resume_hint}" if resume_hint else ""
    try:
        # One tiny constrained request before the episode starts. A server that
        # answers /health but ignores the schema would otherwise be discovered
        # only after every chunk of every track had been dropped.
        if settings["LLM_CHECK_SCHEMA"] == "1":
            client.check_schema_support()
    except llm.SchemaIgnored as exc:
        raise StageError(f"{exc} Try LLM_API=completion.{resume}") from None
    except llm.AuthRejected as exc:
        raise StageError(f"{exc} Fix the key.{resume}") from None
    except llm.ModelUnavailable as exc:
        raise StageError(f"{exc}{resume}") from None

    limits = {
        "max_words": int(settings["LLM_MAX_EDIT_WORDS"]),
        "max_seconds": float(settings["LLM_MAX_EDIT_SECONDS"]),
        "min_confidence": float(settings["LLM_MIN_CONFIDENCE"]),
    }
    total = len(meta["tracks"])
    failed = 0

    for index, track in enumerate(meta["tracks"], start=1):
        participant = track["participant"]
        words_path = os.path.join(work, "words", f"{participant}.words.json")
        target = os.path.join(work, "llm", f"{participant}.edits.json")
        marker = os.path.join(state, f"llm-{participant}.ok")

        if not (os.path.isfile(words_path) and os.path.getsize(words_path) > 0):
            log.warn(f"no transcript for {participant}; skipping edit detection")
            continue
        if (os.path.isfile(marker) and os.path.isfile(target)
                and os.path.getsize(target) > 0):
            log.debug(f"{participant} already analysed by the LLM")
            continue

        log.info(f"llm: {participant} ({index}/{total})")
        try:
            result = llm.detect(
                client, read_json(words_path),
                chunk_words=int(settings["LLM_CHUNK_WORDS"]),
                overlap=int(settings["LLM_CHUNK_OVERLAP"]),
                limits=limits,
                accepted=kinds,
                audit_path=os.path.join(work, "llm", f"{participant}.audit.jsonl"),
                on_progress=lambda done, count, name=participant: log.progress(
                    done, count, name),
                concurrency=int(settings["LLM_CONCURRENCY"]),
            )
        except llm.AuthRejected:
            raise StageError("the LLM endpoint refused our credentials. Fix the "
                             f"key.{resume}") from None
        except llm.SchemaIgnored:
            raise StageError("the LLM endpoint ignored the JSON schema. Try "
                             f"LLM_API=completion.{resume}") from None
        except llm.ModelUnavailable:
            named = (f" for '{settings['LLAMA_MODEL_NAME']}'"
                     if settings["LLAMA_MODEL_NAME"] else "")
            raise StageError(f"the LLM endpoint has no model loaded{named}. "
                             f"Load it.{resume}") from None
        finally:
            log.progress_done()

        render.write_json(target, result)
        describe_edits(result, settings, log)
        if track_was_analysed(result, log):
            open(marker, "w").close()
        else:
            failed += 1
            log.warn(f"edit detection failed for {participant}; that track keeps "
                     "its disfluencies")

    if total > 0 and failed == total:
        raise StageError("edit detection failed for every track")
    log.raw(f"{total - failed}/{total} tracks analysed")


# Encoders with a bit depth worth preserving. For anything else the source's
# sample format is not a property the output can carry.
DEPTH_AWARE_CODECS = ("flac", "alac", "wavpack")


def encode_args(settings, track) -> list[str]:
    """The codec half of an ffmpeg command line, for one track."""
    codec = settings["OUTPUT_CODEC"]
    args = ["-c:a", codec]
    if codec == "flac":
        args += ["-compression_level", str(settings["OUTPUT_COMPRESSION"])]
    # Deliberately split rather than quoted: a user-supplied argument string.
    if settings["OUTPUT_EXTRA_ARGS"].strip():
        args += settings["OUTPUT_EXTRA_ARGS"].split()
    fmt = track.get("sample_fmt") or ""
    if (codec in DEPTH_AWARE_CODECS or codec.startswith("pcm_")) \
            and fmt in ("s16", "s32"):
        args += ["-sample_fmt", fmt]
    return args


def probe_duration(ffprobe: str, path: str) -> float:
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path],
        capture_output=True, text=True, check=False,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        raise StageError(f"could not measure the rendered file: {path}") from None


def verify_durations(expectations, actual, tolerance: float, log) -> None:
    """Compare what was rendered against the frame-exact prediction.

    Not a rule of thumb: expected.json says how many samples each track should
    contain, computed from the same frame size the filtergraph used. A mismatch
    means the cuts that were rendered are not the cuts that were planned, so the
    run stops with everything kept.

    Takes the expectations rather than a path, so the `verify` subcommand can
    check a finished directory by hand against the same rules.
    """
    problems = []
    for participant, measured in sorted(actual.items()):
        if participant not in expectations:
            problems.append(f"{participant}: rendered but not in the plan")
            continue
        expected = expectations[participant]["expected_duration"]
        allowed = max(tolerance * expected, 0.05)
        delta = abs(measured - expected)
        ok = delta <= allowed
        log.report(f"{participant:<16} expected {expected:9.3f}s  "
                   f"actual {measured:9.3f}s  delta {delta:6.3f}s  "
                   f"{'ok' if ok else 'MISMATCH'}")
        if not ok:
            problems.append(
                f"{participant}: expected {expected:.3f}s but rendered "
                f"{measured:.3f}s (delta {delta:.3f}s, allowed {allowed:.3f}s)")

    for participant in expectations:
        if participant not in actual:
            problems.append(f"{participant}: planned but never rendered")

    if problems:
        for problem in problems:
            log.error(problem)
        raise StageError("rendered durations do not match the plan; inputs and "
                         "work directory kept")
    log.report("all tracks verified")


def stage_render(work: str, staging: str, settings, log, ffmpeg: str = "ffmpeg",
                 ffprobe: str = "ffprobe") -> None:
    """Render every track through its filtergraph, then check the lengths."""
    plan_path = os.path.join(work, "plan.json")
    if not (os.path.isfile(plan_path) and os.path.getsize(plan_path) > 0):
        raise StageError("no plan.json; run the plan stage first")

    meta = read_json(os.path.join(work, "meta.json"))
    state = os.path.join(work, "state")
    os.makedirs(staging, exist_ok=True)
    jobs = max(1, int(settings["FFMPEG_JOBS"]))
    suffix, extension = settings["OUTPUT_SUFFIX"], settings["OUTPUT_EXT"]

    jobs_to_run = []
    for track in meta["tracks"]:
        participant = track["participant"]
        target = os.path.join(staging, f"{participant}{suffix}.{extension}")
        marker = os.path.join(state, f"render-{participant}.ok")
        filter_path = os.path.join(work, "render", f"{participant}.filter")
        if os.path.exists(marker):
            os.remove(marker)
        jobs_to_run.append((track, target, marker, filter_path))

    def render_one(item) -> tuple[str, int]:
        track, target, marker, filter_path = item
        participant = track["participant"]
        source = track["source"]
        encode = encode_args(settings, track)

        if not os.path.isfile(filter_path):
            # No edits and no resampling. Copying beats re-encoding, but only
            # when the file is already in the format being asked for.
            same = (track.get("codec") == settings["OUTPUT_CODEC"]
                    and source.rpartition(".")[2] == extension)
            if same:
                log.info(f"{participant} needs no edits and is already "
                         f"{settings['OUTPUT_CODEC']}, copying it through")
                try:
                    shutil.copyfile(source, target)
                except OSError as exc:
                    log.error(f"could not copy {participant}: {exc}")
                    return participant, 1
                open(marker, "w").close()
                return participant, 0

            log.info(f"{participant} needs no edits, converting to "
                     f"{settings['OUTPUT_CODEC']}")
            argv = [ffmpeg, "-nostdin", "-y", "-v", "warning", "-i", source,
                    "-map", "0:a:0", *encode, target]
            status = proc.run(argv, log)
            if status == 0:
                open(marker, "w").close()
            return participant, status

        argv = [ffmpeg, "-nostdin", "-y", "-v", "warning", "-progress", "pipe:1",
                "-nostats", "-i", source,
                "-filter_complex_script", filter_path, "-map", "[out]",
                *encode, target]
        on_line = None
        if jobs <= 1:
            on_line = proc.ffmpeg_progress(
                log, float(track["duration"]) * 1_000_000, f"rendering {participant}")
        status = proc.run(argv, log, on_line=on_line)
        if status == 0:
            open(marker, "w").close()
            log.ok(f"rendered {participant}")
        return participant, status

    if jobs <= 1 or len(jobs_to_run) <= 1:
        results = [render_one(item) for item in jobs_to_run]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(render_one, jobs_to_run))

    failed = [name for name, status in results if status != 0]
    if failed:
        raise StageError(f"rendering failed for: {', '.join(sorted(failed))}")

    actual = {}
    for track in meta["tracks"]:
        participant = track["participant"]
        target = os.path.join(staging, f"{participant}{suffix}.{extension}")
        if not (os.path.isfile(target) and os.path.getsize(target) > 0):
            raise StageError(f"rendered file is missing or empty: {target}")
        actual[participant] = probe_duration(ffprobe, target)

    verify_durations(read_json(os.path.join(work, "expected.json"))["tracks"],
                     actual, float(settings["DURATION_TOLERANCE"]), log)


def human_size(path: str) -> str:
    """`du -h`-ish, for the one line that says what was published."""
    size = float(os.path.getsize(path))
    for unit in ("", "K", "M", "G"):
        if size < 1024 or unit == "G":
            return f"{size:.0f}{unit}" if unit in ("", "K") else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}G"


def build_transcript(work: str, episode_id: str, log) -> None:
    """The final speaker transcript, as JSON, SRT and plain text."""
    current = read_json(os.path.join(work, "plan.json"))
    words = collect(os.path.join(work, "words"), ".words.json")
    result = render.build_transcript(current, words)
    render.write_json(
        os.path.join(work, f"{episode_id}_transcript.json"), result)
    for name, body in (
        (f"{episode_id}_transcript.srt", render.transcript_to_srt(result)),
        (f"{episode_id}_transcript.txt", render.transcript_to_text(result)),
    ):
        with open(os.path.join(work, name), "w", encoding="utf-8") as handle:
            handle.write(body)
    log.raw(f"{len(result['segments'])} segments, {result['removed_words']} "
            "words removed by the edit")


# Sidecars copied beside the audio, each prefixed with the episode id so an
# output directory stays readable when several episodes share a parent.
SIDECARS = (
    "{episode}_transcript.json",
    "{episode}_transcript.srt",
    "{episode}_transcript.txt",
    ("plan.json", "{episode}_plan.json"),
    ("edit-report.txt", "{episode}_edit-report.txt"),
)


def stage_finalize(work: str, out_dir: str, staging: str, settings, log,
                   episode_id: str, sources) -> str:
    """Publish the outputs, then remove what is no longer needed.

    The order is the whole point and is not an accident of how it was written:
    every output is on disk before anything is deleted, and the inputs go only
    after the audio has been moved into place and the sidecars copied. A run that
    dies halfway leaves the originals and the work directory alone, which is what
    makes it safe to resume.

    Returns the run log's path, which moves when the work directory goes.
    """
    build_transcript(work, episode_id, log)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)

    suffix, extension = settings["OUTPUT_SUFFIX"], settings["OUTPUT_EXT"]
    for participant in sorted(sources):
        name = f"{participant}{suffix}.{extension}"
        staged = os.path.join(staging, name)
        final = os.path.join(out_dir, name)
        try:
            # Staging sits inside the output directory, so this is a rename on
            # one filesystem rather than a copy.
            os.replace(staged, final)
        except OSError as exc:
            raise StageError(f"could not publish {final}: {exc}") from None
        log.ok(f"{name} ({human_size(final)})")
    try:
        os.rmdir(staging)
    except OSError:
        pass

    for entry in SIDECARS:
        source_name, target_name = (
            (entry, entry) if isinstance(entry, str) else entry)
        source = os.path.join(work, source_name.format(episode=episode_id))
        if not os.path.isfile(source):
            continue
        shutil.copyfile(source, os.path.join(
            out_dir, target_name.format(episode=episode_id)))

    # Logs outlive everything else, by design.
    published_log = os.path.join(out_dir, "logs", "run.log")
    if log.path and os.path.isfile(log.path):
        shutil.copyfile(log.path, published_log)
    for extra in ("llama-server.log",):
        candidate = os.path.join(work, "logs", extra)
        if os.path.isfile(candidate):
            shutil.copyfile(candidate, os.path.join(out_dir, "logs", extra))
    for audit in sorted(glob.glob(os.path.join(work, "llm", "*.audit.jsonl"))):
        shutil.copyfile(audit, os.path.join(
            out_dir, "logs", os.path.basename(audit)))

    # Inputs go only once every output is on disk.
    if settings["KEEP_INPUTS"] == "1":
        log.info("keeping original inputs (KEEP_INPUTS=1)")
    else:
        for participant in sorted(sources):
            source = sources[participant]
            if os.path.isfile(source):
                os.remove(source)
                log.debug(f"removed input {os.path.basename(source)}")
        log.ok("original inputs removed")

    if settings["KEEP_WORK"] == "1":
        log.info(f"keeping work directory (KEEP_WORK=1): {work}")
        return log.path

    # Everything logged since the copy above would go with the directory, so
    # take the log again and keep writing to the published one from here.
    if log.path and os.path.isfile(log.path):
        shutil.copyfile(log.path, published_log)
    log.path = published_log
    shutil.rmtree(work, ignore_errors=True)
    log.ok("work directory removed")
    return published_log


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
