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

import glob
import json
import os

from . import intervals as iv
from . import plan as planner
from . import render, transcript as tr


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
