"""Turn a plan into ffmpeg filter graphs, and into the final transcript.

The graph per track is:

    asetnsamples -> volume (mutes) -> aselect (cuts) -> asetpts

`aselect` decides per *frame*, not per sample, so cut points land on a frame
boundary. That is why `asetnsamples` comes first and why every track in an
episode is required to share a sample rate: with an identical frame size and an
identical select expression, every track drops exactly the same frames and the
set stays sample-aligned. It also means the output length can be predicted
exactly rather than merely approximately — see `expected_output_samples`.

Both expressions are emitted as a balanced decision tree instead of a flat sum
of `between()` terms. A two-hour episode can carry several hundred cuts, and a
flat expression would be evaluated in full for every one of a million-odd
frames; the tree turns that into a handful of comparisons.
"""

from __future__ import annotations

import json
import math

from . import intervals as iv


def _fmt(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


def _tree(spans, leaf) -> str:
    """Balanced dispatch over sorted, disjoint spans: O(log n) per frame."""
    if not spans:
        return "0"
    if len(spans) == 1:
        return leaf(*spans[0])
    middle = len(spans) // 2
    pivot = spans[middle][0]
    return (
        f"if(lt(t,{_fmt(pivot)}),"
        f"{_tree(spans[:middle], leaf)},"
        f"{_tree(spans[middle:], leaf)})"
    )


def cut_expression(cuts) -> str:
    """1 while inside a cut, 0 otherwise."""
    spans = iv.normalize([(c["start"], c["end"]) for c in cuts])
    return _tree(spans, lambda s, e: f"between(t,{_fmt(s)},{_fmt(e)})")


def mute_gain_expression(mutes, fade: float) -> str:
    """Gain multiplier: 1 normally, 0 inside a mute, ramped over `fade`."""
    spans = iv.normalize([(m["start"], m["end"]) for m in mutes])
    fade = max(fade, 1e-4)

    def leaf(start, end):
        # Trapezoid: ramps down over [start-fade, start], flat 0 to end,
        # ramps back up over [end, end+fade].
        low = _fmt(start - fade)
        high = _fmt(end + fade)
        f = _fmt(fade)
        return f"clip((t-{low})/{f},0,1)*clip(({high}-t)/{f},0,1)"

    # Padded spans are disjoint (mutes were fused across gaps of 2*fade), so the
    # tree pivots on the padded start.
    padded = [(s - fade, e + fade) for s, e in spans]
    ordered = sorted(zip(padded, spans))
    tree = _tree(
        [p for p, _ in ordered],
        lambda ps, pe: leaf(ps + fade, pe - fade),
    )
    return f"1-({tree})"


def build_filter(cuts, mutes, frame_samples: int, fade: float) -> str | None:
    """Full filtergraph for one track, or None when the track is untouched."""
    if not cuts and not mutes:
        return None

    chain = [f"asetnsamples=n={frame_samples}:p=0"]
    if mutes:
        chain.append(
            f"volume=volume='{mute_gain_expression(mutes, fade)}':eval=frame"
        )
    if cuts:
        chain.append(f"aselect='not({cut_expression(cuts)})'")
        chain.append("asetpts=N/SR/TB")
    return "[0:a]" + ",".join(chain) + "[out]"


def expected_output_samples(
    duration: float, sample_rate: int, frame_samples: int, cuts
) -> int:
    """Exactly how many samples the graph above will emit.

    Replicates `aselect`'s frame-level decision: frame k starts at sample
    k*frame_samples, its timestamp is that divided by the sample rate, and the
    whole frame is kept or dropped on the strength of that one timestamp.
    """
    total_samples = int(round(duration * sample_rate))
    if not cuts:
        return total_samples

    spans = iv.normalize([(c["start"], c["end"]) for c in cuts])
    frames = math.ceil(total_samples / frame_samples) if total_samples else 0
    kept = 0
    index = 0

    for frame in range(frames):
        offset = frame * frame_samples
        length = min(frame_samples, total_samples - offset)
        t = offset / float(sample_rate)
        while index < len(spans) and spans[index][1] < t - iv.EPS:
            index += 1
        inside = (
            index < len(spans)
            and spans[index][0] - iv.EPS <= t <= spans[index][1] + iv.EPS
        )
        if not inside:
            kept += length
    return kept


# --- transcript ---------------------------------------------------------------


def _srt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis == 1000:
        millis, secs = 0, secs + 1
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _clock(seconds: float) -> str:
    hours, rest = divmod(int(max(0.0, seconds)), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def build_transcript(plan, transcripts) -> dict:
    """Speaker-labelled transcript on the *rendered* timeline.

    Words removed by a cut, or silenced by a mute, are dropped from the text —
    the transcript describes the audio that now exists, not the audio that was
    recorded.
    """
    timeline = iv.Timeline([tuple(span) for span in plan["keep"]])
    cut_spans = [(c["start"], c["end"]) for c in plan["cuts"]]
    entries = []
    removed_words = 0

    for participant, parsed in transcripts.items():
        mute_spans = [
            (m["start"], m["end"]) for m in plan["mutes"].get(participant, [])
        ]
        gone = iv.union(cut_spans, mute_spans)
        words = parsed["words"]

        for segment in parsed["segments"]:
            first, last = segment["first_word"], segment["last_word"]
            if first < 0 or last < first:
                continue
            surviving = []
            for word in words[first : last + 1]:
                middle = (word["start"] + word["end"]) / 2.0
                if iv.contains(gone, middle):
                    removed_words += 1
                    continue
                surviving.append(word)
            if not surviving:
                continue

            start, end = timeline.map_span(
                surviving[0]["start"], surviving[-1]["end"]
            )
            if end - start < 0.01:
                continue
            entries.append(
                {
                    "participant": participant,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "text": " ".join(w["text"] for w in surviving),
                }
            )

    entries.sort(key=lambda e: (e["start"], e["participant"]))
    return {
        "episode_id": plan["episode_id"],
        "duration": round(timeline.duration, 3),
        "participants": plan["participants"],
        "languages": {
            p: parsed.get("language", "") for p, parsed in transcripts.items()
        },
        "removed_words": removed_words,
        "segments": entries,
    }


def transcript_to_srt(transcript) -> str:
    blocks = []
    for index, entry in enumerate(transcript["segments"], start=1):
        blocks.append(
            f"{index}\n"
            f"{_srt_time(entry['start'])} --> {_srt_time(entry['end'])}\n"
            f"{entry['participant']}: {entry['text']}\n"
        )
    return "\n".join(blocks)


def transcript_to_text(transcript) -> str:
    lines = [
        f"# {transcript['episode_id']}",
        f"# {len(transcript['segments'])} segments, "
        f"{_clock(transcript['duration'])} after cleanup",
        "",
    ]
    previous = None
    for entry in transcript["segments"]:
        if entry["participant"] != previous:
            lines.append("")
            previous = entry["participant"]
        lines.append(
            f"[{_clock(entry['start'])}] {entry['participant']}: {entry['text']}"
        )
    return "\n".join(lines) + "\n"


def write_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
