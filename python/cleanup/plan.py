"""Unify the transcript's silence and the LLM's findings into one edit plan.

Both inputs are now derived from the same transcript: silence is where Whisper
returned no words (`transcript.speech_from_words`), and edits are what the LLM
found in those words. There is no second opinion about the audio to cross-check
against — see `looping_words` for what is still checkable, and DESIGN.md for why
that trade was taken.

Two kinds of operation come out of this, and the distinction is the heart of
the pipeline:

**cut**   Applied to every track at the same time range, shortening the
          timeline. Used for over-long silence, and for a disfluency during
          which nobody else was speaking.

**mute**  Applied to a single track, leaving the timeline untouched. Used when
          a disfluency overlaps another participant's speech: cutting there
          would take a bite out of whoever else was talking, so the stutter is
          silenced on its own track instead and everyone stays in sync.

Because every cut is global and identical, the tracks remain sample-aligned
with each other by construction.
"""

from __future__ import annotations

from . import intervals as iv
from . import transcript as tr

# A mute fragment left over after subtracting cuts is not worth rendering.
MIN_MUTE = 0.02


def _silence_cuts(gaps, duration, params) -> list[dict]:
    """Shorten each qualifying silent gap, rather than removing it outright."""
    edge_margin = max(params["edge_keep"], params["cut_padding"])
    inner_margin = max(params["silence_keep"] / 2.0, params["cut_padding"])
    cuts: list[dict] = []

    for start, end in gaps:
        at_head = start <= iv.EPS
        at_tail = end >= duration - iv.EPS
        if at_head and at_tail:
            continue  # nothing but silence; leave it to the caller to notice

        if at_head:
            span = (0.0, end - edge_margin)
            reason = "lead_in"
        elif at_tail:
            span = (start + edge_margin, duration)
            reason = "lead_out"
        else:
            if end - start < params["silence_min_duration"]:
                continue
            span = (start + inner_margin, end - inner_margin)
            reason = "silence"

        if span[1] - span[0] >= params["min_cut"]:
            cuts.append(
                {
                    "start": round(span[0], 4),
                    "end": round(span[1], 4),
                    "reason": reason,
                    "source": "silence",
                    "gap": round(end - start, 3),
                }
            )
    return cuts


def _pad_edit(words, edit, duration, cut_padding) -> tuple[float, float]:
    """Widen an edit into the silence around it, never into neighbouring words.

    Whisper's word boundaries are approximate, so a little margin protects the
    leading consonant of the word that follows. Claiming at most half of the
    actual gap keeps that margin from swallowing real speech.
    """
    before, after = tr.neighbour_gaps(words, edit["first"], edit["last"])
    start = max(0.0, edit["start"] - min(cut_padding, before / 2.0))
    end = min(duration, edit["end"] + min(cut_padding, after / 2.0))
    return start, end


# Length of the repeated run this looks for, in words. Long enough that ordinary
# speech does not repeat it by accident ("you know what I mean", "and then I"),
# short enough to catch a looping clause rather than only a whole sentence.
LOOP_SHINGLE = 8

# How often a run must recur before it is a loop rather than a turn of phrase.
LOOP_MIN_REPEATS = 5

# And how much of the track those repeats must account for. A podcast can
# legitimately repeat a catchphrase; it cannot legitimately spend a quarter of
# its words on one clause.
LOOP_WARN_FRACTION = 0.25


def looping_words(participants, words) -> dict:
    """Tracks whose transcript repeats one run of words far past plausibility.

    This is the check that replaces comparing a transcript against an
    independent speech map. There is no independent map any more — Whisper's own
    VAD decides what gets transcribed and the map is derived from what comes
    back, so the two can never disagree by construction.

    What remains detectable is the shape of the failure itself. Handed silence
    or noise, Whisper does not invent varied text: it repeats. On the recording
    that prompted all of this it produced one sentence 198 times across 4.6
    minutes of a muted mic, 73% of that track. That is visible in the transcript
    alone, and needs no second opinion about the audio.
    """
    import collections

    looping: dict[str, dict] = {}
    for participant in participants:
        track = [
            "".join(ch for ch in (w.get("text") or "").lower() if ch.isalnum())
            for w in words.get(participant) or []
        ]
        if len(track) < LOOP_SHINGLE * LOOP_MIN_REPEATS:
            continue
        counts = collections.Counter(
            tuple(track[i : i + LOOP_SHINGLE])
            for i in range(len(track) - LOOP_SHINGLE + 1)
        )
        run, repeats = counts.most_common(1)[0]
        if repeats < LOOP_MIN_REPEATS:
            continue
        # Each repeat accounts for at least the run itself. Overlapping shingles
        # of a longer repeated sentence all report the same count, so this
        # under-counts a long loop rather than inflating a short one.
        covered = min(repeats * LOOP_SHINGLE, len(track))
        if covered / len(track) < LOOP_WARN_FRACTION:
            continue
        looping[participant] = {
            "repeats": repeats,
            "words": len(track),
            "covered": covered,
            "text": " ".join(word for word in run if word),
        }
    return looping


def build_plan(meta, speech, edits, words, params) -> dict:
    """Assemble the plan. `speech`/`edits`/`words` are keyed by participant.

    `speech` comes from `transcript.speech_from_words` — the padded union of the
    words Whisper returned, not a separate opinion about the audio. Silence is
    therefore "no words here", and the padding is what keeps a cut off the edge
    of a word whose timing is approximate.
    """
    duration = float(meta["duration"])
    participants = [track["participant"] for track in meta["tracks"]]

    speech_all = iv.union(*(speech.get(p, []) for p in participants))
    gaps = iv.complement(speech_all, 0.0, duration)

    cut_records = _silence_cuts(gaps, duration, params)
    mutes: dict[str, list[dict]] = {p: [] for p in participants}
    warnings: list[str] = []

    if not speech_all:
        warnings.append(
            "no track transcribed a single word, so the whole episode reads as "
            "silence — check that the whisper endpoint returned anything at all "
            "before trusting this plan"
        )

    # Classify every LLM finding as a global cut or a single-track mute.
    for participant in participants:
        found = edits.get(participant) or []
        if not found:
            continue
        track_words = words.get(participant) or []
        others = iv.union(
            *(speech.get(other, []) for other in participants if other != participant)
        )

        for edit in found:
            start, end = _pad_edit(track_words, edit, duration, params["cut_padding"])
            if end - start < iv.EPS:
                continue
            crosstalk = iv.overlap_amount((start, end), others)
            common = {
                "start": round(start, 4),
                "end": round(end, 4),
                "reason": edit["kind"],
                "confidence": edit["confidence"],
                "text": edit["text"],
                "participant": participant,
            }
            if crosstalk > iv.EPS:
                # Someone else is talking here: silence this track only.
                mutes[participant].append(
                    {**common, "crosstalk": round(crosstalk, 3)}
                )
            else:
                cut_records.append({**common, "source": f"llm:{participant}"})

    # Fuse overlapping cuts, remembering what each merged span came from.
    cut_records.sort(key=lambda c: (c["start"], c["end"]))
    cuts: list[dict] = []
    for record in cut_records:
        if cuts and record["start"] <= cuts[-1]["end"] + iv.EPS:
            previous = cuts[-1]
            previous["end"] = max(previous["end"], record["end"])
            previous["reasons"] = sorted(set(previous["reasons"] + [record["reason"]]))
            previous["sources"] = sorted(set(previous["sources"] + [record["source"]]))
            previous["details"].append(record)
        else:
            cuts.append(
                {
                    "start": record["start"],
                    "end": record["end"],
                    "reasons": [record["reason"]],
                    "sources": [record["source"]],
                    "details": [record],
                }
            )

    cut_spans = [(c["start"], c["end"]) for c in cuts]
    keep = iv.complement(cut_spans, 0.0, duration)

    # Nothing here compares the transcript against the audio any more, because
    # the speech map is made of the transcript. What is still visible is Whisper
    # repeating itself, which is how it fails when it is given something that is
    # not speech.
    looping = looping_words(participants, words)
    for participant, detail in sorted(looping.items()):
        warnings.append(
            f"{participant}'s transcript repeats \"{detail['text']}\" "
            f"{detail['repeats']} times, accounting for at least "
            f"{detail['covered'] / detail['words'] * 100:.0f}% of its "
            f"{detail['words']} words. That is what Whisper does when handed "
            "something that is not speech; treat this transcript, the speech map "
            "derived from it, and every edit built on it as unreliable until the "
            "audio is checked. If the server is not running Silero over the "
            "audio first, WHISPER_VAD is what turns that on"
        )

    # A muted stretch inside a cut is moot; trim mutes down to what survives,
    # and fuse those close enough that their fades would otherwise collide.
    fade_gap = 2.0 * params["mute_fade"]
    resolved_mutes: dict[str, list[dict]] = {}
    for participant, items in mutes.items():
        if not items:
            resolved_mutes[participant] = []
            continue
        fused = iv.normalize([(m["start"], m["end"]) for m in items], gap=fade_gap)
        surviving = [
            (start, end)
            for start, end in iv.subtract(fused, cut_spans)
            if end - start >= MIN_MUTE
        ]
        annotated = []
        for start, end in surviving:
            reasons = sorted(
                {m["reason"] for m in items if m["start"] < end and m["end"] > start}
            )
            texts = [
                m["text"] for m in items if m["start"] < end and m["end"] > start
            ]
            annotated.append(
                {
                    "start": round(start, 4),
                    "end": round(end, 4),
                    "reasons": reasons or ["mute"],
                    "text": " / ".join(texts),
                }
            )
        resolved_mutes[participant] = annotated

    removed = iv.total(cut_spans)
    fraction = removed / duration if duration > 0 else 0.0
    if fraction > params["max_cut_fraction"]:
        warnings.append(
            f"plan removes {fraction * 100:.1f}% of the episode, above the "
            f"{params['max_cut_fraction'] * 100:.0f}% safety limit"
        )

    stats = {
        "duration": round(duration, 3),
        "output_duration": round(iv.total(keep), 3),
        "removed": round(removed, 3),
        "removed_fraction": round(fraction, 4),
        "speech_total": round(iv.total(speech_all), 3),
        "silence_gaps": len(gaps),
        "cut_count": len(cuts),
        "cut_from_silence": sum(
            1 for c in cuts if any(s == "silence" for s in c["sources"])
        ),
        "cut_from_llm": sum(
            1 for c in cuts if any(s.startswith("llm:") for s in c["sources"])
        ),
        "mute_count": sum(len(v) for v in resolved_mutes.values()),
        "mute_seconds": round(
            sum(m["end"] - m["start"] for v in resolved_mutes.values() for m in v), 3
        ),
        "per_participant": {
            p: {
                "speech": round(iv.total(speech.get(p, [])), 3),
                "words": len(words.get(p) or []),
                "edits_found": len(edits.get(p) or []),
                "mutes": len(resolved_mutes.get(p, [])),
            }
            for p in participants
        },
    }

    return {
        "episode_id": meta["episode_id"],
        "duration": round(duration, 3),
        "participants": participants,
        "params": params,
        "cuts": cuts,
        "mutes": resolved_mutes,
        "keep": [[round(s, 4), round(e, 4)] for s, e in keep],
        "looping_transcripts": {
            participant: detail for participant, detail in sorted(looping.items())
        },
        "stats": stats,
        "warnings": warnings,
    }


def format_report(plan) -> str:
    """A readable summary of what the plan does, kept alongside the outputs."""
    stats = plan["stats"]
    lines = [
        f"Episode:          {plan['episode_id']}",
        f"Participants:     {', '.join(plan['participants'])}",
        f"Original length:  {_hms(stats['duration'])}",
        f"Cleaned length:   {_hms(stats['output_duration'])}",
        f"Removed:          {_hms(stats['removed'])} "
        f"({stats['removed_fraction'] * 100:.1f}%) in {stats['cut_count']} cuts",
        f"                  {stats['cut_from_silence']} involving silence, "
        f"{stats['cut_from_llm']} involving speech edits",
        f"Muted in place:   {stats['mute_count']} spans, "
        f"{stats['mute_seconds']:.1f}s total (crosstalk — timeline preserved)",
        "",
        "Per participant:",
    ]
    for participant, values in stats["per_participant"].items():
        lines.append(
            f"  {participant:<16} speech {_hms(values['speech'])}  "
            f"words {values['words']:<6} edits found {values['edits_found']:<4} "
            f"muted {values['mutes']}"
        )

    kinds: dict[str, int] = {}
    for cut in plan["cuts"]:
        for reason in cut["reasons"]:
            kinds[reason] = kinds.get(reason, 0) + 1
    for spans in plan["mutes"].values():
        for mute in spans:
            for reason in mute["reasons"]:
                kinds[f"{reason} (muted)"] = kinds.get(f"{reason} (muted)", 0) + 1
    if kinds:
        lines += ["", "By reason:"]
        for reason in sorted(kinds):
            lines.append(f"  {reason:<24} {kinds[reason]}")

    if plan["warnings"]:
        lines += ["", "Warnings:"]
        lines += [f"  ! {w}" for w in plan["warnings"]]

    return "\n".join(lines) + "\n"


def _hms(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    return f"{minutes}m{secs:02d}s"
