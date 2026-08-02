"""Unify the transcript's silence and the LLM's findings into one edit plan.

Both inputs are derived from the same transcript: silence is where Whisper
returned no words (`transcript.speech_from_words`), and edits are what the LLM
found in those words. DESIGN.md covers why that trade was taken.

Two checks guard the obvious hazard in it. `looping_words` catches a transcript
that repeats itself, which is how Whisper fails on material that is not speech.
`untranscribed_audio` compares each transcript against a level scan of its own
track — the only input here that Whisper had no hand in — and the plan refuses to
proceed when a cut would remove loud audio that no transcript accounts for.

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


# A stretch of loud audio with no words has to last this long to be worth
# mentioning. Below it, ordinary things account for the difference: a cough, a
# breath the level scan called loud, the tail of a word whose timing is short.
UNTRANSCRIBED_MIN_SPAN = 3.0

# And a cut has to remove this much of it, in total, before the plan refuses to
# proceed. Set from the incident it exists for: a skipped decode window cost 33
# seconds of clear speech, and a silence cut then removed 6 of them.
UNTRANSCRIBED_BLOCK_SECONDS = 5.0


def untranscribed_audio(participants, words, loud, duration, pad) -> dict:
    """Loud audio a track's own transcript has no words for.

    This is the one opinion about the audio that does not come from Whisper. The
    level scan cannot tell speech from a cough, which is exactly why it is not
    the speech map — but it can tell loud from silent, and a long stretch of loud
    audio that produced no words at all is either something Whisper declined to
    transcribe or something it never saw.

    Both happen. Whisper's Silero pass drops non-speech on purpose, and that is
    the trade this design accepted. But Whisper also discards whole decode
    windows: it works in 30-second windows, and one whose decode ends on a lone
    timestamp token is skipped entirely ("single timestamp ending - skip entire
    chunk"). On the recording this was written for that cost 33 seconds of clear
    speech, and nothing else could see it — the transcript was self-consistent,
    and the speech map is derived from the transcript, so both agreed the audio
    was empty.

    Which window a passage falls in depends on how much speech precedes it, so no
    setting prevents this; it can only be detected after the fact. That is why
    this check blocks rather than warns when a cut is already removing the audio.
    """
    missing: dict[str, list[tuple[float, float]]] = {}
    for participant in participants:
        spans = loud.get(participant) or []
        if not spans:
            continue
        covered = tr.speech_from_words(words.get(participant), duration, pad)
        gaps = [
            (start, end)
            for start, end in iv.subtract(iv.normalize(spans), covered)
            if end - start >= UNTRANSCRIBED_MIN_SPAN
        ]
        if gaps:
            missing[participant] = gaps
    return missing


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


def build_plan(meta, speech, edits, words, params, loud=None) -> dict:
    """Assemble the plan. `speech`/`edits`/`words`/`loud` are keyed by participant.

    `speech` comes from `transcript.speech_from_words` — the padded union of the
    words Whisper returned, not a separate opinion about the audio. Silence is
    therefore "no words here", and the padding is what keeps a cut off the edge
    of a word whose timing is approximate.

    `loud` is the level scan, and is the exception: the one input that does not
    come from Whisper. It is not used to decide anything, only to refuse — see
    `untranscribed_audio`.
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
    # Loud audio nothing transcribed. Reported wherever it is, and refused when a
    # cut actually removes it — that combination is the shape of the bug this
    # exists for, and the only one worth stopping a run over.
    blocking: list[str] = []
    unheard = untranscribed_audio(
        participants, words, loud or {}, duration, params["speech_pad"]
    )
    cut_unheard = 0.0
    for participant, gaps in sorted(unheard.items()):
        total = sum(end - start for start, end in gaps)
        inside = iv.total(iv.intersect(gaps, cut_spans)) if cut_spans else 0.0
        cut_unheard += inside
        widest = max(gaps, key=lambda span: span[1] - span[0])
        warnings.append(
            f"{total:.0f}s of {participant}'s audio is loud enough to be speech "
            f"but produced no words at all, in {len(gaps)} stretch(es), the "
            f"longest {widest[1] - widest[0]:.0f}s at {widest[0]:.0f}s"
            + (f" — {inside:.0f}s of it falls inside cuts" if inside > iv.EPS else "")
        )
    if cut_unheard >= UNTRANSCRIBED_BLOCK_SECONDS:
        blocking.append(
            f"cuts remove {cut_unheard:.0f}s of audio that is loud enough to be "
            "speech and that no transcript accounts for. Either Whisper skipped a "
            "decode window, which lowering WHISPER_CHUNK_SECONDS makes smaller but "
            "cannot prevent, or it declined to transcribe non-speech, which "
            "SPLIT_SILENCE_THRESHOLD can be raised to stop reporting. Listen to "
            "the spans in plan.json before overriding"
        )

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
        limit = (
            f"plan removes {fraction * 100:.1f}% of the episode, above the "
            f"{params['max_cut_fraction'] * 100:.0f}% safety limit"
        )
        warnings.append(limit)
        blocking.append(limit)

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
        "untranscribed_seconds": round(
            sum(e - s for gaps in unheard.values() for s, e in gaps), 3
        ),
        "untranscribed_in_cuts": round(cut_unheard, 3),
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
        # Kept span by span so it can be listened to rather than only counted.
        "untranscribed_audio": {
            participant: [[round(s, 3), round(e, 3)] for s, e in gaps]
            for participant, gaps in sorted(unheard.items())
        },
        "stats": stats,
        "warnings": warnings,
        # The subset of warnings that should stop a run. Kept separate so the
        # decision lives here rather than in a caller matching on message text.
        "blocking": blocking,
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
