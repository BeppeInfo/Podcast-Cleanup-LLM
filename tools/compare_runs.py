#!/usr/bin/env python3
"""Compare two or more finished renders of the same track against a hand edit.

`score_plan.py` answers "how close is this plan to the hand edit". This answers
the question that comes after a change to the pipeline: **is the new one closer
than the old one was.** It takes rendered audio rather than a plan, so anything
that produced a file can be compared — a previous release, a different model, a
different VAD, an export from other software entirely.

    compare_runs.py --original sample-host.flac \\
                    --reference sample-host-result.flac \\
                    --candidate bash=result-whisperhost.flac \\
                    --candidate whisperx=/tmp/eval/output/ep001/host.flac

Each file is aligned against the original by `recover_cuts.py`, which recovers
what was removed; every candidate is then scored against the reference with the
same interval metric `score_plan.py` uses, so the numbers here and there mean the
same thing and can be read side by side.

Read tools/README.md before trusting any of it — in particular "Reading the
numbers honestly", which is the difference between a score and a conclusion.
Three of its warnings apply directly:

  * **Score the talkative track.** On a mostly-silent one this measures where an
    editor happened to lump that track's share of a global cut, which is
    inaudible and tells you nothing. Run it per track and believe the busy one.
  * **A few points of F1 across a handful of cuts is noise.** The fixture this
    was written against has nine cuts in 57 seconds.
  * **The score can improve while the episode gets worse.** It is over removed
    *time*, so a cut taking the right region plus one word more barely moves it.
    Listen to the output, or read its transcript, before believing a win.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
from cleanup import intervals as iv  # noqa: E402

sys.path.insert(0, HERE)
from score_plan import score  # noqa: E402

# recover_cuts splices the original per its own answer and compares the result
# against the edit. Above this the cut list is wrong, and nothing computed from
# it means anything — tools/README.md is explicit that this is fixed first.
MAX_ENVELOPE_ERROR = 0.1

# ...but that check is not sufficient on its own. It compares only the
# overlapping prefix of the rebuilt audio against the edit, so a recovery that
# loses alignment and swallows a long stretch in the middle still scores well on
# the part before the damage. This is the check that catches it: the recovered
# cuts must account for exactly the length the edit is missing, and arithmetic
# cannot be fooled by a prefix. Seen for real — a 33.8s phantom cut in an
# otherwise sensible list, at an envelope error of 0.011.
MAX_DURATION_DISAGREEMENT = 0.25


def fingerprint(*paths: str) -> str:
    """A content hash of the inputs a cached answer was computed from.

    Hashing rather than trusting mtime, and hashing rather than nothing: the
    reference audio *does* get re-cut — an editor who finds a miss fixes it —
    and a cache keyed only on a label would answer the new question with the old
    number, confidently and silently. Hashing a gigabyte costs a second or two
    against a recovery that decodes and correlates the whole track.
    """
    digest = hashlib.sha256()
    for path in paths:
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def recover(original: str, edited: str, cache: str, label: str) -> dict:
    """The cut list turning `original` into `edited`, via recover_cuts.py.

    Cached on disk: recovery is the slow part, correlating whole tracks, and a
    comparison is usually re-run because a candidate changed, not the reference.
    The cache is invalidated by content, not by name.
    """
    out = os.path.join(cache, f"{label}.cuts.json")
    stamp = fingerprint(original, edited)
    if os.path.isfile(out):
        try:
            with open(out, encoding="utf-8") as handle:
                cached = json.load(handle)
            if cached.get("inputs_sha256") == stamp:
                return cached
        except (OSError, ValueError):
            pass
        os.remove(out)
    if not os.path.isfile(out):
        os.makedirs(cache, exist_ok=True)
        result = subprocess.run(
            [sys.executable, os.path.join(HERE, "recover_cuts.py"),
             original, edited, "--json", out, "--quiet"],
            capture_output=True, text=True, check=False)
        if result.returncode != 0 or not os.path.isfile(out):
            raise SystemExit(
                f"could not recover cuts for {label} ({edited}):\n"
                + (result.stderr or result.stdout).strip())
    with open(out, encoding="utf-8") as handle:
        recovered = json.load(handle)
    recovered["inputs_sha256"] = stamp
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(recovered, handle, indent=2)
    return recovered


def duration_disagreement(recovered: dict) -> float:
    """How far the recovered cuts are from explaining the edit's length."""
    removed = sum(end - start for start, end in recovered["removed"])
    expected = recovered["original_duration"] - recovered["edit_duration"]
    return abs(removed - expected)


def as_plan(recovered: dict) -> dict:
    """A recovered cut list in the shape score() reads.

    Deliberately reusing score_plan's function rather than writing a second
    metric: two ways of measuring the same thing is how two numbers that should
    agree quietly stop agreeing.
    """
    return {
        "cuts": [{"start": start, "end": end} for start, end in recovered["removed"]],
        "mutes": {},
    }


def compare(original: str, reference: str, candidates, cache: str, track: str):
    ref = recover(original, reference, cache, f"{track}-reference")
    rows = []
    for name, path in candidates:
        got = recover(original, path, cache, f"{track}-{name}")
        stats = score(as_plan(got), track, ref)
        # What the reference removed and this candidate did not touch at all.
        missed = iv.total(iv.subtract(stats["reference"], stats["cuts"]))
        rows.append({
            "name": name, "path": path,
            "envelope_error": got.get("envelope_error", 0.0),
            "disagreement": duration_disagreement(got),
            "duration": got["edit_duration"],
            "removed": stats["planned"],
            "precision": stats["precision"], "recall": stats["recall"],
            "f1": stats["f1"], "missed": missed,
        })
    return ref, rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original", required=True, help="the track as it went in")
    ap.add_argument("--reference", required=True,
                    help="the same track after editing by hand")
    ap.add_argument("--candidate", action="append", default=[],
                    metavar="NAME=PATH", required=True,
                    help="a rendered result to score; repeatable")
    ap.add_argument("--track", default="track",
                    help="participant name, for the mutes lookup and cache names")
    ap.add_argument("--cache", default=".compare-cache",
                    help="where recovered cut lists are kept between runs")
    ap.add_argument("--json", dest="out", help="also write the table here")
    args = ap.parse_args()

    candidates = []
    for item in args.candidate:
        if "=" not in item:
            raise SystemExit(f"--candidate expects NAME=PATH, got '{item}'")
        name, _, path = item.partition("=")
        if not os.path.isfile(path):
            raise SystemExit(f"no such file: {path}")
        candidates.append((name, path))

    ref, rows = compare(args.original, args.reference, candidates,
                        args.cache, args.track)

    wanted = iv.total(iv.normalize([tuple(v) for v in ref["removed"]]))
    print(f"\n{args.track}: {ref['original_duration']:.2f}s in, "
          f"hand edit {ref['edit_duration']:.2f}s "
          f"({wanted:.2f}s removed in {len(ref['removed'])} cuts)")
    if ref.get("envelope_error", 0.0) >= MAX_ENVELOPE_ERROR:
        print(f"  ! the reference itself did not align "
              f"(envelope error {ref['envelope_error']:.3f}) — nothing below "
              f"means anything until that is fixed")

    width = max([len(r["name"]) for r in rows] + [9]) + 2
    print(f"\n  {'candidate':<{width}}{'length':>9}{'removed':>10}"
          f"{'precision':>11}{'recall':>9}{'F1':>8}{'missed':>9}")
    def unreliable(row):
        return (row["envelope_error"] >= MAX_ENVELOPE_ERROR
                or row["disagreement"] >= MAX_DURATION_DISAGREEMENT)

    # Unreliable rows sort last and are not ranked against the rest: a broken
    # recovery produces numbers, and numbers in a sorted table read as results.
    for row in sorted(rows, key=lambda r: (unreliable(r), -r["f1"])):
        flag = " !" if unreliable(row) else ""
        print(f"  {row['name']:<{width}}{row['duration']:>8.2f}s"
              f"{row['removed']:>9.2f}s"
              f"{row['precision'] * 100:>10.1f}%{row['recall'] * 100:>8.1f}%"
              f"{row['f1'] * 100:>7.1f}%{row['missed']:>8.2f}s{flag}")

    for row in rows:
        if row["disagreement"] >= MAX_DURATION_DISAGREEMENT:
            print(f"\n  ! {row['name']}: recovered cuts total "
                  f"{row['removed']:.2f}s but the edit is only "
                  f"{row['duration'] and (ref['original_duration'] - row['duration']):.2f}s "
                  f"shorter than the original — off by {row['disagreement']:.2f}s. "
                  f"The alignment lost its place; this row is not a result.")
        elif row["envelope_error"] >= MAX_ENVELOPE_ERROR:
            print(f"\n  ! {row['name']}: did not align against the original "
                  f"(envelope error {row['envelope_error']:.3f}); not a result.")

    print("\n  F1 is over removed *time*, not cuts. A few points is noise — see\n"
          "  tools/README.md, 'Reading the numbers honestly', before concluding\n"
          "  anything from a small difference. Listen to the winner.")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump({"track": args.track, "reference": ref, "candidates": rows},
                      handle, indent=2)
        print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
