#!/usr/bin/env python3
"""Score a plan against cut lists recovered from a hand edit.

Reports, per track, how much of what was removed by hand the plan also removes,
which reference cuts it hit and which it missed, and which of its own cuts have
no counterpart. See tools/README.md for the workflow.

Usage:
  score_plan.py PLAN.json host=ref-host.json guest=ref-guest.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python")
)
from cleanup import intervals as iv  # noqa: E402

HIT = 0.6          # fraction of a reference cut the plan must cover to count


def score(plan, name, ref):
    """Overlap statistics for one track."""
    duration = ref["original_duration"]
    reference = iv.normalize([tuple(v) for v in ref["removed"]])
    # A global cut only bites where this track actually has audio.
    cuts = iv.intersect(
        iv.normalize([(c["start"], c["end"]) for c in plan["cuts"]]),
        [(0.0, duration)],
    )
    mutes = iv.normalize(
        [(m["start"], m["end"]) for m in plan["mutes"].get(name, [])]
    )
    overlap = iv.total(iv.intersect(cuts, reference))
    planned, wanted = iv.total(cuts), iv.total(reference)
    precision = overlap / planned if planned else 0.0
    recall = overlap / wanted if wanted else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return dict(duration=duration, reference=reference, cuts=cuts, mutes=mutes,
                overlap=overlap, planned=planned, wanted=wanted,
                precision=precision, recall=recall, f1=f1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan", help="plan.json from a run's work directory")
    ap.add_argument("refs", nargs="+", metavar="NAME=REF.json",
                    help="recover_cuts.py output, per participant")
    args = ap.parse_args()

    plan = json.load(open(args.plan))
    refs = {}
    for item in args.refs:
        if "=" not in item:
            ap.error(f"expected NAME=REF.json, got {item!r}")
        name, path = item.split("=", 1)
        refs[name] = json.load(open(path))

    stats = plan["stats"]
    print(f"episode {plan['episode_id']}  duration {plan['duration']}s")
    print(f"plan: {stats['cut_count']} cuts, {stats['removed']}s removed "
          f"({stats['removed_fraction'] * 100:.1f}%), "
          f"{stats['cut_from_silence']} from silence, "
          f"{stats['cut_from_llm']} from llm, "
          f"{stats['mute_count']} mutes ({stats['mute_seconds']}s)")
    for warning in plan["warnings"]:
        print(f"  ! {warning}")

    for name in sorted(refs):
        s = score(plan, name, refs[name])
        print(f"\n=== {name}  (track {s['duration']:.3f}s)")
        print(f"  reference removes {s['wanted']:.3f}s in "
              f"{len(s['reference'])} cuts")
        print(f"  plan removes      {s['planned']:.3f}s in "
              f"{len(s['cuts'])} spans")
        print(f"  overlap {s['overlap']:.3f}s   "
              f"precision {s['precision'] * 100:.1f}%  "
              f"recall {s['recall'] * 100:.1f}%  F1 {s['f1'] * 100:.1f}%")
        if s["mutes"]:
            print(f"  plus {iv.total(s['mutes']):.3f}s muted in "
                  f"{len(s['mutes'])} spans, which keep the timeline")

        print("\n  reference cuts vs plan:")
        for start, end in s["reference"]:
            covered = iv.total(iv.intersect([(start, end)], s["cuts"]))
            muted = iv.total(iv.intersect([(start, end)], s["mutes"]))
            share = covered / (end - start)
            verdict = ("hit" if share > HIT else
                       "partial" if covered > 0.05 else "MISSED")
            note = f", {muted:.2f}s muted" if muted > 0.05 else ""
            print(f"    {start:7.3f} → {end:7.3f} ({end - start:5.2f}s)  "
                  f"plan covers {covered:5.2f}s ({share * 100:5.1f}%) "
                  f"{verdict}{note}")

        print("\n  plan cuts with no counterpart:")
        spurious = False
        for cut in plan["cuts"]:
            lo = max(cut["start"], 0.0)
            hi = min(cut["end"], s["duration"])
            if hi - lo <= 0.01:
                continue
            covered = iv.total(iv.intersect([(lo, hi)], s["reference"]))
            if covered / (hi - lo) < HIT:
                spurious = True
                print(f"    {lo:7.3f} → {hi:7.3f} ({hi - lo:5.2f}s)  "
                      f"in reference {covered / (hi - lo) * 100:5.1f}%   "
                      f"[{','.join(cut.get('sources', []))}]")
        if not spurious:
            print("    (none)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
