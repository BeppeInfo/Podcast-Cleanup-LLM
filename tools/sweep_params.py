#!/usr/bin/env python3
"""Re-plan an episode under many parameter combinations and score each.

Only the plan stage is re-run, against the transcripts and edits a real run
already produced, so a sweep costs no model time at all. Point it at a work
directory that has been through `--to detect` at least once.

Usage:
  sweep_params.py WORKDIR host=ref-host.json [guest=ref-guest.json] \\
      --vary silence_min_duration=1.5,0.5,0.3 --vary silence_keep=0.4,0.15
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(REPO, "python"))
from cleanup import intervals as iv  # noqa: E402

sys.path.insert(0, HERE)
from score_plan import HIT, score  # noqa: E402


def run_plan(work, params, out_dir, tag):
    params_file = os.path.join(out_dir, f"params-{tag}.json")
    plan_file = os.path.join(out_dir, f"plan-{tag}.json")
    with open(params_file, "w") as handle:
        json.dump(params, handle)
    argv = [
        sys.executable, os.path.join(REPO, "python", "cleanup_cli.py"), "plan",
        "--meta", os.path.join(work, "meta.json"),
        "--params", params_file,
        "--words-dir", os.path.join(work, "words"),
        "--loud-dir", os.path.join(work, "asr"),
        "--out", plan_file,
        "--force",
    ]
    if os.path.isdir(os.path.join(work, "llm")):
        argv += ["--edits-dir", os.path.join(work, "llm")]
    result = subprocess.run(argv, capture_output=True, text=True)
    if not os.path.exists(plan_file):
        return None, (result.stderr or result.stdout)[-300:]
    return json.load(open(plan_file)), None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("work", help="a run's work directory")
    ap.add_argument("refs", nargs="+", metavar="NAME=REF.json")
    ap.add_argument("--vary", action="append", default=[], metavar="KEY=A,B,C",
                    help="repeatable; every combination is tried")
    ap.add_argument("--against", default=None,
                    help="which track to score on (default: the first ref)")
    ap.add_argument("--keep", metavar="DIR",
                    help="keep the generated plans here")
    args = ap.parse_args()

    refs = {}
    for item in args.refs:
        name, path = item.split("=", 1)
        refs[name] = json.load(open(path))
    target = args.against or sorted(refs)[0]

    base = json.load(open(os.path.join(args.work, "params.json")))
    grid = {}
    for item in args.vary:
        key, values = item.split("=", 1)
        if key not in base:
            ap.error(f"unknown parameter {key!r}; known: {', '.join(sorted(base))}")
        grid[key] = [float(v) for v in values.split(",")]
    if not grid:
        ap.error("give at least one --vary")

    out_dir = args.keep or tempfile.mkdtemp(prefix="podcast-sweep-")
    os.makedirs(out_dir, exist_ok=True)

    keys = sorted(grid)
    width = max(len(k) for k in keys)
    print(f"scoring against {target}; {len(list(itertools.product(*(grid[k] for k in keys))))} "
          f"combinations, plans in {out_dir}\n")
    header = "  ".join(f"{k:>{max(width, 8)}}" for k in keys)
    print(f"{header} | {'cuts':>4} {'removed':>8} | {'prec':>6} {'rec':>6} "
          f"{'F1':>6} | hits")
    print("-" * (len(header) + 46))

    rows = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        params = copy.deepcopy(base)
        params.update(dict(zip(keys, combo)))
        tag = "-".join(f"{k}{v}" for k, v in zip(keys, combo))
        plan, error = run_plan(args.work, params, out_dir, tag)
        cells = "  ".join(f"{v:>{max(width, 8)}}" for v in combo)
        if plan is None:
            print(f"{cells} | failed: {error}")
            continue
        s = score(plan, target, refs[target])
        hits = sum(
            1 for lo, hi in s["reference"]
            if iv.total(iv.intersect([(lo, hi)], s["cuts"])) / (hi - lo) > HIT
        )
        rows.append((s["f1"], combo, s, hits))
        print(f"{cells} | {len(plan['cuts']):4d} {s['planned']:7.2f}s | "
              f"{s['precision'] * 100:5.1f}% {s['recall'] * 100:5.1f}% "
              f"{s['f1'] * 100:5.1f}% | {hits}/{len(s['reference'])}")

    if not rows:
        return 1
    rows.sort(key=lambda r: r[0], reverse=True)
    f1, combo, s, hits = rows[0]
    print("\nbest F1: " + ", ".join(f"{k}={v}" for k, v in zip(keys, combo)))
    print(f"  F1 {f1 * 100:.1f}%, recall {s['recall'] * 100:.1f}%, "
          f"{hits}/{len(s['reference'])} reference cuts hit, "
          f"{s['planned']:.2f}s removed against {s['wanted']:.2f}s by hand")
    print("\nA few points of F1 across a handful of cuts is noise. Confirm any "
          "change on a second episode before believing it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
