#!/usr/bin/env python3
"""Recover the cut list of a hand-edited track by aligning it to the original.

Given the track that went in and the version that came out of an editor, work
out which stretches of the original survived and which were removed. The result
is a reference cut list to measure a pipeline run against — see tools/README.md.

The edit is assumed to be an ordered concatenation of stretches of the original,
which is what cutting in a DAW produces. It is *not* assumed to be a byte copy:
an export re-renders, so matching is by normalised cross-correlation rather than
equality. The offset between edit time and original time is then piecewise
constant and jumps at every cut, so the work is to find that offset on a grid of
anchors, group the anchors by it, and refine each jump.

Usage:
  recover_cuts.py ORIGINAL EDITED [--json OUT] [--quiet]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

import numpy as np
from scipy.signal import fftconvolve

SR = 48000
ANCHOR = 4096          # anchor block length, ~85 ms
STRIDE = 2048          # anchor spacing, ~43 ms
ACCEPT = 0.98          # correlation at which the running offset still holds
FIND = 0.90            # below this a fresh search is called lost
HOP = 48               # envelope hop, 1 ms
ENV_WIN = 240          # envelope window, 5 ms
REFINE_SPAN = 200      # hops of context each side when placing a boundary


def decode(path: str) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1",
         "-ar", str(SR), "-"],
        check=True, stdout=subprocess.PIPE,
    ).stdout
    return np.frombuffer(raw, dtype="<f4").astype(np.float64)


def ncc_at(block: np.ndarray, other: np.ndarray, at: int) -> float:
    """Normalised cross-correlation of block against other at one offset."""
    if at < 0 or at + len(block) > len(other):
        return -1.0
    seg = other[at:at + len(block)]
    nb, ns = float(block @ block), float(seg @ seg)
    # Two silent stretches are a legitimate match; say so rather than dividing.
    if nb < 1e-9 and ns < 1e-9:
        return 1.0
    if nb < 1e-12 or ns < 1e-12:
        return 0.0
    return float(seg @ block) / np.sqrt(nb * ns)


def ncc_search(block: np.ndarray, other: np.ndarray, start: int):
    """Best correlation position of block anywhere in other[start:]."""
    seg = other[start:]
    if len(seg) < len(block):
        return None, -1.0
    nb = float(block @ block)
    if nb < 1e-12:
        return None, -1.0
    corr = fftconvolve(seg, block[::-1], mode="valid")
    energy = np.cumsum(np.concatenate([[0.0], seg * seg]))
    win = energy[len(block):] - energy[: len(seg) - len(block) + 1]
    ncc = corr / np.sqrt(np.maximum(win, 1e-12) * nb)
    i = int(np.argmax(ncc))
    return start + i, float(ncc[i])


def envelope(x: np.ndarray) -> np.ndarray:
    """log RMS envelope, one value per hop.

    Boundaries are placed on this rather than on the waveform. A cut lands in
    quiet audio by definition, and quiet audio in a re-render carries its own
    dither, which correlates with nothing — the envelope survives that.
    """
    power = np.convolve(x * x, np.ones(ENV_WIN) / ENV_WIN, mode="same")
    return 0.5 * np.log10(np.maximum(power[::HOP], 1e-12))


def anchor_offsets(edit: np.ndarray, orig: np.ndarray, verbose: bool):
    """(edit_pos, offset) per anchor, where offset = orig_pos - edit_pos."""
    found: list[tuple[int, int]] = []
    delta = 0
    unplaced = 0
    for pos in range(0, len(edit) - ANCHOR, STRIDE):
        block = edit[pos:pos + ANCHOR]
        if float(block @ block) < 1e-9:
            continue                      # digital silence pins nothing
        # The offset rarely changes, so try the running one before searching.
        if ncc_at(block, orig, pos + delta) >= ACCEPT:
            found.append((pos, delta))
            continue
        at, score = ncc_search(block, orig, max(0, pos + delta - ANCHOR))
        if score < FIND:
            at, score = ncc_search(block, orig, 0)
        if at is None or score < FIND:
            # Near-silence carries independent dither in a re-render, so it
            # never correlates. Those anchors simply pin nothing.
            unplaced += 1
            continue
        delta = at - pos
        found.append((pos, delta))
    if verbose:
        print(f"  anchors: {len(found)} placed, {unplaced} too quiet to place",
              file=sys.stderr)
    return found


def refine(edit_env, orig_env, lo: int, hi: int, before: int, after: int) -> int:
    """Place the boundary between two offsets, in [lo, hi] samples of edit time."""
    db, da = before // HOP, after // HOP
    best, best_err = lo, float("inf")

    def err(start, end, delta):
        start, end = max(start, 0), min(end, len(edit_env))
        if end <= start:
            return 0.0
        idx = np.arange(start, end) + delta
        keep = (idx >= 0) & (idx < len(orig_env))
        if not keep.any():
            return 0.0
        return float(np.abs(edit_env[start:end][keep] - orig_env[idx[keep]]).sum())

    for hop in range(lo // HOP, hi // HOP + 1):
        # Before the boundary the edit should follow the old offset, after it
        # the new one. The crossover is where that story costs least.
        total = (err(hop - REFINE_SPAN, hop, db)
                 + err(hop, hop + REFINE_SPAN, da))
        if total < best_err:
            best_err, best = total, hop * HOP
    return best


def segments(edit: np.ndarray, orig: np.ndarray, verbose: bool):
    """Matched stretches as (edit_start, edit_end, orig_start, orig_end)."""
    anchors = anchor_offsets(edit, orig, verbose)
    if not anchors:
        return []
    runs: list[list[int]] = []
    for pos, delta in anchors:
        if runs and runs[-1][2] == delta:
            runs[-1][1] = pos
        else:
            runs.append([pos, pos, delta])

    edit_env, orig_env = envelope(edit), envelope(orig)
    out: list[tuple[int, int, int, int]] = []
    for index, (_, last, delta) in enumerate(runs):
        start = 0 if index == 0 else out[-1][1]
        if index + 1 < len(runs):
            end = refine(edit_env, orig_env, last, runs[index + 1][0] + ANCHOR,
                         delta, runs[index + 1][2])
        else:
            end = len(edit)
        if end - start < HOP:
            continue
        out.append((start, end, start + delta, end + delta))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("original", help="the track as it went in")
    ap.add_argument("edited", help="the same track after editing")
    ap.add_argument("--json", dest="out", help="write the cut list here")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    orig, edit = decode(args.original), decode(args.edited)
    if not args.quiet:
        print(f"original {len(orig) / SR:.3f}s   edit {len(edit) / SR:.3f}s")

    segs = segments(edit, orig, not args.quiet)
    if not segs:
        print("could not align: no anchor matched", file=sys.stderr)
        return 1

    kept = [(a / SR, b / SR) for _, _, a, b in segs]
    removed: list[tuple[float, float]] = []
    cursor = 0
    for _, _, start, end in segs:
        if start > cursor:
            removed.append((cursor / SR, start / SR))
        cursor = max(cursor, end)
    if len(orig) - cursor > HOP:
        removed.append((cursor / SR, len(orig) / SR))

    if not args.quiet:
        print(f"\nkept, in original time ({len(segs)} segments):")
        for _, _, start, end in segs:
            print(f"  {start / SR:8.3f} → {end / SR:8.3f}  "
                  f"({(end - start) / SR:6.3f}s)")
        print(f"\nremoved, in original time ({len(removed)} cuts):")
        for start, end in removed:
            print(f"  {start:8.3f} → {end:8.3f}  ({end - start:6.3f}s)")
        print(f"\ntotal removed {sum(e - s for s, e in removed):.3f}s "
              f"of {len(orig) / SR:.3f}s")

    # Splice the original per the recovered list and see whether it reproduces
    # the edit. A wrong cut list shows up here as envelope error.
    rebuilt = np.concatenate([orig[a:b] for _, _, a, b in segs])
    n = min(len(rebuilt), len(edit))
    error = np.abs(envelope(rebuilt[:n]) - envelope(edit[:n]))
    verdict = "matches" if error.mean() < 0.1 else "DOES NOT MATCH"
    print(f"\ncheck: rebuilt {len(rebuilt) / SR:.3f}s vs edit "
          f"{len(edit) / SR:.3f}s, envelope error mean {error.mean():.3f} "
          f"p95 {np.percentile(error, 95):.3f} — {verdict}")

    if args.out:
        with open(args.out, "w") as handle:
            json.dump({
                "original": args.original,
                "edit": args.edited,
                "original_duration": len(orig) / SR,
                "edit_duration": len(edit) / SR,
                "kept": [list(k) for k in kept],
                "removed": [list(r) for r in removed],
                "envelope_error": round(float(error.mean()), 4),
            }, handle, indent=2)
        print(f"wrote {args.out}")
    return 0 if error.mean() < 0.1 else 2


if __name__ == "__main__":
    sys.exit(main())
