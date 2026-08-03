#!/usr/bin/env python3
"""Transcribe around each hand-made cut, to read off what was removed.

A recovered cut list says where an editor cut; this says what they cut, which is
what decides whether the pipeline could ever have found it. For each cut it
transcribes the original before, inside and after, and the edit at the seam —
the words present in the first and absent in the last are what went.

Short clips make Whisper hallucinate, so the seam is what to trust: read the
"joined" line against the three above it rather than the "inside" line alone.

Usage:
  what_was_cut.py ORIGINAL EDITED REF.json --endpoint http://host:8081
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile


def transcribe(path, start, end, endpoint, prompt, workdir) -> str:
    start, end = max(0.0, start), max(0.0, end)
    if end - start < 0.05:
        return "—"
    clip = os.path.join(workdir, "clip.wav")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-ss", f"{start:.3f}",
         "-to", f"{end:.3f}", "-ar", "16000", "-ac", "1", "-y", clip],
        check=True,
    )
    argv = ["curl", "-s", "-m", "120", "-F", f"file=@{clip}",
            "-F", "temperature=0", "-F", "response_format=json"]
    if prompt:
        argv += ["-F", f"prompt={prompt}"]
    argv.append(endpoint)
    out = subprocess.run(argv, check=True, stdout=subprocess.PIPE).stdout
    try:
        return " ".join(json.loads(out)["text"].split()) or "(silence)"
    except Exception:
        return f"<no transcript: {out[:60]!r}>"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("original")
    ap.add_argument("edited")
    ap.add_argument("ref", help="recover_cuts.py --json output")
    ap.add_argument("--endpoint", required=True,
                    help="whisper-server inference URL")
    ap.add_argument("--prompt", default="",
                    help="initial prompt, as WHISPER_PROMPT")
    ap.add_argument("--pad", type=float, default=3.0)
    args = ap.parse_args()

    ref = json.load(open(args.ref))
    kept = [tuple(k) for k in ref["kept"]]

    def edit_time(instant: float) -> float:
        """Where an original instant ended up in the edit."""
        seen = 0.0
        for start, end in kept:
            if instant < start:
                return seen
            if instant <= end:
                return seen + (instant - start)
            seen += end - start
        return seen

    workdir = tempfile.mkdtemp(prefix="podcast-cut-")
    try:
        print(f"{len(ref['removed'])} cuts in {args.original}")
        for start, end in ref["removed"]:
            seam = edit_time(start)
            print(f"\ncut {start:7.3f} → {end:7.3f}  ({end - start:.2f}s)   "
                  f"[the edit joins at {seam:.3f}]")
            for label, path, lo, hi in [
                ("before", args.original, start - args.pad, start),
                ("INSIDE", args.original, start, end),
                ("after ", args.original, end, end + args.pad),
                ("joined", args.edited, seam - args.pad, seam + args.pad),
            ]:
                text = transcribe(path, lo, hi, args.endpoint, args.prompt,
                                  workdir)
                print(f"  {label}: {text}")
    finally:
        clip = os.path.join(workdir, "clip.wav")
        if os.path.exists(clip):
            os.unlink(clip)
        os.rmdir(workdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
