# tools/

Measuring a run against an episode someone edited by hand. Nothing here is part
of the pipeline — `clean-podcast.sh` never calls it, and the tests do not need
it. It exists so that "the cleanup got closer" can be a number rather than an
impression.

The audio these work on does not belong in git. `.gitignore` keeps `*.flac` and
friends out of the checkout root for that reason; keep the fixtures wherever you
like outside the repo and pass paths in.

`recover_cuts.py` needs numpy and scipy, and `sweep_params.py` imports its
scorer. Nothing under `python/cleanup/` does — the pipeline stays on the
standard library, and that is deliberate. Install them for the measuring, not
for the running.

**Steps 2 and 5 still POST to a whisper-server.** The pipeline stopped needing
one when transcription moved to WhisperX, but these two only ever wanted *a*
transcript of some audio, and they were not ported along with it. They work
against any whisper-server you still have; if you have none, transcribe the
audio however you like and read the text — the tool is a convenience, the
comparison is the point.

## The workflow

Edit an episode by hand, keep the originals, and the difference between them is
a reference edit to aim at.

**1. Recover what the editor did.** The export is a re-render, not a byte copy,
so this aligns by correlation rather than equality and reports which stretches
of the original survived.

```
tools/recover_cuts.py orig/host.flac edited/host.flac --json ref-host.json
tools/recover_cuts.py orig/guest.flac edited/guest.flac --json ref-guest.json
```

It ends with a check: the recovered list is spliced out of the original and
compared against the edit. Envelope error well under 0.1 means the cut list is
right. If it is not, nothing downstream means anything — fix that first.

**2. Find out what was cut, not just where.** This decides whether a miss is
tunable at all. A filler the transcript never contained is not a threshold
problem, and no amount of sweeping will find it.

```
tools/what_was_cut.py orig/host.flac edited/host.flac ref-host.json \
    --endpoint http://whisper-host:8081/inference
```

**3. Run the pipeline and score the plan.**

```
clean-podcast.sh --root /tmp/eval
tools/score_plan.py /tmp/eval/work/EPISODE/plan.json \
    host=ref-host.json guest=ref-guest.json
```

Per track: how much of the hand-removed time the plan also removes, which
reference cuts were hit, missed or muted, and which plan cuts have no
counterpart.

**4. Sweep parameters, if the transcript actually contains the material.** Only
the plan stage re-runs, against transcripts a real run already produced, so this
costs no model time.

```
tools/sweep_params.py /tmp/eval/work/EPISODE host=ref-host.json \
    --vary silence_min_duration=1.5,0.5,0.3 \
    --vary silence_keep=0.4,0.25,0.15
```

**4b. Compare against what the pipeline used to produce.** After a change big
enough to be worth doubting — a new transcription engine, a different model — the
question is not only "how close to the hand edit" but "closer than before". This
takes rendered audio, so any previous release, export or experiment can be a
candidate.

```
tools/compare_runs.py --track host \
    --original sample-host.flac --reference sample-host-result.flac \
    --candidate bash=result-whisperhost.flac \
    --candidate whisperx=/tmp/eval/output/ep001/host.flac
```

It reuses `score_plan.py`'s metric rather than defining a second one, so its
numbers can be read beside that tool's. Recovered cut lists are cached, because
recovery is the slow part and it is usually the candidate that changed, not the
reference. Every warning in "Reading the numbers honestly" applies here too, and
the last one especially: a candidate can win on F1 and be worse to listen to.

**5. Read the output.** Transcribe the rendered track and compare it to the
hand edit as text. This is not optional politeness — it is the step that catches
what the score cannot.

```
ffmpeg -i output/EPISODE/host.flac -ar 16000 -ac 1 -y /tmp/out.wav
curl -s -F file=@/tmp/out.wav -F temperature=0 -F response_format=json \
    http://whisper-host:8081/inference
```

## Reading the numbers honestly

**The score can improve while the episode gets worse.** This happened. The LLM
was returning spans covering *every* copy of a repeated word — "pancakes,
pancakes" as one span — so the rendered episode said "we're talking about those
things that you eat", subject deleted. Precision and recall are over removed
*time*, and a span that takes the right region plus one word more still overlaps
the reference well, so the damage scored as ordinary boundary slop. Fixing it
moved F1 *down* from 63.3% to 62.9% and the output from missing three words to
matching the hand edit verbatim. Step 5 is what caught it.

**Precision and recall are over removed *time*, not cuts.** A plan that removes
the right seconds in differently placed spans scores well, which is the intent:
where a cut falls inside a stretch that was silent anyway is not a real
difference.

**Score the talkative track.** Cuts are global, but where a track is silent it
does not matter *where* its share of the removal was taken from — an editor
lumping a guest's silence into one cut at the head is audibly identical to
taking it in pieces. Scoring a mostly-silent track by interval overlap measures
that arbitrary placement and tells you nothing. `--against` picks the track.

**A few points of F1 across a handful of cuts is noise.** The fixture this was
built on has nine cuts in 57 seconds. Sweeping found a 5-point spread across 42
combinations, most of it one knob, and the gap between the best and second-best
settings was well inside what a single differently-placed cut would move.
Confirm on a second episode before believing a parameter change.

**Some gaps are not parameter gaps.** On that fixture the honest split was: two
of nine cuts unreachable at any setting — one a false start Whisper does not
transcribe under any configuration, one a single-track cut that global cuts
cannot express — and the rest closed by fixing the transcript and the speech
map, not by tuning. Step 2 is what tells you which kind you have.
