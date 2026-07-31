# System description

How the pipeline is put together and why it makes the choices it does. The
README is the operating manual; this is the reference for changing the thing.

**Keep this updated.** When behaviour changes, the sections most likely to go
stale are [Cuts and mutes](#cuts-and-mutes), [The data flow](#the-data-flow) and
[Invariants](#invariants) — the last of these is the list a change can quietly
break without any test noticing.

---

## 1. The problem

Several audio tracks, one per participant, from a single ~2 h podcast recording
(FLAC in practice, but any format ffmpeg decodes). They are already
synchronised: sample 0 of each is the same instant. Wanted, per episode:

1. dead air shortened, where *nobody* is speaking;
2. stutters, accidental repetitions and false starts removed;
3. the tracks still separate, still in sync, ready for mixing;
4. everything else — inputs and intermediates — gone at the end, logs kept.

Constraint from the operator: Whisper and the LLM must never be resident in
memory at the same time.

## 2. Shape of the solution

```
                 ┌─────────────────────── clean-podcast.sh ──────────────────────┐
                 │  CLI · config · stage sequencing · failure handling           │
                 └───┬───────────────────────────────────────────────────────┬───┘
                     │ drives                                                │ calls
        ┌────────────▼────────────┐                        ┌─────────────────▼──────────────┐
        │ ffmpeg  whisper  llama  │                        │ python/cleanup_cli.py          │
        │ (audio and the models)  │                        │ + python/cleanup/*  (decisions)│
        └─────────────────────────┘                        └────────────────────────────────┘
```

The split is deliberate and worth preserving: **the shell touches audio and
processes, the Python decides what to do and never touches audio.** Every
interesting judgement — interval algebra, which findings to trust, what to cut —
is therefore reachable from a unit test without any external tool.

The draft this replaced generated Python inside unquoted shell heredocs, so
every `$` in a regex was a hazard and nothing was testable. There are no
heredocs in the pipeline now; Python lives in `.py` files and is invoked with
argv.

## 3. Cuts and mutes

This is the core of the design. Two operations come out of the analysis:

| | scope | timeline | used for |
| --- | --- | --- | --- |
| **cut** | every track, same range | shortened | over-long silence; a disfluency with no one else speaking |
| **mute** | one track | unchanged | a disfluency overlapping another participant's speech |

The mute exists because of a conflict that has no good answer otherwise. Cuts
have to be global — cutting one track alone would desynchronise it from the
rest. But a stutter often happens while somebody else is talking, and cutting
globally there takes a bite out of the other speaker. So that case is not cut at
all: the stutter is silenced on its own track, with a short fade at each end, and
the timeline is left exactly as it was. Nobody else is damaged, and sync is free.

The classification, in `plan.py`:

```
for each LLM finding on track T:
    others = union of speech intervals of every track except T
    if the finding's time range overlaps `others`:  →  mute on T
    else:                                           →  global cut
```

Silence is **shortened, not removed**. A gap where no track has speech, longer
than `SILENCE_MIN_DURATION`, has its middle removed and `SILENCE_KEEP` of quiet
left behind. Removing it entirely makes conversation sound gasped.

## 4. The data flow

Everything lives in `$WORK_ROOT/<episode>/` and is JSON, so any stage can be
re-run by hand.

```
inputs/<episode>_<participant>.<ext>       any format ffmpeg can decode
   │
   ├─ discover ──→ meta.json         participants, codecs, rates, durations
   │
   ├─ prepare ───→ prep/<p>.wav      16 kHz mono, what Whisper and Silero want
   │               meta.json         durations replaced with measured ones
   │
   ├─ vad ───────→ vad/<p>.json      {"speech": [[start, end], ...]}
   │
   ├─ transcribe → words/<p>.words.json   words with timings, and segments
   │                                      (local whisper-cli, or a server)
   │
   ├─ detect ────→ llm/<p>.edits.json     validated findings, word-index based
   │               llm/<p>.audit.jsonl    every response, accepted or not
   │
   ├─ plan ──────→ plan.json         cuts, mutes, keep-list, stats, warnings
   │               expected.json     frame-exact predicted output length
   │               render/<p>.filter one ffmpeg filtergraph per track
   │
   ├─ render ────→ output/<ep>/.staging/<p>.flac   then verified
   │
   └─ finalize ──→ output/<ep>/      tracks, transcripts, plan, report, logs
                                     work dir and inputs removed
```

Stage completion is recorded in `state/<stage>.done` and per-item work in
`state/<stage>-<participant>.ok`, which is what makes `--from`, `--only` and
`--stages` work without special cases. A resumed run rebuilds its context from
`meta.json` via the `meta-shell` subcommand, which emits shell assignments
through `shlex.quote`.

## 5. Rendering

One ffmpeg pass per track:

```
[0:a] aresample? → asetnsamples → volume (mutes) → aselect (cuts) → asetpts → [out]
```

- `aresample` is only present when `RESAMPLE_TO` is in play, and it must come
  **first**. Cuts are decided per frame, so resampling on the output side
  instead would leave each track chunked at its own rate, and one cut list would
  remove slightly different spans from each — the one arrangement that breaks
  sync. There is a test asserting this ordering.
- `asetnsamples=n=RENDER_FRAME_SAMPLES` fixes the frame size next, because
  everything downstream decides per frame.
- `volume=eval=frame` with a gain expression applies the mutes. Mutes are fused
  when closer together than twice the fade, so their trapezoidal ramps can never
  overlap and the summed gain stays within [0, 1].
- `aselect` drops the frames inside a cut, `asetpts=N/SR/TB` re-times what
  survives.
- The graph goes to a file and is passed with `-filter_complex_script`, so its
  length is not an argv problem.

Both expressions are emitted as a **balanced decision tree**
(`if(lt(t,pivot), left, right)`) rather than a flat sum of `between()` terms. A
2 h episode can carry several hundred cuts, and a flat expression would be
evaluated in full for every one of a million-odd frames; the tree makes it a
handful of comparisons.

### Why cuts are frame-aligned, and why that is fine

`aselect` decides per frame, not per sample, so a cut boundary lands on a frame
edge — about 11 ms at the default 512 samples. Inaudible for silence and
stutters. What matters is that **all tracks share a frame size and an identical
select expression, so they drop exactly the same frames** and stay
sample-aligned. That is also why tracks of one episode are required to share a
sample rate: at different rates the same frame count is a different duration, and
the error would compound a few milliseconds per cut into seconds of drift.

Being deterministic, the output length is *predictable*, so
`render.expected_output_samples` replicates the frame decision and verification
compares against an exact figure rather than a tolerance. In the self-test the
prediction lands within a millisecond of the rendered file.

That prediction depends on knowing the true input length, which **is not what the
container says**. Measured on this machine: AAC decodes 5.33 ms longer than its
container claims, and Opus's container claims 6.5 ms more than it decodes. A
truncated file of any format can claim anything. So the `prepare` stage — which
has just fully decoded every track anyway — hands its output to `meta-refresh`,
and the decoded length replaces the container's figure for everything downstream.
`container_duration` is kept for reference. The error was within the 50 ms
verification tolerance, so nothing was failing; this keeps the word "exact"
honest.

### Input and output formats

`INPUT_EXTS` is only a discovery filter — `prepare` decodes to PCM, so the
container never reaches the editing logic. `OUTPUT_CODEC` / `OUTPUT_EXT` are
independent of it, defaulting to FLAC. `TRACK_EXT` used to mean both at once and
survives as a deprecated input-only alias.

Mixed formats within one episode are safe. This was checked rather than assumed:
a click placed at exactly 5.000 s round-trips through FLAC, MP3, AAC, Opus and
Vorbis and comes back at the identical sample in every case, because ffmpeg's
decoders account for encoder delay and pre-skip. What is *not* safe is mixed
sample rates, for the frame-quantisation reason above — refused unless
`RESAMPLE_TO` is set.

The render's "nothing to change" shortcut copies the source file only when it is
already in the requested codec and extension; otherwise a track with no edits is
still transcoded. Getting this wrong would produce a file named `.flac`
containing MP3.

Sample-accurate alternatives were considered and rejected: `atrim` with
`start_sample` is exact but needs `asplit` into one branch per kept segment,
which makes ffmpeg buffer most of the file; splitting to hundreds of pieces and
concatenating means hundreds of processes per track.

## 6. Keeping the LLM honest

A model asked for timestamps will confidently invent them, and a fabricated
number becomes a cut in the wrong place. So it is never given or asked for one.

- Whisper's word timings stay on our side. The prompt contains a **numbered word
  list**; the reply is word-index ranges.
- `llm._validate` rejects anything out of the window, inverted, longer than
  `LLM_MAX_EDIT_WORDS`, spanning more than `LLM_MAX_EDIT_SECONDS`, of an
  unaccepted kind, or below `LLM_MIN_CONFIDENCE`. Each rejection is recorded with
  its reason in the audit log.
- Responses are constrained by a JSON schema server-side, so nothing is scraped
  out of prose.
- Long transcripts go in overlapping windows; findings reported twice are fused
  by index range, keeping the more confident label.
- A window whose response cannot be used is **dropped with a warning, not
  fatal**. Losing a few stutters beats losing the episode.

Cut times come from word boundaries, widened by at most `CUT_PADDING` and never
by more than half the real gap to the neighbouring word — so padding cannot eat
into speech that should stay.

`filler` ("um", "uh") is implemented but excluded from `LLM_ACCEPT_KINDS` by
default; removing every one reads as over-editing.

## 7. Model placement and memory

Whisper and the LLM are configured independently. Empty `WHISPER_ENDPOINT` or
`LLAMA_ENDPOINT` means local; a URL means an existing server is used as-is and no
process is ever spawned or stopped. All four combinations are supported and the
arrangement is printed at the start of every run.

The memory constraint is honoured structurally rather than by hoping:

- stages are strictly sequential;
- `transcribe` waits for every Whisper process to exit — including background
  ones when `WHISPER_JOBS > 1` — before returning, and logs that it has;
- `llama_start` and `llama_stop` are both inside `stage_detect`, so the server's
  lifetime is that stage and not the run. `llama_stop` also runs from the exit
  trap, so a crash or Ctrl-C does not leave a model resident.

This only covers processes the script manages. Two remote endpoints sharing a
machine are that machine's problem, and the run warns rather than implying a
guarantee it cannot make.

### Remote transcription

`asr.py` POSTs the prepared 16 kHz WAV to `/inference` as multipart and converts
the reply into the same segment shape `whisper-cli` produces, so
`transcript.build_from_segments` reassembles words from one code path either way.

Two wrinkles. **Chunking**: a 2 h track is ~230 MB and one request means no
progress for as long as it takes, so it is split — with boundaries nudged onto
the middle of a silence the VAD stage already found, which is why `transcribe`
runs after `vad`. **Tolerant parsing**: whisper-server's response shape varies by
version and `response_format`, so several are accepted (`segments` with float
seconds, with clock strings, or whisper-cli's `transcription` with `offsets`;
token lists are used when they carry timings and ignored when they are bare ids).
What is *not* tolerated is a response with no timings at all — that raises rather
than guessing.

Word timing quality differs: `max_len=1` is requested so each segment is one
word and timings are exact, but a build that ignores it returns sentence
segments whose word positions are interpolated. The run says which it got,
because it decides how tightly a stutter can be cut.

## 8. Safety and failure

- A plan removing more than `MAX_CUT_FRACTION` of the episode is refused;
  `--force` overrides. That almost always means a wrong detection threshold.
- Rendered durations are checked against the frame-exact prediction. A mismatch
  fails the run.
- Rejected up front: mismatched sample rates, tracks from two episodes,
  duplicate participants, unparseable filenames, a missing input directory.
- Outputs are rendered into `output/<episode>/.staging/` and moved into place
  only once all are verified — a rename within one filesystem, so it is atomic
  and not a second copy of a gigabyte.
- **Inputs are deleted only after that.** Any failure leaves them untouched,
  keeps the work directory, writes a `FAILED` marker and copies the log to
  `FAILED_DIR`. `FAILED_ACTION="move"` additionally parks the inputs there, so an
  unattended run cannot retry the same broken episode forever.
- `KEEP_INPUTS` / `--keep-inputs` and `KEEP_WORK` / `--keep-work` opt out of the
  deletion entirely; `--dry-run` implies both.

## 9. Invariants

The properties a change must not break. Most have a test; the ones that do not
are marked.

1. **All output tracks of an episode have identical length.** Guaranteed by
   global cuts, one frame size, one shared cut expression, and — when resampling
   — `aresample` sitting ahead of `asetnsamples` so every track is chunked at the
   same rate.
2. **Mutes never change the timeline.** They are `volume`, never `aselect`.
3. **A cut never overlaps any track's speech**, beyond the deliberate
   `CUT_PADDING`.
4. **`keep` is exactly the complement of `cuts`** over `[0, duration]`, and
   `output_duration == total(keep)`.
5. **No LLM-supplied number reaches the audio unvalidated.** Indices are checked
   against the transcript; timings come from Whisper.
6. **No locally managed Whisper process overlaps a locally managed llama server.**
   *(Structural — no test; check `stage_transcribe` and `stage_detect` when
   touching either.)*
7. **Inputs are deleted only after every output is verified.**
8. **The transcript describes the audio that now exists** — words removed by a
   cut or silenced by a mute are gone from it, and timestamps are on the rendered
   timeline.
9. **Failure is recoverable**: the work directory plus `--from` reproduces the
   rest of the run.

## 10. Testing

`python3 tests/test_pipeline.py` — 85 unit tests, stdlib only. Interval algebra,
transcript parsing (including the fallback for builds with dead token timings),
LLM response validation, remote-ASR parsing and chunking, the cut/mute decision,
and the generated ffmpeg expressions. Those last are checked by **evaluating them
with a miniature interpreter of the grammar we emit** — a wrong expression would
mangle audio without failing anything else. The llama client is exercised against
a stub HTTP server: schema payload, retries, malformed responses, dedupe.

`./tests/selftest.sh` — 63 end-to-end checks needing only ffmpeg. Synthetic
episodes with silences in known places, run through the real pipeline in a
sandbox, with the *rendered audio* inspected. Cases: sub-threshold gaps left
alone; a long gap shortened to its residual; the safety limit refusing and
`--force` overriding; a dry run touching nothing; mixed sample rates rejected; a
stutter over crosstalk muted with the other speaker measurably intact; the full
eight stages against stub whisper and llama servers; a mixed-format episode
(WAV + AAC in, FLAC out) staying length-identical and matching its prediction;
the same participant in two formats refused; and mixed rates working under
`RESAMPLE_TO=auto`.

Neither suite runs a real model. **Run one real episode with `--keep-work` after
changing anything in `transcribe` or `detect`.**

## 11. Known limitations

- Tracks must already be synchronised at sample 0. Nothing here aligns them.
- All tracks of an episode must end up at one sample rate — either they already
  agree, or `RESAMPLE_TO` converts them.
- Cut boundaries are frame-quantised (~11 ms by default).
- Lossy input is accepted but cannot be improved: what the codec discarded stays
  discarded, and a FLAC output of it is merely a larger file.
- One episode per run. Several in `INPUT_DIR` at once is an error, not a queue.
- Mute rendering evaluates a per-frame expression over the whole track; with
  very many mutes on a long track it is the slowest part of a render.
- Silero holds a torch model and so runs one track at a time regardless of
  `FFMPEG_JOBS`.
- `filler` detection exists but is off by default.
- Speaker attribution comes from track membership, not diarisation: bleed of one
  voice into another's mic is not separated, though the mute-over-crosstalk rule
  means it does no damage.

## 12. Configuration reference

`podcast-cleanup.conf.example` documents every setting inline and is the
authority. The ones whose meaning is easy to get wrong:

| Setting | Meaning |
| --- | --- |
| `INPUT_EXTS` | a discovery filter only; the format never reaches the editing logic |
| `OUTPUT_CODEC`/`OUTPUT_EXT` | the output format, unrelated to what came in |
| `RESAMPLE_TO` | empty means a rate mismatch is an error, not that nothing happens |
| `VAD_MIN_SILENCE` | detection *granularity* of the speech map, not the editing threshold |
| `SILENCE_MIN_DURATION` | how long a gap must be before it is worth shortening |
| `SILENCE_KEEP` | quiet left behind in place of a shortened gap |
| `CUT_PADDING` | speech margin kept either side of every cut |
| `RENDER_FRAME_SAMPLES` | timing granularity of the whole edit; must be uniform across an episode |
| `MAX_CUT_FRACTION` | refusal threshold, not a target |
| `WHISPER_CHUNK_SECONDS` | upload chunk size for remote transcription; 0 sends the lot |
