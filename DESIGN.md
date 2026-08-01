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

### Where it all lives

One root, four directories under it, created on first use:

```
$PODCAST_ROOT/
    incoming/    tracks waiting to be processed
    output/      finished episodes, one directory each
    work/        per-episode intermediates
    failed/      logs, and inputs when FAILED_ACTION="move"
```

`PODCAST_ROOT` defaults to the directory holding `clean-podcast.sh`, so a
checkout is self-contained and runs with no arguments and no config file. That
default suits a workstation, which is now the expected case: with both models
reachable over HTTP there is no reason for the pipeline itself to live on the
server. A server install points the root at its media volume instead, or sets
the four individually when they belong on different ones.

Precedence is *specific beats general, command line beats config file*: a
`INPUT_DIR` setting overrides the root, but `--root` on the command line
overrides that `INPUT_DIR`, since relocating the whole layout is the reason to
pass it. `--input`/`--output`/`--work` alongside `--root` still win, being
equally explicit and more specific.

Creating the tree up front is what lets a missing input directory mean *empty*
rather than *misconfigured* — the run reports nothing to do and exits 0, leaving
an obvious place to put files.

### Per-episode state

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
  out of prose. Because that constraint is load-bearing, its absence is checked
  for rather than assumed — see below.
- Long transcripts go in overlapping windows; findings reported twice are fused
  by index range, keeping the more confident label.
- A window whose response cannot be used is **dropped with a warning, not
  fatal**. Losing a few stutters beats losing the episode.

Cut times come from word boundaries, widened by at most `CUT_PADDING` and never
by more than half the real gap to the neighbouring word — so padding cannot eat
into speech that should stay.

`filler` ("um", "uh") is implemented but excluded from `LLM_ACCEPT_KINDS` by
default; removing every one reads as over-editing.

### The transcript has already removed some of them

Whisper's decoder normalises. Speech synthesised as *"So I I I think … the the
weather"* transcribed as *"So I think … the weather"* — every stutter gone before
the LLM saw a word of it. `uh` survived; the repetitions did not.

Two consequences, and the second is easy to miss:

- **This stage cannot find what never reached it.** Its real yield is bounded by
  what survives transcription, which is less than the raw speech contains. A
  disappointing edit count is not necessarily the model being cautious.
- **Synthetic audio cannot exercise it at all.** No matter how carefully the
  disfluencies are planted, they are gone by the time the words arrive. That is
  why a real recording is kept in `tests/samples/` — see §10.

Nothing is done about the first. Reconstructing what Whisper discarded would mean
distrusting the transcript the rest of the pipeline is built on, and the timings
would be guesswork.

### Nothing here knows which model it is talking to

Both models are named only by a path — `WHISPER_MODEL`, `LLAMA_MODEL` — and the
prompt is plain instructions plus a schema, with no model-specific token or
phrasing in it. Swapping either, or upgrading a version, needs no code change.
Two decisions keep it that way:

- **The LLM is called through `/v1/chat/completions`, not `/completion`.** The
  raw-prompt endpoint applies no chat template, so every model would see an
  instruction formatted for none of them, and how well it coped would be a
  property of the model rather than of this code. Going through the chat
  endpoint hands that job to the server, which knows the template of whatever it
  has loaded. `LLM_API="completion"` reverts to the raw path for a build without
  the chat endpoint, or to compare the two on one episode.
- **Whisper responses are parsed tolerantly** (see §7), which is the same
  argument applied to version drift rather than model choice.

One thing the server may still need told: a llama-server started without a model
path becomes a *router* for several, and refuses any request that does not name
one. `LLAMA_MODEL_NAME` supplies it, empty for the single-model case that ignores
it. That this was missing for so long is instructive — a single-model server
ignores the field, so nothing local ever objected, and neither did stubs written
against the same assumption.

The schema constraint is sent under both the nested OpenAI spelling and the flat
one, because llama.cpp builds have read one or the other. A build reading
*neither* is the dangerous case: unconstrained prose fails to parse, every chunk
is dropped by the rule above, and the run finishes reporting no edits at all —
which is exactly what a clean recording also looks like. `LLM_CHECK_SCHEMA`
therefore spends one small request before the first track confirming the reply
really is constrained, and the stage aborts if not. It is the same reasoning as
the per-chunk tolerance, inverted: dropping a window is survivable precisely
because it is rare, so a fault that would drop *every* window must not be
allowed to look like the survivable case.

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

**A word with timings is not evidence a word was spoken.** On the sample
recording Whisper returned a final `right` spanning 9.94–11.34 s, over audio
whose peak is −50.4 dB — the noise floor. The transcript is a model's output, not
a measurement, and it will place words over near-silence. Everything downstream
treats it as one input among two: the VAD decides where speech is, the transcript
decides what was said, and §8 covers what happens when they disagree.

### Authenticating to either endpoint

Both clients send `Authorization: Bearer <key>` when a key is configured, inline
(`WHISPER_API_KEY`, `LLAMA_API_KEY`) or from a file (the `_FILE` variants, which
are preferred and warn when readable beyond their owner). whisper.cpp has no
auth of its own, so its key is for whatever fronts it; llama.cpp's matches
`--api-key`.

Where the key must *not* end up drove the design. It reaches Python through the
environment, never argv, because a command line is readable by any process on
the machine; and `config_dump` records only whether one was set, because the run
log is copied into the output directory and outlives the episode.

A refusal is treated as fatal rather than transient, which is the opposite of
how this code treats most errors:

- the readiness probes report a 401 immediately instead of polling out their
  timeout, since no amount of waiting fixes a wrong key;
- `detect` propagates `AuthRejected` rather than swallowing it per chunk.

The reasoning is the one in §6: a per-chunk drop is survivable because it is
rare, so anything that would fail *every* chunk identically has to stop the run
instead of quietly producing an episode with no edits. The schema check exists
for the same reason and aborts the same way — exit 2 for a refused key, exit 3
for an ignored schema, each naming the setting to change and the `--from detect`
command to resume with.

## 8. Safety and failure

- A plan removing more than `MAX_CUT_FRACTION` of the episode is refused;
  `--force` overrides. That almost always means a wrong detection threshold.
- Rendered durations are checked against the frame-exact prediction. A mismatch
  fails the run.
- A cut that swallows transcribed words no edit asked for is warned about, and
  the words are listed in `plan.json` under `words_lost_to_silence`. The VAD and
  the transcript are two independent opinions about where speech is, and nothing
  else compares them — the published transcript is rebuilt from the rendered
  timeline, so a word cut away disappears from it and the output stays
  self-consistent. That self-consistency is exactly what hides the
  disagreement. Either side can be wrong (a level-based VAD misses quiet speech;
  Whisper places words over near-silence), so this reports and does not correct.
  A word must be more than half swallowed to count: cut padding grazing its
  neighbour is normal.
- Rejected up front: mismatched sample rates, tracks from two episodes,
  duplicate participants, unparseable filenames. A missing input directory is
  *not* an error — it is created, and an empty one simply means nothing to do.
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
10. **A fault that would affect every chunk identically stops the run.** Per-item
    tolerance — dropping a window whose response cannot be used — is only safe
    because it is rare. A refused key or an unhonoured schema would drop all of
    them and finish reporting no edits, which is indistinguishable from clean
    speech, so both abort instead. Any new whole-run failure mode belongs here
    rather than in the per-chunk `except`.

## 10. Testing strategy

```sh
python3 tests/test_pipeline.py    # 88 tests, 13 classes, ~14 s, stdlib only
./tests/selftest.sh               # 64 checks, 10 cases, ~17 s, ffmpeg only
```

### What makes this awkward to test

The characteristic failure here is not a crash. It is **a valid output file, of
plausible length, containing wrong audio.** A cut expression with a mistake in it
still produces a well-formed FLAC that ffmpeg is perfectly happy with; a mute
applied to the wrong track still yields two files of identical length. Nothing
raises, nothing exits non-zero, and the damage is only apparent on listening —
by which point the inputs have been deleted.

Everything below follows from that. The strategy is not "cover the code"; it is
**make wrongness observable**, and prefer a measurement over an assertion about
what a tool probably does.

A secondary constraint shapes it too: development happens on a machine with
neither model installed. Both suites therefore run with no configuration, no
models, and nothing on the network.

### Three layers

| Layer | Runs | Needs | Catches | Cannot catch |
| --- | --- | --- | --- | --- |
| Unit — `tests/test_pipeline.py` | pure decision code | python3 | wrong intervals, bad validation, malformed expressions | anything about what ffmpeg or a model actually does |
| Stubbed integration — same file + `tests/stub_servers.py` | real HTTP clients against fake servers | python3 | wrong request payloads, bad response handling, retry behaviour | whether a real server would answer that way |
| End to end — `tests/selftest.sh` | the real pipeline over synthetic audio | ffmpeg | wrong rendered audio, wrong stage wiring, wrong file layout | model quality, real-world audio, long-file behaviour |

The unit layer is where a bug should be reproduced if it possibly can be: it is
fast, needs no audio, and a failure points at one function. The end-to-end layer
exists for the claims that only ffmpeg can settle.

### The four techniques doing the real work

**1. The generated ffmpeg expressions are evaluated, not eyeballed.**
`eval_expr` in the test file is a ~15-line interpreter of exactly the grammar
`render.py` emits — `if`, `lt`, `between`, `clip`, arithmetic. Each generated
expression is then compared against an independent Python statement of the
intent, across thousands of sample points:

```python
for step in range(0, 6000):
    t = step / 100.0
    expected = any(s <= t <= e for s, e in
                   ((c["start"], c["end"]) for c in cuts))
    self.assertEqual(bool(eval_expr(expression, t)), expected)
```

The interpreter also **rejects any symbol it does not know**. If the generator
starts emitting a new function, the test fails rather than quietly evaluating
something it has never checked. This is the single most valuable test in the
suite, because it covers the component whose failure is least visible.

**2. The rendered audio is measured, not just its metadata.**
`peak_db` reads `volumedetect` over a named window of an output file. The
crosstalk case does not merely assert two equal durations — it asserts that
across the muted span the affected track reads −91 dB *and the other speaker
reads −8 dB at that same instant*. A test comparing only durations would pass
with entirely wrong audio, which is precisely the failure mode that matters.

**3. Both models have stubs, so all eight stages can run.**
`tests/stub_servers.py` impersonates `whisper-server` and `llama-server`: canned
replies handed out in order, and a request log the test asserts against. That
gives coverage of the whole pipeline including `transcribe` and `detect`, and
lets the assertions reach the *payload* — that the multipart body really carries
the audio, that the request went to the chat endpoint with the edit schema in
`response_format`, that the transcript reached the message content — rather than
only checking that nothing blew up. The stubs answer in the envelope matching
the endpoint that was called, so a client posting to the wrong one is not
rewarded with a well-formed reply.

**4. Predict, then verify — no tolerances where an exact answer exists.**
`expected_output_samples` replicates ffmpeg's per-frame decision, and the
self-test compares the real rendered file against that prediction rather than
against a rule of thumb. The prediction itself is cross-checked in the unit
suite by brute-force simulation through `eval_expr`, so the two implementations
have to agree. In practice it lands within a millisecond.

### Measure, don't assume

Where behaviour of an external tool matters, it gets measured and the number
recorded, rather than asserted from memory. Two cases in this codebase:

- Mixed-format sync was going to carry a warning. A click at exactly 5.000 s
  round-tripped through five codecs came back at the identical sample every
  time, so the warning was dropped. An assumption would have produced a caveat
  that was simply false. That measurement is now a self-test check in case 8,
  since a regression in it would appear as silent misalignment rather than as an
  error.
- Container durations were assumed accurate. Measurement found AAC decoding
  +5.33 ms and Opus −6.5 ms against their headers, which is what prompted
  `meta-refresh`.

When adding a claim about ffmpeg, whisper.cpp or llama.cpp, measure it first and
put the figure in the test or its comment. When the design *depends* on the
claim, make the measurement a check — the assumption is then load-bearing, and
load-bearing assumptions deserve a guard.

### Keeping the arithmetic hand-checkable

The synthetic episodes use deliberately round numbers, so every expected value
can be derived on paper and the test asserts a figure rather than whatever the
code happened to produce. Case 2, for instance: a 10 s gap with
`SILENCE_KEEP=0.4` must yield a 9.6 s cut, and 30 s minus that cut minus a
3.75 s tail must leave 16.65 s. Both are checked as numbers.

This matters because a test that asserts the current output is not a test — it is
a change detector. When one of these fails, the arithmetic in the comment says
which of the two is wrong.

### What is deliberately not covered

- **Real Whisper and real llama.cpp.** Their output is neither cheap nor
  deterministic, and what needs testing is our handling of it, not their quality.
  The consequence: the flags passed to `whisper-cli`, and llama-server's launch
  arguments, are exercised by nothing. Those are the lines to re-read by hand.

  The sharper consequence is that **the stubs accept whatever the client sends**,
  because both sides were written here. A green suite proves the request has the
  shape we intended, never that a real server agrees — which is exactly why the
  `response_format` spelling is sent twice and why `LLM_CHECK_SCHEMA` exists. A
  first run against a real server remains the only evidence that the wire format
  is right, and the schema check is what makes that run fail loudly instead of
  silently.
- **Silero VAD** — torch is absent from the test environment. The ffmpeg backend
  is covered; the Silero path is not.
- **Real audio.** Synthetic tracks are sine bursts against digital silence, so
  every level is unambiguous and every word is invented. Neither the `detect`
  stage nor anything depending on real acoustics can be reached that way. A
  recording is kept in `tests/samples/` for manual runs; nothing automated
  consumes it, so the suites stay offline.
- **Whether the edit sounds good.** Not a testable property. That is what the
  edit report and a listen are for.
- **Long-file behaviour.** Synthetic episodes are 10–30 s. Nothing here would
  catch a filter graph that is correct but unusably slow across two hours, or a
  memory problem that only appears at scale.
- **Concurrency.** `FFMPEG_JOBS > 1` and `WHISPER_JOBS > 1` paths are not
  exercised; the suites run serially.

### What the first real run found

The suites were green throughout. The first run against a real whisper-server and
a real llama-server, on a real recording, found four things — worth listing
because they share a shape:

1. **The client never sent a `model` field.** A single-model server ignores it, so
   the assumption held everywhere it was tested; a router-mode server refuses.
2. **The VAD and the transcript disagreed**, and nothing compared them. Quiet
   speech at −39.2 dB fell under the −35 dB threshold and was cut while Whisper
   had transcribed it. Now warned about (§8).
3. **The default `SILENCE_THRESHOLD` cut through the middle of the speech.** A
   threshold belongs below the quietest speech and above the noise floor; this
   recording put speech at −21 dB and −39 dB with a floor near −50 dB, and
   `-35dB` fell *between the two kinds of speech*. Lowered to `-45dB`, which
   lands in the intended band and errs toward leaving room tone rather than
   cutting speech — those two mistakes do not cost the same.
4. **The lost-word warning gave advice for the wrong backend**, telling a Silero
   run to try Silero.

Every one of them was invisible to the suites for the same reason: *both sides of
each test were written here*. The stubs answer as the client expects, synthetic
audio has no quiet speech, digital silence has no noise floor, and a fixture
never runs with a configuration nobody thought to write down. Passing tests
constrain the code to the author's model of the world, and these were all places
where that model was wrong rather than where the code was.

The response is not more stubs. It is the schema check, the lost-word warning,
and a real recording to run by hand — each of which turns a silent wrong answer
into a loud one at the point where reality first disagrees.

### Invariants and where they are guarded

Cross-reference for [§9](#9-invariants):

| Invariant | Guarded by |
| --- | --- |
| 1. identical output lengths | selftest cases 1, 6, 7, 8, 10; unit test on filter ordering |
| 2. mutes never change the timeline | selftest cases 6, 7 (durations plus measured audio) |
| 3. no cut overlaps speech | selftest case 1, checked against the VAD output |
| 4. `keep` complements `cuts` | `TestPlanBuilder.test_keep_and_cuts_are_complementary` |
| 5. no unvalidated LLM number reaches audio | `TestLlmValidation`, `TestAgainstStubServer` |
| 6. local models never overlap | **nothing** — structural only, read the two stages |
| 7. inputs deleted only after verification | selftest cases 1 (deleted), 3 (preserved on failure) |
| 8. transcript matches the rendered audio | `TestFinalTranscript`, selftest case 6 |
| 9. failure is recoverable | selftest case 3 (work dir kept), 6 and 7 (resume via `--from`) |

Invariant 6 is the gap worth remembering. It cannot be observed without loading
two real models, so it is held structurally instead: `stage_transcribe` waits for
every process to exit, and `llama_start`/`llama_stop` both live inside
`stage_detect`. **Anyone touching either stage has to verify it by reading.**

### Adding a test

- A wrong editing decision → `tests/test_pipeline.py`, reproduced as an interval
  or word-list fixture. No audio required, and the failure localises.
- A wrong thing done *to the audio* → `tests/selftest.sh`, with the output
  inspected via `peak_db` or `duration_of`, not merely asserted to exist.
- A new external-tool assumption → measure it, then encode the figure.
- Keep expected values derived, not observed.

The self-test sandboxes itself under `mktemp -d` and cleans up; run with
`KEEP_SANDBOX=1` to keep the work directories, rendered files and stub request
logs for inspection.

### Before trusting a change

1. Both suites pass.
2. If `transcribe` or `detect` was touched: **one real episode with
   `--keep-work`.** Nothing automated covers the real models.
3. Read `<episode>_edit-report.txt` — cut counts and removed fraction are the
   cheapest signal that a threshold has drifted.
4. If the change was anywhere near rendering, sample the audio, and check the
   invariants table above for what is guarding you.

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
- Whisper removes some disfluencies during transcription, so `detect` can only
  work on what survives (§6). Its yield is lower than the raw speech would
  suggest, and nothing recovers the difference.
- The VAD and the transcript can disagree about where speech is. The
  disagreement is reported, never reconciled — neither side is reliable enough to
  overrule the other (§8).
- `SILENCE_THRESHOLD` is a single level for a whole episode. A recording whose
  loudness varies across it has no single right answer; `VAD_BACKEND="silero"`
  judges speech instead and is the better tool there.
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
| `SILENCE_THRESHOLD` | ffmpeg backend only; errs low because cutting quiet speech is damage while keeping room tone is only a looser edit |
| `LLAMA_MODEL_NAME` | required by a router-mode server, ignored by a single-model one |
| `SILENCE_MIN_DURATION` | how long a gap must be before it is worth shortening |
| `SILENCE_KEEP` | quiet left behind in place of a shortened gap |
| `CUT_PADDING` | speech margin kept either side of every cut |
| `RENDER_FRAME_SAMPLES` | timing granularity of the whole edit; must be uniform across an episode |
| `MAX_CUT_FRACTION` | refusal threshold, not a target |
| `WHISPER_CHUNK_SECONDS` | upload chunk size for remote transcription; 0 sends the lot |
| `PODCAST_ROOT` | the whole layout; the four directory settings override it individually |
| `LLM_API` | `chat` lets the server apply the model's template; `completion` is the raw-prompt fallback |
| `LLM_CHECK_SCHEMA` | one request that turns a silent whole-run failure into an immediate one |
| `*_API_KEY_FILE` | preferred over the inline form; the key never reaches argv or the log |
