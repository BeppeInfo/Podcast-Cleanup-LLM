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

Both models are reached over HTTP. Whether they fit in memory together is the
business of whatever machine serves them; see §7 for what that gave up.

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
   ├─ prepare ───→ prep/<p>.wav      16 kHz mono, what Whisper wants
   │               meta.json         durations replaced with measured ones
   │
   ├─ transcribe → words/<p>.words.json   words with timings, and segments
   │                                      (one request per chunk, per track)
   │               asr/<p>.loud.json      the level scan: chunk boundaries, and
   │                                      the only opinion here that is not
   │                                      Whisper's (§8)
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

### Word-like crutches are a different problem, and are not covered

`filler` means the non-lexical sounds only — the guidance the model receives is
*"a hesitation sound carrying no meaning"*. Crutches built from real words —
*"well…"*, *"you know"*, *"like"*, *"I mean"*, pt-BR *"tipo"*, *"sabe"*, *"né"* —
are **not** detected, and switching `filler` on does not start detecting them.

Considered and deferred, not overlooked. The model would likely be good at it.
Unlike the four mechanical kinds, this is a semantic judgement, which is what a
language model is actually for: every candidate word has a load-bearing sense —
*"the well ran dry"*, *"you know what I mean?"*, the quotative *"he was like,
'no way'"* whose removal collapses the sentence — and no word list resolves
that. Being real words, they also survive transcription, so the yield ceiling
described below does not apply: there is more to find here than there is for
stutters, not less.

Three things stand in the way, and none of them is the code. Adding a kind is
`KINDS` plus a `KIND_GUIDANCE` entry — the schema enum, the prompt and the
validator all derive from that list, and `LLM_ACCEPT_KINDS` already supplies the
switch.

- **The confidence question would be the wrong question.** The prompt asks how
  sure the model is that a span was *accidental*. A crutch is not accidental —
  *"you know"* is deliberately uttered as a hedge — so the honest answer is low,
  and `LLM_MIN_CONFIDENCE` would filter away precisely the correct findings. The
  question has to become whether removal leaves the meaning intact, and that is
  a change to the shared prompt, so it lands on the four existing kinds too.
- **`CUT_PADDING` assumes a gap that is not there.** Padding claims at most half
  the real distance to the neighbouring word (above), which works because a
  stutter or a false start sits in a natural break. These do not: *"it was
  y'know weird"* is one coarticulated breath group with no gap to claim, so
  padding computes to nearly nothing and the cut lands mid-consonant. *"Well,"*
  opening a clause carries a pitch reset as well, and taking it makes what
  follows start abruptly. Cheap in code, expensive in audio — this is the real
  obstacle.
- **Density defeats the safety net.** *"You know"* can recur every few seconds.
  `MAX_CUT_FRACTION` would not notice, because the seconds stay small; what
  accumulates is the count of splices, not the total removed.

If it is ever taken up, `detect` is a standalone subcommand and `--audit` records
the raw reply per chunk, so a new kind can be run against a transcript that
already exists and read as text — no re-transcription, no render, and no audio
touched until the model has shown it is any good at this on real material.

### A repetition has a survivor, and the model will take it

"pancakes, pancakes" is two words in the transcript and one word in the
sentence. The edit is to remove *a* copy, not the span — and the model does not
reliably see the difference. Asked about that pair it returned both indices;
about "would, would, would" all three; about "and, and" both. Rendered, that is
an episode saying "we're talking about those things that you eat" with the
subject deleted, which is worse than leaving the stumble in.

The prompt already said to keep the completed attempt and showed it in the
example. That was not enough, so the rule is now stated for the repeated case
specifically — *the later version is the one that survives* — and enforced
regardless. `transcript.spare_the_survivor` hands back the last copy of a span
that is one word repeated. It fires only in that unambiguous case: a repeated
*phrase* is left alone, because which copy survives cannot be read off safely,
and a filler is left alone because it has no survivor — both halves of "um, um"
should go.

Enforced in the plan stage rather than only where the reply is validated. An
edits file can be resumed from an earlier run or written by hand, and a span
that takes every copy damages the episode however it arrived. The LLM stage
applies it too, so the audit records what the model actually asked for; the
operation is idempotent, so doing it twice changes nothing.

**What this cost to find, and the lesson for measuring.** The over-wide spans
were invisible in the scoring: precision and recall are computed over removed
*time*, and a span that removes a correct region plus one word more still
overlaps the reference well. Fixing it moved the F1 on the fixture from 63.3%
*down* to 62.9% while the rendered episode went from missing three words to
matching the hand edit verbatim. A transcript of the output is the check that
catches this; the score is not.

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

Reconstructing what Whisper discarded after the fact would mean distrusting the
transcript the rest of the pipeline is built on, and the timings would be
guesswork. So nothing is reconstructed. What is done instead is to stop it being
discarded in the first place, which costs nothing downstream because the words
arrive through the ordinary path with the server's own timings.

**The prompt is what does it.** `WHISPER_PROMPT` is whisper's initial prompt:
conditioning text seeded into the decoder as though it were the transcript
preceding the audio. It is not an instruction and nothing in it is obeyed — the
decode simply carries on in the register it was handed, so a prompt full of
fillers biases it towards the fillers that were actually said. Wording it as a
request does nothing; the fillers themselves are the mechanism.

Measured on the two-track sample fixture, against an edit cut by hand. Six
disfluencies were removed by hand; with no prompt, five of them were absent from
the transcript altogether, and the LLM stage returned `{"edits": []}` for both
tracks — not caution, blindness. With a prompt, five of the six came back
(*"and, um, um"*, *"pancakes, pancakes"*, *"Uh,"*, *"would, would, would"*,
*"and, and"*), and the stage found eight edits. Nothing else moved the needle:
VAD on and off, `max_len=1` on and off, `no_context`, and temperature 0 through
0.4 all returned byte-identical text.

The cost is the mirror image of the gain, and it is real: a decode biased towards
fillers can invent one in silence, and §7 derives the speech map from these
words, so an invented filler becomes speech in the plan. The Silero pass is what
keeps that in check — with VAD on there is no silence for an invented filler to
appear in. A prompt is a bias, not a switch, and it comes with no guarantees in
either direction.

### Re-asking where one word swallowed the audio

The recovery in §7 answers *"loud audio, no words at all"*. `WHISPER_REASK`
answers the graded version: loud audio with implausibly *few* words on it. When
Whisper reads a passage fluently it hangs the whole passage on one word, and that
word ends up carrying seconds of continuous speech. Nothing in the response marks
it, and both stages downstream go blind in the same spot — the LLM stage cannot
cut what is not in the transcript, and the plan stage sees one long word where
there was a pause, so the silence never becomes a gap either.

The trigger is the level scan, not the clock: a word is only questioned when more
than `WHISPER_REASK_WORD_SECONDS` of *measured loud audio* sits inside it. A word
stretched across silence is badly timestamped, not hiding anything, and asking
about it would spend a request to be told the same thing.

Asked again in a short window the same audio comes back verbatim, because there
is no fluent context left to smooth into. The window length is the mechanism, not
the re-asking: re-sending one span as a nineteen-second request returned the same
cleaned-up reading, while five-second windows over the same audio returned
*"Yeah, I wonder what Fairpunk, uh, would, would, would talk about this"*.

A replacement is only accepted where the second pass found strictly more words
than the first. Equal counts are the same reading spelled differently — proper
nouns come back unstable from a short window, the same name arriving as
*Shwereponk*, *SharePunk* and *Fairpunk* — and trading one spelling for another
is churn, not recovery.

On the sample fixture this fired once and correctly declined to replace anything:
with the prompt in place there was nothing left for it to find. It is kept for
the case the prompt does not reach, which is the one it was written for.

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

### Windows may go in parallel, and nothing observable changes

The first real episode made the cost of one-at-a-time obvious: the server was
busy the whole time and the machine was not, because a single request decodes at
batch size 1, which is bound by memory bandwidth rather than by cores.
`LLM_CONCURRENCY` puts several windows in flight against a llama-server given
matching `--parallel` slots.

Nothing about the design resisted this. Windows are planned up front by
`plan_chunks`, none depends on another's answer, and overlaps are reconciled
afterwards by `_dedupe` sorting on index — so the wire order was never load
bearing. What *was* at risk is everything a person looks at afterwards. The
audit JSONL is meant to be diffed against a previous run, and an edit list that
shuffles between runs cannot be. So results are consumed in **submission order,
not completion order**: `detect` blocks on the oldest outstanding window,
validates it, writes its audit line, then advances the progress count. This
costs no throughput, because the window being waited on is itself still in
flight — the same number of requests are outstanding either way.

Two things follow from the same choice:

- **Only `LLM_CONCURRENCY` windows are ever submitted**, topped up one at a time
  as each is consumed, rather than the whole track being queued into the pool.
  That is what keeps a refused key costing one wasted request instead of one per
  chunk. Submitting everything up front broke exactly that, and the test written
  long before this feature — "one attempt, not three" — is what caught it.
- **The default stays 1.** A server with one slot gains nothing, and the setting
  is only correct when it matches how the server was actually launched. Too high
  is not an error: the surplus queues inside the server where the client cannot
  see it, and the run merely looks slow.

The sharp edge is context, not concurrency. Giving llama-server an explicit
`-np` takes the slot count out of "auto", which is also what turns `--kv-unified`
off, so `-c` is split and each slot gets `n_ctx / n_parallel`. Raising the slot
count therefore *shrinks* the window each chunk must fit into, and a chunk that
no longer fits is refused, dropped, and shows up only as a track that found
suspiciously little. `stage_detect` does that arithmetic before the episode and
warns — but only for a server it starts itself, since a remote one's `-c` and
`-np` cannot be read back.

## 7. Model placement

Both models are servers someone else runs. `WHISPER_ENDPOINT` and
`LLAMA_ENDPOINT` are required — the second only when `LLM_ENABLE=1` — and
nothing here starts, stops, waits on or otherwise manages a model process.
Pointing them at `127.0.0.1` is how a single-machine install works; that is a
deployment choice, and this side cannot tell the difference.

**There used to be a local mode**, and most of §7 used to be about defending it.
`whisper-cli` was run per track and `llama-server` was started and stopped inside
`stage_detect`, because the two must never hold memory at once on a machine that
cannot fit both. That constraint drove real structure: `llama_stop` in the exit
trap so a Ctrl-C could not leave a model resident, `pool_wait` proving every
Whisper process had exited before `detect` began, `WHISPER_JOBS`, `LLAMA_NGL`,
`LLAMA_CTX`, and a warning that computed whether `-c` divided by `--parallel`
still left room for a chunk.

All of it is gone. The reasoning: it was two ways to do one thing, and the
second way is strictly more capable — a server on `127.0.0.1` does everything a
subprocess did, and can be shared, restarted, swapped or moved to another
machine without this script knowing. Keeping both meant every stage carrying a
branch, fourteen settings existing only to construct command lines, and a
process lifetime to get right on every failure path.

**What was genuinely lost, and is worth stating plainly.** The memory constraint
was previously enforced structurally, and now it is not enforced at all. If both
servers share a machine that cannot hold both models, that machine will swap or
OOM, and nothing here will prevent it or explain it. That was already true for
anyone using two endpoints, which is why the old code warned about it; the
change is that it is now the only mode. It is the right trade for this project —
the models have lived on `choppaserver` for a while — but a single-machine user
with tight RAM has to sequence the two servers themselves.

The context-size check went the same way. It could compute `-c / --parallel`
only because it was the thing passing those flags. Against a server it did not
start, the arithmetic is unavailable, so the guidance moved into the config
comment and the README sizing table. A chunk that does not fit is still refused
silently and still shows up only as a track with suspiciously few edits — there
is simply no longer anything positioned to warn first.

### Remote transcription

`asr.py` POSTs the prepared 16 kHz WAV to `/inference` as multipart and converts
the reply into the same segment shape `whisper-cli` produces, so
`transcript.build_from_segments` reassembles words from one code path either way.

Three wrinkles. **Server-side VAD**: the request asks the server to run Silero
first and transcribe only what it calls speech, which is the whole reason this
pipeline no longer detects speech itself — see below. **Chunking**: a 2 h track is
~230 MB and one request means no progress for as long as it takes, so it is split,
with boundaries nudged onto a quiet spot ffmpeg found so a split does not land
mid-word. `WHISPER_CHUNK_SECONDS` is also the blast radius of a skipped decode
window, which is a separate matter covered below. **Tolerant parsing**: whisper-server's response shape varies by
version and `response_format`, so several are accepted (`segments` with float
seconds, with clock strings, or whisper-cli's `transcription` with `offsets`;
token lists are used when they carry timings and ignored when they are bare ids).
What is *not* tolerated is a response with no timings at all — that raises rather
than guessing.

Word timing quality differs: `max_len=1` is requested so each segment is one
word and timings are exact, but a build that ignores it returns sentence
segments whose word positions are interpolated. The run says which it got,
because it decides how tightly a stutter can be cut.

**An empty answer is an answer.** A chunk holding no speech comes back as `200`
with an empty segment list, and that is the truth about that chunk. Treating it as
a failure cost a whole track the first time chunking met a silent stretch — on a
two-mic recording each participant is silent for minutes while the other talks, so
those stretches are the norm rather than the exception. What still raises is a
response that had something to say and nowhere to put it: text without timings.

**A word with timings is not evidence a word was spoken.** On the sample
recording Whisper returned a final `right` spanning 9.94–11.34 s, over audio
whose peak is −50.4 dB — the noise floor. The transcript is a model's output, not
a measurement, and it will place words over near-silence.

### The transcript is the speech map

There used to be a `vad` stage: Silero (or ffmpeg's silencedetect) ran over each
prepared track and produced a speech map, and the plan treated it as one input
among two — the map decided where speech was, the transcript decided what was
said, and a disagreement between them was reported.

That stage is gone. whisper.cpp runs Silero itself, and once the server is
transcribing speech only, the words coming back already carry that judgement:
silence is where the transcript has no words, and `transcript.speech_from_words`
is the padded union of the words. One Silero, in one place, and the Python side
has no model dependency at all.

**What that buys.** Whisper handed silence invents text, and a local detector
existed largely to withhold silence from it — a job the model's own VAD does
better, inside the model, without slicing WAVs and re-offsetting timings. The
`plan_speech_chunks` layer that did it here, its padding and merge-gap constants,
both VAD backends, the Silero packages, `VAD_BACKEND`, `SILERO_THRESHOLD` and the
stage itself are all gone. What remains is one number, `SPEECH_PAD`.

**What it costs, precisely.** The map is now a subset of what Silero passed, and
the residue — audible material Whisper wrote nothing for — reads as silence and is
eligible for a cut:

- fillers Whisper drops, mumbles, a too-quiet phrase, crosstalk bleed;
- laughter, a cough, a breath, a chair, an intro sting.

Cuts are global, so a laugh in a mutual pause is the realistic loss. That was
accepted deliberately: it is rare in the recordings this is for, and the answer to
non-verbal material is a model that transcribes it as a tag, not a second
detector here.

**And two checks became tautologies, so they were deleted rather than kept as
theatre.** `_words_without_speech` asked whether a track's audio supported its own
transcript, and `_words_lost_to_silence` asked whether a cut removed words nothing
asked for. Both compared the transcript against an independent opinion. With the
map derived from the transcript, neither can ever fire: a word is always inside
its own padded span, and a cut only happens where no padded span is. Keeping code
that cannot fail would read as a safeguard that is not there.

The first of those caught something real — 198 repetitions of one sentence over a
muted mic — so its *purpose* is kept by `plan.looping_words`, which needs no
second opinion. Whisper handed noise does not invent varied text; it repeats. A
run of eight words recurring five times and accounting for a quarter of a track is
warned about. That detects the failure by its shape rather than by comparison, and
it works on a transcript alone.

**The residual risks worth knowing.** A build too old to parse the `vad_*` form
fields ignores them silently, so nothing here can confirm the pass actually ran.
And where a segment arrives without usable token timings,
`_segment_words_by_proportion` spreads its words evenly across its whole span, so
that segment tiles continuously and silence inside it is invisible — which loses a
cut rather than inventing one, and is the direction to err in.

#### The second of those risks turned out to be the larger one

It is not confined to segments without usable timings, and losing a cut is not
all it does. Whisper's word timings run past the audio as a matter of course: on
the sample fixture a single word spanned 19.0 seconds of one track's silence, and
across the two tracks 41 seconds of word-time sat over measured silence. A map
built from those spans reads as wall-to-wall speech — 57s of a 57.7s track — with
two effects, and the second is not "losing a cut" at all:

- No gap is ever long enough to shorten, so `SILENCE_MIN_DURATION` never fires.
- One participant's stretched word covers everything the other says underneath
  it, so their disfluencies are classified as crosstalk and *muted rather than
  cut* — the timeline preserved to protect a speaker who was not talking.

`SPEECH_MAP_CLIP` bounds each word by the level scan: a word claims only the parts
of its own span that measurably had sound in them. This is narrower than the `vad`
stage that was deleted, and does not bring it back. The scan is not consulted
about *whether* something is speech — it cannot tell speech from a cough, which is
why it is not the map — only about *when* the audio it was attributed to actually
occurred. It trims and never extends, so it cannot make speech out of silence, and
material Whisper wrote nothing for stays invisible here exactly as before. The
trade above is unchanged.

Where a track has no level scan, its timings are trusted as before. A word with no
measured sound anywhere in it claims nothing: there is no audio there to protect,
and after a filler-biased decode (see §6) that is the shape an invented word
takes — so the same clip is what keeps `WHISPER_PROMPT` from turning a
hallucinated filler into protected speech.

Measured on the fixture, against an edit cut by hand: cuts went from 3 to 6, the
share of the episode removed from 6.6% to 16.1% against the 20.4% removed by
hand, and mutes from 4 spans to 1 — the two the plan had been muting were the two
the hand edit cut.

### Whisper throws away decode windows, and that is why the level scan survived

The first real episode run under this design lost 33 seconds of clear speech from
one track, and every part of the pipeline agreed that stretch was silent.

Whisper decodes in 30-second windows (`WHISPER_CHUNK_SIZE`), and a window whose
decode ends on a lone timestamp token is discarded whole — `"single timestamp
ending - skip entire chunk"`, `src/whisper.cpp`. With VAD the windows are cut from
*filtered* audio, so one skipped 30s window spanned 33s of original time once the
silence inside it is counted back. The response says nothing about it. The
transcript that came back was internally consistent, so the speech map derived
from it was consistent too, and a silence cut removed the audio.

Request length does not fix this. Which window a passage lands in depends on how
much speech precedes it, so length only reshuffles the alignment: the same passage
survived a 100s request and vanished from a 300s and a 600s one. `120` is chosen
to keep any single loss small, not because it is safe.

Two things follow from it. The first is that the loss is *recoverable*: re-sending
the missing span as a short request of its own puts it at a different offset in the
windows, so it no longer falls in a discarded one. `asr.recover_missing` does that
after the first pass — one request per stretch, VAD still on so a loud stretch that
is not speech comes back empty rather than invented, and recovered words dropped
where they overlap what is already known, since the retry is padded and
re-transcribes its neighbours. It runs in `asr.py` rather than in the plan stage
because the plan stage has no endpoint and touches no audio, and it runs before the
words are written so nothing downstream ever sees the damaged transcript.

The second is that recovery cannot be trusted to succeed, so something still has to
refuse. That is the level scan, which is why it now runs for every track rather than
only the ones long enough to split. It is the only input in the whole
pipeline that Whisper had no hand in. It cannot tell speech from a cough — that is
precisely why it is not the speech map — but it can tell loud from silent, and
`plan.untranscribed_audio` compares each transcript against its own track's loud
stretches. Over 3s is reported; once cuts remove more than 5s of it the run
refuses.

That restores, on a different footing, the independence the removed
`_words_without_speech` check had. It does not need Silero, or any model: ffmpeg
was already a hard dependency. The lesson worth keeping is narrower than "keep the
old check" — it is that a transcript cannot be its own witness, and something in
the pipeline has to look at the audio directly.

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
- A transcript that repeats one run of words past plausibility is warned about
  and recorded in `plan.json` under `looping_transcripts`. This is the one check
  that survives deriving the speech map from the transcript, because it needs no
  independent opinion about the audio — see §7. A missing transcript is a hard
  failure rather than a silent track: silence is the absence of words, so a track
  without any would read as silent everywhere, and since cuts happen only where
  every track is silent, it would stop protecting its own audio from everyone
  else's cuts.
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
    because it is rare. A refused key, an unhonoured schema, or a server with no
    model loaded would drop all of them and finish reporting no edits, which is
    indistinguishable from clean speech, so all three abort instead (exit 2, 3
    and 5). A track where *every* window failed for ordinary reasons is the
    weaker case: exit 4, which fails that track and lets the rest of the episode
    finish. Any new whole-run failure mode belongs here rather than in the
    per-chunk `except`.

    The unloaded model earned its place the hard way. A router with
    `--no-models-autoload` dropped the model *during* the transcribe stage, so
    the readiness check at the top of `detect` had passed several minutes
    earlier and every window then failed with an ordinary-looking HTTP 400.
    Checking a server before a long stage does not prove it will still be there
    after it.
11. **Detection does not depend on how many windows are in flight.** Raising
    `LLM_CONCURRENCY` changes speed and nothing else: the edits, the audit file
    and the failure count come out byte-identical to a sequential run, because
    windows are consumed in submission order however they are answered.

## 10. Testing strategy

```sh
python3 tests/test_pipeline.py    # 156 tests, 21 classes, ~60 s, stdlib only
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

### Four layers

| Layer | Runs | Needs | Catches | Cannot catch |
| --- | --- | --- | --- | --- |
| Unit — `tests/test_pipeline.py` | pure decision code | python3 | wrong intervals, bad validation, malformed expressions | anything about what ffmpeg or a model actually does |
| Stubbed integration — same file + `tests/stub_servers.py` | real HTTP clients against fake servers | python3 | wrong request payloads, bad response handling, retry behaviour | whether a real server would answer that way |
| End to end — `tests/selftest.sh` | the real pipeline over synthetic audio | ffmpeg | wrong rendered audio, wrong stage wiring, wrong file layout | model quality, real-world audio, long-file behaviour |
| Manual — `tests/samples/` against real servers | the whole pipeline over a real recording | ffmpeg, a whisper-server, a llama-server | wire formats a stub would have accepted, quiet speech, invented words, a threshold in the wrong place | anything absent from one 11-second clip |

The unit layer is where a bug should be reproduced if it possibly can be: it is
fast, needs no audio, and a failure points at one function. The end-to-end layer
exists for the claims that only ffmpeg can settle.

The fourth layer is deliberately **not** automated. It needs two servers, a
model, and a judgement about how the result sounds, none of which belong in a
suite that must stay runnable offline. It earns its place anyway: the first time
it ran it found four things the other three had been green through, and "What the
first real run found" below is about why that is not a coincidence.

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
  The consequence: the request fields sent to each server — the `vad_*` form
  values, `prompt`, `max_len` — are asserted against stubs that accept anything.
  Those are the lines to re-read by hand.

  The sharper consequence is that **the stubs accept whatever the client sends**,
  because both sides were written here. A green suite proves the request has the
  shape we intended, never that a real server agrees — which is exactly why the
  `response_format` spelling is sent twice and why `LLM_CHECK_SCHEMA` exists. A
  first run against a real server remains the only evidence that the wire format
  is right, and the schema check is what makes that run fail loudly instead of
  silently.
- **Silero's judgement** — it runs inside whisper.cpp now, so nothing here can
  reach it and synthetic audio would have nothing for it to judge anyway. What is
  covered is that the request asks for it and spells every field the way
  whisper-server parses it (`TestServerSideVadRequest`, selftest case 1). Whether
  the server on the other end honours those fields, or is old enough to drop them
  without complaint, is not something this side can test — or detect at runtime.
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
- **Concurrency.** The `FFMPEG_JOBS > 1` path is not exercised; the suites run
  serially.

### What the first real run found

The suites were green throughout. The first run against a real whisper-server and
a real llama-server, on a real recording, found four things — worth listing
because they share a shape:

1. **The client never sent a `model` field.** A single-model server ignores it, so
   the assumption held everywhere it was tested; a router-mode server refuses.
2. **The VAD and the transcript disagreed**, and nothing compared them. Quiet
   speech at −39.2 dB fell under the −35 dB threshold and was cut while Whisper
   had transcribed it.
3. **The default `SILENCE_THRESHOLD` cut through the middle of the speech.** A
   threshold belongs below the quietest speech and above the noise floor; this
   recording put speech at −21 dB and −39 dB with a floor near −50 dB, and
   `-35dB` fell *between the two kinds of speech*.
4. **The lost-word warning gave advice for the wrong backend**, telling a Silero
   run to try Silero.

Findings 2, 3 and 4 are all about a local speech detector that no longer exists;
the level threshold they argued over now only picks chunk boundaries, where being
wrong costs one word rather than an edit (§7).

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
| 3. no cut overlaps speech | selftest case 1, checked against the transcribed words rather than the plan's own map |
| 4. `keep` complements `cuts` | `TestPlanBuilder.test_keep_and_cuts_are_complementary` |
| 5. no unvalidated LLM number reaches audio | `TestLlmValidation`, `TestAgainstStubServer` |
| 6. local models never overlap | **nothing** — structural only, read the two stages |
| 7. inputs deleted only after verification | selftest cases 1 (deleted), 3 (preserved on failure) |
| 8. transcript matches the rendered audio | `TestFinalTranscript`, selftest case 6 |
| 9. failure is recoverable | selftest case 3 (work dir kept), 6 and 7 (resume via `--from`) |
| 10. a whole-run fault aborts | `TestApiKeyAuth`, `TestAgainstStubServer` schema cases |
| 11. concurrency changes speed only | `TestDetectConcurrency` (result and audit compared against a sequential run) |

Invariant 6 no longer has a mechanism behind it. It described this script
keeping two local models out of each other's way, and there are no local models
any more — both are endpoints, and their memory is the serving machine's
problem. It is recorded in §7 as something given up deliberately rather than
quietly dropped.

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
2. If `transcribe` or `detect` was touched: **run `tests/samples/` against real
   servers.** Nothing automated covers the real models, and the request shapes
   this code sends have been wrong in ways every stub accepted. The fixture's
   README has the commands and the numbers a correct run produces.
3. Read `<episode>_edit-report.txt` — cut counts and removed fraction are the
   cheapest signal that a threshold has drifted, and the warnings block is where
   a looping transcript shows up.
4. If the change was anywhere near rendering, sample the audio, and check the
   invariants table above for what is guarding you.
5. If a request payload changed, check that `LLM_CHECK_SCHEMA` still passes
   against a real server. It is the only thing standing between a wrong wire
   format and an episode that silently finds no edits.

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
- `filler` detection exists but is off by default, and covers only the
  non-lexical sounds. Crutches made of real words — *"well…"*, *"you know"*,
  *"né"* — are detected by no kind at all (§6).
- Whisper removes some disfluencies during transcription, so `detect` can only
  work on what survives (§6). Its yield is lower than the raw speech would
  suggest, and nothing recovers the difference.
- **Nothing checks the transcript against the audio**, because the speech map is
  derived from the transcript (§7). Audible material Whisper wrote nothing for —
  laughter, a cough, a dropped filler — reads as silence and can be cut. Only the
  shape of a looping transcript is still detectable.
- **A whisper build that ignores the `vad_*` request fields is undetectable from
  here.** It answers normally and transcribes the silence too, and what it invents
  there becomes speech in the plan. `whisper-server --help | grep -c vad` should
  report 8.
- `SPEECH_PAD` is a single margin for a whole episode, and it is doing two jobs at
  once: absorbing Whisper's timing error and setting how long a gap must be to
  count. A recording where those want different values has no right answer.
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
| `WHISPER_VAD` | the only speech detection there is; off means silence gets transcribed and whatever is invented there becomes speech in the plan |
| `WHISPER_ENDPOINT`/`LLAMA_ENDPOINT` | required, not optional; there is no local mode to fall back to, and `127.0.0.1` is how a one-machine install is spelled |
| `SPEECH_PAD` | how far each word is widened before the union that makes the speech map; a gap needs `SILENCE_MIN_DURATION` **plus twice this** to be silence |
| `SPLIT_SILENCE_THRESHOLD` | picks chunk boundaries, and sets how much loud-but-untranscribed audio gets reported; it never decides what is cut |
| `WHISPER_RECOVER` | re-asks about stretches the first pass returned nothing for — the only thing that recovers a discarded decode window |
| `WHISPER_PROMPT` | conditioning text, not an instruction; empty means Whisper returns fluent prose and the disfluencies never reach the LLM stage at all |
| `WHISPER_REASK` | questions a word carrying more *measured* speech than `WHISPER_REASK_WORD_SECONDS`; a word merely stretched across silence is not questioned |
| `WHISPER_REASK_WINDOW` | short is the mechanism — a long window returns the same fluent reading that hid the disfluency |
| `SPEECH_MAP_CLIP` | bounds each word by the level scan when building the speech map; off means a word stretched across silence protects all of it, from both cutting and the other track's disfluencies |
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
| `LLM_CONCURRENCY` | throughput only; correct only when it matches the server's `--parallel`, whose `-c` is split that many ways unless it was given `--kv-unified` |
| `*_API_KEY_FILE` | preferred over the inline form; the key never reaches argv or the log |
