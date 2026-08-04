# Podcast-Cleanup-LLM

Set of scripts for cleaning up a multi-track podcast recording using FFMPEG,
Whisper and llama.cpp.

Takes the synchronised per-participant tracks of one episode, works out which
stretches are dead air and which are speech disfluencies, and renders the tracks
back out — still separate, still in sync, ready for mixing.

```sh
clean-podcast.sh                      # uses ./incoming, creating it if need be
clean-podcast.sh --root /srv/media/podcast
```

## What it does to the audio

Two kinds of operation come out of the analysis, and the distinction between
them is the heart of the design.

**Cuts** are applied to every track at the same time range, so the timeline gets
shorter. Used for over-long silence, and for a disfluency during which nobody
else was speaking.

**Mutes** are applied to a single track and leave the timeline alone. Used when a
disfluency overlaps another participant's speech: cutting there would take a bite
out of whoever else was talking, so the stutter is silenced on its own track
instead, with a short fade at each end to avoid a click.

Because every cut is global and identical, the tracks stay sample-aligned with
each other by construction — the property the whole pipeline exists to preserve.

Silence is **shortened, not removed**. A gap longer than `SILENCE_MIN_DURATION`
is trimmed down to `SILENCE_KEEP` of residual quiet rather than spliced out
entirely, so the edit keeps its breathing room instead of sounding gasped.

## Requirements

| Tool | Needed for | Notes |
| --- | --- | --- |
| ffmpeg + ffprobe | everything | the only hard dependency |
| python3 | everything | standard library only |
| a whisper-server | the `transcribe` stage | reached over HTTP; not started here |
| a llama-server | the `detect` stage | reached over HTTP; not started here |
| a Silero VAD model | the `transcribe` stage | `ggml-silero-*.bin`, run by whisper.cpp itself — no Python package |

## Where the models run

Both are servers you run yourself, reached over HTTP. `WHISPER_ENDPOINT` is
required, and `LLAMA_ENDPOINT` too unless `LLM_ENABLE=0`. Nothing here starts,
stops or waits on a model process, and every run prints the two URLs it is
using. `--whisper-endpoint` and `--llama-endpoint` override them for one run.

On one machine, point both at `127.0.0.1`. That is a deployment choice and this
side cannot tell the difference — which also means **it will not keep the two
models out of each other's memory.** Earlier versions ran both as subprocesses
and sequenced them so peak usage was the larger model rather than the sum; that
went away with the local mode. If both servers share a machine that cannot hold
both, sequence them yourself.

**One caveat on word timings.** They are are only as good as what the
server returns. A build honouring `max_len=1` returns one word per segment and
the timings are exact; one that ignores it returns sentence segments, and word
positions inside them are interpolated, which makes stutter cuts less tight. The
run warns when that happens.
Audio is uploaded in chunks (`WHISPER_CHUNK_SECONDS`, default 120, 0 for one
request), with boundaries nudged onto a quiet spot ffmpeg found so no chunk edge
lands inside a word.

**Whisper discards decode windows, and this is the one to know about.** It works
in 30-second windows, and a window whose decode ends on a lone timestamp token is
skipped whole — `"single timestamp ending - skip entire chunk"` in
`src/whisper.cpp`. On a real 600s track that silently cost 33 seconds of clear
speech: with VAD the windows are cut from *filtered* audio, so one skipped 30s
window spanned 33s of the original once the silence inside it is counted back.
Which window a passage lands in depends on how much speech precedes it, so
request length shuffles the alignment rather than curing anything — the same
passage survived a 100s request and vanished from a 300s and a 600s one. Nothing
in the response reports it, and since the speech map is derived from the
transcript, the loss reads as silence and gets cut. `WHISPER_CHUNK_SECONDS=120`
keeps any single loss small, `WHISPER_RECOVER=1` re-asks about the missing span on
its own — which recovers it, because the window alignment is then different — and
the audio cross-check below refuses to cut whatever still cannot be recovered.

**And one that matters more.** The words that come back are also the speech map —
silence is where the transcript has none — so the server must be running Silero
over the audio first, or it will transcribe the silence too and whatever it
invents there will read as speech. `WHISPER_VAD=1` asks for that per request;
what the server needs at launch is `-vm` (see below). A build too old to parse
the `vad_*` form fields ignores them silently, and nothing on this side can tell:
check `whisper-server --help | grep -c vad` reports 8.

## Running the servers

If you serve either model remotely, these are the settings that matter to this
pipeline. Both projects' flags drift between versions — check `--help` before
copying.

### Authentication

Neither server requires a key by default, and on a trusted LAN neither needs
one. If an endpoint is behind something that does — a reverse proxy in front of
whisper-server, or `llama-server --api-key` — set the matching pair:

| | inline | from a file |
| --- | --- | --- |
| **Whisper** | `WHISPER_API_KEY` | `WHISPER_API_KEY_FILE` |
| **LLM** | `LLAMA_API_KEY` | `LLAMA_API_KEY_FILE` |

Either form sends `Authorization: Bearer <key>`. Prefer the `_FILE` form: a key
in the config file is also a key in your editor's backups, and readable by anyone
who can read the config, whereas a key in its own file can be `chmod 600`. A file
readable beyond its owner draws a warning.

The key is never passed as a command-line argument, since argv is readable by
every process on the machine, and the config dump records only whether one was
set — the run log is copied into the output directory and outlives the episode.

**A refused key fails fast rather than being retried.** The readiness check
reports it before the episode starts instead of polling out its timeout, and the
detection stage stops the whole run on the first refusal. Retrying cannot help,
and since that stage drops a chunk it cannot process, carrying on would otherwise
deliver an episode that had quietly found no edits at all.

### whisper-server

```sh
whisper-server \
    -m /srv/llm/models/whisper/ggml-large-v3-turbo.bin \
    --host 0.0.0.0 --port 8080 \
    -vm /srv/llm/models/whisper/ggml-silero-v6.2.0.bin \
    -t "$(nproc)" \
    -ml 1 -sow \
    -fa
```

- **`--host 0.0.0.0`** — the default binds to localhost, which is the usual
  reason a "remote" server cannot be reached.
- **`-vm <silero model>`** is the one this pipeline cannot work properly without.
  It points at the Silero VAD model so the server can transcribe speech only;
  fetch one with whisper.cpp's `models/download-vad-model.sh silero-v6.2.0`.
  Everything *else* about the VAD travels per request, so nothing here needs
  `--vad` at launch and other clients of the same server are unaffected:

  | request field | set from | default here | whisper.cpp default |
  | --- | --- | --- | --- |
  | `vad` | `WHISPER_VAD` | `true` | `false` |
  | `vad_threshold` | `WHISPER_VAD_THRESHOLD` | `0.5` | `0.5` |
  | `vad_min_speech_duration_ms` | `WHISPER_VAD_MIN_SPEECH_MS` | `250` | `250` |
  | `vad_min_silence_duration_ms` | `WHISPER_VAD_MIN_SILENCE_MS` | `1000` | `100` |
  | `vad_speech_pad_ms` | `WHISPER_VAD_SPEECH_PAD_MS` | `300` | `30` |
  | `vad_samples_overlap` | `WHISPER_VAD_SAMPLES_OVERLAP` | `0.1` | `0.1` |

  The two that differ from upstream do so for the same reason in opposite
  directions. `min_silence` is raised because 100 ms ends a segment at every
  breath, and each segment then reaches Whisper with less of the context it
  punctuates from. `speech_pad` is raised because a word's opening consonant is
  quieter than the vowel behind it, so segment edges are where Silero errs — and
  a clipped edge is a word that never reaches the transcript at all.

  A build too old to parse those fields ignores them without complaint, and the
  symptom is a transcript with invented speech in it. `whisper-server --help |
  grep -c vad` should report 8.
- **`-ml 1 -sow`** is the one that affects output quality. It makes every
  returned segment a single word, so word timings are exact rather than
  interpolated across a sentence, which is what decides how tightly a stutter
  can be cut. The client asks for `max_len=1` per request as well, but not every
  build honours per-request parameters — setting it on the server removes the
  doubt. The run reports which it got; look for a line about interpolated
  timings.
- **`-fa`** (flash attention) on CUDA builds; `-t` matters mainly on CPU ones.
- Uploads are **19.2 MB per request** at the default `WHISPER_CHUNK_SECONDS=600`
  (16 kHz mono 16-bit). If your build rejects bodies that size, lower the
  setting rather than raising a limit — `300` halves it. `0` sends a whole 2 h
  track as one ~230 MB request, which is rarely a good idea.
- No `-cv/--convert` needed: the audio is already 16 kHz mono WAV.
- Requests are sent one at a time, since a single model instance serialises
  them anyway.

### llama-server

```sh
llama-server \
    -m /srv/llm/models/qwen/Qwen3.6-35B-A3B.gguf \
    --host 0.0.0.0 --port 8081 \
    -c 8192 --parallel 1 \
    -ngl 99 -fa \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --mlock
```

- **`-c 8192 --parallel 1`.** The trap here is that `-c` is the *total* context
  divided among slots, so `-c 8192 --parallel 4` leaves 2048 per slot and
  silently truncates the prompt. At the default `LLM_CHUNK_WORDS=350` a request
  is ~3.3k prompt tokens plus 2048 reserved for the reply, so a slot needs
  ~6k. Multiply `-c` by the slot count.

  | `LLM_CHUNK_WORDS` | prompt tokens | minimum context per slot |
  | --- | --- | --- |
  | 150 | ~1.8k | 4096 |
  | 250 | ~2.5k | 5120 |
  | 350 (default) | ~3.3k | 6144 |
  | 500 | ~4.5k | 7168 |

  The division is what llama.cpp does when the KV cache is split, which is the
  default as soon as `--parallel` is given a number — passing `-np` takes the
  slot count out of "auto", and `--kv-unified` defaults to on only while it is
  auto. With `--kv-unified` forced on, one sequence may use the whole `-c`, but
  the buffer is still a single pool every slot draws from, so the total budget
  is unchanged and a busy moment fails on load rather than deterministically.
  For a batch job that is the worse trade: prefer sizing `-c` for the slots.

- **More slots is the way to make the detect stage faster.** At one request in
  flight the server decodes at batch size 1, which is bound by memory bandwidth
  and leaves most of a CPU idle; several slots let it batch the decode steps
  together. Give the server `--parallel N` and set `LLM_CONCURRENCY=N` to
  match — the two have to agree, since anything beyond the slot count just
  queues inside the server where the client cannot see it. A server this script
  starts gets `--parallel` derived from `LLM_CONCURRENCY` automatically, and the
  detect stage checks the context arithmetic above before the episode starts.

- **The server's `-c` is the only context there is**, and nothing here checks it
  against `LLM_CHUNK_WORDS`. A prompt that does not fit is truncated silently and
  shows up only as a track with suspiciously few edits, so size it by hand.
- **Any instruct model works**, not just the one in the example. Requests go to
  `/v1/chat/completions`, so llama-server applies the chat template of whatever
  model it has loaded — see "Choosing a model" below.
- **`-fa` plus q8_0 KV cache** keeps VRAM down at no cost worth measuring here;
  `--mlock` stops the model being swapped out between episodes.
- Expect **~58 requests per participant** for a 2 h track at ~150 wpm, so the
  model is loaded once and hit repeatedly. The client sends `cache_prompt: true`
  and the ~570-token instruction prefix is identical every time, so prompt
  caching earns its keep. Slots each keep their own copy of that prefix, so
  running several costs one extra prefill apiece and no more — not a reason to
  stay at one slot.
- Sampling is fixed per request (`temperature: 0`, `top_k: 1`) and the JSON
  schema travels with each call, so server-side sampling defaults and
  `--grammar-file` are irrelevant.
- Raise `--timeout` if your build defaults below the client's
  `LLAMA_REQUEST_TIMEOUT` (600 s).
- For a MoE model on limited VRAM, newer builds have `--n-cpu-moe` to keep
  experts on the CPU while the dense layers stay on the GPU.

### Choosing a model

Nothing here is tied to the models in the examples. Which model each server has
loaded is that server's business — swap either, or upgrade to a new version of
the same one, and the pipeline does not need to know.

**Whisper.** Any ggml model `whisper-cli` or `whisper-server` accepts. Bigger
models transcribe better and are slower; what matters most downstream is word
timing precision, which is why `-ml 1 -sow` is worth more than model size for
this job. Remote responses are parsed tolerantly, so server versions that shape
their JSON differently are handled.

**The LLM.** Any instruct model llama-server can load. Requests go to
`/v1/chat/completions`, so the server applies that model's own chat template —
the prompt is plain instructions plus a JSON schema, with nothing model-specific
in it. Two things worth knowing:

- The schema is what guarantees parseable output, so a model that follows
  instructions moderately well is enough; you do not need a large one. What a
  bigger model buys is judgement about which repetitions are accidental, which
  is exactly what the `confidence` field and `LLM_MIN_CONFIDENCE` exist to
  filter.
- **A server in router mode needs `LLAMA_MODEL_NAME`.** A llama-server hosting a
  single model ignores the `model` field, but one started without a model path
  becomes a router for several and refuses any request that does not name one.
  `curl -s <endpoint>/v1/models` lists the ids. Such a server also has to have
  the model *loaded*: with `--no-models-autoload` it will not load on demand, and
  answers "model is not loaded". Both failures are reported before the first
  track, with the available ids listed.
- `LLM_API="completion"` reverts to the old raw-prompt `/completion` path, for a
  build without the chat endpoint or to compare the two on one episode. The
  audit log (`*.audit.jsonl`, kept with the outputs) records every response, so
  comparing two models — or two endpoints — on the same episode is just a diff.

`LLM_CHECK_SCHEMA` sends one small request before the first track to confirm the
server really constrains its output. It is on by default because the failure it
catches is otherwise invisible: a server that ignores `response_format` returns
prose, every chunk is dropped as unparseable, and the run finishes reporting no
edits at all — which looks exactly like a clean recording.

## Setup

```sh
cp podcast-cleanup.conf.example ~/.config/podcast-cleanup/config
$EDITOR ~/.config/podcast-cleanup/config
```

The config file is searched for in this order, first match winning:

1. the path given to `--config`
2. `$PODCAST_CLEANUP_CONF`
3. `./podcast-cleanup.conf`
4. `${XDG_CONFIG_HOME:-~/.config}/podcast-cleanup/config`
5. `/etc/podcast-cleanup.conf`

It is sourced as bash, so it holds assignments and nothing else. Every setting is
documented inline in the example, and command-line options override it.

## Input and output

Everything lives under one **root**, which defaults to the directory holding
`clean-podcast.sh`. The four subdirectories are created on first use, so a fresh
checkout is usable straight away — run it once, then drop tracks into the
`incoming/` it made:

```
<root>/
    incoming/      tracks waiting to be processed
    output/        finished episodes, one directory each
    work/          per-episode intermediates
    failed/        logs, and inputs if FAILED_ACTION="move"
```

Point the whole layout somewhere else with `--root DIR` or `PODCAST_ROOT`. Any
one of the four can still be set on its own (`--input`, `--output`, `--work`, or
`INPUT_DIR` and friends) for the case where they genuinely belong apart — scratch
space on a different volume from the finished audio, say. A `--root` on the
command line re-derives the ones you have not named explicitly.

The default suits a workstation, where the models are usually remote anyway; a
server install sets `PODCAST_ROOT` to its media volume in the config file and
never thinks about it again.

Input tracks are named `<episode><separator><participant>.<ext>`:

```
<root>/incoming/
    ep042_leonardo.flac
    ep042_marta.flac
    ep042_guest.flac
```

**Any format ffmpeg can decode** works — FLAC, WAV, AIFF, ALAC, MP3, AAC, Opus,
Vorbis, WavPack, or a video container with an audio track. `INPUT_EXTS` lists what
to look for (case-insensitively); the `prepare` stage decodes to PCM regardless,
so the container is only a matter of finding the files. Two tracks of one episode
may even be in different formats: decoders handle encoder delay correctly, so
they stay sample-aligned.

Output format is **independent of input** — FLAC by default via `OUTPUT_CODEC` /
`OUTPUT_EXT`, since the tracks are going on to be mixed. A lossy input is
accepted and noted: cutting and re-encoding cannot recover what the codec already
discarded, and a lossless output of it will be larger for no gain.

Two requirements remain. Tracks must be **aligned at sample 0** — nothing here
synchronises them. And they must **share a sample rate**, because cuts land on
frame boundaries and a track quantising at its own rate would drift from the
others; a mismatch is refused unless `RESAMPLE_TO` (`auto`, or a rate) tells the
run to resample, which it then does *before* frames are fixed so alignment holds.

Container headers are not trusted for length. AAC decodes a few milliseconds
longer than it claims, Opus shorter, and a truncated file of any format can claim
anything — so once `prepare` has decoded each track, its true length is measured
from the decode. That keeps the frame-exact output prediction exact whatever the
input format was.

After a successful run:

```
<root>/output/ep042/
    leonardo.flac                 cleaned tracks, all exactly the same length
    marta.flac
    guest.flac
    ep042_transcript.json         speaker-labelled, on the *rendered* timeline
    ep042_transcript.srt
    ep042_transcript.txt
    ep042_plan.json               every cut and mute, with its reason
    ep042_edit-report.txt         human-readable summary
    logs/
        run.log                   the whole run
        llama-server.log
        *.audit.jsonl             every LLM response, accepted or rejected
```

Everything else — the work directory, the 16 kHz intermediates, the raw Whisper
output, and the original inputs — is deleted. Inputs go **only** after every
output track has been rendered and its duration verified against the plan; any
failure leaves them untouched. `--keep-inputs` and `--keep-work` opt out.

## Stages

```
discover    find the episode's tracks, probe them, agree on an episode id
prepare     decode each track to 16 kHz mono (what Whisper wants)
transcribe  Whisper per track, to completion, then every process exits
            (its Silero pass is what decides where speech is at all)
detect      the LLM finds disfluencies; the server starts and stops here
plan        unify silence and edits into cuts and mutes
render      one ffmpeg pass per track, into staging, then verified
finalize    publish outputs, delete intermediates and inputs
```

Each stage reads and writes JSON in the work directory and records its
completion, so a failed run can be resumed rather than restarted:

```sh
clean-podcast.sh --episode ep042 --from plan      # redo the edit decisions
clean-podcast.sh --episode ep042 --only render    # just re-render
clean-podcast.sh --no-llm                         # silence editing only
clean-podcast.sh --dry-run                        # show the commands, touch nothing
clean-podcast.sh --list-stages
```

A note on `transcribe` and `detect`: `detect` reads the transcripts `transcribe`
wrote, so that boundary is a real dependency — but each stage's work is recorded
per participant, so re-running either alone is safe.

## How the LLM stage is kept honest

A model asked for timestamps will confidently invent them. So it never sees or
returns one:

- Whisper's word timings stay on our side. The transcript goes to the model as a
  **numbered word list**, and it replies with word-index ranges.
- Indices are checked against the transcript we already hold. Out-of-window,
  inverted, over-long and low-confidence spans are all rejected, and every
  rejection is recorded in the audit log with its reason.
- Responses are constrained by a JSON schema server-side, so there is no scraping
  of prose for something that looks like JSON.
- Long transcripts are sent in overlapping windows; findings reported twice are
  fused by index range.
- A window whose response cannot be used is dropped with a warning rather than
  failing the run. Losing a few stutters beats losing the episode.

Cut timings come from Whisper's word boundaries, widened by at most
`CUT_PADDING` and never by more than half the actual gap to the neighbouring
word, so padding cannot eat into real speech.

## Safety rails

- A plan that would remove more than `MAX_CUT_FRACTION` of the episode is
  refused — that almost always means the detection threshold is wrong, not that
  the episode is half silence. `--force` overrides.
- Rendered durations are checked against a **frame-exact prediction** of what the
  filter graph will emit, not a rule of thumb. A mismatch fails the run and
  preserves the inputs.
- **A skipped decode window is re-asked about, then refused over.** After the
  first pass, any loud stretch over 3s that produced no words is re-sent as a
  short request of its own; that changes where it falls in Whisper's 30-second
  windows, which is what gets the words back. The retry asks for VAD too, so a
  loud stretch that is not speech comes back empty rather than invented. What
  cannot be recovered is what the next guarantee refuses over — so the refusal is
  about genuine loss rather than about a transient.
- **A cut over audio nothing transcribed is refused.** One ffmpeg level scan per
  track is the only input Whisper had no hand in. It cannot tell speech from a
  cough — which is why it is not the speech map — but it can tell loud from
  silent, and loud audio that produced no words at all is either something
  Whisper declined to transcribe or a decode window it threw away. Stretches over
  3s are reported and recorded in `plan.json` under `untranscribed_audio`; once
  cuts remove more than 5s of it in total the run refuses, and `--force` is the
  way past. This is the check that catches the 30-second-window loss described
  above, and it is the reason the level scan now runs for every track.
- **A looping transcript is reported.** Nothing compares the transcript against
  the audio any more — the speech map is derived from the transcript, so the two
  agree by construction. What is still visible is the shape of the failure:
  handed something that is not speech, Whisper repeats itself rather than
  inventing varied text. On the recording that prompted this it produced one
  sentence 198 times across 4.6 minutes, 73% of a track whose mic was muted. A
  run of eight words recurring five or more times, and accounting for a quarter
  of a track, is warned about and recorded in `plan.json` under
  `looping_transcripts`. Treat that transcript, the speech map derived from it,
  and every edit built on it as unreliable until the audio has been listened to.
- Mismatched sample rates, tracks from two different episodes, duplicate
  participants and unparseable filenames are all rejected up front.
- On failure the work directory is kept, a `FAILED` marker is written, and the run
  log is copied to `FAILED_DIR`. Set `FAILED_ACTION="move"` to also move the
  inputs there, which stops an unattended run retrying the same broken episode
  forever.

## Tests

```sh
python3 tests/test_pipeline.py    # 156 unit tests, no external tools needed
./tests/selftest.sh               # 64 end-to-end checks, needs only ffmpeg
```

The unit tests cover the interval algebra, transcript parsing, LLM response
validation, the cut-versus-mute decision, and the generated ffmpeg expressions —
those last by evaluating the expressions with a miniature interpreter of the
grammar we emit, since a silently wrong expression would mangle audio without
failing anything.

The self-test builds synthetic episodes with silences in known places, runs the
real pipeline over them in a sandbox, and inspects the rendered audio. Among
other things it confirms that a stutter over crosstalk leaves both tracks exactly
their original length, that the affected track is silent across the muted span,
and that the *other* speaker's audio at that same instant is untouched.

Its last case runs all eight stages against stub HTTP servers standing in for
whisper-server and llama-server, which covers the remote clients, the multipart
upload, and the response handling. Real models are still never exercised, so run
one real episode with `--keep-work` the first time.

## Layout

```
clean-podcast.sh              CLI, config, stage sequencing, failure handling
lib/log.sh                    logging, progress, command execution
lib/config.sh                 defaults, config loading, validation
lib/stages.sh                 the seven stages, and the llama server lifecycle
python/cleanup_cli.py         subcommands the shell calls
python/cleanup/intervals.py   interval algebra and timeline remapping
python/cleanup/silence.py     silencedetect parsing: chunk boundaries, and the
                              only opinion about the audio that is not Whisper's
python/cleanup/transcript.py  Whisper tokens to words, and the speech map
python/cleanup/asr.py         remote whisper-server client, chunking, and
                              recovering a skipped decode window
python/cleanup/llm.py         chunking, prompting, response validation
python/cleanup/plan.py        the cut-versus-mute decision
python/cleanup/render.py      ffmpeg expressions, duration prediction, transcript
tests/stub_servers.py         stand-ins for both servers, for the self-test
DESIGN.md                     how it all fits together, and why
```

The shell drives ffmpeg and the models; everything under `python/cleanup/` reads
and writes JSON and never touches audio.

## Tuning notes

- `SPEECH_PAD` is the first thing to adjust if too much or too little is being
  cut, and it is the only threshold left on this side. Each word is widened by it
  before the union that makes the speech map, so a gap must clear
  `SILENCE_MIN_DURATION` plus twice it before anything is removed. Raise it for
  rarer, safer cuts; lower it for tighter ones. Read `ep042_edit-report.txt`
  before trusting a run.
- `WHISPER_VAD_THRESHOLD` is the other end of the same question, and it acts
  earlier: it decides what Whisper transcribes at all, and therefore what exists
  to be protected. Lower it if quiet speech is going missing from the transcript.
- `SPLIT_SILENCE_THRESHOLD` decides how much the audio cross-check complains
  about. Raise it toward `-35dB` if it keeps reporting non-speech you are content
  to lose; lower it to hear about more. It has no effect on what gets cut.
- `INPUT_EXTS` is worth narrowing to just your own format if the input directory
  holds anything else you would rather not sweep up. The same participant present
  in two formats is an error, not a preference.
- `LLM_ACCEPT_KINDS` excludes `filler` by default; removing every "um" tends to
  read as over-editing. Add it if you want that.
- `RENDER_FRAME_SAMPLES` is the timing granularity of the edit (512 samples is
  ~11 ms at 48 kHz). Smaller is more precise and fades more smoothly, but slower
  to render. It has to be the same for every track of an episode, which is why it
  is a config setting rather than a per-track choice.
- `LLM_CONCURRENCY` is the one to reach for when the detect stage is the slow
  part and the machine running the model looks idle — one request at a time
  decodes at batch size 1 and cannot use the cores. Set it to the server's
  `--parallel` slot count, and raise the server's `-c` by the same factor, since
  the slots divide it unless it was started with `--kv-unified`. It changes speed only: the edits, the audit log and the
  report come out identical to a sequential run.
