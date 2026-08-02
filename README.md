# Podcast-Cleanup-LLM

Set of scripts for cleaning up a multi-track podcast recording using FFMPEG,
Silero, Whisper and llama.cpp.

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
| whisper.cpp | the `transcribe` stage | `whisper-cli` locally, or a `whisper-server` endpoint |
| llama.cpp | the `detect` stage | `llama-server` locally, or an endpoint |
| `silero-vad` or `pysilero_vad` | only for `VAD_BACKEND=silero` | optional; the latter is 2 MB and needs no torch |

## Where the models run

Whisper and the LLM are configured independently, so all four combinations work:

| | local Whisper | remote Whisper |
| --- | --- | --- |
| **local LLM** | both managed here | `--whisper-endpoint URL` |
| **remote LLM** | `--llama-endpoint URL` | both endpoints |

Local means the binary is run here and its lifetime is managed here. Remote means
an endpoint is used as-is and no process is ever spawned or stopped. Config keys
are `WHISPER_ENDPOINT` and `LLAMA_ENDPOINT` — empty for local; `--local-whisper`
and `--local-llama` force local for one run when the config names an endpoint.
Every run prints which arrangement it is using.

**Memory.** When both are local they are **never resident at the same time**:
every track is transcribed and each Whisper process exits before a model is
loaded for detection, and the llama server is shut down before rendering begins,
so peak memory is whichever single model is largest rather than their sum. That
guarantee only covers processes this script manages — if both endpoints are
remote and share a machine, keeping them out of each other's way is that
machine's business, and the run says so.

**One caveat on remote Whisper.** Word timings are only as good as what the
server returns. A build honouring `max_len=1` returns one word per segment and
the timings are exact; one that ignores it returns sentence segments, and word
positions inside them are interpolated, which makes stutter cuts less tight. The
run warns when that happens. Local `whisper-cli` always gives per-token timings.
Audio is uploaded in chunks (`WHISPER_CHUNK_SECONDS`, 0 for one request), with
boundaries nudged onto silence the VAD stage already found so no chunk edge lands
inside a word.

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
    -t "$(nproc)" \
    -ml 1 -sow \
    -fa
```

- **`--host 0.0.0.0`** — the default binds to localhost, which is the usual
  reason a "remote" server cannot be reached.
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
- Requests are sent one at a time, since a single model instance serialises them
  anyway. `WHISPER_JOBS` applies only to the local path.

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

- **`LLAMA_CTX` in the config does nothing for a remote server** — it is only
  passed to a server this script starts itself. The remote server's own `-c`
  governs, and nothing checks that the two agree.
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

Nothing here is tied to the models in the examples. Both are named only by
`WHISPER_MODEL` and `LLAMA_MODEL`, which are paths — swap either, or upgrade to a
new version of the same one, and the pipeline does not need to know.

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
prepare     decode each track to 16 kHz mono (what Whisper and Silero want)
vad         per-track speech detection
transcribe  Whisper per track, to completion, then every process exits
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
clean-podcast.sh --stages discover,prepare,vad    # silence analysis only
clean-podcast.sh --no-llm                         # skip the LLM stage
clean-podcast.sh --dry-run                        # show the commands, touch nothing
clean-podcast.sh --list-stages
```

A note on `transcribe` and `detect`: their stage boundary is also the memory
boundary when both models are local, so re-running only one of them is safe.

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
- **A silence cut swallowing transcribed words is reported.** The VAD's idea of
  speech and the transcript's are independent, and when they disagree the output
  hides it: the published transcript is rebuilt from the rendered timeline, so a
  word cut away vanishes from both and the result still looks consistent. The
  plan lists the words and the report warns. Either side can be the wrong one —
  a level-based VAD misses quiet speech, and Whisper places words over
  near-silence — so it warns rather than deciding, and `plan.json` keeps the
  full list under `words_lost_to_silence` for a listen. Usually it means
  `SILENCE_THRESHOLD` is too high for the recording, or that
  `VAD_BACKEND="silero"` would suit it better.
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
lib/stages.sh                 the eight stages, and the llama server lifecycle
python/cleanup_cli.py         subcommands the shell calls
python/cleanup/intervals.py   interval algebra and timeline remapping
python/cleanup/vad.py         silencedetect parsing and Silero
python/cleanup/transcript.py  Whisper tokens to words
python/cleanup/asr.py         remote whisper-server client and chunking
python/cleanup/llm.py         chunking, prompting, response validation
python/cleanup/plan.py        the cut-versus-mute decision
python/cleanup/render.py      ffmpeg expressions, duration prediction, transcript
tests/stub_servers.py         stand-ins for both servers, for the self-test
DESIGN.md                     how it all fits together, and why
```

The shell drives ffmpeg and the models; everything under `python/cleanup/` reads
and writes JSON and never touches audio.

## Tuning notes

- `SILENCE_THRESHOLD` is the first thing to adjust if too much or too little is
  being cut. The default `-45dB` errs on the safe side, since cutting quiet
  speech is damage while leaving room tone is only a looser edit. Raise it
  toward `-35dB` for a loud, close-mic'd recording. Read
  `ep042_edit-report.txt` before trusting a run — and if it warns that a cut
  swallowed transcribed words, this is the setting at fault.
- `INPUT_EXTS` is worth narrowing to just your own format if the input directory
  holds anything else you would rather not sweep up. The same participant present
  in two formats is an error, not a preference.
- `VAD_BACKEND="silero"` is markedly better at telling speech from breathing and
  room tone, at the cost of a torch dependency.
- `LLM_ACCEPT_KINDS` excludes `filler` by default; removing every "um" tends to
  read as over-editing. Add it if you want that.
- `RENDER_FRAME_SAMPLES` is the timing granularity of the edit (512 samples is
  ~11 ms at 48 kHz). Smaller is more precise and fades more smoothly, but slower
  to render. It has to be the same for every track of an episode, which is why it
  is a config setting rather than a per-track choice.
- `WHISPER_JOBS` stays at 1 unless there is RAM for several Whisper instances. It
  never overlaps with the LLM stage either way.
- `LLM_CONCURRENCY` is the one to reach for when the detect stage is the slow
  part and the machine running the model looks idle — one request at a time
  decodes at batch size 1 and cannot use the cores. Set it to the server's
  `--parallel` slot count, and raise `LLAMA_CTX` by the same factor, since the
  slots divide it. It changes speed only: the edits, the audit log and the
  report come out identical to a sequential run.
