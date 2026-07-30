# Podcast-Cleanup-LLM

Set of scripts for cleaning up a multi-track podcast recording using FFMPEG,
Silero, Whisper and Qwen.

Takes the synchronised per-participant tracks of one episode, works out which
stretches are dead air and which are speech disfluencies, and renders the tracks
back out — still separate, still in sync, ready for mixing.

```sh
clean-podcast.sh --input /srv/media/podcast/incoming
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
| torch + `silero-vad` | only for `VAD_BACKEND=silero` | optional |

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

Input tracks are named `<episode><separator><participant>.<ext>`:

```
/srv/media/podcast/incoming/
    ep042_leonardo.flac
    ep042_marta.flac
    ep042_guest.flac
```

All tracks of an episode must **share a sample rate** — cuts land on frame
boundaries, and identical frame sizes across tracks are what keeps them aligned.
A mismatch is refused rather than allowed to drift silently. They must also
already be aligned at sample 0; this tool does not synchronise anything.

After a successful run:

```
/srv/media/podcast/output/ep042/
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
detect      Qwen finds disfluencies; the server starts and stops inside here
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
clean-podcast.sh --no-llm                         # skip Qwen entirely
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
- Mismatched sample rates, tracks from two different episodes, duplicate
  participants and unparseable filenames are all rejected up front.
- On failure the work directory is kept, a `FAILED` marker is written, and the run
  log is copied to `FAILED_DIR`. Set `FAILED_ACTION="move"` to also move the
  inputs there, which stops an unattended run retrying the same broken episode
  forever.

## Tests

```sh
python3 tests/test_pipeline.py    # 85 unit tests, no external tools needed
./tests/selftest.sh               # 46 end-to-end checks, needs only ffmpeg
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
  being cut. Read `ep042_edit-report.txt` before trusting a run.
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
