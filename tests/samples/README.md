# Test samples

Real recorded speech, for the checks the synthetic suites cannot make.

## `sample_speaker.flac`

11.33 s, 44.1 kHz stereo, converted losslessly from the original WAV (the decoded
PCM is bit-identical). Named `sample_speaker` so the pipeline reads it as episode
`sample`, participant `speaker`, under the default `TRACK_SEPARATOR="_"`.

Transcribed by whisper.cpp `large-v3-turbo` as:

> well you know I don't know you remember that **there's a way there's a way** to
> fix that right

**Why this one is worth keeping.** It contains a repetition that *survived
transcription*. That is not a given: speech synthesised with `espeak-ng` and fed
through the same server came back with its stutters silently removed —
`"So I I I think ... the the weather"` transcribed as `"So I think ... the
weather"`. Whisper's decoder normalises away the very disfluencies the `detect`
stage exists to find, so synthetic audio cannot exercise that stage at all. Real
speech can.

## Using it

The offline suites do not touch this file — they synthesise their own audio and
stub both servers, and stay runnable with no network and no models. This is for
manual checks against real servers:

```sh
mkdir -p /tmp/podcast-check/incoming
cp tests/samples/sample_speaker.flac /tmp/podcast-check/incoming/

# Everything but detect, against a real whisper-server. That server must have
# been launched with -vm pointing at a Silero model, or it will transcribe the
# silence too and the plan will read the invented words as speech.
clean-podcast.sh --root /tmp/podcast-check \
    --whisper-endpoint http://your-server:8081 --no-llm

# Add detect once an LLM endpoint is available. A router-mode server also
# needs LLAMA_MODEL_NAME set to one of the ids in /v1/models.
clean-podcast.sh --root /tmp/podcast-check \
    --whisper-endpoint http://your-server:8081 \
    --llama-endpoint http://your-server:8080
```

## What a real run showed

Verified end to end against real servers: whisper.cpp `large-v3-turbo` and
llama.cpp serving `Qwen3.6-35B-A3B-Q4_K_M`. `detect` flagged words 9–11, the
first `there's a way`, at confidence 0.9 — cutting the abandoned attempt and
keeping the completed one. Nothing was rejected by validation, and the render
matched the frame-exact prediction to 0.000 s.

Two things this recording exposes that synthetic audio cannot:

**Whisper places words where there is no audio.** It transcribes a final `right`
across 9.94–11.34 s, a span whose peak is −50.4 dB — the noise floor. Word timings
are not evidence that a word was spoken. This is the reason `SPEECH_PAD` exists
and the reason `WHISPER_VAD` matters: the server's Silero pass is what keeps that
kind of invention out of the transcript in the first place, and once the speech map
is derived from the transcript, anything that gets in becomes speech in the plan.

**The levels, which is why a level threshold was the wrong tool.**

| | measured peak |
| --- | --- |
| `well you`, `there's a way` | −21.4 dB, −22.2 dB |
| `I don't know` | **−39.2 dB** |
| the invented `right` | −50.4 dB (noise floor) |

A level threshold has to sit below the quietest speech and above the noise floor,
and this clip shows how narrow that band can be: the old `-35dB` default fell
*between two kinds of speech in the same sentence*, cutting `I don't know` while
the rest survived. Nothing was broken — the threshold was simply in the wrong
place, and a single number for a whole episode has no right value when loudness
varies within one breath.

### Historical: the two local VAD backends

Kept as the measurement that justified removing them. Neither `VAD_BACKEND` nor
`SILERO_THRESHOLD` exists any more — speech detection is whisper.cpp's Silero pass
(§7 of DESIGN.md) — so this is a record, not guidance.

Same audio, same servers, only the backend differing. The `ffmpeg` column was
measured when `-35dB` was still the default:

| | `ffmpeg` (−35dB) | `silero` |
| --- | --- | --- |
| speech detected | 3.68 s in 7 spans | **5.50 s in 5 spans** |
| removed | 52.3% | **27.6%** |
| cuts | 4 (3 silence, 1 LLM) | 3 (2 silence, 1 LLM) |
| words lost to silence | 4 — `know I don't right` | **1 — `right`** |

```
original   well you know I don't know you remember that there's a way there's a way to fix that right
ffmpeg     well you know            you remember that there's a way            to fix that
silero     well you know I don't know you remember that there's a way          to fix that
```

Silero produced the edit a person would: it heard the quiet `I don't know` that the
level threshold missed. Both backends made the *same* LLM cut — 7.080–8.170, the
repeated `there's a way` — so the detection stage was unaffected by the choice, and
both renders matched their frame-exact prediction to 0.000 s.

That result is the argument for the current design taken one step further: if
Silero is the right judge, and whisper.cpp will run Silero itself before
transcribing, then the transcript already carries its judgement and a second copy
here was redundant. The two Silero packages this used to choose between agreed to
within one 32 ms chunk of each other, which is quantisation rather than
disagreement — and neither is a dependency any more.

### What to check on a re-run

The lost-word warning these numbers came from no longer exists: it compared the
transcript against an independent speech map, and there is no longer one to compare
against (DESIGN.md §7 explains why keeping a check that cannot fail is worse than
removing it). What is worth looking at instead:

- **Is `right` in the transcript at all?** With the server running Silero it should
  not be — a −50.4 dB span is not speech, and keeping it out is the whole point of
  doing the VAD before transcription rather than after.
- **Does `I don't know` survive?** At −39.2 dB it is the quiet speech that a level
  threshold cut. Silero should hear it; if it goes missing, lower
  `WHISPER_VAD_THRESHOLD`.
- **Removed fraction and cut count**, from `sample_edit-report.txt`. Roughly 27%
  was the figure with a speech-based judge; a jump toward 50% means silence is
  being found where speech is.
- **The repetition at 7.080–8.170**, which is the one thing only real speech can
  exercise: the LLM cut it in a verified run at confidence 0.9, and synthetic audio
  cannot reach that stage at all because Whisper normalises invented stutters away.
