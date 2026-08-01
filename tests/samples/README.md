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

# Everything but detect, against a real whisper-server.
clean-podcast.sh --root /tmp/podcast-check \
    --whisper-endpoint http://your-server:8081 \
    --stages discover,prepare,vad,transcribe,plan,render,finalize --no-llm

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

**`SILENCE_THRESHOLD="-35dB"` is too aggressive for a quiet recording** — this
clip is what moved the default down to `-45dB`. Silence cuts only, `--no-llm`:

| `SILENCE_THRESHOLD` | removed | cuts |
| --- | --- | --- |
| `-35dB` (the old default) | 42.7% | 3 |
| `-45dB` (current default) | 29.5% | 3 |
| `-55dB` | 1.9% | 1 |

At `-35dB` the span carrying `I don't know` (measured peak −39.2 dB) falls below
the threshold and is cut as silence, while `well you` (−21.4 dB) and `there's a
way` (−22.2 dB) are comfortably above it and survive. Nothing was broken — the
VAD applied the threshold it was given — but that threshold sat between this
recording's speech and its noise floor, which is the one place it must not be.
`VAD_BACKEND="silero"` decides on speech rather than level and is the better
answer when levels vary.

**Whisper places words where there is no audio.** It transcribes a final
`right` across 9.94–11.34 s, a span whose peak is −50.4 dB — the noise floor.
Word timings are not evidence that a word was spoken.

### The two backends, same audio, same servers

Only `VAD_BACKEND` differs. Measured when `-35dB` was still the default, so the
`ffmpeg` column is the old behaviour — it is what the change to `-45dB` was
reacting to:

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

Silero produces the edit a person would: it hears the quiet `I don't know` that
the level threshold missed, so the mid-clip silence cut disappears entirely. Both
backends make the *same* LLM cut — 7.080–8.170, the repeated `there's a way` — so
the detection stage is unaffected by the VAD choice, as it should be. Both renders
matched their frame-exact prediction to 0.000 s.

The one word silero still loses is `right`, and that is the correct answer:
it is a Whisper hallucination over a −50.4 dB noise floor, so a speech-based VAD
is right to call it silence. Which is the warning behaving as intended — under a
good VAD it stops crying wolf about quiet speech and reports only the genuine
anomaly.

Silero needs `torch`, `torchaudio` and `silero-vad`, which the offline suites do
not have; install them into a venv and point `PYTHON_BIN` at it rather than
adding a hard dependency. On a CPU-only machine take torch and torchaudio from
`--index-url https://download.pytorch.org/whl/cpu`, or `silero-vad` will pull a
CUDA torchaudio that fails on `libcudart`.

### Both findings are caught

A full run at the current `-45dB` default removes 39.1%, finishes without
`--force`, and warns:

```
! 2 transcribed word(s) on speaker fall inside cuts nothing asked for:
  "know right". The VAD heard silence where the transcript has words …
```

At the old `-35dB` the same warning named four words, `know I don't right`, and
the run reached 52.3% — over `MAX_CUT_FRACTION`, so it refused without `--force`.
Both are the rails working: the refusal was reporting a misconfiguration, not
being fussy, and lowering the threshold removed the cause rather than the symptom.

Either way the repetition the LLM removed on purpose is correctly left out of the
count. The residual `right` is the Whisper hallucination and is expected to stay
until something reconciles the two notions of speech.

This sample is the regression test for that warning in its real form. The unit
tests cover the logic; only real audio produces speech at −39 dB and a
hallucination at −50 dB.
