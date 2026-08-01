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

Expect roughly 42% of it to be removed as silence — it is mostly lead-in and
lead-out — which is close to `MAX_CUT_FRACTION` and would trip the safety refusal
if that were lowered.
