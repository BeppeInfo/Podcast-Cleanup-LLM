"""A stand-in for the whisperx package, for the self-test.

The real one is three gigabytes of torch and downloads model weights on first
use. Neither belongs in a test that has to run offline in under two minutes, so
this takes its place on `PYTHONPATH` and answers from a file — the same bargain
the stub whisper-server made, and it reads the same fixtures.

`FAKE_WHISPERX_RESPONSES` names a JSON object keyed by the basename of the audio
handed to `load_audio`. Each value is either whisper.cpp's shape (a
`transcription` list of one-word segments with millisecond `offsets`) or the
OpenAI-ish one (`segments` with `start`/`end` in seconds). Both are what the old
stub answered with, so the fixtures did not have to be rewritten to come here.

A name with no entry transcribes as silence, which is a real answer: one
participant is quiet for minutes while the other talks.
"""

from __future__ import annotations

import json
import os


def _responses() -> dict:
    path = os.environ.get("FAKE_WHISPERX_RESPONSES", "")
    if not path or not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _words_from_reply(reply) -> list[dict]:
    """Either fixture shape, as words with seconds."""
    words = []
    for item in reply.get("transcription") or []:
        offsets = item.get("offsets") or {}
        if "from" not in offsets or "to" not in offsets:
            continue
        words.append({"word": (item.get("text") or "").strip(),
                      "start": offsets["from"] / 1000.0,
                      "end": offsets["to"] / 1000.0})
    for item in reply.get("segments") or []:
        if item.get("start") is None or item.get("end") is None:
            continue
        words.append({"word": (item.get("text") or "").strip(),
                      "start": float(item["start"]), "end": float(item["end"])})
    return [word for word in words if word["word"]]


class _Model:
    def __init__(self, language):
        self._language = language

    # The real FasterWhisperPipeline.transcribe takes exactly these. Keeping the
    # signature narrow is deliberate: a fake that accepted **anything would have
    # hidden the initial_prompt mismatch that the real package raises on.
    def transcribe(self, audio, batch_size=None, num_workers=0, language=None,
                   task=None, chunk_size=30, print_progress=False,
                   combined_progress=False, verbose=False,
                   progress_callback=None):
        words = _words_from_reply(_responses().get(os.path.basename(audio), {}))
        if not words:
            return {"segments": [], "language": language or self._language or "en"}
        # One segment carrying the lot; align() is what splits it into words,
        # and this fake's align is where they come back out.
        return {
            "segments": [{"start": words[0]["start"], "end": words[-1]["end"],
                          "text": " ".join(w["word"] for w in words),
                          "words": words}],
            "language": language or self._language or "en",
        }


def load_model(whisper_arch, device="cpu", compute_type="default",
               asr_options=None, language=None, vad_method=None,
               vad_options=None, threads=4, **kwargs):
    return _Model(language)


def load_audio(path):
    # The real one returns samples; nothing in the pipeline looks inside, and
    # this fake needs the name to answer from the fixtures.
    return path


def load_align_model(language_code, device):
    return ("fake-align-model", {"language": language_code})


def align(segments, model, meta, audio, device, return_char_alignments=False):
    return {"segments": segments}
