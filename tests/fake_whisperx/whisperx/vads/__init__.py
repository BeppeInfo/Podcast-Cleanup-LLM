"""The two VADs, answering from a file like the rest of this fake.

`FAKE_WHISPERX_VAD` names a JSON object keyed by the basename of the audio,
each value a list of [start, end] speech turns in seconds. A name with no entry
is speech from its first sample to its last, which leaves the transcript alone
to decide where the silence is — what every case written before the detector
bounded the map still expects.
"""

from __future__ import annotations

import json
import os
import types
import wave


def _turns(path) -> list:
    table = {}
    source = os.environ.get("FAKE_WHISPERX_VAD", "")
    if source and os.path.isfile(source):
        with open(source, encoding="utf-8") as handle:
            table = json.load(handle)
    name = os.path.basename(path)
    if name in table:
        spans = table[name]
    else:
        with wave.open(path, "rb") as handle:
            spans = [[0.0, handle.getnframes() / float(handle.getframerate())]]
    return [types.SimpleNamespace(start=float(s), end=float(e)) for s, e in spans]


class Silero:
    def __init__(self, **options):
        pass

    @staticmethod
    def preprocess_audio(audio):
        return audio

    def __call__(self, audio):
        return _turns(audio["waveform"])


from .pyannote import Pyannote  # noqa: E402  (needs _turns above)
