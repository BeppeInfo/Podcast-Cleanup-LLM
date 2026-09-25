"""pyannote's VAD and the Binarize that thresholds it, with nothing behind them."""

from __future__ import annotations

import types


class Pyannote:
    def __init__(self, device, token=None, **options):
        pass

    @staticmethod
    def preprocess_audio(audio):
        return audio

    def __call__(self, audio):
        from . import _turns
        return _turns(audio["waveform"])


class Binarize:
    def __init__(self, **options):
        pass

    def __call__(self, scores):
        return types.SimpleNamespace(get_timeline=lambda: scores)
