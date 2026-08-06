"""Which settings the web interface offers, and where its answers are kept.

Not all 58 settings belong on a form. Three kinds are deliberately absent:

**Paths.** `PODCAST_ROOT` and the four directories under it are how the container
is wired to its volume. Letting a page rewrite them would let a page write
anywhere the process can.

**Naming.** `TRACK_SEPARATOR` and `INPUT_EXTS` exist so a human can drop files in
a directory and have them understood. The upload form asks for the episode and
each participant by name and writes the filenames itself, so the convention never
reaches the user and nothing would be served by letting them change it.

**Secrets.** `LLAMA_API_KEY` is not here. Saving it would write it in clear to
the volume, and there is already a better route: the `_FILE` variant, or the
environment. A form that quietly makes a secret less safe is worse than no form.
There is no whisper key any more — transcription is not a server.

What is left is the tuning: what counts as silence, how bold the disfluency
detection is, what the servers are, and what comes out. Those are the ones worth
changing between episodes, which is the whole reason for the page.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

from cleanup import config as cfg  # noqa: E402

SETTINGS_FILE = "settings.json"


class Group:
    def __init__(self, key: str, title: str, blurb: str, fields):
        self.key = key
        self.title = title
        self.blurb = blurb
        self.fields = fields


# (name, label, hint). The hint is for what the label cannot carry on its own;
# the full reasoning for every one of these is in podcast-cleanup.conf.example.
GROUPS = [
    Group("editing", "Silence", "What counts as dead air, and how much is left behind.", [
        ("SILENCE_MIN_DURATION", "Shorten gaps longer than",
         "seconds; a gap needs this plus twice the speech padding"),
        ("SILENCE_KEEP", "Quiet left behind", "seconds; removing it all sounds gasped"),
        ("EDGE_KEEP", "Kept at the start and end", "seconds"),
        ("SPEECH_PAD", "Speech padding",
         "seconds of margin around every word, for Whisper's timing error"),
        ("CUT_PADDING", "Cut padding", "seconds claimed around a disfluency"),
        ("MIN_CUT", "Shortest cut worth making", "seconds"),
        ("MUTE_FADE", "Mute fade", "seconds at each end of a muted span"),
        ("MAX_CUT_FRACTION", "Refuse a plan removing more than",
         "fraction of the episode; almost always a wrong threshold"),
    ]),
    Group("detect", "Disfluency detection",
          "What the model is asked to find, and how much of it to believe.", [
        ("LLM_ENABLE", "Enabled", "off means silence editing only"),
        ("LLM_ACCEPT_KINDS", "Kinds to remove",
         "comma separated: stutter, repetition, false_start, filler"),
        ("LLM_MIN_CONFIDENCE", "Minimum confidence", "0 to 1"),
        ("LLM_MAX_EDIT_WORDS", "Longest edit, in words", ""),
        ("LLM_MAX_EDIT_SECONDS", "Longest edit, in seconds", ""),
        ("LLM_CHUNK_WORDS", "Words per request", ""),
        ("LLM_CHUNK_OVERLAP", "Overlap between requests", "words"),
        ("LLM_CONCURRENCY", "Requests in flight",
         "must match the server's --parallel; speed only"),
    ]),
    Group("transcribe", "Transcription",
          "What reaches the detector at all — see the note below the prompt.", [
        ("WHISPER_MODEL", "Model",
         "tiny, base, small, medium, large-v3 — bigger is a better transcript "
         "and, on a CPU, a much longer wait"),
        ("WHISPER_DEVICE", "Device", "cuda needs an NVIDIA card; Radeon means cpu"),
        ("WHISPER_COMPUTE_TYPE", "Precision", "int8 on a CPU; float16 needs cuda"),
        ("WHISPER_BATCH_SIZE", "Batch size", "speed against memory"),
        ("WHISPER_THREADS", "Threads", "0 lets the runtime decide"),
        ("WHISPER_LANG", "Language", "a code, or auto"),
        ("WHISPER_PROMPT", "Initial prompt",
         "conditioning text, not an instruction; empty means Whisper returns "
         "fluent prose and the disfluencies never reach the detector"),
        ("WHISPER_VAD_METHOD", "Speech detection",
         "silero is what whisper-server used; pyannote needs no runtime "
         "download but is a different detector"),
        ("WHISPER_VAD_ONSET", "Speech starts above", "0 to 1"),
        ("WHISPER_VAD_OFFSET", "Speech ends below",
         "0 to 1; pyannote only, silero uses the onset alone"),
        ("SPEECH_MAP_CLIP", "Bound word timings by the level scan",
         "off means a word stretched across silence protects all of it"),
    ]),
    Group("servers", "Server",
          "The detector is reached over HTTP and is not started here. "
          "Transcription runs in this process and needs no server at all.", [
        ("LLAMA_ENDPOINT", "Llama endpoint", "e.g. http://127.0.0.1:8080"),
        ("LLAMA_MODEL_NAME", "Model name", "required by a router-mode server"),
        ("LLM_API", "API", ""),
        ("LLM_CHECK_SCHEMA", "Check the schema before starting",
         "one request that catches a server which would drop every window"),
        ("LLAMA_REQUEST_TIMEOUT", "Llama timeout", "seconds"),
    ]),
    Group("output", "Output", "", [
        ("OUTPUT_CODEC", "Codec", ""),
        ("OUTPUT_EXT", "Extension", ""),
        ("OUTPUT_COMPRESSION", "FLAC compression", "0 to 12"),
        ("OUTPUT_SUFFIX", "Filename suffix", ""),
        ("RESAMPLE_TO", "Resample to", "a rate, auto, or empty to refuse a mismatch"),
    ]),
]

EDITABLE = [name for group in GROUPS for name, _, _ in group.fields]


def _path(root: str) -> str:
    return os.path.join(root, SETTINGS_FILE)


def load(root: str) -> dict[str, str]:
    """The saved overrides. Only what was changed; anything else is a default.

    Storing the whole set would freeze this deployment's values against a later
    image whose defaults are better. Storing only the differences means an
    untouched setting keeps following the default.
    """
    try:
        with open(_path(root), encoding="utf-8") as handle:
            saved = json.load(handle)
    except (OSError, ValueError):
        return {}
    return {name: str(value) for name, value in saved.items()
            if name in EDITABLE}


def save(root: str, values) -> dict[str, str]:
    """Write the overrides, and return what was written."""
    defaults = cfg.defaults()
    overrides = {
        name: str(values[name]) for name in EDITABLE
        if name in values and str(values[name]) != str(defaults.get(name, ""))
    }
    os.makedirs(root, exist_ok=True)
    tmp = _path(root) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(overrides, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, _path(root))
    return overrides


def effective(root: str, environ=None) -> dict[str, str]:
    """The settings a run would use: saved answers over the deployment's own.

    Saved values sit above the environment on purpose. The environment is how
    the container was configured; the form is what the person operating it asked
    for since, and that is the more specific statement.
    """
    settings = cfg.from_environment(environ)
    saved = load(root)
    settings.update(saved)
    # So the run log names what the values came from, the way the CLI's dump
    # names a config file. Only when the form has actually saved something —
    # otherwise the file does not exist and the environment is the whole story.
    if saved:
        settings["_CONFIG_FILE"] = _path(root)
    return settings


def validate(values) -> list[str]:
    """Complaints about a proposed set, in the words the CLI would use."""
    candidate = cfg.defaults()
    candidate.update({k: v for k, v in values.items() if k in EDITABLE})
    problems = []
    try:
        cfg.validate(candidate)
    except cfg.ConfigError as exc:
        problems.append(str(exc))
    return problems


def field_spec(name: str) -> dict:
    """What the form needs to render one setting."""
    default, kind, choices = cfg.SETTINGS[name]
    if default is None:
        default = cfg.defaults()[name]
    return {"name": name, "kind": kind, "choices": choices, "default": default}
