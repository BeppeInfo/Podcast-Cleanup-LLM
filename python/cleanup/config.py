"""Settings: where they come from, what they have to look like.

The prose explaining what each one *does* lives in
`podcast-cleanup.conf.example`, which is the file people actually read. What
lives here is the machine-readable half: the default, the shape the value has to
have, and the cross-checks between settings.

**Precedence is command line, then environment, then config file, then default.**
That is a change from the shell, where the config file was sourced over the top
of the environment and so beat it. Environment-first is what a container needs —
one image, settings injected per deployment — and it is the conventional order
everywhere else, so a config file that is present now furnishes values rather
than overriding what was asked for explicitly.

The config file is still bash and is still sourced by bash. Parsing it here
would mean reimplementing quoting, `$VAR` references and command substitution,
and getting one of them subtly wrong on somebody's working config. Instead bash
sources it and prints back the values of the names we know about, which is
exactly the old semantics with no parser of our own.
"""

from __future__ import annotations

import os
import shlex
import subprocess

# Kinds: how a value is checked and converted. Everything is stored as a string
# so that the shell bridge stays lossless; the typed accessors below convert.
STR, INT, NUM, FLAG, CHOICE = "str", "int", "num", "flag", "choice"


class ConfigError(Exception):
    """A setting is missing, malformed, or contradicts another one."""


def _cpu_half() -> str:
    return str(max(1, (os.cpu_count() or 2) // 2))


# name -> (default, kind, choices)
SETTINGS: dict[str, tuple] = {
    # Filesystem layout. The four directories are derived from the root unless
    # set; resolve_paths fills them in once the command line has been applied.
    "PODCAST_ROOT": ("", STR, None),
    "INPUT_DIR": ("", STR, None),
    "OUTPUT_DIR": ("", STR, None),
    "WORK_ROOT": ("", STR, None),
    "FAILED_DIR": ("", STR, None),
    "TRACK_SEPARATOR": ("_", STR, None),
    "INPUT_EXTS": (
        "flac wav wave aiff aif m4a mp4 mka mkv mp3 ogg oga opus wv ape alac",
        STR, None,
    ),
    # Deprecated: an input filter only. validate() folds it into INPUT_EXTS.
    "TRACK_EXT": ("", STR, None),

    # Whisper. The endpoint is required — there is no local mode.
    "WHISPER_ENDPOINT": ("", STR, None),
    "WHISPER_ENDPOINT_PATH": ("/inference", STR, None),
    "WHISPER_API_KEY": ("", STR, None),
    "WHISPER_API_KEY_FILE": ("", STR, None),
    "WHISPER_CHUNK_SECONDS": ("120", NUM, None),
    "WHISPER_REQUEST_TIMEOUT": ("1800", NUM, None),
    "WHISPER_LANG": ("auto", STR, None),

    # Whisper's own Silero pass — the only speech detection there is.
    "WHISPER_VAD": ("1", FLAG, None),
    "WHISPER_VAD_THRESHOLD": ("0.5", NUM, None),
    "WHISPER_VAD_MIN_SPEECH_MS": ("250", INT, None),
    "WHISPER_VAD_MIN_SILENCE_MS": ("1000", INT, None),
    "WHISPER_VAD_SPEECH_PAD_MS": ("300", INT, None),
    "WHISPER_VAD_SAMPLES_OVERLAP": ("0.1", NUM, None),

    "WHISPER_RECOVER": ("1", FLAG, None),
    "SPEECH_MAP_CLIP": ("1", FLAG, None),
    "WHISPER_PROMPT": ("", STR, None),
    "WHISPER_REASK": ("1", FLAG, None),
    "WHISPER_REASK_WORD_SECONDS": ("1.2", NUM, None),
    "WHISPER_REASK_WINDOW": ("5.0", NUM, None),

    # llama.cpp. Also an endpoint, also required unless LLM_ENABLE=0.
    "LLAMA_ENDPOINT": ("", STR, None),
    "LLAMA_API_KEY": ("", STR, None),
    "LLAMA_API_KEY_FILE": ("", STR, None),
    "LLAMA_REQUEST_TIMEOUT": ("600", NUM, None),
    "LLM_API": ("chat", CHOICE, ("chat", "completion")),
    "LLAMA_MODEL_NAME": ("", STR, None),
    "LLM_MAX_REPLY_TOKENS": ("2048", INT, None),
    "LLM_CHECK_SCHEMA": ("1", FLAG, None),
    "LLM_CONCURRENCY": ("1", INT, None),

    # Chunk boundaries: where a long track is split, never what is cut.
    "SPLIT_SILENCE_THRESHOLD": ("-45dB", STR, None),
    "SPLIT_MIN_SILENCE": ("0.30", NUM, None),

    # What gets cut.
    "SPEECH_PAD": ("0.25", NUM, None),
    "SILENCE_MIN_DURATION": ("1.5", NUM, None),
    "SILENCE_KEEP": ("0.40", NUM, None),
    "EDGE_KEEP": ("0.25", NUM, None),
    "CUT_PADDING": ("0.10", NUM, None),
    "MIN_CUT": ("0.15", NUM, None),
    "MUTE_FADE": ("0.030", NUM, None),

    "LLM_ENABLE": ("1", FLAG, None),
    "LLM_CHUNK_WORDS": ("350", INT, None),
    "LLM_CHUNK_OVERLAP": ("40", INT, None),
    "LLM_MAX_EDIT_WORDS": ("12", INT, None),
    "LLM_MAX_EDIT_SECONDS": ("4.0", NUM, None),
    "LLM_MIN_CONFIDENCE": ("0.6", NUM, None),
    "LLM_TEMP": ("0", NUM, None),
    "LLM_ACCEPT_KINDS": ("stutter,repetition,false_start", STR, None),
    "MAX_CUT_FRACTION": ("0.5", NUM, None),

    # Output.
    "OUTPUT_CODEC": ("flac", STR, None),
    "OUTPUT_EXT": ("flac", STR, None),
    "OUTPUT_COMPRESSION": ("8", INT, None),
    "OUTPUT_EXTRA_ARGS": ("", STR, None),
    "OUTPUT_SUFFIX": ("", STR, None),
    "RESAMPLE_TO": ("", STR, None),
    "RENDER_FRAME_SAMPLES": ("512", INT, None),

    # Behaviour.
    "FFMPEG_JOBS": (None, INT, None),          # filled in by defaults()
    "KEEP_WORK": ("0", FLAG, None),
    "KEEP_INPUTS": ("0", FLAG, None),
    "FAILED_ACTION": ("log", CHOICE, ("log", "move")),
    "DURATION_TOLERANCE": ("0.02", NUM, None),
}

# Never written to the log by value. That log is copied into the output
# directory and outlives the work directory, the inputs and the container.
SECRETS = ("WHISPER_API_KEY", "LLAMA_API_KEY")

# Searched in order, first match wins. Mirrors the shell.
CONF_CANDIDATES = (
    "$PODCAST_CLEANUP_CONF",
    "$PWD/podcast-cleanup.conf",
    "${XDG_CONFIG_HOME:-$HOME/.config}/podcast-cleanup/config",
    "/etc/podcast-cleanup.conf",
)


def defaults() -> dict[str, str]:
    out = {name: spec[0] for name, spec in SETTINGS.items()}
    out["FFMPEG_JOBS"] = _cpu_half()
    return out


def from_environment(environ=None) -> dict[str, str]:
    """Defaults overlaid with whatever the environment carries.

    For a stage the launcher invokes: it has already resolved everything —
    file, environment, command line — and exported the answers, so re-reading
    the config file here could only disagree with it. No file is consulted.
    """
    environ = os.environ if environ is None else environ
    settings = defaults()
    for name in SETTINGS:
        if environ.get(name) is not None:
            settings[name] = environ[name]
    # Only so the dump can name it; the file itself is not re-read here.
    if environ.get("PODCAST_CONFIG_FILE"):
        settings["_CONFIG_FILE"] = environ["PODCAST_CONFIG_FILE"]
    return settings


def find_config_file(environ=None) -> str:
    environ = os.environ if environ is None else environ
    explicit = environ.get("PODCAST_CLEANUP_CONF") or ""
    if explicit and os.path.isfile(explicit):
        return explicit
    home = environ.get("HOME", "")
    xdg = environ.get("XDG_CONFIG_HOME") or (os.path.join(home, ".config") if home else "")
    for path in (
        os.path.join(environ.get("PWD", os.getcwd()), "podcast-cleanup.conf"),
        os.path.join(xdg, "podcast-cleanup", "config") if xdg else "",
        "/etc/podcast-cleanup.conf",
    ):
        if path and os.path.isfile(path):
            return path
    return ""


def read_config_file(path: str) -> dict[str, str]:
    """Source the file in bash and read back the names we know about.

    Only names in SETTINGS are read, and only ones the file actually set — an
    unset name must fall through to the environment, not arrive as an empty
    string that shadows it. `declare -p` after sourcing tells us which is which.
    """
    if not os.path.isfile(path):
        raise ConfigError(f"config file not found: {path}")
    names = " ".join(SETTINGS)
    script = f'''
set -u
source {shlex.quote(path)} || exit 3
for name in {names}; do
    if [[ -n "${{!name+set}}" ]]; then
        printf '%s=%s\\0' "$name" "${{!name}}"
    fi
done
'''
    result = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=False,
    )
    if result.returncode == 3:
        raise ConfigError(f"failed to parse config file: {path}")
    if result.returncode != 0:
        detail = (result.stderr or "").strip().splitlines()
        raise ConfigError(
            f"config file {path} could not be read"
            + (f": {detail[-1]}" if detail else "")
        )
    found = {}
    for chunk in result.stdout.split("\0"):
        if "=" in chunk:
            name, _, value = chunk.partition("=")
            found[name] = value
    return found


def load(config_file: str = "", overrides=None, environ=None) -> dict[str, str]:
    """Assemble the settings. Later sources win over earlier ones."""
    environ = os.environ if environ is None else environ
    settings = defaults()

    path = config_file or find_config_file(environ)
    if config_file and not os.path.isfile(config_file):
        raise ConfigError(f"config file not found: {config_file}")
    if path:
        settings.update(read_config_file(path))
        settings["_CONFIG_FILE"] = path

    for name in SETTINGS:
        if environ.get(name) is not None:
            settings[name] = environ[name]

    # An empty override is meaningful and is applied: `--root` clears the four
    # directories so they re-derive from the new root, and only a value the
    # caller actually passed reaches here at all.
    for name, value in (overrides or {}).items():
        if value is not None:
            settings[name] = str(value)
    return settings


def _abs(path: str, base: str) -> str:
    return os.path.normpath(
        path if os.path.isabs(path) else os.path.join(base, path)
    )


def resolve_paths(settings: dict, script_root: str) -> None:
    """Fill in the layout, in place. Run after the command line is applied."""
    root = settings.get("PODCAST_ROOT") or script_root
    root = _abs(root, os.getcwd())
    settings["PODCAST_ROOT"] = root
    for name, leaf in (
        ("INPUT_DIR", "incoming"), ("OUTPUT_DIR", "output"),
        ("WORK_ROOT", "work"), ("FAILED_DIR", "failed"),
    ):
        settings[name] = _abs(settings.get(name) or os.path.join(root, leaf), root)


def read_key_file(label: str, path: str, warn=None) -> str:
    if not os.path.isfile(path):
        raise ConfigError(f"{label} file not found: {path}")
    if not os.access(path, os.R_OK):
        raise ConfigError(f"{label} file is not readable: {path}")
    # A secret readable by anyone on the box is not much of a secret.
    mode = os.stat(path).st_mode & 0o777
    if mode & 0o044 and warn:
        warn(f"{label} file {path} is readable beyond its owner "
             f"(mode {mode:03o}); chmod 600 it")
    with open(path, encoding="utf-8") as handle:
        value = (handle.readline() or "").strip()
    if not value:
        raise ConfigError(f"{label} file {path} is empty")
    return value


def resolve_api_keys(settings: dict, warn=None) -> None:
    for label in ("WHISPER_API_KEY", "LLAMA_API_KEY"):
        path = settings.get(f"{label}_FILE") or ""
        if path:
            settings[label] = read_key_file(label, path, warn)


def as_int(settings, name) -> int:
    return int(str(settings[name]).strip())


def as_num(settings, name) -> float:
    return float(str(settings[name]).strip())


def as_flag(settings, name) -> bool:
    return str(settings[name]).strip() == "1"


def validate(settings: dict, warn=None) -> None:
    """Every check the shell did, in the order it did them."""
    for name, (_, kind, choices) in SETTINGS.items():
        raw = str(settings.get(name, "")).strip()
        if kind in (INT, NUM):
            try:
                float(raw) if kind == NUM else int(raw)
            except ValueError:
                raise ConfigError(
                    f"{name} must be {'a number' if kind == NUM else 'an integer'}"
                    f", got '{raw}'"
                ) from None
        elif kind == CHOICE and raw not in choices:
            raise ConfigError(
                f"{name} must be {' or '.join(repr(c) for c in choices)}, "
                f"got '{raw}'"
            )
        elif kind == FLAG and raw not in ("0", "1"):
            raise ConfigError(f"{name} must be 0 or 1, got '{raw}'")

    if as_int(settings, "LLM_CONCURRENCY") < 1:
        raise ConfigError(
            f"LLM_CONCURRENCY must be at least 1, got {settings['LLM_CONCURRENCY']}")

    frame = as_int(settings, "RENDER_FRAME_SAMPLES")
    if not 64 <= frame <= 8192:
        raise ConfigError(
            f"RENDER_FRAME_SAMPLES must be between 64 and 8192, got {frame}")

    resample = str(settings["RESAMPLE_TO"]).strip()
    if resample not in ("", "auto"):
        if not resample.isdigit():
            raise ConfigError(
                "RESAMPLE_TO must be empty, 'auto', or a sample rate, "
                f"got '{resample}'")
        if int(resample) < 8000:
            raise ConfigError(f"RESAMPLE_TO looks too low: {resample}")

    if settings.get("TRACK_EXT"):
        settings["INPUT_EXTS"] = settings["TRACK_EXT"]
        if warn:
            warn("TRACK_EXT is deprecated: it now only restricts which inputs "
                 "are found. Use INPUT_EXTS for that and OUTPUT_CODEC/OUTPUT_EXT "
                 "to choose the output format (currently "
                 f"{settings['OUTPUT_CODEC']}/.{settings['OUTPUT_EXT']})")

    if not settings["INPUT_EXTS"].strip():
        raise ConfigError("INPUT_EXTS is empty; nothing could ever be found")
    if not settings["OUTPUT_EXT"].isalnum():
        raise ConfigError(
            f"OUTPUT_EXT should be a bare extension, got '{settings['OUTPUT_EXT']}'")

    if as_int(settings, "LLM_CHUNK_OVERLAP") >= as_int(settings, "LLM_CHUNK_WORDS"):
        raise ConfigError(
            f"LLM_CHUNK_OVERLAP ({settings['LLM_CHUNK_OVERLAP']}) must be smaller "
            f"than LLM_CHUNK_WORDS ({settings['LLM_CHUNK_WORDS']})")

    if as_num(settings, "SILENCE_KEEP") >= as_num(settings, "SILENCE_MIN_DURATION"):
        raise ConfigError(
            f"SILENCE_KEEP ({settings['SILENCE_KEEP']}) must be smaller than "
            f"SILENCE_MIN_DURATION ({settings['SILENCE_MIN_DURATION']})")

    if not settings["TRACK_SEPARATOR"]:
        raise ConfigError("TRACK_SEPARATOR must not be empty")


def dump(settings: dict, log) -> None:
    """Every effective setting, into the run log.

    Driven by SETTINGS rather than by a list kept alongside it. The shell kept
    such a list and it had drifted six names behind by the time this moved, so
    six settings could change a run without the log admitting it.

    run_episode calls this, which is what puts it on both front ends: a web
    run's log is the only part of the job that survives the download, and it
    was not saying what produced the audio.
    """
    log.debug("config file: "
              f"{settings.get('_CONFIG_FILE') or '<none, using defaults>'}")
    # Presence and length, never the value: enough to tell "the wrong key" from
    # "no key at all" when reading the log of a run nobody can repeat.
    for name in SECRETS:
        value = str(settings.get(name) or "")
        log.raw(f"  config {name}=" + (
            f"<set, {len(value)} chars, redacted>" if value else "<unset>"))
    for name in SETTINGS:
        if name not in SECRETS:
            log.raw(f"  config {name}={settings.get(name, '')}")


def to_shell(settings: dict) -> str:
    """Shell assignments for the launcher, quoted so any value survives."""
    lines = []
    for name in SETTINGS:
        lines.append(f"{name}={shlex.quote(str(settings.get(name, '')))}")
    if settings.get("_CONFIG_FILE"):
        lines.append(f"CONFIG_FILE={shlex.quote(settings['_CONFIG_FILE'])}")
    return "\n".join(lines)
