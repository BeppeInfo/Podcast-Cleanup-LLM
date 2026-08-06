"""Running external commands, and reporting while they run.

The shell's `run_streaming` did three things at once: log every line the command
produced, watch those lines for progress, and get the exit status out. Doing it
here is mostly simpler — a pipeline's exit status needed a temp file in bash and
needs nothing here — but the reporting rules are worth keeping deliberately.

**Everything the command says goes to the run log, nothing to the console.**
ffmpeg is talkative and most of it is uninteresting until something fails, at
which point the last of it is exactly what is wanted. So a failure prints the
tail of the log rather than the whole stream.

**Progress is shown only when one command owns the console.** Several concurrent
writers on one in-place counter garble each other, so a parallel batch reports
per completion instead. That was the shell's rule and it was the right one.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess

TAIL_LINES = 20


class ToolError(Exception):
    """A tool the run needs is missing, or cannot do what the run needs."""


def require_bin(label: str, candidate: str) -> str:
    """The resolved path to a tool. Raises ToolError rather than exiting.

    An explicit path is taken as given; a bare name is looked up on PATH. The
    shell's version could not raise and so returned a status its callers had to
    remember to check — twice, in a `$(...)` where an exit would only have ended
    the subshell.
    """
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    found = shutil.which(candidate)
    if found:
        return found
    raise ToolError(
        f"{label} not found: '{candidate}' is neither an executable path "
        "nor on PATH")


def has_encoder(ffmpeg: str, codec: str) -> bool:
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-h", f"encoder={codec}"],
            capture_output=True, text=True, check=False)
    except OSError:
        return False
    return f"Encoder {codec}" in result.stdout


def resolve_ffmpeg(settings, log, ffmpeg: str = "", ffprobe: str = "",
                   environ=None) -> tuple[str, str]:
    """ffmpeg and ffprobe, resolved and checked against OUTPUT_CODEC.

    The encoder check is up here, before the first stage, rather than at render
    time where the failure would actually happen. Render is last: an ffmpeg that
    cannot build the configured codec would otherwise be discovered after the
    whole episode had been decoded, sent to whisper and analysed.

    FFMPEG_BIN and FFPROBE_BIN name a build to use instead of whatever is on
    PATH. They are read from the environment, not from the settings — they say
    which tool runs this pipeline, not how it should edit.
    """
    environ = os.environ if environ is None else environ
    ffmpeg = require_bin("ffmpeg", ffmpeg or environ.get("FFMPEG_BIN") or "ffmpeg")
    ffprobe = require_bin(
        "ffprobe", ffprobe or environ.get("FFPROBE_BIN") or "ffprobe")
    log.debug(f"ffmpeg: {ffmpeg}")

    codec = str(settings.get("OUTPUT_CODEC", ""))
    if not has_encoder(ffmpeg, codec):
        raise ToolError(
            f"this ffmpeg has no '{codec}' encoder (see: ffmpeg -encoders)")
    return ffmpeg, ffprobe


def run(argv, log, on_line=None, dry_run: bool = False) -> int:
    """Run a command, streaming its output into the run log. Returns its status.

    `on_line` sees every line as it arrives, for progress. It is called inside
    the read loop, so it should be cheap and must not raise.
    """
    log.raw("$ " + " ".join(shlex.quote(str(a)) for a in argv))
    if dry_run:
        log.line(f"{log.c.dim}      would run:{log.c.reset} "
                 + " ".join(shlex.quote(str(a)) for a in argv))
        return 0

    process = subprocess.Popen(
        [str(a) for a in argv],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, errors="replace",
    )
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip("\n")
        log.raw(line)
        if on_line is not None:
            on_line(line)
    status = process.wait()
    log.progress_done()

    if status != 0:
        log.error(f"command failed (exit {status}): {argv[0]}")
        log.line(f"{log.c.dim}{tail_of_log(log)}{log.c.reset}")
    return status


def tail_of_log(log, lines: int = TAIL_LINES) -> str:
    """The last of the run log — what a failing command actually said."""
    if not log.path:
        return ""
    try:
        with open(log.path, encoding="utf-8", errors="replace") as handle:
            return "\n".join(handle.read().splitlines()[-lines:])
    except OSError:
        return ""


def run_to_file(argv, log, destination: str, dry_run: bool = False) -> int:
    """Run a command with its output captured to a file of its own.

    For output that is voluminous and worth keeping separately — ffmpeg's
    silencedetect scan is thousands of lines and stays in the work directory as
    an artefact, rather than burying the run log.
    """
    log.raw("$ " + " ".join(shlex.quote(str(a)) for a in argv) + f" > {destination}")
    if dry_run:
        log.line(f"{log.c.dim}      would run:{log.c.reset} "
                 + " ".join(shlex.quote(str(a)) for a in argv))
        open(destination, "w").close()
        return 0

    with open(destination, "w", encoding="utf-8", errors="replace") as handle:
        status = subprocess.run(
            [str(a) for a in argv], stdout=handle, stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    if status != 0:
        log.error(f"command failed (exit {status}): {argv[0]}")
        try:
            with open(destination, encoding="utf-8", errors="replace") as handle:
                tail = "\n".join(handle.read().splitlines()[-25:])
            log.line(f"{log.c.dim}{tail}{log.c.reset}")
        except OSError:
            pass
    return status


def ffmpeg_progress(log, total_us, label: str):
    """A line handler turning ffmpeg's -progress output into a counter.

    ffmpeg reports `out_time_ms` in microseconds despite the name. Both sides
    are divided down to centiseconds so the numbers stay small enough to read.
    """
    total = int(total_us or 0)

    def on_line(line: str) -> None:
        if not line.startswith("out_time_ms="):
            return
        value = line.partition("=")[2].strip()
        if not value.isdigit() or total <= 0:
            return
        log.progress(int(value) // 10000, total // 10000, label)

    return on_line
