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

import shlex
import subprocess

TAIL_LINES = 20


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
