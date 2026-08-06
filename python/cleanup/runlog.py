"""The run log and the console display.

Once a port of `lib/log.sh`, format for format, so that a line written here and
a line written there were indistinguishable while stages moved across. That file
is gone and this is the only implementation; the format survives it because it is
what the console and the kept log have always looked like, not because something
else still has to match. The launcher's one remaining error line is copied from
`Log.error` and pinned by a test.

Two audiences, deliberately different:

**The log file** gets everything, timestamped, with the level and stage named.
It is copied into the output directory and outlives the work directory, so it
is the record of what happened.

**stderr** gets what the level allows, without timestamps, and is where the
in-place progress counter lives. stdout is left alone: subcommands use it for
data the shell `eval`s or reads, and a stray log line there has already broken
this once.
"""

from __future__ import annotations

import os
import sys
import time

LEVELS = {"debug": 10, "info": 20, "warn": 30, "error": 40}


def fmt_duration(seconds) -> str:
    total = int(float(seconds)) if seconds else 0
    if total < 0:
        total = 0
    if total >= 3600:
        return f"{total // 3600}h{(total % 3600) // 60:02d}m{total % 60:02d}s"
    if total >= 60:
        return f"{total // 60}m{total % 60:02d}s"
    return f"{total}s"


class Palette:
    """ANSI codes, or nothing at all when the output is not a terminal."""

    NAMES = ("reset", "dim", "bold", "red", "green", "yellow", "blue", "cyan")
    CODES = ("0", "2", "1", "31", "32", "33", "34", "36")

    def __init__(self, enabled: bool):
        for name, code in zip(self.NAMES, self.CODES):
            setattr(self, name, f"\033[{code}m" if enabled else "")


class Log:
    """One run's log: the file that is kept, and the console that is watched."""

    def __init__(self, path: str = "", level: str = "info", stream=None,
                 colour=None):
        self.path = path
        self.level = level if level in LEVELS else "info"
        self.stream = stream if stream is not None else sys.stderr
        if colour is None:
            colour = (
                hasattr(self.stream, "isatty") and self.stream.isatty()
                and not os.environ.get("NO_COLOR")
            )
        self.c = Palette(bool(colour))
        self.stage_name = ""
        self._stage_index = 0
        self._stage_total = 0
        self._stage_started = 0.0
        self._run_started = time.monotonic()
        self._progress_active = False

    # --- construction --------------------------------------------------------

    @classmethod
    def from_env(cls, environ=None, stream=None):
        """The log the shell is already writing to, taken from the environment.

        `PODCAST_LOG_FILE`, `LOG_LEVEL` and `PODCAST_LOG_STAGE` are exported by
        the launcher, so a stage that has moved to Python appends to the same
        file, at the same level, under the same stage name.
        """
        environ = os.environ if environ is None else environ
        log = cls(
            path=environ.get("PODCAST_LOG_FILE", ""),
            level=environ.get("LOG_LEVEL", "info"),
            stream=stream,
        )
        # So a line from a ported stage is tagged with the stage the shell is
        # already displaying, and the log file reads as one sequence.
        log.stage_name = environ.get("PODCAST_LOG_STAGE", "")
        return log

    def init(self, path: str, note: str = "") -> None:
        self.path = path
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        open(path, "a", encoding="utf-8").close()
        self.raw(f"=== run started: {note} ===")

    # --- the log file --------------------------------------------------------

    def raw(self, message: str) -> None:
        """Append to the run log only. A no-op before a path is known."""
        if not self.path:
            return
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(f"{stamp} {message}\n")
        except OSError:
            # Losing the log must not lose the run; the console still has it.
            pass

    # --- the console ---------------------------------------------------------

    def enabled(self, level: str) -> bool:
        return LEVELS.get(level, 20) >= LEVELS[self.level]

    def _clear_progress(self) -> None:
        if self._progress_active:
            self.stream.write("\r\033[K")
            self.stream.flush()
            self._progress_active = False

    def _emit(self, level: str, colour: str, tag: str, message: str) -> None:
        stage = f" ({self.stage_name})" if self.stage_name else ""
        self.raw(f"[{level}]{stage} {message}")
        if not self.enabled(level):
            return
        self._clear_progress()
        self.stream.write(f"{colour}{tag}{self.c.reset} {message}\n")
        self.stream.flush()

    def debug(self, message: str) -> None:
        self._emit("debug", self.c.dim, "  ·", message)

    def info(self, message: str) -> None:
        self._emit("info", self.c.blue, "  →", message)

    def ok(self, message: str) -> None:
        self._emit("info", self.c.green, "  ✓", message)

    def warn(self, message: str) -> None:
        self._emit("warn", self.c.yellow, "  !", message)

    def error(self, message: str) -> None:
        self._emit("error", self.c.red, "  ✗", message)

    def line(self, message: str = "") -> None:
        """A plain, unprefixed line — headers and summaries. Always shown."""
        self.raw(message)
        self._clear_progress()
        self.stream.write(f"{message}\n")
        self.stream.flush()

    def report(self, text: str) -> None:
        """Multi-line output: always logged, shown unless the user asked quiet."""
        for entry in text.splitlines():
            self.raw(entry)
            if self.enabled("info"):
                self._clear_progress()
                self.stream.write(f"      {entry}\n")
        self.stream.flush()

    # --- stages --------------------------------------------------------------

    def stage_total(self, total: int) -> None:
        self._stage_total = int(total)
        self._stage_index = 0

    def stage_begin(self, name: str, description: str = "") -> None:
        self.stage_name = name
        self._stage_started = time.monotonic()
        self._stage_index += 1
        self.raw(f"--- stage {self._stage_index}/{self._stage_total}: {name} "
                 f"{'- ' + description if description else ''}---")
        self._clear_progress()
        tail = f"  {self.c.dim}{description}{self.c.reset}" if description else ""
        self.stream.write(
            f"{self.c.dim}[{self._stage_index}/{self._stage_total}]{self.c.reset} "
            f"{self.c.bold}{name}{self.c.reset}{tail}\n")
        self.stream.flush()

    def stage_end(self, note: str = "") -> None:
        elapsed = fmt_duration(time.monotonic() - self._stage_started)
        self.raw(f"--- stage {self.stage_name} done in {elapsed} "
                 f"{'- ' + note if note else ''}---")
        self._clear_progress()
        self.stream.write(
            f"{self.c.dim}      {note or 'done'} in {elapsed}{self.c.reset}\n")
        self.stream.flush()
        self.stage_name = ""

    def stage_skip(self, name: str, reason: str) -> None:
        self._stage_index += 1
        self.raw(f"--- stage {self._stage_index}/{self._stage_total}: {name} "
                 f"SKIPPED ({reason}) ---")
        self._clear_progress()
        self.stream.write(
            f"{self.c.dim}[{self._stage_index}/{self._stage_total}] {name} "
            f"— skipped ({reason}){self.c.reset}\n")
        self.stream.flush()

    # --- progress ------------------------------------------------------------

    def progress(self, current: int, total: int, label: str = "") -> None:
        """In place on a terminal, one line per item otherwise.

        A non-interactive log with a hundred rewritten lines in it is unreadable,
        and a CI log is exactly where the count matters most.
        """
        self.raw(f"[progress] {current}/{total} {label}")
        if not self.enabled("info"):
            return
        if getattr(self.stream, "isatty", lambda: False)() and self.c.reset:
            self.stream.write(
                f"\r\033[K{self.c.dim}      {self.c.cyan}({current}/{total})"
                f"{self.c.dim} {label}{self.c.reset}")
            self._progress_active = True
        else:
            self.stream.write(f"      ({current}/{total}) {label}\n")
        self.stream.flush()

    def progress_done(self) -> None:
        self._clear_progress()

    def elapsed(self) -> str:
        return fmt_duration(time.monotonic() - self._run_started)
