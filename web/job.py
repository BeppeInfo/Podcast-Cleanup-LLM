"""One episode at a time, and what it is doing.

Deliberately not a queue. The pipeline holds a whole episode's audio in a work
directory and talks to two model servers that serialise requests anyway, so a
second concurrent run would contend for both and finish neither sooner. One lock,
one job, and the upload form is closed while it is held — which is also the whole
of the concurrency design, and why there is no database here.

The run happens on a thread because a request must not block on something that
takes minutes. Everything the page needs is read back through `status()`, which
takes the same lock, so a half-written state is never rendered.
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
import time

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

from cleanup import discover, pipeline, runlog  # noqa: E402

IDLE, RUNNING, DONE, FAILED = "idle", "running", "done", "failed"

# Kept for the page; the run log on disk is the complete record.
RECENT_LINES = 40


class Busy(Exception):
    """A run is already in progress."""


class WebLog(runlog.Log):
    """A run log that also remembers enough to draw a progress page.

    The console format is unchanged — the log file this writes is the same one
    the CLI would produce — so nothing here is a second way of reporting. It is
    the same reporting, observed.
    """

    def __init__(self, path: str, stages, **kwargs):
        super().__init__(path=path, colour=False, **kwargs)
        self.stage_status = {name: "pending" for name in stages}
        self.current = ""
        self.recent: list[str] = []

    def _remember(self, text: str) -> None:
        self.recent.append(text)
        del self.recent[:-RECENT_LINES]

    def stage_begin(self, name, description=""):
        super().stage_begin(name, description)
        self.stage_status[name] = "running"
        self.current = name

    def stage_end(self, note=""):
        if self.stage_name:
            self.stage_status[self.stage_name] = "done"
        super().stage_end(note)
        self.current = ""

    def stage_skip(self, name, reason):
        super().stage_skip(name, reason)
        self.stage_status[name] = "skipped"

    def info(self, message):
        super().info(message)
        self._remember(message)

    def ok(self, message):
        super().ok(message)
        self._remember(message)

    def warn(self, message):
        super().warn(message)
        self._remember(f"! {message}")

    def error(self, message):
        super().error(message)
        self._remember(f"✗ {message}")


class Runner:
    """The single job slot."""

    def __init__(self, root: str):
        self.root = root
        self._lock = threading.Lock()
        self._state = IDLE
        self._episode = ""
        self._error = ""
        self._log: WebLog | None = None
        self._started = 0.0
        self._finished = 0.0
        self._thread: threading.Thread | None = None

    # --- what the page reads -------------------------------------------------

    def status(self) -> dict:
        with self._lock:
            log = self._log
            stages = (list(log.stage_status.items()) if log else
                      [(name, "pending") for name in pipeline.ALL_STAGES])
            elapsed = ((self._finished or time.monotonic()) - self._started
                       if self._started else 0.0)
            return {
                "state": self._state,
                "episode": self._episode,
                "error": self._error,
                "stages": stages,
                "current": log.current if log else "",
                "recent": list(log.recent) if log else [],
                "elapsed": runlog.fmt_duration(elapsed),
                "busy": self._state == RUNNING,
                "output": self.output_dir(self._episode) if self._episode else "",
            }

    def output_dir(self, episode: str) -> str:
        return os.path.join(self.root, "output", episode)

    def outputs(self, episode: str) -> list[str]:
        """Published files, newest layout first, for the download list."""
        where = self.output_dir(episode)
        if not os.path.isdir(where):
            return []
        return sorted(
            name for name in os.listdir(where)
            if os.path.isfile(os.path.join(where, name)))

    # --- starting one --------------------------------------------------------

    def accept(self, episode: str, uploads, settings) -> None:
        """Write the uploads under the names the pipeline expects, then run.

        The filename convention is the CLI's input handling; here the form asks
        for an episode and a name per track, and this writes
        `<episode><sep><participant>.<ext>` so the user never meets it.
        """
        with self._lock:
            if self._state == RUNNING:
                raise Busy("a run is already in progress")
            self._state = RUNNING
            self._episode = episode
            self._error = ""
            self._started = time.monotonic()
            self._finished = 0.0

        incoming = os.path.join(self.root, "incoming")
        os.makedirs(incoming, exist_ok=True)
        separator = settings.get("TRACK_SEPARATOR", "_")
        written = []
        try:
            for participant, stream, extension in uploads:
                target = os.path.join(
                    incoming, f"{episode}{separator}{participant}.{extension}")
                # Werkzeug's upload has .save(); a plain file object does not,
                # which is what the tests hand it.
                if hasattr(stream, "save"):
                    stream.save(target)
                else:
                    with open(target, "wb") as handle:
                        shutil.copyfileobj(stream, handle)
                written.append(target)
        except Exception:
            for path in written:
                if os.path.isfile(path):
                    os.remove(path)
            with self._lock:
                self._state = IDLE
            raise

        stages = list(pipeline.ALL_STAGES)
        work_log = os.path.join(
            discover.episode_paths(episode, settings["WORK_ROOT"],
                                   settings["OUTPUT_DIR"])["work"],
            "logs", "run.log")
        os.makedirs(os.path.dirname(work_log), exist_ok=True)
        log = WebLog(work_log, stages, level="info")
        with self._lock:
            self._log = log

        self._thread = threading.Thread(
            target=self._run, args=(episode, settings, log, stages, written),
            daemon=True)
        self._thread.start()

    def _run(self, episode, settings, log, stages, written) -> None:
        try:
            status = pipeline.run_episode(
                settings, log, stages,
                episode_override=episode,
                input_files=written,
                api_keys={"whisper": os.environ.get("PODCAST_WHISPER_API_KEY"),
                          "llama": os.environ.get("PODCAST_LLAMA_API_KEY")},
            )
            failed = status != 0
            message = "the run failed; the log has the detail" if failed else ""
        except Exception as exc:                      # noqa: BLE001
            # A crash here would otherwise leave the page saying "running"
            # forever, which is worse than an ugly message.
            failed, message = True, f"{type(exc).__name__}: {exc}"
            log.error(message)

        with self._lock:
            self._state = FAILED if failed else DONE
            self._error = message
            self._finished = time.monotonic()

    # --- afterwards ----------------------------------------------------------

    def discard(self, episode: str) -> None:
        """Remove a finished episode's outputs. What "downloaded" leads to."""
        with self._lock:
            if self._state == RUNNING:
                raise Busy("a run is in progress")
        where = self.output_dir(episode)
        if os.path.isdir(where):
            shutil.rmtree(where, ignore_errors=True)
        with self._lock:
            if self._episode == episode:
                self._state = IDLE
                self._episode = ""
                self._log = None

    def reset(self) -> None:
        """Back to an empty form, leaving whatever is on disk alone."""
        with self._lock:
            if self._state == RUNNING:
                raise Busy("a run is in progress")
            self._state = IDLE
            self._episode = ""
            self._error = ""
            self._log = None
