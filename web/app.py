"""The web interface: upload an episode, watch it, take the result away.

Server-side rendered on purpose. There is one job, a handful of states and a
form; a page that re-renders itself every couple of seconds says everything a
client-side app would, and it works with JavaScript off. The progress page uses
`<meta http-equiv="refresh">` rather than polling, which is the smallest thing
that does the job.

This is a local tool. It has no authentication and it hands ffmpeg whatever is
uploaded, so it belongs on a machine you trust, not on the internet. That is
stated again in the README and it is not a detail.
"""

from __future__ import annotations

import io
import os
import sys
import zipfile

from flask import (Flask, redirect, render_template, request, send_file,
                   url_for)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import job as job_module  # noqa: E402
import store  # noqa: E402
from cleanup import config as cfg  # noqa: E402

# Anything ffmpeg can open; the pipeline decodes everything to PCM anyway, so
# this only decides what the form will accept.
ALLOWED = {"flac", "wav", "wave", "aiff", "aif", "m4a", "mp4", "mka", "mkv",
           "mp3", "ogg", "oga", "opus", "wv", "ape", "alac"}

MAX_UPLOAD_BYTES = 8 * 1024 * 1024 * 1024      # two hours of 24-bit stereo, twice


def clean_name(raw: str, fallback: str) -> str:
    """A filename component that cannot escape the directory it belongs in."""
    kept = "".join(c for c in (raw or "").strip() if c.isalnum() or c in "-_")
    return kept or fallback


def create_app(root: str | None = None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    data_root = root or os.environ.get("PODCAST_ROOT") or "/data"
    app.config["DATA_ROOT"] = data_root
    runner = job_module.Runner(data_root)
    app.config["RUNNER"] = runner

    def settings_now():
        values = store.effective(data_root)
        # The layout is the container's, never the form's.
        values["PODCAST_ROOT"] = data_root
        cfg.resolve_paths(values, data_root)
        return values

    @app.get("/")
    def index():
        status = runner.status()
        return render_template(
            "index.html", status=status,
            outputs=runner.outputs(status["episode"]) if status["episode"] else [],
            refresh=2 if status["busy"] else 0)

    @app.post("/start")
    def start():
        if runner.status()["busy"]:
            return redirect(url_for("index"))

        episode = clean_name(request.form.get("episode"), "episode")
        uploads = []
        for index, storage in enumerate(request.files.getlist("track")):
            if not storage or not storage.filename:
                continue
            extension = storage.filename.rpartition(".")[2].lower()
            if extension not in ALLOWED:
                return render_template(
                    "index.html", status=runner.status(), outputs=[], refresh=0,
                    error=f"{storage.filename}: not an audio file this can read"), 400
            participant = clean_name(
                request.form.getlist("participant")[index]
                if index < len(request.form.getlist("participant")) else "",
                f"track{index + 1}")
            uploads.append((participant, storage, extension))

        if len(uploads) < 1:
            return render_template(
                "index.html", status=runner.status(), outputs=[], refresh=0,
                error="Add at least one track."), 400
        names = [name for name, _, _ in uploads]
        if len(set(names)) != len(names):
            return render_template(
                "index.html", status=runner.status(), outputs=[], refresh=0,
                error="Two tracks have the same participant name."), 400

        try:
            runner.accept(episode, uploads, settings_now())
        except job_module.Busy:
            pass
        return redirect(url_for("index"))

    @app.get("/download/<episode>")
    def download(episode):
        """Everything the run produced, as one zip.

        Downloading is what "done" leads to: the outputs go once the response
        has been sent. The zip is built in memory first so that a failure while
        building it cannot leave the directory half-deleted.
        """
        episode = clean_name(episode, "")
        where = runner.output_dir(episode)
        if not episode or not os.path.isdir(where):
            return redirect(url_for("index"))

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
            for folder, _, names in os.walk(where):
                for name in sorted(names):
                    full = os.path.join(folder, name)
                    archive.write(full, os.path.relpath(full, where))
        buffer.seek(0)

        keep = request.args.get("keep") == "1"
        if not keep:
            runner.discard(episode)
        return send_file(buffer, mimetype="application/zip", as_attachment=True,
                         download_name=f"{episode}.zip")

    @app.post("/discard")
    def discard():
        try:
            runner.discard(clean_name(request.form.get("episode"), ""))
        except job_module.Busy:
            pass
        return redirect(url_for("index"))

    @app.post("/reset")
    def reset():
        try:
            runner.reset()
        except job_module.Busy:
            pass
        return redirect(url_for("index"))

    @app.get("/settings")
    def settings_page():
        return render_template(
            "settings.html", groups=store.GROUPS, values=settings_now(),
            spec=store.field_spec, saved=store.load(data_root),
            busy=runner.status()["busy"])

    @app.post("/settings")
    def settings_save():
        # An unchecked box is simply absent from the post, so "off" and "this
        # form did not carry the field" look identical. The form names what it
        # rendered in `present`, and only those are touched — otherwise a
        # partial post would silently switch features off.
        present = set(request.form.getlist("present"))
        proposed = dict(settings_now())
        for name in store.EDITABLE:
            if name not in present:
                continue
            if cfg.SETTINGS[name][1] == cfg.FLAG:
                proposed[name] = "1" if request.form.get(name) else "0"
            elif name in request.form:
                proposed[name] = request.form[name].strip()

        problems = store.validate(proposed)
        if problems:
            return render_template(
                "settings.html", groups=store.GROUPS, values=proposed,
                spec=store.field_spec, saved=store.load(data_root),
                busy=runner.status()["busy"], problems=problems), 400

        store.save(data_root, proposed)
        return redirect(url_for("settings_page", saved="1"))

    @app.post("/settings/reset")
    def settings_reset():
        store.save(data_root, cfg.defaults())
        return redirect(url_for("settings_page", saved="1"))

    return app


def main() -> int:
    root = os.environ.get("PODCAST_ROOT", "/data")
    os.makedirs(root, exist_ok=True)
    app = create_app(root)
    app.run(host=os.environ.get("PODCAST_WEB_HOST", "0.0.0.0"),
            port=int(os.environ.get("PODCAST_WEB_PORT", "8000")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
