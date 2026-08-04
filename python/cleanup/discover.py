"""Working out which files are one episode, and what to call its participants.

Inputs are named `<episode><separator><participant>.<ext>`. That convention is
the whole of the CLI's input handling, and the errors it can produce are the
first thing anyone meets, so they say what to do rather than what went wrong.

Nothing here touches the filesystem beyond listing a directory: given a list of
paths it is pure, which is what makes the awkward cases — two episodes at once,
one participant in two formats — testable without fixtures.
"""

from __future__ import annotations

import os


class DiscoverError(Exception):
    """The inputs do not describe one episode."""


class NothingToDo(Exception):
    """No candidate files at all. Not a failure: an empty inbox."""


def find_tracks(input_dir: str, extensions) -> list[str]:
    """Every file in `input_dir` with one of these extensions, sorted.

    Matched case-insensitively — some recorders write .WAV, and ffmpeg does not
    care either way. Not recursive: a subdirectory is somebody's own filing.
    """
    wanted = {ext.strip().lower().lstrip(".") for ext in extensions if ext.strip()}
    if not wanted:
        raise DiscoverError("no input extensions configured; nothing could be found")
    if not os.path.isdir(input_dir):
        raise NothingToDo(f"input directory does not exist yet: {input_dir}")

    found = [
        os.path.join(input_dir, name)
        for name in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, name))
        and name.rpartition(".")[2].lower() in wanted
    ]
    if not found:
        raise NothingToDo(
            f"no files matching [{' '.join(sorted(wanted))}] in {input_dir}")
    return sorted(found)


def parse_tracks(paths, separator: str, episode_override: str = ""):
    """(episode_id, {participant: path}) for one episode's worth of tracks."""
    if not separator:
        raise DiscoverError("TRACK_SEPARATOR must not be empty")

    episode_id = ""
    tracks: dict[str, str] = {}
    for path in paths:
        base = os.path.basename(path)
        stem = base.rpartition(".")[0] or base
        if separator not in stem:
            raise DiscoverError(
                f"cannot parse '{stem}': expected "
                f"<episode>{separator}<participant>.<ext>")
        # Split on the first separator: a participant may contain one, an
        # episode may not, and that is the rule that makes "ep_1_bob" readable.
        episode, _, participant = stem.partition(separator)
        if not episode or not participant:
            raise DiscoverError(
                f"cannot parse '{stem}': empty episode or participant")
        if "/" in participant:
            raise DiscoverError(
                f"participant name may not contain a slash: '{participant}'")

        if episode_override:
            episode = episode_override
        if not episode_id:
            episode_id = episode
        elif episode != episode_id:
            raise DiscoverError(
                f"found tracks from two episodes ('{episode_id}' and "
                f"'{episode}'); process them separately or pass --episode")

        # Also catches one track present in two formats, which would otherwise
        # silently pick whichever sorted first.
        if participant in tracks:
            raise DiscoverError(
                f"participant '{participant}' appears twice: "
                f"{os.path.basename(tracks[participant])} and {base}. "
                "Keep one and remove the other, or narrow INPUT_EXTS.")
        tracks[participant] = path

    return episode_id, tracks


WORK_SUBDIRS = ("prep", "asr", "words", "llm", "render", "logs", "state")


def episode_paths(episode_id: str, work_root: str, output_dir: str) -> dict:
    """Where this episode's intermediates and outputs live."""
    work = os.path.join(work_root, episode_id)
    out = os.path.join(output_dir, episode_id)
    return {
        "work": work,
        "state": os.path.join(work, "state"),
        "output": out,
        "staging": os.path.join(out, ".staging"),
    }


def make_work_tree(paths: dict) -> None:
    for leaf in WORK_SUBDIRS:
        os.makedirs(os.path.join(paths["work"], leaf), exist_ok=True)
    os.makedirs(paths["output"], exist_ok=True)
