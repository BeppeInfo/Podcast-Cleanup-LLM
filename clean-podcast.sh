#!/usr/bin/env bash
#
# Clean up a multi-track podcast recording.
#
# Takes the synchronised per-participant tracks of one episode, works out which
# stretches are dead air and which are speech disfluencies, and renders the
# tracks back out — still separate, still in sync, ready for mixing.
#
# The pipeline runs as ordered stages; see --list-stages. Both models are
# reached over HTTP; neither process is managed here.
#
# Usage: clean-podcast.sh [options] [TRACK...]
#        clean-podcast.sh --help

set -euo pipefail

LIB_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=lib/log.sh
source "$LIB_ROOT/lib/log.sh"
# shellcheck source=lib/config.sh
source "$LIB_ROOT/lib/config.sh"

CONFIG_FILE=""
EPISODE_ID_OVERRIDE=""
FORCE=0
FROM_STAGE=""
TO_STAGE=""
EXPLICIT_STAGES=""
EXIT_SUMMARY=""

usage() {
    cat <<'USAGE'
Clean up a multi-track podcast recording.

Usage:
  clean-podcast.sh [options] [TRACK...]

With no TRACK arguments the input directory is scanned for one episode's worth
of tracks, named <episode><separator><participant>.<ext> — for example
ep042_leonardo.flac.

Everything lives under one root, which defaults to the directory holding this
script, so a fresh checkout is usable as-is: run it once and drop tracks into
the incoming/ directory it creates.

Options:
  -c, --config FILE     configuration file (default: search the usual places)
  -r, --root DIR        root of the working layout; incoming/, output/, work/
                        and failed/ are created under it as needed
                        (default: the directory holding this script)
  -i, --input DIR       directory to scan for input tracks   (default: ROOT/incoming)
  -o, --output DIR      where finished tracks are published  (default: ROOT/output)
      --work DIR        root of the per-episode work directory (default: ROOT/work)
  -e, --episode ID      override the episode id derived from filenames

      --from STAGE      start at STAGE, reusing what an earlier run produced
      --to STAGE        stop after STAGE
      --only STAGE      run just STAGE
      --stages A,B,C    run exactly these stages, in this order
      --list-stages     show the stages and exit

      --no-whisper-vad  do not have Whisper run Silero over the audio first.
                        Silence then gets transcribed too, and since the speech
                        map is derived from the transcript, anything invented
                        there becomes speech in the plan
      --no-llm          silence editing only; skip the LLM stage entirely
  -j, --jobs N          parallel ffmpeg jobs
      --force           proceed even when the plan trips a safety limit

  Both models are servers someone else runs; nothing is started or stopped
  here. Point these at 127.0.0.1 when they happen to be on this machine.
      --whisper-endpoint URL    transcribe via this whisper-server
      --llama-endpoint URL      detect via this llama-server

      --keep-inputs     do not delete the originals after a successful run
      --keep-work       do not delete the work directory
  -n, --dry-run         show what would happen; touch nothing

  -v, --verbose         include debug detail on the console
  -q, --quiet           warnings and errors only
      --self-test       run the offline end-to-end check and exit
  -h, --help            this text

Inputs are deleted only after every output track has been rendered and its
duration verified against the plan. Any failure leaves the originals and the
work directory in place so the run can be resumed with --from.
USAGE
}

# --- argument parsing ---------------------------------------------------------

parse_args() {
    while (( $# )); do
        case "$1" in
            -c|--config)   CONFIG_FILE="${2:?--config needs a file}"; shift 2 ;;
            -r|--root)     ARG_PODCAST_ROOT="${2:?--root needs a directory}"; shift 2 ;;
            -i|--input)    ARG_INPUT_DIR="${2:?--input needs a directory}"; shift 2 ;;
            -o|--output)   ARG_OUTPUT_DIR="${2:?--output needs a directory}"; shift 2 ;;
            --work)        ARG_WORK_ROOT="${2:?--work needs a directory}"; shift 2 ;;
            -e|--episode)  EPISODE_ID_OVERRIDE="${2:?--episode needs an id}"; shift 2 ;;
            --from)        FROM_STAGE="${2:?--from needs a stage}"; shift 2 ;;
            --to)          TO_STAGE="${2:?--to needs a stage}"; shift 2 ;;
            --only)        EXPLICIT_STAGES="${2:?--only needs a stage}"; shift 2 ;;
            --stages)      EXPLICIT_STAGES="${2:?--stages needs a list}"; shift 2 ;;
            --list-stages)
                config_need_python
                printf 'Stages, in order:\n'
                "$PYTHON" "$LIB_ROOT/python/cleanup_cli.py" list-stages
                exit 0 ;;
            --no-whisper-vad) ARG_WHISPER_VAD=0; shift ;;
            --no-llm)      ARG_LLM_ENABLE=0; shift ;;
            --whisper-endpoint) ARG_WHISPER_ENDPOINT="${2:?--whisper-endpoint needs a URL}"; shift 2 ;;
            --llama-endpoint)   ARG_LLAMA_ENDPOINT="${2:?--llama-endpoint needs a URL}"; shift 2 ;;
            -j|--jobs)     ARG_FFMPEG_JOBS="${2:?--jobs needs a number}"; shift 2 ;;
            --force)       FORCE=1; shift ;;
            --keep-inputs) ARG_KEEP_INPUTS=1; shift ;;
            --keep-work)   ARG_KEEP_WORK=1; shift ;;
            -n|--dry-run)  DRY_RUN=1; ARG_KEEP_WORK=1; ARG_KEEP_INPUTS=1; shift ;;
            -v|--verbose)  LOG_LEVEL=debug; shift ;;
            -q|--quiet)    LOG_LEVEL=warn; shift ;;
            --self-test)   exec "$LIB_ROOT/tests/selftest.sh" ;;
            -h|--help)     usage; exit 0 ;;
            --)            shift; INPUT_FILES+=("$@"); break ;;
            -*)            die "unknown option: $1 (try --help)" ;;
            *)             INPUT_FILES+=("$1"); shift ;;
        esac
    done
}

# --- main --------------------------------------------------------------------

main() {
    parse_args "$@"

    # Until the episode is identified the log goes somewhere temporary; the
    # discover stage adopts it into the work directory and carries it over.
    log_init "$(mktemp -t podcast-cleanup-XXXXXX.log)"

    # One call: defaults, config file, environment and the options above are
    # resolved and checked on the Python side, and come back as assignments.
    config_load "$CONFIG_FILE"
    config_export
    config_make_tree

    # And from here it is Python's run: the stage sequence, the resume
    # behaviour and the failure handling all live in cleanup/pipeline.py, so
    # anything else driving this pipeline gets them too.
    local -a args=(run)
    [[ -n "$FROM_STAGE" ]] && args+=(--from "$FROM_STAGE")
    [[ -n "$TO_STAGE" ]] && args+=(--to "$TO_STAGE")
    [[ -n "$EXPLICIT_STAGES" ]] && args+=(--stages "$EXPLICIT_STAGES")
    [[ -n "$EPISODE_ID_OVERRIDE" ]] && args+=(--episode "$EPISODE_ID_OVERRIDE")
    [[ "$DRY_RUN" == 1 ]] && args+=(--dry-run)
    [[ "${FORCE:-0}" == 1 ]] && args+=(--force)
    args+=(--program "$0")
    local file
    for file in "${INPUT_FILES[@]}"; do
        args+=(--file "$file")
    done

    # PODCAST_CONFIG_FILE is for the config dump alone — which file was read,
    # not a request to read it again; config_export already carried the values.
    PODCAST_WHISPER_API_KEY="$WHISPER_API_KEY" \
        PODCAST_LLAMA_API_KEY="$LLAMA_API_KEY" \
        PODCAST_CONFIG_FILE="${CONFIG_FILE:-}" \
        PODCAST_LOG_FILE="$LOG_FILE" LOG_LEVEL="$LOG_LEVEL" \
        exec "$PYTHON" "$LIB_ROOT/python/cleanup_cli.py" "${args[@]}"
}

main "$@"
