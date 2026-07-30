#!/usr/bin/env bash
#
# Clean up a multi-track podcast recording.
#
# Takes the synchronised per-participant tracks of one episode, works out which
# stretches are dead air and which are speech disfluencies, and renders the
# tracks back out — still separate, still in sync, ready for mixing.
#
# The pipeline runs as ordered stages; see --list-stages. Whisper and llama.cpp
# are never resident at the same time.
#
# Usage: clean-podcast.sh [options] [TRACK...]
#        clean-podcast.sh --help

set -euo pipefail

LIB_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=lib/log.sh
source "$LIB_ROOT/lib/log.sh"
# shellcheck source=lib/config.sh
source "$LIB_ROOT/lib/config.sh"
# shellcheck source=lib/stages.sh
source "$LIB_ROOT/lib/stages.sh"

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

Options:
  -c, --config FILE     configuration file (default: search the usual places)
  -i, --input DIR       directory to scan for input tracks
  -o, --output DIR      where finished tracks are published
      --work DIR        root of the per-episode work directory
  -e, --episode ID      override the episode id derived from filenames

      --from STAGE      start at STAGE, reusing what an earlier run produced
      --to STAGE        stop after STAGE
      --only STAGE      run just STAGE
      --stages A,B,C    run exactly these stages, in this order
      --list-stages     show the stages and exit

      --vad BACKEND     ffmpeg (level based) or silero (speech based)
      --no-llm          silence editing only; skip Qwen entirely
      --llama-endpoint URL
                        use a llama server that is already running
  -j, --jobs N          parallel ffmpeg jobs
      --force           proceed even when the plan trips a safety limit

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
            -i|--input)    ARG_INPUT_DIR="${2:?--input needs a directory}"; shift 2 ;;
            -o|--output)   ARG_OUTPUT_DIR="${2:?--output needs a directory}"; shift 2 ;;
            --work)        ARG_WORK_ROOT="${2:?--work needs a directory}"; shift 2 ;;
            -e|--episode)  EPISODE_ID_OVERRIDE="${2:?--episode needs an id}"; shift 2 ;;
            --from)        FROM_STAGE="${2:?--from needs a stage}"; shift 2 ;;
            --to)          TO_STAGE="${2:?--to needs a stage}"; shift 2 ;;
            --only)        EXPLICIT_STAGES="${2:?--only needs a stage}"; shift 2 ;;
            --stages)      EXPLICIT_STAGES="${2:?--stages needs a list}"; shift 2 ;;
            --list-stages)
                printf 'Stages, in order:\n'
                printf '  %s\n' "${ALL_STAGES[@]}"
                exit 0 ;;
            --vad)         ARG_VAD_BACKEND="${2:?--vad needs a backend}"; shift 2 ;;
            --no-llm)      ARG_LLM_ENABLE=0; shift ;;
            --llama-endpoint) ARG_LLAMA_ENDPOINT="${2:?--llama-endpoint needs a URL}"; shift 2 ;;
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

# Command-line values are applied after the config file, so they win.
apply_overrides() {
    [[ -n "${ARG_INPUT_DIR:-}" ]]       && INPUT_DIR="$ARG_INPUT_DIR"
    [[ -n "${ARG_OUTPUT_DIR:-}" ]]      && OUTPUT_DIR="$ARG_OUTPUT_DIR"
    [[ -n "${ARG_WORK_ROOT:-}" ]]       && WORK_ROOT="$ARG_WORK_ROOT"
    [[ -n "${ARG_VAD_BACKEND:-}" ]]     && VAD_BACKEND="$ARG_VAD_BACKEND"
    [[ -n "${ARG_LLM_ENABLE:-}" ]]      && LLM_ENABLE="$ARG_LLM_ENABLE"
    [[ -n "${ARG_LLAMA_ENDPOINT:-}" ]]  && LLAMA_ENDPOINT="$ARG_LLAMA_ENDPOINT"
    [[ -n "${ARG_FFMPEG_JOBS:-}" ]]     && FFMPEG_JOBS="$ARG_FFMPEG_JOBS"
    [[ -n "${ARG_KEEP_INPUTS:-}" ]]     && KEEP_INPUTS="$ARG_KEEP_INPUTS"
    [[ -n "${ARG_KEEP_WORK:-}" ]]       && KEEP_WORK="$ARG_KEEP_WORK"
    return 0
}

# --- stage selection ---------------------------------------------------------

stage_index() {
    local wanted="$1" index=0 name
    for name in "${ALL_STAGES[@]}"; do
        if [[ "$name" == "$wanted" ]]; then
            printf '%d' "$index"
            return 0
        fi
        index=$(( index + 1 ))
    done
    die "unknown stage '$wanted' (known: ${ALL_STAGES[*]})"
}

selected_stages() {
    local -a chosen=()
    if [[ -n "$EXPLICIT_STAGES" ]]; then
        local IFS=','
        read -r -a chosen <<<"$EXPLICIT_STAGES"
        local name
        for name in "${chosen[@]}"; do
            stage_index "$name" >/dev/null
        done
    else
        local first=0 last=$(( ${#ALL_STAGES[@]} - 1 ))
        [[ -n "$FROM_STAGE" ]] && first=$(stage_index "$FROM_STAGE")
        [[ -n "$TO_STAGE" ]] && last=$(stage_index "$TO_STAGE")
        (( first <= last )) || die "--from $FROM_STAGE comes after --to $TO_STAGE"
        local index
        for (( index = first; index <= last; index++ )); do
            chosen+=("${ALL_STAGES[$index]}")
        done
    fi
    printf '%s\n' "${chosen[@]}"
}

# --- failure handling --------------------------------------------------------

on_exit() {
    local code=$?
    trap - EXIT
    progress_done

    # Whatever went wrong, the model must not be left holding memory.
    llama_stop || true

    if (( code == 0 )); then
        [[ -n "$EXIT_SUMMARY" ]] && log_line "$EXIT_SUMMARY"
        return 0
    fi

    log_raw "=== run failed with exit code $code ==="
    log_line ""
    log_error "run failed (exit $code)"

    if [[ -n "$EPISODE_ID" && -d "${WORK:-}" && "$DRY_RUN" != 1 ]]; then
        printf 'failed at %s with exit code %s\n' "$(date -Is)" "$code" \
            >"$WORK/FAILED" 2>/dev/null || true
        log_line "      ${C_DIM}work directory kept: $WORK${C_RESET}"
        log_line "      ${C_DIM}resume with: $0 --episode $EPISODE_ID --from <stage>${C_RESET}"

        mkdir -p "$FAILED_DIR" 2>/dev/null || true
        if [[ -d "$FAILED_DIR" && -f "$LOG_FILE" ]]; then
            cp -f -- "$LOG_FILE" "$FAILED_DIR/${EPISODE_ID}.log" 2>/dev/null || true
            log_line "      ${C_DIM}log copied to $FAILED_DIR/${EPISODE_ID}.log${C_RESET}"
        fi

        # Unattended runs can ask for the inputs to be parked, so that a cron
        # job does not retry the same broken episode forever.
        if [[ "${FAILED_ACTION:-log}" == move ]] && (( ${#PARTICIPANTS[@]} )); then
            local target="$FAILED_DIR/$EPISODE_ID"
            mkdir -p "$target" 2>/dev/null || true
            local participant
            for participant in "${PARTICIPANTS[@]}"; do
                [[ -f "${TRACK_SOURCE[$participant]:-}" ]] || continue
                mv -f -- "${TRACK_SOURCE[$participant]}" "$target/" 2>/dev/null || true
            done
            log_line "      ${C_DIM}inputs moved to $target${C_RESET}"
        else
            log_line "      ${C_DIM}inputs left untouched${C_RESET}"
        fi
    fi
    log_line ""
    return "$code"
}

# --- main --------------------------------------------------------------------

main() {
    parse_args "$@"

    # Until the episode is identified the log goes somewhere temporary; the
    # discover stage adopts it into the work directory and carries it over.
    log_init "$(mktemp -t podcast-cleanup-XXXXXX.log)"

    config_load "$CONFIG_FILE"
    apply_overrides
    config_validate
    config_dump

    local -a stages=()
    mapfile -t stages < <(selected_stages)

    log_line ""
    log_line "${C_BOLD}Podcast cleanup${C_RESET}${C_DIM} — stages: ${stages[*]}${C_RESET}"
    [[ "$DRY_RUN" == 1 ]] && log_warn "dry run: no files will be written or removed"

    stage_total "${#stages[@]}"

    # A run that does not start at discover has to rebuild its context.
    if [[ "${stages[0]}" != "discover" ]]; then
        EPISODE_ID="$EPISODE_ID_OVERRIDE"
        resume_context
    fi

    local stage rc
    for stage in "${stages[@]}"; do
        rc=0
        "stage_$stage" || rc=$?
        if (( rc == 2 )) && [[ "$stage" == discover ]]; then
            log_line ""
            log_line "Nothing to do."
            return 0
        fi
        (( rc == 0 )) || die "stage '$stage' failed with exit code $rc"
    done

    EXIT_SUMMARY=$(printf '\n%s✓%s %sfinished %s in %s%s' \
        "$C_GREEN" "$C_RESET" "$C_BOLD" "$EPISODE_ID" "$(run_elapsed)" "$C_RESET")
    if [[ -d "${OUT_DIR:-}" ]]; then
        EXIT_SUMMARY+=$(printf '\n  %soutputs: %s%s' "$C_DIM" "$OUT_DIR" "$C_RESET")
    fi
    return 0
}

trap on_exit EXIT
main "$@"
