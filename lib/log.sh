# shellcheck shell=bash
#
# Logging, status output and command execution helpers.
#
# Everything printed to the user also lands in the run log, timestamped.
# The run log is the only artifact that survives cleanup, so it has to be
# complete enough to diagnose a failed episode after the work dir is gone.

# --- state -------------------------------------------------------------------

LOG_FILE="${LOG_FILE:-}"
LOG_LEVEL="${LOG_LEVEL:-info}"     # debug | info | warn | error
DRY_RUN="${DRY_RUN:-0}"

_STAGE_NAME=""
_STAGE_INDEX=0
_STAGE_TOTAL=0
_STAGE_STARTED=0
_RUN_STARTED=$SECONDS
_PROGRESS_ACTIVE=0

# --- colours -----------------------------------------------------------------

if [[ -t 2 && "${NO_COLOR:-}" == "" && "${TERM:-dumb}" != "dumb" ]]; then
    C_RESET=$'\033[0m'
    C_DIM=$'\033[2m'
    C_BOLD=$'\033[1m'
    C_RED=$'\033[31m'
    C_GREEN=$'\033[32m'
    C_YELLOW=$'\033[33m'
    C_BLUE=$'\033[34m'
    C_CYAN=$'\033[36m'
    _TTY=1
else
    C_RESET='' C_DIM='' C_BOLD='' C_RED='' C_GREEN='' C_YELLOW='' C_BLUE='' C_CYAN=''
    _TTY=0
fi

# --- internals ---------------------------------------------------------------

_log_level_num() {
    case "$1" in
        debug) echo 10 ;;
        info)  echo 20 ;;
        warn)  echo 30 ;;
        error) echo 40 ;;
        *)     echo 20 ;;
    esac
}

_log_enabled() {
    (( $(_log_level_num "$1") >= $(_log_level_num "$LOG_LEVEL") ))
}

_now() { date '+%Y-%m-%d %H:%M:%S'; }

# Seconds -> compact human duration (1h02m03s / 2m03s / 3.4s).
fmt_duration() {
    local t=${1%.*}
    (( t < 0 )) && t=0
    if (( t >= 3600 )); then
        printf '%dh%02dm%02ds' $((t/3600)) $(((t%3600)/60)) $((t%60))
    elif (( t >= 60 )); then
        printf '%dm%02ds' $((t/60)) $((t%60))
    else
        printf '%ds' "$t"
    fi
}

# Clear an in-place progress line before emitting anything else on stderr.
_clear_progress() {
    if (( _PROGRESS_ACTIVE )); then
        printf '\r\033[K' >&2
        _PROGRESS_ACTIVE=0
    fi
}

# Append to the run log only. Safe before log_init (no-op).
log_raw() {
    [[ -n "$LOG_FILE" ]] || return 0
    printf '%s %s\n' "$(_now)" "$*" >>"$LOG_FILE"
}

_emit() {
    local level="$1" colour="$2" tag="$3"; shift 3
    log_raw "[$level]${_STAGE_NAME:+ ($_STAGE_NAME)} $*"
    _log_enabled "$level" || return 0
    _clear_progress
    printf '%s%s%s %s\n' "$colour" "$tag" "$C_RESET" "$*" >&2
}

# --- public API --------------------------------------------------------------

# log_init <logfile>
log_init() {
    LOG_FILE="$1"
    mkdir -p "$(dirname "$LOG_FILE")"
    : >>"$LOG_FILE"
    log_raw "=== run started: $* ==="
}

log_debug() { _emit debug "$C_DIM"    "  ·" "$*"; }
log_info()  { _emit info  "$C_BLUE"   "  →" "$*"; }
log_ok()    { _emit info  "$C_GREEN"  "  ✓" "$*"; }
log_warn()  { _emit warn  "$C_YELLOW" "  !" "$*"; }
log_error() { _emit error "$C_RED"    "  ✗" "$*"; }

# A plain, unprefixed user-facing line (headers, summaries). Always shown.
log_line() {
    log_raw "$*"
    _clear_progress
    printf '%s\n' "$*" >&2
}

# Multi-line output that belongs in the log unconditionally but on the console
# only when the user has not asked for quiet.
log_report() {
    local line
    while IFS= read -r line; do
        log_raw "$line"
        if _log_enabled info; then
            _clear_progress
            printf '      %s\n' "$line" >&2
        fi
    done <<<"$1"
}

die() {
    log_error "$*"
    exit 1
}

# stage_total <n> — how many stages this run will execute.
stage_total() { _STAGE_TOTAL="$1"; _STAGE_INDEX=0; }

# stage_begin <name> [description]
stage_begin() {
    _STAGE_NAME="$1"
    _STAGE_STARTED=$SECONDS
    _STAGE_INDEX=$(( _STAGE_INDEX + 1 ))
    local desc="${2:-}"
    log_raw "--- stage $_STAGE_INDEX/$_STAGE_TOTAL: $1 ${desc:+- $desc} ---"
    _clear_progress
    printf '%s[%d/%d]%s %s%s%s%s\n' \
        "$C_DIM" "$_STAGE_INDEX" "$_STAGE_TOTAL" "$C_RESET" \
        "$C_BOLD" "$1" "$C_RESET" "${desc:+  $C_DIM$desc$C_RESET}" >&2
}

# stage_end [note]
stage_end() {
    local elapsed=$(( SECONDS - _STAGE_STARTED ))
    local note="${1:-}"
    log_raw "--- stage $_STAGE_NAME done in $(fmt_duration "$elapsed") ${note:+- $note} ---"
    _clear_progress
    printf '%s      %s in %s%s\n' \
        "$C_DIM" "${note:-done}" "$(fmt_duration "$elapsed")" "$C_RESET" >&2
    _STAGE_NAME=""
}

# stage_skip <name> <reason> — for stages short-circuited by resume/flags.
stage_skip() {
    _STAGE_INDEX=$(( _STAGE_INDEX + 1 ))
    log_raw "--- stage $_STAGE_INDEX/$_STAGE_TOTAL: $1 SKIPPED ($2) ---"
    _clear_progress
    printf '%s[%d/%d] %s — skipped (%s)%s\n' \
        "$C_DIM" "$_STAGE_INDEX" "$_STAGE_TOTAL" "$1" "$2" "$C_RESET" >&2
}

# progress <current> <total> <label> — in-place counter on a TTY, one line
# per item otherwise (so non-interactive logs stay readable).
progress() {
    local cur="$1" total="$2" label="$3"
    log_raw "[progress] $cur/$total $label"
    _log_enabled info || return 0
    if (( _TTY )); then
        printf '\r\033[K%s      %s(%d/%d)%s %s' \
            "$C_DIM" "$C_CYAN" "$cur" "$total" "$C_DIM" "$label$C_RESET" >&2
        _PROGRESS_ACTIVE=1
    else
        printf '      (%d/%d) %s\n' "$cur" "$total" "$label" >&2
    fi
}

progress_done() { _clear_progress; }

run_elapsed() { fmt_duration $(( SECONDS - _RUN_STARTED )); }

# --- command execution -------------------------------------------------------
#
# run  — run a command, streaming its output to the run log only. On failure
#        the tail of that output is surfaced, so a broken ffmpeg invocation
#        does not require digging through the log by hand.
# run_quiet — same, but a non-zero exit is the caller's business.

_run_impl() {
    local tolerate_failure="$1"; shift
    log_raw "\$ $*"
    if [[ "$DRY_RUN" == 1 ]]; then
        _clear_progress
        printf '%s      would run:%s %s\n' "$C_DIM" "$C_RESET" "$*" >&2
        return 0
    fi

    local out rc=0
    out=$(mktemp)
    if "$@" >"$out" 2>&1; then
        rc=0
    else
        rc=$?
    fi
    if [[ -n "$LOG_FILE" ]]; then
        cat "$out" >>"$LOG_FILE"
    fi
    if (( rc != 0 && ! tolerate_failure )); then
        log_error "command failed (exit $rc): $1"
        _clear_progress
        printf '%s%s%s\n' "$C_DIM" "$(tail -n 25 "$out")" "$C_RESET" >&2
    fi
    rm -f "$out"
    return "$rc"
}

run()       { _run_impl 0 "$@"; }
run_quiet() { _run_impl 1 "$@"; }

# run_streaming <parser-fn> <label> <cmd...>
#
# Like run(), but every output line is fed to <parser-fn> as it arrives, so a
# long-running Whisper or LLM stage can show live progress. The parser receives
# (line, label) and is expected to call progress() when it recognises something.
run_streaming() {
    local parser="$1" label="$2"; shift 2
    log_raw "\$ $*"
    if [[ "$DRY_RUN" == 1 ]]; then
        _clear_progress
        printf '%s      would run:%s %s\n' "$C_DIM" "$C_RESET" "$*" >&2
        return 0
    fi

    local rc_file
    rc_file=$(mktemp)
    printf '0' >"$rc_file"

    # The exit status has to travel out of the pipeline's first stage, hence the
    # temp file: $PIPESTATUS would refer to the reader on the right.
    set +e
    {
        "$@" 2>&1
        printf '%s' "$?" >"$rc_file"
    } | {
        while IFS= read -r stream_line; do
            log_raw "$stream_line"
            "$parser" "$stream_line" "$label" || true
        done
    }
    set -e
    # The reader ran in a subshell, so its progress state never made it back
    # here; assume a counter was left on screen and clear it.
    _PROGRESS_ACTIVE=$_TTY
    progress_done

    local rc
    rc=$(<"$rc_file")
    rm -f "$rc_file"
    if [[ "$rc" != 0 ]]; then
        log_error "command failed (exit $rc): $1"
        if [[ -n "$LOG_FILE" ]]; then
            _clear_progress
            printf '%s%s%s\n' "$C_DIM" "$(tail -n 20 "$LOG_FILE")" "$C_RESET" >&2
        fi
    fi
    return "$rc"
}

# --- output parsers used with run_streaming ----------------------------------

# whisper-cli reports "progress = 42%" when --print-progress is on.
parse_whisper_progress() {
    if [[ "$1" =~ progress[[:space:]]*=[[:space:]]*([0-9]+)% ]]; then
        progress "${BASH_REMATCH[1]}" 100 "$2"
    fi
}

# Our Python CLI emits "PROGRESS <done> <total>", and "WARN <text>" for anything
# the operator has to see. Without the WARN case a stage can fail in every chunk
# and still look healthy on screen: only the file log would carry it, and the
# report would show a track with no edits, which reads exactly like clean audio.
parse_python_progress() {
    if [[ "$1" == PROGRESS\ * ]]; then
        local _tag done total
        read -r _tag done total <<<"$1"
        progress "$done" "$total" "$2"
    elif [[ "$1" == WARN\ * ]]; then
        log_warn "${1#WARN }"
    fi
}

# ffmpeg with -progress pipe:1 emits "out_time_ms=<microseconds>" (the name is
# a long-standing misnomer). FFMPEG_TOTAL_US must hold the expected length.
parse_ffmpeg_progress() {
    if [[ "$1" == out_time_ms=* ]]; then
        local current="${1#out_time_ms=}"
        [[ "$current" =~ ^[0-9]+$ ]] || return 0
        local total="${FFMPEG_TOTAL_US:-0}"
        if (( total > 0 )); then
            progress $(( current / 10000 )) $(( total / 10000 )) "$2"
        fi
    fi
}

# Discards everything; for commands whose output only belongs in the log.
parse_nothing() { :; }

# run_to_file <destination> <cmd...> — capture combined output to a file of its
# own (silencedetect's report, a server's log) rather than into the run log.
run_to_file() {
    local destination="$1"; shift
    log_raw "\$ $* > $destination"
    if [[ "$DRY_RUN" == 1 ]]; then
        _clear_progress
        printf '%s      would run:%s %s\n' "$C_DIM" "$C_RESET" "$*" >&2
        : >"$destination"
        return 0
    fi
    local rc=0
    "$@" >"$destination" 2>&1 || rc=$?
    if (( rc != 0 )); then
        log_error "command failed (exit $rc): $1"
        _clear_progress
        printf '%s%s%s\n' "$C_DIM" "$(tail -n 25 "$destination")" "$C_RESET" >&2
    fi
    return "$rc"
}

# require_bin <name> <path-or-command>
#
# Prints the resolved path, or logs an error and returns 1. Deliberately does
# not call die(): callers use it inside $(...), where an exit would only end the
# subshell and let a missing tool slip through. Callers pair it with `|| exit 1`.
require_bin() {
    local label="$1" candidate="$2"
    if [[ -x "$candidate" ]]; then
        printf '%s' "$candidate"
        return 0
    fi
    local resolved
    if resolved=$(command -v "$candidate" 2>/dev/null); then
        printf '%s' "$resolved"
        return 0
    fi
    log_error "$label not found: '$candidate' is neither an executable path nor on PATH"
    return 1
}
