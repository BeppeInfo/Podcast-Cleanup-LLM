# shellcheck shell=bash
#
# What the launcher needs to say something before Python takes over.
#
# The display — stage headers, the progress counter, per-command output — lives
# in python/cleanup/runlog.py, which the run adopts by way of PODCAST_LOG_FILE.
# What is left here covers the handful of lines emitted before that: finding
# ffmpeg, the config dump, and a configuration error that ends the run before
# there is an episode to log against.
#
# The format matches runlog.py exactly, because both write to the same file and
# a reader should not be able to tell which produced a line. TestRunLog compares
# them.

LOG_FILE="${LOG_FILE:-}"
LOG_LEVEL="${LOG_LEVEL:-info}"     # debug | info | warn | error
DRY_RUN="${DRY_RUN:-0}"

if [[ -t 2 && "${NO_COLOR:-}" == "" && "${TERM:-dumb}" != "dumb" ]]; then
    C_RESET=$'\033[0m'
    C_DIM=$'\033[2m'
    C_BOLD=$'\033[1m'
    C_RED=$'\033[31m'
    C_GREEN=$'\033[32m'
    C_YELLOW=$'\033[33m'
    C_BLUE=$'\033[34m'
    C_CYAN=$'\033[36m'
else
    C_RESET='' C_DIM='' C_BOLD='' C_RED='' C_GREEN='' C_YELLOW='' C_BLUE='' C_CYAN=''
fi

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

# Append to the run log only. Safe before log_init (no-op).
log_raw() {
    [[ -n "$LOG_FILE" ]] || return 0
    printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >>"$LOG_FILE"
}

_emit() {
    local level="$1" colour="$2" tag="$3"; shift 3
    log_raw "[$level] $*"
    _log_enabled "$level" || return 0
    printf '%s%s%s %s\n' "$colour" "$tag" "$C_RESET" "$*" >&2
}

# log_init <logfile> — where the run log starts, before an episode is known.
# The discover stage moves it into the work directory, carrying this over.
log_init() {
    LOG_FILE="$1"
    mkdir -p "$(dirname "$LOG_FILE")"
    : >>"$LOG_FILE"
    log_raw "=== run started: $* ==="
}

log_debug() { _emit debug "$C_DIM"    "  ·" "$*"; }
log_info()  { _emit info  "$C_BLUE"   "  →" "$*"; }
log_warn()  { _emit warn  "$C_YELLOW" "  !" "$*"; }
log_error() { _emit error "$C_RED"    "  ✗" "$*"; }

die() {
    log_error "$*"
    exit 1
}

# require_bin <label> <candidate> — the resolved path, or an error and 1.
#
# Deliberately does not call die(): callers use it inside $(...), where an exit
# would only end the subshell and let a missing tool slip through. They pair it
# with `|| exit 1`.
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
