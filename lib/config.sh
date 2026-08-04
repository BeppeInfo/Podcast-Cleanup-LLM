# shellcheck shell=bash
#
# Defaults, config-file loading and validation.
#
# The config file is sourced as bash, so it can only set variables (and may
# reference earlier ones). Tool resolution is deliberately lazy: a run that
# only re-renders should not require whisper or llama to be installed.

# --- defaults ----------------------------------------------------------------

# The Python side is the authority for defaults, precedence and validation; see
# python/cleanup/config.py. This asks it for the resolved settings and evals the
# answer, so there is exactly one list of settings and one set of rules, shared
# with whatever else drives the pipeline.
#
# PYTHON has to be found first, and can be: it comes from PYTHON_BIN or the path,
# never from the config file.
config_load() {
    local explicit="${1:-}"
    config_need_python

    local -a args=(config --script-root "$LIB_ROOT")
    [[ -n "$explicit" ]] && args+=(--config "$explicit")

    # Only options actually given are passed, so an absent one falls through to
    # the environment and then the file. An explicit --root re-derives the whole
    # layout: a config file's INPUT_DIR is a weaker statement than the root asked
    # for on the command line, so the four are cleared and rebuilt. The
    # per-directory options below still win, being equally explicit and more
    # specific.
    if [[ -n "${ARG_PODCAST_ROOT:-}" ]]; then
        args+=(--set "PODCAST_ROOT=$ARG_PODCAST_ROOT"
               --set "INPUT_DIR=" --set "OUTPUT_DIR="
               --set "WORK_ROOT=" --set "FAILED_DIR=")
    fi
    local name
    for name in INPUT_DIR OUTPUT_DIR WORK_ROOT WHISPER_VAD LLM_ENABLE \
                WHISPER_ENDPOINT LLAMA_ENDPOINT FFMPEG_JOBS KEEP_INPUTS KEEP_WORK; do
        local -n value="ARG_$name"
        [[ -n "${value:-}" ]] && args+=(--set "$name=$value")
    done

    local resolved
    resolved=$("$PYTHON" "$LIB_ROOT/python/cleanup_cli.py" "${args[@]}") || exit 1
    eval "$resolved"
}

# Hand every resolved setting to the Python stages through the environment.
# They have to see exactly what this run resolved — file, environment and command
# line together — and re-reading the config file on their side could only
# disagree with it. config.from_environment is the other half.
config_export() {
    local name
    for name in $("$PYTHON" "$LIB_ROOT/python/cleanup_cli.py" config-names); do
        export "$name"
    done
}

# Creating the tree up front is what lets a missing input directory mean *empty*
# rather than *misconfigured*.
config_make_tree() {
    [[ "$DRY_RUN" == 1 ]] && return 0
    local dir
    for dir in "$INPUT_DIR" "$OUTPUT_DIR" "$WORK_ROOT" "$FAILED_DIR"; do
        mkdir -p -- "$dir" || die "cannot create directory: $dir"
    done
    return 0
}

config_need_ffmpeg() {
    [[ -n "${FFMPEG:-}" ]] && return 0
    FFMPEG=$(require_bin ffmpeg "${FFMPEG_BIN:-ffmpeg}") || exit 1
    FFPROBE=$(require_bin ffprobe "${FFPROBE_BIN:-ffprobe}") || exit 1
    log_debug "ffmpeg: $FFMPEG"

    # Catch an unbuildable output format now rather than after transcribing two
    # hours of audio.
    if ! "$FFMPEG" -hide_banner -h "encoder=$OUTPUT_CODEC" 2>/dev/null \
        | grep -q "Encoder $OUTPUT_CODEC"; then
        die "this ffmpeg has no '$OUTPUT_CODEC' encoder (see: ffmpeg -encoders)"
    fi
}

config_need_whisper() {
    [[ -n "$WHISPER_ENDPOINT" ]] \
        || die "WHISPER_ENDPOINT is required: transcription is always sent to a whisper-server. Point it at one — http://127.0.0.1:8081 if it runs on this machine."
    log_debug "whisper: $WHISPER_ENDPOINT"
}

config_describe_models() {
    log_info "Whisper: $WHISPER_ENDPOINT   LLM: $LLAMA_ENDPOINT"
}

config_need_llama() {
    [[ -n "$LLAMA_ENDPOINT" ]] \
        || die "LLAMA_ENDPOINT is required: edit detection is always sent to a llama-server. Point it at one, or set LLM_ENABLE=0 to do silence editing only."
    log_debug "llama: $LLAMA_ENDPOINT"
}

config_need_python() {
    [[ -n "${PYTHON:-}" ]] && return 0
    PYTHON=$(require_bin python3 "${PYTHON_BIN:-python3}") || exit 1
    log_debug "python: $PYTHON"
}

# config_dump — every effective setting, to the log (and to stderr at -v).
#
# API keys are reported as present or absent and never by value: this log is
# copied into the output directory and kept after everything else is deleted.
config_dump() {
    local v
    log_debug "config file: ${CONFIG_FILE:-<none, using defaults>}"
    local secret
    for v in WHISPER_API_KEY LLAMA_API_KEY; do
        secret="${!v}"
        if [[ -n "$secret" ]]; then
            log_raw "  config $v=<set, ${#secret} chars, redacted>"
        else
            log_raw "  config $v=<unset>"
        fi
    done
    for v in PODCAST_ROOT INPUT_DIR OUTPUT_DIR WORK_ROOT FAILED_DIR \
             INPUT_EXTS TRACK_SEPARATOR \
             OUTPUT_CODEC OUTPUT_EXT OUTPUT_EXTRA_ARGS RESAMPLE_TO \
             WHISPER_ENDPOINT WHISPER_ENDPOINT_PATH WHISPER_CHUNK_SECONDS \
             WHISPER_REQUEST_TIMEOUT \
             WHISPER_LANG \
             WHISPER_VAD WHISPER_VAD_THRESHOLD \
             WHISPER_VAD_MIN_SPEECH_MS WHISPER_VAD_MIN_SILENCE_MS \
             WHISPER_VAD_SPEECH_PAD_MS WHISPER_VAD_SAMPLES_OVERLAP \
             WHISPER_RECOVER WHISPER_PROMPT WHISPER_REASK \
             WHISPER_REASK_WORD_SECONDS WHISPER_REASK_WINDOW SPEECH_MAP_CLIP \
             LLAMA_ENDPOINT \
             LLAMA_MODEL_NAME LLM_API LLM_MAX_REPLY_TOKENS \
             LLM_CHECK_SCHEMA LLM_CONCURRENCY \
             SPLIT_SILENCE_THRESHOLD SPLIT_MIN_SILENCE SPEECH_PAD \
             SILENCE_MIN_DURATION SILENCE_KEEP EDGE_KEEP CUT_PADDING MIN_CUT \
             MUTE_FADE LLM_ENABLE LLM_CHUNK_WORDS LLM_CHUNK_OVERLAP \
             LLM_MAX_EDIT_WORDS LLM_MAX_EDIT_SECONDS LLM_MIN_CONFIDENCE \
             LLM_ACCEPT_KINDS MAX_CUT_FRACTION OUTPUT_COMPRESSION OUTPUT_SUFFIX \
             RENDER_FRAME_SAMPLES FFMPEG_JOBS KEEP_WORK KEEP_INPUTS \
             DURATION_TOLERANCE; do
        log_raw "  config $v=${!v}"
    done
}
