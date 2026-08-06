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

# What used to live below here is all on the Python side now, called from
# run_episode so the web front end is covered by the same code:
#
#   config_need_ffmpeg    -> proc.resolve_ffmpeg   (and the encoder check)
#   config_need_whisper   -> pipeline.stage_transcribe
#   config_need_llama     -> pipeline.stage_detect
#   config_dump           -> config.dump
#
# config_describe_models went with the driver loop; cleanup_cli prints that line.

config_need_python() {
    [[ -n "${PYTHON:-}" ]] && return 0
    PYTHON=$(require_bin python3 "${PYTHON_BIN:-python3}") || exit 1
}
