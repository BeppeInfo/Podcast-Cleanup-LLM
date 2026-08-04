# shellcheck shell=bash
#
# The pipeline stages.
#
# Each stage is self-contained: it reads what it needs from the episode work
# directory, writes its results back there, and drops a marker in state/ so a
# later run can pick up where this one stopped. That is what makes --from and
# --only work without special cases.
#
# Stage order still has one real constraint: `detect` reads the transcripts, so
# `transcribe` must finish every track first. Keeping the two models out of each
# other's way is no longer this script's problem — both are endpoints someone
# else runs, and whether they share a machine is that machine's business.

# --- episode state (populated by discover, or reloaded from meta.json) --------

EPISODE_ID=""
EPISODE_DURATION="0"
EPISODE_SAMPLE_RATE="0"
WORK=""
STAGE_DIR=""
OUT_DIR=""
STAGING_DIR=""
declare -a PARTICIPANTS=()
declare -a INPUT_FILES=()
declare -A TRACK_SOURCE=()
declare -A TRACK_DURATION=()
declare -A TRACK_SAMPLE_FMT=()
declare -A TRACK_CODEC=()

ALL_STAGES=(discover prepare transcribe detect plan render finalize)

# --- helpers -----------------------------------------------------------------

py() {
    config_need_python
    "$PYTHON" "$LIB_ROOT/python/cleanup_cli.py" "$@"
}

state_done()  { [[ -f "$STAGE_DIR/$1.done" ]]; }
state_mark()  { [[ "$DRY_RUN" == 1 ]] || touch "$STAGE_DIR/$1.done"; }
state_clear() { rm -f "$STAGE_DIR/$1.done"; }
# Per-item completion marker; a no-op in a dry run, where pool_wait skips too.
state_touch() { [[ "$DRY_RUN" == 1 ]] || touch "$1"; }

track_total_us() {
    awk -v d="${TRACK_DURATION[$1]:-0}" 'BEGIN{printf "%d", d*1000000}'
}

# Wait for every background job, then confirm each expected marker landed.
# Background failures are otherwise easy to lose track of.
pool_wait() {
    local label="$1"; shift
    wait
    [[ "$DRY_RUN" == 1 ]] && return 0
    local missing=()
    local marker
    for marker in "$@"; do
        [[ -f "$marker" ]] || missing+=("$(basename "$marker" .ok)")
    done
    if (( ${#missing[@]} )); then
        die "$label failed for: ${missing[*]} (see $LOG_FILE)"
    fi
}

pool_slot() {
    local limit="$1"
    while (( $(jobs -rp | wc -l) >= limit )); do
        wait -n 2>/dev/null || true
    done
}

load_meta() {
    local meta="$WORK/meta.json"
    [[ -f "$meta" ]] || die "meta.json missing — run the discover stage first"
    local assignments
    assignments=$(py meta-shell --meta "$meta") || die "could not read $meta"
    eval "$assignments"
    log_debug "episode $EPISODE_ID: ${#PARTICIPANTS[@]} tracks, ${EPISODE_DURATION}s"
}


probe_duration() {
    "$FFPROBE" -v error -show_entries format=duration -of csv=p=0 "$1"
}

# ============================================================================
# discover
# ============================================================================

stage_discover() {
    stage_begin discover "locating the episode and its tracks"
    config_need_ffmpeg
    config_need_python

    local -a args=(
        discover --exts "$INPUT_EXTS" --separator "$TRACK_SEPARATOR"
        --work-root "$WORK_ROOT" --output-dir "$OUTPUT_DIR"
        --ffprobe "$FFPROBE" --input-dir "$INPUT_DIR"
    )
    [[ -n "$RESAMPLE_TO" ]] && args+=(--resample-to "$RESAMPLE_TO")
    [[ -n "${EPISODE_ID_OVERRIDE:-}" ]] && args+=(--episode "$EPISODE_ID_OVERRIDE")
    [[ "$DRY_RUN" == 1 ]] && args+=(--dry-run)
    local file
    for file in "${INPUT_FILES[@]}"; do
        args+=(--file "$file")
    done

    # Python writes its own notes and refusals into the same run log, in the
    # same format, so nothing here has to translate them. Exit 2 is an empty
    # inbox rather than a fault; anything else has already explained itself.
    local resolved rc=0
    resolved=$(PODCAST_LOG_FILE="$LOG_FILE" LOG_LEVEL="$LOG_LEVEL" \
        PODCAST_LOG_STAGE=discover \
        "$PYTHON" "$LIB_ROOT/python/cleanup_cli.py" "${args[@]}") || rc=$?
    if (( rc == 2 )); then
        return 2
    elif (( rc != 0 )); then
        exit 1
    fi
    eval "$resolved"

    # From here the run log lives with the episode. What the temporary log
    # already holds is carried over rather than lost — it has the config dump.
    local episode_log="$WORK/logs/run.log"
    if [[ "$DRY_RUN" != 1 && "$LOG_FILE" != "$episode_log" ]]; then
        if [[ -f "$LOG_FILE" ]]; then
            cat "$LOG_FILE" >>"$episode_log"
            rm -f "$LOG_FILE"
        fi
        LOG_FILE="$episode_log"
        log_raw "=== log adopted into $WORK ==="
    fi

    log_line ""
    log_line "  ${C_BOLD}Episode${C_RESET} $EPISODE_ID  ${C_DIM}(${#PARTICIPANTS[@]} tracks)${C_RESET}"
    local participant
    for participant in "${PARTICIPANTS[@]}"; do
        log_line "    ${C_CYAN}${participant}${C_RESET} ${C_DIM}← $(basename "${TRACK_SOURCE[$participant]}")${C_RESET}"
    done
    log_line ""

    if [[ "$DRY_RUN" == 1 ]]; then
        # Nothing was written, so track detail can only come from a meta.json an
        # earlier run left behind.
        if [[ -f "$WORK/meta.json" ]]; then
            load_meta
            log_info "reusing the meta.json from a previous run"
        else
            log_warn "no meta.json yet, so later stages can only be listed"
        fi
    else
        load_meta
    fi

    state_mark discover
    stage_end "${#PARTICIPANTS[@]} tracks"
    return 0
}

# Re-establish the paths a resumed run needs without re-scanning the input dir.
resume_context() {
    [[ -n "$EPISODE_ID" ]] || die "--from requires --episode (or an input dir to scan)"
    WORK="$WORK_ROOT/$EPISODE_ID"
    STAGE_DIR="$WORK/state"
    OUT_DIR="$OUTPUT_DIR/$EPISODE_ID"
    STAGING_DIR="$OUT_DIR/.staging"
    [[ -d "$WORK" ]] || die "no work directory for episode '$EPISODE_ID' at $WORK"
    mkdir -p "$WORK"/{prep,asr,words,llm,render,logs} "$STAGE_DIR" "$OUT_DIR"
    log_init "$WORK/logs/run.log"
    config_need_ffmpeg
    load_meta
}

# ============================================================================
# prepare — 16 kHz mono WAV, the one format Whisper and Silero both want
# ============================================================================

stage_prepare() {
    stage_begin prepare "decoding tracks to 16 kHz mono"
    config_need_ffmpeg

    if [[ "$DRY_RUN" == 1 ]]; then
        log_info "would decode ${#PARTICIPANTS[@]} tracks into $WORK/prep"
        stage_end "dry run"
        return 0
    fi

    PODCAST_LOG_FILE="$LOG_FILE" LOG_LEVEL="$LOG_LEVEL" PODCAST_LOG_STAGE=prepare \
        "$PYTHON" "$LIB_ROOT/python/cleanup_cli.py" stage-prepare \
        --work "$WORK" --ffmpeg "$FFMPEG" \
        || exit 1

    # Durations were replaced with the measured ones, so the shell's copy is out
    # of date.
    load_meta

    state_mark prepare
    stage_end "${#PARTICIPANTS[@]} tracks decoded"
}


# ============================================================================
# transcribe — Whisper per track, all of it finished before any LLM work
# ============================================================================

# ============================================================================
# transcribe — one request per track against a whisper-server
# ============================================================================

stage_transcribe() {
    stage_begin transcribe "transcribing via $WHISPER_ENDPOINT"
    config_need_whisper
    config_need_ffmpeg

    if [[ "$DRY_RUN" == 1 ]]; then
        log_info "would transcribe ${#PARTICIPANTS[@]} tracks via $WHISPER_ENDPOINT"
        stage_end "dry run"
        return 0
    fi

    PODCAST_WHISPER_API_KEY="$WHISPER_API_KEY" \
        PODCAST_LOG_FILE="$LOG_FILE" LOG_LEVEL="$LOG_LEVEL" \
        PODCAST_LOG_STAGE=transcribe \
        "$PYTHON" "$LIB_ROOT/python/cleanup_cli.py" stage-transcribe \
        --work "$WORK" --ffmpeg "$FFMPEG" \
        || exit 1

    state_mark transcribe
    stage_end "${#PARTICIPANTS[@]} tracks transcribed remotely"
}

# ============================================================================
# detect — the LLM finds disfluencies; the server lives and dies in this stage
# ============================================================================


stage_detect() {
    if [[ "$LLM_ENABLE" != 1 ]]; then
        stage_skip detect "LLM_ENABLE=0"
        state_mark detect
        return 0
    fi

    stage_begin detect "finding stutters and false starts"
    config_need_llama
    state_done transcribe || log_warn "transcribe stage has not completed in this work dir"

    if [[ "$DRY_RUN" == 1 ]]; then
        log_info "would analyse ${#PARTICIPANTS[@]} transcripts via $LLAMA_ENDPOINT"
        stage_end "dry run"
        return 0
    fi

    PODCAST_LLAMA_API_KEY="$LLAMA_API_KEY" \
        PODCAST_LOG_FILE="$LOG_FILE" LOG_LEVEL="$LOG_LEVEL" \
        PODCAST_LOG_STAGE=detect \
        "$PYTHON" "$LIB_ROOT/python/cleanup_cli.py" stage-detect \
        --work "$WORK" \
        --resume-hint "$0 --episode $EPISODE_ID --from detect" \
        || exit 1

    state_mark detect
    stage_end "${#PARTICIPANTS[@]} tracks analysed"
}

# ============================================================================
# plan — unify silence and edits into cuts and mutes
# ============================================================================

stage_plan() {
    stage_begin plan "deciding what to cut and what to mute"

    if [[ "$DRY_RUN" == 1 ]]; then
        log_info "would unify $WORK/words and $WORK/llm into $WORK/plan.json"
        stage_end "dry run"
        return 0
    fi

    local -a args=(stage-plan --work "$WORK")
    [[ "${FORCE:-0}" == 1 ]] && args+=(--force)

    # The settings reach it through the environment (config_export), and it
    # writes its own report and refusals into the run log.
    PODCAST_LOG_FILE="$LOG_FILE" LOG_LEVEL="$LOG_LEVEL" PODCAST_LOG_STAGE=plan \
        "$PYTHON" "$LIB_ROOT/python/cleanup_cli.py" "${args[@]}" \
        || exit 1

    state_mark plan
    stage_end "plan written"
}

# ============================================================================
# render — one ffmpeg pass per track, into a staging dir
# ============================================================================

stage_render() {
    stage_begin render "rendering cleaned tracks"
    config_need_ffmpeg

    if [[ "$DRY_RUN" == 1 ]]; then
        log_info "would render ${#PARTICIPANTS[@]} tracks into $STAGING_DIR"
        state_mark render
        stage_end "dry run"
        return 0
    fi

    PODCAST_LOG_FILE="$LOG_FILE" LOG_LEVEL="$LOG_LEVEL" PODCAST_LOG_STAGE=render \
        "$PYTHON" "$LIB_ROOT/python/cleanup_cli.py" stage-render \
        --work "$WORK" --staging "$STAGING_DIR" \
        --ffmpeg "$FFMPEG" --ffprobe "$FFPROBE" \
        || exit 1

    state_mark render
    stage_end "${#PARTICIPANTS[@]} tracks rendered and verified"
}

# ============================================================================
# finalize — publish outputs, then dispose of inputs and intermediates
# ============================================================================

stage_finalize() {
    stage_begin finalize "publishing outputs and cleaning up"

    local -a args=(
        stage-finalize --work "$WORK" --output "$OUT_DIR"
        --staging "$STAGING_DIR" --episode "$EPISODE_ID"
    )
    local participant
    for participant in "${PARTICIPANTS[@]}"; do
        args+=(--source "$participant=${TRACK_SOURCE[$participant]}")
    done

    if [[ "$DRY_RUN" == 1 ]]; then
        log_info "would publish ${#PARTICIPANTS[@]} tracks into $OUT_DIR"
        stage_end "dry run"
        return 0
    fi

    # The run log moves when the work directory is removed, and this shell keeps
    # logging afterwards — the stage says where it ended up.
    local moved
    moved=$(PODCAST_LOG_FILE="$LOG_FILE" LOG_LEVEL="$LOG_LEVEL" \
        PODCAST_LOG_STAGE=finalize \
        "$PYTHON" "$LIB_ROOT/python/cleanup_cli.py" "${args[@]}") || exit 1
    eval "$moved"

    stage_end "outputs in $OUT_DIR"
}
