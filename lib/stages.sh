# shellcheck shell=bash
#
# The pipeline stages.
#
# Each stage is self-contained: it reads what it needs from the episode work
# directory, writes its results back there, and drops a marker in state/ so a
# later run can pick up where this one stopped. That is what makes --from and
# --only work without special cases.
#
# Stage order is not arbitrary. Whisper and llama.cpp must never be resident at
# the same time, so `transcribe` runs every track to completion and lets each
# process exit before `detect` loads a model at all — and `detect` shuts its
# server down before returning, so `render` gets the machine to itself.

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

LLAMA_PID=""
LLAMA_URL=""

ALL_STAGES=(discover prepare vad transcribe detect plan render finalize)

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

# The plan's numeric parameters, handed to Python as one file. Every value has
# already been checked as numeric by config_validate.
write_params() {
    local target="$1"
    printf '{\n' >"$target"
    printf '  "silence_min_duration": %s,\n' "$SILENCE_MIN_DURATION" >>"$target"
    printf '  "silence_keep": %s,\n'         "$SILENCE_KEEP"         >>"$target"
    printf '  "edge_keep": %s,\n'            "$EDGE_KEEP"            >>"$target"
    printf '  "cut_padding": %s,\n'          "$CUT_PADDING"          >>"$target"
    printf '  "min_cut": %s,\n'              "$MIN_CUT"              >>"$target"
    printf '  "mute_fade": %s,\n'            "$MUTE_FADE"            >>"$target"
    printf '  "max_cut_fraction": %s\n'      "$MAX_CUT_FRACTION"     >>"$target"
    printf '}\n' >>"$target"
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

    if (( ${#INPUT_FILES[@]} == 0 )); then
        [[ -d "$INPUT_DIR" ]] || die "input directory does not exist: $INPUT_DIR"
        # Any of the configured extensions, matched case-insensitively — some
        # recorders write .WAV, and ffmpeg does not care either way.
        local -a name_test=()
        local ext
        for ext in $INPUT_EXTS; do
            (( ${#name_test[@]} )) && name_test+=(-o)
            name_test+=(-iname "*.${ext}")
        done
        mapfile -t INPUT_FILES < <(
            find "$INPUT_DIR" -maxdepth 1 -type f \( "${name_test[@]}" \) | sort
        )
    fi

    if (( ${#INPUT_FILES[@]} == 0 )); then
        log_warn "no files matching [$INPUT_EXTS] in $INPUT_DIR — nothing to do"
        return 2
    fi

    # Episode id comes from the part of the filename before the separator; all
    # tracks must agree on it, otherwise two recordings got mixed together.
    local -a track_args=()
    local -A seen=()
    local file base participant episode

    for file in "${INPUT_FILES[@]}"; do
        base=$(basename "$file")
        base="${base%.*}"
        if [[ "$base" != *"$TRACK_SEPARATOR"* ]]; then
            die "cannot parse '$base': expected <episode>${TRACK_SEPARATOR}<participant>.<ext>"
        fi
        episode="${base%%"$TRACK_SEPARATOR"*}"
        participant="${base#*"$TRACK_SEPARATOR"}"

        [[ -n "$episode" && -n "$participant" ]] \
            || die "cannot parse '$base': empty episode or participant"
        [[ "$participant" != *"/"* ]] \
            || die "participant name may not contain a slash: '$participant'"

        if [[ -n "${EPISODE_ID_OVERRIDE:-}" ]]; then
            episode="$EPISODE_ID_OVERRIDE"
        fi
        if [[ -z "$EPISODE_ID" ]]; then
            EPISODE_ID="$episode"
        elif [[ "$episode" != "$EPISODE_ID" ]]; then
            die "found tracks from two episodes ('$EPISODE_ID' and '$episode'); process them separately or pass --episode"
        fi

        # Also catches the same track present in two formats, which would
        # otherwise silently pick whichever sorted first.
        [[ -z "${seen[$participant]:-}" ]] \
            || die "participant '$participant' appears twice: $(basename "${seen[$participant]}") and $(basename "$file"). Keep one and remove the other, or narrow INPUT_EXTS."
        seen["$participant"]="$file"
        track_args+=("$participant=$file")
    done

    WORK="$WORK_ROOT/$EPISODE_ID"
    STAGE_DIR="$WORK/state"
    OUT_DIR="$OUTPUT_DIR/$EPISODE_ID"
    STAGING_DIR="$OUT_DIR/.staging"

    if [[ "$DRY_RUN" != 1 ]]; then
        mkdir -p "$WORK"/{prep,vad,asr,words,llm,render,logs} "$STAGE_DIR" "$OUT_DIR"
    fi

    # From here on the run log lives with the episode. Carry over what the
    # temporary log already holds, rather than starting a fresh one and losing
    # the config dump that precedes episode discovery.
    local episode_log="$WORK/logs/run.log"
    if [[ "$DRY_RUN" != 1 && "$LOG_FILE" != "$episode_log" ]]; then
        local previous="$LOG_FILE"
        if [[ -f "$previous" ]]; then
            cat "$previous" >>"$episode_log"
            rm -f "$previous"
        fi
        LOG_FILE="$episode_log"
        log_raw "=== log adopted into $WORK ==="
    fi

    log_line ""
    log_line "  ${C_BOLD}Episode${C_RESET} $EPISODE_ID  ${C_DIM}(${#track_args[@]} tracks)${C_RESET}"
    for participant in $(printf '%s\n' "${!seen[@]}" | sort); do
        log_line "    ${C_CYAN}${participant}${C_RESET} ${C_DIM}← $(basename "${seen[$participant]}")${C_RESET}"
    done
    log_line ""

    if [[ "$DRY_RUN" == 1 ]]; then
        # Nothing gets written, so the track details can only come from a
        # meta.json an earlier run left behind.
        if [[ -f "$WORK/meta.json" ]]; then
            load_meta
            log_info "reusing the meta.json from a previous run"
        else
            log_warn "no meta.json yet, so later stages can only be listed"
        fi
    else
        local -a meta_args=(
            meta --episode "$EPISODE_ID" --ffprobe "$FFPROBE"
            --out "$WORK/meta.json"
        )
        [[ -n "$RESAMPLE_TO" ]] && meta_args+=(--resample-to "$RESAMPLE_TO")
        local probe_report
        if ! probe_report=$(py "${meta_args[@]}" "${track_args[@]}" 2>&1); then
            # log_line, not log_report: an explanation of a failure has to be
            # visible even under --quiet.
            log_line "$probe_report"
            die "could not inspect the input tracks"
        fi
        log_report "$probe_report"
        load_meta
    fi

    state_mark discover
    stage_end "${#track_args[@]} tracks"
}

# Re-establish the paths a resumed run needs without re-scanning the input dir.
resume_context() {
    [[ -n "$EPISODE_ID" ]] || die "--from requires --episode (or an input dir to scan)"
    WORK="$WORK_ROOT/$EPISODE_ID"
    STAGE_DIR="$WORK/state"
    OUT_DIR="$OUTPUT_DIR/$EPISODE_ID"
    STAGING_DIR="$OUT_DIR/.staging"
    [[ -d "$WORK" ]] || die "no work directory for episode '$EPISODE_ID' at $WORK"
    mkdir -p "$WORK"/{prep,vad,asr,words,llm,render,logs} "$STAGE_DIR" "$OUT_DIR"
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

    local -a markers=()
    local participant target marker

    for participant in "${PARTICIPANTS[@]}"; do
        target="$WORK/prep/$participant.wav"
        marker="$STAGE_DIR/prep-$participant.ok"
        markers+=("$marker")

        if [[ -s "$target" && -f "$marker" ]]; then
            log_debug "$participant already prepared"
            continue
        fi
        rm -f "$marker"

        local -a decode_cmd=(
            "$FFMPEG" -nostdin -y -v warning -progress pipe:1 -nostats
            -i "${TRACK_SOURCE[$participant]}"
            -map 0:a:0 -ac 1 -ar 16000 -c:a pcm_s16le
            -f wav "$target"
        )

        # Live progress only makes sense when one job owns the console; several
        # concurrent writers would just garble each other's counter.
        if (( FFMPEG_JOBS <= 1 )); then
            FFMPEG_TOTAL_US=$(track_total_us "$participant") \
                run_streaming parse_ffmpeg_progress "decoding $participant" \
                "${decode_cmd[@]}" || die "could not decode $participant"
            state_touch "$marker"
        else
            pool_slot "$FFMPEG_JOBS"
            (
                run "${decode_cmd[@]}" \
                    && state_touch "$marker" \
                    && log_ok "decoded $participant"
            ) &
        fi
    done

    pool_wait "decoding" "${markers[@]}"

    # Now that every track has actually been decoded, take its length from the
    # decoded audio rather than from a container header that may be wrong — an
    # AAC file decodes longer than it claims, an Opus one shorter, and a
    # truncated file of any format can claim anything. The frame-exact render
    # prediction is only exact if this is right.
    if [[ "$DRY_RUN" != 1 ]]; then
        local measure_report
        if ! measure_report=$(py meta-refresh --meta "$WORK/meta.json" \
            --prep-dir "$WORK/prep" 2>&1); then
            log_line "$measure_report"
            die "could not measure the decoded track lengths"
        fi
        log_report "$measure_report"
        load_meta
    fi

    state_mark prepare
    stage_end "${#PARTICIPANTS[@]} tracks decoded"
}

# ============================================================================
# vad — where is there speech, per track
# ============================================================================

stage_vad() {
    stage_begin vad "detecting speech (backend: $VAD_BACKEND)"
    config_need_ffmpeg

    local -a markers=()
    local participant target marker wav

    for participant in "${PARTICIPANTS[@]}"; do
        target="$WORK/vad/$participant.json"
        marker="$STAGE_DIR/vad-$participant.ok"
        wav="$WORK/prep/$participant.wav"
        markers+=("$marker")

        if [[ -s "$target" && -f "$marker" ]]; then
            log_debug "$participant already analysed"
            continue
        fi
        rm -f "$marker"
        [[ -s "$wav" || "$DRY_RUN" == 1 ]] || die "missing prepared track: $wav"

        if [[ "$VAD_BACKEND" == silero ]]; then
            # Silero holds a torch model, so these run strictly one at a time.
            log_info "silero: $participant"
            run py vad-silero --wav "$wav" --participant "$participant" \
                --threshold "$SILERO_THRESHOLD" \
                --duration "${TRACK_DURATION[$participant]}" --out "$target" \
                || die "Silero VAD failed on $participant"
            state_touch "$marker"
        else
            pool_slot "$FFMPEG_JOBS"
            (
                silence_log="$WORK/vad/$participant.silence.log"
                run_to_file "$silence_log" \
                    "$FFMPEG" -nostdin -v info -i "$wav" \
                    -af "silencedetect=noise=${SILENCE_THRESHOLD}:d=${VAD_MIN_SILENCE}" \
                    -f null - \
                    && run py vad-ffmpeg --log "$silence_log" \
                        --duration "${TRACK_DURATION[$participant]}" \
                        --participant "$participant" --out "$target" \
                    && state_touch "$marker"
            ) &
        fi
    done

    pool_wait "speech detection" "${markers[@]}"
    state_mark vad
    stage_end "${#PARTICIPANTS[@]} tracks analysed"
}

# ============================================================================
# transcribe — Whisper per track, all of it finished before any LLM work
# ============================================================================

stage_transcribe() {
    if [[ -n "$WHISPER_ENDPOINT" ]]; then
        stage_transcribe_remote
        return $?
    fi

    stage_begin transcribe "transcribing with Whisper (local)"
    config_need_whisper

    local -a markers=()
    local participant marker prefix wav index=0
    local total="${#PARTICIPANTS[@]}"

    for participant in "${PARTICIPANTS[@]}"; do
        index=$(( index + 1 ))
        marker="$STAGE_DIR/asr-$participant.ok"
        prefix="$WORK/asr/$participant"
        wav="$WORK/prep/$participant.wav"
        markers+=("$marker")

        if [[ -s "$WORK/words/$participant.words.json" && -f "$marker" ]]; then
            log_debug "$participant already transcribed"
            continue
        fi
        rm -f "$marker"
        [[ -s "$wav" || "$DRY_RUN" == 1 ]] || die "missing prepared track: $wav"

        local -a whisper_args=(
            -m "$WHISPER_MODEL" -f "$wav" -of "$prefix"
            --output-json-full --print-progress -t "$WHISPER_THREADS"
        )
        [[ "$WHISPER_LANG" != "auto" ]] && whisper_args+=(-l "$WHISPER_LANG")
        # Deliberately unquoted: this is a user-supplied argument string.
        # shellcheck disable=SC2206
        [[ -n "$WHISPER_EXTRA_ARGS" ]] && whisper_args+=($WHISPER_EXTRA_ARGS)

        if (( WHISPER_JOBS <= 1 )); then
            log_info "whisper: $participant ($index/$total)"
            run_streaming parse_whisper_progress "$participant" \
                "$WHISPER" "${whisper_args[@]}" \
                || die "Whisper failed on $participant"
            run py words --whisper-json "$prefix.json" \
                --participant "$participant" \
                --out "$WORK/words/$participant.words.json" \
                || die "could not parse Whisper output for $participant"
            touch "$marker"
        else
            pool_slot "$WHISPER_JOBS"
            (
                run "$WHISPER" "${whisper_args[@]}" \
                    && run py words --whisper-json "$prefix.json" \
                        --participant "$participant" \
                        --out "$WORK/words/$participant.words.json" \
                    && touch "$marker"
            ) &
        fi
    done

    # Nothing may proceed while a Whisper process still holds memory.
    pool_wait "transcription" "${markers[@]}"
    log_ok "all Whisper processes have exited; memory released"
    state_mark transcribe
    stage_end "${#PARTICIPANTS[@]} tracks transcribed"
}

# Transcription against a whisper-server someone else is running. No local
# process, so nothing to serialise against the LLM stage on this machine.
stage_transcribe_remote() {
    stage_begin transcribe "transcribing via $WHISPER_ENDPOINT"
    config_need_python

    if [[ "$DRY_RUN" != 1 ]]; then
        PODCAST_WHISPER_API_KEY="$WHISPER_API_KEY" \
            py whisper-wait --endpoint "$WHISPER_ENDPOINT" \
            --path "$WHISPER_ENDPOINT_PATH" --timeout 60 \
            || die "the whisper endpoint at $WHISPER_ENDPOINT is not usable"
    fi

    local participant target wav index=0
    local total="${#PARTICIPANTS[@]}"

    for participant in "${PARTICIPANTS[@]}"; do
        index=$(( index + 1 ))
        target="$WORK/words/$participant.words.json"
        wav="$WORK/prep/$participant.wav"

        if [[ -s "$target" && -f "$STAGE_DIR/asr-$participant.ok" ]]; then
            log_debug "$participant already transcribed"
            continue
        fi
        [[ -s "$wav" || "$DRY_RUN" == 1 ]] || die "missing prepared track: $wav"

        log_info "whisper (remote): $participant ($index/$total)"
        local -a args=(
            transcribe-remote
            --wav "$wav" --participant "$participant" --out "$target"
            --endpoint "$WHISPER_ENDPOINT" --path "$WHISPER_ENDPOINT_PATH"
            --vad "$WORK/vad/$participant.json"
            --chunk-seconds "$WHISPER_CHUNK_SECONDS"
            --language "$WHISPER_LANG"
            --request-timeout "$WHISPER_REQUEST_TIMEOUT"
        )
        if [[ "$DRY_RUN" == 1 ]]; then
            run py "${args[@]}"
        else
            PODCAST_WHISPER_API_KEY="$WHISPER_API_KEY" \
                run_streaming parse_python_progress "$participant" \
                "$PYTHON" "$LIB_ROOT/python/cleanup_cli.py" "${args[@]}" \
                || die "remote transcription failed for $participant"
            state_touch "$STAGE_DIR/asr-$participant.ok"
        fi
    done

    state_mark transcribe
    stage_end "$total tracks transcribed remotely"
}

# ============================================================================
# detect — Qwen finds disfluencies; the server lives and dies inside this stage
# ============================================================================

llama_start() {
    config_need_python

    # An endpoint supplied by the user is still checked for life before we start
    # sending it two hours of transcript.
    if [[ -n "$LLAMA_ENDPOINT" ]]; then
        LLAMA_URL="$LLAMA_ENDPOINT"
        log_info "using the llama server already at $LLAMA_URL"
        [[ "$DRY_RUN" == 1 ]] && return 0
        PODCAST_LLAMA_API_KEY="$LLAMA_API_KEY" \
            py llm-wait --endpoint "$LLAMA_URL" --timeout 30 \
            || die "the configured llama endpoint at $LLAMA_URL is not usable"
        log_ok "endpoint is responding"
        return 0
    fi

    LLAMA_URL="http://${LLAMA_HOST}:${LLAMA_PORT}"
    local server_log="$WORK/logs/llama-server.log"
    log_info "starting llama-server on $LLAMA_URL ($(basename "$LLAMA_MODEL"))"
    log_raw "\$ $LLAMA_SERVER -m $LLAMA_MODEL --host $LLAMA_HOST --port $LLAMA_PORT -c $LLAMA_CTX -ngl $LLAMA_NGL $LLAMA_EXTRA_ARGS"

    if [[ "$DRY_RUN" == 1 ]]; then
        return 0
    fi

    # shellcheck disable=SC2206
    local -a extra=()
    [[ -n "$LLAMA_EXTRA_ARGS" ]] && extra=($LLAMA_EXTRA_ARGS)

    "$LLAMA_SERVER" \
        -m "$LLAMA_MODEL" \
        --host "$LLAMA_HOST" \
        --port "$LLAMA_PORT" \
        -c "$LLAMA_CTX" \
        -ngl "$LLAMA_NGL" \
        "${extra[@]}" \
        >"$server_log" 2>&1 &
    LLAMA_PID=$!
    log_debug "llama-server pid $LLAMA_PID, log $server_log"

    if ! PODCAST_LLAMA_API_KEY="$LLAMA_API_KEY" \
        py llm-wait --endpoint "$LLAMA_URL" --timeout "$LLAMA_STARTUP_TIMEOUT"; then
        log_error "llama-server never became ready; last lines of its log:"
        [[ -f "$server_log" ]] && log_line "$(tail -n 20 "$server_log")"
        llama_stop
        die "giving up on the LLM stage"
    fi
    log_ok "model loaded and serving"
}

llama_stop() {
    [[ -n "$LLAMA_PID" ]] || return 0
    local pid="$LLAMA_PID"
    LLAMA_PID=""
    if kill -0 "$pid" 2>/dev/null; then
        log_info "stopping llama-server (pid $pid) to free memory"
        kill "$pid" 2>/dev/null || true
        local waited=0
        while kill -0 "$pid" 2>/dev/null && (( waited < 30 )); do
            sleep 1
            waited=$(( waited + 1 ))
        done
        if kill -0 "$pid" 2>/dev/null; then
            log_warn "llama-server ignored SIGTERM; sending SIGKILL"
            kill -9 "$pid" 2>/dev/null || true
        fi
        wait "$pid" 2>/dev/null || true
        log_ok "llama-server stopped"
    fi
}

stage_detect() {
    if [[ "$LLM_ENABLE" != 1 ]]; then
        stage_skip detect "LLM_ENABLE=0"
        state_mark detect
        return 0
    fi

    stage_begin detect "finding stutters and false starts"
    config_need_llama
    state_done transcribe || log_warn "transcribe stage has not completed in this work dir"

    llama_start

    local participant words target index=0
    local total="${#PARTICIPANTS[@]}"
    local failed=0

    for participant in "${PARTICIPANTS[@]}"; do
        index=$(( index + 1 ))
        words="$WORK/words/$participant.words.json"
        target="$WORK/llm/$participant.edits.json"

        if [[ ! -s "$words" ]]; then
            if [[ "$DRY_RUN" == 1 ]]; then
                continue
            fi
            log_warn "no transcript for $participant; skipping edit detection"
            continue
        fi
        if [[ -s "$target" && -f "$STAGE_DIR/llm-$participant.ok" ]]; then
            log_debug "$participant already analysed by the LLM"
            continue
        fi

        log_info "qwen: $participant ($index/$total)"
        local rc=0
        PODCAST_LLAMA_API_KEY="$LLAMA_API_KEY" \
            run_streaming parse_python_progress "$participant" \
            "$PYTHON" "$LIB_ROOT/python/cleanup_cli.py" detect \
            --words "$words" --endpoint "$LLAMA_URL" --out "$target" \
            --audit "$WORK/llm/$participant.audit.jsonl" \
            --chunk-words "$LLM_CHUNK_WORDS" --overlap "$LLM_CHUNK_OVERLAP" \
            --max-words "$LLM_MAX_EDIT_WORDS" --max-seconds "$LLM_MAX_EDIT_SECONDS" \
            --min-confidence "$LLM_MIN_CONFIDENCE" --temperature "$LLM_TEMP" \
            --request-timeout "$LLAMA_REQUEST_TIMEOUT" --kinds "$LLM_ACCEPT_KINDS" \
            || rc=$?

        if (( rc == 0 )); then
            touch "$STAGE_DIR/llm-$participant.ok"
        elif (( rc == 2 )); then
            # Exit 2 is a refused API key. Every remaining track would fail the
            # same way, and carrying on would deliver an episode that quietly
            # found no edits at all — so this one stops the run.
            llama_stop
            die "the LLM endpoint refused our credentials. Fix the key, then resume with: $0 --episode $EPISODE_ID --from detect"
        else
            failed=$(( failed + 1 ))
            log_warn "edit detection failed for $participant; that track keeps its disfluencies"
        fi
    done

    # Free the model before anything else runs, whatever happened above.
    llama_stop

    if (( failed == total )) && (( total > 0 )); then
        die "edit detection failed for every track"
    fi
    state_mark detect
    stage_end "$(( total - failed ))/$total tracks analysed"
}

# ============================================================================
# plan — unify silence and edits into cuts and mutes
# ============================================================================

stage_plan() {
    stage_begin plan "deciding what to cut and what to mute"

    if [[ "$DRY_RUN" == 1 ]]; then
        log_info "would unify $WORK/vad and $WORK/llm into $WORK/plan.json"
        stage_end "dry run"
        return 0
    fi

    write_params "$WORK/params.json"
    local -a plan_args=(
        plan
        --meta "$WORK/meta.json"
        --params "$WORK/params.json"
        --vad-dir "$WORK/vad"
        --words-dir "$WORK/words"
        --out "$WORK/plan.json"
        --report "$WORK/edit-report.txt"
    )
    [[ "$LLM_ENABLE" == 1 ]] && plan_args+=(--edits-dir "$WORK/llm")
    [[ "${FORCE:-0}" == 1 ]] && plan_args+=(--force)

    # The report is the one piece of stage output worth showing in full.
    local report
    if ! report=$(py "${plan_args[@]}" 2>&1); then
        log_raw "$report"
        log_line "$report"
        die "planning failed"
    fi
    log_report "$report"

    run py filters --meta "$WORK/meta.json" --plan "$WORK/plan.json" \
        --dir "$WORK/render" --out "$WORK/expected.json" \
        --frame-samples "$RENDER_FRAME_SAMPLES" --fade "$MUTE_FADE" \
        || die "could not build the render filters"

    state_mark plan
    stage_end "plan written"
}

# ============================================================================
# render — one ffmpeg pass per track, into a staging dir
# ============================================================================

stage_render() {
    stage_begin render "rendering cleaned tracks"
    config_need_ffmpeg
    [[ -s "$WORK/plan.json" || "$DRY_RUN" == 1 ]] || die "no plan.json; run the plan stage first"

    [[ "$DRY_RUN" == 1 ]] || mkdir -p "$STAGING_DIR"

    local -a markers=()
    local participant target marker filter fmt source
    local -a fmt_args=() encode_args=()

    for participant in "${PARTICIPANTS[@]}"; do
        source="${TRACK_SOURCE[$participant]}"
        target="$STAGING_DIR/${participant}${OUTPUT_SUFFIX}.${OUTPUT_EXT}"
        marker="$STAGE_DIR/render-$participant.ok"
        filter="$WORK/render/$participant.filter"
        markers+=("$marker")
        rm -f "$marker"

        # Bit depth is only meaningful to encoders that have one to preserve.
        fmt="${TRACK_SAMPLE_FMT[$participant]:-}"
        fmt_args=()
        case "$OUTPUT_CODEC" in
            flac|alac|pcm_*|wavpack)
                case "$fmt" in
                    s16|s32) fmt_args=(-sample_fmt "$fmt") ;;
                esac
                ;;
        esac

        encode_args=(-c:a "$OUTPUT_CODEC")
        [[ "$OUTPUT_CODEC" == flac ]] \
            && encode_args+=(-compression_level "$OUTPUT_COMPRESSION")
        # Deliberately unquoted: a user-supplied argument string.
        # shellcheck disable=SC2206
        [[ -n "$OUTPUT_EXTRA_ARGS" ]] && encode_args+=($OUTPUT_EXTRA_ARGS)
        encode_args+=("${fmt_args[@]}")

        if [[ ! -f "$filter" ]]; then
            # No edits and no resampling. Copying beats re-encoding, but only
            # when the file is already in the format being asked for —
            # otherwise it still has to be converted.
            if [[ "${TRACK_CODEC[$participant]:-}" == "$OUTPUT_CODEC" \
                  && "${source##*.}" == "$OUTPUT_EXT" ]]; then
                log_info "$participant needs no edits and is already $OUTPUT_CODEC, copying it through"
                if [[ "$DRY_RUN" != 1 ]]; then
                    cp -- "$source" "$target" || die "could not copy $participant"
                    state_touch "$marker"
                fi
                continue
            fi
            log_info "$participant needs no edits, converting to $OUTPUT_CODEC"
            if [[ "$DRY_RUN" != 1 ]]; then
                run "$FFMPEG" -nostdin -y -v warning -i "$source" \
                    -map 0:a:0 "${encode_args[@]}" "$target" \
                    || die "could not convert $participant"
                state_touch "$marker"
            fi
            continue
        fi

        local -a render_cmd=(
            "$FFMPEG" -nostdin -y -v warning -progress pipe:1 -nostats
            -i "$source"
            -filter_complex_script "$filter" -map '[out]'
            "${encode_args[@]}" "$target"
        )

        if (( FFMPEG_JOBS <= 1 )); then
            FFMPEG_TOTAL_US=$(track_total_us "$participant") \
                run_streaming parse_ffmpeg_progress "rendering $participant" \
                "${render_cmd[@]}" || die "could not render $participant"
            state_touch "$marker"
        else
            pool_slot "$FFMPEG_JOBS"
            (
                run "${render_cmd[@]}" \
                    && state_touch "$marker" \
                    && log_ok "rendered $participant"
            ) &
        fi
    done

    pool_wait "rendering" "${markers[@]}"

    if [[ "$DRY_RUN" == 1 ]]; then
        state_mark render
        stage_end "dry run"
        return 0
    fi

    # Verify against the frame-exact prediction, not a rule of thumb.
    local -a actual=()
    for participant in "${PARTICIPANTS[@]}"; do
        target="$STAGING_DIR/${participant}${OUTPUT_SUFFIX}.${OUTPUT_EXT}"
        [[ -s "$target" ]] || die "rendered file is missing or empty: $target"
        actual+=("$participant=$(probe_duration "$target")")
    done

    local verdict
    if ! verdict=$(py verify --expected "$WORK/expected.json" \
        --tolerance "$DURATION_TOLERANCE" "${actual[@]}" 2>&1); then
        log_raw "$verdict"
        log_line "$verdict"
        die "rendered durations do not match the plan; inputs and work directory kept"
    fi
    log_report "$verdict"

    state_mark render
    stage_end "${#PARTICIPANTS[@]} tracks rendered and verified"
}

# ============================================================================
# finalize — publish outputs, then dispose of inputs and intermediates
# ============================================================================

stage_finalize() {
    stage_begin finalize "publishing outputs and cleaning up"

    run py transcript --plan "$WORK/plan.json" --words-dir "$WORK/words" \
        --out-json "$WORK/${EPISODE_ID}_transcript.json" \
        --out-srt "$WORK/${EPISODE_ID}_transcript.srt" \
        --out-txt "$WORK/${EPISODE_ID}_transcript.txt" \
        || die "could not build the final transcript"

    if [[ "$DRY_RUN" == 1 ]]; then
        stage_end "dry run"
        return 0
    fi

    mkdir -p "$OUT_DIR/logs"

    # Move the rendered audio into place. Staging sits inside the output
    # directory so this is a rename on the same filesystem, not a copy.
    local participant staged final
    for participant in "${PARTICIPANTS[@]}"; do
        staged="$STAGING_DIR/${participant}${OUTPUT_SUFFIX}.${OUTPUT_EXT}"
        final="$OUT_DIR/${participant}${OUTPUT_SUFFIX}.${OUTPUT_EXT}"
        mv -f -- "$staged" "$final" || die "could not publish $final"
        log_ok "$(basename "$final") ($(du -h "$final" | cut -f1))"
    done
    rmdir "$STAGING_DIR" 2>/dev/null || true

    # Sidecar artefacts, every one prefixed with the episode id.
    local pair source_name target_name
    for pair in \
        "${EPISODE_ID}_transcript.json:${EPISODE_ID}_transcript.json" \
        "${EPISODE_ID}_transcript.srt:${EPISODE_ID}_transcript.srt" \
        "${EPISODE_ID}_transcript.txt:${EPISODE_ID}_transcript.txt" \
        "plan.json:${EPISODE_ID}_plan.json" \
        "edit-report.txt:${EPISODE_ID}_edit-report.txt"
    do
        source_name="${pair%%:*}"
        target_name="${pair#*:}"
        [[ -f "$WORK/$source_name" ]] || continue
        cp -f -- "$WORK/$source_name" "$OUT_DIR/$target_name"
    done

    # Logs outlive everything else, by design.
    cp -f -- "$LOG_FILE" "$OUT_DIR/logs/run.log"
    [[ -f "$WORK/logs/llama-server.log" ]] \
        && cp -f -- "$WORK/logs/llama-server.log" "$OUT_DIR/logs/"
    local audit
    for audit in "$WORK"/llm/*.audit.jsonl; do
        [[ -f "$audit" ]] && cp -f -- "$audit" "$OUT_DIR/logs/"
    done

    # Inputs go only after every output is on disk and verified.
    if [[ "$KEEP_INPUTS" == 1 ]]; then
        log_info "keeping original inputs (KEEP_INPUTS=1)"
    else
        local source
        for participant in "${PARTICIPANTS[@]}"; do
            source="${TRACK_SOURCE[$participant]}"
            [[ -f "$source" ]] || continue
            rm -f -- "$source" && log_debug "removed input $(basename "$source")"
        done
        log_ok "original inputs removed"
    fi

    if [[ "$KEEP_WORK" == 1 ]]; then
        log_info "keeping work directory (KEEP_WORK=1): $WORK"
    else
        cp -f -- "$LOG_FILE" "$OUT_DIR/logs/run.log"
        LOG_FILE="$OUT_DIR/logs/run.log"
        rm -rf -- "$WORK"
        log_ok "work directory removed"
    fi

    stage_end "outputs in $OUT_DIR"
}
