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
    printf '  "speech_pad": %s,\n'           "$SPEECH_PAD"           >>"$target"
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

    if (( ${#INPUT_FILES[@]} == 0 )) && [[ ! -d "$INPUT_DIR" ]]; then
        # Only reachable on a dry run, which creates nothing: every other run
        # has had config_make_tree build the layout by now. An absent input
        # directory is the same situation as an empty one.
        log_warn "input directory does not exist yet: $INPUT_DIR"
        return 2
    fi

    if (( ${#INPUT_FILES[@]} == 0 )); then
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
        mkdir -p "$WORK"/{prep,asr,words,llm,render,logs} "$STAGE_DIR" "$OUT_DIR"
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

# Where a long track may be split, for the transcribe stage. Only the boundary
# placement rides on this: whether a spot is quiet enough to split on is a much
# smaller question than what counts as speech, which Whisper's own Silero pass
# answers. Written per track and skipped entirely for a track short enough to go
# in one request.
write_silence_log() {
    local participant="$1" wav="$2" target="$3"

    run_to_file "$target" \
        "$FFMPEG" -nostdin -v info -i "$wav" \
        -af "silencedetect=noise=${SPLIT_SILENCE_THRESHOLD}:d=${SPLIT_MIN_SILENCE}" \
        -f null - \
        || log_warn "could not scan $participant for quiet spots; chunk boundaries may land mid-word"
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
        # Same Silero pass the remote path asks the server for. Without it this
        # track's silence gets transcribed, and whatever Whisper invents there
        # becomes speech in the plan.
        if [[ "$WHISPER_VAD" == 1 ]]; then
            whisper_args+=(
                --vad -vm "$WHISPER_VAD_MODEL"
                -vt "$WHISPER_VAD_THRESHOLD"
                -vspd "$WHISPER_VAD_MIN_SPEECH_MS"
                -vsd "$WHISPER_VAD_MIN_SILENCE_MS"
                -vp "$WHISPER_VAD_SPEECH_PAD_MS"
                -vo "$WHISPER_VAD_SAMPLES_OVERLAP"
            )
        fi
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

        # Only a track that will actually be split needs somewhere to split it.
        local silence_log=""
        if [[ "$WHISPER_CHUNK_SECONDS" != 0 && "$DRY_RUN" != 1 ]] \
           && awk -v d="${TRACK_DURATION[$participant]:-0}" \
                  -v c="$WHISPER_CHUNK_SECONDS" 'BEGIN{exit !(d > c)}'; then
            silence_log="$WORK/asr/$participant.silence.log"
            write_silence_log "$participant" "$wav" "$silence_log"
        fi

        local -a args=(
            transcribe-remote
            --wav "$wav" --participant "$participant" --out "$target"
            --endpoint "$WHISPER_ENDPOINT" --path "$WHISPER_ENDPOINT_PATH"
            --duration "${TRACK_DURATION[$participant]}"
            --chunk-seconds "$WHISPER_CHUNK_SECONDS"
            --language "$WHISPER_LANG"
            --request-timeout "$WHISPER_REQUEST_TIMEOUT"
        )
        [[ -n "$silence_log" ]] && args+=(--silence-log "$silence_log")
        if [[ "$WHISPER_VAD" == 1 ]]; then
            args+=(
                --vad-threshold "$WHISPER_VAD_THRESHOLD"
                --vad-min-speech-ms "$WHISPER_VAD_MIN_SPEECH_MS"
                --vad-min-silence-ms "$WHISPER_VAD_MIN_SILENCE_MS"
                --vad-speech-pad-ms "$WHISPER_VAD_SPEECH_PAD_MS"
                --vad-samples-overlap "$WHISPER_VAD_SAMPLES_OVERLAP"
            )
        else
            args+=(--no-vad)
        fi
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
# detect — the LLM finds disfluencies; the server lives and dies in this stage
# ============================================================================

# Whether to spend one request confirming the server honours a JSON schema.
llm_schema_check_flag() {
    [[ "$LLM_CHECK_SCHEMA" == 1 ]] && printf '%s' "--check-schema"
    return 0
}

# The slot count LLAMA_EXTRA_ARGS pins, if it pins one. An explicit -np there is
# the user being specific about their server, so it wins over LLM_CONCURRENCY
# and we add no flag of our own.
llm_explicit_slots() {
    # shellcheck disable=SC2206
    local -a extra=($LLAMA_EXTRA_ARGS)
    local i
    for (( i = 0; i < ${#extra[@]}; i++ )); do
        case "${extra[i]}" in
            -np|--parallel)     printf '%s' "${extra[i + 1]:-}"; return 0 ;;
            -np=*|--parallel=*) printf '%s' "${extra[i]#*=}";    return 0 ;;
        esac
    done
    return 0
}

# Unless llama-server is given --kv-unified, -c is the whole KV budget and each
# slot gets -c / -np (rounded down to a multiple of 256). So raising the slot
# count silently shrinks the window every chunk has to fit into, and a chunk
# that no longer fits is refused, dropped, and reported only as a track that
# found suspiciously few disfluencies.
#
# Roughly eight tokens per word — each one travels twice, in the flowing text
# and again in the indexed listing — plus the ~570-token instruction prefix and
# whatever is reserved for the reply, rounded up to the next 1024. That is the
# arithmetic behind the sizing table in the README, and it reproduces it
# exactly; if one moves, move the other.
llm_context_warning() {
    local slots="$1"
    [[ "$slots" =~ ^[0-9]+$ ]] && (( slots > 0 )) || return 0

    local per_slot=$(( LLAMA_CTX / slots ))
    local needed=$(( LLM_CHUNK_WORDS * 8 + 570 + LLM_MAX_REPLY_TOKENS ))
    needed=$(( (needed + 1023) / 1024 * 1024 ))
    (( per_slot >= needed )) && return 0

    log_warn "LLAMA_CTX=$LLAMA_CTX split over $slots slots leaves ~$per_slot tokens each, but a ${LLM_CHUNK_WORDS}-word window needs roughly $needed"
    log_warn "raise LLAMA_CTX to $(( needed * slots )), lower LLM_CHUNK_WORDS, or add --kv-unified to LLAMA_EXTRA_ARGS — otherwise chunks are refused and those disfluencies survive"
}

llama_start() {
    config_need_python

    # An endpoint supplied by the user is still checked for life before we start
    # sending it two hours of transcript.
    if [[ -n "$LLAMA_ENDPOINT" ]]; then
        LLAMA_URL="$LLAMA_ENDPOINT"
        log_info "using the llama server already at $LLAMA_URL"
        # Its slot count and context are its own business — nothing here can
        # read them back, so the arithmetic llm_context_warning does for a
        # server we start cannot be done for this one.
        if (( LLM_CONCURRENCY > 1 )); then
            log_info "sending $LLM_CONCURRENCY windows at a time; that server needs at least that many --parallel slots, each with room for a ${LLM_CHUNK_WORDS}-word window"
        fi
        [[ "$DRY_RUN" == 1 ]] && return 0
        # Unquoted on purpose: the helper yields one flag or no argument at all.
        # shellcheck disable=SC2046
        PODCAST_LLAMA_API_KEY="$LLAMA_API_KEY" \
            py llm-wait --endpoint "$LLAMA_URL" --timeout 30 \
            --api "$LLM_API" --model-name "$LLAMA_MODEL_NAME" $(llm_schema_check_flag) \
            || die "the configured llama endpoint at $LLAMA_URL is not usable"
        log_ok "endpoint is responding"
        return 0
    fi

    LLAMA_URL="http://${LLAMA_HOST}:${LLAMA_PORT}"
    local server_log="$WORK/logs/llama-server.log"

    # One slot per window we intend to have in flight, unless LLAMA_EXTRA_ARGS
    # already says otherwise. Deriving it here is what keeps the two settings
    # from drifting apart: a server with fewer slots than LLM_CONCURRENCY does
    # not fail, it just queues the surplus internally and looks slow for no
    # visible reason.
    local -a parallel=()
    local slots
    slots=$(llm_explicit_slots)
    if [[ -n "$slots" ]]; then
        log_debug "LLAMA_EXTRA_ARGS pins the slot count at $slots; not deriving one"
    elif (( LLM_CONCURRENCY > 1 )); then
        slots="$LLM_CONCURRENCY"
        parallel=(--parallel "$LLM_CONCURRENCY")
    else
        slots=1
    fi
    llm_context_warning "$slots"

    log_info "starting llama-server on $LLAMA_URL ($(basename "$LLAMA_MODEL"))"
    log_raw "\$ $LLAMA_SERVER -m $LLAMA_MODEL --host $LLAMA_HOST --port $LLAMA_PORT -c $LLAMA_CTX -ngl $LLAMA_NGL ${parallel[*]} $LLAMA_EXTRA_ARGS"

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
        "${parallel[@]}" \
        "${extra[@]}" \
        >"$server_log" 2>&1 &
    LLAMA_PID=$!
    log_debug "llama-server pid $LLAMA_PID, log $server_log"

    # shellcheck disable=SC2046
    if ! PODCAST_LLAMA_API_KEY="$LLAMA_API_KEY" \
        py llm-wait --endpoint "$LLAMA_URL" --timeout "$LLAMA_STARTUP_TIMEOUT" \
        --api "$LLM_API" --model-name "$LLAMA_MODEL_NAME" $(llm_schema_check_flag); then
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

        log_info "llm: $participant ($index/$total)"
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
            --api "$LLM_API" --max-reply-tokens "$LLM_MAX_REPLY_TOKENS" \
            --model-name "$LLAMA_MODEL_NAME" \
            --concurrency "$LLM_CONCURRENCY" \
            || rc=$?

        if (( rc == 0 )); then
            touch "$STAGE_DIR/llm-$participant.ok"
        elif (( rc == 2 )); then
            # Exit 2 is a refused API key. Every remaining track would fail the
            # same way, and carrying on would deliver an episode that quietly
            # found no edits at all — so this one stops the run.
            llama_stop
            die "the LLM endpoint refused our credentials. Fix the key, then resume with: $0 --episode $EPISODE_ID --from detect"
        elif (( rc == 3 )); then
            # Exit 3 is a server that will not constrain its output. Same
            # reasoning: every track would fail identically and silently.
            llama_stop
            die "the LLM endpoint ignored the JSON schema. Try LLM_API=completion, then resume with: $0 --episode $EPISODE_ID --from detect"
        elif (( rc == 5 )); then
            # Exit 5 is a server that is up but has no model to serve. Same
            # reasoning as 2 and 3: it is a property of the server, so every
            # remaining track would fail identically. A router can drop the
            # model *during* the run — transcribe takes minutes, and nothing
            # reloads it with --no-models-autoload — so this is worth its own
            # message rather than one wasted track after another.
            llama_stop
            die "the LLM endpoint has no model loaded${LLAMA_MODEL_NAME:+ for '$LLAMA_MODEL_NAME'}. Load it, then resume with: $0 --episode $EPISODE_ID --from detect"
        else
            # Exit 4 is every chunk of this track failing — the track was not
            # analysed at all, rather than analysed and found clean. It lands
            # here with the other per-track failures on purpose: one bad track
            # is survivable, and the run should still deliver the rest.
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
        log_info "would unify $WORK/words and $WORK/llm into $WORK/plan.json"
        stage_end "dry run"
        return 0
    fi

    write_params "$WORK/params.json"
    local -a plan_args=(
        plan
        --meta "$WORK/meta.json"
        --params "$WORK/params.json"
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
