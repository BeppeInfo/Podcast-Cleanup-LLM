# shellcheck shell=bash
#
# Defaults, config-file loading and validation.
#
# The config file is sourced as bash, so it can only set variables (and may
# reference earlier ones). Tool resolution is deliberately lazy: a run that
# only re-renders should not require whisper or llama to be installed.

# --- defaults ----------------------------------------------------------------

config_defaults() {
    # Filesystem layout ------------------------------------------------------
    # Everything the run reads and writes lives under one root, which defaults
    # to the directory holding clean-podcast.sh: a checkout run in place is
    # self-contained, which is what a workstation wants. A server install points
    # PODCAST_ROOT at its media volume, or sets the four directories below
    # individually when they live on different ones.
    #
    # These are deliberately left empty here and filled in by
    # config_resolve_paths, which runs after the command line has been applied —
    # otherwise --root could not affect a directory that had already defaulted.
    : "${PODCAST_ROOT:=}"
    : "${INPUT_DIR:=}"
    : "${OUTPUT_DIR:=}"
    : "${WORK_ROOT:=}"
    : "${FAILED_DIR:=}"

    # Track naming: <episode><SEP><participant>.<ext>
    : "${TRACK_SEPARATOR:=_}"

    # Input extensions to look for, space separated and matched
    # case-insensitively. Anything ffmpeg can decode works; the prepare stage
    # decodes to PCM regardless, so the container is only a discovery matter.
    # Video containers are fine too — the first audio stream is taken.
    : "${INPUT_EXTS:=flac wav wave aiff aif m4a mp4 mka mkv mp3 ogg oga opus wv ape alac}"

    # Deprecated: TRACK_EXT used to mean both "what to look for" and "what to
    # write". Honoured as an input filter so existing configs keep working.
    : "${TRACK_EXT:=}"

    # Whisper ---------------------------------------------------------------
    # WHISPER_ENDPOINT: when set, transcription is sent to that whisper-server
    # instead of running whisper-cli locally, and no local process is managed.
    # Independent of how the LLM stage is served: local Whisper with a remote
    # llama, or the reverse, are both fine.
    : "${WHISPER_ENDPOINT:=}"
    : "${WHISPER_ENDPOINT_PATH:=/inference}"
    # Sent as "Authorization: Bearer <key>". whisper.cpp's own server has no
    # auth, so this is for a reverse proxy in front of it. Prefer the _FILE form:
    # a key in the config file is a key in your editor's backups.
    : "${WHISPER_API_KEY:=}"
    : "${WHISPER_API_KEY_FILE:=}"
    # Audio is uploaded in chunks of this many seconds (0 sends the whole
    # track). Boundaries are nudged onto a quiet spot ffmpeg found, so a split
    # does not land mid-word.
    #
    # This is not only an upload-size knob, but it is not a fix for the thing it
    # mitigates either, so it is worth knowing what it does and does not buy.
    #
    # Whisper decodes in 30-second windows (WHISPER_CHUNK_SIZE in whisper.h), and
    # a window whose decode ends on a lone timestamp token is discarded whole —
    # "single timestamp ending - skip entire chunk" in whisper.cpp. On a real 600s
    # track that cost 33 seconds of clear speech, silently: with VAD the windows
    # are cut from *filtered* audio, so one skipped 30s window spanned 33s of the
    # original once the silence inside it is counted back.
    #
    # Which window a passage lands in depends on how much speech precedes it, so
    # request length shuffles the alignment rather than curing anything: the same
    # passage survived a 100s request and vanished from a 300s and a 600s one.
    # 120 is short enough to keep any single loss small and to give the audio
    # cross-check in `plan` a useful signal. That check, not this number, is what
    # actually stops the damage.
    : "${WHISPER_CHUNK_SECONDS:=120}"
    : "${WHISPER_REQUEST_TIMEOUT:=1800}"

    : "${WHISPER_BIN:=/opt/whisper.cpp/bin/whisper-cli}"
    : "${WHISPER_MODEL:=/srv/llm/models/whisper/ggml-large-v3-turbo.bin}"
    : "${WHISPER_THREADS:=$(nproc 2>/dev/null || echo 4)}"
    : "${WHISPER_LANG:=auto}"
    : "${WHISPER_EXTRA_ARGS:=}"

    # Whisper's own Silero pass ----------------------------------------------
    # This is the only speech detection in the pipeline. Whisper transcribes just
    # what Silero calls speech, and the plan derives its speech map from the words
    # that come back — so these settings decide both what gets transcribed and
    # what counts as silence to cut. Turning it off (0) means silence is
    # transcribed too, and whatever Whisper invents there becomes speech in the
    # plan, because nothing else looks at the audio.
    : "${WHISPER_VAD:=1}"
    # The Silero model. A remote server takes it at launch (-vm) and cannot be
    # told per request, so this is for local whisper-cli runs only; get one with
    # whisper.cpp's models/download-vad-model.sh.
    : "${WHISPER_VAD_MODEL:=/srv/llm/models/whisper/ggml-silero-v6.2.0.bin}"
    : "${WHISPER_VAD_THRESHOLD:=0.5}"
    : "${WHISPER_VAD_MIN_SPEECH_MS:=250}"
    # whisper.cpp defaults this to 100ms, which ends a speech segment at every
    # breath. Each segment is transcribed with less of its surroundings, and
    # Whisper punctuates from context, so it is raised well past a natural pause.
    : "${WHISPER_VAD_MIN_SILENCE_MS:=1000}"
    # whisper.cpp defaults to 30ms. A word's opening consonant is quieter than
    # the vowel behind it, so segment edges are exactly where Silero errs, and
    # here a clipped edge is a word that never reaches the transcript at all.
    : "${WHISPER_VAD_SPEECH_PAD_MS:=300}"
    : "${WHISPER_VAD_SAMPLES_OVERLAP:=0.1}"

    # Re-ask about any loud stretch the first pass returned no words for, one span
    # at a time. This is the only thing that recovers a decode window Whisper threw
    # away: re-sending the span on its own changes the window alignment, so the
    # audio that fell in a discarded window no longer does. Costs one extra request
    # per missing stretch, and nothing at all on a track with none.
    : "${WHISPER_RECOVER:=1}"

    # llama.cpp -------------------------------------------------------------
    # LLAMA_ENDPOINT: if set, an already-running server is used as-is and this
    # script never spawns or stops one.
    : "${LLAMA_ENDPOINT:=}"
    # Matches llama-server's own --api-key. Applies to a local server too, if
    # LLAMA_EXTRA_ARGS starts one with --api-key.
    : "${LLAMA_API_KEY:=}"
    : "${LLAMA_API_KEY_FILE:=}"
    : "${LLAMA_SERVER_BIN:=/opt/llama.cpp/bin/llama-server}"
    : "${LLAMA_MODEL:=/srv/llm/models/qwen/Qwen3.6-35B-A3B.gguf}"
    : "${LLAMA_HOST:=127.0.0.1}"
    : "${LLAMA_PORT:=8081}"
    : "${LLAMA_CTX:=8192}"
    : "${LLAMA_NGL:=99}"
    : "${LLAMA_EXTRA_ARGS:=}"
    : "${LLAMA_STARTUP_TIMEOUT:=600}"     # seconds to wait for /health
    : "${LLAMA_REQUEST_TIMEOUT:=600}"     # seconds per completion request

    # Which llama.cpp endpoint the detection stage talks to.
    #   chat        /v1/chat/completions — the server applies the loaded model's
    #               own chat template, so the model can be swapped or upgraded
    #               without anything here knowing which one it is
    #   completion  /completion — a raw prompt with no template applied, the way
    #               this used to work. Kept for a build without the chat
    #               endpoint, or to compare the two on the same episode.
    : "${LLM_API:=chat}"
    # The model id to name in each request. A single-model llama-server ignores
    # it; one in router mode serves several and refuses a request that does not
    # name one, so it is required there. `curl <endpoint>/v1/models` lists them.
    : "${LLAMA_MODEL_NAME:=}"
    # Ceiling on one reply. A chunk of 350 words rarely yields more than a
    # handful of edits, so this is generous; raise it only if replies are being
    # truncated mid-JSON.
    : "${LLM_MAX_REPLY_TOKENS:=2048}"
    # Send one tiny schema-constrained request before the episode starts, to
    # catch a server that answers /health but does not honour response_format.
    : "${LLM_CHECK_SCHEMA:=1}"
    # How many transcript windows are in flight at once. Chunks are independent,
    # so this is pure throughput: at 1 the server decodes at batch size 1, which
    # is memory-bandwidth-bound and leaves most of a CPU idle.
    #
    # It has to match the server's slot count. A server we start ourselves gets
    # --parallel from this value; for LLAMA_ENDPOINT, set it to the -np the
    # server was launched with. Going higher only queues requests inside the
    # server, where this side cannot see them.
    #
    # Mind LLAMA_CTX with it: unless llama-server was given --kv-unified, -c is
    # the total and each slot gets -c / -np. The detect stage checks that
    # arithmetic against LLM_CHUNK_WORDS and warns before wasting an episode.
    : "${LLM_CONCURRENCY:=1}"

    # Chunk boundaries ------------------------------------------------------
    # Only used to pick where a long track is split into requests. Nothing here
    # decides what is speech — Whisper's Silero pass does that — so these want to
    # be loose enough to find *a* quiet spot, not accurate enough to trust.
    #
    # The threshold errs low on purpose: too high and a split lands in quiet
    # speech, costing that word; too low and it just falls at the nominal
    # position instead. On a real recording here, speech ran from -21dB down to
    # -39dB over a floor near -50dB, so -45dB clears the quiet end of speech.
    : "${SPLIT_SILENCE_THRESHOLD:=-45dB}"
    : "${SPLIT_MIN_SILENCE:=0.30}"

    # Speech map ------------------------------------------------------------
    # Every transcribed word is widened by this much before the union that makes
    # the speech map, which decides two things: how much of Whisper's timing
    # error a cut may eat, and how long a gap between words has to be before it
    # is silence at all (SILENCE_MIN_DURATION plus twice this). Raising it makes
    # cuts rarer and safer; lowering it makes them tighter and more numerous.
    : "${SPEECH_PAD:=0.25}"

    # A gap where no track has speech becomes a candidate only past this length.
    : "${SILENCE_MIN_DURATION:=1.5}"
    # Residual silence left in place of a shortened gap.
    : "${SILENCE_KEEP:=0.40}"
    # Silence retained at the very start and end of the episode.
    : "${EDGE_KEEP:=0.25}"
    # Speech margin preserved on both sides of every cut.
    : "${CUT_PADDING:=0.10}"
    # Cuts shorter than this are not worth the splice.
    : "${MIN_CUT:=0.15}"
    # Fade applied at both ends of a per-track mute, to avoid clicks.
    : "${MUTE_FADE:=0.030}"

    # LLM edit detection ----------------------------------------------------
    : "${LLM_ENABLE:=1}"
    : "${LLM_CHUNK_WORDS:=350}"
    : "${LLM_CHUNK_OVERLAP:=40}"
    : "${LLM_MAX_EDIT_WORDS:=12}"
    : "${LLM_MAX_EDIT_SECONDS:=4.0}"
    : "${LLM_MIN_CONFIDENCE:=0.6}"
    : "${LLM_TEMP:=0}"
    # Accepted edit kinds. "filler" (um/uh/er — the non-lexical sounds, and
    # nothing else) is available but off by default: removing every filler tends
    # to over-edit natural speech. Crutches made of real words ("well", "you
    # know", "né") are detected by no kind; DESIGN.md §6 says why not.
    : "${LLM_ACCEPT_KINDS:=stutter,repetition,false_start}"

    # Safety ----------------------------------------------------------------
    # A plan that removes more than this fraction of the episode is refused
    # unless --force is given: it almost always means bad VAD settings.
    : "${MAX_CUT_FRACTION:=0.5}"

    # Rendering -------------------------------------------------------------
    # Output format, independent of what came in. FLAC by default: the tracks
    # are going on to be mixed, so losing nothing matters more than size.
    : "${OUTPUT_CODEC:=flac}"
    : "${OUTPUT_EXT:=flac}"
    : "${OUTPUT_COMPRESSION:=8}"          # FLAC only
    : "${OUTPUT_EXTRA_ARGS:=}"            # e.g. "-b:a 192k" for a lossy codec
    : "${OUTPUT_SUFFIX:=}"                # e.g. "_clean"

    # Cuts land on frame boundaries, so every track has to be frame-aligned at
    # the same rate or they drift apart. Empty means a rate mismatch is an
    # error; "auto" resamples everything to the highest rate present; a number
    # resamples everything to that. Resampling is a real transformation of the
    # audio, hence opt-in.
    : "${RESAMPLE_TO:=}"
    # Frame size used for cut and mute decisions. Cuts land on frame
    # boundaries, so this is the timing granularity of the edit — and it must
    # be the same for every track of an episode, which is what keeps them in
    # sync. Smaller is more precise and smoother-fading but slower to render.
    : "${RENDER_FRAME_SAMPLES:=512}"

    # Concurrency -----------------------------------------------------------
    # Whisper runs one track at a time by default: the model is large and must
    # not compete with itself for RAM/VRAM.
    : "${WHISPER_JOBS:=1}"
    : "${FFMPEG_JOBS:=$(( $(nproc 2>/dev/null || echo 2) / 2 ))}"
    (( FFMPEG_JOBS < 1 )) && FFMPEG_JOBS=1

    # Behaviour -------------------------------------------------------------
    : "${KEEP_WORK:=0}"
    : "${KEEP_INPUTS:=0}"
    # What a failure does with the originals.
    #   log   keep them where they are, copy the run log to FAILED_DIR
    #   move  additionally move them into FAILED_DIR/<episode>, so an
    #         unattended run cannot retry the same broken episode forever
    : "${FAILED_ACTION:=log}"
    # Rendered output must be within this fraction of the planned duration.
    : "${DURATION_TOLERANCE:=0.02}"
}

# --- loading -----------------------------------------------------------------

# config_load [explicit-file]
# Sources the first config file found. An explicit path that does not exist is
# an error; the implicit search path silently falls through to defaults.
config_load() {
    local explicit="${1:-}"
    local candidates=()

    if [[ -n "$explicit" ]]; then
        [[ -f "$explicit" ]] || die "config file not found: $explicit"
        candidates=("$explicit")
    else
        candidates=(
            "${PODCAST_CLEANUP_CONF:-}"
            "$PWD/podcast-cleanup.conf"
            "${XDG_CONFIG_HOME:-$HOME/.config}/podcast-cleanup/config"
            "/etc/podcast-cleanup.conf"
        )
    fi

    local f
    for f in "${candidates[@]}"; do
        [[ -n "$f" && -f "$f" ]] || continue
        # shellcheck disable=SC1090
        source "$f" || die "failed to parse config file: $f"
        CONFIG_FILE="$f"
        break
    done

    config_defaults
}

# --- filesystem layout -------------------------------------------------------
#
# One root with four subdirectories under it. Each of the four can still be
# pointed somewhere else on its own — a server with output on a different volume
# keeps working — but the ordinary case is one --root and nothing more.

# Absolute form of a path, whether or not it exists yet: the log and every error
# message should name a directory the reader can act on, and "output" alone is
# not one once the run has been started from somewhere else.
_abs_path() {
    local path="$1"
    if [[ -d "$path" ]]; then
        (cd -- "$path" && pwd)
    else
        [[ "$path" == /* ]] || path="$PWD/${path#./}"
        printf '%s\n' "$path"
    fi
}

# Called after the config file and the command line have both had their say, so
# that an explicit setting always wins and only what is left derives from the
# root.
config_resolve_paths() {
    [[ -n "$PODCAST_ROOT" ]] || PODCAST_ROOT="$LIB_ROOT"
    PODCAST_ROOT=$(_abs_path "$PODCAST_ROOT")

    : "${INPUT_DIR:=$PODCAST_ROOT/incoming}"
    : "${OUTPUT_DIR:=$PODCAST_ROOT/output}"
    : "${WORK_ROOT:=$PODCAST_ROOT/work}"
    : "${FAILED_DIR:=$PODCAST_ROOT/failed}"

    local v
    for v in INPUT_DIR OUTPUT_DIR WORK_ROOT FAILED_DIR; do
        printf -v "$v" '%s' "$(_abs_path "${!v}")"
    done
}

# config_make_tree — create the layout up front, so a fresh install is one run
# away from being usable and there is an obvious place to drop tracks. It also
# makes a missing input directory mean "empty" rather than "misconfigured".
config_make_tree() {
    [[ "$DRY_RUN" == 1 ]] && return 0
    local dir
    for dir in "$INPUT_DIR" "$OUTPUT_DIR" "$WORK_ROOT" "$FAILED_DIR"; do
        mkdir -p -- "$dir" || die "cannot create directory: $dir"
    done
    return 0
}

# --- api keys ----------------------------------------------------------------
#
# Keys are resolved once, held only in shell variables, and handed to Python
# through the environment — never as a command-line argument, since argv is
# readable by any process on the machine, and never through the run log, which
# outlives the episode.

_read_key_file() {
    local label="$1" path="$2"
    [[ -f "$path" ]] || die "$label file not found: $path"
    [[ -r "$path" ]] || die "$label file is not readable: $path"

    # A secret readable by anyone on the box is not much of a secret.
    local mode
    if mode=$(stat -c '%a' "$path" 2>/dev/null) && [[ "$mode" =~ ^[0-7]+$ ]]; then
        local group=$(( (10#$mode / 10) % 10 )) other=$(( 10#$mode % 10 ))
        if (( (group & 4) || (other & 4) )); then
            log_warn "$label file $path is readable beyond its owner (mode $mode); chmod 600 it"
        fi
    fi

    local value
    IFS= read -r value <"$path" || true
    # Trim surrounding whitespace, which a stray newline or editor would add.
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    [[ -n "$value" ]] || die "$label file $path is empty"
    printf '%s' "$value"
}

config_resolve_api_keys() {
    if [[ -n "$WHISPER_API_KEY_FILE" ]]; then
        WHISPER_API_KEY=$(_read_key_file WHISPER_API_KEY "$WHISPER_API_KEY_FILE") || exit 1
    fi
    if [[ -n "$LLAMA_API_KEY_FILE" ]]; then
        LLAMA_API_KEY=$(_read_key_file LLAMA_API_KEY "$LLAMA_API_KEY_FILE") || exit 1
    fi
    return 0
}

# --- validation --------------------------------------------------------------

_is_number() { [[ "$1" =~ ^-?[0-9]+([.][0-9]+)?$ ]]; }

_require_number() {
    _is_number "${!1}" || die "config $1 must be a number, got '${!1}'"
}

_require_int() {
    [[ "${!1}" =~ ^[0-9]+$ ]] || die "config $1 must be a non-negative integer, got '${!1}'"
}

config_validate() {
    local v
    for v in SILENCE_MIN_DURATION SILENCE_KEEP EDGE_KEEP CUT_PADDING MIN_CUT \
             MUTE_FADE LLM_MAX_EDIT_SECONDS LLM_MIN_CONFIDENCE DURATION_TOLERANCE \
             MAX_CUT_FRACTION SPLIT_MIN_SILENCE SPEECH_PAD \
             WHISPER_VAD_THRESHOLD WHISPER_VAD_SAMPLES_OVERLAP \
             WHISPER_CHUNK_SECONDS WHISPER_REQUEST_TIMEOUT; do
        _require_number "$v"
    done
    for v in LLM_CHUNK_WORDS LLM_CHUNK_OVERLAP LLM_MAX_EDIT_WORDS WHISPER_JOBS \
             FFMPEG_JOBS LLAMA_PORT LLAMA_CTX OUTPUT_COMPRESSION \
             LLM_CONCURRENCY RENDER_FRAME_SAMPLES WHISPER_VAD_MIN_SPEECH_MS \
             WHISPER_VAD_MIN_SILENCE_MS WHISPER_VAD_SPEECH_PAD_MS; do
        _require_int "$v"
    done

    (( LLM_CONCURRENCY >= 1 )) \
        || die "LLM_CONCURRENCY must be at least 1, got $LLM_CONCURRENCY"

    (( RENDER_FRAME_SAMPLES >= 64 && RENDER_FRAME_SAMPLES <= 8192 )) \
        || die "RENDER_FRAME_SAMPLES must be between 64 and 8192, got $RENDER_FRAME_SAMPLES"

    case "$FAILED_ACTION" in
        log|move) ;;
        *) die "FAILED_ACTION must be 'log' or 'move', got '$FAILED_ACTION'" ;;
    esac

    case "$LLM_API" in
        chat|completion) ;;
        *) die "LLM_API must be 'chat' or 'completion', got '$LLM_API'" ;;
    esac

    case "$RESAMPLE_TO" in
        ""|auto) ;;
        *[!0-9]*) die "RESAMPLE_TO must be empty, 'auto', or a sample rate, got '$RESAMPLE_TO'" ;;
        *) (( RESAMPLE_TO >= 8000 )) || die "RESAMPLE_TO looks too low: $RESAMPLE_TO" ;;
    esac

    # The old single setting meant input filter and output format at once.
    if [[ -n "$TRACK_EXT" ]]; then
        INPUT_EXTS="$TRACK_EXT"
        log_warn "TRACK_EXT is deprecated: it now only restricts which inputs are found. Use INPUT_EXTS for that and OUTPUT_CODEC/OUTPUT_EXT to choose the output format (currently $OUTPUT_CODEC/.$OUTPUT_EXT)"
    fi
    [[ -n "${INPUT_EXTS// /}" ]] || die "INPUT_EXTS is empty; nothing could ever be found"
    [[ "$OUTPUT_EXT" == *[!a-zA-Z0-9]* ]] \
        && die "OUTPUT_EXT should be a bare extension, got '$OUTPUT_EXT'"

    (( LLM_CHUNK_OVERLAP < LLM_CHUNK_WORDS )) \
        || die "LLM_CHUNK_OVERLAP ($LLM_CHUNK_OVERLAP) must be smaller than LLM_CHUNK_WORDS ($LLM_CHUNK_WORDS)"

    (( WHISPER_JOBS > 1 )) && log_warn \
        "WHISPER_JOBS=$WHISPER_JOBS runs several Whisper instances at once; make sure the RAM is there"

    awk -v k="$SILENCE_KEEP" -v m="$SILENCE_MIN_DURATION" 'BEGIN{exit !(k < m)}' \
        || die "SILENCE_KEEP ($SILENCE_KEEP) must be smaller than SILENCE_MIN_DURATION ($SILENCE_MIN_DURATION)"

    [[ "$TRACK_SEPARATOR" == ?* ]] || die "TRACK_SEPARATOR must not be empty"
    return 0
}

# --- lazy tool resolution ----------------------------------------------------
#
# Each of these is called by the stage that actually needs the tool, so the
# absence of (say) llama-server never blocks a silence-only run.

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
    if [[ -n "$WHISPER_ENDPOINT" ]]; then
        log_debug "whisper: using remote endpoint $WHISPER_ENDPOINT"
        return 0
    fi
    [[ -n "${WHISPER:-}" ]] && return 0
    # A dry run is a preview, so a missing model is reported and stepped over
    # rather than being the end of it. ffmpeg and python3 stay hard
    # requirements — nothing can be previewed without them.
    if ! WHISPER=$(require_bin whisper-cli "$WHISPER_BIN"); then
        [[ "$DRY_RUN" == 1 ]] || exit 1
        WHISPER="$WHISPER_BIN"
        log_warn "dry run: carrying on without whisper-cli"
    fi
    if [[ ! -f "$WHISPER_MODEL" ]]; then
        [[ "$DRY_RUN" == 1 ]] || die "Whisper model not found: $WHISPER_MODEL"
        log_warn "dry run: no Whisper model at $WHISPER_MODEL"
    fi
    # The Silero model is whisper-cli's own argument, so it has to be here for a
    # local run. A remote server was given one at launch instead, and nothing on
    # this side can check that from here.
    if [[ "$WHISPER_VAD" == 1 && ! -f "$WHISPER_VAD_MODEL" ]]; then
        [[ "$DRY_RUN" == 1 ]] \
            && log_warn "dry run: no Silero model at $WHISPER_VAD_MODEL" \
            || die "WHISPER_VAD=1 needs the Silero model at WHISPER_VAD_MODEL ($WHISPER_VAD_MODEL). Fetch one with whisper.cpp's models/download-vad-model.sh silero-v6.2.0 — or set WHISPER_VAD=0, and accept that silence gets transcribed and whatever Whisper invents there becomes speech in the plan"
    fi
    log_debug "whisper: $WHISPER ($(basename "$WHISPER_MODEL"))"
}

# Which halves of the run this script is responsible for keeping out of memory's
# way. Anything remote is the other machine's problem, and worth saying so.
config_describe_models() {
    local whisper_where llama_where
    [[ -n "$WHISPER_ENDPOINT" ]] && whisper_where="remote" || whisper_where="local"
    [[ -n "$LLAMA_ENDPOINT" ]] && llama_where="remote" || llama_where="local"
    log_info "Whisper: $whisper_where${WHISPER_ENDPOINT:+ ($WHISPER_ENDPOINT)}   LLM: $llama_where${LLAMA_ENDPOINT:+ ($LLAMA_ENDPOINT)}"
    if [[ "$whisper_where" == local && "$llama_where" == local ]]; then
        log_debug "both models are local, so this run serialises them itself"
    elif [[ "$whisper_where" == remote && "$llama_where" == remote ]]; then
        log_warn "both models are remote: if they share a machine, keeping them from overlapping in memory is that machine's business, not this script's"
    fi
}

config_need_llama() {
    if [[ -n "$LLAMA_ENDPOINT" ]]; then
        log_debug "llama: using external endpoint $LLAMA_ENDPOINT"
        return 0
    fi
    [[ -n "${LLAMA_SERVER:-}" ]] && return 0
    if ! LLAMA_SERVER=$(require_bin llama-server "$LLAMA_SERVER_BIN"); then
        [[ "$DRY_RUN" == 1 ]] || exit 1
        LLAMA_SERVER="$LLAMA_SERVER_BIN"
        log_warn "dry run: carrying on without llama-server"
    fi
    if [[ ! -f "$LLAMA_MODEL" ]]; then
        [[ "$DRY_RUN" == 1 ]] || die "llama model not found: $LLAMA_MODEL"
        log_warn "dry run: no llama model at $LLAMA_MODEL"
    fi
    log_debug "llama: $LLAMA_SERVER ($(basename "$LLAMA_MODEL"))"
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
             WHISPER_BIN WHISPER_MODEL WHISPER_THREADS WHISPER_LANG WHISPER_JOBS \
             WHISPER_VAD WHISPER_VAD_MODEL WHISPER_VAD_THRESHOLD \
             WHISPER_VAD_MIN_SPEECH_MS WHISPER_VAD_MIN_SILENCE_MS \
             WHISPER_VAD_SPEECH_PAD_MS WHISPER_VAD_SAMPLES_OVERLAP \
             LLAMA_ENDPOINT LLAMA_SERVER_BIN LLAMA_MODEL LLAMA_HOST LLAMA_PORT \
             LLAMA_CTX LLAMA_NGL LLAMA_MODEL_NAME LLM_API LLM_MAX_REPLY_TOKENS \
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
