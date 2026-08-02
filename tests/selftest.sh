#!/usr/bin/env bash
#
# Offline end-to-end check.
#
# Builds a synthetic two-track "episode" with silences in known places, runs the
# real pipeline over it in a sandbox, and checks the results. It needs nothing
# but ffmpeg and python3 — the models are stubbed over HTTP — and it exercises
# the parts that actually touch audio, which are the parts that are awkward to
# reason about on paper.
#
# Transcription is stubbed rather than skipped, because there is nothing left to
# skip it to: the pipeline no longer detects speech from the audio. Whisper's own
# Silero pass decides what gets transcribed and the speech map is derived from
# the words that come back, so a synthetic episode needs a synthetic transcript,
# and each case generates one that agrees with the tone in its audio.
#
# The synthetic episode is 30 seconds long:
#
#   track a speaks   0-4      10-14        20-24
#   track b speaks        5-9        15-19        25-29
#   nobody speaks       4-5   9-10  14-15  19-20  24-25  29-30
#
# Every internal gap is 1 s, below the 1.5 s threshold, so the only cut should
# be the 1 s tail — which makes the arithmetic easy to check by hand. A second
# pass then widens one gap to prove that shortening works too. Those cases pin
# SPEECH_PAD to 0 so the arithmetic stays the transcript's own; case 11 is where
# the padding itself is checked.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SANDBOX="$(mktemp -d -t podcast-selftest-XXXXXX)"
FAILURES=0
CHECKS=0

cleanup() { [[ "${KEEP_SANDBOX:-0}" == 1 ]] || rm -rf "$SANDBOX"; }
trap cleanup EXIT

if [[ -t 1 ]]; then
    GREEN=$'\033[32m'; RED=$'\033[31m'; DIM=$'\033[2m'; BOLD=$'\033[1m'; RESET=$'\033[0m'
else
    GREEN=''; RED=''; DIM=''; BOLD=''; RESET=''
fi

check() {
    local description="$1"; shift
    CHECKS=$(( CHECKS + 1 ))
    if "$@"; then
        printf '  %s✓%s %s\n' "$GREEN" "$RESET" "$description"
    else
        printf '  %s✗%s %s\n' "$RED" "$RESET" "$description"
        FAILURES=$(( FAILURES + 1 ))
    fi
}

fail_note() {
    printf '      %s%s%s\n' "$DIM" "$1" "$RESET"
}

# --- numeric helpers ----------------------------------------------------------

approx() {  # approx <actual> <expected> <tolerance>
    awk -v a="$1" -v b="$2" -v t="$3" 'BEGIN{d=a-b; if(d<0)d=-d; exit !(d<=t)}'
}

duration_of() {
    ffprobe -v error -show_entries format=duration -of csv=p=0 "$1"
}

# peak_db <file> <start> <end> — peak level in dB over one window. Digital
# silence reports as -inf or -91; both are normalised to a very low number.
peak_db() {
    local out
    out=$(ffmpeg -v info -nostdin -i "$1" \
        -af "atrim=start=$2:end=$3,volumedetect" -f null - 2>&1 |
        sed -n 's/.*max_volume: \(-\?[0-9.a-z]*\) dB.*/\1/p' | tail -1)
    [[ -z "$out" || "$out" == *inf* ]] && out="-999"
    printf '%s' "$out"
}

quieter_than() { awk -v v="$1" -v t="$2" 'BEGIN{exit !(v < t)}'; }
louder_than()  { awk -v v="$1" -v t="$2" 'BEGIN{exit !(v > t)}'; }

json_number() {  # json_number <file> <dotted.path>
    python3 -c '
import json, sys
data = json.load(open(sys.argv[1]))
for key in sys.argv[2].split("."):
    data = data[key] if not key.isdigit() else data[int(key)]
print(data)
' "$1" "$2"
}

# --- synthetic audio ----------------------------------------------------------

has_encoder() {
    ffmpeg -hide_banner -h "encoder=$1" 2>/dev/null | grep -q "Encoder $1"
}

# make_track_as <output> <codec> <extra-args> <sample-rate> <duration> <windows>
# Same synthetic signal, written through an arbitrary encoder.
make_track_as() {
    local output="$1" codec="$2" extra="$3" rate="$4" duration="$5" windows="$6"
    # shellcheck disable=SC2086
    ffmpeg -v error -y -f lavfi \
        -i "aevalsrc=exprs='0.4*sin(2*PI*440*t)*(${windows})':s=${rate}:d=${duration}" \
        -c:a "$codec" $extra "$output"
}

# make_track <output> <sample-rate> <duration> <expr-of-speech-windows>
make_track() {
    local output="$1" rate="$2" duration="$3" windows="$4"
    # A 440 Hz tone inside the speech windows, digital silence outside. Level is
    # well above the -35 dB detection threshold; silence is exactly zero.
    # The expression is quoted for ffmpeg's own parser, or its commas would read
    # as option separators.
    ffmpeg -v error -y -f lavfi \
        -i "aevalsrc=exprs='0.4*sin(2*PI*440*t)*(${windows})':s=${rate}:d=${duration}" \
        -c:a flac "$output"
}

build_episode() {
    local episode="$1" incoming="$2" a_windows="$3" b_windows="$4" duration="$5"
    mkdir -p "$incoming"
    make_track "$incoming/${episode}_alice.flac" 48000 "$duration" "$a_windows"
    make_track "$incoming/${episode}_bob.flac" 48000 "$duration" "$b_windows"
}

# --- stub whisper -------------------------------------------------------------
#
# Speech is no longer detected from the audio: Whisper's own Silero pass decides
# what gets transcribed, and the pipeline derives its speech map from the words
# that come back. So a synthetic episode needs a synthetic transcript, and the
# two have to agree — the tone is in the audio for the render checks to measure,
# and the words say where that tone is.
#
# Windows are written once, as "start,end start,end", and both the audio
# expression and the canned transcript are generated from them.

# windows_expr "0,4 10,14" -> between(t,0,4)+between(t,10,14)
windows_expr() {
    local expr="" window
    for window in $1; do
        [[ -n "$expr" ]] && expr+="+"
        expr+="between(t,${window%,*},${window#*,})"
    done
    printf '%s' "$expr"
}

# whisper_responses <target.json> <name=windows> ...
# One reply per track, keyed by the prepared track's filename, each speech window
# becoming a run of one-word segments. One word per segment is what max_len=1
# gives on a real server, and it makes every word timing exact rather than
# interpolated across a sentence.
whisper_responses() {
    local target="$1"; shift
    python3 - "$target" "$@" <<'PY'
import json, sys

target, tracks = sys.argv[1], sys.argv[2:]
WORD = 0.5
replies = {}
for spec in tracks:
    name, _, windows = spec.partition("=")
    segments = []
    for window in windows.split():
        start, end = (float(value) for value in window.split(","))
        moment = start
        while moment < end - 1e-9:
            stop = min(moment + WORD, end)
            segments.append({
                "text": f"w{len(segments)}",
                "offsets": {
                    "from": int(round(moment * 1000)),
                    "to": int(round(stop * 1000)),
                },
            })
            moment = stop
    replies[f"{name}.wav"] = {"transcription": segments}
with open(target, "w", encoding="utf-8") as handle:
    json.dump(replies, handle)
PY
}

WHISPER_STUB_PID=""
WHISPER_STUB_ENDPOINT=""

start_whisper_stub() {
    local responses="$1"
    stop_whisper_stub
    local port_file="$SANDBOX/whisper-port-$$-$CHECKS"
    rm -f "$port_file"
    python3 "$ROOT/tests/stub_servers.py" whisper \
        --responses "$responses" --port-file "$port_file" \
        ${2:+--request-log "$2"} >/dev/null 2>&1 &
    WHISPER_STUB_PID=$!
    local waited=0
    while [[ ! -f "$port_file" ]]; do
        sleep 0.05
        waited=$(( waited + 1 ))
        if (( waited > 200 )); then
            printf 'stub whisper server did not start\n' >&2
            return 1
        fi
    done
    WHISPER_STUB_ENDPOINT="http://127.0.0.1:$(cat "$port_file")"
}

stop_whisper_stub() {
    if [[ -n "$WHISPER_STUB_PID" ]]; then
        kill "$WHISPER_STUB_PID" 2>/dev/null || true
        wait "$WHISPER_STUB_PID" 2>/dev/null || true
        WHISPER_STUB_PID=""
    fi
}

trap 'stop_whisper_stub; cleanup' EXIT

run_pipeline() {
    local incoming="$1" output="$2" work="$3"; shift 3
    # llama.cpp is out of scope here, so the detect stage is left out of the list
    # entirely rather than stubbed. Transcription cannot be: the speech map comes
    # from the transcript now, so the stub whisper server is part of the pipeline
    # under test rather than a convenience.
    # --root keeps everything this run might write inside the sandbox, including
    # the directories the three explicit options do not name — FAILED_DIR, which
    # the cases that fail on purpose write their logs to.
    "$ROOT/clean-podcast.sh" --root "$SANDBOX" \
        --input "$incoming" --output "$output" --work "$work" \
        --whisper-endpoint "$WHISPER_STUB_ENDPOINT" \
        --stages discover,prepare,transcribe,plan,render,finalize \
        --no-llm --quiet "$@"
}

printf '\n%sPodcast cleanup self-test%s %s(%s)%s\n\n' \
    "$BOLD" "$RESET" "$DIM" "$SANDBOX" "$RESET"

# ============================================================================
printf '%sCase 1: gaps below the threshold are left alone%s\n' "$BOLD" "$RESET"
# ============================================================================

CASE1="$SANDBOX/case1"
A_SPEECH='0,4 10,14 20,24'
B_SPEECH='5,9 15,19 25,29'
A_WINDOWS="$(windows_expr "$A_SPEECH")"
B_WINDOWS="$(windows_expr "$B_SPEECH")"
build_episode ep001 "$CASE1/incoming" "$A_WINDOWS" "$B_WINDOWS" 30
whisper_responses "$SANDBOX/case1.words.json" \
    "alice=$A_SPEECH" "bob=$B_SPEECH"
start_whisper_stub "$SANDBOX/case1.words.json" "$SANDBOX/case1.requests.jsonl"

CONF="$SANDBOX/case1.conf"
cat >"$CONF" <<'EOF'
SILENCE_MIN_DURATION="1.5"
SILENCE_KEEP="0.40"
EDGE_KEEP="0.25"
CUT_PADDING="0.10"
MIN_CUT="0.15"
RENDER_FRAME_SAMPLES="512"
# Zero, so the gap arithmetic below is the transcript's own and can be checked by
# hand. Case 11 covers what the padding does.
SPEECH_PAD="0"
EOF

if run_pipeline "$CASE1/incoming" "$CASE1/output" "$CASE1/work" \
    --config "$CONF" --keep-work >"$SANDBOX/case1.stdout" 2>&1
then
    check "pipeline completed" true
else
    check "pipeline completed" false
    fail_note "$(tail -n 25 "$SANDBOX/case1.stdout")"
fi

OUT1="$CASE1/output/ep001"
PLAN1="$CASE1/work/ep001/plan.json"

check "alice track published" test -s "$OUT1/alice.flac"
check "bob track published"   test -s "$OUT1/bob.flac"
check "transcript json written" test -s "$OUT1/ep001_transcript.json"
check "edit report written"   test -s "$OUT1/ep001_edit-report.txt"
check "run log kept"          test -s "$OUT1/logs/run.log"
check "plan.json published"   test -s "$OUT1/ep001_plan.json"

if [[ -f "$PLAN1" ]]; then
    CUT_COUNT=$(json_number "$PLAN1" stats.cut_count)
    REMOVED=$(json_number "$PLAN1" stats.removed)
    # Only the 1 s tail qualifies: 1 - 0.25 of retained edge = 0.75 s.
    check "exactly one cut planned (the tail)" test "$CUT_COUNT" = "1"
    check "removed ≈ 0.75 s" approx "$REMOVED" 0.75 0.06
    fail_note "cuts=$CUT_COUNT removed=${REMOVED}s"

    # A cut that lands inside anybody's speech would be a serious bug. Checked
    # against the words themselves rather than the plan's own map, so a mistake
    # in deriving that map cannot hide here.
    check "no cut overlaps a transcribed word" python3 -c '
import json, sys
plan = json.load(open(sys.argv[1]))
words = []
for path in sys.argv[2:]:
    words += [(w["start"], w["end"], w["text"])
              for w in json.load(open(path))["words"]]
for cut in plan["cuts"]:
    for start, end, text in words:
        overlap = min(cut["end"], end) - max(cut["start"], start)
        if overlap > 0.02:
            print(f"cut {cut} overlaps word {text} {(start, end)} by {overlap:.3f}s")
            sys.exit(1)
' "$PLAN1" "$CASE1/work/ep001/words/alice.words.json" \
  "$CASE1/work/ep001/words/bob.words.json"

    check "server-side VAD was actually requested" python3 -c '
import json, sys
for line in open(sys.argv[1], encoding="utf-8"):
    record = json.loads(line)
    values = record.get("values") or {}
    if values.get("vad") != "true":
        print("request", record["index"], "did not ask for VAD:", values)
        sys.exit(1)
    for field in ("vad_threshold", "vad_min_silence_duration_ms",
                  "vad_speech_pad_ms"):
        if field not in values:
            print("request", record["index"], "left out", field, ":", values)
            sys.exit(1)
' "$SANDBOX/case1.requests.jsonl"
fi

if [[ -s "$OUT1/alice.flac" && -s "$OUT1/bob.flac" ]]; then
    DUR_A=$(duration_of "$OUT1/alice.flac")
    DUR_B=$(duration_of "$OUT1/bob.flac")
    EXPECT_A=$(json_number "$CASE1/work/ep001/expected.json" tracks.alice.expected_duration)
    check "alice matches the predicted length" approx "$DUR_A" "$EXPECT_A" 0.05
    # The property everything else depends on: the tracks stay the same length.
    check "both tracks are identically long" approx "$DUR_A" "$DUR_B" 0.0005
    fail_note "alice=${DUR_A}s bob=${DUR_B}s predicted=${EXPECT_A}s"
fi

check "inputs deleted after success" bash -c \
    '[[ -z "$(find "$1" -name "*.flac" -print -quit)" ]]' _ "$CASE1/incoming"

# ============================================================================
printf '\n%sCase 2: a long silence is shortened, not removed%s\n' "$BOLD" "$RESET"
# ============================================================================

CASE2="$SANDBOX/case2"
# Nobody speaks between 8 s and 18 s: a 10 s gap that must shrink to 0.4 s.
A2_SPEECH='0,4 18,22'
B2_SPEECH='4,8 22,26'
build_episode ep002 "$CASE2/incoming" \
    "$(windows_expr "$A2_SPEECH")" "$(windows_expr "$B2_SPEECH")" 30
whisper_responses "$SANDBOX/case2.words.json" \
    "alice=$A2_SPEECH" "bob=$B2_SPEECH"
start_whisper_stub "$SANDBOX/case2.words.json"

if run_pipeline "$CASE2/incoming" "$CASE2/output" "$CASE2/work" \
    --config "$CONF" --keep-work >"$SANDBOX/case2.stdout" 2>&1
then
    check "pipeline completed" true
else
    check "pipeline completed" false
    fail_note "$(tail -n 25 "$SANDBOX/case2.stdout")"
fi

PLAN2="$CASE2/work/ep002/plan.json"
if [[ -f "$PLAN2" ]]; then
    check "the long gap survives as residual silence" python3 -c '
import json, sys
plan = json.load(open(sys.argv[1]))
inner = [c for c in plan["cuts"] if "silence" in c["reasons"]]
if len(inner) != 1:
    print(f"expected 1 internal cut, got {len(inner)}: {inner}")
    sys.exit(1)
cut = inner[0]
length = cut["end"] - cut["start"]
# A 10 s gap keeping 0.4 s leaves a 9.6 s cut, give or take word boundaries.
if not 9.0 <= length <= 9.8:
    print(f"internal cut is {length:.3f}s, expected about 9.6s")
    sys.exit(1)
print(f"internal cut {length:.3f}s out of a 10s gap")
' "$PLAN2"

    OUT2="$CASE2/output/ep002"
    DUR2_A=$(duration_of "$OUT2/alice.flac")
    DUR2_B=$(duration_of "$OUT2/bob.flac")
    check "both tracks still identically long" approx "$DUR2_A" "$DUR2_B" 0.0005
    # 30 s, minus ~9.6 s of gap, minus the head (none) and the ~3.75 s tail.
    check "output is between 16 s and 21 s" \
        awk -v d="$DUR2_A" 'BEGIN{exit !(d>16 && d<21)}'
    fail_note "output length ${DUR2_A}s (was 30s)"
fi

# ============================================================================
printf '\n%sCase 3: safety rails%s\n' "$BOLD" "$RESET"
# ============================================================================

CASE3="$SANDBOX/case3"
# Two brief bursts in 60 s: cutting this to the threshold would remove most of
# the episode, which must be refused rather than silently accepted.
build_episode ep003 "$CASE3/incoming" \
    "$(windows_expr '1,2')" "$(windows_expr '58,59')" 60
whisper_responses "$SANDBOX/case3.words.json" "alice=1,2" "bob=58,59"
start_whisper_stub "$SANDBOX/case3.words.json"

if run_pipeline "$CASE3/incoming" "$CASE3/output" "$CASE3/work" \
    --config "$CONF" --keep-work >"$SANDBOX/case3.stdout" 2>&1
then
    check "an over-aggressive plan is refused" false
    fail_note "the run should have failed but did not"
else
    check "an over-aggressive plan is refused" true
fi
check "refusal mentions the safety limit" \
    grep -qi "safety limit" "$SANDBOX/case3.stdout"
check "inputs preserved after a failure" bash -c \
    '[[ -n "$(find "$1" -name "*.flac" -print -quit)" ]]' _ "$CASE3/incoming"
check "work directory preserved after a failure" test -d "$CASE3/work/ep003"

if run_pipeline "$CASE3/incoming" "$CASE3/output" "$CASE3/work" \
    --config "$CONF" --keep-work --force >"$SANDBOX/case3-force.stdout" 2>&1
then
    check "--force overrides the refusal" true
else
    check "--force overrides the refusal" false
    fail_note "$(tail -n 25 "$SANDBOX/case3-force.stdout")"
fi

# ============================================================================
printf '\n%sCase 4: dry run touches nothing%s\n' "$BOLD" "$RESET"
# ============================================================================

CASE4="$SANDBOX/case4"
build_episode ep004 "$CASE4/incoming" "$A_WINDOWS" "$B_WINDOWS" 30
# A dry run sends nothing, so the case 1 replies serve; the stub is restarted
# only so a failure here cannot be blamed on case 3's leftovers.
start_whisper_stub "$SANDBOX/case1.words.json"

if run_pipeline "$CASE4/incoming" "$CASE4/output" "$CASE4/work" \
    --config "$CONF" --dry-run >"$SANDBOX/case4.stdout" 2>&1
then
    check "dry run completed" true
else
    check "dry run completed" false
    fail_note "$(tail -n 25 "$SANDBOX/case4.stdout")"
fi
check "dry run left the inputs alone" bash -c \
    '[[ -n "$(find "$1" -name "*.flac" -print -quit)" ]]' _ "$CASE4/incoming"
check "dry run published nothing" bash -c \
    '[[ -z "$(find "$1" -name "*.flac" 2>/dev/null -print -quit)" ]]' _ "$CASE4/output"

# ============================================================================
printf '\n%sCase 5: mismatched sample rates are rejected%s\n' "$BOLD" "$RESET"
# ============================================================================

CASE5="$SANDBOX/case5"
mkdir -p "$CASE5/incoming"
make_track "$CASE5/incoming/ep005_alice.flac" 48000 10 "$(windows_expr '0,5')"
make_track "$CASE5/incoming/ep005_bob.flac"   44100 10 "$(windows_expr '5,9')"

if run_pipeline "$CASE5/incoming" "$CASE5/output" "$CASE5/work" \
    --config "$CONF" >"$SANDBOX/case5.stdout" 2>&1
then
    check "mixed sample rates rejected" false
    fail_note "the run should have failed but did not"
else
    check "mixed sample rates rejected" true
fi
check "the error explains why" \
    grep -qi "share a sample rate" "$SANDBOX/case5.stdout"

# ============================================================================
printf '\n%sCase 6: a stutter over crosstalk is muted, not cut%s\n' "$BOLD" "$RESET"
# ============================================================================
#
# This is the case the whole design turns on. Alice stutters at 5.0-5.5 s while
# Bob is talking (4-6 s). Cutting there would take a bite out of Bob, so instead
# Alice's track alone is silenced and the timeline is left exactly as it was.
#
# The LLM is not available here, and the transcript is written by hand rather
# than through the stub, because these exact word timings are the point. Note
# that the words now carry two jobs: they are what the LLM edit refers to, and
# they are the speech map — so they have to cover the audio the tracks actually
# contain, or the plan would read the untranscribed parts as silence and cut
# them.

CASE6="$SANDBOX/case6"
# Both speak throughout their windows, so there is no silence anywhere to cut.
build_episode ep006 "$CASE6/incoming" \
    "$(windows_expr '0,20')" "$(windows_expr '4,6')" 20

run_pipeline "$CASE6/incoming" "$CASE6/output" "$CASE6/work" \
    --config "$CONF" --keep-work --stages discover,prepare \
    >"$SANDBOX/case6a.stdout" 2>&1 \
    || { check "case 6 setup" false; fail_note "$(tail -n 20 "$SANDBOX/case6a.stdout")"; }

WORK6="$CASE6/work/ep006"
python3 - "$WORK6" <<'INJECT'
import json, os, sys

work = sys.argv[1]

# Alice talks across the whole 20 s — every 0.5 s, so her speech map covers the
# tone — except at 5.0, where "eu eu" is the accidental repetition the LLM found.
words = []
moment = 0.0
stutter = []
# The 4.5 and 5.5 slots are left empty so the stutter has ordinary room around
# it: a neighbouring word 0.1 s away would cap the cut padding at half that gap
# and the mute would come out narrower than CUT_PADDING asked for. Both gaps stay
# under SILENCE_MIN_DURATION, so neither becomes a silence cut.
while moment < 20.0 - 1e-9:
    if abs(moment - 5.0) < 1e-9:
        for half in (0.0, 0.25):
            stutter.append(len(words))
            words.append({"i": len(words), "text": "eu",
                          "start": 5.0 + half, "end": 5.25 + half, "segment": 0})
    elif abs(moment - 4.5) < 1e-9 or abs(moment - 5.5) < 1e-9:
        pass
    else:
        words.append({"i": len(words), "text": f"w{len(words)}",
                      "start": moment, "end": moment + 0.4, "segment": 0})
    moment += 0.5

# Bob speaks 4-6, which is what makes alice's 5.0-5.5 stutter crosstalk.
bob = [
    {"i": 0, "text": "sim",   "start": 4.0, "end": 4.9, "segment": 0},
    {"i": 1, "text": "claro", "start": 5.0, "end": 6.0, "segment": 0},
]

for name, payload in (("alice", words), ("bob", bob)):
    json.dump(
        {
            "participant": name,
            "language": "pt",
            "words": payload,
            "segments": [{
                "i": 0, "start": payload[0]["start"], "end": payload[-1]["end"],
                "text": " ".join(w["text"] for w in payload),
                "first_word": 0, "last_word": len(payload) - 1,
            }],
            "approximated_segments": 0,
        },
        open(os.path.join(work, "words", f"{name}.words.json"), "w"),
    )

json.dump(
    {
        "participant": "alice",
        "chunks": 1,
        "chunk_failures": 0,
        "rejected_count": 0,
        "edits": [{
            "first": stutter[0], "last": stutter[1], "kind": "repetition",
            "confidence": 0.95, "start": 5.0, "end": 5.5, "text": "eu eu",
        }],
    },
    open(os.path.join(work, "llm", "alice.edits.json"), "w"),
)
INJECT

# Deliberately without --no-llm: the injected edits must be picked up.
if "$ROOT/clean-podcast.sh" --root "$SANDBOX" \
    --input "$CASE6/incoming" --output "$CASE6/output" \
    --work "$CASE6/work" --config "$CONF" --episode ep006 --keep-work --quiet \
    --stages plan,render,finalize >"$SANDBOX/case6b.stdout" 2>&1
then
    check "pipeline completed" true
else
    check "pipeline completed" false
    fail_note "$(tail -n 25 "$SANDBOX/case6b.stdout")"
fi

PLAN6="$WORK6/plan.json"
OUT6="$CASE6/output/ep006"

if [[ -f "$PLAN6" ]]; then
    check "the edit became a mute, not a cut" python3 -c '
import json, sys
plan = json.load(open(sys.argv[1]))
cuts, mutes = plan["cuts"], plan["mutes"]
if cuts:
    print("expected no cuts at all, got:", cuts)
    sys.exit(1)
if len(mutes["alice"]) != 1:
    print("expected exactly 1 mute on alice, got:", mutes["alice"])
    sys.exit(1)
if mutes["bob"]:
    print("bob should not be muted at all, got:", mutes["bob"])
    sys.exit(1)
mutes = mutes["alice"]
start, end = mutes[0]["start"], mutes[0]["end"]
# 5.0-5.5 widened by 0.1 s of padding on each side, the neighbouring words
# being far enough away to allow the full amount.
if abs(start - 4.9) > 0.001 or abs(end - 5.6) > 0.001:
    print(f"mute is {start}-{end}, expected 4.9-5.6")
    sys.exit(1)
' "$PLAN6"
fi

if [[ -s "$OUT6/alice.flac" && -s "$OUT6/bob.flac" ]]; then
    D6A=$(duration_of "$OUT6/alice.flac")
    D6B=$(duration_of "$OUT6/bob.flac")
    check "alice keeps her original length" approx "$D6A" 20 0.05
    check "bob keeps his original length"   approx "$D6B" 20 0.05
    fail_note "alice=${D6A}s bob=${D6B}s (input was 20s)"

    ALICE_MUTED=$(peak_db "$OUT6/alice.flac" 4.95 5.55)
    ALICE_BEFORE=$(peak_db "$OUT6/alice.flac" 1.0 3.0)
    BOB_SAME_SPOT=$(peak_db "$OUT6/bob.flac" 5.0 5.5)

    check "alice is silent across the muted span" quieter_than "$ALICE_MUTED" -60
    check "alice is untouched either side of it"  louder_than  "$ALICE_BEFORE" -20
    # The decisive check: Bob's audio at that instant survives intact.
    check "bob is unaffected at the same instant" louder_than "$BOB_SAME_SPOT" -20
    fail_note "alice muted=${ALICE_MUTED}dB elsewhere=${ALICE_BEFORE}dB bob=${BOB_SAME_SPOT}dB"

    check "the muted words left the transcript" python3 -c '
import json, sys
data = json.load(open(sys.argv[1]))
alice = " ".join(s["text"] for s in data["segments"] if s["participant"] == "alice")
if "eu" in alice.split():
    print(f"muted words are still in the transcript: {alice!r}")
    sys.exit(1)
if "w0" not in alice.split():
    print(f"surviving words went missing: {alice!r}")
    sys.exit(1)
' "$OUT6/ep006_transcript.json"
fi

# ============================================================================
printf '\n%sCase 7: the full pipeline against remote Whisper and remote LLM%s\n' \
    "$BOLD" "$RESET"
# ============================================================================
#
# All eight stages, with both models served over HTTP by stubs. This is the only
# case that exercises the remote transcription client and the LLM detection
# client, and the same crosstalk scenario as case 6 rides along on top: a
# repetition Alice makes while Bob is talking has to end up muted, not cut.

CASE7="$SANDBOX/case7"
build_episode ep007 "$CASE7/incoming" \
    "$(windows_expr '0,20')" "$(windows_expr '4,6')" 20

# Both stubs' replies, generated together because they have to agree: the LLM
# names the repetition by word index, so the indices depend on the transcript the
# whisper stub hands back. That transcript also has to cover the tone in the
# audio — it is the speech map now, and anything it leaves out reads as silence
# and gets cut.
python3 - "$SANDBOX" <<'REPLIES'
import json, os, sys

sandbox = sys.argv[1]

# whisper-server verbose_json, one word per segment (what max_len=1 produces),
# in the OpenAI-ish start/end shape rather than whisper.cpp's own offsets — the
# client accepts several, and this is the one nothing else covers.
alice, stutter = [], []
moment = 0.0
# 4.5 and 5.5 are left empty for the same reason as case 6: room for the cut
# padding, without either gap reaching SILENCE_MIN_DURATION.
while moment < 20.0 - 1e-9:
    if abs(moment - 5.0) < 1e-9:
        for half in (0.0, 0.25):
            stutter.append(len(alice))
            alice.append({"id": len(alice), "start": 5.0 + half,
                          "end": 5.25 + half, "text": " eu"})
    elif abs(moment - 4.5) < 1e-9 or abs(moment - 5.5) < 1e-9:
        pass
    else:
        alice.append({"id": len(alice), "start": moment,
                      "end": moment + 0.4, "text": f" w{len(alice)}"})
    moment += 0.5

bob = [
    {"id": 0, "start": 4.0, "end": 4.9, "text": " sim"},
    {"id": 1, "start": 5.0, "end": 6.0, "text": " claro"},
]

with open(os.path.join(sandbox, "whisper-replies.json"), "w") as handle:
    json.dump([
        {"task": "transcribe", "language": "pt", "duration": 20.0,
         "text": " ".join(s["text"].strip() for s in alice), "segments": alice},
        {"task": "transcribe", "language": "pt", "duration": 20.0,
         "text": " ".join(s["text"].strip() for s in bob), "segments": bob},
    ], handle)

# The model's replies. The first is consumed by the startup schema check, which
# sends one tiny constrained request before any track is analysed; the rest are
# the tracks in processing order (alice, then bob).
with open(os.path.join(sandbox, "llama-replies.json"), "w") as handle:
    json.dump([
        {"edits": []},
        {"edits": [{"first": stutter[0], "last": stutter[1],
                    "kind": "repetition", "confidence": 0.95}]},
        {"edits": []},
    ], handle)

with open(os.path.join(sandbox, "case7-stutter.txt"), "w") as handle:
    handle.write(f"{stutter[0]} {stutter[1]}\n")
REPLIES

read -r STUTTER_FIRST STUTTER_LAST < "$SANDBOX/case7-stutter.txt"

start_stub() {  # start_stub <role> <responses> <port-file> <request-log>
    python3 "$ROOT/tests/stub_servers.py" "$1" \
        --responses "$2" --port-file "$3" --request-log "$4" \
        >>"$SANDBOX/stubs.log" 2>&1 &
    printf '%s' "$!"
}

wait_for_port_file() {
    local file="$1" waited=0
    while [[ ! -f "$file" ]] && (( waited < 100 )); do
        sleep 0.05
        waited=$(( waited + 1 ))
    done
    [[ -f "$file" ]]
}

W_PORT_FILE="$SANDBOX/whisper.port"
L_PORT_FILE="$SANDBOX/llama.port"
W_PID=$(start_stub whisper "$SANDBOX/whisper-replies.json" "$W_PORT_FILE" "$SANDBOX/whisper-requests.jsonl")
L_PID=$(start_stub llama "$SANDBOX/llama-replies.json" "$L_PORT_FILE" "$SANDBOX/llama-requests.jsonl")
stop_stubs() { kill "$W_PID" "$L_PID" 2>/dev/null || true; }
trap 'stop_stubs; stop_whisper_stub; cleanup' EXIT

if wait_for_port_file "$W_PORT_FILE" && wait_for_port_file "$L_PORT_FILE"; then
    check "stub servers started" true
else
    check "stub servers started" false
    fail_note "$(cat "$SANDBOX/stubs.log" 2>/dev/null)"
fi

WHISPER_URL="http://127.0.0.1:$(cat "$W_PORT_FILE")"
LLAMA_URL="http://127.0.0.1:$(cat "$L_PORT_FILE")"

if "$ROOT/clean-podcast.sh" --root "$SANDBOX" \
    --input "$CASE7/incoming" --output "$CASE7/output" --work "$CASE7/work" \
    --config "$CONF" --keep-work --quiet \
    --whisper-endpoint "$WHISPER_URL" --llama-endpoint "$LLAMA_URL" \
    >"$SANDBOX/case7.stdout" 2>&1
then
    check "pipeline completed with both models remote" true
else
    check "pipeline completed with both models remote" false
    fail_note "$(tail -n 30 "$SANDBOX/case7.stdout")"
fi

WORK7="$CASE7/work/ep007"
OUT7="$CASE7/output/ep007"

check "no local whisper or llama process was needed" bash -c \
    '! grep -qE "whisper-cli not found|llama-server not found" "$1"' _ \
    "$SANDBOX/case7.stdout"

check "audio was uploaded as multipart with a file part" python3 -c '
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1])]
if len(rows) != 2:
    print(f"expected 2 uploads (one per track), got {len(rows)}")
    sys.exit(1)
for row in rows:
    if "multipart/form-data" not in row["content_type"]:
        print("not multipart:", row["content_type"])
        sys.exit(1)
    if not row["has_file_part"]:
        print("no file part in upload", row)
        sys.exit(1)
    if "response_format" not in row["fields"]:
        print("response_format was not sent:", row["fields"])
        sys.exit(1)
    # 20 s of 16 kHz mono 16-bit is ~640 KB; anything tiny means the audio
    # never made it into the body.
    if row["length"] < 500_000:
        print("upload suspiciously small, bytes:", row["length"])
        sys.exit(1)
' "$SANDBOX/whisper-requests.jsonl"

check "remote segments became word timings" python3 -c '
import json, sys
data = json.load(open(sys.argv[1]))
first, last = (int(v) for v in sys.argv[2:4])
words = data["words"]
if [w["text"] for w in words[first:last + 1]] != ["eu", "eu"]:
    print("the repetition is not where the LLM was told it was:", words[first:last + 2])
    sys.exit(1)
spans = [(w["start"], w["end"]) for w in words[first:last + 1]]
if spans != [(5.0, 5.25), (5.25, 5.5)]:
    print("timings did not survive the round trip:", spans)
    sys.exit(1)
' "$WORK7/words/alice.words.json" "$STUTTER_FIRST" "$STUTTER_LAST"

check "the LLM was asked through the chat endpoint, with a JSON schema" python3 -c '
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1])]
if not rows:
    print("the llama endpoint was never called")
    sys.exit(1)
# The default LLM_API=chat must reach /v1/chat/completions, so the server
# applies the loaded model chat template rather than seeing a raw prompt.
paths = {row["path"] for row in rows}
if paths != {"/v1/chat/completions"}:
    print("unexpected endpoints used:", sorted(paths))
    sys.exit(1)
# Row 0 is the startup schema check; the transcript goes in the ones after it.
if len(rows) < 2:
    print("expected a schema check plus per-track requests, got", len(rows))
    sys.exit(1)
payload = rows[1]["payload"]
if "response_format" not in payload:
    print("no response_format in the request:", sorted(payload))
    sys.exit(1)
schema = payload["response_format"].get("json_schema", {}).get("schema", {})
if "edits" not in schema.get("properties", {}):
    print("the edit schema did not survive into response_format:", payload["response_format"])
    sys.exit(1)
sent = json.dumps(payload.get("messages"))
if "eu" not in sent:
    print("the transcript did not reach the message content")
    sys.exit(1)
' "$SANDBOX/llama-requests.jsonl"

if [[ -f "$WORK7/plan.json" ]]; then
    check "the remote finding became a mute on alice only" python3 -c '
import json, sys
plan = json.load(open(sys.argv[1]))
if plan["cuts"]:
    print("expected no cuts at all, got:", plan["cuts"])
    sys.exit(1)
if len(plan["mutes"]["alice"]) != 1 or plan["mutes"]["bob"]:
    print("mutes are wrong:", plan["mutes"])
    sys.exit(1)
mute = plan["mutes"]["alice"][0]
if abs(mute["start"] - 4.9) > 0.001 or abs(mute["end"] - 5.6) > 0.001:
    print("mute is", mute["start"], "-", mute["end"], "expected 4.9-5.6")
    sys.exit(1)
' "$WORK7/plan.json"
fi

if [[ -s "$OUT7/alice.flac" && -s "$OUT7/bob.flac" ]]; then
    D7A=$(duration_of "$OUT7/alice.flac")
    D7B=$(duration_of "$OUT7/bob.flac")
    check "both tracks keep their original length" approx "$D7A" "$D7B" 0.0005
    check "length is unchanged at 20 s" approx "$D7A" 20 0.05
    A7_MUTED=$(peak_db "$OUT7/alice.flac" 4.95 5.55)
    B7_SAME=$(peak_db "$OUT7/bob.flac" 5.0 5.5)
    check "alice is silent across the muted span" quieter_than "$A7_MUTED" -60
    check "bob is unaffected at the same instant"  louder_than  "$B7_SAME" -20
    fail_note "alice muted=${A7_MUTED}dB bob=${B7_SAME}dB"
fi

stop_stubs
trap 'stop_whisper_stub; cleanup' EXIT

# ============================================================================
printf '\n%sCase 8: input format is independent of output format%s\n' "$BOLD" "$RESET"
# ============================================================================
#
# The same episode as case 1, but each track arrives in a different container
# and codec, and the output is asked for as FLAC regardless. Also checks the
# thing that makes lossy input safe to accept: a container's duration is not
# always the length of the audio inside it, so the real length is measured from
# the decode rather than believed from the header.

CASE8="$SANDBOX/case8"
mkdir -p "$CASE8/incoming"

# alice as WAV (lossless, honest header), bob as AAC in m4a if it is available —
# AAC decodes a few ms longer than its container claims, which is the case worth
# covering. Falls back to MP3, then to plain WAV.
make_track_as "$CASE8/incoming/ep008_alice.wav" pcm_s16le "" 48000 30 "$A_WINDOWS"
BOB_KIND="wav"
if has_encoder aac; then
    make_track_as "$CASE8/incoming/ep008_bob.m4a" aac "-b:a 192k" 48000 30 "$B_WINDOWS"
    BOB_KIND="m4a/aac"
elif has_encoder libmp3lame; then
    make_track_as "$CASE8/incoming/ep008_bob.mp3" libmp3lame "-b:a 192k" 48000 30 "$B_WINDOWS"
    BOB_KIND="mp3"
else
    make_track_as "$CASE8/incoming/ep008_bob.wav" pcm_s16le "" 48000 30 "$B_WINDOWS"
fi
fail_note "alice=wav bob=$BOB_KIND, output requested as flac"

# Same speech windows as case 1, so the same canned transcript fits.
start_whisper_stub "$SANDBOX/case1.words.json"

CONF8="$SANDBOX/case8.conf"
cat "$CONF" >"$CONF8"
cat >>"$CONF8" <<'EOF'
INPUT_EXTS="flac wav m4a mp3"
OUTPUT_CODEC="flac"
OUTPUT_EXT="flac"
EOF

if run_pipeline "$CASE8/incoming" "$CASE8/output" "$CASE8/work" \
    --config "$CONF8" --keep-work >"$SANDBOX/case8.stdout" 2>&1
then
    check "mixed-format episode completed" true
else
    check "mixed-format episode completed" false
    fail_note "$(tail -n 25 "$SANDBOX/case8.stdout")"
fi

OUT8="$CASE8/output/ep008"
WORK8="$CASE8/work/ep008"

check "output is flac whatever went in" test -s "$OUT8/alice.flac"
check "the other track too"             test -s "$OUT8/bob.flac"
check "no input extension leaked into the output names" bash -c \
    '[[ -z "$(find "$1" -maxdepth 1 \( -name "*.wav" -o -name "*.m4a" -o -name "*.mp3" \) -print -quit)" ]]' \
    _ "$OUT8"

if [[ -f "$OUT8/alice.flac" ]]; then
    check "outputs really are FLAC streams" bash -c \
        '[[ "$(ffprobe -v error -select_streams a:0 -show_entries stream=codec_name -of csv=p=0 "$1")" == flac ]]' \
        _ "$OUT8/alice.flac"
fi

if [[ -f "$WORK8/meta.json" ]]; then
    check "durations were measured, not taken from the container" python3 -c '
import json, sys
meta = json.load(open(sys.argv[1]))
if not meta.get("durations_measured"):
    print("meta.json was never refreshed from the decoded audio")
    sys.exit(1)
for track in meta["tracks"]:
    if "container_duration" not in track:
        print("no container duration recorded for", track["participant"])
        sys.exit(1)
    # The measured value is what the rest of the pipeline uses.
    if track["duration"] <= 0:
        print("bad measured duration:", track)
        sys.exit(1)
' "$WORK8/meta.json"

    check "the input codec of each track was recorded" python3 -c '
import json, sys
meta = json.load(open(sys.argv[1]))
codecs = {t["participant"]: t["codec"] for t in meta["tracks"]}
if not all(codecs.values()):
    print("a codec was not identified:", codecs)
    sys.exit(1)
print("codecs:", codecs)
' "$WORK8/meta.json"
fi

# The assumption that makes mixed-format episodes safe at all, and the reason
# this pipeline carries no warning about them: every decoder accounts for its
# codec's encoder delay, so identical audio in different formats decodes to the
# same sample offset. Guarded here because a regression would show up as silent
# misalignment between tracks rather than as any kind of error.
check "codecs agree on where a sample sits" python3 -c '
import subprocess, sys, struct, shutil

SPECS = [("flac", "flac", []), ("mp3", "libmp3lame", ["-b:a", "192k"]),
         ("m4a", "aac", ["-b:a", "192k"]), ("opus", "libopus", ["-b:a", "128k"]),
         ("ogg", "libvorbis", ["-q:a", "6"])]
work = sys.argv[1]

def has(codec):
    out = subprocess.run(["ffmpeg", "-hide_banner", "-h", f"encoder={codec}"],
                         capture_output=True, text=True).stdout
    return f"Encoder {codec}" in out

def onset(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "s16le",
                          "-ac", "1", "-ar", "48000", "-"],
                         capture_output=True).stdout
    for index in range(0, len(raw) - 1, 2):
        if abs(struct.unpack_from("<h", raw, index)[0]) > 2000:
            return index // 2
    return None

source = f"{work}/click.wav"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                "aevalsrc=exprs=0.8*sin(2*PI*1000*t)*between(t\\,5\\,5.01)"
                ":s=48000:d=10", "-c:a", "pcm_s16le", source], check=True)

reference = onset(source)
if reference is None:
    print("the reference click was not found at all")
    sys.exit(1)

results = {"source": reference}
for ext, codec, extra in SPECS:
    if not has(codec):
        continue
    encoded = f"{work}/click.{ext}"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", source, "-c:a", codec]
                   + extra + [encoded], check=True)
    results[ext] = onset(encoded)

if len(results) < 3:
    print("too few encoders available to prove anything:", results)
    sys.exit(1)
if len(set(results.values())) != 1:
    print("codecs disagree on the click position:", results)
    sys.exit(1)
print("all agree at sample", reference, "-", ", ".join(sorted(results)))
' "$SANDBOX"

if [[ -s "$OUT8/alice.flac" && -s "$OUT8/bob.flac" ]]; then
    D8A=$(duration_of "$OUT8/alice.flac")
    D8B=$(duration_of "$OUT8/bob.flac")
    # The invariant that matters, now across two different input formats.
    check "both tracks are identically long across formats" approx "$D8A" "$D8B" 0.0005
    EXPECT8=$(json_number "$WORK8/expected.json" tracks.alice.expected_duration)
    check "still matches the frame-exact prediction" approx "$D8A" "$EXPECT8" 0.05
    fail_note "alice=${D8A}s bob=${D8B}s predicted=${EXPECT8}s"
fi

# ============================================================================
printf '\n%sCase 9: the same participant in two formats is refused%s\n' "$BOLD" "$RESET"
# ============================================================================

CASE9="$SANDBOX/case9"
mkdir -p "$CASE9/incoming"
make_track "$CASE9/incoming/ep009_alice.flac" 48000 10 "$(windows_expr '0,5')"
make_track_as "$CASE9/incoming/ep009_alice.wav" pcm_s16le "" 48000 10 \
    "$(windows_expr '0,5')"

if run_pipeline "$CASE9/incoming" "$CASE9/output" "$CASE9/work" \
    --config "$CONF8" >"$SANDBOX/case9.stdout" 2>&1
then
    check "duplicate track in two formats refused" false
    fail_note "the run should have failed but did not"
else
    check "duplicate track in two formats refused" true
fi
check "the error names both files" \
    grep -qE "appears twice.*alice" "$SANDBOX/case9.stdout"

# ============================================================================
printf '\n%sCase 10: mixed sample rates, refused by default and fixed by RESAMPLE_TO%s\n' \
    "$BOLD" "$RESET"
# ============================================================================
#
# Two tracks at different rates cannot be cut consistently: the same cut list
# quantised at 48 kHz and at 44.1 kHz removes slightly different spans, and the
# error compounds over hundreds of cuts. Refused unless resampling is asked for.

CASE10="$SANDBOX/case10"
mkdir -p "$CASE10/incoming"
make_track "$CASE10/incoming/ep010_alice.flac" 48000 30 "$A_WINDOWS"
make_track "$CASE10/incoming/ep010_bob.flac"   44100 30 "$B_WINDOWS"
# Case 1's windows again, so its transcript fits both tracks.
start_whisper_stub "$SANDBOX/case1.words.json"

if run_pipeline "$CASE10/incoming" "$CASE10/output" "$CASE10/work" \
    --config "$CONF" >"$SANDBOX/case10a.stdout" 2>&1
then
    check "mixed rates refused without RESAMPLE_TO" false
else
    check "mixed rates refused without RESAMPLE_TO" true
fi
check "the refusal explains the consequence" \
    grep -qi "same span from every track" "$SANDBOX/case10a.stdout"

CONF10="$SANDBOX/case10.conf"
cat "$CONF" >"$CONF10"
printf 'RESAMPLE_TO="auto"\n' >>"$CONF10"

if run_pipeline "$CASE10/incoming" "$CASE10/output" "$CASE10/work" \
    --config "$CONF10" --keep-work >"$SANDBOX/case10b.stdout" 2>&1
then
    check "RESAMPLE_TO=auto makes it work" true
else
    check "RESAMPLE_TO=auto makes it work" false
    fail_note "$(tail -n 25 "$SANDBOX/case10b.stdout")"
fi

OUT10="$CASE10/output/ep010"
if [[ -s "$OUT10/alice.flac" && -s "$OUT10/bob.flac" ]]; then
    R10A=$(ffprobe -v error -select_streams a:0 -show_entries stream=sample_rate -of csv=p=0 "$OUT10/alice.flac")
    R10B=$(ffprobe -v error -select_streams a:0 -show_entries stream=sample_rate -of csv=p=0 "$OUT10/bob.flac")
    check "both outputs are at the higher rate" bash -c \
        '[[ "$1" == 48000 && "$2" == 48000 ]]' _ "$R10A" "$R10B"
    D10A=$(duration_of "$OUT10/alice.flac")
    D10B=$(duration_of "$OUT10/bob.flac")
    # The point of resampling: after it, the sync invariant holds again.
    check "resampled tracks are identically long" approx "$D10A" "$D10B" 0.0005
    fail_note "rates ${R10A}/${R10B} Hz, lengths ${D10A}s/${D10B}s"

    check "the resample runs before frames are fixed" bash -c \
        'head -c 200 "$1" | grep -q "aresample=48000,asetnsamples"' \
        _ "$CASE10/work/ep010/render/bob.filter"
fi

# ============================================================================
printf '\n%sCase 11: SPEECH_PAD decides what counts as a gap%s\n' "$BOLD" "$RESET"
# ============================================================================
#
# The only threshold left. Whisper's word timings are approximate, so each word
# is widened before the union that makes the speech map, and a gap has to clear
# SILENCE_MIN_DURATION *plus twice the padding* before it is silence. Here the
# gap is 1.8 s: a cut at SPEECH_PAD=0, nothing at 0.25, same audio and same
# transcript both times.

CASE11="$SANDBOX/case11"
A11_SPEECH='0,10'
B11_SPEECH='11.8,20'
whisper_responses "$SANDBOX/case11.words.json" \
    "alice=$A11_SPEECH" "bob=$B11_SPEECH"
start_whisper_stub "$SANDBOX/case11.words.json"

internal_cuts() {  # internal_cuts <plan.json>
    python3 -c '
import json, sys
plan = json.load(open(sys.argv[1]))
print(len([c for c in plan["cuts"] if "silence" in c["reasons"]]))
' "$1"
}

for PAD in 0 0.25; do
    SLUG="${PAD/./_}"
    build_episode ep011 "$CASE11/incoming-$SLUG" \
        "$(windows_expr "$A11_SPEECH")" "$(windows_expr "$B11_SPEECH")" 20
    CONF11="$SANDBOX/case11-$SLUG.conf"
    cat "$CONF" >"$CONF11"
    printf 'SPEECH_PAD="%s"\n' "$PAD" >>"$CONF11"

    if run_pipeline "$CASE11/incoming-$SLUG" "$CASE11/output-$SLUG" \
        "$CASE11/work-$SLUG" --config "$CONF11" --keep-work \
        >"$SANDBOX/case11-$SLUG.stdout" 2>&1
    then
        CUTS11=$(internal_cuts "$CASE11/work-$SLUG/ep011/plan.json")
        if [[ "$PAD" == 0 ]]; then
            check "an unpadded 1.8s gap is a cut" test "$CUTS11" = "1"
        else
            check "the same gap padded to 1.3s is left alone" test "$CUTS11" = "0"
        fi
        fail_note "SPEECH_PAD=$PAD -> $CUTS11 internal cut(s)"
    else
        check "case 11 run at SPEECH_PAD=$PAD" false
        fail_note "$(tail -n 20 "$SANDBOX/case11-$SLUG.stdout")"
    fi
done

# ============================================================================

printf '\n'
if (( FAILURES == 0 )); then
    printf '%s✓ all %d checks passed%s\n\n' "$GREEN" "$CHECKS" "$RESET"
    exit 0
fi
printf '%s✗ %d of %d checks failed%s\n' "$RED" "$FAILURES" "$CHECKS" "$RESET"
printf '%sre-run with KEEP_SANDBOX=1 to inspect %s%s\n\n' "$DIM" "$SANDBOX" "$RESET"
exit 1
