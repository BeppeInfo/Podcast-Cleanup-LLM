#!/usr/bin/env bash
#
# Offline end-to-end check.
#
# Builds a synthetic two-track "episode" with silences in known places, runs the
# real pipeline over it in a sandbox, and checks the results. Whisper and
# llama.cpp are skipped, so this needs nothing but ffmpeg and python3 — it
# exercises the parts that actually touch audio, which are the parts that are
# awkward to reason about on paper.
#
# The synthetic episode is 30 seconds long:
#
#   track a speaks   0-4      10-14        20-24
#   track b speaks        5-9        15-19        25-29
#   nobody speaks       4-5   9-10  14-15  19-20  24-25  29-30
#
# Every internal gap is 1 s, below the 1.5 s threshold, so the only cut should
# be the 1 s tail — which makes the arithmetic easy to check by hand. A second
# pass then widens one gap to prove that shortening works too.

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

run_pipeline() {
    local incoming="$1" output="$2" work="$3"; shift 3
    # Whisper and llama.cpp are out of scope here, so their stages are left out
    # of the list entirely rather than stubbed.
    "$ROOT/clean-podcast.sh" \
        --input "$incoming" --output "$output" --work "$work" \
        --stages discover,prepare,vad,plan,render,finalize \
        --no-llm --quiet "$@"
}

printf '\n%sPodcast cleanup self-test%s %s(%s)%s\n\n' \
    "$BOLD" "$RESET" "$DIM" "$SANDBOX" "$RESET"

# ============================================================================
printf '%sCase 1: gaps below the threshold are left alone%s\n' "$BOLD" "$RESET"
# ============================================================================

CASE1="$SANDBOX/case1"
A_WINDOWS='between(t,0,4)+between(t,10,14)+between(t,20,24)'
B_WINDOWS='between(t,5,9)+between(t,15,19)+between(t,25,29)'
build_episode ep001 "$CASE1/incoming" "$A_WINDOWS" "$B_WINDOWS" 30

CONF="$SANDBOX/case1.conf"
cat >"$CONF" <<'EOF'
SILENCE_MIN_DURATION="1.5"
SILENCE_KEEP="0.40"
EDGE_KEEP="0.25"
CUT_PADDING="0.10"
MIN_CUT="0.15"
VAD_MIN_SILENCE="0.30"
SILENCE_THRESHOLD="-35dB"
RENDER_FRAME_SAMPLES="512"
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

    # A cut that lands inside anybody's speech would be a serious bug.
    check "no cut overlaps speech" python3 -c '
import json, sys
plan = json.load(open(sys.argv[1]))
speech = []
for path in sys.argv[2:]:
    speech += [tuple(s) for s in json.load(open(path))["speech"]]
for cut in plan["cuts"]:
    for start, end in speech:
        overlap = min(cut["end"], end) - max(cut["start"], start)
        if overlap > 0.02:
            print(f"cut {cut} overlaps speech {(start, end)} by {overlap:.3f}s")
            sys.exit(1)
' "$PLAN1" "$CASE1/work/ep001/vad/alice.json" "$CASE1/work/ep001/vad/bob.json"
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
A2='between(t,0,4)+between(t,18,22)'
B2='between(t,4,8)+between(t,22,26)'
build_episode ep002 "$CASE2/incoming" "$A2" "$B2" 30

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
# A 10 s gap keeping 0.4 s leaves a 9.6 s cut, give or take VAD boundaries.
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
build_episode ep003 "$CASE3/incoming" 'between(t,1,2)' 'between(t,58,59)' 60

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
make_track "$CASE5/incoming/ep005_alice.flac" 48000 10 'between(t,0,5)'
make_track "$CASE5/incoming/ep005_bob.flac"   44100 10 'between(t,5,9)'

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
# Whisper and Qwen are not available here, so their outputs are injected
# directly — the point under test is what the plan and the render do with them.

CASE6="$SANDBOX/case6"
# Both speak throughout, so there is no silence anywhere to cut.
build_episode ep006 "$CASE6/incoming" 'between(t,0,20)' 'between(t,4,6)' 20

run_pipeline "$CASE6/incoming" "$CASE6/output" "$CASE6/work" \
    --config "$CONF" --keep-work --stages discover,prepare,vad \
    >"$SANDBOX/case6a.stdout" 2>&1 \
    || { check "case 6 setup" false; fail_note "$(tail -n 20 "$SANDBOX/case6a.stdout")"; }

WORK6="$CASE6/work/ep006"
python3 - "$WORK6" <<'INJECT'
import json, os, sys

work = sys.argv[1]
# Alice says "a ... eu eu acho": words 1 and 2 are the accidental repetition.
words = [
    {"i": 0, "text": "a",    "start": 3.00, "end": 3.20, "segment": 0},
    {"i": 1, "text": "eu",   "start": 5.00, "end": 5.25, "segment": 0},
    {"i": 2, "text": "eu",   "start": 5.25, "end": 5.50, "segment": 0},
    {"i": 3, "text": "acho", "start": 7.00, "end": 7.50, "segment": 0},
]
for name, payload in (
    ("alice", words),
    ("bob", [{"i": 0, "text": "sim", "start": 4.5, "end": 5.0, "segment": 0}]),
):
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
            "first": 1, "last": 2, "kind": "repetition", "confidence": 0.95,
            "start": 5.0, "end": 5.5, "text": "eu eu",
        }],
    },
    open(os.path.join(work, "llm", "alice.edits.json"), "w"),
)
INJECT

# Deliberately without --no-llm: the injected edits must be picked up.
if "$ROOT/clean-podcast.sh" --input "$CASE6/incoming" --output "$CASE6/output" \
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
if "acho" not in alice:
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
build_episode ep007 "$CASE7/incoming" 'between(t,0,20)' 'between(t,4,6)' 20

# whisper-server verbose_json, one word per segment (what max_len=1 produces).
cat >"$SANDBOX/whisper-replies.json" <<'EOF'
[
  {"task": "transcribe", "language": "pt", "duration": 20.0,
   "text": " a eu eu acho",
   "segments": [
     {"id": 0, "start": 3.00, "end": 3.20, "text": " a"},
     {"id": 1, "start": 5.00, "end": 5.25, "text": " eu"},
     {"id": 2, "start": 5.25, "end": 5.50, "text": " eu"},
     {"id": 3, "start": 7.00, "end": 7.50, "text": " acho"}
   ]},
  {"task": "transcribe", "language": "pt", "duration": 20.0,
   "text": " sim",
   "segments": [
     {"id": 0, "start": 4.50, "end": 5.00, "text": " sim"}
   ]}
]
EOF

# The model's replies, in the order the tracks are processed (alice, then bob).
cat >"$SANDBOX/llama-replies.json" <<'EOF'
[
  {"edits": [{"first": 1, "last": 2, "kind": "repetition", "confidence": 0.95}]},
  {"edits": []}
]
EOF

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
trap 'stop_stubs; cleanup' EXIT

if wait_for_port_file "$W_PORT_FILE" && wait_for_port_file "$L_PORT_FILE"; then
    check "stub servers started" true
else
    check "stub servers started" false
    fail_note "$(cat "$SANDBOX/stubs.log" 2>/dev/null)"
fi

WHISPER_URL="http://127.0.0.1:$(cat "$W_PORT_FILE")"
LLAMA_URL="http://127.0.0.1:$(cat "$L_PORT_FILE")"

if "$ROOT/clean-podcast.sh" \
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
texts = [w["text"] for w in data["words"]]
if texts != ["a", "eu", "eu", "acho"]:
    print("unexpected words:", texts)
    sys.exit(1)
spans = [(w["start"], w["end"]) for w in data["words"]]
if spans != [(3.0, 3.2), (5.0, 5.25), (5.25, 5.5), (7.0, 7.5)]:
    print("timings did not survive the round trip:", spans)
    sys.exit(1)
' "$WORK7/words/alice.words.json"

check "the LLM was asked with a JSON schema" python3 -c '
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1])]
if not rows:
    print("the llama endpoint was never called")
    sys.exit(1)
payload = rows[0]["payload"]
if "json_schema" not in payload:
    print("no json_schema in the request:", sorted(payload))
    sys.exit(1)
if "eu" not in payload["prompt"]:
    print("the transcript did not reach the prompt")
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
trap cleanup EXIT

# ============================================================================

printf '\n'
if (( FAILURES == 0 )); then
    printf '%s✓ all %d checks passed%s\n\n' "$GREEN" "$CHECKS" "$RESET"
    exit 0
fi
printf '%s✗ %d of %d checks failed%s\n' "$RED" "$FAILURES" "$CHECKS" "$RESET"
printf '%sre-run with KEEP_SANDBOX=1 to inspect %s%s\n\n' "$DIM" "$SANDBOX" "$RESET"
exit 1
