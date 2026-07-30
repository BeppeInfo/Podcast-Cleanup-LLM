#!/usr/bin/env python3
"""Unit tests for the decision-making half of the pipeline.

Stdlib only, no pytest:  python3 tests/test_pipeline.py

The interesting ones are `TestFilterExpressions`, which evaluate the generated
ffmpeg expressions with a miniature interpreter of the exact grammar we emit.
A silently wrong expression would mangle the audio without failing anything, so
it gets checked rather than eyeballed.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

from cleanup import intervals as iv
from cleanup import asr, llm, plan as planner, render, transcript as tr, vad


# --- a tiny evaluator for the ffmpeg expression subset we generate ------------

def eval_expr(expression: str, t: float) -> float:
    scope = {
        "t": t,
        "if_": lambda c, a, b: a if c else b,
        "lt": lambda a, b: 1.0 if a < b else 0.0,
        "between": lambda x, lo, hi: 1.0 if lo <= x <= hi else 0.0,
        "clip": lambda x, lo, hi: max(lo, min(hi, x)),
    }
    known = {"if_", "lt", "between", "clip", "t"}
    translated = re.sub(r"\bif\(", "if_(", expression)
    for name in re.findall(r"[A-Za-z_][A-Za-z_0-9]*", translated):
        if name not in known:
            raise AssertionError(f"expression uses unexpected symbol {name!r}")
    return float(eval(translated, {"__builtins__": {}}, scope))  # noqa: S307


class TestIntervals(unittest.TestCase):
    def test_normalize_fuses_touching(self):
        self.assertEqual(
            iv.normalize([(2, 3), (0, 1), (1, 2)]), [(0.0, 3.0)]
        )

    def test_normalize_drops_empty(self):
        self.assertEqual(iv.normalize([(1, 1), (2, 5)]), [(2.0, 5.0)])

    def test_normalize_gap_fusion(self):
        self.assertEqual(iv.normalize([(0, 1), (1.05, 2)], gap=0.1), [(0.0, 2.0)])
        self.assertEqual(
            iv.normalize([(0, 1), (1.5, 2)], gap=0.1), [(0.0, 1.0), (1.5, 2.0)]
        )

    def test_subtract_splits(self):
        self.assertEqual(
            iv.subtract([(0, 10)], [(2, 3), (5, 6)]),
            [(0.0, 2.0), (3.0, 5.0), (6.0, 10.0)],
        )

    def test_subtract_engulfed(self):
        self.assertEqual(iv.subtract([(2, 3)], [(0, 10)]), [])

    def test_intersect(self):
        self.assertEqual(
            iv.intersect([(0, 5), (7, 10)], [(3, 8)]), [(3.0, 5.0), (7.0, 8.0)]
        )

    def test_complement(self):
        self.assertEqual(
            iv.complement([(1, 2)], 0, 5), [(0.0, 1.0), (2.0, 5.0)]
        )

    def test_overlaps_tolerance(self):
        self.assertTrue(iv.overlaps((1, 2), [(1.5, 3)]))
        self.assertFalse(iv.overlaps((1, 2), [(2, 3)]))

    def test_timeline_maps_and_drops(self):
        timeline = iv.Timeline([(0, 10), (20, 30)])
        self.assertAlmostEqual(timeline.duration, 20.0)
        self.assertAlmostEqual(timeline.map(5), 5.0)
        self.assertAlmostEqual(timeline.map(25), 15.0)
        # Anything inside the removed stretch collapses onto the splice point.
        self.assertAlmostEqual(timeline.map(15), 10.0)
        self.assertTrue(timeline.dropped(15))
        self.assertFalse(timeline.dropped(25))
        self.assertAlmostEqual(timeline.map(100), 20.0)

    def test_timeline_is_monotonic(self):
        timeline = iv.Timeline([(1, 2), (5, 6), (9, 12)])
        previous = -1.0
        for step in range(0, 130):
            value = timeline.map(step / 10.0)
            self.assertGreaterEqual(value + 1e-9, previous)
            previous = value


class TestSilenceDetectParsing(unittest.TestCase):
    def test_pairs_events_in_order(self):
        log = """
        [silencedetect @ 0x1] silence_start: 1.5
        [silencedetect @ 0x1] silence_end: 3.0 | silence_duration: 1.5
        [silencedetect @ 0x1] silence_start: 8.0
        [silencedetect @ 0x1] silence_end: 9.25 | silence_duration: 1.25
        """
        self.assertEqual(
            vad.parse_silencedetect(log, 10.0),
            [(0.0, 1.5), (3.0, 8.0), (9.25, 10.0)],
        )

    def test_unterminated_trailing_silence(self):
        """The draft's zip() approach mispaired everything after this case."""
        log = "silence_start: 2.0\nsilence_end: 4.0\nsilence_start: 9.0\n"
        self.assertEqual(
            vad.parse_silencedetect(log, 10.0), [(0.0, 2.0), (4.0, 9.0)]
        )

    def test_leading_silence(self):
        log = "silence_start: 0\nsilence_end: 2.5\n"
        self.assertEqual(vad.parse_silencedetect(log, 6.0), [(2.5, 6.0)])

    def test_no_silence_at_all(self):
        self.assertEqual(vad.parse_silencedetect("", 5.0), [(0.0, 5.0)])


def _whisper_segment(text, start_ms, end_ms, tokens):
    return {
        "text": text,
        "offsets": {"from": start_ms, "to": end_ms},
        "tokens": [
            {
                "text": token,
                "offsets": {"from": t_from, "to": t_to},
            }
            for token, t_from, t_to in tokens
        ],
    }


class TestTranscriptParsing(unittest.TestCase):
    def _write(self, payload):
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(payload, handle)
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        return handle.name

    def test_words_from_tokens(self):
        payload = {
            "result": {"language": "pt"},
            "transcription": [
                _whisper_segment(
                    "I I think so",
                    0,
                    2000,
                    [
                        ("[_BEG_]", 0, 0),
                        (" I", 0, 300),
                        (" I", 300, 600),
                        (" th", 600, 800),
                        ("ink", 800, 1000),
                        (" so", 1000, 2000),
                    ],
                )
            ],
        }
        parsed = tr.parse_whisper_json(self._write(payload), "alice")
        self.assertEqual([w["text"] for w in parsed["words"]], ["I", "I", "think", "so"])
        self.assertEqual([w["i"] for w in parsed["words"]], [0, 1, 2, 3])
        # Sub-word tokens fuse into one word spanning both.
        self.assertAlmostEqual(parsed["words"][2]["start"], 0.6)
        self.assertAlmostEqual(parsed["words"][2]["end"], 1.0)
        self.assertEqual(parsed["segments"][0]["first_word"], 0)
        self.assertEqual(parsed["segments"][0]["last_word"], 3)
        self.assertEqual(parsed["approximated_segments"], 0)

    def test_falls_back_when_token_timings_are_dead(self):
        payload = {
            "transcription": [
                _whisper_segment(
                    "hello there world",
                    1000,
                    4000,
                    [(" hello", 0, 0), (" there", 0, 0), (" world", 0, 0)],
                )
            ]
        }
        parsed = tr.parse_whisper_json(self._write(payload), "bob")
        self.assertEqual(parsed["approximated_segments"], 1)
        self.assertEqual(len(parsed["words"]), 3)
        self.assertAlmostEqual(parsed["words"][0]["start"], 1.0)
        self.assertAlmostEqual(parsed["words"][-1]["end"], 4.0, places=3)
        for word in parsed["words"]:
            self.assertGreater(word["end"], word["start"])

    def test_neighbour_gaps(self):
        words = [
            {"start": 0.0, "end": 1.0},
            {"start": 1.4, "end": 2.0},
            {"start": 2.9, "end": 3.0},
        ]
        before, after = tr.neighbour_gaps(words, 1, 1)
        self.assertAlmostEqual(before, 0.4)
        self.assertAlmostEqual(after, 0.9)
        self.assertEqual(tr.neighbour_gaps(words, 0, 2), (0.0, 0.0))


class TestLlmValidation(unittest.TestCase):
    def setUp(self):
        self.words = [
            {"i": index, "text": f"w{index}", "start": index * 1.0, "end": index * 1.0 + 0.5}
            for index in range(20)
        ]
        self.limits = {"max_words": 4, "max_seconds": 3.0, "min_confidence": 0.6}
        self.accepted = ["stutter", "repetition"]

    def _validate(self, edits):
        return llm._validate(edits, self.words, 0, 19, self.limits, self.accepted)

    def test_accepts_good_edit(self):
        kept, rejected = self._validate(
            [{"first": 2, "last": 3, "kind": "stutter", "confidence": 0.9}]
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(rejected, [])
        self.assertAlmostEqual(kept[0]["start"], 2.0)
        self.assertAlmostEqual(kept[0]["end"], 3.5)
        self.assertEqual(kept[0]["text"], "w2 w3")

    def test_rejects_hallucinated_index(self):
        kept, rejected = self._validate(
            [{"first": 500, "last": 501, "kind": "stutter", "confidence": 1.0}]
        )
        self.assertEqual(kept, [])
        self.assertIn("outside", rejected[0]["reason"])

    def test_rejects_inverted_reversed_and_oversized(self):
        kept, rejected = self._validate(
            [
                {"first": 5, "last": 2, "kind": "stutter", "confidence": 0.9},
                {"first": 0, "last": 9, "kind": "stutter", "confidence": 0.9},
                {"first": 0, "last": 1, "kind": "filler", "confidence": 0.9},
                {"first": 0, "last": 1, "kind": "stutter", "confidence": 0.2},
                {"first": 0, "last": 1},
            ]
        )
        self.assertEqual(kept, [])
        self.assertEqual(len(rejected), 5)

    def test_rejects_span_longer_than_max_seconds(self):
        # 4 words is within max_words but spans 3.5s, over max_seconds.
        kept, rejected = self._validate(
            [{"first": 0, "last": 3, "kind": "stutter", "confidence": 0.9}]
        )
        self.assertEqual(kept, [])
        self.assertIn("max_seconds", rejected[0]["reason"])

    def test_dedupe_merges_overlapping_ranges(self):
        merged = llm._dedupe(
            [
                {"first": 5, "last": 7, "kind": "stutter", "confidence": 0.7},
                {"first": 6, "last": 9, "kind": "repetition", "confidence": 0.95},
                {"first": 20, "last": 21, "kind": "stutter", "confidence": 0.8},
            ]
        )
        self.assertEqual(len(merged), 2)
        self.assertEqual((merged[0]["first"], merged[0]["last"]), (5, 9))
        self.assertEqual(merged[0]["kind"], "repetition")  # higher confidence wins
        self.assertAlmostEqual(merged[0]["confidence"], 0.95)

    def test_chunk_planning_covers_every_word(self):
        for count in (1, 5, 100, 349, 350, 351, 1000, 7231):
            chunks = llm.plan_chunks(count, 350, 40)
            self.assertEqual(chunks[0][0], 0)
            self.assertEqual(chunks[-1][1], count - 1)
            covered = set()
            for first, last in chunks:
                covered.update(range(first, last + 1))
            self.assertEqual(len(covered), count, f"gap at word_count={count}")

    def test_prompt_mentions_real_indices_only(self):
        words = [
            {"i": index, "text": f"word{index}", "start": index, "end": index + 0.5}
            for index in range(100)
        ]
        prompt = llm.build_prompt(words, 40, 59, ["stutter"], 6)
        self.assertIn("they start at 40 and end at 59", prompt)
        self.assertIn("word40", prompt)
        self.assertNotIn("word39", prompt)
        self.assertNotIn("word60", prompt)
        self.assertIn("at most 6 words", prompt)
        # Only the enabled kinds are described.
        self.assertNotIn("hesitation sound", prompt)


class TestRemoteAsrTimeParsing(unittest.TestCase):
    def test_numbers_and_numeric_strings(self):
        self.assertEqual(asr.to_seconds(12), 12.0)
        self.assertEqual(asr.to_seconds(12.5), 12.5)
        self.assertEqual(asr.to_seconds("12.5"), 12.5)
        self.assertEqual(asr.to_seconds(" 12.5 "), 12.5)

    def test_clock_strings(self):
        self.assertAlmostEqual(asr.to_seconds("00:00:01.500"), 1.5)
        self.assertAlmostEqual(asr.to_seconds("00:00:01,500"), 1.5)
        self.assertAlmostEqual(asr.to_seconds("01:02:03.250"), 3723.25)
        self.assertAlmostEqual(asr.to_seconds("02:03"), 123.0)

    def test_rejects_junk(self):
        for value in (None, "", "  ", "abc", True, [], {}, "1:2:3:4"):
            self.assertIsNone(asr.to_seconds(value), repr(value))


class TestRemoteAsrNormalization(unittest.TestCase):
    def test_openai_style_segments(self):
        payload = {
            "text": " hello there",
            "segments": [
                {"id": 0, "start": 1.0, "end": 1.5, "text": " hello"},
                {"id": 1, "start": 1.5, "end": 2.25, "text": " there"},
            ],
        }
        segments = asr.normalize_response(payload)
        self.assertEqual(
            [(s["offsets"]["from"], s["offsets"]["to"]) for s in segments],
            [(1000, 1500), (1500, 2250)],
        )
        self.assertEqual([s["text"] for s in segments], ["hello", "there"])

    def test_offset_shifts_a_chunk_onto_the_episode_timeline(self):
        payload = {"segments": [{"start": 2.0, "end": 3.0, "text": "x"}]}
        segments = asr.normalize_response(payload, offset=600.0)
        self.assertEqual(segments[0]["offsets"], {"from": 602000, "to": 603000})

    def test_whisper_cli_shape_is_accepted_too(self):
        payload = {
            "transcription": [
                {"text": " word", "offsets": {"from": 500, "to": 900}}
            ]
        }
        segments = asr.normalize_response(payload)
        self.assertEqual(segments[0]["offsets"], {"from": 500, "to": 900})

    def test_clock_string_timings(self):
        payload = {
            "segments": [
                {"start": "00:00:01.250", "end": "00:00:02.000", "text": "a"}
            ]
        }
        segments = asr.normalize_response(payload)
        self.assertEqual(segments[0]["offsets"], {"from": 1250, "to": 2000})

    def test_token_ids_are_discarded_rather_than_crashing(self):
        """whisper-server's verbose_json lists tokens as bare ids."""
        payload = {
            "segments": [
                {"start": 1.0, "end": 2.0, "text": "hi there",
                 "tokens": [50364, 2088, 616]}
            ]
        }
        segments = asr.normalize_response(payload)
        self.assertNotIn("tokens", segments[0])
        # And the words still come out, interpolated across the segment.
        parsed = tr.build_from_segments(segments, "alice")
        self.assertEqual([w["text"] for w in parsed["words"]], ["hi", "there"])
        self.assertEqual(parsed["approximated_segments"], 1)

    def test_token_objects_with_timings_are_kept_and_shifted(self):
        payload = {
            "segments": [{
                "start": 1.0, "end": 2.0, "text": "hi there",
                "tokens": [
                    {"text": " hi", "offsets": {"from": 1000, "to": 1400}},
                    {"text": " there", "offsets": {"from": 1400, "to": 2000}},
                ],
            }]
        }
        segments = asr.normalize_response(payload, offset=10.0)
        self.assertEqual(len(segments[0]["tokens"]), 2)
        self.assertEqual(segments[0]["tokens"][0]["offsets"]["from"], 11000)
        parsed = tr.build_from_segments(segments, "alice")
        self.assertEqual(parsed["approximated_segments"], 0)
        self.assertAlmostEqual(parsed["words"][0]["start"], 11.0)

    def test_text_without_timings_is_a_clear_error(self):
        with self.assertRaises(ValueError) as caught:
            asr.normalize_response({"text": "just a transcript"})
        self.assertIn("verbose_json", str(caught.exception))

    def test_segments_missing_timings_are_skipped(self):
        payload = {
            "segments": [
                {"text": "no timing here"},
                {"start": 1.0, "end": 2.0, "text": "good"},
            ]
        }
        segments = asr.normalize_response(payload)
        self.assertEqual([s["text"] for s in segments], ["good"])

    def test_empty_result_is_an_error_not_a_silent_pass(self):
        with self.assertRaises(ValueError):
            asr.normalize_response({"segments": []})
        with self.assertRaises(ValueError):
            asr.normalize_response({"nothing": True})

    def test_accepts_a_json_string(self):
        segments = asr.normalize_response(
            '{"segments": [{"start": 0, "end": 1, "text": "x"}]}'
        )
        self.assertEqual(len(segments), 1)


class TestRemoteAsrChunking(unittest.TestCase):
    def test_short_track_is_one_chunk(self):
        self.assertEqual(asr.plan_audio_chunks(300.0, 600.0, []), [(0.0, 300.0)])

    def test_zero_target_disables_chunking(self):
        self.assertEqual(asr.plan_audio_chunks(7200.0, 0, []), [(0.0, 7200.0)])

    def test_chunks_cover_the_track_without_gaps_or_overlap(self):
        for duration in (601.0, 1000.0, 3600.0, 7231.5):
            chunks = asr.plan_audio_chunks(duration, 600.0, [])
            self.assertEqual(chunks[0][0], 0.0)
            self.assertAlmostEqual(chunks[-1][1], duration)
            for before, after in zip(chunks, chunks[1:]):
                self.assertAlmostEqual(before[1], after[0])
            for start, end in chunks:
                self.assertGreater(end - start, 0)

    def test_boundaries_move_into_silence(self):
        # Speech everywhere except a gap at 590-610, straddling the ideal 600 s
        # boundary. The split should land in the middle of that gap.
        speech = [(0.0, 590.0), (610.0, 1200.0)]
        chunks = asr.plan_audio_chunks(1200.0, 600.0, speech)
        self.assertEqual(len(chunks), 2)
        self.assertAlmostEqual(chunks[0][1], 600.0, places=3)

    def test_boundary_prefers_nearby_silence_over_the_exact_target(self):
        speech = [(0.0, 500.0), (530.0, 1200.0)]
        chunks = asr.plan_audio_chunks(1200.0, 600.0, speech)
        # The gap's midpoint is 515 s, well inside the quarter-target window,
        # so it is chosen over cutting a word at 600 s.
        self.assertAlmostEqual(chunks[0][1], 515.0, places=3)

    def test_distant_silence_is_not_worth_the_detour(self):
        # The only silence is at the very start, far from the 600 s mark.
        speech = [(0.0, 10.0), (20.0, 1200.0)]
        chunks = asr.plan_audio_chunks(1200.0, 600.0, speech)
        self.assertAlmostEqual(chunks[0][1], 600.0, places=3)

    def test_no_runt_final_chunk(self):
        # 605 s with a 600 s target would leave a 5 s tail; it stays whole.
        chunks = asr.plan_audio_chunks(605.0, 600.0, [])
        self.assertEqual(chunks, [(0.0, 605.0)])
        for start, end in asr.plan_audio_chunks(1900.0, 600.0, []):
            self.assertGreaterEqual(end - start, asr.MIN_CHUNK_SECONDS)


class TestWavSlicing(unittest.TestCase):
    def _make_wav(self, path, seconds, rate=16000):
        import struct
        import wave as wavemod

        with wavemod.open(path, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            # A ramp, so a slice can be identified by its content.
            handle.writeframes(
                b"".join(
                    struct.pack("<h", (index % 1000) - 500)
                    for index in range(int(seconds * rate))
                )
            )

    def test_slice_is_sample_accurate(self):
        directory = tempfile.mkdtemp()
        source = os.path.join(directory, "src.wav")
        target = os.path.join(directory, "cut.wav")
        self._make_wav(source, 3.0)

        duration, rate = asr.wav_info(source)
        self.assertAlmostEqual(duration, 3.0)
        self.assertEqual(rate, 16000)

        asr.slice_wav(source, 1.0, 2.5, target)
        sliced, _ = asr.wav_info(target)
        self.assertAlmostEqual(sliced, 1.5, places=6)

    def test_slice_clamps_to_the_file(self):
        directory = tempfile.mkdtemp()
        source = os.path.join(directory, "src.wav")
        target = os.path.join(directory, "cut.wav")
        self._make_wav(source, 1.0)
        asr.slice_wav(source, 0.5, 99.0, target)
        sliced, _ = asr.wav_info(target)
        self.assertAlmostEqual(sliced, 0.5, places=6)


class _StubLlamaServer:
    """A stand-in for llama-server, to exercise the real HTTP client.

    Serves /health and /completion, hands out canned replies in order, and
    records the requests it received so the payload contract can be checked.
    """

    def __init__(self, replies, health_status=200):
        import http.server
        import threading

        self.replies = list(replies)
        self.requests = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _send(self, status, payload):
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/health":
                    self._send(health_status, {"status": "ok"})
                else:
                    self._send(404, {})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                outer.requests.append(json.loads(self.rfile.read(length)))
                reply = (
                    outer.replies.pop(0) if outer.replies else {"edits": []}
                )
                if isinstance(reply, str):
                    self._send(200, {"content": reply})
                elif reply is None:
                    self._send(500, {"error": "boom"})
                else:
                    self._send(200, {"content": json.dumps(reply)})

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.endpoint = "http://127.0.0.1:%d" % self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class TestAgainstStubServer(unittest.TestCase):
    def _words(self, texts):
        return {
            "participant": "alice",
            "language": "pt",
            "words": [
                {"i": index, "text": text, "start": index * 0.5,
                 "end": index * 0.5 + 0.4, "segment": 0}
                for index, text in enumerate(texts)
            ],
            "segments": [],
        }

    def _detect(self, parsed, replies, **overrides):
        server = _StubLlamaServer(replies)
        self.addCleanup(server.close)
        client = llm.LlamaClient(server.endpoint, timeout=10)
        options = {
            "chunk_words": 100,
            "overlap": 10,
            "limits": {"max_words": 4, "max_seconds": 3.0, "min_confidence": 0.6},
            "accepted": ["stutter", "repetition"],
        }
        options.update(overrides)
        result = llm.detect(client, parsed, **options)
        return result, server

    def test_health_check(self):
        server = _StubLlamaServer([])
        self.addCleanup(server.close)
        client = llm.LlamaClient(server.endpoint)
        self.assertTrue(client.wait_until_ready(timeout=5))

    def test_health_check_gives_up_on_a_dead_endpoint(self):
        # Port 1 is not going to answer.
        client = llm.LlamaClient("http://127.0.0.1:1")
        self.assertFalse(client.wait_until_ready(timeout=2, poll=0.2))

    def test_end_to_end_detection(self):
        parsed = self._words(["eu", "eu", "acho", "que", "sim"])
        result, server = self._detect(
            parsed,
            [{"edits": [{"first": 0, "last": 1, "kind": "repetition",
                         "confidence": 0.9}]}],
        )
        self.assertEqual(len(result["edits"]), 1)
        edit = result["edits"][0]
        self.assertEqual(edit["text"], "eu eu")
        self.assertAlmostEqual(edit["start"], 0.0)
        self.assertAlmostEqual(edit["end"], 0.9)
        self.assertEqual(result["chunk_failures"], 0)

    def test_request_payload_carries_the_schema(self):
        parsed = self._words(["um", "teste"])
        _, server = self._detect(parsed, [{"edits": []}])
        self.assertEqual(len(server.requests), 1)
        payload = server.requests[0]
        self.assertIn("json_schema", payload)
        self.assertEqual(payload["temperature"], 0.0)
        self.assertTrue(payload["cache_prompt"])
        schema = payload["json_schema"]
        kinds = schema["properties"]["edits"]["items"]["properties"]["kind"]["enum"]
        # Only the kinds we accept are offered to the model.
        self.assertEqual(kinds, ["stutter", "repetition"])
        self.assertIn("teste", payload["prompt"])

    def test_unparseable_response_is_survived(self):
        parsed = self._words(["um", "teste"])
        result, _ = self._detect(parsed, ["not json at all"], retries=0)
        self.assertEqual(result["edits"], [])
        self.assertEqual(result["chunk_failures"], 1)

    def test_server_error_is_retried_then_survived(self):
        parsed = self._words(["um", "teste"])
        result, server = self._detect(parsed, [None, None], retries=1)
        self.assertEqual(result["chunk_failures"], 1)
        self.assertEqual(len(server.requests), 2, "should have retried once")

    def test_retry_recovers(self):
        parsed = self._words(["eu", "eu", "acho"])
        result, _ = self._detect(
            parsed,
            [None, {"edits": [{"first": 0, "last": 1, "kind": "stutter",
                               "confidence": 0.8}]}],
            retries=1,
        )
        self.assertEqual(result["chunk_failures"], 0)
        self.assertEqual(len(result["edits"]), 1)

    def test_hallucinated_indices_are_dropped_not_trusted(self):
        parsed = self._words(["um", "teste", "curto"])
        result, _ = self._detect(
            parsed,
            [{"edits": [
                {"first": 900, "last": 901, "kind": "stutter", "confidence": 1.0},
                {"first": 0, "last": 0, "kind": "stutter", "confidence": 0.9},
            ]}],
        )
        self.assertEqual([e["first"] for e in result["edits"]], [0])
        self.assertEqual(result["rejected_count"], 1)

    def test_audit_log_records_every_chunk(self):
        parsed = self._words(["eu", "eu", "acho"])
        audit = os.path.join(tempfile.mkdtemp(), "audit.jsonl")
        self._detect(
            parsed,
            [{"edits": [{"first": 0, "last": 1, "kind": "stutter",
                         "confidence": 0.9}]}],
            audit_path=audit,
        )
        with open(audit, encoding="utf-8") as handle:
            lines = [json.loads(line) for line in handle if line.strip()]
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["accepted"], 1)
        self.assertEqual(lines[0]["words"], [0, 2])

    def test_multiple_chunks_are_deduplicated(self):
        # 30 words with a 10-word window and 4-word overlap: several chunks, and
        # the same finding reported by two of them must collapse into one.
        parsed = self._words([f"w{index}" for index in range(30)])
        reply = {"edits": [{"first": 6, "last": 7, "kind": "stutter",
                            "confidence": 0.9}]}
        result, server = self._detect(
            parsed, [reply, reply, reply, reply, reply],
            chunk_words=10, overlap=4,
        )
        self.assertGreater(len(server.requests), 1)
        # Only the chunks whose window contains 6-7 can contribute it.
        self.assertEqual(len(result["edits"]), 1)
        self.assertEqual((result["edits"][0]["first"], result["edits"][0]["last"]),
                         (6, 7))


PARAMS = {
    "silence_min_duration": 1.5,
    "silence_keep": 0.4,
    "edge_keep": 0.25,
    "cut_padding": 0.1,
    "min_cut": 0.15,
    "mute_fade": 0.03,
    "max_cut_fraction": 0.5,
}


def _meta(participants, duration):
    return {
        "episode_id": "test",
        "duration": duration,
        "sample_rate": 48000,
        "tracks": [
            {
                "participant": name,
                "duration": duration,
                "sample_rate": 48000,
                "sample_fmt": "s16",
            }
            for name in participants
        ],
    }


class TestPlanBuilder(unittest.TestCase):
    def test_silence_is_shortened_not_removed(self):
        # One 5s gap between two bursts of speech.
        speech = {"a": [(0.0, 10.0), (15.0, 20.0)]}
        result = planner.build_plan(_meta(["a"], 20.0), speech, {}, {}, PARAMS)
        internal = [c for c in result["cuts"] if "silence" in c["reasons"]]
        self.assertEqual(len(internal), 1)
        cut = internal[0]
        # 0.4s residual means 0.2s margin either side of the 10..15 gap.
        self.assertAlmostEqual(cut["start"], 10.2, places=3)
        self.assertAlmostEqual(cut["end"], 14.8, places=3)
        remaining = 5.0 - (cut["end"] - cut["start"])
        self.assertAlmostEqual(remaining, PARAMS["silence_keep"], places=3)

    def test_short_gaps_are_left_alone(self):
        speech = {"a": [(0.0, 10.0), (11.0, 20.0)]}  # 1.0s gap, under the threshold
        result = planner.build_plan(_meta(["a"], 20.0), speech, {}, {}, PARAMS)
        self.assertEqual([c for c in result["cuts"] if "silence" in c["reasons"]], [])

    def test_head_and_tail_are_trimmed_to_edge_keep(self):
        speech = {"a": [(5.0, 15.0)]}
        result = planner.build_plan(_meta(["a"], 20.0), speech, {}, {}, PARAMS)
        head = next(c for c in result["cuts"] if "lead_in" in c["reasons"])
        tail = next(c for c in result["cuts"] if "lead_out" in c["reasons"])
        self.assertAlmostEqual(head["start"], 0.0)
        self.assertAlmostEqual(head["end"], 4.75, places=3)
        self.assertAlmostEqual(tail["start"], 15.25, places=3)
        self.assertAlmostEqual(tail["end"], 20.0)

    def test_silence_requires_all_tracks_quiet(self):
        # b keeps talking through a's pause, so there is no silence to shorten.
        speech = {"a": [(0.0, 5.0), (12.0, 20.0)], "b": [(0.0, 20.0)]}
        result = planner.build_plan(_meta(["a", "b"], 20.0), speech, {}, {}, PARAMS)
        self.assertEqual(result["cuts"], [])
        self.assertAlmostEqual(result["stats"]["output_duration"], 20.0)

    def test_solo_stutter_becomes_a_global_cut(self):
        speech = {"a": [(0.0, 20.0)], "b": [(0.0, 2.0)]}
        words = {
            "a": [
                {"i": 0, "text": "I", "start": 10.0, "end": 10.3},
                {"i": 1, "text": "I", "start": 10.4, "end": 10.7},
                {"i": 2, "text": "think", "start": 11.0, "end": 11.5},
            ]
        }
        edits = {
            "a": [
                {
                    "first": 0,
                    "last": 1,
                    "kind": "stutter",
                    "confidence": 0.9,
                    "start": 10.0,
                    "end": 10.7,
                    "text": "I I",
                }
            ]
        }
        result = planner.build_plan(_meta(["a", "b"], 20.0), speech, edits, words, PARAMS)
        stutter = next(c for c in result["cuts"] if "stutter" in c["reasons"])
        self.assertEqual(stutter["sources"], ["llm:a"])
        self.assertEqual(result["mutes"]["a"], [])
        # The gap to the next word is 0.3s, half of which is 0.15s, so the full
        # 0.1s of cut_padding is available and is what gets claimed.
        self.assertAlmostEqual(stutter["end"], 10.8, places=3)
        self.assertAlmostEqual(stutter["start"], 10.0, places=3)  # nothing before it

    def test_stutter_over_crosstalk_becomes_a_mute(self):
        """The decisive case: b is talking, so a's stutter is silenced in place."""
        speech = {"a": [(0.0, 20.0)], "b": [(9.0, 13.0)]}
        words = {
            "a": [
                {"i": 0, "text": "I", "start": 10.0, "end": 10.3},
                {"i": 1, "text": "I", "start": 10.4, "end": 10.7},
                {"i": 2, "text": "think", "start": 11.0, "end": 11.5},
            ]
        }
        edits = {
            "a": [
                {
                    "first": 0,
                    "last": 1,
                    "kind": "stutter",
                    "confidence": 0.9,
                    "start": 10.0,
                    "end": 10.7,
                    "text": "I I",
                }
            ]
        }
        result = planner.build_plan(_meta(["a", "b"], 20.0), speech, edits, words, PARAMS)
        self.assertEqual(result["cuts"], [], "timeline must not be shortened")
        self.assertEqual(len(result["mutes"]["a"]), 1)
        self.assertEqual(result["mutes"]["b"], [])
        mute = result["mutes"]["a"][0]
        self.assertAlmostEqual(mute["start"], 10.0)
        self.assertAlmostEqual(mute["end"], 10.8, places=3)
        self.assertAlmostEqual(result["stats"]["output_duration"], 20.0)

    def test_mutes_inside_a_cut_are_dropped(self):
        speech = {"a": [(0.0, 20.0)], "b": [(0.0, 20.0)]}
        words = {"a": [{"i": 0, "text": "um", "start": 5.0, "end": 5.4}]}
        edits = {
            "a": [
                {
                    "first": 0,
                    "last": 0,
                    "kind": "stutter",
                    "confidence": 0.9,
                    "start": 5.0,
                    "end": 5.4,
                    "text": "um",
                }
            ]
        }
        base = planner.build_plan(_meta(["a", "b"], 20.0), speech, edits, words, PARAMS)
        self.assertEqual(len(base["mutes"]["a"]), 1)

        # Now make that region silent on every track: the gap cut swallows it.
        speech2 = {"a": [(0.0, 3.0), (12.0, 20.0)], "b": [(0.0, 3.0), (12.0, 20.0)]}
        result = planner.build_plan(
            _meta(["a", "b"], 20.0), speech2, edits, words, PARAMS
        )
        self.assertEqual(result["mutes"]["a"], [])

    def test_nearby_mutes_fuse_so_fades_cannot_collide(self):
        speech = {"a": [(0.0, 20.0)], "b": [(0.0, 20.0)]}
        words = {
            "a": [
                {"i": 0, "text": "the", "start": 5.0, "end": 5.2},
                {"i": 1, "text": "the", "start": 5.22, "end": 5.4},
            ]
        }
        edits = {
            "a": [
                {"first": 0, "last": 0, "kind": "stutter", "confidence": 0.9,
                 "start": 5.0, "end": 5.2, "text": "the"},
                {"first": 1, "last": 1, "kind": "stutter", "confidence": 0.9,
                 "start": 5.22, "end": 5.4, "text": "the"},
            ]
        }
        result = planner.build_plan(_meta(["a", "b"], 20.0), speech, edits, words, PARAMS)
        self.assertEqual(len(result["mutes"]["a"]), 1)
        self.assertAlmostEqual(result["mutes"]["a"][0]["start"], 5.0)
        self.assertAlmostEqual(result["mutes"]["a"][0]["end"], 5.4)

    def test_keep_and_cuts_are_complementary(self):
        speech = {"a": [(0.0, 4.0), (10.0, 14.0), (25.0, 30.0)]}
        result = planner.build_plan(_meta(["a"], 30.0), speech, {}, {}, PARAMS)
        keep = [tuple(k) for k in result["keep"]]
        cuts = [(c["start"], c["end"]) for c in result["cuts"]]
        self.assertEqual(iv.subtract([(0.0, 30.0)], cuts), keep)
        self.assertAlmostEqual(
            iv.total(keep) + iv.total(cuts), 30.0, places=3
        )
        self.assertAlmostEqual(
            result["stats"]["output_duration"], iv.total(keep), places=3
        )

    def test_excessive_cutting_raises_a_warning(self):
        speech = {"a": [(0.0, 1.0), (29.0, 30.0)]}
        result = planner.build_plan(_meta(["a"], 30.0), speech, {}, {}, PARAMS)
        self.assertTrue(
            any("safety limit" in w for w in result["warnings"]), result["warnings"]
        )

    def test_no_speech_at_all_is_flagged(self):
        result = planner.build_plan(_meta(["a"], 30.0), {"a": []}, {}, {}, PARAMS)
        self.assertTrue(any("no speech" in w for w in result["warnings"]))

    def test_report_renders(self):
        speech = {"a": [(0.0, 10.0), (15.0, 20.0)]}
        result = planner.build_plan(_meta(["a"], 20.0), speech, {}, {}, PARAMS)
        report = planner.format_report(result)
        self.assertIn("Episode:", report)
        self.assertIn("Removed:", report)


class TestFilterExpressions(unittest.TestCase):
    def test_cut_expression_matches_the_cut_list(self):
        cuts = [
            {"start": 1.0, "end": 2.0},
            {"start": 5.5, "end": 6.25},
            {"start": 10.0, "end": 30.0},
            {"start": 44.0, "end": 44.5},
            {"start": 50.0, "end": 51.0},
        ]
        expression = render.cut_expression(cuts)
        for step in range(0, 6000):
            t = step / 100.0
            expected = any(s <= t <= e for s, e in
                           ((c["start"], c["end"]) for c in cuts))
            self.assertEqual(
                bool(eval_expr(expression, t)), expected, f"disagreement at t={t}"
            )

    def test_cut_expression_scales_logarithmically(self):
        cuts = [{"start": i * 10.0, "end": i * 10.0 + 1.0} for i in range(400)]
        expression = render.cut_expression(cuts)
        # A flat sum would nest 400 terms deep on every evaluation; the tree
        # keeps the per-frame comparison count near log2(n).
        self.assertLess(expression.count("if("), 400)
        for t in (0.5, 9.5, 1000.5, 3995.5, 3999.9):
            expected = any(
                c["start"] <= t <= c["end"] for c in cuts
            )
            self.assertEqual(bool(eval_expr(expression, t)), expected, f"t={t}")

    def test_single_cut(self):
        expression = render.cut_expression([{"start": 2.0, "end": 3.0}])
        self.assertEqual(eval_expr(expression, 1.0), 0.0)
        self.assertEqual(eval_expr(expression, 2.5), 1.0)
        self.assertEqual(eval_expr(expression, 4.0), 0.0)

    def test_mute_gain_is_zero_inside_and_one_outside(self):
        fade = 0.03
        mutes = [{"start": 10.0, "end": 10.5}, {"start": 20.0, "end": 21.0}]
        expression = render.mute_gain_expression(mutes, fade)
        for t in (0.0, 5.0, 9.9, 10.6, 19.0, 21.2, 100.0):
            self.assertAlmostEqual(eval_expr(expression, t), 1.0, places=6, msg=f"t={t}")
        for t in (10.0, 10.25, 10.5, 20.0, 20.5, 21.0):
            self.assertAlmostEqual(eval_expr(expression, t), 0.0, places=6, msg=f"t={t}")

    def test_mute_gain_ramps_monotonically(self):
        fade = 0.05
        expression = render.mute_gain_expression([{"start": 10.0, "end": 11.0}], fade)
        # Fade in over [9.95, 10.0]: gain falls 1 -> 0.
        self.assertAlmostEqual(eval_expr(expression, 9.975), 0.5, places=3)
        self.assertAlmostEqual(eval_expr(expression, 11.025), 0.5, places=3)
        previous = 1.1
        for step in range(0, 51):
            value = eval_expr(expression, 9.95 + step * 0.001)
            self.assertLessEqual(value, previous + 1e-9)
            previous = value

    def test_mute_gain_never_leaves_zero_to_one(self):
        mutes = [{"start": i * 1.0, "end": i * 1.0 + 0.2} for i in range(50)]
        expression = render.mute_gain_expression(mutes, 0.03)
        for step in range(0, 5200):
            value = eval_expr(expression, step / 100.0)
            self.assertGreaterEqual(value, -1e-9)
            self.assertLessEqual(value, 1.0 + 1e-9)

    def test_build_filter_passthrough(self):
        self.assertIsNone(render.build_filter([], [], 512, 0.03))

    def test_build_filter_shapes(self):
        cuts = [{"start": 1.0, "end": 2.0}]
        mutes = [{"start": 5.0, "end": 5.5}]

        only_cuts = render.build_filter(cuts, [], 512, 0.03)
        self.assertIn("asetnsamples=n=512:p=0", only_cuts)
        self.assertIn("aselect=", only_cuts)
        self.assertIn("asetpts=N/SR/TB", only_cuts)
        self.assertNotIn("volume=", only_cuts)

        only_mutes = render.build_filter([], mutes, 512, 0.03)
        self.assertIn("volume=", only_mutes)
        self.assertNotIn("aselect=", only_mutes)
        self.assertNotIn("asetpts", only_mutes)

        both = render.build_filter(cuts, mutes, 256, 0.03)
        self.assertTrue(both.startswith("[0:a]"))
        self.assertTrue(both.endswith("[out]"))
        self.assertLess(both.index("volume="), both.index("aselect="))
        self.assertEqual(both.count("\n"), 0, "filter script must be a single line")


class TestExpectedDuration(unittest.TestCase):
    def test_no_cuts_is_the_full_length(self):
        self.assertEqual(
            render.expected_output_samples(10.0, 48000, 512, []), 480000
        )

    def test_frame_exact_prediction(self):
        rate, frame = 48000, 512
        cuts = [{"start": 2.0, "end": 3.0}]
        samples = render.expected_output_samples(10.0, rate, frame, cuts)
        # Frames whose start timestamp lands in [2, 3] are dropped, so the
        # result is quantised to whole frames rather than exactly 9s.
        self.assertLessEqual(samples, 9 * rate + frame)
        self.assertGreaterEqual(samples, 9 * rate - frame)

    def test_matches_a_brute_force_simulation(self):
        rate, frame = 44100, 1024
        duration = 37.5
        cuts = [
            {"start": 1.0, "end": 1.5},
            {"start": 12.3, "end": 15.75},
            {"start": 30.0, "end": 30.2},
        ]
        expression = render.cut_expression(cuts)
        total = int(round(duration * rate))
        brute = 0
        frame_index = 0
        while frame_index * frame < total:
            offset = frame_index * frame
            length = min(frame, total - offset)
            if not eval_expr(expression, offset / rate):
                brute += length
            frame_index += 1
        self.assertEqual(
            render.expected_output_samples(duration, rate, frame, cuts), brute
        )

    def test_quantisation_stays_within_one_frame_per_cut(self):
        """Frame granularity is the only source of error, and it is bounded.

        Whether a given cut rounds up or down depends on where it happens to
        fall relative to the frame grid, so a smaller frame is not guaranteed to
        be closer on any *particular* cut list — but the worst case shrinks in
        proportion to the frame size, and that bound is what has to hold.
        """
        rate = 48000
        cuts = [{"start": i * 5.0 + 1.0, "end": i * 5.0 + 1.4} for i in range(20)]
        ideal = (100.0 - 20 * 0.4) * rate
        for frame in (256, 512, 1024, 4096):
            samples = render.expected_output_samples(100.0, rate, frame, cuts)
            self.assertLessEqual(
                abs(samples - ideal),
                len(cuts) * frame,
                f"frame={frame} drifted by more than one frame per cut",
            )

    def test_prediction_depends_only_on_rate_frame_and_cuts(self):
        """The guarantee that keeps tracks in sync with each other.

        Two tracks of an episode differ in channel count and bit depth but not
        in sample rate, so they drop identical frames and stay aligned.
        """
        cuts = [{"start": 3.3, "end": 4.7}, {"start": 11.0, "end": 11.9}]
        first = render.expected_output_samples(60.0, 48000, 512, cuts)
        second = render.expected_output_samples(60.0, 48000, 512, cuts)
        self.assertEqual(first, second)
        # A different frame size would break that alignment, hence one setting
        # for the whole episode.
        self.assertNotEqual(
            first, render.expected_output_samples(60.0, 48000, 4096, cuts)
        )


class TestFinalTranscript(unittest.TestCase):
    def _plan(self):
        return {
            "episode_id": "test",
            "duration": 30.0,
            "participants": ["a", "b"],
            "cuts": [{"start": 10.0, "end": 20.0, "reasons": ["silence"],
                      "sources": ["vad"], "details": []}],
            "mutes": {"a": [{"start": 3.0, "end": 4.0, "reasons": ["stutter"],
                             "text": "I I"}], "b": []},
            "keep": [[0.0, 10.0], [20.0, 30.0]],
            "stats": {},
            "warnings": [],
        }

    def _transcripts(self):
        def parsed(name, words):
            return {
                "participant": name,
                "language": "en",
                "words": [
                    {"i": index, "text": text, "start": start, "end": end,
                     "segment": 0}
                    for index, (text, start, end) in enumerate(words)
                ],
                "segments": [
                    {"i": 0, "start": words[0][1], "end": words[-1][2],
                     "text": " ".join(w[0] for w in words),
                     "first_word": 0, "last_word": len(words) - 1}
                ],
            }

        return {
            "a": parsed("a", [("I", 3.0, 3.4), ("I", 3.5, 3.9),
                              ("think", 5.0, 5.5), ("so", 5.6, 6.0)]),
            "b": parsed("b", [("dropped", 12.0, 13.0)]),
        }

    def test_muted_words_leave_the_transcript(self):
        result = render.build_transcript(self._plan(), self._transcripts())
        texts = [entry["text"] for entry in result["segments"] if entry["participant"] == "a"]
        self.assertEqual(texts, ["think so"])
        self.assertEqual(result["removed_words"], 3)  # two muted, one cut

    def test_cut_segments_disappear_entirely(self):
        result = render.build_transcript(self._plan(), self._transcripts())
        self.assertEqual(
            [e for e in result["segments"] if e["participant"] == "b"], []
        )

    def test_timestamps_are_on_the_rendered_timeline(self):
        plan = self._plan()
        transcripts = self._transcripts()
        transcripts["b"] = {
            "participant": "b", "language": "en",
            "words": [{"i": 0, "text": "later", "start": 25.0, "end": 25.5,
                       "segment": 0}],
            "segments": [{"i": 0, "start": 25.0, "end": 25.5, "text": "later",
                          "first_word": 0, "last_word": 0}],
        }
        result = render.build_transcript(plan, transcripts)
        later = next(e for e in result["segments"] if e["text"] == "later")
        # 25s original, minus the 10s cut, is 15s in the rendered file.
        self.assertAlmostEqual(later["start"], 15.0)
        self.assertAlmostEqual(result["duration"], 20.0)

    def test_srt_and_text_render(self):
        result = render.build_transcript(self._plan(), self._transcripts())
        srt = render.transcript_to_srt(result)
        self.assertIn("-->", srt)
        self.assertIn("00:00:0", srt)
        text = render.transcript_to_text(result)
        self.assertIn("think so", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
