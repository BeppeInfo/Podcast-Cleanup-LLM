#!/usr/bin/env python3
"""Unit tests for the decision-making half of the pipeline.

Stdlib only, no pytest:  python3 tests/test_pipeline.py

The interesting ones are `TestFilterExpressions`, which evaluate the generated
ffmpeg expressions with a miniature interpreter of the exact grammar we emit.
A silently wrong expression would mangle the audio without failing anything, so
it gets checked rather than eyeballed.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import unittest.mock
import wave

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

from cleanup import intervals as iv
from cleanup import runlog
from cleanup import config as cfg, discover, llm, pipeline, proc  # noqa: E501
from cleanup import plan as planner, render, silence, transcript as tr
from cleanup import whisperx_asr as wx
# The CLI itself, for the exit codes the shell branches on. Importing is safe:
# it only runs main() under __main__.
import cleanup_cli as cli

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
            silence.parse_silencedetect(log, 10.0),
            [(0.0, 1.5), (3.0, 8.0), (9.25, 10.0)],
        )

    def test_unterminated_trailing_silence(self):
        """The draft's zip() approach mispaired everything after this case."""
        log = "silence_start: 2.0\nsilence_end: 4.0\nsilence_start: 9.0\n"
        self.assertEqual(
            silence.parse_silencedetect(log, 10.0), [(0.0, 2.0), (4.0, 9.0)]
        )

    def test_leading_silence(self):
        log = "silence_start: 0\nsilence_end: 2.5\n"
        self.assertEqual(silence.parse_silencedetect(log, 6.0), [(2.5, 6.0)])

    def test_no_silence_at_all(self):
        self.assertEqual(silence.parse_silencedetect("", 5.0), [(0.0, 5.0)])


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
        parsed = tr.build_from_segments(
            payload["transcription"], "alice", payload["result"]["language"]
        )
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
        parsed = tr.build_from_segments(payload["transcription"], "bob")
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


class _StubLlamaServer:
    """A stand-in for llama-server, to exercise the real HTTP client.

    Serves /health and /completion, hands out canned replies in order, and
    records the requests it received so the payload contract can be checked.
    """

    def __init__(self, replies, health_status=200, api_key=None, responder=None,
                 auth_delay=0.0):
        import http.server
        import threading

        self.replies = list(replies)
        self.requests = []
        self.paths = []
        self.seen_auth = []
        # Canned replies are handed out in arrival order, which says nothing
        # useful once requests overlap. A responder answers from the request
        # itself instead, so a reply belongs to its window however they race.
        self.responder = responder
        # How many POSTs were being served at the same moment, at the busiest
        # point. The only direct evidence that concurrency reached the wire.
        self.in_flight = 0
        self.peak_in_flight = 0
        self._counter_lock = threading.Lock()
        # Set to send a hand-built envelope instead of the usual wrapping, for
        # the response shapes only some servers produce.
        self.raw_reply = None
        self.raw_status = 200
        # Model ids returned from /v1/models, for the router-mode error path.
        self.models = []
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

            def _authorised(self):
                outer.seen_auth.append(self.headers.get("Authorization"))
                if not api_key:
                    return True
                if self.headers.get("Authorization") == f"Bearer {api_key}":
                    return True
                # A refusal that takes a moment, so a test measuring how many
                # requests escape before the client gives up is measuring the
                # client's reaction and not the scheduler's mood.
                if auth_delay:
                    time.sleep(auth_delay)
                length = int(self.headers.get("Content-Length", 0))
                if length:
                    self.rfile.read(length)
                self._send(401, {"error": "invalid api key"})
                return False

            def do_GET(self):
                if not self._authorised():
                    return
                if self.path == "/health":
                    self._send(health_status, {"status": "ok"})
                elif self.path == "/v1/models":
                    self._send(200, {"data": [{"id": m} for m in outer.models]})
                else:
                    self._send(404, {})

            def do_POST(self):
                if not self._authorised():
                    return
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length))
                outer.requests.append(payload)
                outer.paths.append(self.path)
                with outer._counter_lock:
                    outer.in_flight += 1
                    outer.peak_in_flight = max(
                        outer.peak_in_flight, outer.in_flight
                    )
                try:
                    self._reply_to(payload)
                finally:
                    with outer._counter_lock:
                        outer.in_flight -= 1

            def _reply_to(self, payload):
                if outer.raw_reply is not None:
                    self._send(outer.raw_status, outer.raw_reply)
                    return
                if outer.responder is not None:
                    reply = outer.responder(payload)
                else:
                    reply = (
                        outer.replies.pop(0) if outer.replies else {"edits": []}
                    )
                if reply is None:
                    self._send(500, {"error": "boom"})
                    return
                text = reply if isinstance(reply, str) else json.dumps(reply)
                # Reply in the envelope matching the endpoint that was called,
                # so a client posting to the wrong one gets nothing usable.
                if self.path.startswith("/v1/chat/completions"):
                    self._send(200, {
                        "choices": [{"message": {"role": "assistant",
                                                 "content": text}}]
                    })
                else:
                    self._send(200, {"content": text})

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
        # The model asked for both copies of "eu". Only the first is taken: the
        # sentence still needs one, and it is the last. The reply is recorded
        # as it arrived, so the trim is visible rather than silent.
        self.assertEqual(edit["text"], "eu")
        self.assertEqual(edit["spared"], "eu")
        self.assertAlmostEqual(edit["start"], 0.0)
        self.assertAlmostEqual(edit["end"], 0.4)
        self.assertEqual(result["chunk_failures"], 0)

    def test_request_payload_carries_the_schema(self):
        parsed = self._words(["um", "teste"])
        _, server = self._detect(parsed, [{"edits": []}])
        self.assertEqual(len(server.requests), 1)
        payload = server.requests[0]
        self.assertEqual(payload["temperature"], 0.0)
        self.assertTrue(payload["cache_prompt"])

        # Both spellings of the constraint are sent, since llama.cpp builds have
        # read one or the other; each must carry the same schema.
        response_format = payload["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        for schema in (response_format["json_schema"]["schema"],
                       response_format["schema"]):
            kinds = schema["properties"]["edits"]["items"]["properties"]["kind"]["enum"]
            # Only the kinds we accept are offered to the model.
            self.assertEqual(kinds, ["stutter", "repetition"])

        # The transcript travels as a chat message, not a raw prompt: that is
        # what lets the server apply the loaded model's own template.
        self.assertNotIn("prompt", payload)
        self.assertIn("teste", payload["messages"][0]["content"])
        self.assertEqual(payload["messages"][0]["role"], "user")

    def test_completion_escape_hatch_keeps_the_old_request_shape(self):
        """LLM_API=completion must reach the raw endpoint, unchanged.

        It exists for a build without the chat endpoint, so it has to keep
        working exactly as it did rather than quietly becoming a chat call.
        """
        server = _StubLlamaServer([{"edits": []}])
        self.addCleanup(server.close)
        client = llm.LlamaClient(server.endpoint, timeout=10, api="completion")
        client.complete("hi", llm.response_schema(["stutter"]))

        payload = server.requests[0]
        self.assertEqual(server.paths, ["/completion"])
        self.assertIn("json_schema", payload)
        self.assertIn("prompt", payload)
        self.assertNotIn("messages", payload)
        self.assertNotIn("response_format", payload)

    def test_reply_token_ceiling_is_configurable(self):
        server = _StubLlamaServer([{"edits": []}])
        self.addCleanup(server.close)
        client = llm.LlamaClient(server.endpoint, timeout=10, max_reply_tokens=64)
        client.complete("hi", llm.response_schema(["stutter"]))
        self.assertEqual(server.requests[0]["max_tokens"], 64)

    def test_model_name_is_sent_when_configured(self):
        """A server in router mode refuses a request that names no model."""
        server = _StubLlamaServer([{"edits": []}])
        self.addCleanup(server.close)
        client = llm.LlamaClient(server.endpoint, timeout=10, model="qwen-x")
        client.complete("hi", llm.response_schema(["stutter"]))
        self.assertEqual(server.requests[0]["model"], "qwen-x")

    def test_model_name_is_omitted_when_not_configured(self):
        """A single-model server has nothing to route, so nothing is claimed."""
        server = _StubLlamaServer([{"edits": []}])
        self.addCleanup(server.close)
        client = llm.LlamaClient(server.endpoint, timeout=10)
        client.complete("hi", llm.response_schema(["stutter"]))
        self.assertNotIn("model", server.requests[0])

    def test_a_refusal_carries_the_servers_own_words(self):
        """"HTTP Error 400: Bad Request" alone sends you reading code.

        The example used to be "model is not loaded"; that one now raises
        ModelUnavailable, because it condemns every window rather than this one.
        A context overflow is the genuinely per-chunk kind of refusal.
        """
        server = _StubLlamaServer([])
        self.addCleanup(server.close)
        server.raw_status = 400
        server.raw_reply = {"error": {"message": "the prompt is too long"}}
        client = llm.LlamaClient(server.endpoint, timeout=10)
        with self.assertRaises(llm.EndpointError) as caught:
            client.complete("hi", llm.response_schema(["stutter"]))
        self.assertIn("the prompt is too long", str(caught.exception))

    def test_the_unloaded_model_refusal_also_quotes_the_server(self):
        server = _StubLlamaServer([])
        self.addCleanup(server.close)
        server.raw_status = 400
        server.raw_reply = {"error": {"message": "model is not loaded"}}
        client = llm.LlamaClient(server.endpoint, timeout=10)
        with self.assertRaises(llm.ModelUnavailable) as caught:
            client.complete("hi", llm.response_schema(["stutter"]))
        self.assertIn("model is not loaded", str(caught.exception))

    def test_a_missing_model_name_is_explained_with_the_choices(self):
        server = _StubLlamaServer([])
        self.addCleanup(server.close)
        server.raw_status = 400
        server.raw_reply = {"error": {"message": "model name is missing"}}
        server.models = ["qwen-a", "gemma-b"]
        client = llm.LlamaClient(server.endpoint, timeout=10)
        with self.assertRaises(llm.SchemaIgnored) as caught:
            client.check_schema_support()
        message = str(caught.exception)
        self.assertIn("LLAMA_MODEL_NAME", message)
        self.assertIn("qwen-a", message)
        self.assertIn("gemma-b", message)

    def test_an_unknown_api_is_refused_at_construction(self):
        with self.assertRaises(ValueError):
            llm.LlamaClient("http://localhost:1", api="grpc")

    def test_reasoning_content_is_read_when_content_is_empty(self):
        """A reasoning model with reasoning_format=none leaves content empty.

        The schema still constrained what it generated, so the JSON is in
        reasoning_content and is worth reading rather than discarding.
        """
        server = _StubLlamaServer([])
        self.addCleanup(server.close)
        server.raw_reply = {
            "choices": [{"message": {
                "content": "",
                "reasoning_content": '{"edits": []}',
            }}]
        }
        client = llm.LlamaClient(server.endpoint, timeout=10)
        content = client.complete("hi", llm.response_schema(["stutter"]))
        self.assertEqual(json.loads(content), {"edits": []})

    # --- the startup schema check --------------------------------------------

    def test_schema_check_passes_against_a_constraining_server(self):
        server = _StubLlamaServer([{"edits": []}])
        self.addCleanup(server.close)
        client = llm.LlamaClient(server.endpoint, timeout=10)
        client.check_schema_support()          # must not raise
        self.assertEqual(len(server.requests), 1)

    def test_schema_check_catches_a_server_that_answers_prose(self):
        """The failure this exists to prevent is a silent one.

        A server that ignores response_format returns prose, every chunk fails
        to parse and is dropped, and the episode reports no edits — which looks
        exactly like clean speech.
        """
        server = _StubLlamaServer(["Certainly! Here are the edits you asked for."])
        self.addCleanup(server.close)
        client = llm.LlamaClient(server.endpoint, timeout=10)
        with self.assertRaises(llm.SchemaIgnored) as caught:
            client.check_schema_support()
        self.assertIn("LLM_API=completion", str(caught.exception))

    def test_schema_check_catches_valid_json_of_the_wrong_shape(self):
        server = _StubLlamaServer([{"something_else": 1}])
        self.addCleanup(server.close)
        client = llm.LlamaClient(server.endpoint, timeout=10)
        with self.assertRaises(llm.SchemaIgnored):
            client.check_schema_support()

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
        self.assertIsNone(lines[0]["error"])

    def test_audit_says_outright_when_a_chunk_failed(self):
        """A failed window and a clean one must not read alike.

        A real episode had every window of a track time out, and the audit was
        read as the model finding nothing to cut. Both write empty `raw`, zero
        accepted and nothing rejected; they differed only in `content`, which is
        null for a failure *and* null whenever edits were kept — overloaded
        enough that the reader guessed wrong. `error` states it instead.
        """
        parsed = self._words(["eu", "eu", "acho"])

        audit = os.path.join(tempfile.mkdtemp(), "audit.jsonl")
        # `None` makes the stub answer HTTP 500, so the chunk exhausts its
        # retries and is dropped exactly as a timeout would be.
        self._detect(parsed, [None, None, None], audit_path=audit, retries=2)
        with open(audit, encoding="utf-8") as handle:
            failed = json.loads(handle.readline())

        clean_audit = os.path.join(tempfile.mkdtemp(), "audit.jsonl")
        self._detect(parsed, [{"edits": []}], audit_path=clean_audit)
        with open(clean_audit, encoding="utf-8") as handle:
            clean = json.loads(handle.readline())

        # Everything describing the outcome agrees...
        for field in ("raw", "accepted", "rejected"):
            self.assertEqual(failed[field], clean[field], field)
        # ...so `error` has to carry the difference, and carry the reason.
        self.assertIsNone(clean["error"])
        self.assertIn("500", failed["error"])

        # The overload that caused the misreading: `content` is null for a
        # failure and also null for a chunk whose edits were all accepted, so it
        # cannot be used to tell the two apart.
        kept_audit = os.path.join(tempfile.mkdtemp(), "audit.jsonl")
        self._detect(
            parsed,
            [{"edits": [{"first": 0, "last": 1, "kind": "stutter",
                         "confidence": 0.9}]}],
            audit_path=kept_audit,
        )
        with open(kept_audit, encoding="utf-8") as handle:
            kept = json.loads(handle.readline())
        self.assertIsNone(kept["content"])
        self.assertIsNone(failed["content"])
        self.assertIsNone(kept["error"])

    def test_every_chunk_failing_is_not_reported_as_success(self):
        """A track nothing could be learned about must not exit 0.

        Dropping the odd window is deliberate tolerance; dropping all of them
        means the track was never analysed, and exiting 0 would let the shell
        mark it done, skip it on resume, and publish a report showing zero edits
        — which reads exactly like clean speech. Invariant 10.
        """
        import io, contextlib
        parsed = self._words(["eu", "eu", "acho", "que", "sim"])
        server = _StubLlamaServer([None] * 12)   # every attempt answers HTTP 500
        self.addCleanup(server.close)

        out = os.path.join(tempfile.mkdtemp(), "edits.json")
        words = os.path.join(tempfile.mkdtemp(), "words.json")
        with open(words, "w", encoding="utf-8") as handle:
            json.dump(parsed, handle)

        argv = ["detect", "--words", words, "--endpoint", server.endpoint,
                "--out", out, "--chunk-words", "2", "--overlap", "0",
                "--kinds", "stutter"]
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            with self.assertRaises(SystemExit) as caught:
                cli.main(argv)
        self.assertEqual(caught.exception.code, 4)
        self.assertIn("every chunk failed", buffer.getvalue())

        # The edits file is still written, so the rest of the run can proceed
        # with this track simply keeping its disfluencies.
        with open(out, encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["edits"], [])

    def test_some_chunks_failing_is_still_a_success(self):
        """Partial loss is the tolerated case and must stay exit 0."""
        import io, contextlib
        parsed = self._words(["eu", "eu", "acho", "que", "sim"])
        # First window answers, the rest fail: a survivable partial result.
        server = _StubLlamaServer(
            [{"edits": [{"first": 0, "last": 1, "kind": "stutter",
                         "confidence": 0.9}]}] + [None] * 12
        )
        self.addCleanup(server.close)
        out = os.path.join(tempfile.mkdtemp(), "edits.json")
        words = os.path.join(tempfile.mkdtemp(), "words.json")
        with open(words, "w", encoding="utf-8") as handle:
            json.dump(parsed, handle)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(
                cli.main(["detect", "--words", words, "--endpoint",
                          server.endpoint, "--out", out, "--chunk-words", "2",
                          "--overlap", "0", "--kinds", "stutter"]),
                0,
            )

    def _truncated_reply(self, count, cut_mid_object=True):
        """A reply of `count` complete edits, then cut off part-way through one.

        Shaped like what llama.cpp actually returned: pretty-printed, and the
        cut landing inside an object rather than neatly between them.
        """
        body = ",\n".join(
            '    {\n      "first": %d,\n      "last": %d,\n'
            '      "kind": "repetition",\n      "confidence": 0.99\n    }'
            % (index * 2, index * 2 + 1)
            for index in range(count)
        )
        text = '{\n  "edits": [\n' + body
        return text + ',\n    {\n      "first": 998,\n      "las' if cut_mid_object else text

    def test_a_reply_cut_off_at_the_token_ceiling_keeps_what_arrived(self):
        """45 real findings should not be lost to a missing closing bracket.

        This is the shape of an actual failure: the model emitted dozens of
        edits, ran into max_reply_tokens mid-object, and the whole window was
        discarded because the JSON would not parse.
        """
        parsed = self._words([f"w{index}" for index in range(40)])
        server = _StubLlamaServer([])
        self.addCleanup(server.close)
        server.raw_reply = {
            "choices": [{
                "message": {"role": "assistant",
                            "content": self._truncated_reply(12)},
                "finish_reason": "length",
            }]
        }
        client = llm.LlamaClient(server.endpoint, timeout=10)
        audit = os.path.join(tempfile.mkdtemp(), "audit.jsonl")
        result = llm.detect(
            client, parsed, chunk_words=40, overlap=0,
            limits={"max_words": 4, "max_seconds": 3.0, "min_confidence": 0.6},
            accepted=["stutter", "repetition"], audit_path=audit,
        )
        # All 12 complete edits survive; only the 13th, cut in half, is lost.
        self.assertEqual(len(result["edits"]), 12)
        # Truncation is not a failure — the edits are as valid as any other —
        # but it is reported, because the rest of the window went unjudged.
        self.assertEqual(result["chunk_failures"], 0)
        self.assertEqual(result["chunks_truncated"], 1)

        with open(audit, encoding="utf-8") as handle:
            record = json.loads(handle.readline())
        self.assertTrue(record["truncated"])
        self.assertEqual(record["salvaged"], 12)
        self.assertIsNone(record["error"])

    def test_truncation_is_not_retried(self):
        """Deterministic decoding means a retry reproduces it exactly.

        Retrying cost three full 2048-token generations per window on a real
        episode, for three identical truncated replies.
        """
        parsed = self._words([f"w{index}" for index in range(40)])
        server = _StubLlamaServer([])
        self.addCleanup(server.close)
        server.raw_reply = {
            "choices": [{"message": {"content": self._truncated_reply(3)},
                         "finish_reason": "length"}]
        }
        client = llm.LlamaClient(server.endpoint, timeout=10)
        llm.detect(
            client, parsed, chunk_words=40, overlap=0,
            limits={"max_words": 4, "max_seconds": 3.0, "min_confidence": 0.6},
            accepted=["stutter", "repetition"], retries=2,
        )
        self.assertEqual(len(server.requests), 1)

    def test_truncation_with_nothing_salvageable_is_a_failure(self):
        parsed = self._words([f"w{index}" for index in range(40)])
        server = _StubLlamaServer([])
        self.addCleanup(server.close)
        server.raw_reply = {
            "choices": [{"message": {"content": '{\n  "edits": [\n    {\n  "fir'},
                         "finish_reason": "length"}]
        }
        client = llm.LlamaClient(server.endpoint, timeout=10)
        result = llm.detect(
            client, parsed, chunk_words=40, overlap=0,
            limits={"max_words": 4, "max_seconds": 3.0, "min_confidence": 0.6},
            accepted=["stutter", "repetition"],
        )
        self.assertEqual(result["chunk_failures"], 1)
        self.assertEqual(result["edits"], [])

    def test_salvage_handles_the_shapes_a_cut_can_land_on(self):
        whole = self._truncated_reply(3, cut_mid_object=False)
        self.assertEqual(len(llm._salvage_edits(whole)), 3)
        self.assertEqual(len(llm._salvage_edits(self._truncated_reply(3))), 3)
        # Cut exactly on the separator, and cut before any object closes.
        self.assertEqual(len(llm._salvage_edits(whole + ",")), 3)
        self.assertEqual(llm._salvage_edits('{"edits": ['), [])
        self.assertEqual(llm._salvage_edits(""), [])
        self.assertEqual(llm._salvage_edits("not json at all"), [])
        # A complete, well-formed reply salvages to exactly its own contents.
        self.assertEqual(
            llm._salvage_edits(json.dumps({"edits": [{"first": 1, "last": 2}]})),
            [{"first": 1, "last": 2}],
        )

    def test_a_server_with_no_model_ends_the_run(self):
        """The model is the server's state, so one window's 400 condemns all.

        Seen for real: the router unloaded the model while transcribe was
        running, and detect burned a whole stage rediscovering that per window.
        """
        parsed = self._words([f"w{index}" for index in range(40)])
        server = _StubLlamaServer([])
        self.addCleanup(server.close)
        server.raw_reply = {"error": {"message": "model is not loaded"}}
        server.raw_status = 400
        client = llm.LlamaClient(server.endpoint, timeout=10, model="qwen-podcast")
        with self.assertRaises(llm.ModelUnavailable) as caught:
            llm.detect(
                client, parsed, chunk_words=4, overlap=0,
                limits={"max_words": 4, "max_seconds": 3.0, "min_confidence": 0.6},
                accepted=["stutter"], retries=2, concurrency=2,
            )
        self.assertIn("qwen-podcast", str(caught.exception))
        # Not retried, and the queued windows never go out: one or two requests
        # for the workers already in flight, not one per window.
        self.assertLessEqual(len(server.requests), 2)

    def test_it_exits_5_so_the_shell_can_tell_it_apart(self):
        import io, contextlib
        parsed = self._words([f"w{index}" for index in range(8)])
        server = _StubLlamaServer([])
        self.addCleanup(server.close)
        server.raw_reply = {"error": {"message": "model is not loaded"}}
        server.raw_status = 400
        words = os.path.join(tempfile.mkdtemp(), "words.json")
        with open(words, "w", encoding="utf-8") as handle:
            json.dump(parsed, handle)
        buffer = io.StringIO()
        with contextlib.redirect_stderr(buffer):
            code = cli.main(["detect", "--words", words, "--endpoint",
                             server.endpoint, "--out",
                             os.path.join(tempfile.mkdtemp(), "e.json"),
                             "--kinds", "stutter"])
        self.assertEqual(code, 5)
        self.assertIn("not serving", buffer.getvalue())

    def test_an_ordinary_400_is_still_only_that_chunk(self):
        """Narrow matching matters: a context overflow must stay survivable."""
        parsed = self._words([f"w{index}" for index in range(8)])
        server = _StubLlamaServer([])
        self.addCleanup(server.close)
        server.raw_reply = {"error": {"message": "the prompt is too long"}}
        server.raw_status = 400
        client = llm.LlamaClient(server.endpoint, timeout=10)
        result = llm.detect(
            client, parsed, chunk_words=8, overlap=0,
            limits={"max_words": 4, "max_seconds": 3.0, "min_confidence": 0.6},
            accepted=["stutter"], retries=0,
        )
        self.assertEqual(result["chunk_failures"], 1)

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


class TestDetectConcurrency(unittest.TestCase):
    """Several windows in flight at once, against a --parallel llama-server.

    The claim being tested is that this is a throughput change and nothing
    else. Chunks are planned up front and reconciled by index afterwards, so
    the wire order genuinely does not matter — but only if nothing downstream
    quietly depends on it. These pin that down, because the failure mode is not
    a crash: it is an audit file or an edit list that shuffles between runs and
    can no longer be compared against the previous one.
    """

    WINDOW = re.compile(r"they start at (\d+) and end at (\d+)")

    def _words(self, count):
        return {
            "participant": "alice",
            "language": "pt",
            "words": [
                {"i": index, "text": f"w{index}", "start": index * 0.5,
                 "end": index * 0.5 + 0.4, "segment": 0}
                for index in range(count)
            ],
            "segments": [],
        }

    def _run(self, concurrency, *, delay=None, audit=None, words=350,
             api_key=None, client_key=None, auth_delay=0.0):
        """Detect over `words` words, one edit reported per window.

        `delay` is called with the window's first index and returns how long
        that reply should take, so a test can decide the order the answers come
        back in — including the reverse of the order they were asked for.
        """
        def responder(payload):
            first, _ = self.WINDOW.search(
                payload["messages"][0]["content"]
            ).groups()
            first = int(first)
            if delay:
                time.sleep(delay(first))
            return {"edits": [{"first": first, "last": first + 1,
                               "kind": "repetition", "confidence": 0.9}]}

        server = _StubLlamaServer([], responder=responder, api_key=api_key,
                                  auth_delay=auth_delay)
        self.addCleanup(server.close)
        # Kept for the tests whose subject is a run that raises before
        # returning, and so never gets the server back the usual way.
        self.server = server
        client = llm.LlamaClient(server.endpoint, timeout=10, api_key=client_key)
        result = llm.detect(
            client,
            self._words(words),
            chunk_words=100,
            overlap=10,
            limits={"max_words": 4, "max_seconds": 3.0, "min_confidence": 0.6},
            accepted=["stutter", "repetition"],
            audit_path=audit,
            concurrency=concurrency,
        )
        return result, server

    def _audit(self):
        return os.path.join(tempfile.mkdtemp(), "audit.jsonl")

    def test_parallel_run_matches_the_sequential_one_exactly(self):
        sequential_audit, parallel_audit = self._audit(), self._audit()
        sequential, _ = self._run(1, audit=sequential_audit)
        parallel, _ = self._run(4, audit=parallel_audit)

        # The fixture has to span several windows, or this proves nothing.
        self.assertEqual(sequential["chunks"], 4)
        self.assertGreater(len(sequential["edits"]), 1)

        self.assertEqual(sequential, parallel)
        with open(sequential_audit) as handle:
            first_run = handle.read()
        with open(parallel_audit) as handle:
            self.assertEqual(first_run, handle.read())

    def test_windows_really_do_overlap_on_the_wire(self):
        _, server = self._run(4, delay=lambda first: 0.05)
        self.assertGreater(server.peak_in_flight, 1)

    def test_the_default_keeps_one_request_in_flight(self):
        _, server = self._run(1, delay=lambda first: 0.02)
        self.assertEqual(server.peak_in_flight, 1)

    def test_answers_arriving_backwards_still_record_in_chunk_order(self):
        """The audit is a diffable record, so its order cannot be a race.

        Chunk 0 is made the slowest, so the replies finish in the reverse of
        the order they were asked for.
        """
        audit = self._audit()
        self._run(4, delay=lambda first: max(0.0, 0.2 - first * 0.0005),
                  audit=audit)
        with open(audit) as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        self.assertEqual([r["chunk"] for r in records], [0, 1, 2, 3])
        self.assertEqual([r["words"][0] for r in records], [0, 90, 180, 270])

    def test_a_refused_key_stops_the_track_without_sending_every_window(self):
        """A bad key must not become hundreds of doomed requests.

        Sequentially this cost one wasted request. Every window is submitted to
        the pool up front now, so without cancelling the queue, giving up would
        still work through all of it first.
        """
        total = len(llm.plan_chunks(4000, 100, 10))
        self.assertGreater(total, 20)
        with self.assertRaises(llm.AuthRejected):
            self._run(4, words=4000, api_key="right", client_key="wrong",
                      auth_delay=0.05)
        self.assertLess(len(self.server.seen_auth), total)

    def test_more_workers_than_windows_is_harmless(self):
        result, server = self._run(16, words=50)
        self.assertEqual(result["chunks"], 1)
        self.assertEqual(len(server.requests), 1)


class TestApiKeyAuth(unittest.TestCase):
    """Authentication for both remote clients.

    The properties that matter: the header is exactly what llama-server's
    --api-key expects, a refusal is reported rather than retried into silence,
    and the key never reaches a file we keep.
    """

    KEY = "sk-test-abc123"

    def _words(self, texts=("eu", "eu", "acho")):
        return {
            "participant": "alice",
            "language": "pt",
            "words": [
                {"i": i, "text": t, "start": i * 0.5, "end": i * 0.5 + 0.4,
                 "segment": 0}
                for i, t in enumerate(texts)
            ],
            "segments": [],
        }

    def _server(self, replies, api_key=None):
        server = _StubLlamaServer(replies, api_key=api_key)
        self.addCleanup(server.close)
        return server

    # --- llama client ---------------------------------------------------------

    def test_bearer_header_is_sent(self):
        server = self._server([{"edits": []}], api_key=self.KEY)
        client = llm.LlamaClient(server.endpoint, timeout=10, api_key=self.KEY)
        self.assertTrue(client.wait_until_ready(timeout=5))
        client.complete("hi", llm.response_schema(["stutter"]))
        self.assertIn(f"Bearer {self.KEY}", server.seen_auth)

    def test_no_header_when_no_key_configured(self):
        server = self._server([{"edits": []}])
        client = llm.LlamaClient(server.endpoint, timeout=10)
        client.complete("hi", llm.response_schema(["stutter"]))
        self.assertEqual(server.seen_auth, [None])

    def test_missing_key_raises_auth_rejected(self):
        server = self._server([{"edits": []}], api_key=self.KEY)
        client = llm.LlamaClient(server.endpoint, timeout=10)  # no key
        with self.assertRaises(llm.AuthRejected) as caught:
            client.complete("hi", llm.response_schema(["stutter"]))
        message = str(caught.exception)
        self.assertIn("401", message)
        self.assertIn("LLAMA_API_KEY", message)

    def test_wrong_key_names_the_setting_to_check(self):
        server = self._server([{"edits": []}], api_key=self.KEY)
        client = llm.LlamaClient(server.endpoint, timeout=10, api_key="wrong")
        with self.assertRaises(llm.AuthRejected) as caught:
            client.complete("hi", llm.response_schema(["stutter"]))
        self.assertIn("--api-key", str(caught.exception))

    def test_health_check_fails_fast_rather_than_waiting_out_the_timeout(self):
        """A refused key is not a loading model; waiting cannot fix it."""
        import time

        server = self._server([], api_key=self.KEY)
        client = llm.LlamaClient(server.endpoint, api_key="wrong")
        started = time.monotonic()
        self.assertFalse(client.wait_until_ready(timeout=30, poll=2.0))
        self.assertLess(
            time.monotonic() - started, 5.0,
            "should give up at once on a 401, not poll for the full timeout",
        )

    def test_detection_aborts_instead_of_dropping_every_chunk(self):
        """The failure mode this guards: an episode that quietly finds nothing.

        detect() swallows most errors per chunk on purpose. A refused key must
        not be swallowed, or a misconfiguration would look like clean speech.
        """
        server = self._server([{"edits": []}], api_key=self.KEY)
        client = llm.LlamaClient(server.endpoint, timeout=10)  # no key
        with self.assertRaises(llm.AuthRejected):
            llm.detect(
                client, self._words(),
                chunk_words=2, overlap=0,
                limits={"max_words": 4, "max_seconds": 3.0, "min_confidence": 0.6},
                accepted=["stutter"], retries=2,
            )
        # One attempt, not three: an auth failure is never retried.
        self.assertEqual(len(server.seen_auth), 1)

    def test_key_is_not_written_to_the_audit_log(self):
        server = self._server(
            [{"edits": [{"first": 0, "last": 1, "kind": "stutter",
                         "confidence": 0.9}]}],
            api_key=self.KEY,
        )
        audit = os.path.join(tempfile.mkdtemp(), "audit.jsonl")
        client = llm.LlamaClient(server.endpoint, timeout=10, api_key=self.KEY)
        llm.detect(
            client, self._words(),
            chunk_words=100, overlap=10,
            limits={"max_words": 4, "max_seconds": 3.0, "min_confidence": 0.6},
            accepted=["stutter"], audit_path=audit,
        )
        with open(audit, encoding="utf-8") as handle:
            self.assertNotIn(self.KEY, handle.read())


PARAMS = {
    "silence_min_duration": 1.5,
    "silence_keep": 0.4,
    "edge_keep": 0.25,
    "cut_padding": 0.1,
    "min_cut": 0.15,
    "mute_fade": 0.03,
    "max_cut_fraction": 0.5,
    "speech_pad": 0.25,
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


class TestSpeechFromWords(unittest.TestCase):
    """The speech map, derived from the transcript rather than from the audio.

    Whisper runs Silero itself and only transcribes what it calls speech, so the
    words that come back carry that judgement and nothing here looks at the audio
    a second time. What this pins down is the padding, which is the only knob
    left: it decides how much of Whisper's timing error a cut may eat, and how
    long a gap has to be before it counts as silence at all.
    """

    def _words(self, spec):
        return [
            {"i": i, "text": t, "start": s, "end": e, "segment": 0}
            for i, (t, s, e) in enumerate(spec)
        ]

    def test_each_word_becomes_a_padded_span(self):
        speech = tr.speech_from_words(
            self._words([("hello", 5.0, 5.5)]), 20.0, pad=0.25
        )
        self.assertEqual(speech, [(4.75, 5.75)])

    def test_adjacent_words_fuse_into_one_region(self):
        speech = tr.speech_from_words(
            self._words([("one", 5.0, 5.4), ("two", 5.5, 5.9)]), 20.0, pad=0.25
        )
        self.assertEqual(speech, [(4.75, 6.15)])

    def test_padding_never_runs_past_the_track(self):
        speech = tr.speech_from_words(
            self._words([("edge", 0.1, 0.4), ("end", 19.8, 20.0)]), 20.0, pad=0.25
        )
        self.assertEqual(speech[0][0], 0.0)
        self.assertEqual(speech[-1][1], 20.0)

    def test_a_gap_must_clear_the_padding_on_both_sides(self):
        """Which is why the effective silence threshold is min_duration + 2*pad."""
        words = self._words([("before", 4.0, 5.0), ("after", 7.0, 8.0)])
        speech = tr.speech_from_words(words, 20.0, pad=0.25)
        gap = speech[1][0] - speech[0][1]
        self.assertAlmostEqual(gap, 2.0 - 0.5, places=6)

    def test_no_words_is_no_speech(self):
        self.assertEqual(tr.speech_from_words([], 20.0), [])
        self.assertEqual(tr.speech_from_words(None, 20.0), [])

    def test_unusable_timings_are_skipped_not_guessed(self):
        words = [
            {"i": 0, "text": "fine", "start": 1.0, "end": 2.0},
            {"i": 1, "text": "broken", "start": None, "end": 4.0},
            {"i": 2, "text": "missing"},
            {"i": 3, "text": "backwards", "start": 9.0, "end": 8.0},
        ]
        self.assertEqual(tr.speech_from_words(words, 20.0, pad=0.0), [(1.0, 2.0)])

    def test_an_interpolated_segment_tiles_its_whole_span(self):
        """So silence inside it is invisible — losing a cut, never inventing one."""
        segment = {
            "text": "one two three four",
            "offsets": {"from": 0, "to": 10000},
            "tokens": [{"text": " one", "offsets": {"from": 0, "to": 0}}],
        }
        parsed = tr.build_from_segments([segment], "a")
        self.assertEqual(parsed["approximated_segments"], 1)
        speech = tr.speech_from_words(parsed["words"], 10.0, pad=0.0)
        self.assertEqual(speech, [(0.0, 10.0)])


class TestLoopingTranscript(unittest.TestCase):
    """What replaces comparing a transcript against an independent speech map.

    There is no independent map any more: it is derived from the transcript, so
    the two agree by construction. The failure is still detectable in the
    transcript alone, because Whisper handed something that is not speech does
    not invent varied text — it repeats. On the episode behind this it produced
    one sentence 198 times across 4.6 minutes, 73% of that track.
    """

    def _meta(self, duration=600.0):
        return {
            "episode_id": "ep001",
            "duration": duration,
            "tracks": [{"participant": "host"}, {"participant": "guest"}],
        }

    def _params(self):
        return {
            "silence_min_duration": 1.5, "silence_keep": 0.4, "edge_keep": 0.25,
            "cut_padding": 0.1, "min_cut": 0.15, "mute_fade": 0.03,
            "max_cut_fraction": 0.5, "speech_pad": 0.25,
        }

    def _loop(self, repeats, start, step=0.2):
        """One phrase over and over, the way a looping Whisper emits it."""
        phrase = ["I", "lost", "my", "train", "of", "thought", "here", "again"]
        return [
            {"i": index, "text": phrase[index % len(phrase)],
             "start": start + index * step, "end": start + index * step + step * 0.8,
             "segment": 0}
            for index in range(repeats * len(phrase))
        ]

    def _speech(self, words):
        return {
            name: tr.speech_from_words(track, 600.0, pad=0.25)
            for name, track in words.items()
        }

    def test_a_looping_track_is_reported(self):
        words = {"host": self._loop(40, 100.0), "guest": []}
        plan = planner.build_plan(
            self._meta(), self._speech(words), {"host": [], "guest": []},
            words, self._params(),
        )
        warning = " ".join(plan["warnings"])
        self.assertIn("host's transcript repeats", warning)
        self.assertIn("i lost my train of thought here again", warning)
        self.assertIn("WHISPER_VAD_ONSET", warning)

        record = plan["looping_transcripts"]["host"]
        self.assertEqual(record["repeats"], 40)
        self.assertEqual(record["words"], 320)

    def test_the_real_shape_is_caught(self):
        """198 repetitions over 73% of the track, which is what happened."""
        words = {"host": self._loop(198, 60.0) + self._loop(20, 400.0), "guest": []}
        plan = planner.build_plan(
            self._meta(), self._speech(words), {"host": [], "guest": []},
            words, self._params(),
        )
        self.assertIn("host", plan["looping_transcripts"])

    def test_ordinary_speech_raises_nothing(self):
        words = {
            "host": [
                {"i": i, "text": f"word{i}", "start": 10.0 + i * 0.4,
                 "end": 10.3 + i * 0.4, "segment": 0}
                for i in range(200)
            ],
            "guest": [],
        }
        plan = planner.build_plan(
            self._meta(), self._speech(words), {"host": [], "guest": []},
            words, self._params(),
        )
        self.assertEqual(plan["looping_transcripts"], {})
        self.assertNotIn("repeats", " ".join(plan["warnings"]))

    def test_a_catchphrase_is_not_a_loop(self):
        """A repeated phrase has to account for a real share of the track."""
        phrase = self._loop(6, 20.0)
        filler = [
            {"i": 1000 + i, "text": f"other{i}", "start": 200.0 + i * 0.4,
             "end": 200.3 + i * 0.4, "segment": 1}
            for i in range(400)
        ]
        words = {"host": phrase + filler, "guest": []}
        plan = planner.build_plan(
            self._meta(), self._speech(words), {"host": [], "guest": []},
            words, self._params(),
        )
        self.assertEqual(plan["looping_transcripts"], {})

    def test_a_transcript_too_short_to_judge_is_left_alone(self):
        words = {"host": self._loop(2, 10.0), "guest": []}
        self.assertEqual(planner.looping_words(["host"], words), {})


class TestUntranscribedAudio(unittest.TestCase):
    """The check that would have caught the bug that prompted it.

    A 600s request to whisper-server silently omitted 22 seconds of clear speech
    that a 100s request transcribed fine. Nothing downstream could see it: the
    speech map is derived from the transcript, so both agreed that audio was
    empty, and a silence cut then removed it. A level scan is the only input
    Whisper had no hand in, and this is what it is for.
    """

    def _words(self, spec):
        return [
            {"i": i, "text": t, "start": s, "end": e, "segment": 0}
            for i, (t, s, e) in enumerate(spec)
        ]

    def _plan(self, words, loud, params=None):
        speech = {
            p: tr.speech_from_words(w, 60.0, pad=0.25) for p, w in words.items()
        }
        # These fixtures are mostly silence by construction, so the cut-fraction
        # limit is lifted: the point is what the audio cross-check does, and a
        # second blocking reason would only obscure it.
        return planner.build_plan(
            _meta(list(words), 60.0), speech, {}, words,
            params or dict(PARAMS, max_cut_fraction=1.0), loud=loud,
        )

    def test_loud_audio_with_no_words_is_reported(self):
        words = {"a": self._words([("hi", 2.0, 2.5), ("bye", 50.0, 50.5)])}
        # The level scan heard speech from 30-45s; the transcript has none.
        result = self._plan(words, {"a": [(2.0, 2.5), (30.0, 45.0), (50.0, 50.5)]})
        warning = " ".join(result["warnings"])
        self.assertIn("loud enough to be speech but produced no words", warning)
        self.assertIn("a", result["untranscribed_audio"])
        self.assertAlmostEqual(result["stats"]["untranscribed_seconds"], 15.0, places=1)

    def test_a_cut_over_it_blocks_the_run(self):
        """Reporting is not enough when the audio is already being removed."""
        words = {"a": self._words([("hi", 2.0, 2.5), ("bye", 50.0, 50.5)])}
        result = self._plan(words, {"a": [(2.0, 2.5), (30.0, 45.0), (50.0, 50.5)]})
        self.assertTrue(result["blocking"], result["warnings"])
        self.assertIn("no transcript accounts for", " ".join(result["blocking"]))
        self.assertIn("WHISPER_VAD_ONSET", " ".join(result["blocking"]))
        self.assertGreater(result["stats"]["untranscribed_in_cuts"], 5.0)

    def test_short_stretches_are_not_worth_a_warning(self):
        """A cough or a breath the scan called loud must not cry wolf."""
        words = {"a": self._words([("hi", 2.0, 2.5), ("bye", 50.0, 50.5)])}
        result = self._plan(words, {"a": [(2.0, 2.5), (20.0, 21.0), (50.0, 50.5)]})
        self.assertEqual(result["untranscribed_audio"], {})
        self.assertEqual(result["blocking"], [])

    def test_a_transcript_covering_its_audio_raises_nothing(self):
        words = {"a": self._words([(f"w{i}", 2.0 + i, 2.5 + i) for i in range(40)])}
        result = self._plan(words, {"a": [(2.0, 42.0)]})
        self.assertEqual(result["untranscribed_audio"], {})
        self.assertEqual(result["blocking"], [])

    def test_without_a_level_scan_the_check_is_simply_absent(self):
        """It must not invent a finding from a missing input."""
        words = {"a": self._words([("hi", 2.0, 2.5), ("bye", 50.0, 50.5)])}
        result = self._plan(words, {})
        self.assertEqual(result["untranscribed_audio"], {})
        self.assertEqual(result["blocking"], [])

    def test_the_padding_is_honoured_so_edges_do_not_count(self):
        """A word's span plus SPEECH_PAD covers the scan's slightly wider idea."""
        words = {"a": self._words([("hi", 10.0, 12.0)])}
        # The scan starts 0.2s earlier and ends 0.2s later than the word.
        result = self._plan(words, {"a": [(9.8, 12.2)]})
        self.assertEqual(result["untranscribed_audio"], {})


class TestProc(unittest.TestCase):
    """Running a command: what reaches the log, and what reaches the console."""

    def _log(self):
        path = os.path.join(tempfile.mkdtemp(), "run.log")
        self.addCleanup(shutil.rmtree, os.path.dirname(path), ignore_errors=True)
        buf = io.StringIO()
        return runlog.Log(path=path, stream=buf, colour=False), buf

    def test_output_goes_to_the_log_not_the_console(self):
        log, buf = self._log()
        status = proc.run(["sh", "-c", "echo chatty; echo more"], log)
        self.assertEqual(status, 0)
        written = open(log.path, encoding="utf-8").read()
        self.assertIn("chatty", written)
        self.assertIn("more", written)
        # Nothing interesting happened, so the console stays quiet.
        self.assertEqual(buf.getvalue(), "")

    def test_a_failure_reports_and_shows_the_tail(self):
        log, buf = self._log()
        status = proc.run(["sh", "-c", "echo boom >&2; exit 3"], log)
        self.assertEqual(status, 3)
        self.assertIn("command failed (exit 3)", buf.getvalue())
        self.assertIn("boom", buf.getvalue())

    def test_the_command_itself_is_logged_quoted(self):
        log, _ = self._log()
        proc.run(["sh", "-c", "true", "arg with spaces"], log)
        self.assertIn("'arg with spaces'", open(log.path, encoding="utf-8").read())

    def test_dry_run_says_what_it_would_do_and_runs_nothing(self):
        log, buf = self._log()
        target = os.path.join(os.path.dirname(log.path), "must-not-exist")
        status = proc.run(["touch", target], log, dry_run=True)
        self.assertEqual(status, 0)
        self.assertFalse(os.path.exists(target))
        self.assertIn("would run", buf.getvalue())

    def test_ffmpeg_progress_reads_microseconds(self):
        log, buf = self._log()
        on_line = proc.ffmpeg_progress(log, 10_000_000, "decoding alice")
        on_line("frame=1")                 # not a progress line
        on_line("out_time_ms=5000000")     # halfway, in microseconds
        on_line("out_time_ms=garbage")     # survives nonsense
        self.assertIn("(500/1000) decoding alice", buf.getvalue())

    def test_no_total_means_no_counter(self):
        log, buf = self._log()
        proc.ffmpeg_progress(log, 0, "x")("out_time_ms=5000000")
        self.assertEqual(buf.getvalue(), "")

    def test_require_bin_finds_a_name_on_the_path(self):
        found = proc.require_bin("sh", "sh")
        self.assertTrue(os.path.isabs(found))
        self.assertTrue(os.access(found, os.X_OK))

    def test_require_bin_takes_an_explicit_path_as_given(self):
        explicit = shutil.which("sh")
        self.assertEqual(proc.require_bin("sh", explicit), explicit)

    def test_require_bin_raises_rather_than_returning_a_status(self):
        # The shell version could only return 1, which every caller had to
        # remember to check inside a $(...) where exit would not have worked.
        with self.assertRaises(proc.ToolError) as caught:
            proc.require_bin("nosuchtool", "definitely-not-installed-xyz")
        self.assertIn("nosuchtool not found", str(caught.exception))

    def test_an_unbuildable_codec_is_refused_before_any_stage(self):
        # The point of checking up front: render is last, so without this the
        # missing encoder surfaces after the episode has been transcribed.
        log, buf = self._log()
        settings = cfg.defaults()
        settings["OUTPUT_CODEC"] = "not_a_real_encoder"
        status = pipeline.run_episode(
            settings, log, ["discover"], input_files=("nowhere.flac",))
        self.assertEqual(status, 1)
        self.assertIn("no 'not_a_real_encoder' encoder", buf.getvalue())
        # Nothing began: the discover stage never announced itself.
        self.assertNotIn("locating the episode", buf.getvalue())

    def test_ffmpeg_bin_names_the_build_to_use(self):
        log, _ = self._log()
        real = shutil.which("ffmpeg")
        ffmpeg, _probe = proc.resolve_ffmpeg(
            cfg.defaults(), log, environ={"FFMPEG_BIN": real})
        self.assertEqual(ffmpeg, real)


class FakeWhisperX:
    """Stands in for the whisperx package. No model, no torch, no download.

    The real thing is 3GB of dependencies and downloads weights on first use,
    which is not something the unit layer should ever touch — the same reason
    the remote client was tested against a stub server rather than a real one.
    """

    def __init__(self, segments, language="en", words=None):
        self._segments, self._language, self._words = segments, language, words
        self.load_model_calls, self.align_model_calls = [], []
        self.transcribe_options = []

    # --- the three entry points whisperx_asr uses ---------------------------
    def load_model(self, name, **kwargs):
        self.load_model_calls.append((name, kwargs))
        return self

    def load_align_model(self, language_code, device):
        self.align_model_calls.append(language_code)
        return ("align-model", {"language": language_code})

    def load_audio(self, path):
        return [0.0] * 16000

    def align(self, segments, model, meta, audio, device, **kwargs):
        return {"segments": self._words if self._words is not None else segments}

    # --- the model object load_model returns --------------------------------
    # Deliberately narrow: FasterWhisperPipeline.transcribe takes these and
    # nothing else. A fake that swallowed **anything would have accepted
    # initial_prompt here, which the real one raises on — and had it been
    # merely ignored instead, the transcript would have come back fluent with
    # the fillers gone and every test still green.
    def transcribe(self, audio, batch_size=None, num_workers=0, language=None,
                   task=None, chunk_size=30, print_progress=False,
                   combined_progress=False, verbose=False,
                   progress_callback=None):
        self.transcribe_options.append(
            {"batch_size": batch_size, "language": language})
        return {"segments": self._segments, "language": self._language}


class TestWhisperXTiming(unittest.TestCase):
    """Turning aligned words into the segments the rest of the pipeline reads."""

    def test_one_segment_per_word_with_millisecond_offsets(self):
        aligned = {"segments": [{"start": 0.0, "end": 2.0, "words": [
            {"word": "well", "start": 0.5, "end": 0.8},
            {"word": "um", "start": 1.2, "end": 1.35},
        ]}]}
        got = wx.segments_from_alignment(aligned)
        self.assertEqual(
            got,
            [{"text": "well", "offsets": {"from": 500, "to": 800}},
             {"text": "um", "offsets": {"from": 1200, "to": 1350}}])

    def test_an_untimed_word_is_never_dropped(self):
        # The safety property. The speech map is derived from these words and
        # silence is their absence, so a dropped word is audio that can then be
        # cut out from under the speaker.
        aligned = {"segments": [{"start": 0.0, "end": 3.0, "words": [
            {"word": "chapter", "start": 0.0, "end": 1.0},
            {"word": "19"},                                  # no timings
            {"word": "begins", "start": 2.0, "end": 3.0},
        ]}]}
        got = wx.segments_from_alignment(aligned)
        self.assertEqual([s["text"] for s in got], ["chapter", "19", "begins"])
        # It landed in the gap its neighbours left, not at zero.
        self.assertEqual(got[1]["offsets"], {"from": 1000, "to": 2000})

    def test_an_untimed_run_is_shared_out_evenly(self):
        words = [{"word": "a", "start": 0.0, "end": 1.0},
                 {"word": "b"}, {"word": "c"}, {"word": "d"},
                 {"word": "e", "start": 4.0, "end": 5.0}]
        filled = wx.fill_missing_timings(words, 0.0, 5.0)
        self.assertEqual([round(w["start"], 3) for w in filled],
                         [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_untimed_at_the_edges_borrows_the_segment_bounds(self):
        words = [{"word": "first"},
                 {"word": "middle", "start": 2.0, "end": 3.0},
                 {"word": "last"}]
        filled = wx.fill_missing_timings(words, 1.0, 6.0)
        self.assertEqual(round(filled[0]["start"], 3), 1.0)
        self.assertEqual(round(filled[2]["end"], 3), 6.0)

    def test_the_result_feeds_build_from_segments_unchanged(self):
        # The whole point of the one-word-per-segment shape: nothing downstream
        # of transcript.py had to learn about WhisperX.
        aligned = {"segments": [{"start": 0.0, "end": 2.0, "words": [
            {"word": "so", "start": 0.1, "end": 0.3},
            {"word": "anyway", "start": 0.4, "end": 0.9},
        ]}]}
        built = tr.build_from_segments(wx.segments_from_alignment(aligned), "alice")
        self.assertEqual([w["text"] for w in built["words"]], ["so", "anyway"])
        self.assertAlmostEqual(built["words"][0]["start"], 0.1, places=3)
        self.assertAlmostEqual(built["words"][1]["end"], 0.9, places=3)


class TestWhisperXTranscriber(unittest.TestCase):
    """Loading the model, and what gets asked of it."""

    def _fake(self, **kwargs):
        fake = FakeWhisperX(**kwargs)
        return fake, (lambda: fake)

    def test_the_model_is_loaded_once_and_reused(self):
        # Loading costs tens of seconds on a CPU and an episode is several
        # tracks; per-track loading would dominate the run.
        fake, loader = self._fake(segments=[{"start": 0.0, "end": 1.0, "words": [
            {"word": "hello", "start": 0.0, "end": 1.0}]}])
        t = wx.Transcriber(model="small", loader=loader)
        t.transcribe("/one.wav")
        t.transcribe("/two.wav")
        self.assertEqual(len(fake.load_model_calls), 1)
        self.assertEqual(fake.load_model_calls[0][0], "small")

    def test_the_aligner_is_loaded_once_per_language(self):
        fake, loader = self._fake(segments=[{"start": 0.0, "end": 1.0, "words": [
            {"word": "hello", "start": 0.0, "end": 1.0}]}])
        t = wx.Transcriber(loader=loader)
        t.transcribe("/one.wav")
        t.transcribe("/two.wav")
        self.assertEqual(fake.align_model_calls, ["en"])

    def test_the_prompt_reaches_the_model_at_load_time(self):
        # Without it Whisper returns fluent prose and the disfluencies never
        # reach the detector — DESIGN.md §6. WhisperX fixes its decode options
        # when the model is built; its transcribe() takes no initial_prompt, so
        # passing one per call is a TypeError rather than a silent no-op.
        fake, loader = self._fake(segments=[{"start": 0.0, "end": 1.0, "words": [
            {"word": "um", "start": 0.0, "end": 1.0}]}])
        wx.Transcriber(loader=loader, prompt="Um, uh, so").transcribe("/a.wav")
        _name, kwargs = fake.load_model_calls[0]
        self.assertEqual(kwargs["asr_options"]["initial_prompt"], "Um, uh, so")
        self.assertNotIn("initial_prompt", fake.transcribe_options[0])

    def test_the_decode_does_not_retry_at_a_higher_temperature(self):
        # WhisperX re-decodes at rising temperatures when a pass looks too
        # repetitive. A stutter *is* repetitive, so the retry removes what this
        # pipeline exists to find — and above zero it samples, which made three
        # identical runs of the same audio give two different transcripts.
        fake, loader = self._fake(segments=[])
        wx.Transcriber(loader=loader)
        self.assertEqual(fake.load_model_calls[0][1]["asr_options"]["temperatures"],
                         [0.0])

    def test_the_ladder_can_be_asked_for(self):
        # For a track with a genuine repetition loop, which is what the ladder
        # is actually for.
        fake, loader = self._fake(segments=[])
        wx.Transcriber(loader=loader, temperature_fallback=True)
        self.assertNotIn("temperatures",
                         fake.load_model_calls[0][1].get("asr_options") or {})

    def test_the_vad_method_is_stated_not_defaulted(self):
        # It was defined as a setting and wired to nothing, so WhisperX fell
        # back to pyannote while the config, the form and the run log all said
        # whatever the operator had chosen. whisper-server ran Silero, so the
        # silent default was also a change of detector.
        fake, loader = self._fake(segments=[])
        wx.Transcriber(loader=loader, vad_method="silero")
        _name, kwargs = fake.load_model_calls[0]
        self.assertEqual(kwargs["vad_method"], "silero")

    def test_the_vad_method_is_never_left_unsaid(self):
        # Even with nothing asked for, something is stated: the failure mode
        # being guarded is WhisperX silently choosing for us.
        fake, loader = self._fake(segments=[])
        wx.Transcriber(loader=loader)
        self.assertEqual(fake.load_model_calls[0][1]["vad_method"], "pyannote")

    def test_the_default_is_what_whisperx_intends(self):
        # pyannote, whose weights ship in the package. Silero — what
        # whisper-server ran — stays reachable for a like-for-like comparison.
        self.assertEqual(cfg.defaults()["WHISPER_VAD_METHOD"], "pyannote")
        self.assertIn("silero", cfg.SETTINGS["WHISPER_VAD_METHOD"][2])

    def test_the_vad_settings_reach_the_model(self):
        # They used to be form fields on an HTTP request the self-test could
        # read back. Now they are constructor arguments, so this is where the
        # property lives.
        fake, loader = self._fake(segments=[])
        wx.Transcriber(loader=loader,
                       vad_options={"vad_onset": 0.4, "vad_offset": 0.3})
        _name, kwargs = fake.load_model_calls[0]
        self.assertEqual(kwargs["vad_options"], {"vad_onset": 0.4, "vad_offset": 0.3})

    def test_the_settings_map_onto_whisperx_names(self):
        # WHISPER_VAD_ONSET/OFFSET are what pyannote takes. The old Silero
        # threshold and its four durations had no honest translation, which is
        # why they went rather than being renamed onto something else.
        got = pipeline.vad_options(
            cfg.defaults() | {"WHISPER_VAD_ONSET": "0.6", "WHISPER_VAD_OFFSET": "0.2"})
        self.assertEqual(got, {"vad_onset": 0.6, "vad_offset": 0.2})

    def test_silence_is_an_answer_not_a_failure(self):
        # One participant is silent for minutes while the other talks; a track
        # with nothing in it must not look like a broken run.
        fake, loader = self._fake(segments=[])
        segments, language = wx.Transcriber(loader=loader).transcribe("/quiet.wav")
        self.assertEqual(segments, [])
        self.assertEqual(language, "en")
        self.assertEqual(fake.align_model_calls, [])   # nothing to align

    def test_a_missing_package_says_where_to_get_it(self):
        def absent():
            raise wx.WhisperXMissing(
                "whisperx is not installed in this interpreter. Use the "
                "container image, which carries it.")
        with self.assertRaises(wx.WhisperXMissing) as caught:
            wx.Transcriber(loader=absent)
        self.assertIn("container image", str(caught.exception))


class TestRunComparison(unittest.TestCase):
    """Scoring one finished render against another, for tools/compare_runs.py.

    The recovery half needs numpy, scipy and real audio and is not exercised
    here. What is exercised is the arithmetic that turns two recovered cut lists
    into a verdict, which is the part that would silently mislead.
    """

    def setUp(self):
        sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))
        import compare_runs
        self.cr = compare_runs

    def _recovered(self, removed, duration=60.0):
        return {"original_duration": duration, "edit_duration": duration - sum(
            end - start for start, end in removed),
            "removed": [list(r) for r in removed], "envelope_error": 0.01}

    def test_an_identical_edit_scores_perfectly(self):
        ref = self._recovered([(10.0, 12.0), (30.0, 33.0)])
        stats = self.cr.score(self.cr.as_plan(ref), "host", ref)
        self.assertAlmostEqual(stats["f1"], 1.0, places=6)

    def test_removing_nothing_scores_zero_without_dividing_by_zero(self):
        ref = self._recovered([(10.0, 12.0)])
        stats = self.cr.score(self.cr.as_plan(self._recovered([])), "host", ref)
        self.assertEqual(stats["f1"], 0.0)
        self.assertEqual(stats["recall"], 0.0)

    def test_cutting_everything_is_caught_by_precision(self):
        # The failure the score exists to catch: a candidate that removes the
        # whole track overlaps every reference cut, so recall is perfect. Only
        # precision says it destroyed the episode.
        ref = self._recovered([(10.0, 12.0), (30.0, 33.0)])
        stats = self.cr.score(self.cr.as_plan(self._recovered([(0.0, 60.0)])),
                              "host", ref)
        self.assertAlmostEqual(stats["recall"], 1.0, places=6)
        self.assertLess(stats["precision"], 0.1)

    def test_what_was_missed_is_reported_in_seconds(self):
        ref = self._recovered([(10.0, 12.0), (30.0, 33.0)])
        got = self._recovered([(10.0, 12.0)])
        stats = self.cr.score(self.cr.as_plan(got), "host", ref)
        missed = iv.total(iv.subtract(stats["reference"], stats["cuts"]))
        self.assertAlmostEqual(missed, 3.0, places=6)

    def test_the_cache_is_invalidated_when_an_input_changes(self):
        # The reference audio does get re-cut — an editor who finds a miss
        # fixes it — and a cache keyed only on a label would answer the new
        # question with the old number, silently.
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        one = os.path.join(root, "a.bin")
        two = os.path.join(root, "b.bin")
        for path in (one, two):
            with open(path, "wb") as handle:
                handle.write(b"original")
        before = self.cr.fingerprint(one, two)
        with open(two, "wb") as handle:
            handle.write(b"re-cut by hand")
        self.assertNotEqual(before, self.cr.fingerprint(one, two))

    def test_a_reference_that_did_not_align_is_not_silently_scored(self):
        # tools/README.md is explicit: above this the cut list is wrong and
        # nothing computed from it means anything.
        self.assertEqual(self.cr.MAX_ENVELOPE_ERROR, 0.1)


class TestEndpointsAreRequired(unittest.TestCase):
    """A missing endpoint is a sentence, not a traceback.

    The shell refused these before the stage ran, in config_need_whisper and
    config_need_llama. The port left them behind: nothing called those functions
    once the stages moved, so an unset WHISPER_ENDPOINT reached urllib and came
    back as `ValueError: unknown url type: '/inference'`.
    """

    def _log(self):
        return runlog.Log(stream=io.StringIO(), colour=False)

    def test_detect_points_at_the_way_out(self):
        settings = cfg.defaults()
        with self.assertRaises(pipeline.StageError) as caught:
            pipeline.stage_detect("/nonexistent", settings, self._log())
        self.assertIn("LLAMA_ENDPOINT is required", str(caught.exception))
        # The other answer is not to run the stage at all.
        self.assertIn("LLM_ENABLE=0", str(caught.exception))


class _NullTranscriber:
    """Answers every track with silence. For the paths that never get that far."""

    def transcribe(self, wav_path, *, batch_size=8):
        return [], "en"


class TestPipelineTranscribeStage(unittest.TestCase):
    """The level scan, and what the transcript summary reports."""

    def _log(self, level="info"):
        path = os.path.join(tempfile.mkdtemp(), "run.log")
        self.addCleanup(shutil.rmtree, os.path.dirname(path), ignore_errors=True)
        buf = io.StringIO()
        return runlog.Log(path=path, level=level, stream=buf, colour=False), buf

    def _work(self, seconds=1.0):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        for leaf in ("prep", "asr", "words", "state"):
            os.makedirs(os.path.join(root, leaf), exist_ok=True)
        wav = os.path.join(root, "prep", "alice.wav")
        with wave.open(wav, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            # Half silence, half a tone, so silencedetect has something to find.
            quiet = b"\0\0" * int(16000 * seconds / 2)
            loud = b"\x00\x40" * int(16000 * seconds / 2)
            handle.writeframes(quiet + loud)
        return root, wav

    def _settings(self, **over):
        values = cfg.defaults()
        values.update(over)
        return values

    def test_the_level_scan_writes_a_json_and_a_log(self):
        root, wav = self._work(2.0)
        log, _ = self._log()
        loud = pipeline.scan_levels(
            root, "alice", wav, 2.0,
            self._settings(SPLIT_SILENCE_THRESHOLD="-45dB",
                           SPLIT_MIN_SILENCE="0.30"),
            log, "ffmpeg")
        self.assertIsNotNone(loud)
        self.assertTrue(os.path.isfile(os.path.join(root, "asr", "alice.silence.log")))
        written = json.load(open(os.path.join(root, "asr", "alice.loud.json")))
        self.assertEqual(written["participant"], "alice")
        # Records which threshold produced it, where the shell left it empty.
        self.assertEqual(written["threshold"], "-45dB")
        self.assertTrue(written["loud"])

    def test_a_failed_scan_warns_and_yields_nothing(self):
        root, _ = self._work()
        log, buf = self._log()
        loud = pipeline.scan_levels(
            root, "alice", os.path.join(root, "prep", "absent.wav"), 1.0,
            self._settings(), log, "ffmpeg")
        self.assertIsNone(loud)
        self.assertIn("could not scan alice", buf.getvalue())

    def test_the_summary_is_a_note_not_a_console_line(self):
        # It goes to the log only, as it did when bash read it off a pipe.
        log, buf = self._log()
        parsed = {"words": [{"i": 0}, {"i": 1}], "segments": [{"i": 0}]}
        pipeline.describe_transcript(parsed, self._settings(), log)
        self.assertIn("2 words in 1 segments",
                      open(log.path, encoding="utf-8").read())
        self.assertEqual(buf.getvalue(), "")

    def test_a_track_with_no_speech_says_so(self):
        # Normal on a two-mic recording; the plan stage is what refuses if that
        # silence turns out to have been audible speech.
        log, _ = self._log()
        pipeline.describe_transcript({"words": [], "segments": []},
                                     self._settings(), log)
        self.assertIn("no speech at all", open(log.path, encoding="utf-8").read())

    def test_a_missing_prepared_track_is_refused_by_path(self):
        root, _ = self._work()
        os.remove(os.path.join(root, "prep", "alice.wav"))
        with open(os.path.join(root, "meta.json"), "w", encoding="utf-8") as handle:
            json.dump({"episode_id": "ep", "duration": 1.0, "sample_rate": 16000,
                       "tracks": [{"participant": "alice", "duration": 1.0,
                                   "sample_rate": 16000, "sample_fmt": "s16"}]},
                      handle)
        log, _ = self._log()
        # A transcriber is injected so the stage gets as far as the missing
        # file rather than trying to load three gigabytes of model first.
        with self.assertRaises(pipeline.StageError) as caught:
            pipeline.stage_transcribe(root, self._settings(), log,
                                      transcriber=_NullTranscriber())
        self.assertIn("missing prepared track", str(caught.exception))


class TestPipelineDetectStage(unittest.TestCase):
    """The three faults that end the run, and the one that ends a track.

    This is what the shell's exit codes 2, 3, 5 and 4 carried. They are
    exceptions now, so what is worth pinning is that each still stops what it
    used to stop, and still says what to do about it.
    """

    HINT = "clean-podcast.sh --episode ep --from detect"

    def _work(self, participants=("alice",), words=True):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        for leaf in ("words", "llm", "state"):
            os.makedirs(os.path.join(root, leaf), exist_ok=True)
        with open(os.path.join(root, "meta.json"), "w", encoding="utf-8") as handle:
            json.dump({"episode_id": "ep", "duration": 10.0, "sample_rate": 48000,
                       "tracks": [{"participant": p, "duration": 10.0,
                                   "sample_rate": 48000, "sample_fmt": "s16"}
                                  for p in participants]}, handle)
        if words:
            for name in participants:
                with open(os.path.join(root, "words", f"{name}.words.json"),
                          "w", encoding="utf-8") as handle:
                    json.dump({"participant": name, "words": [
                        {"i": 0, "text": "hi", "start": 0.0, "end": 0.4,
                         "segment": 0}], "segments": []}, handle)
        return root

    def _log(self):
        path = os.path.join(tempfile.mkdtemp(), "run.log")
        self.addCleanup(shutil.rmtree, os.path.dirname(path), ignore_errors=True)
        buf = io.StringIO()
        return runlog.Log(path=path, stream=buf, colour=False), buf

    class _Client:
        def __init__(self, *a, **k):
            pass

        def wait_until_ready(self, *a, **k):
            return True

        def check_schema_support(self):
            return None

    def _run(self, root, log, detect, settings=None):
        values = cfg.defaults()
        values.update(settings or {})
        values["LLAMA_ENDPOINT"] = "http://stub"
        with unittest.mock.patch.object(llm, "LlamaClient", self._Client), \
                unittest.mock.patch.object(llm, "detect", detect):
            pipeline.stage_detect(root, values, log, resume_hint=self.HINT)

    def test_a_refused_key_ends_the_run_and_says_so(self):
        root, (log, _) = self._work(), self._log()
        def detect(*a, **k):
            raise llm.AuthRejected("nope")
        with self.assertRaises(pipeline.StageError) as caught:
            self._run(root, log, detect)
        message = str(caught.exception)
        self.assertIn("refused our credentials", message)
        self.assertIn(self.HINT, message)

    def test_an_ignored_schema_suggests_the_other_api(self):
        root, (log, _) = self._work(), self._log()
        def detect(*a, **k):
            raise llm.SchemaIgnored("prose")
        with self.assertRaises(pipeline.StageError) as caught:
            self._run(root, log, detect)
        self.assertIn("LLM_API=completion", str(caught.exception))

    def test_no_model_loaded_names_the_model_when_one_was_asked_for(self):
        root, (log, _) = self._work(), self._log()
        def detect(*a, **k):
            raise llm.ModelUnavailable("gone")
        with self.assertRaises(pipeline.StageError) as caught:
            self._run(root, log, detect, {"LLAMA_MODEL_NAME": "qwen-podcast"})
        self.assertIn("qwen-podcast", str(caught.exception))

    def test_every_chunk_failing_leaves_the_track_unmarked(self):
        root = self._work()
        log, buf = self._log()
        def detect(*a, **k):
            return {"participant": "alice", "edits": [], "rejected_count": 0,
                    "chunks": 3, "chunk_failures": 3}
        # One track, and it failed, so the stage refuses the episode.
        with self.assertRaises(pipeline.StageError) as caught:
            self._run(root, log, detect)
        self.assertIn("every track", str(caught.exception))
        self.assertIn("not analysed at all", buf.getvalue())
        self.assertFalse(
            os.path.exists(os.path.join(root, "state", "llm-alice.ok")))

    def test_one_bad_track_of_two_is_survivable(self):
        root = self._work(("alice", "bob"))
        log, buf = self._log()
        def detect(client, parsed, **k):
            if parsed["participant"] == "alice":
                return {"participant": "alice", "edits": [], "rejected_count": 0,
                        "chunks": 2, "chunk_failures": 2}
            return {"participant": "bob", "edits": [], "rejected_count": 0,
                    "chunks": 2, "chunk_failures": 0}
        self._run(root, log, detect)
        self.assertFalse(os.path.exists(os.path.join(root, "state", "llm-alice.ok")))
        self.assertTrue(os.path.exists(os.path.join(root, "state", "llm-bob.ok")))
        self.assertIn("keeps its disfluencies", buf.getvalue())

    def test_a_track_without_a_transcript_is_skipped_not_fatal(self):
        root = self._work(("alice",), words=False)
        log, buf = self._log()
        def detect(*a, **k):
            raise AssertionError("should not be called")
        self._run(root, log, detect)
        self.assertIn("no transcript for alice", buf.getvalue())

    def test_an_unknown_edit_kind_is_refused_by_name(self):
        root, (log, _) = self._work(), self._log()
        def detect(*a, **k):
            raise AssertionError("should not be called")
        with self.assertRaises(pipeline.StageError) as caught:
            self._run(root, log, detect, {"LLM_ACCEPT_KINDS": "stutter,banana"})
        self.assertIn("banana", str(caught.exception))


class TestPipelineRenderStage(unittest.TestCase):
    """Rendering, the two no-edit shortcuts, and the length check."""

    def _log(self):
        path = os.path.join(tempfile.mkdtemp(), "run.log")
        self.addCleanup(shutil.rmtree, os.path.dirname(path), ignore_errors=True)
        buf = io.StringIO()
        return runlog.Log(path=path, stream=buf, colour=False), buf

    def _settings(self, **over):
        values = cfg.defaults()
        values.update(over)
        return values

    def _episode(self, source_ext="flac", codec="flac", cuts=(), seconds=2.0):
        """A work directory with one track, optionally with a filtergraph."""
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        for leaf in ("render", "state"):
            os.makedirs(os.path.join(root, leaf), exist_ok=True)
        staging = os.path.join(root, "staging")

        raw = os.path.join(root, "alice.wav")
        with wave.open(raw, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(48000)
            handle.writeframes(b"\x00\x20" * int(48000 * seconds))
        source = raw
        if source_ext != "wav":
            source = os.path.join(root, f"alice.{source_ext}")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", raw, source],
                           check=True)

        track = {"participant": "alice", "source": source, "duration": seconds,
                 "sample_rate": 48000, "render_rate": 48000, "sample_fmt": "s16",
                 "codec": codec, "lossless": True}
        with open(os.path.join(root, "meta.json"), "w", encoding="utf-8") as handle:
            json.dump({"episode_id": "ep", "duration": seconds,
                       "sample_rate": 48000, "tracks": [track]}, handle)
        plan = {"duration": seconds, "cuts": list(cuts), "mutes": {"alice": []},
                "keep": [], "stats": {}, "warnings": [], "blocking": []}
        with open(os.path.join(root, "plan.json"), "w", encoding="utf-8") as handle:
            json.dump(plan, handle)
        # write_filters decides whether a graph is needed, exactly as the stage does.
        pipeline.write_filters(root, {"tracks": [track]}, plan, self._settings())
        return root, staging, track

    def test_bit_depth_is_only_passed_to_encoders_that_have_one(self):
        track = {"sample_fmt": "s16"}
        self.assertIn("-sample_fmt", pipeline.encode_args(
            self._settings(OUTPUT_CODEC="flac"), track))
        self.assertIn("-sample_fmt", pipeline.encode_args(
            self._settings(OUTPUT_CODEC="pcm_s16le"), track))
        self.assertNotIn("-sample_fmt", pipeline.encode_args(
            self._settings(OUTPUT_CODEC="libopus"), track))
        # And not for a format the encoder could not carry anyway.
        self.assertNotIn("-sample_fmt", pipeline.encode_args(
            self._settings(OUTPUT_CODEC="flac"), {"sample_fmt": "fltp"}))

    def test_flac_gets_its_compression_level_and_extra_args_are_split(self):
        args = pipeline.encode_args(
            self._settings(OUTPUT_CODEC="flac", OUTPUT_COMPRESSION="5",
                           OUTPUT_EXTRA_ARGS="-b:a 192k"), {})
        self.assertIn("-compression_level", args)
        self.assertIn("5", args)
        self.assertIn("-b:a", args)
        self.assertIn("192k", args)

    def test_no_edits_and_already_the_right_format_is_copied_through(self):
        root, staging, _ = self._episode(source_ext="flac", codec="flac")
        log, buf = self._log()
        pipeline.stage_render(root, staging, self._settings(), log)
        self.assertIn("copying it through", buf.getvalue())
        target = os.path.join(staging, "alice.flac")
        self.assertEqual(open(target, "rb").read(),
                         open(os.path.join(root, "alice.flac"), "rb").read())

    def test_no_edits_but_the_wrong_format_is_converted(self):
        root, staging, _ = self._episode(source_ext="wav", codec="pcm_s16le")
        log, buf = self._log()
        pipeline.stage_render(root, staging, self._settings(OUTPUT_CODEC="flac",
                                                           OUTPUT_EXT="flac"), log)
        self.assertIn("converting to flac", buf.getvalue())
        self.assertTrue(os.path.isfile(os.path.join(staging, "alice.flac")))

    def test_a_cut_means_a_filtergraph_and_a_shorter_file(self):
        root, staging, _ = self._episode(
            source_ext="flac", codec="flac", cuts=[{"start": 0.5, "end": 1.0}])
        log, buf = self._log()
        pipeline.stage_render(root, staging, self._settings(FFMPEG_JOBS="1"), log)
        self.assertIn("rendered alice", buf.getvalue())
        self.assertIn("all tracks verified", buf.getvalue())
        rendered = pipeline.probe_duration(
            "ffprobe", os.path.join(staging, "alice.flac"))
        self.assertAlmostEqual(rendered, 1.5, places=1)

    def test_a_length_that_does_not_match_the_plan_stops_the_run(self):
        root, staging, _ = self._episode(source_ext="flac", codec="flac")
        log, _ = self._log()
        # Claim the track should come out twice as long as it can.
        path = os.path.join(root, "expected.json")
        expected = json.load(open(path))
        expected["tracks"]["alice"]["expected_duration"] = 99.0
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(expected, handle)
        with self.assertRaises(pipeline.StageError) as caught:
            pipeline.stage_render(root, staging, self._settings(), log)
        self.assertIn("do not match the plan", str(caught.exception))

    def test_a_missing_plan_is_refused_before_anything_runs(self):
        root, staging, _ = self._episode()
        os.remove(os.path.join(root, "plan.json"))
        log, _ = self._log()
        with self.assertRaises(pipeline.StageError) as caught:
            pipeline.stage_render(root, staging, self._settings(), log)
        self.assertIn("no plan.json", str(caught.exception))


class TestPipelineFinalizeStage(unittest.TestCase):
    """Publishing, and the order in which things are allowed to be deleted."""

    def _log(self, path=None):
        root = path or tempfile.mkdtemp()
        if path is None:
            self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        log_path = os.path.join(root, "run.log")
        open(log_path, "w", encoding="utf-8").close()
        buf = io.StringIO()
        return runlog.Log(path=log_path, stream=buf, colour=False), buf

    def _settings(self, **over):
        values = cfg.defaults()
        values.update(over)
        return values

    def _episode(self, staged=True):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        work = os.path.join(root, "work", "ep")
        out = os.path.join(root, "output", "ep")
        staging = os.path.join(out, ".staging")
        for path in (os.path.join(work, "words"), os.path.join(work, "llm"),
                     os.path.join(work, "logs"), staging):
            os.makedirs(path, exist_ok=True)

        inputs = os.path.join(root, "incoming")
        os.makedirs(inputs, exist_ok=True)
        source = os.path.join(inputs, "ep_alice.flac")
        with open(source, "wb") as handle:
            handle.write(b"original audio")

        if staged:
            with open(os.path.join(staging, "alice.flac"), "wb") as handle:
                handle.write(b"rendered audio")
        for name, body in (
            ("plan.json", json.dumps({
                "episode_id": "ep", "participants": ["alice"], "duration": 1.0,
                "cuts": [], "mutes": {"alice": []}, "keep": [[0.0, 1.0]]})),
            ("edit-report.txt", "a report"),
        ):
            with open(os.path.join(work, name), "w", encoding="utf-8") as handle:
                handle.write(body)
        with open(os.path.join(work, "words", "alice.words.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({"participant": "alice", "words": [], "segments": []}, handle)
        with open(os.path.join(work, "llm", "alice.audit.jsonl"), "w",
                  encoding="utf-8") as handle:
            handle.write("{}\n")
        return work, out, staging, {"alice": source}

    def test_publishes_audio_sidecars_and_logs(self):
        work, out, staging, sources = self._episode()
        log, buf = self._log(work)
        pipeline.stage_finalize(work, out, staging, self._settings(KEEP_WORK="1"),
                                log, "ep", sources)
        self.assertEqual(open(os.path.join(out, "alice.flac"), "rb").read(),
                         b"rendered audio")
        for name in ("ep_plan.json", "ep_edit-report.txt", "ep_transcript.json",
                     "ep_transcript.srt", "ep_transcript.txt"):
            self.assertTrue(os.path.isfile(os.path.join(out, name)), name)
        self.assertTrue(os.path.isfile(os.path.join(out, "logs", "run.log")))
        self.assertTrue(os.path.isfile(
            os.path.join(out, "logs", "alice.audit.jsonl")))
        # Staging was inside the output directory and is gone again.
        self.assertFalse(os.path.exists(staging))

    def test_keep_inputs_and_keep_work_are_honoured(self):
        work, out, staging, sources = self._episode()
        log, _ = self._log(work)
        pipeline.stage_finalize(
            work, out, staging,
            self._settings(KEEP_INPUTS="1", KEEP_WORK="1"), log, "ep", sources)
        self.assertTrue(os.path.isfile(sources["alice"]))
        self.assertTrue(os.path.isdir(work))

    def test_without_them_the_inputs_and_work_directory_go(self):
        work, out, staging, sources = self._episode()
        log, _ = self._log(work)
        published = pipeline.stage_finalize(
            work, out, staging,
            self._settings(KEEP_INPUTS="0", KEEP_WORK="0"), log, "ep", sources)
        self.assertFalse(os.path.exists(sources["alice"]))
        self.assertFalse(os.path.exists(work))
        # The log moved, and the caller is told where, because it keeps writing.
        self.assertEqual(published, os.path.join(out, "logs", "run.log"))
        self.assertEqual(log.path, published)
        log.info("written after the work directory went")
        self.assertIn("after the work directory went",
                      open(published, encoding="utf-8").read())

    def test_nothing_is_deleted_when_publishing_fails(self):
        """The ordering guarantee: a failure leaves the originals alone."""
        work, out, staging, sources = self._episode(staged=False)
        log, _ = self._log(work)
        with self.assertRaises(pipeline.StageError) as caught:
            pipeline.stage_finalize(
                work, out, staging,
                self._settings(KEEP_INPUTS="0", KEEP_WORK="0"), log, "ep", sources)
        self.assertIn("could not publish", str(caught.exception))
        self.assertTrue(os.path.isfile(sources["alice"]),
                        "the original input must survive a failed publish")
        self.assertTrue(os.path.isdir(work),
                        "the work directory must survive a failed publish")

    def test_human_size_reads_like_du(self):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        path = os.path.join(root, "f")
        for size, want in ((512, "512"), (2048, "2K"), (3 * 1024 * 1024, "3.0M")):
            with open(path, "wb") as handle:
                handle.write(b"\0" * size)
            self.assertEqual(pipeline.human_size(path), want)


class TestPipelinePlanStage(unittest.TestCase):
    """The plan stage as one call, taking its numbers from the settings."""

    def _work(self, **files):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        for leaf in ("words", "asr", "llm", "render"):
            os.makedirs(os.path.join(root, leaf), exist_ok=True)
        for name, payload in files.items():
            path = os.path.join(root, name.replace("__", os.sep))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
        return root

    def _settings(self, **over):
        values = cfg.defaults()
        values.update(over)
        return values

    def test_params_come_from_the_settings(self):
        values = self._settings(SILENCE_KEEP="0.15", SPEECH_PAD="0.15")
        params = pipeline.plan_params(values)
        self.assertEqual(params["silence_keep"], 0.15)
        self.assertEqual(params["speech_pad"], 0.15)
        # Exactly the keys plan.build_plan reads, no more and no fewer.
        self.assertEqual(set(params), set(pipeline.PLAN_PARAMS))

    def test_collect_keys_by_participant(self):
        root = self._work(**{"words__alice.words.json": {"words": []},
                             "words__bob.words.json": {"words": []}})
        found = pipeline.collect(os.path.join(root, "words"), ".words.json")
        self.assertEqual(sorted(found), ["alice", "bob"])

    def test_a_missing_transcript_is_refused_by_name(self):
        root = self._work(**{
            "meta.json": {"episode_id": "ep", "duration": 10.0,
                          "sample_rate": 48000,
                          "tracks": [{"participant": "alice", "duration": 10.0,
                                      "sample_rate": 48000, "sample_fmt": "s16"},
                                     {"participant": "bob", "duration": 10.0,
                                      "sample_rate": 48000, "sample_fmt": "s16"}]},
            "words__alice.words.json": {"words": []},
        })
        with self.assertRaises(pipeline.StageError) as caught:
            pipeline.stage_plan(root, self._settings(),
                                runlog.Log(stream=io.StringIO(), colour=False))
        self.assertIn("bob", str(caught.exception))

    def test_it_writes_everything_the_render_stage_reads(self):
        # Speech either side of one long gap. Dense enough that the plan removes
        # a plausible slice rather than tripping MAX_CUT_FRACTION, which is what
        # a sparser fixture does — correctly, and unhelpfully for this test.
        moments = [n / 2 for n in range(1, 16)] + [n / 2 for n in range(26, 39)]
        words = [{"i": i, "text": f"w{i}", "start": at, "end": at + 0.4,
                  "segment": 0} for i, at in enumerate(moments)]
        loud = [[at - 0.05, at + 0.45] for at in moments]
        root = self._work(**{
            "meta.json": {"episode_id": "ep", "duration": 20.0,
                          "sample_rate": 48000,
                          "tracks": [{"participant": "alice", "duration": 20.0,
                                      "sample_rate": 48000, "render_rate": 48000,
                                      "sample_fmt": "s16", "lossless": True}]},
            "words__alice.words.json": {"words": words},
            "asr__alice.loud.json": {"loud": loud},
        })
        plan = pipeline.stage_plan(
            root, self._settings(), runlog.Log(stream=io.StringIO(), colour=False))
        self.assertTrue(plan["cuts"])
        for name in ("params.json", "plan.json", "edit-report.txt",
                     "expected.json"):
            self.assertTrue(os.path.isfile(os.path.join(root, name)), name)
        expected = json.load(open(os.path.join(root, "expected.json")))
        # The render stage reads these exact keys; a rename here breaks it.
        entry = expected["tracks"]["alice"]
        for key in ("passthrough", "mutes", "expected_samples",
                    "expected_duration", "sample_rate", "sample_fmt"):
            self.assertIn(key, entry)

    def test_clipping_the_speech_map_is_reported(self):
        buf = io.StringIO()
        log = runlog.Log(stream=buf, colour=False)
        # One word claiming twenty seconds, with sound for only half of one.
        words = {"a": [{"i": 0, "text": "x", "start": 1.0, "end": 21.0}]}
        pipeline.build_speech_map(
            ["a"], words, {"a": 30.0}, {"a": [[1.0, 1.5]]},
            pad=0.25, clip=True, log=log)
        self.assertIn("word timings ran past the audio", buf.getvalue())


class TestPipelinePrepareStage(unittest.TestCase):
    """Decoding, skipping what is done, and always re-measuring."""

    def _episode(self, sources):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        for leaf in ("prep", "state", "asr", "words", "llm", "render"):
            os.makedirs(os.path.join(root, leaf), exist_ok=True)
        tracks = []
        for participant, seconds in sources.items():
            path = os.path.join(root, f"{participant}.wav")
            with wave.open(path, "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(48000)
                handle.writeframes(b"\0\0" * int(48000 * seconds))
            tracks.append({
                "participant": participant, "source": path,
                # Deliberately wrong: the point of the measure step.
                "duration": seconds + 5.0,
                "sample_rate": 48000, "render_rate": 48000,
                "sample_fmt": "s16", "lossless": True,
            })
        meta = {"episode_id": "ep", "duration": max(sources.values()) + 5.0,
                "sample_rate": 48000, "durations_measured": False,
                "tracks": tracks}
        with open(os.path.join(root, "meta.json"), "w", encoding="utf-8") as handle:
            json.dump(meta, handle)
        return root

    def _log(self):
        buf = io.StringIO()
        return runlog.Log(stream=buf, colour=False), buf

    def _settings(self, **over):
        values = cfg.defaults()
        values.update(over)
        return values

    def test_decodes_and_replaces_the_container_duration(self):
        root = self._episode({"alice": 1.0})
        log, buf = self._log()
        pipeline.stage_prepare(root, self._settings(FFMPEG_JOBS="1"), log)
        self.assertTrue(os.path.isfile(os.path.join(root, "prep", "alice.wav")))
        self.assertTrue(os.path.isfile(os.path.join(root, "state", "prep-alice.ok")))
        meta = json.load(open(os.path.join(root, "meta.json")))
        self.assertTrue(meta["durations_measured"])
        # 1.0s of audio, not the 6.0s the fixture claimed.
        self.assertAlmostEqual(meta["tracks"][0]["duration"], 1.0, places=2)
        self.assertIn("container said", buf.getvalue())

    def test_an_already_decoded_track_is_skipped_but_still_measured(self):
        root = self._episode({"alice": 1.0})
        log, _ = self._log()
        pipeline.stage_prepare(root, self._settings(FFMPEG_JOBS="1"), log)

        # Reset the durations the way `discover` would, then resume.
        meta = json.load(open(os.path.join(root, "meta.json")))
        meta["tracks"][0]["duration"] = 99.0
        meta["durations_measured"] = False
        with open(os.path.join(root, "meta.json"), "w", encoding="utf-8") as handle:
            json.dump(meta, handle)

        # "already prepared" is a debug line, as it was in the shell.
        buf = io.StringIO()
        log = runlog.Log(level="debug", stream=buf, colour=False)
        pipeline.stage_prepare(root, self._settings(FFMPEG_JOBS="1"), log)
        self.assertIn("already prepared", buf.getvalue())
        again = json.load(open(os.path.join(root, "meta.json")))
        self.assertAlmostEqual(again["tracks"][0]["duration"], 1.0, places=2)

    def test_several_tracks_decode_in_parallel(self):
        root = self._episode({"alice": 0.5, "bob": 0.5, "carol": 0.5})
        log, buf = self._log()
        pipeline.stage_prepare(root, self._settings(FFMPEG_JOBS="3"), log)
        for name in ("alice", "bob", "carol"):
            self.assertTrue(
                os.path.isfile(os.path.join(root, "prep", f"{name}.wav")), name)
            self.assertIn(f"decoded {name}", buf.getvalue())

    def test_a_decode_failure_names_the_track(self):
        root = self._episode({"alice": 0.5})
        meta = json.load(open(os.path.join(root, "meta.json")))
        meta["tracks"][0]["source"] = os.path.join(root, "nothing-here.wav")
        with open(os.path.join(root, "meta.json"), "w", encoding="utf-8") as handle:
            json.dump(meta, handle)
        log, _ = self._log()
        with self.assertRaises(pipeline.StageError) as caught:
            pipeline.stage_prepare(root, self._settings(FFMPEG_JOBS="1"), log)
        self.assertIn("alice", str(caught.exception))


class TestRunLog(unittest.TestCase):
    """The console format.

    This used to be compared line for line against lib/log.sh, which wrote into
    the same file while the port was under way. That file is gone and there is
    one implementation, so the format is pinned to its own output instead: still
    a promise, just without a second implementation to check it against.

    One shared surface is left. The launcher prints an error of its own for the
    failures that happen before Python starts — an unknown option, a missing
    python3 — and that line has to look like every other error the run produces.
    """

    def _launcher(self, argv, env=None):
        """stderr from running the launcher itself, for real."""
        result = subprocess.run(
            [os.path.join(REPO_ROOT, "clean-podcast.sh"), *argv],
            capture_output=True, text=True, cwd=REPO_ROOT,
            env={**os.environ, "NO_COLOR": "1", "TERM": "dumb", **(env or {})},
            check=False)
        self.assertEqual(result.returncode, 1)
        return result.stderr

    def _as_error(self, message):
        buf = io.StringIO()
        runlog.Log(level="error", stream=buf, colour=False).error(message)
        return buf.getvalue()

    def test_the_launchers_error_looks_like_every_other_error(self):
        self.assertEqual(
            self._launcher(["--no-such-option"]),
            self._as_error("unknown option: --no-such-option (try --help)"))

    def test_the_launchers_missing_tool_message_is_the_python_one(self):
        # proc.require_bin says this once Python is running; the launcher has to
        # say it when the tool that is missing is Python itself.
        with self.assertRaises(proc.ToolError) as caught:
            proc.require_bin("python3", "definitely-not-installed-xyz")
        self.assertEqual(
            self._launcher(["--list-stages"],
                           env={"PYTHON_BIN": "definitely-not-installed-xyz"}),
            self._as_error(str(caught.exception)))

    def test_stage_headers_and_counter_keep_their_format(self):
        buf = io.StringIO()
        log = runlog.Log(level="info", stream=buf, colour=False)
        log.stage_total(3)
        log.stage_begin("prepare", "decoding tracks to 16 kHz mono")
        log.ok("decoded host")
        log.progress(1, 2, "host")
        log.line("")
        log.line("  a header")
        log.report("first\nsecond")
        log.stage_skip("detect", "LLM_ENABLE=0")
        self.assertEqual(buf.getvalue(), (
            "[1/3] prepare  decoding tracks to 16 kHz mono\n"
            "  \u2713 decoded host\n"
            "      (1/2) host\n"
            "\n"
            "  a header\n"
            "      first\n"
            "      second\n"
            "[2/3] detect \u2014 skipped (LLM_ENABLE=0)\n"
        ))

    def test_stage_end_reports_a_duration(self):
        buf = io.StringIO()
        log = runlog.Log(level="info", stream=buf, colour=False)
        log.stage_total(1)
        log.stage_begin("plan")
        log.stage_end("plan written")
        self.assertIn("      plan written in 0s\n", buf.getvalue())

    def test_duration_formatting(self):
        for seconds, want in ((0, "0s"), (5, "5s"), (59, "59s"), (60, "1m00s"),
                              (61, "1m01s"), (3599, "59m59s"), (3600, "1h00m00s"),
                              (3661, "1h01m01s"), (86399, "23h59m59s"),
                              (-5, "0s")):
            with self.subTest(seconds=seconds):
                self.assertEqual(runlog.fmt_duration(seconds), want)

    def test_the_log_file_carries_level_and_stage(self):
        path = os.path.join(tempfile.mkdtemp(), "run.log")
        self.addCleanup(shutil.rmtree, os.path.dirname(path), ignore_errors=True)
        log = runlog.Log(path=path, stream=io.StringIO(), colour=False)
        log.stage_name = "discover"
        log.warn("two episodes at once")
        self.assertIn("[warn] (discover) two episodes at once",
                      open(path, encoding="utf-8").read())

    def test_from_env_adopts_the_launchers_log(self):
        path = os.path.join(tempfile.mkdtemp(), "run.log")
        self.addCleanup(shutil.rmtree, os.path.dirname(path), ignore_errors=True)
        log = runlog.Log.from_env(
            environ={"PODCAST_LOG_FILE": path, "LOG_LEVEL": "warn",
                     "PODCAST_LOG_STAGE": "prepare"},
            stream=io.StringIO())
        self.assertEqual(log.level, "warn")
        self.assertEqual(log.stage_name, "prepare")
        self.assertFalse(log.enabled("info"))

    def test_no_path_is_a_no_op_not_a_crash(self):
        runlog.Log(stream=io.StringIO(), colour=False).warn("nowhere to write")

class TestDiscover(unittest.TestCase):
    """Which files are one episode, and what the refusals say."""

    def _dir(self, *names):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        for name in names:
            open(os.path.join(root, name), "w").close()
        return root

    def test_finds_by_extension_case_insensitively(self):
        root = self._dir("ep1_a.flac", "ep1_b.FLAC", "notes.txt", "ep1_c.wav")
        found = discover.find_tracks(root, ["flac"])
        self.assertEqual([os.path.basename(p) for p in found],
                         ["ep1_a.flac", "ep1_b.FLAC"])

    def test_an_empty_inbox_is_not_a_failure(self):
        with self.assertRaises(discover.NothingToDo):
            discover.find_tracks(self._dir("notes.txt"), ["flac"])
        with self.assertRaises(discover.NothingToDo):
            discover.find_tracks("/nowhere/at/all", ["flac"])

    def test_splits_on_the_first_separator(self):
        episode, tracks = discover.parse_tracks(
            ["/in/ep_1_bob.flac"], "_")
        self.assertEqual(episode, "ep")
        self.assertEqual(list(tracks), ["1_bob"])

    def test_two_episodes_at_once_is_refused_by_name(self):
        with self.assertRaises(discover.DiscoverError) as caught:
            discover.parse_tracks(["/in/ep1_a.flac", "/in/ep2_b.flac"], "_")
        self.assertIn("ep1", str(caught.exception))
        self.assertIn("ep2", str(caught.exception))

    def test_episode_override_unifies_them(self):
        episode, tracks = discover.parse_tracks(
            ["/in/ep1_a.flac", "/in/ep2_b.flac"], "_", episode_override="only")
        self.assertEqual(episode, "only")
        self.assertEqual(sorted(tracks), ["a", "b"])

    def test_the_same_participant_twice_names_both_files(self):
        with self.assertRaises(discover.DiscoverError) as caught:
            discover.parse_tracks(["/in/ep1_a.flac", "/in/ep1_a.wav"], "_")
        message = str(caught.exception)
        self.assertIn("ep1_a.flac", message)
        self.assertIn("ep1_a.wav", message)
        self.assertIn("INPUT_EXTS", message)

    def test_a_name_without_the_separator_says_the_shape_wanted(self):
        with self.assertRaises(discover.DiscoverError) as caught:
            discover.parse_tracks(["/in/justaname.flac"], "_")
        self.assertIn("<episode>_<participant>", str(caught.exception))

    def test_empty_halves_are_refused(self):
        for name in ("_bob.flac", "ep1_.flac"):
            with self.subTest(name=name):
                with self.assertRaises(discover.DiscoverError):
                    discover.parse_tracks([f"/in/{name}"], "_")

    def test_a_dash_separator_works_too(self):
        episode, tracks = discover.parse_tracks(
            ["/in/sample-host.flac", "/in/sample-guest.flac"], "-")
        self.assertEqual(episode, "sample")
        self.assertEqual(sorted(tracks), ["guest", "host"])

    def test_episode_paths_and_tree(self):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        where = discover.episode_paths(
            "ep1", os.path.join(root, "work"), os.path.join(root, "out"))
        discover.make_work_tree(where)
        for leaf in discover.WORK_SUBDIRS:
            self.assertTrue(os.path.isdir(os.path.join(where["work"], leaf)), leaf)
        self.assertTrue(os.path.isdir(where["output"]))


class TestConfig(unittest.TestCase):
    """The settings layer: where a value comes from and what shape it must have."""

    def _conf(self, body):
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".conf", delete=False, encoding="utf-8")
        handle.write(body)
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        return handle.name

    def test_defaults_cover_every_setting(self):
        values = cfg.defaults()
        self.assertEqual(set(values), set(cfg.SETTINGS))
        self.assertTrue(values["FFMPEG_JOBS"].isdigit())

    def _dumped(self, settings):
        """The dump's log lines, as `name -> value`."""
        path = os.path.join(tempfile.mkdtemp(), "run.log")
        self.addCleanup(shutil.rmtree, os.path.dirname(path), True)
        cfg.dump(settings, runlog.Log(path, level="error", stream=io.StringIO()))
        with open(path, encoding="utf-8") as handle:
            body = handle.read()
        out = {}
        for line in body.splitlines():
            _, _, rest = line.partition("  config ")
            if rest:
                name, _, value = rest.partition("=")
                out[name] = value
        return out, body

    def test_dump_covers_every_setting(self):
        # The shell kept this list by hand and it fell six names behind, so a
        # setting could change a run without the log showing it. Driven by
        # SETTINGS now; this fails if that ever stops being true.
        dumped, _ = self._dumped(cfg.defaults())
        self.assertEqual(set(dumped), set(cfg.SETTINGS))

    def test_dump_redacts_the_api_key(self):
        settings = cfg.defaults()
        settings["LLAMA_API_KEY"] = "sk-secret-value"
        dumped, body = self._dumped(settings)
        # Present, and its length, but never the key: this log is copied into
        # the output directory and outlives everything else in the run.
        self.assertNotIn("sk-secret-value", body)
        self.assertEqual(dumped["LLAMA_API_KEY"],
                         f"<set, {len('sk-secret-value')} chars, redacted>")

    def test_dump_says_so_when_no_key_is_set(self):
        dumped, _ = self._dumped(cfg.defaults())
        self.assertEqual(dumped["LLAMA_API_KEY"], "<unset>")

    def test_config_file_is_sourced_by_bash(self):
        # Not parsed here: a value referencing another one has to keep working.
        path = self._conf('SILENCE_KEEP="0.2"\nOUTPUT_EXT="wav"\n'
                          'OUTPUT_CODEC="pcm_${OUTPUT_EXT}"\n')
        values = cfg.load(config_file=path, environ={})
        self.assertEqual(values["SILENCE_KEEP"], "0.2")
        self.assertEqual(values["OUTPUT_CODEC"], "pcm_wav")

    def test_environment_beats_the_file_and_the_cli_beats_both(self):
        path = self._conf('SILENCE_KEEP="0.2"\n')
        self.assertEqual(
            cfg.load(config_file=path, environ={})["SILENCE_KEEP"], "0.2")
        self.assertEqual(
            cfg.load(config_file=path,
                     environ={"SILENCE_KEEP": "0.3"})["SILENCE_KEEP"], "0.3")
        self.assertEqual(
            cfg.load(config_file=path, environ={"SILENCE_KEEP": "0.3"},
                     overrides={"SILENCE_KEEP": "0.4"})["SILENCE_KEEP"], "0.4")

    def test_a_name_the_file_does_not_set_falls_through(self):
        path = self._conf('SILENCE_KEEP="0.2"\n')
        values = cfg.load(config_file=path, environ={"MIN_CUT": "0.9"})
        self.assertEqual(values["MIN_CUT"], "0.9")

    def test_root_override_re_derives_the_layout(self):
        path = self._conf('PODCAST_ROOT="/from/file"\nINPUT_DIR="/from/file/in"\n')
        # What the launcher sends for --root: the root, and the four cleared.
        values = cfg.load(config_file=path, environ={}, overrides={
            "PODCAST_ROOT": "/cli", "INPUT_DIR": "", "OUTPUT_DIR": "",
            "WORK_ROOT": "", "FAILED_DIR": "",
        })
        cfg.resolve_paths(values, "/script")
        self.assertEqual(values["INPUT_DIR"], "/cli/incoming")

    def test_paths_fall_back_to_the_script_directory(self):
        values = cfg.load(environ={})
        values["PODCAST_ROOT"] = ""
        cfg.resolve_paths(values, "/opt/app")
        self.assertEqual(values["PODCAST_ROOT"], "/opt/app")
        self.assertEqual(values["WORK_ROOT"], "/opt/app/work")

    def test_validation_rejects_what_the_shell_rejected(self):
        for override, expected in [
            ({"SILENCE_KEEP": "9"}, "SILENCE_KEEP"),
            ({"LLM_API": "grpc"}, "LLM_API"),
            ({"RENDER_FRAME_SAMPLES": "3"}, "RENDER_FRAME_SAMPLES"),
            ({"CUT_PADDING": "abc"}, "CUT_PADDING"),
            ({"RESAMPLE_TO": "44"}, "RESAMPLE_TO"),
            ({"LLM_CHUNK_OVERLAP": "900"}, "LLM_CHUNK_OVERLAP"),
            ({"TRACK_SEPARATOR": ""}, "TRACK_SEPARATOR"),
            ({"FAILED_ACTION": "burn"}, "FAILED_ACTION"),
        ]:
            with self.subTest(override=override):
                values = cfg.load(environ={}, overrides=override)
                with self.assertRaises(cfg.ConfigError) as caught:
                    cfg.validate(values)
                self.assertIn(expected, str(caught.exception))

    def test_a_clean_configuration_validates(self):
        values = cfg.load(environ={})
        cfg.resolve_paths(values, "/opt/app")
        cfg.validate(values)

    def test_deprecated_track_ext_becomes_an_input_filter(self):
        values = cfg.load(environ={}, overrides={"TRACK_EXT": "wav"})
        said = []
        cfg.validate(values, warn=said.append)
        self.assertEqual(values["INPUT_EXTS"], "wav")
        self.assertTrue(any("TRACK_EXT is deprecated" in m for m in said))

    def test_key_file_is_read_and_trimmed(self):
        path = self._conf("  s3cret  \n")
        values = cfg.load(environ={}, overrides={"LLAMA_API_KEY_FILE": path})
        cfg.resolve_api_keys(values)
        self.assertEqual(values["LLAMA_API_KEY"], "s3cret")

    def test_empty_key_file_is_an_error(self):
        path = self._conf("\n")
        values = cfg.load(environ={}, overrides={"LLAMA_API_KEY_FILE": path})
        with self.assertRaises(cfg.ConfigError):
            cfg.resolve_api_keys(values)

    def test_shell_output_survives_a_hostile_value(self):
        values = cfg.load(environ={}, overrides={
            "WHISPER_PROMPT": "it's \"quoted\"; rm -rf /"})
        emitted = cfg.to_shell(values)
        probe = subprocess.run(
            ["bash", "-c", f'{emitted}\nprintf "%s" "$WHISPER_PROMPT"'],
            capture_output=True, text=True, check=True)
        self.assertEqual(probe.stdout, "it's \"quoted\"; rm -rf /")



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
        # "I I" was reported as both words; the second is given back, so the cut
        # covers only word 0 (10.0-10.3). The gap to the surviving "I" is 0.1s,
        # half of which caps the padding at 0.05s.
        self.assertAlmostEqual(stutter["end"], 10.35, places=3)
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
        # Trimmed to the first "I" for the same reason as the cut case above:
        # muting both would leave the sentence without the word either way.
        self.assertAlmostEqual(mute["end"], 10.35, places=3)
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

    def test_no_words_at_all_is_flagged(self):
        """An empty speech map now means an empty transcript, so it says so."""
        result = planner.build_plan(_meta(["a"], 30.0), {"a": []}, {}, {}, PARAMS)
        warning = " ".join(result["warnings"])
        self.assertIn("transcribed a single word", warning)
        self.assertIn("whisper endpoint", warning)

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

    def test_resample_alone_still_needs_a_graph(self):
        graph = render.build_filter([], [], 512, 0.03, resample=48000)
        self.assertIsNotNone(graph)
        self.assertIn("aresample=48000", graph)
        # Nothing is being cut, so there is no reason to re-chunk the frames.
        self.assertNotIn("asetnsamples", graph)
        self.assertNotIn("aselect", graph)

    def test_resample_comes_before_frames_are_fixed(self):
        """The ordering the whole sync guarantee rests on.

        Cuts are decided per frame. If each track were resampled after its
        frames were fixed, every track would quantise at its own rate and an
        identical cut list would remove slightly different spans from each.
        """
        graph = render.build_filter(
            [{"start": 1.0, "end": 2.0}],
            [{"start": 5.0, "end": 5.5}],
            512, 0.03, resample=48000,
        )
        order = [
            graph.index("aresample="),
            graph.index("asetnsamples="),
            graph.index("volume="),
            graph.index("aselect="),
            graph.index("asetpts="),
        ]
        self.assertEqual(order, sorted(order), f"wrong filter order: {graph}")

    def test_expected_samples_follow_the_render_rate(self):
        """A resampled track's length is predicted at its output rate."""
        cuts = [{"start": 1.0, "end": 2.0}]
        at_44 = render.expected_output_samples(10.0, 44100, 512, cuts)
        at_48 = render.expected_output_samples(10.0, 48000, 512, cuts)
        # Same span removed either way, so the durations agree even though the
        # sample counts do not.
        self.assertAlmostEqual(at_44 / 44100, at_48 / 48000, places=2)

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
                      "sources": ["silence"], "details": []}],
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
