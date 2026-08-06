#!/usr/bin/env python3
"""A stand-in for llama-server, for the self-test.

llama-server is not available in a test environment, but the *client* is worth
testing against something that speaks the same HTTP: response shape, endpoint
probing, and the stage wiring around them.

    stub_servers.py llama --responses r.json --port-file port.txt

`--responses` is a JSON array handed out one entry per POST, in order. Requests
past the end of the list get the final entry again. Each request is appended to
`--request-log` as one JSON line, so the test can assert on what was sent.

There was a whisper role here too, until transcription stopped being a server.
Its replacement is tests/fake_whisperx/, which is imported rather than dialled.
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import sys
import threading


def _uploaded_filename(raw: bytes) -> str:
    """The filename from the multipart file part, or "" if there is none."""
    marker = b'name="file"; filename="'
    at = raw.find(marker)
    if at < 0:
        return ""
    rest = raw[at + len(marker):]
    return rest[: rest.find(b'"')].decode("utf-8", errors="replace")


def _field_values(raw: bytes) -> dict:
    """The plain (non-file) form fields as {name: value}."""
    values = {}
    for part in raw.split(b"Content-Disposition: form-data; ")[1:]:
        head, _, body = part.partition(b"\r\n\r\n")
        if b"filename=" in head:
            continue
        name = head.split(b'name="', 1)[1].split(b'"', 1)[0].decode()
        values[name] = body.split(b"\r\n--")[0].decode("utf-8", errors="replace")
    return values


def build_handler(role, responses, request_log, api_key=None):
    counter = {"n": 0}
    lock = threading.Lock()

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send_json(self, status, payload):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorised(self):
            """Reject like llama-server's --api-key does, on every route."""
            if not api_key:
                return True
            if self.headers.get("Authorization") == f"Bearer {api_key}":
                return True
            # Drain any body first, or the client sees a closed connection
            # instead of the 401.
            length = int(self.headers.get("Content-Length", 0))
            if length:
                self.rfile.read(length)
            self._send_json(401, {"error": {"message": "invalid api key"}})
            return False

        def do_GET(self):
            if not self._authorised():
                return
            # llama-server answers /health; whisper-server has no such endpoint,
            # and its /inference rejects GET — both are treated as "alive".
            if role == "llama" and self.path == "/health":
                self._send_json(200, {"status": "ok"})
            elif role == "whisper" and self.path == "/inference":
                self._send_json(405, {"error": "POST only"})
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self):
            if not self._authorised():
                return
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            filename = _uploaded_filename(raw)
            with lock:
                index = counter["n"]
                counter["n"] += 1
                if isinstance(responses, dict):
                    # Keyed by filename: a resumed or re-run stage must get the
                    # same answer for the same track, not the next one in line.
                    reply = responses.get(filename)
                    if reply is None:
                        self._send_json(
                            404, {"error": f"no canned reply for '{filename}'"}
                        )
                        return
                elif responses:
                    reply = responses[min(index, len(responses) - 1)]
                else:
                    reply = {}
                if request_log:
                    record = {
                        "index": index,
                        "path": self.path,
                        "content_type": self.headers.get("Content-Type", ""),
                        "length": length,
                    }
                    if role == "llama":
                        try:
                            record["payload"] = json.loads(raw)
                        except ValueError:
                            record["payload"] = None
                    else:
                        # Note the audio's size rather than its contents, and
                        # confirm the file part actually arrived.
                        record["has_file_part"] = b'name="file"' in raw
                        record["filename"] = filename
                        record["fields"] = sorted(
                            part.split(b'name="', 1)[1].split(b'"', 1)[0].decode()
                            for part in raw.split(b"Content-Disposition: form-data; ")[1:]
                        )
                        # Values of the non-file parts, so a test can assert that
                        # e.g. vad=true really travelled rather than just that a
                        # field by that name existed.
                        record["values"] = _field_values(raw)
                    with open(request_log, "a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record) + "\n")

            if role == "llama":
                # The two endpoints wrap the same text differently: /completion
                # puts it in "content", /v1/chat/completions in the OpenAI
                # choices/message envelope. Answer in the shape that was asked
                # for, so a client sending to the wrong one is not quietly
                # rewarded with a well-formed reply.
                text = json.dumps(reply)
                if self.path.startswith("/v1/chat/completions"):
                    self._send_json(200, {
                        "choices": [{"message": {"role": "assistant", "content": text}}]
                    })
                else:
                    self._send_json(200, {"content": text})
            else:
                self._send_json(200, reply)

        def log_message(self, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=["llama"])
    parser.add_argument("--responses", help="JSON array of canned replies")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--port-file", required=True)
    parser.add_argument("--request-log")
    parser.add_argument(
        "--api-key",
        help="require this key as 'Authorization: Bearer <key>', like "
             "llama-server's own --api-key",
    )
    args = parser.parse_args()

    responses = []
    if args.responses:
        with open(args.responses, encoding="utf-8") as handle:
            responses = json.load(handle)

    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", args.port),
        build_handler(args.role, responses, args.request_log, args.api_key),
    )
    port = server.server_address[1]

    # Written last and atomically, so the test can treat its appearance as the
    # signal that the port is accepting connections.
    temporary = args.port_file + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(str(port))
    os.replace(temporary, args.port_file)

    print(f"stub {args.role} server on 127.0.0.1:{port}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
