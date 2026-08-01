"""Disfluency detection via llama.cpp.

Two decisions shape this module:

1. The model is asked for *word indices*, never timestamps. Indices are checked
   against the transcript we already hold, so a hallucinated number is caught
   instead of becoming a cut in the wrong place. Timings come from Whisper.

2. Output is constrained by a JSON schema server-side, so there is no regex
   scraping of prose. A chunk whose response still fails to parse is dropped
   with a warning rather than failing the run: losing a few stutters is a much
   better outcome than losing the episode.

The server is expected to be up already — process lifetime is the shell's job,
because Whisper must be finished and gone before the model is loaded.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from . import transcript as tr

KINDS = ("stutter", "repetition", "false_start", "filler")

KIND_GUIDANCE = {
    "stutter": (
        'a word or syllable broken off and restarted — "I-I-I think", "wh- what"'
    ),
    "repetition": (
        "the same word or short phrase accidentally said twice in a row — "
        '"the the plan", "we we should"'
    ),
    "false_start": (
        "an abandoned phrase the speaker immediately replaces — "
        '"I went to- actually we drove"'
    ),
    "filler": 'a hesitation sound carrying no meaning — "um", "uh", "er"',
}

_EXAMPLE = """\
EXAMPLE (its own numbering, unrelated to the transcript below)

  0 So
  1 I
  2 I
  3 I
  4 think
  5 we
  6 should
  7 go
  8 to
  9 the
 10 uh
 11 the
 12 market
 13 on
 14 Saturday
 15 very
 16 very
 17 early

Correct answer for that example:

{"edits": [{"first": 1, "last": 2, "kind": "stutter", "confidence": 0.95},
           {"first": 10, "last": 11, "kind": "false_start", "confidence": 0.8}]}

Note what was *not* removed: word 3 is the attempt that succeeded, and 15-16
("very very") is deliberate emphasis, not an accident.
"""


def response_schema(kinds) -> dict:
    return {
        "type": "object",
        "properties": {
            "edits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "first": {"type": "integer", "minimum": 0},
                        "last": {"type": "integer", "minimum": 0},
                        "kind": {"type": "string", "enum": list(kinds)},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["first", "last", "kind", "confidence"],
                },
            }
        },
        "required": ["edits"],
    }


def build_prompt(words, first: int, last: int, kinds, max_edit_words: int) -> str:
    """Prompt for one window, [first, last] inclusive over the global indices."""
    listing = "\n".join(
        f"{w['i']:6d} {w['text']}" for w in words[first : last + 1]
    )
    plain = " ".join(w["text"] for w in words[first : last + 1])
    removable = "\n".join(f"- {kind}: {KIND_GUIDANCE[kind]}" for kind in kinds)

    return f"""\
You are a meticulous podcast editor working on one speaker's track.

Your task: find spans of words that should be removed because they are speech
disfluencies rather than content.

Remove only these kinds:
{removable}

Never remove:
- deliberate emphasis or rhetorical repetition
- any word the sentence still needs to make sense
- names, numbers, quotations, or items in a list
- anything you are not sure about

Rules:
- Report spans of consecutive word indices, using the numbering given below.
- Keep every span minimal: cut the broken attempt, keep the completed one.
- A span may cover at most {max_edit_words} words.
- Spans must not overlap each other.
- "confidence" is 0.0-1.0: how sure you are that this was accidental.
- Reporting nothing is the correct answer for clean speech. Prefer an empty
  list over a doubtful span.
- The transcript may be in any language. Judge it in its own language.

{_EXAMPLE}
Now the real transcript. Reading it as continuous speech first:

{plain}

The same words, one per line, with the indices you must use
(they start at {first} and end at {last}):

{listing}

Reply with JSON only.
"""


def plan_chunks(word_count: int, chunk_words: int, overlap: int):
    """Window boundaries covering every word, with overlap for context.

    Overlap means the same disfluency can be reported twice; duplicates are
    merged later by index range.
    """
    if word_count == 0:
        return []
    stride = max(1, chunk_words - overlap)
    chunks = []
    start = 0
    while start < word_count:
        end = min(start + chunk_words, word_count) - 1
        chunks.append((start, end))
        if end >= word_count - 1:
            break
        start += stride
    return chunks


class AuthRejected(Exception):
    """The server refused our credentials. Retrying cannot help."""


class EndpointError(OSError):
    """The server rejected the request and said why.

    Subclasses OSError so the per-chunk handler in detect() still treats it as
    a droppable failure — a chunk that overflows the context is a real per-item
    error — while carrying the server's own words instead of a bare status code.
    A misconfiguration that would reject *every* chunk is caught earlier, by
    check_schema_support.
    """


class SchemaIgnored(Exception):
    """The server answered but did not honour the JSON schema.

    Worth its own type because the consequence is silent: unconstrained prose
    fails to parse, every chunk is dropped, and the episode finishes reporting
    no edits — which looks exactly like clean speech.
    """


class LlamaClient:
    """Client for a llama.cpp server, whichever model it happens to be serving.

    Requests go to /v1/chat/completions by default, so the server applies the
    chat template of the model it has loaded. That is the whole reason to prefer
    it: /completion takes a raw prompt and applies no template, so every model
    sees an instruction formatted for none of them, and how well it copes is a
    property of the model rather than of this code. Routing through the chat
    endpoint is what lets the model be swapped or upgraded without touching
    anything here.

    api="completion" keeps the old raw-prompt path for a server that lacks the
    chat endpoint, or to compare the two against the same episode.
    """

    def __init__(
        self,
        endpoint: str,
        timeout: float = 600.0,
        temperature: float = 0.0,
        api_key: str | None = None,
        api: str = "chat",
        max_reply_tokens: int = 2048,
        model: str | None = None,
    ):
        if api not in ("chat", "completion"):
            raise ValueError(f"api must be 'chat' or 'completion', got {api!r}")
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.api_key = api_key or None
        self.api = api
        self.max_reply_tokens = max_reply_tokens
        # Which model to ask for by name. A single-model llama-server ignores
        # this; one in router mode serves several and refuses a request that
        # does not name one, so it is not optional there.
        self.model = model or None

    def _headers(self, **extra) -> dict:
        headers = dict(extra)
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post(self, path: str, payload: dict, timeout: float | None = None) -> dict:
        request = urllib.request.Request(
            f"{self.endpoint}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(**{"Content-Type": "application/json"}),
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=timeout or self.timeout
            ) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                exc.close()  # or its buffered body leaks a ResourceWarning
                raise AuthRejected(
                    f"the llama endpoint rejected our credentials (HTTP {exc.code}). "
                    + (
                        "Check LLAMA_API_KEY against the server's --api-key."
                        if self.api_key
                        else "It requires an API key; set LLAMA_API_KEY or "
                        "LLAMA_API_KEY_FILE."
                    )
                ) from exc
            # Any other refusal: keep what the server actually said. "HTTP Error
            # 400: Bad Request" alone sends you reading code instead of config.
            detail = _error_detail(exc)
            raise EndpointError(
                f"the llama endpoint refused the request (HTTP {exc.code})"
                + (f": {detail}" if detail else "")
            ) from exc

    def available_models(self) -> list[str]:
        """Model ids the server admits to, for naming them in an error."""
        try:
            request = urllib.request.Request(
                f"{self.endpoint}/v1/models", headers=self._headers()
            )
            with urllib.request.urlopen(request, timeout=10) as resp:
                listing = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return []
        return [
            entry["id"] for entry in listing.get("data", []) if entry.get("id")
        ]

    def wait_until_ready(self, timeout: float, poll: float = 2.0) -> bool:
        """Block until the server reports it can serve, or the timeout expires.

        A loading model answers /health with 503, so a refused connection and a
        busy server are treated the same way: keep waiting. A rejected key is
        different — no amount of waiting fixes it, so it fails at once rather
        than after the full timeout.
        """
        deadline = time.monotonic() + timeout
        last_error = "no response"
        while time.monotonic() < deadline:
            for path in ("/health", "/v1/models"):
                request = urllib.request.Request(
                    f"{self.endpoint}{path}", headers=self._headers()
                )
                try:
                    with urllib.request.urlopen(request, timeout=10) as resp:
                        if resp.status == 200:
                            return True
                except urllib.error.HTTPError as exc:
                    if exc.code in (401, 403):
                        print(
                            f"llama endpoint rejected our credentials (HTTP {exc.code}) "
                            f"on {path}: "
                            + (
                                "check LLAMA_API_KEY against the server's --api-key"
                                if self.api_key
                                else "it wants an API key; set LLAMA_API_KEY or "
                                "LLAMA_API_KEY_FILE"
                            )
                        )
                        return False
                    last_error = f"HTTP {exc.code} from {path}"
                except Exception as exc:  # connection refused, DNS, timeout
                    last_error = f"{type(exc).__name__} on {path}"
            time.sleep(poll)
        print(f"llama endpoint not ready: {last_error}")
        return False

    def complete(self, prompt: str, schema: dict, max_tokens: int | None = None) -> str:
        budget = max_tokens or self.max_reply_tokens
        if self.api == "completion":
            result = self._post(
                "/completion",
                {
                    "prompt": prompt,
                    "n_predict": budget,
                    "temperature": self.temperature,
                    "top_k": 1,
                    "cache_prompt": True,
                    "json_schema": schema,
                },
            )
            return result.get("content", "")

        result = self._post(
            "/v1/chat/completions", self._chat_payload(prompt, schema, budget)
        )
        return _chat_content(result)

    def _chat_payload(self, prompt: str, schema: dict, budget: int) -> dict:
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": budget,
            "temperature": self.temperature,
            "top_k": 1,
            "cache_prompt": True,
            # Two spellings of the same constraint. llama.cpp has documented
            # both the nested OpenAI form and a flat "schema" shorthand, and
            # which one a given build reads has moved around; sending both means
            # a build that understands either is constrained, and neither form
            # upsets a build that ignores it. check_schema_support() exists
            # because a build that reads *neither* would otherwise fail silently.
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "edits", "schema": schema},
                "schema": schema,
            },
        }
        if self.model:
            payload["model"] = self.model
        return payload

    def check_schema_support(self) -> None:
        """Confirm the server actually constrains output to the schema.

        Sent once before an episode. If the schema is not applied the reply is
        prose, every chunk of the real run fails to parse and is dropped, and
        the episode reports no edits at all — indistinguishable from a clean
        recording. Better to spend one tiny request finding out.
        """
        schema = response_schema(KINDS)
        probe = (
            "Reply with JSON only, matching the required schema: an object with "
            'an "edits" key holding an empty array.'
        )
        try:
            content = self.complete(probe, schema, max_tokens=128)
        except AuthRejected:
            raise
        except Exception as exc:
            # A server in router mode serves several models and refuses a
            # request that does not name one — the commonest way this fails, and
            # not obvious from the wording alone, so the setting and the actual
            # choices are spelled out.
            if "model" in str(exc).lower():
                offered = self.available_models()
                raise SchemaIgnored(
                    f"{exc}. This server wants the model named in the request; "
                    "set LLAMA_MODEL_NAME"
                    + (f" to one of: {', '.join(offered)}" if offered else "")
                    + ". A model listed but not loaded also has to be loaded "
                    "first, unless the server was started with model autoload."
                ) from exc
            raise SchemaIgnored(
                f"could not reach the llama endpoint for a schema check: {exc}"
            ) from exc

        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, TypeError) as exc:
            raise SchemaIgnored(
                "the llama endpoint did not constrain its reply to the JSON "
                f"schema (it answered {content[:120]!r}). Its build may not "
                "support response_format; set LLM_API=completion to use the "
                "raw /completion endpoint instead."
            ) from exc
        if not isinstance(parsed, dict) or "edits" not in parsed:
            raise SchemaIgnored(
                "the llama endpoint returned JSON that does not match the "
                f"schema ({content[:120]!r}); set LLM_API=completion to use the "
                "raw /completion endpoint instead."
            )


def _error_detail(exc: urllib.error.HTTPError) -> str:
    """The message out of an error body, in whichever shape it arrived."""
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:
        return ""
    finally:
        exc.close()  # or the buffered body leaks a ResourceWarning
    try:
        parsed = json.loads(body)
    except ValueError:
        return body.strip()[:200]
    error = parsed.get("error", parsed)
    if isinstance(error, dict):
        return str(error.get("message") or error)[:200]
    return str(error)[:200]


def _chat_content(result: dict) -> str:
    """The assistant text of a chat completion, whatever the server wrapped it in."""
    choices = result.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if content:
        return content
    # A reasoning model with reasoning_format=none puts everything in
    # reasoning_content and leaves content empty. The schema still applies to
    # what it generated, so it is worth reading rather than discarding.
    return message.get("reasoning_content") or ""


def _validate(raw_edits, words, first, last, limits, accepted):
    """Keep only edits that survive every sanity check, with a reason per drop."""
    kept, rejected = [], []

    def reject(edit, why):
        rejected.append({"edit": edit, "reason": why})

    for edit in raw_edits:
        try:
            lo = int(edit["first"])
            hi = int(edit["last"])
            kind = str(edit["kind"])
            confidence = float(edit.get("confidence", 0.0))
        except (KeyError, TypeError, ValueError):
            reject(edit, "malformed")
            continue

        if kind not in accepted:
            reject(edit, f"kind '{kind}' not accepted")
            continue
        if lo > hi:
            reject(edit, "first > last")
            continue
        if lo < first or hi > last:
            reject(edit, "index outside the window given to the model")
            continue
        if hi - lo + 1 > limits["max_words"]:
            reject(edit, f"{hi - lo + 1} words exceeds max_words")
            continue
        if confidence < limits["min_confidence"]:
            reject(edit, f"confidence {confidence:.2f} below threshold")
            continue

        start, end = tr.word_span(words, lo, hi)
        if end - start > limits["max_seconds"]:
            reject(edit, f"span {end - start:.2f}s exceeds max_seconds")
            continue
        if end <= start:
            reject(edit, "empty time span")
            continue

        kept.append(
            {
                "first": lo,
                "last": hi,
                "kind": kind,
                "confidence": round(confidence, 3),
                "start": round(start, 3),
                "end": round(end, 3),
                "text": tr.word_text(words, lo, hi),
            }
        )
    return kept, rejected


def _dedupe(edits):
    """Fuse edits whose word ranges overlap, keeping the most confident label."""
    ordered = sorted(edits, key=lambda e: (e["first"], e["last"]))
    out: list[dict] = []
    for edit in ordered:
        if out and edit["first"] <= out[-1]["last"]:
            previous = out[-1]
            merged = {
                "first": previous["first"],
                "last": max(previous["last"], edit["last"]),
                "kind": (
                    previous["kind"]
                    if previous["confidence"] >= edit["confidence"]
                    else edit["kind"]
                ),
                "confidence": max(previous["confidence"], edit["confidence"]),
            }
            out[-1] = merged
        else:
            out.append(dict(edit))
    return out


def detect(
    client: LlamaClient,
    parsed,
    *,
    chunk_words: int,
    overlap: int,
    limits: dict,
    accepted,
    audit_path: str | None = None,
    retries: int = 2,
    on_progress=None,
) -> dict:
    """Run detection over one participant's transcript."""
    words = parsed["words"]
    chunks = plan_chunks(len(words), chunk_words, overlap)
    schema = response_schema(accepted)

    collected: list[dict] = []
    rejected: list[dict] = []
    failures = 0
    audit = open(audit_path, "w", encoding="utf-8") if audit_path else None

    try:
        for index, (first, last) in enumerate(chunks):
            if on_progress:
                on_progress(index + 1, len(chunks))
            prompt = build_prompt(words, first, last, accepted, limits["max_words"])

            content = None
            for attempt in range(retries + 1):
                try:
                    content = client.complete(prompt, schema)
                    parsed_response = json.loads(content)
                    break
                # Deliberately not caught: AuthRejected. Retrying a refused key
                # only wastes time, and dropping the chunk would turn a
                # misconfiguration into an episode that quietly found no edits.
                except (
                    urllib.error.URLError,
                    json.JSONDecodeError,
                    TimeoutError,
                    OSError,
                ) as exc:
                    if attempt == retries:
                        failures += 1
                        print(
                            f"  chunk {index + 1}/{len(chunks)} "
                            f"(words {first}-{last}) failed: {exc}"
                        )
                        parsed_response = {"edits": []}
                    else:
                        time.sleep(1.5 * (attempt + 1))

            raw_edits = parsed_response.get("edits") or []
            kept, dropped = _validate(raw_edits, words, first, last, limits, accepted)
            collected.extend(kept)
            rejected.extend(dropped)

            if audit:
                audit.write(
                    json.dumps(
                        {
                            "chunk": index,
                            "words": [first, last],
                            "raw": raw_edits,
                            "accepted": len(kept),
                            "rejected": dropped,
                            "content": content if not kept and content else None,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    finally:
        if audit:
            audit.close()

    merged = _dedupe(collected)
    # Re-derive timings and text after merging, so the record matches the range.
    for edit in merged:
        start, end = tr.word_span(words, edit["first"], edit["last"])
        edit["start"] = round(start, 3)
        edit["end"] = round(end, 3)
        edit["text"] = tr.word_text(words, edit["first"], edit["last"])

    return {
        "participant": parsed["participant"],
        "chunks": len(chunks),
        "chunk_failures": failures,
        "edits": merged,
        "rejected_count": len(rejected),
    }
