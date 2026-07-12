"""Stdlib OpenAI-compatible HTTP proxy for an Ollama upstream."""

import json
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from gateway.middleware import ShortCircuit
from gateway.usage import StreamUsageAccumulator, extract_usage


class GatewayServer(ThreadingHTTPServer):
    """Proxy OpenAI chat-completion requests through a middleware pipeline."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, upstream_url, middlewares=(), ledger=None, port=8000, shadow=None):
        self.upstream_url = str(upstream_url).rstrip("/")
        self.middlewares = tuple(middlewares)
        self.ledger = ledger
        # Optional gateway.receipts.ShadowJudge: when middlewares changed the body and the
        # judge samples this request, the ORIGINAL body is also sent upstream and the two
        # responses are judged; the verdict lands in the ledger entry (the receipts).
        self.shadow = shadow
        super().__init__(("127.0.0.1", port), _GatewayHandler)

    def status_payload(self):
        middleware_status = {}
        for middleware in self.middlewares:
            status = getattr(middleware, "status_metadata", None)
            middleware_status[type(middleware).__name__] = status() if callable(status) else {}
        return {
            "status": "ok",
            "upstream": self.upstream_url,
            "middlewares": middleware_status,
            "receipts_enabled": self.ledger is not None,
            "shadow_enabled": self.shadow is not None and self.shadow.rate > 0.0,
            "shadow_rate": self.shadow.rate if self.shadow is not None else 0.0,
        }


class _GatewayHandler(BaseHTTPRequestHandler):
    server: GatewayServer

    # OpenAI-compatible chat plus Ollama's native generate, so local research/agent
    # traffic (OllamaClient uses /api/generate) can flow through the same pipeline.
    PROXIED_PATHS = ("/v1/chat/completions", "/v1/messages", "/api/generate", "/api/chat")
    FORWARDED_REQUEST_HEADERS = (
        "Authorization",
        "Accept",
        "Anthropic-Version",
        "Anthropic-Beta",
        "OpenAI-Organization",
        "OpenAI-Project",
        "X-API-Key",
    )

    def do_POST(self):
        started = time.perf_counter()
        if urlsplit(self.path).path not in self.PROXIED_PATHS:
            response_bytes = self._write_json(
                404, {"error": {"message": "Not found", "type": "not_found"}}
            )
            self._record(0, len(response_bytes.decode("utf-8")), started)
            return

        request_chars = 0
        response_chars = 0
        self._extra = {}
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_request = self.rfile.read(content_length)
            original_chars = len(raw_request.decode("utf-8"))
            original_body = json.loads(raw_request)
            self._extra = {
                "path": self.path,
                "model": original_body.get("model"),
            }
            body = original_body
            applied_middlewares = []
            short_circuit = None
            for middleware in self.server.middlewares:
                middleware_input = body
                result = middleware.before_request(body)
                if isinstance(result, ShortCircuit):
                    short_circuit = result
                    break
                body = result
                applied_middlewares.append((middleware, middleware_input))
            optimized_payload = json.dumps(body, ensure_ascii=False)
            request_chars = len(optimized_payload)
            optimized = body != original_body
            self._extra.update(
                {"request_chars_original": original_chars, "optimized": optimized}
            )
            middleware_receipts = {}
            for middleware in self.server.middlewares:
                receipt = getattr(middleware, "receipt_metadata", None)
                if callable(receipt):
                    middleware_receipts[type(middleware).__name__] = receipt()
            if middleware_receipts:
                self._extra["middleware_receipts"] = middleware_receipts

            if short_circuit is not None:
                status = 200
                response = short_circuit.response
                self._extra["cached"] = True
            else:
                self._extra["upstream_called"] = True
                if body.get("stream") is True:
                    status, response_chars, usage = self._stream_upstream(optimized_payload)
                    self._extra.update({"status": status, "streamed": True})
                    if usage is not None:
                        self._extra["usage"] = usage
                    return
                status, response = self._upstream(optimized_payload)
                usage = extract_usage(response)
                if usage is not None:
                    self._extra["usage"] = usage

                # Receipts: judge a deterministic sample of optimized requests against the
                # unoptimized original, so savings always ship with quality evidence.
                shadow = self.server.shadow
                if optimized and shadow is not None and shadow.should_sample(original_body):
                    from gateway.receipts import response_text

                    _, raw_response = self._upstream(json.dumps(original_body, ensure_ascii=False))
                    raw_usage = extract_usage(raw_response)
                    if raw_usage is not None:
                        self._extra["shadow_usage"] = raw_usage
                    self._extra["shadow"] = shadow.judge(
                        response_text(raw_response), response_text(response)
                    )

            for middleware, middleware_input in reversed(applied_middlewares):
                response = middleware.after_response(middleware_input, response)
            response_bytes = self._write_json(status, response)
            self._extra["status"] = status
            response_chars = len(response_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
            response_bytes = self._write_json(
                400, {"error": {"message": str(error), "type": "invalid_request_error"}}
            )
            response_chars = len(response_bytes.decode("utf-8"))
        except urllib.error.URLError as error:
            response_bytes = self._write_json(
                502, {"error": {"message": str(error.reason), "type": "upstream_error"}}
            )
            response_chars = len(response_bytes.decode("utf-8"))
        finally:
            self._record(request_chars, response_chars, started)

    def _upstream(self, payload):
        upstream_request = urllib.request.Request(
            self.server.upstream_url + self.path,
            data=payload.encode("utf-8"),
            headers=self._upstream_headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(upstream_request) as upstream_response:
                return upstream_response.status, json.loads(upstream_response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def _upstream_headers(self):
        headers = {"Content-Type": "application/json"}
        for name in self.FORWARDED_REQUEST_HEADERS:
            value = self.headers.get(name)
            if value is not None:
                headers[name] = value
        return headers

    def _stream_upstream(self, payload):
        """Relay an upstream event/JSON stream without buffering or decoding it."""
        upstream_request = urllib.request.Request(
            self.server.upstream_url + self.path,
            data=payload.encode("utf-8"),
            headers=self._upstream_headers(),
            method="POST",
        )
        try:
            upstream = urllib.request.urlopen(upstream_request)
        except urllib.error.HTTPError as error:
            upstream = error
        with upstream:
            self.send_response(upstream.status)
            content_type = upstream.headers.get("Content-Type", "application/octet-stream")
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            response_chars = 0
            usage = StreamUsageAccumulator()
            while True:
                chunk = upstream.read(64 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
                response_chars += len(chunk)
                usage.feed(chunk)
            return upstream.status, response_chars, usage.finish()

    def do_GET(self):
        if self.path in {"/health", "/status"}:
            body = {"status": "ok"} if self.path == "/health" else self.server.status_payload()
            self._write_json(200, body)
            return
        # Transparent passthrough for Ollama utility endpoints (/api/tags, /api/ps, ...).
        try:
            with urllib.request.urlopen(self.server.upstream_url + self.path) as upstream:
                payload = upstream.read()
                self.send_response(upstream.status)
        except urllib.error.HTTPError as error:
            payload = error.read()
            self.send_response(error.code)
        except urllib.error.URLError as error:
            payload = json.dumps({"error": {"message": str(error.reason)}}).encode("utf-8")
            self.send_response(502)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _record(self, request_chars, response_chars, started):
        if self.server.ledger is not None:
            entry = {
                "request_chars": request_chars,
                "response_chars": response_chars,
                "latency_ms": (time.perf_counter() - started) * 1000,
                "middlewares": [
                    type(middleware).__name__ for middleware in self.server.middlewares
                ],
            }
            entry.update(getattr(self, "_extra", {}))
            self.server.ledger.record(entry)

    def _write_json(self, status, body):
        payload = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
        return payload

    def log_message(self, format, *args):
        return
