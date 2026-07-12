"""Stdlib OpenAI-compatible HTTP proxy for an Ollama upstream."""

import json
import socket
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from gateway.middleware import ShortCircuit
from gateway.receipts import response_text
from gateway.requirements import evaluate_requirements, extract_requirements
from gateway.usage import StreamUsageAccumulator, extract_usage


class UpstreamProtocolError(RuntimeError):
    """The provider replied, but its response violated the expected JSON protocol."""


class GatewayServer(ThreadingHTTPServer):
    """Proxy OpenAI chat-completion requests through a middleware pipeline."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, upstream_url, middlewares=(), ledger=None, port=8000, shadow=None,
                 upstream_timeout=300.0):
        self.upstream_url = str(upstream_url).rstrip("/")
        self.middlewares = tuple(middlewares)
        self.ledger = ledger
        # Optional gateway.receipts.ShadowJudge: when middlewares changed the body and the
        # judge samples this request, the ORIGINAL body is also sent upstream and the two
        # responses are judged; the verdict lands in the ledger entry (the receipts).
        self.shadow = shadow
        if (
            not isinstance(upstream_timeout, (int, float))
            or isinstance(upstream_timeout, bool)
            or upstream_timeout <= 0
        ):
            raise ValueError("upstream_timeout must be a positive number")
        self.upstream_timeout = float(upstream_timeout)
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
            "upstream_timeout_seconds": self.upstream_timeout,
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
        self._extra = {"path": self.path}
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_request = self.rfile.read(content_length)
            original_chars = len(raw_request.decode("utf-8"))
            original_body, requirements = extract_requirements(json.loads(raw_request))
            if requirements:
                original_chars = len(json.dumps(original_body, ensure_ascii=False))
            self._extra["model"] = original_body.get("model")
            if requirements:
                self._extra["requirement_contracts"] = len(requirements)
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
                    status, response_chars, usage, output_text, complete = self._stream_upstream(
                        optimized_payload
                    )
                    self._extra.update({
                        "status": status, "streamed": True, "stream_complete": complete
                    })
                    if usage is not None:
                        self._extra["usage"] = usage
                    requirement_receipt = evaluate_requirements(output_text, requirements)
                    if requirement_receipt is not None:
                        self._extra["requirements"] = requirement_receipt
                    return
                status, response = self._upstream(optimized_payload)
                usage = extract_usage(response)
                if usage is not None:
                    self._extra["usage"] = usage

                # Receipts: judge a deterministic sample of optimized requests against the
                # unoptimized original, so savings always ship with quality evidence.
                shadow = self.server.shadow
                if optimized and shadow is not None and shadow.should_sample(original_body):
                    try:
                        _, raw_response = self._upstream(
                            json.dumps(original_body, ensure_ascii=False)
                        )
                        raw_usage = extract_usage(raw_response)
                        if raw_usage is not None:
                            self._extra["shadow_usage"] = raw_usage
                        self._extra["shadow"] = shadow.judge(
                            response_text(raw_response), response_text(response)
                        )
                        shadow_requirements = evaluate_requirements(
                            response_text(raw_response), requirements
                        )
                        if shadow_requirements is not None:
                            self._extra["shadow_requirements"] = shadow_requirements
                    except (
                        TimeoutError,
                        socket.timeout,
                        urllib.error.URLError,
                        UpstreamProtocolError,
                    ) as error:
                        reason = getattr(error, "reason", error)
                        self._extra["shadow_error"] = {
                            "type": "upstream_timeout"
                            if isinstance(reason, (TimeoutError, socket.timeout))
                            else (
                                "upstream_invalid_response"
                                if isinstance(error, UpstreamProtocolError)
                                else "upstream_error"
                            ),
                            "message": str(reason),
                        }

            for middleware, middleware_input in reversed(applied_middlewares):
                response = middleware.after_response(middleware_input, response)
            requirement_receipt = evaluate_requirements(response_text(response), requirements)
            if requirement_receipt is not None:
                self._extra["requirements"] = requirement_receipt
            response_bytes = self._write_json(status, response)
            self._extra["status"] = status
            response_chars = len(response_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
            self._extra["status"] = 400
            response_bytes = self._write_json(
                400, {"error": {"message": str(error), "type": "invalid_request_error"}}
            )
            response_chars = len(response_bytes.decode("utf-8"))
        except UpstreamProtocolError as error:
            self._extra["status"] = 502
            response_bytes = self._write_json(
                502,
                {"error": {"message": str(error), "type": "upstream_invalid_response"}},
            )
            response_chars = len(response_bytes.decode("utf-8"))
        except (TimeoutError, socket.timeout) as error:
            self._extra["status"] = 504
            response_bytes = self._write_json(
                504, {"error": {"message": str(error), "type": "upstream_timeout"}}
            )
            response_chars = len(response_bytes.decode("utf-8"))
        except urllib.error.URLError as error:
            timed_out = isinstance(error.reason, (TimeoutError, socket.timeout))
            status = 504 if timed_out else 502
            error_type = "upstream_timeout" if timed_out else "upstream_error"
            self._extra["status"] = status
            response_bytes = self._write_json(
                status, {"error": {"message": str(error.reason), "type": error_type}}
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
            with urllib.request.urlopen(
                upstream_request, timeout=self.server.upstream_timeout
            ) as upstream_response:
                return upstream_response.status, self._decode_upstream_json(upstream_response.read())
        except urllib.error.HTTPError as error:
            return error.code, self._decode_upstream_json(error.read())

    @staticmethod
    def _decode_upstream_json(payload):
        try:
            return json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise UpstreamProtocolError("upstream returned invalid JSON") from error

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
            upstream = urllib.request.urlopen(
                upstream_request, timeout=self.server.upstream_timeout
            )
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
            complete = True
            try:
                while True:
                    chunk = upstream.read(64 * 1024)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
                    response_chars += len(chunk)
                    usage.feed(chunk)
            except (
                TimeoutError,
                socket.timeout,
                BrokenPipeError,
                ConnectionResetError,
                ConnectionAbortedError,
            ):
                complete = False
            normalized_usage = usage.finish()
            return upstream.status, response_chars, normalized_usage, usage.text(), complete

    def do_GET(self):
        if self.path in {"/health", "/status"}:
            body = {"status": "ok"} if self.path == "/health" else self.server.status_payload()
            self._write_json(200, body)
            return
        # Transparent passthrough for Ollama utility endpoints (/api/tags, /api/ps, ...).
        try:
            with urllib.request.urlopen(
                self.server.upstream_url + self.path,
                timeout=self.server.upstream_timeout,
            ) as upstream:
                payload = upstream.read()
                self.send_response(upstream.status)
        except urllib.error.HTTPError as error:
            payload = error.read()
            self.send_response(error.code)
        except (TimeoutError, socket.timeout):
            payload = json.dumps({"error": {"message": "upstream timed out"}}).encode("utf-8")
            self.send_response(504)
        except urllib.error.URLError as error:
            timed_out = isinstance(error.reason, (TimeoutError, socket.timeout))
            payload = json.dumps({"error": {"message": str(error.reason)}}).encode("utf-8")
            self.send_response(504 if timed_out else 502)
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
