"""Stdlib OpenAI-compatible HTTP proxy for an Ollama upstream."""

import json
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class GatewayServer(ThreadingHTTPServer):
    """Proxy OpenAI chat-completion requests through a middleware pipeline."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, upstream_url, middlewares=(), ledger=None, port=8000):
        self.upstream_url = str(upstream_url).rstrip("/")
        self.middlewares = tuple(middlewares)
        self.ledger = ledger
        super().__init__(("127.0.0.1", port), _GatewayHandler)


class _GatewayHandler(BaseHTTPRequestHandler):
    server: GatewayServer

    def do_POST(self):
        started = time.perf_counter()
        if self.path != "/v1/chat/completions":
            response_bytes = self._write_json(
                404, {"error": {"message": "Not found", "type": "not_found"}}
            )
            self._record(0, len(response_bytes.decode("utf-8")), started)
            return

        request_chars = 0
        response_chars = 0
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_request = self.rfile.read(content_length)
            request_chars = len(raw_request.decode("utf-8"))
            original_body = json.loads(raw_request)
            body = original_body
            for middleware in self.server.middlewares:
                body = middleware.before_request(body)

            upstream_request = urllib.request.Request(
                self.server.upstream_url + "/v1/chat/completions",
                data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(upstream_request) as upstream_response:
                    status = upstream_response.status
                    response = json.loads(upstream_response.read())
            except urllib.error.HTTPError as error:
                status = error.code
                response = json.loads(error.read())

            for middleware in reversed(self.server.middlewares):
                response = middleware.after_response(body, response)
            response_bytes = self._write_json(status, response)
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

    def _record(self, request_chars, response_chars, started):
        if self.server.ledger is not None:
            self.server.ledger.record(
                {
                    "request_chars": request_chars,
                    "response_chars": response_chars,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "middlewares": [
                        type(middleware).__name__ for middleware in self.server.middlewares
                    ],
                }
            )

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
