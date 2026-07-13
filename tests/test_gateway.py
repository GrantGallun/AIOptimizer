import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from aioptimizer.ledger import JsonlLedger
from aioptimizer.cache_middleware import ExactCacheMiddleware
from aioptimizer.server import GatewayServer, _GatewayHandler
from aioptimizer.receipts import ShadowJudge, response_text


class _StubHandler(BaseHTTPRequestHandler):
    requests = []
    headers_seen = []

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        type(self).requests.append((self.path, body))
        type(self).headers_seen.append(dict(self.headers.items()))
        if body.get("slow"):
            time.sleep(0.1)
        if body.get("malformed_response"):
            payload = b"not-json"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if body.get("stream"):
            payload = (
                b'data: {"choices":[{"delta":{"content":"one"}}]}\n\n'
                b'data: {"usage":{"prompt_tokens":9,"completion_tokens":2,'
                b'"total_tokens":11}}\n\ndata: [DONE]\n\n'
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(payload)
            return
        response = {"id": "chatcmpl-test", "object": "chat.completion", "tag": body.get("tag")}
        if isinstance(body.get("answer"), str):
            response["choices"] = [{"message": {"content": body["answer"]}}]
        if body.get("usage_test"):
            response["usage"] = {
                "prompt_tokens": 15,
                "completion_tokens": 3,
                "total_tokens": 18,
            }
        payload = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def log_message(self, format, *args):
        return


class _TagMiddleware:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    def before_request(self, body):
        self.events.append("before:" + self.name)
        return {**body, "tag": body.get("tag", "") + self.name}

    def after_response(self, body, response):
        self.events.append("after:" + self.name)
        return {**response, "after": response.get("after", "") + self.name}


class _ReceiptMiddleware(_TagMiddleware):
    def receipt_metadata(self):
        return {"applied": True, "compiler_fingerprint": "sha256:test"}


class _RemoveSlowMiddleware:
    def before_request(self, body):
        return {key: value for key, value in body.items() if key != "slow"}

    def after_response(self, body, response):
        return response


class GatewayTests(unittest.TestCase):
    def test_json_writer_quietly_handles_disconnected_local_client(self):
        class _DisconnectedWriter:
            def write(self, payload):
                raise ConnectionAbortedError("client closed")

        handler = object.__new__(_GatewayHandler)
        handler.wfile = _DisconnectedWriter()
        handler.send_response = lambda status: None
        handler.send_header = lambda name, value: None
        handler.end_headers = lambda: None

        payload = handler._write_json(200, {"route": "attention"})

        self.assertEqual(payload, b"")
        self.assertTrue(handler._extra["client_disconnected"])

    def setUp(self):
        _StubHandler.requests = []
        _StubHandler.headers_seen = []
        self.upstream = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
        self.upstream_thread = threading.Thread(target=self.upstream.serve_forever, daemon=True)
        self.upstream_thread.start()

    def tearDown(self):
        self.upstream.shutdown()
        self.upstream.server_close()
        self.upstream_thread.join()

    def _start_gateway(self, **kwargs):
        upstream_url = "http://127.0.0.1:%d" % self.upstream.server_port
        gateway = GatewayServer(upstream_url, port=0, **kwargs)
        thread = threading.Thread(target=gateway.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(gateway.server_close)
        self.addCleanup(gateway.shutdown)
        return "http://127.0.0.1:%d" % gateway.server_port

    def _post(self, url, body):
        request = urllib.request.Request(
            url + "/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read())

    def test_passthrough_returns_upstream_json(self):
        url = self._start_gateway()

        response = self._post(url, {"model": "qwen3:8b", "messages": []})

        self.assertEqual(
            response,
            {"id": "chatcmpl-test", "object": "chat.completion", "tag": None},
        )
        self.assertEqual(_StubHandler.requests[0][0], "/v1/chat/completions")

    def test_anthropic_response_text_supports_multiple_text_blocks(self):
        response = {
            "content": [
                {"type": "text", "text": "first"},
                {"type": "tool_use", "id": "tool-1"},
                {"type": "text", "text": "second"},
            ]
        }

        self.assertEqual(response_text(response), "first\nsecond")

    def test_ollama_chat_response_text_uses_message_content(self):
        self.assertEqual(
            response_text({"message": {"role": "assistant", "content": "ready"}}),
            "ready",
        )

    def test_query_string_and_provider_headers_are_forwarded(self):
        url = self._start_gateway()
        request = urllib.request.Request(
            url + "/v1/messages?beta=true",
            data=json.dumps({"model": "claude-test", "messages": []}).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer test-token",
                "X-Api-Key": "test-key",
                "Anthropic-Version": "2023-06-01",
            },
            method="POST",
        )

        with urllib.request.urlopen(request) as response:
            json.loads(response.read())

        self.assertEqual(_StubHandler.requests[0][0], "/v1/messages?beta=true")
        seen = {key.lower(): value for key, value in _StubHandler.headers_seen[0].items()}
        self.assertEqual(seen["authorization"], "Bearer test-token")
        self.assertEqual(seen["x-api-key"], "test-key")
        self.assertEqual(seen["anthropic-version"], "2023-06-01")

    def test_stream_is_relayed_and_recorded_without_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger_path = Path(directory) / "stream.jsonl"
            url = self._start_gateway(
                middlewares=(ExactCacheMiddleware(),), ledger=JsonlLedger(ledger_path)
            )
            body = {
                "model": "test",
                "messages": [],
                "stream": True,
                "aioptimizer": {"requirements": [{
                    "id": "stream-word", "must_include": ["one"]
                }]},
            }

            responses = []
            for _ in range(2):
                request = urllib.request.Request(
                    url + "/v1/chat/completions",
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request) as response:
                    self.assertEqual(response.headers.get_content_type(), "text/event-stream")
                    responses.append(response.read())

            self.assertIn(b'"content":"one"', responses[0])
            self.assertIn(b'data: [DONE]', responses[0])
            self.assertEqual(responses[0], responses[1])
            self.assertEqual(len(_StubHandler.requests), 2)
            entries = [json.loads(line) for line in ledger_path.read_text().splitlines()]
            self.assertEqual(len(entries), 2)
            self.assertTrue(all(entry["streamed"] for entry in entries))
            self.assertTrue(all(entry["stream_complete"] for entry in entries))
            self.assertTrue(all(entry["status"] == 200 for entry in entries))
            self.assertTrue(all(entry["response_chars"] == len(responses[0]) for entry in entries))
            self.assertTrue(all(entry["usage"] == {
                "input_tokens": 9, "output_tokens": 2, "total_tokens": 11
            } for entry in entries))
            self.assertTrue(all(entry["requirements"]["all_passed"] for entry in entries))
            self.assertTrue(all("aioptimizer" not in body for _, body in _StubHandler.requests))

    def test_json_usage_is_recorded_only_when_upstream_is_called(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger_path = Path(directory) / "usage.jsonl"
            url = self._start_gateway(
                middlewares=(ExactCacheMiddleware(),), ledger=JsonlLedger(ledger_path)
            )
            body = {"model": "test", "messages": [], "usage_test": True}

            self._post(url, body)
            self._post(url, body)

            entries = [json.loads(line) for line in ledger_path.read_text().splitlines()]
            self.assertEqual(entries[0]["usage"], {
                "input_tokens": 15, "output_tokens": 3, "total_tokens": 18
            })
            self.assertTrue(entries[0]["upstream_called"])
            self.assertNotIn("usage", entries[1])
            self.assertNotIn("upstream_called", entries[1])
            self.assertTrue(entries[1]["cached"])

    def test_private_requirement_contract_is_stripped_and_receipted_on_cache_hits(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger_path = Path(directory) / "requirements.jsonl"
            url = self._start_gateway(
                middlewares=(ExactCacheMiddleware(),), ledger=JsonlLedger(ledger_path)
            )
            base = {"model": "test", "messages": [], "answer": "HELLO"}
            passing = {**base, "aioptimizer": {"requirements": [{
                "id": "greeting", "must_include": ["hello"]
            }]}}
            failing = {**base, "aioptimizer": {"requirements": [{
                "id": "other", "must_include": ["goodbye"]
            }]}}

            self._post(url, passing)
            self._post(url, failing)

            self.assertEqual(len(_StubHandler.requests), 1)
            self.assertNotIn("aioptimizer", _StubHandler.requests[0][1])
            for _ in range(100):
                if (
                    ledger_path.exists()
                    and len(ledger_path.read_text().splitlines()) == 2
                ):
                    break
                time.sleep(0.01)
            entries = [json.loads(line) for line in ledger_path.read_text().splitlines()]
            self.assertTrue(entries[0]["requirements"]["all_passed"])
            self.assertFalse(entries[1]["requirements"]["all_passed"])
            self.assertTrue(entries[1]["cached"])
            self.assertEqual(entries[0]["requirements"]["results"][0]["id"], "greeting")
            self.assertNotIn("hello", json.dumps(entries[0]["requirements"]).lower())

    def test_invalid_requirement_contract_returns_400_without_upstream_call(self):
        url = self._start_gateway()
        request = urllib.request.Request(
            url + "/v1/chat/completions",
            data=json.dumps({"messages": [], "aioptimizer": {"requirements": []}}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(request)
        error = raised.exception
        try:
            self.assertEqual(error.code, 400)
        finally:
            error.close()
        self.assertEqual(_StubHandler.requests, [])

    def test_shadow_receipts_compare_explicit_requirements_without_forwarding_them(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger_path = Path(directory) / "shadow-requirements.jsonl"
            url = self._start_gateway(
                ledger=JsonlLedger(ledger_path),
                middlewares=(_TagMiddleware("A", []),),
                shadow=ShadowJudge(rate=1.0, embed_fn=lambda texts: [[1.0] for _ in texts]),
            )
            body = {
                "messages": [],
                "answer": "STATUS: ready",
                "aioptimizer": {"requirements": [{
                    "id": "status", "must_include": ["STATUS:"]
                }]},
            }

            self._post(url, body)

            for _ in range(100):
                if ledger_path.exists() and ledger_path.read_text().strip():
                    break
                time.sleep(0.01)
            entry = json.loads(ledger_path.read_text().splitlines()[0])
            self.assertTrue(entry["requirements"]["all_passed"])
            self.assertTrue(entry["shadow_requirements"]["all_passed"])
            self.assertEqual(len(_StubHandler.requests), 2)
            self.assertTrue(all(
                "aioptimizer" not in request_body
                for _, request_body in _StubHandler.requests
            ))

    def test_upstream_timeout_returns_504_and_records_failure_status(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger_path = Path(directory) / "timeout.jsonl"
            url = self._start_gateway(
                upstream_timeout=0.01, ledger=JsonlLedger(ledger_path)
            )
            request = urllib.request.Request(
                url + "/v1/chat/completions",
                data=json.dumps({"messages": [], "slow": True}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with self.assertRaises(urllib.error.HTTPError) as raised:
                urllib.request.urlopen(request)
            error = raised.exception
            try:
                self.assertEqual(error.code, 504)
                self.assertEqual(json.loads(error.read())["error"]["type"], "upstream_timeout")
            finally:
                error.close()

            for _ in range(100):
                if ledger_path.exists() and ledger_path.read_text().strip():
                    break
                time.sleep(0.01)
            entry = json.loads(ledger_path.read_text().splitlines()[0])
            self.assertEqual(entry["status"], 504)
            self.assertTrue(entry["upstream_called"])

    def test_shadow_timeout_does_not_fail_primary_request(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger_path = Path(directory) / "shadow-timeout.jsonl"
            url = self._start_gateway(
                upstream_timeout=0.01,
                ledger=JsonlLedger(ledger_path),
                middlewares=(_RemoveSlowMiddleware(),),
                shadow=ShadowJudge(rate=1.0),
            )

            response = self._post(url, {"messages": [], "slow": True})

            self.assertEqual(response["id"], "chatcmpl-test")
            for _ in range(100):
                if ledger_path.exists() and ledger_path.read_text().strip():
                    break
                time.sleep(0.01)
            entry = json.loads(ledger_path.read_text().splitlines()[0])
            self.assertEqual(entry["status"], 200)
            self.assertEqual(entry["shadow_error"]["type"], "upstream_timeout")
            self.assertNotIn("shadow", entry)

    def test_invalid_upstream_json_returns_502_not_client_400(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger_path = Path(directory) / "invalid-upstream.jsonl"
            url = self._start_gateway(ledger=JsonlLedger(ledger_path))
            request = urllib.request.Request(
                url + "/v1/chat/completions",
                data=json.dumps({"messages": [], "malformed_response": True}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with self.assertRaises(urllib.error.HTTPError) as raised:
                urllib.request.urlopen(request)
            error = raised.exception
            try:
                self.assertEqual(error.code, 502)
                self.assertEqual(
                    json.loads(error.read())["error"]["type"],
                    "upstream_invalid_response",
                )
            finally:
                error.close()
            for _ in range(100):
                if ledger_path.exists() and ledger_path.read_text().strip():
                    break
                time.sleep(0.01)
            self.assertEqual(
                json.loads(ledger_path.read_text().splitlines()[0])["status"], 502
            )

    def test_middlewares_run_before_in_order_and_after_in_reverse(self):
        events = []
        middlewares = (_TagMiddleware("A", events), _TagMiddleware("B", events))
        url = self._start_gateway(middlewares=middlewares)

        response = self._post(url, {"messages": []})

        self.assertEqual(response["tag"], "AB")
        self.assertEqual(response["after"], "BA")
        self.assertEqual(events, ["before:A", "before:B", "after:B", "after:A"])

    def test_exact_cache_uses_its_original_input_when_later_middleware_rewrites(self):
        events = []
        url = self._start_gateway(
            middlewares=(ExactCacheMiddleware(), _TagMiddleware("A", events))
        )
        body = {"messages": [{"role": "user", "content": "repeat"}]}

        first = self._post(url, body)
        second = self._post(url, body)

        self.assertEqual(first, second)
        self.assertEqual(len(_StubHandler.requests), 1)
        self.assertEqual(events, ["before:A", "after:A"])

    def test_unknown_path_returns_json_404(self):
        url = self._start_gateway()
        request = urllib.request.Request(url + "/unknown", data=b"{}", method="POST")

        with self.assertRaises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(request)

        error = raised.exception
        try:
            self.assertEqual(error.code, 404)
            self.assertEqual(json.loads(error.read())["error"]["type"], "not_found")
        finally:
            error.close()

    def test_health_and_status_are_local_and_describe_active_middlewares(self):
        middleware = _ReceiptMiddleware("R", [])
        url = self._start_gateway(middlewares=(middleware,))

        with urllib.request.urlopen(url + "/health") as response:
            self.assertEqual(json.loads(response.read()), {"status": "ok"})
        with urllib.request.urlopen(url + "/status") as response:
            status = json.loads(response.read())

        self.assertEqual(status["status"], "ok")
        self.assertIn("_ReceiptMiddleware", status["middlewares"])
        self.assertFalse(status["receipts_enabled"])
        self.assertEqual(status["upstream_timeout_seconds"], 300.0)
        self.assertEqual(_StubHandler.requests, [])

    def test_ledger_records_one_valid_line_per_request(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requests.jsonl"
            url = self._start_gateway(ledger=JsonlLedger(path), middlewares=(_TagMiddleware("A", []),))

            self._post(url, {"messages": [{"role": "user", "content": "hello"}]})
            self._post(url, {"messages": [{"role": "user", "content": "again"}]})

            entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(entries), 2)
            for entry in entries:
                self.assertGreater(entry["ts"], 0)
                self.assertGreater(entry["request_chars"], 0)
                self.assertGreater(entry["response_chars"], 0)
                self.assertGreaterEqual(entry["latency_ms"], 0)
                self.assertEqual(entry["middlewares"], ["_TagMiddleware"])
                self.assertEqual(entry["path"], "/v1/chat/completions")
                self.assertEqual(entry["status"], 200)

    def test_ledger_records_middleware_specific_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requests.jsonl"
            middleware = _ReceiptMiddleware("R", [])
            url = self._start_gateway(ledger=JsonlLedger(path), middlewares=(middleware,))

            self._post(url, {"messages": [{"role": "user", "content": "hello"}]})

            for _ in range(100):
                if path.exists() and path.read_text(encoding="utf-8").strip():
                    break
                import time
                time.sleep(0.01)
            entry = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            receipt = entry["middleware_receipts"]["_ReceiptMiddleware"]
            self.assertTrue(receipt["applied"])
            self.assertEqual(receipt["compiler_fingerprint"], "sha256:test")


if __name__ == "__main__":
    unittest.main()
