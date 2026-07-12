import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from gateway.ledger import JsonlLedger
from gateway.cache_middleware import ExactCacheMiddleware
from gateway.server import GatewayServer


class _StubHandler(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        type(self).requests.append((self.path, body))
        response = {"id": "chatcmpl-test", "object": "chat.completion", "tag": body.get("tag")}
        payload = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

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


class GatewayTests(unittest.TestCase):
    def setUp(self):
        _StubHandler.requests = []
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
