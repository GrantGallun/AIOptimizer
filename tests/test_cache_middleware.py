import json
import tempfile
import threading
import time
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from gateway.cache_middleware import ExactCacheMiddleware
from gateway.ledger import JsonlLedger
from gateway.middleware import ShortCircuit
from gateway.server import GatewayServer


class _FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class ExactCacheMiddlewareTests(unittest.TestCase):
    def test_miss_then_hit_returns_independent_cached_response(self):
        cache = ExactCacheMiddleware()
        body = {"messages": [], "model": "qwen3:8b"}
        self.assertIs(cache.before_request(body), body)

        response = {"choices": [{"message": {"content": "hello"}}]}
        self.assertIs(cache.after_response(body, response), response)
        response["choices"][0]["message"]["content"] = "changed"

        hit = cache.before_request({"model": "qwen3:8b", "messages": []})
        self.assertIsInstance(hit, ShortCircuit)
        self.assertEqual(hit.response["choices"][0]["message"]["content"], "hello")
        hit.response["choices"][0]["message"]["content"] = "mutated"
        self.assertEqual(
            cache.before_request(body).response["choices"][0]["message"]["content"],
            "hello",
        )

    def test_ttl_expiry_uses_injected_clock(self):
        clock = _FakeClock()
        cache = ExactCacheMiddleware(ttl_seconds=10, clock=clock)
        body = {"messages": []}
        cache.after_response(body, {"answer": 1})
        clock.now = 10
        self.assertIsInstance(cache.before_request(body), ShortCircuit)
        clock.now = 10.01
        self.assertIs(cache.before_request(body), body)

    def test_evicts_oldest_entry(self):
        cache = ExactCacheMiddleware(max_entries=2)
        bodies = [{"request": number} for number in range(3)]
        for number, body in enumerate(bodies):
            cache.after_response(body, {"answer": number})

        self.assertIs(cache.before_request(bodies[0]), bodies[0])
        self.assertIsInstance(cache.before_request(bodies[1]), ShortCircuit)
        self.assertIsInstance(cache.before_request(bodies[2]), ShortCircuit)


class _CountingUpstream(BaseHTTPRequestHandler):
    calls = 0

    def do_POST(self):
        type(self).calls += 1
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        payload = json.dumps({"id": "cached-test", "model": body.get("model")}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        return


class CacheEndToEndTests(unittest.TestCase):
    def test_second_identical_request_skips_upstream_and_records_cached(self):
        _CountingUpstream.calls = 0
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), _CountingUpstream)
        upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        upstream_thread.start()
        with tempfile.TemporaryDirectory() as directory:
            ledger_path = Path(directory) / "ledger.jsonl"
            gateway = GatewayServer(
                "http://127.0.0.1:%d" % upstream.server_port,
                middlewares=(ExactCacheMiddleware(),),
                ledger=JsonlLedger(ledger_path),
                port=0,
            )
            gateway_thread = threading.Thread(target=gateway.serve_forever, daemon=True)
            gateway_thread.start()
            try:
                url = "http://127.0.0.1:%d/v1/chat/completions" % gateway.server_port
                body = {"model": "qwen3:8b", "messages": []}
                responses = []
                for _ in range(2):
                    request = urllib.request.Request(
                        url,
                        data=json.dumps(body).encode(),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(request) as response:
                        responses.append(json.loads(response.read()))

                for _ in range(100):
                    if (
                        ledger_path.exists()
                        and len(ledger_path.read_text(encoding="utf-8").splitlines()) == 2
                    ):
                        break
                    time.sleep(0.01)
                entries = [
                    json.loads(line)
                    for line in ledger_path.read_text(encoding="utf-8").splitlines()
                ]
                self.assertEqual(responses[0], responses[1])
                self.assertEqual(_CountingUpstream.calls, 1)
                self.assertNotIn("cached", entries[0])
                self.assertTrue(entries[1]["cached"])
            finally:
                gateway.shutdown()
                gateway.server_close()
                gateway_thread.join()
                upstream.shutdown()
                upstream.server_close()
                upstream_thread.join()


if __name__ == "__main__":
    unittest.main()
