import json
import tempfile
import threading
import time
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from aioptimizer.cache_middleware import ExactCacheMiddleware
from aioptimizer.context_compiler import ConversationCompiler
from aioptimizer.context_middleware import AttentionContextMiddleware
from aioptimizer.ledger import JsonlLedger
from aioptimizer.server import GatewayServer


def _embed(texts):
    vocabulary = ("cache", "latency", "weather", "memory")
    return [[text.lower().count(word) for word in vocabulary] for text in texts]


class _CaptureUpstream(BaseHTTPRequestHandler):
    bodies = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).bodies.append(body)
        payload = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers(); self.wfile.write(payload)

    def log_message(self, format, *args):
        return


class AdaptiveGatewayEndToEndTests(unittest.TestCase):
    def setUp(self):
        _CaptureUpstream.bodies = []
        self.upstream = ThreadingHTTPServer(("127.0.0.1", 0), _CaptureUpstream)
        self.upstream_thread = threading.Thread(target=self.upstream.serve_forever, daemon=True)
        self.upstream_thread.start()
        self.temp = tempfile.TemporaryDirectory()
        self.ledger_path = Path(self.temp.name) / "ledger.jsonl"
        attention = AttentionContextMiddleware(
            budget_chars=420, compiler=ConversationCompiler(embed_fn=_embed), min_relevance=.5
        )
        self.gateway = GatewayServer(
            f"http://127.0.0.1:{self.upstream.server_port}",
            middlewares=(ExactCacheMiddleware(), attention),
            ledger=JsonlLedger(self.ledger_path), port=0,
        )
        self.gateway_thread = threading.Thread(target=self.gateway.serve_forever, daemon=True)
        self.gateway_thread.start()
        self.url = f"http://127.0.0.1:{self.gateway.server_port}/v1/chat/completions"
        self.optimize_url = f"http://127.0.0.1:{self.gateway.server_port}/optimize/context"

    def tearDown(self):
        self.gateway.shutdown(); self.gateway.server_close(); self.gateway_thread.join()
        self.upstream.shutdown(); self.upstream.server_close(); self.upstream_thread.join()
        self.temp.cleanup()

    def _post(self, body):
        request = urllib.request.Request(
            self.url, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read())

    def _optimize(self, body):
        request = urllib.request.Request(
            self.optimize_url, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read())

    @staticmethod
    def _body(query):
        return {"model": "test", "messages": [
            {"role": "system", "content": "Keep the request verbatim."},
            {"role": "user", "content": "Weather background " * 16},
            {"role": "assistant", "content": "Weather discussion " * 16},
            {"role": "user", "content": "Cache latency must stay below 20ms."},
            {"role": "assistant", "content": "Exact cache lowers cache latency."},
            {"role": "user", "content": query},
        ]}

    def test_adaptive_routes_cache_and_receipts_work_together(self):
        focused = self._body("What did we decide about cache latency?")
        vague = self._body("What stands out?")

        self._post(focused)
        self._post(vague)
        self._post(focused)

        self.assertEqual(len(_CaptureUpstream.bodies), 2)
        optimized, passthrough = _CaptureUpstream.bodies
        self.assertLess(len(json.dumps(optimized)), len(json.dumps(focused)))
        self.assertEqual(optimized["messages"][-1], focused["messages"][-1])
        self.assertEqual(passthrough, vague)

        for _ in range(100):
            lines = self.ledger_path.read_text(encoding="utf-8").splitlines()
            if len(lines) == 3:
                break
            time.sleep(.01)
        entries = [json.loads(line) for line in lines]
        receipts = [
            entry.get("middleware_receipts", {}).get("AttentionContextMiddleware", {})
            for entry in entries
        ]
        self.assertEqual(receipts[0]["route"], "attention")
        self.assertEqual(receipts[1]["route"], "raw")
        self.assertTrue(entries[2]["cached"])
        self.assertEqual(receipts[2], {"applied": False})  # cache short-circuited before routing

    def test_local_compiler_endpoint_never_calls_upstream(self):
        messages = self._body("unused")["messages"][1:-1]

        focused = self._optimize({
            "messages": messages,
            "query": "What did we decide about cache latency?",
            "output_budget_chars": 300,
        })
        vague = self._optimize({
            "messages": messages,
            "query": "What stands out?",
            "output_budget_chars": 300,
        })

        self.assertEqual(focused["route"], "attention")
        self.assertIn("cache", focused["context"].lower())
        self.assertEqual(vague["route"], "raw")
        self.assertEqual(vague["context"], "")
        self.assertEqual(_CaptureUpstream.bodies, [])


if __name__ == "__main__":
    unittest.main()
