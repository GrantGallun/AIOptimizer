import json
import tempfile
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from gateway.ledger import JsonlLedger
from gateway.receipts import ShadowJudge, response_text
from gateway.report import summarize
from gateway.server import GatewayServer


def _embed(texts):
    vocabulary = ("alpha", "beta", "gamma", "delta")
    return [[text.lower().count(word) for word in vocabulary] for text in texts]


class _EchoUpstream(BaseHTTPRequestHandler):
    """Returns the request's first message content as the completion text."""

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        content = body["messages"][0]["content"]
        payload = json.dumps(
            {"choices": [{"message": {"role": "assistant", "content": content}}]}
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        return


class _TrimMiddleware:
    """Test optimizer: drops a filler suffix from the prompt (changes the body)."""

    def before_request(self, body):
        body = json.loads(json.dumps(body))
        body["messages"][0]["content"] = body["messages"][0]["content"].replace(" FILLER", "")
        return body

    def after_response(self, body, response):
        return response


class ShadowJudgeTests(unittest.TestCase):
    def test_sampling_is_deterministic_per_content(self):
        judge = ShadowJudge(rate=0.5, embed_fn=_embed)
        body = {"messages": [{"content": "alpha"}]}
        self.assertEqual(judge.should_sample(body), judge.should_sample(body))
        self.assertTrue(ShadowJudge(rate=1.0).should_sample(body))
        self.assertFalse(ShadowJudge(rate=0.0).should_sample(body))

    def test_judge_identical_and_divergent(self):
        judge = ShadowJudge(rate=1.0, embed_fn=_embed, parity_threshold=0.9)
        same = judge.judge("alpha beta", "alpha beta")
        self.assertEqual(same["similarity"], 1.0)
        self.assertTrue(same["parity"])
        different = judge.judge("alpha alpha", "delta delta")
        self.assertFalse(different["parity"])


class ShadowEndToEndTests(unittest.TestCase):
    def test_optimized_request_produces_shadow_receipt(self):
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), _EchoUpstream)
        threading.Thread(target=upstream.serve_forever, daemon=True).start()
        ledger_path = Path(tempfile.mkdtemp()) / "ledger.jsonl"
        gateway = GatewayServer(
            f"http://127.0.0.1:{upstream.server_address[1]}",
            middlewares=(_TrimMiddleware(),),
            ledger=JsonlLedger(str(ledger_path)),
            port=0,
            shadow=ShadowJudge(rate=1.0, embed_fn=_embed),
        )
        threading.Thread(target=gateway.serve_forever, daemon=True).start()
        try:
            request = urllib.request.Request(
                f"http://127.0.0.1:{gateway.server_address[1]}/v1/chat/completions",
                data=json.dumps(
                    {"messages": [{"role": "user", "content": "alpha beta FILLER"}]}
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request) as response:
                completion = json.loads(response.read())
            # optimized request went upstream (FILLER trimmed)
            self.assertEqual(response_text(completion), "alpha beta")

            # the ledger write happens in the handler's finally, after the response is
            # flushed to the client — poll briefly instead of racing it
            import time as _time

            for _ in range(100):
                if ledger_path.exists() and ledger_path.read_text(encoding="utf-8").strip():
                    break
                _time.sleep(0.02)
            entry = json.loads(ledger_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue(entry["optimized"])
            self.assertGreater(entry["request_chars_original"], entry["request_chars"])
            self.assertIn("shadow", entry)
            self.assertIn("similarity", entry["shadow"])
            self.assertIn("parity", entry["shadow"])

            summary = summarize(str(ledger_path))
            self.assertEqual(summary["requests"], 1)
            self.assertEqual(summary["optimized_requests"], 1)
            self.assertEqual(summary["cached_requests"], 0)
            self.assertEqual(summary["passthrough_requests"], 0)
            self.assertGreater(summary["request_chars_saved"], 0)
            self.assertEqual(summary["requests_by_path"], {"/v1/chat/completions": 1})
            self.assertEqual(summary["responses_by_status"], {"200": 1})
            self.assertGreaterEqual(summary["p95_latency_ms"], summary["p50_latency_ms"])
            self.assertEqual(summary["shadow_samples"], 1)
            self.assertEqual(len(summary["quality_parity_ci"]), 2)
        finally:
            gateway.shutdown()
            upstream.shutdown()


if __name__ == "__main__":
    unittest.main()
