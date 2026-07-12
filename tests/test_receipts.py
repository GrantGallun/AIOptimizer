import json
import tempfile
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from aioptimizer.ledger import JsonlLedger
from aioptimizer.receipts import ShadowJudge, response_text
from aioptimizer.report import summarize
from aioptimizer.server import GatewayServer


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
        upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        upstream_thread.start()
        temporary_directory = tempfile.TemporaryDirectory()
        ledger_path = Path(temporary_directory.name) / "ledger.jsonl"
        gateway = GatewayServer(
            f"http://127.0.0.1:{upstream.server_address[1]}",
            middlewares=(_TrimMiddleware(),),
            ledger=JsonlLedger(str(ledger_path)),
            port=0,
            shadow=ShadowJudge(rate=1.0, embed_fn=_embed),
        )
        gateway_thread = threading.Thread(target=gateway.serve_forever, daemon=True)
        gateway_thread.start()
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
            gateway.server_close()
            gateway_thread.join()
            upstream.shutdown()
            upstream.server_close()
            upstream_thread.join()
            temporary_directory.cleanup()

    def test_report_aggregates_adaptive_routes_and_embedding_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            rows = [
                {"request_chars": 10, "response_chars": 2, "latency_ms": 1,
                 "optimized": True, "request_chars_original": 20,
                 "upstream_called": True,
                 "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
                 "shadow_usage": {"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
                 "requirement_contracts": 2,
                 "requirements": {"requirements": 2, "passed": 1, "all_passed": False},
                 "shadow_requirements": {"requirements": 2, "passed": 2, "all_passed": True},
                 "middleware_receipts": {"AttentionContextMiddleware": {
                     "route": "attention", "applied": True,
                     "embedding_cache_hits": 7, "embedding_cache_misses": 3}}},
                {"request_chars": 10, "response_chars": 2, "latency_ms": 2,
                 "optimized": False,
                 "middleware_receipts": {"AttentionContextMiddleware": {
                     "route": "raw", "applied": False,
                     "embedding_cache_hits": 4, "embedding_cache_misses": 1}}},
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            summary = summarize(str(path))

        self.assertEqual(summary["attention_route_counts"], {"attention": 1, "raw": 1})
        self.assertEqual(summary["attention_applied_requests"], 1)
        self.assertEqual(summary["embedding_cache_hits"], 11)
        self.assertEqual(summary["embedding_cache_misses"], 4)
        self.assertEqual(summary["upstream_requests"], 1)
        self.assertEqual(summary["usage_receipts"], 1)
        self.assertEqual(summary["usage_coverage_rate"], 1.0)
        self.assertEqual(summary["input_tokens"], 10)
        self.assertEqual(summary["output_tokens"], 2)
        self.assertEqual(summary["total_tokens"], 12)
        self.assertEqual(summary["shadow_total_tokens"], 13)
        self.assertEqual(summary["provider_total_tokens_consumed"], 25)
        self.assertEqual(summary["paired_usage_receipts"], 1)
        self.assertEqual(summary["measured_input_token_savings"], 0)
        self.assertEqual(summary["streamed_requests"], 0)
        self.assertEqual(summary["incomplete_streams"], 0)
        self.assertEqual(summary["shadow_failures"], 0)
        self.assertEqual(summary["shadow_failure_types"], {})
        self.assertEqual(summary["requirement_contract_requests"], 1)
        self.assertEqual(summary["requirement_receipts"], 1)
        self.assertEqual(summary["requirement_receipt_coverage_rate"], 1.0)
        self.assertEqual(summary["requirements_checked"], 2)
        self.assertEqual(summary["requirements_passed"], 1)
        self.assertEqual(summary["requirement_retention_rate"], 0.5)
        self.assertEqual(summary["all_requirements_passed_requests"], 0)
        self.assertEqual(summary["shadow_requirement_receipts"], 1)
        self.assertEqual(summary["shadow_requirements_checked"], 2)
        self.assertEqual(summary["shadow_requirements_passed"], 2)
        self.assertEqual(summary["paired_requirement_receipts"], 1)
        self.assertEqual(summary["measured_requirement_pass_delta"], -1)


if __name__ == "__main__":
    unittest.main()
