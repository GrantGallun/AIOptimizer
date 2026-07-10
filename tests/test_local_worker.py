import unittest
from unittest.mock import patch

from experiments.local_worker.ollama_client import DEFAULT_MODEL, OllamaClient, diagnose


class LocalWorkerTests(unittest.TestCase):
    @patch("experiments.local_worker.ollama_client.OllamaClient.list_models")
    def test_diagnose_recognizes_installed_recommended_model(self, list_models):
        list_models.return_value = [{"name": DEFAULT_MODEL}, {"name": "other:latest"}]

        status = diagnose()

        self.assertTrue(status.reachable)
        self.assertTrue(status.recommended_model_ready)
        self.assertIn(DEFAULT_MODEL, status.installed_models)

    @patch("experiments.local_worker.ollama_client.OllamaClient.list_models")
    def test_diagnose_reports_unreachable_service(self, list_models):
        list_models.side_effect = RuntimeError("connection refused")

        status = diagnose()

        self.assertFalse(status.reachable)
        self.assertFalse(status.recommended_model_ready)
        self.assertIn("connection refused", status.message)

    def test_generation_disables_thinking_for_machine_scored_runs(self):
        client = OllamaClient()
        captured = {}

        def request(_path, body=None):
            captured.update(body or {})
            return {"response": "ready"}

        client._request = request

        result = client.generate_with_metrics("test")

        self.assertEqual(result.text, "ready")
        self.assertFalse(captured["think"])


if __name__ == "__main__":
    unittest.main()
