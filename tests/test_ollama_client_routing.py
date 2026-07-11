import os
import unittest
from unittest.mock import patch

from experiments.local_worker.ollama_client import DEFAULT_ENDPOINT, OllamaClient


class OllamaClientRoutingTests(unittest.TestCase):
    def test_default_client_uses_gateway_environment_variable(self):
        with patch.dict(os.environ, {"AIOPT_GATEWAY": "http://127.0.0.1:8800"}):
            client = OllamaClient()

        self.assertEqual(client.endpoint, "http://127.0.0.1:8800")

    def test_explicit_endpoint_overrides_gateway_environment_variable(self):
        with patch.dict(os.environ, {"AIOPT_GATEWAY": "http://127.0.0.1:8800"}):
            client = OllamaClient("http://127.0.0.1:11434")

        self.assertEqual(client.endpoint, "http://127.0.0.1:11434")

    def test_unset_gateway_falls_back_to_default_endpoint(self):
        with patch.dict(os.environ, {}, clear=True):
            client = OllamaClient()

        self.assertEqual(client.endpoint, DEFAULT_ENDPOINT)


if __name__ == "__main__":
    unittest.main()
