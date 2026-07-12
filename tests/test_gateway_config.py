import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gateway.__main__ import build_middlewares, parse_args
from gateway.config import DEFAULT_CONFIG, load_config, validate_config, write_default


class GatewayConfigTests(unittest.TestCase):
    def test_default_template_is_explicitly_attention_off_and_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aioptimizer.json"
            write_default(path)
            loaded = load_config(path)
        self.assertEqual(loaded, DEFAULT_CONFIG)
        self.assertFalse(loaded["attention_context"])

    def test_unknown_and_invalid_values_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "unknown"):
            validate_config({"mystery": True})
        with self.assertRaisesRegex(ValueError, "between"):
            validate_config({"shadow_rate": 2.0})
        with self.assertRaisesRegex(ValueError, "boolean"):
            validate_config({"attention_context": "yes"})

    def test_config_overrides_defaults_and_cli_overrides_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"port": 9900, "attention_context": True}), encoding="utf-8")
            args = parse_args(["--config", str(path)])
            overridden = parse_args(["--config", str(path), "--port", "9911"])
            disabled = parse_args(["--config", str(path), "--no-attention-context"])
        self.assertEqual(args.port, 9900)
        self.assertTrue(args.attention_context)
        self.assertEqual(overridden.port, 9911)
        self.assertFalse(disabled.attention_context)

    def test_environment_selects_config_without_recreating_variables(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"ledger": "persistent.jsonl"}), encoding="utf-8")
            with patch.dict("os.environ", {"AIOPT_CONFIG": str(path)}):
                args = parse_args([])
        self.assertEqual(args.ledger, "persistent.jsonl")

    def test_middleware_order_preserves_original_cache_boundary(self):
        args = parse_args(["--attention-context"])
        names = [type(middleware).__name__ for middleware in build_middlewares(args)]
        self.assertEqual(names, [
            "ExactCacheMiddleware", "AttentionContextMiddleware", "CompactContextMiddleware",
        ])
