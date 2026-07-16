import json
import hashlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from aioptimizer.episodes import (
    DATASET_SCHEMA,
    EVENT_SCHEMA,
    HEALTH_SCHEMA,
    EpisodeEventLedger,
    build_dataset,
    build_hard_cases,
    content_free_measurements,
    discover_event_paths,
    episode_id_from_payload,
    inspect_event_stream,
    inspect_workspace,
    main,
    make_event,
    replay_rows,
)


class EpisodeReceiptTests(unittest.TestCase):
    def test_episode_id_is_stable_without_exposing_session_value(self):
        payload = {"session_id": "private-session-name", "transcript_path": "ignored"}
        first = episode_id_from_payload(payload)
        self.assertEqual(first, episode_id_from_payload(payload))
        self.assertNotIn("private", first)
        self.assertTrue(first.startswith("ep-"))

    def test_event_contract_rejects_content_and_unbounded_categories(self):
        with self.assertRaisesRegex(ValueError, "content-free"):
            content_free_measurements({"prompt_text": "do the secret task"})
        with self.assertRaisesRegex(ValueError, "bounded category"):
            make_event(
                source="hook", event_type="route", episode_id="episode-1234",
                route="raw with arbitrary prose",
            )

    def test_event_ledger_appends_schema_valid_rows(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            ledger = EpisodeEventLedger(path, source="agent_bus")
            ledger.record(
                "task_executed", episode_id="bus-t0001", producer_ok=False,
                cost_seconds=0.25, attempt=0,
            )
            row = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(row["schema"], EVENT_SCHEMA)
            self.assertFalse(row["producer_ok"])
            self.assertNotIn("producer_result", row)


class EpisodeDatasetTests(unittest.TestCase):
    def _events(self, root: Path) -> Path:
        path = root / "events_v1.jsonl"
        rows = [
            make_event(
                source="codex_hook", event_type="context_route",
                episode_id="episode-0001", turn_id="turn-00000001",
                route="attention", injected=True, history_chars=12000,
            ),
            make_event(
                source="agent_bus", event_type="task_verified",
                episode_id="episode-0001", verification_ok=False,
            ),
            make_event(
                source="agent_bus", event_type="task_retry",
                episode_id="episode-0001", retry_scheduled=True, attempt=1,
            ),
            make_event(
                source="agent_bus", event_type="task_terminal",
                episode_id="episode-0001", eventual_success=True,
            ),
            make_event(
                source="claude_hook", event_type="context_route",
                episode_id="episode-0002", turn_id="turn-00000002",
                route="raw", injected=False, history_chars=100,
            ),
            make_event(
                source="agent_bus", event_type="task_terminal",
                episode_id="episode-0002", eventual_success=True,
            ),
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path

    def test_builds_joined_frozen_split_and_hard_case_view(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = self._events(root)
            dataset_path = root / "dataset_v1.json"
            dataset = build_dataset(
                [events], dataset_path, split_salt="frozen-test-salt",
                hidden_fraction=0.5,
            )
            self.assertEqual(dataset["schema"], DATASET_SCHEMA)
            self.assertEqual(dataset["episode_count"], 2)
            hard = next(row for row in dataset["episodes"] if row["episode_id"] == "episode-0001")
            self.assertEqual(
                hard["hard_reason_codes"], ["retry_required", "verification_failure"]
            )
            self.assertTrue(hard["eventual_success"])

            hard_path = root / "hard_v1.json"
            hard_artifact = build_hard_cases(dataset_path, hard_path, split=hard["split"])
            self.assertIn("episode-0001", [row["episode_id"] for row in hard_artifact["episodes"]])
            replay = replay_rows(dataset_path, split=hard["split"], hard_only=True)
            self.assertIn("episode-0001", [row["episode_id"] for row in replay])

    def test_split_is_stable_and_versioned_artifacts_never_overwrite(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = self._events(root)
            first_path = root / "dataset_v1.json"
            second_path = root / "dataset_v2.json"
            first = build_dataset([events], first_path, split_salt="same-salt")
            second = build_dataset([events], second_path, split_salt="same-salt")
            self.assertEqual(
                [(row["episode_id"], row["split"]) for row in first["episodes"]],
                [(row["episode_id"], row["split"]) for row in second["episodes"]],
            )
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                build_dataset([events], first_path, split_salt="same-salt")

    def test_dataset_rejects_legacy_or_malformed_rows_instead_of_guessing(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "legacy.jsonl"
            path.write_text('{"episode_id":"episode-0001","prompt":"secret"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "content-bearing"):
                build_dataset([path], root / "dataset_v1.json", split_salt="salt")

    def test_known_legacy_query_fingerprint_rows_are_ignored_not_imported(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "mixed.jsonl"
            event = make_event(
                source="codex_hook", event_type="context_route",
                episode_id="episode-legacy-0001", route="raw",
            )
            path.write_text(
                json.dumps({"query_sha256_16": "deadbeef", "route": "raw"})
                + "\n" + json.dumps(event) + "\n",
                encoding="utf-8",
            )
            dataset = build_dataset(
                [path], root / "dataset_v1.json", split_salt="salt"
            )
            self.assertEqual(dataset["event_count"], 1)

    def test_dataset_refuses_to_freeze_an_empty_or_legacy_only_stream(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "legacy.jsonl"
            path.write_text('{"route":"raw"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "at least one"):
                build_dataset([path], root / "dataset_v1.json", split_salt="salt")

    def test_checked_in_plumbing_artifacts_cover_dev_hidden_and_hard_replay(self):
        root = Path(__file__).resolve().parents[1] / "results" / "episodes"
        dataset = json.loads(
            (root / "plumbing_dataset_v1.json").read_text(encoding="utf-8")
        )
        hard = json.loads(
            (root / "plumbing_hard_cases_dev_v1.json").read_text(encoding="utf-8")
        )
        self.assertEqual(dataset["split_counts"], {"dev": 1, "hidden": 1})
        self.assertEqual(dataset["hard_episode_count"], 1)
        self.assertEqual(hard["episode_count"], 1)
        self.assertEqual(hard["episodes"][0]["split"], "dev")
        self.assertEqual(
            dataset["source_manifest"][0]["sha256"],
            hashlib.sha256((root / "plumbing_events_v1.jsonl").read_bytes()).hexdigest(),
        )
        self.assertEqual(
            hard["dataset_sha256"],
            hashlib.sha256((root / "plumbing_dataset_v1.json").read_bytes()).hexdigest(),
        )
        serialized = json.dumps(dataset).lower()
        for forbidden in ("prompt_text", "response_text", "producer_result", "acceptance"):
            self.assertNotIn(forbidden, serialized)

    def test_v2_plumbing_artifacts_cover_the_complete_product_signal_path(self):
        root = Path(__file__).resolve().parents[1] / "results" / "episodes"
        events = root / "plumbing_events_v2.jsonl"
        dataset_path = root / "plumbing_dataset_v2.json"
        hard_path = root / "plumbing_hard_cases_dev_v2.json"
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        hard = json.loads(hard_path.read_text(encoding="utf-8"))
        health = inspect_event_stream([events])
        self.assertEqual(dataset["event_count"], 12)
        self.assertEqual(dataset["split_counts"], {"dev": 1, "hidden": 1})
        self.assertEqual(hard["episode_count"], 1)
        for signal in (
            "context_route", "provider_response", "provider_usage", "latency",
            "acceptance_test", "verification", "requirements", "eventual_outcome",
            "cross_source_join",
        ):
            self.assertEqual(health["coverage"][signal]["rate"], 1.0, signal)
        self.assertEqual(health["warnings"], [])
        self.assertEqual(
            dataset["source_manifest"][0]["sha256"], hashlib.sha256(events.read_bytes()).hexdigest()
        )
        self.assertEqual(
            hard["dataset_sha256"], hashlib.sha256(dataset_path.read_bytes()).hexdigest()
        )


class EpisodeHealthTests(unittest.TestCase):
    def test_workspace_discovery_finds_standard_and_extra_streams_once(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            standard = root / ".aioptimizer" / "codex_hook_ledger.jsonl"
            extra = root / "custom" / "events.jsonl"
            standard.parent.mkdir(parents=True)
            extra.parent.mkdir(parents=True)
            standard.write_text("", encoding="utf-8")
            extra.write_text("", encoding="utf-8")
            discovered = discover_event_paths(
                root, extra_paths=("custom/events.jsonl", extra)
            )
            self.assertEqual(discovered, (standard, extra))

    def test_inspector_reports_cross_layer_and_outcome_coverage_without_ids(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "events.jsonl"
            rows = [
                make_event(
                    source="codex_hook", event_type="context_route",
                    episode_id="episode-health-0001", turn_id="turn-health-000001",
                    route="attention", latency_ms=1.5,
                ),
                make_event(
                    source="gateway", event_type="provider_response",
                    episode_id="episode-health-0001", turn_id="turn-health-000001",
                    status=200, input_tokens=100, output_tokens=20,
                ),
                make_event(
                    source="agent_bus", event_type="task_executed",
                    episode_id="episode-health-0001", test_executed=True, test_ok=True,
                ),
                make_event(
                    source="agent_bus", event_type="task_verified",
                    episode_id="episode-health-0001", verification_ok=False,
                ),
                make_event(
                    source="agent_bus", event_type="task_retry",
                    episode_id="episode-health-0001", retry_scheduled=True,
                ),
                make_event(
                    source="agent_bus", event_type="task_terminal",
                    episode_id="episode-health-0001", eventual_success=True,
                ),
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            summary = inspect_event_stream([path])
            self.assertEqual(summary["schema"], HEALTH_SCHEMA)
            self.assertEqual(summary["episode_count"], 1)
            self.assertEqual(summary["hard_episode_count"], 1)
            self.assertEqual(summary["eventual_success_episodes"], 1)
            self.assertEqual(summary["coverage"]["cross_source_join"]["rate"], 1.0)
            self.assertEqual(summary["coverage"]["provider_usage"]["rate"], 1.0)
            self.assertEqual(summary["coverage"]["acceptance_test"]["rate"], 1.0)
            serialized = json.dumps(summary)
            self.assertNotIn("episode-health-0001", serialized)
            self.assertNotIn("turn-health-000001", serialized)

    def test_empty_workspace_is_diagnostic_not_an_exception(self):
        with TemporaryDirectory() as tmp:
            summary = inspect_workspace(tmp)
        self.assertEqual(summary["event_count"], 0)
        self.assertEqual(summary["coverage"]["eventual_outcome"]["rate"], None)
        self.assertEqual(summary["warnings"], ["no_event_files"])

    def test_build_cli_discovers_workspace_stream_without_manual_event_paths(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            event_path = root / ".aioptimizer" / "codex_hook_ledger.jsonl"
            event_path.parent.mkdir(parents=True)
            event_path.write_text(json.dumps(make_event(
                source="codex_hook", event_type="context_route",
                episode_id="episode-cli-000001", route="raw",
            )) + "\n", encoding="utf-8")
            output_path = root / "results" / "episodes" / "dataset_v1.json"
            stdout = io.StringIO()
            with patch("sys.argv", [
                "aioptimizer-episodes", "build", "--workspace", str(root),
                "--out", str(output_path), "--split-salt", "cli-salt-v1",
            ]), patch("sys.stdout", stdout):
                main()
            result = json.loads(stdout.getvalue())
            self.assertEqual(result["event_count"], 1)
            self.assertEqual(result["episode_count"], 1)
            self.assertTrue(output_path.is_file())


if __name__ == "__main__":
    unittest.main()
