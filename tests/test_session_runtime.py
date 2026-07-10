import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.brain_runtime.session_runtime import PersistentBrainRuntime, SNAPSHOT_SCHEMA


class PersistentBrainRuntimeTests(unittest.TestCase):
    def _populated_runtime(self) -> PersistentBrainRuntime:
        runtime = PersistentBrainRuntime(decay_half_life=4.0, working_memory_limit=8)
        old = runtime.remember(
            "parser",
            "Delimiter is comma.",
            long_term=True,
            key="delimiter",
            value="comma",
            confidence=0.4,
            utility=0.6,
            source="old-test",
        )
        runtime.tick(3)
        current = runtime.publish_cache(
            "parser",
            "Failing test proves delimiter is pipe.",
            scope="project",
            key="delimiter",
            value="pipe",
            confidence=0.95,
            utility=0.9,
            source="new-test",
        )
        linked = runtime.remember(
            "parser-procedure",
            "Use structured parsing after retrieval.",
            kind="procedure",
            links={current.id, old.id},
            source="procedure-test",
        )
        runtime.retrieve("parser delimiter", limit=2)
        runtime.spawn_task("Verify parser behavior", reason="session test", memory_ids=[current.id, linked.id])
        runtime.tick(7)
        return runtime

    def test_snapshot_round_trip_preserves_complete_state(self):
        runtime = self._populated_runtime()
        payload = runtime.snapshot()
        restored = PersistentBrainRuntime.from_snapshot(payload)

        self.assertEqual(payload["schema"], SNAPSHOT_SCHEMA)
        self.assertEqual(restored.snapshot(), payload)
        self.assertEqual(restored.clock, runtime.clock)
        self.assertEqual(restored.contradictions, runtime.contradictions)
        self.assertEqual(restored.task_hooks, runtime.task_hooks)

    def test_round_trip_preserves_shared_memory_object_identity_and_scope(self):
        restored = PersistentBrainRuntime.from_snapshot(self._populated_runtime().snapshot())
        shared = next(iter(restored.shared_cache.values()))

        self.assertIs(restored.working_memory[shared.id], shared)
        self.assertEqual(restored.read_cache("delimiter", scope="worker-a", limit=1)[0].value, "pipe")
        restored.publish_cache("secret", "private", scope="worker-b", source="worker-b")
        visible = restored.read_cache("private", scope="worker-a", limit=5)
        self.assertNotIn("secret", [item.topic for item in visible])

    def test_save_load_is_valid_json_and_counters_continue(self):
        runtime = self._populated_runtime()
        with TemporaryDirectory() as tmp:
            path = runtime.save(Path(tmp) / "nested" / "session.json")
            parsed = json.loads(path.read_text(encoding="utf-8"))
            restored = PersistentBrainRuntime.load(path)

        self.assertEqual(parsed["schema"], SNAPSHOT_SCHEMA)
        existing_ids = set(restored.working_memory) | set(restored.long_term_memory)
        new_memory = restored.remember("next", "Later-session observation.")
        new_task = restored.spawn_task("Later-session task", reason="counter test", memory_ids=[new_memory.id])
        self.assertNotIn(new_memory.id, existing_ids)
        self.assertEqual(new_task.id, "task-002")

    def test_rejects_unknown_schema_and_dangling_store_reference(self):
        payload = self._populated_runtime().snapshot()
        payload["schema"] = "future-schema"
        with self.assertRaisesRegex(ValueError, "unsupported snapshot schema"):
            PersistentBrainRuntime.from_snapshot(payload)

        payload = self._populated_runtime().snapshot()
        payload["stores"]["working_memory"]["broken"] = "mem-missing"
        with self.assertRaisesRegex(ValueError, "references missing memory"):
            PersistentBrainRuntime.from_snapshot(payload)


if __name__ == "__main__":
    unittest.main()

