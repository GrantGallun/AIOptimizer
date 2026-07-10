import unittest

from experiments.brain_runtime.backend_comparison_v2 import DEV_SEEDS, HIDDEN_SEEDS, run_v2
from experiments.brain_runtime.memory_backends import MemoryRecord
from experiments.brain_runtime.memory_backends_v2 import PriorityNativeMemoryBackend


class PriorityNativeMemoryBackendTests(unittest.TestCase):
    def test_frequency_cannot_override_more_relevant_scenario(self):
        backend = PriorityNativeMemoryBackend("project")
        old = MemoryRecord(
            "old-popular",
            "s11-procedure",
            "For scenario s11, validate with python unittest.",
            "project",
            "procedure",
            confidence=0.95,
            utility=0.95,
        )
        relevant = MemoryRecord(
            "relevant",
            "s23-nested-payload",
            "Scenario s23 fixed nested payload parsing with structured JSON.",
            "project",
            "retrospective",
            confidence=0.8,
            utility=0.8,
        )
        backend.add(old)
        for _ in range(20):
            backend.search("scenario s11 unittest", scope="worker-a", limit=1)
        backend.add(relevant)

        hits = backend.search("scenario s23 nested payload parsing", scope="worker-a", limit=1)
        self.assertEqual(hits[0].record.id, "relevant")
        self.assertGreater(hits[0].backend_metadata["rank"]["relevance"], 0)

    def test_contradicted_memory_loses_when_relevance_ties(self):
        backend = PriorityNativeMemoryBackend("project")
        backend.add(MemoryRecord("old", "parser", "Parser delimiter is comma.", "project", "spec", key="delimiter", value="comma", confidence=0.4))
        backend.add(MemoryRecord("new", "parser", "Parser delimiter is pipe.", "project", "test", key="delimiter", value="pipe", confidence=0.95))
        hits = backend.search("parser delimiter", scope="worker-a", limit=2)
        self.assertEqual(hits[0].record.id, "new")

    def test_scope_filter_precedes_ranking(self):
        backend = PriorityNativeMemoryBackend("project")
        backend.add(MemoryRecord("private", "parser", "Worker B parser secret.", "worker-b", "worker-b", confidence=1.0, utility=1.0))
        backend.add(MemoryRecord("public", "parser", "Project parser guidance.", "project", "public", confidence=0.4, utility=0.4))
        hits = backend.search("parser", scope="worker-a", limit=5)
        self.assertEqual([hit.record.id for hit in hits], ["public"])


class BackendComparisonV2Tests(unittest.TestCase):
    def test_splits_are_fixed_and_disjoint(self):
        self.assertEqual(DEV_SEEDS, [11, 23, 37, 41, 59])
        self.assertEqual(HIDDEN_SEEDS, [101, 103, 107, 109, 113])
        self.assertFalse(set(DEV_SEEDS) & set(HIDDEN_SEEDS))

    def test_dev_reproduces_v1_failure_and_priority_v2_fixes_it(self):
        payload = run_v2("dev", backend_names=["native_v1", "native_priority_v2"])
        summaries = {result["requested_name"]: result["summary"] for result in payload["results"]}
        self.assertEqual(summaries["native_v1"]["successes"], 16)
        self.assertEqual(summaries["native_priority_v2"]["successes"], 20)
        self.assertEqual(summaries["native_priority_v2"]["privacy_leaks"], 0)


if __name__ == "__main__":
    unittest.main()

