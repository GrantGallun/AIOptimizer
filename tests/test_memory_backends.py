import asyncio
import unittest
from types import SimpleNamespace

from experiments.brain_runtime.backend_comparison_v1 import run_comparison
from experiments.brain_runtime.memory_backends import (
    BackendUnavailable,
    GraphitiMemoryBackend,
    LangGraphMemoryBackend,
    Mem0MemoryBackend,
    MemoryRecord,
    NativeMemoryBackend,
    create_backend,
    deterministic_embeddings,
)


class MemoryBackendContractTests(unittest.TestCase):
    def _records(self):
        return [
            MemoryRecord("public", "parser", "Verified delimiter is pipe.", "project", "test", key="delimiter", value="pipe", confidence=0.9, utility=0.9),
            MemoryRecord("private", "secret", "Worker B token is secret-1234.", "worker-b", "worker-b"),
        ]

    def _assert_scope_contract(self, backend):
        for record in self._records():
            backend.add(record)
        backend.begin_session("later")
        hits = backend.search("verified parser delimiter", scope="worker-a", limit=5)
        self.assertEqual(hits[0].record.id, "public")
        self.assertNotIn("private", [hit.record.id for hit in hits])

    def test_native_contract_and_cross_session_snapshot(self):
        self._assert_scope_contract(NativeMemoryBackend("test"))

    def test_langgraph_contract_with_real_inmemory_store(self):
        self._assert_scope_contract(LangGraphMemoryBackend("test"))

    def test_deterministic_embedding_is_stable_and_normalized(self):
        first, second = deterministic_embeddings(["parser delimiter", "parser delimiter"])
        self.assertEqual(first, second)
        self.assertAlmostEqual(sum(value * value for value in first), 1.0)


class FakeMem0:
    def __init__(self):
        self.rows = []

    def add(self, content, **kwargs):
        self.rows.append((content, kwargs))

    def search(self, query, *, filters, limit):
        user_id = filters["user_id"]
        matches = []
        for index, (content, kwargs) in enumerate(self.rows):
            if kwargs["user_id"] == user_id:
                matches.append({"id": str(index), "memory": content, "metadata": kwargs["metadata"], "score": 1.0})
        return {"results": matches[:limit]}


class FakeGraphiti:
    def __init__(self):
        self.episodes = []

    async def add_episode(self, **kwargs):
        self.episodes.append(kwargs)

    async def search(self, *, query, group_id):
        return [SimpleNamespace(uuid="edge-1", fact=f"Fact for {group_id}: {query}", score=0.8)]

    async def close(self):
        return None


class OptionalAdapterTests(unittest.TestCase):
    def test_mem0_maps_scope_to_filtered_entities_and_preserves_metadata(self):
        backend = Mem0MemoryBackend("project", memory=FakeMem0())
        record = MemoryRecord("r1", "parser", "Verified delimiter is pipe.", "project", "test")
        backend.add(record)
        hits = backend.search("delimiter", scope="worker-a", limit=3)
        self.assertEqual(hits[0].record, record)
        self.assertEqual(backend.memory.rows[0][1]["infer"], False)

    def test_graphiti_maps_scope_to_group_id(self):
        graph = FakeGraphiti()
        backend = GraphitiMemoryBackend("project", graph=graph, episode_type_json="json")
        backend.add(MemoryRecord("r1", "parser", "Delimiter is pipe.", "project", "test"))
        hits = backend.search("delimiter", scope="worker-a", limit=2)
        self.assertEqual(graph.episodes[0]["group_id"], "project:project")
        self.assertNotIn("project:worker-b", [hit.record.scope for hit in hits])
        backend.close()

    def test_factories_report_missing_service_configuration_cleanly(self):
        with self.assertRaises(BackendUnavailable):
            create_backend("mem0", "project")


class ComparisonHarnessTests(unittest.TestCase):
    def test_native_and_langgraph_run_same_seeded_cases(self):
        payload = run_comparison(["native", "langgraph"], seeds=[11, 23])
        self.assertFalse(payload["preregistered"])
        self.assertFalse(payload["model_calls"])
        self.assertEqual([item["available"] for item in payload["availability"]], [True, True])
        self.assertEqual({result["summary"]["tasks"] for result in payload["results"]}, {8})
        self.assertTrue(all(result["summary"]["privacy_leaks"] == 0 for result in payload["results"]))

    def test_unavailable_backend_is_reported_without_aborting_available_ones(self):
        payload = run_comparison(["native", "mem0"], seeds=[11])
        self.assertEqual(len(payload["results"]), 1)
        self.assertTrue(payload["availability"][0]["available"])
        self.assertFalse(payload["availability"][1]["available"])


if __name__ == "__main__":
    unittest.main()

