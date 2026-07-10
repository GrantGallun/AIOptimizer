import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.activation_memory.memory_harness import (
    CHANNELS,
    DEFAULT_ADVERSARIAL,
    DEFAULT_DEV,
    DEFAULT_HIDDEN,
    DEFAULT_MEMORIES,
    MEMORY_TYPES,
    aggregate,
    interaction_and_gate,
    load_memories,
    rank_sweep,
    run_toy,
    wrong_memory_id,
)
from experiments.activation_steering.steering_harness import read_jsonl


TASK_FILES = [DEFAULT_DEV, DEFAULT_HIDDEN, DEFAULT_ADVERSARIAL]


class DataIntegrityTests(unittest.TestCase):
    def test_memories_have_required_fields(self):
        memories = load_memories(DEFAULT_MEMORIES)
        self.assertGreaterEqual(len(memories), 4)
        for mem in memories.values():
            self.assertIn(mem["type"], MEMORY_TYPES)
            self.assertTrue(mem["text"])
            if mem["type"] == "procedural":
                self.assertIn(mem["behavior"], {"evidence", "clarify", "concise"})
            else:
                self.assertIsNotNone(mem["fact_value"])
                self.assertIsNotNone(mem["distractor_value"])

    def test_each_type_has_a_wrong_memory_control(self):
        # wrong_memory_id needs >= 2 memories per type to build the same-type control.
        memories = load_memories(DEFAULT_MEMORIES)
        for mid, mem in memories.items():
            wrong = wrong_memory_id(mid, memories)
            self.assertNotEqual(wrong, mid)
            self.assertEqual(memories[wrong]["type"], mem["type"])

    def test_tasks_reference_real_memories_and_both_types_present(self):
        memories = load_memories(DEFAULT_MEMORIES)
        for path in TASK_FILES:
            rows = read_jsonl(path)
            self.assertGreater(len(rows), 0)
            types = {row["type"] for row in rows}
            self.assertEqual(types, set(MEMORY_TYPES), f"{path} must cover both memory types")
            for row in rows:
                self.assertIn(row["memory_id"], memories)
                self.assertTrue(row["positive_answer"])
                self.assertTrue(row["negative_answer"])


class ToyInteractionTests(unittest.TestCase):
    def _run(self, tasks: Path) -> dict:
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "toy_run.json"
            return run_toy(
                Namespace(
                    memories=str(DEFAULT_MEMORIES),
                    tasks=str(tasks),
                    out=str(out),
                    dims=12,
                    coefficient=1.5,
                    min_gap=0.5,
                    include_rows=False,
                )
            )

    def test_toy_shows_predicted_crossover_on_hidden(self):
        payload = self._run(DEFAULT_HIDDEN)
        agg = payload["aggregate"]
        # Procedural: activation should beat in-context.
        self.assertGreater(
            agg["procedural"]["activation"]["margin_lift"],
            agg["procedural"]["in_context"]["margin_lift"],
        )
        # Factual: in-context should beat activation.
        self.assertGreater(
            agg["factual"]["in_context"]["margin_lift"],
            agg["factual"]["activation"]["margin_lift"],
        )

    def test_toy_gate_passes_on_hidden(self):
        payload = self._run(DEFAULT_HIDDEN)
        interaction = payload["interaction"]
        self.assertTrue(interaction["passed"], interaction)
        self.assertEqual(interaction["verdict"], "Confirmed")
        self.assertGreaterEqual(interaction["interaction_gap"], 0.5)

    def test_all_channels_scored(self):
        payload = self._run(DEFAULT_DEV)
        for mtype in MEMORY_TYPES:
            for channel in CHANNELS:
                self.assertIn(channel, payload["aggregate"][mtype])


class AggregateTests(unittest.TestCase):
    def test_interaction_gate_refutes_when_no_gap(self):
        # Flat aggregate: every channel identical -> no interaction, must refute.
        flat = {
            mtype: {ch: {"tasks": 1, "success_rate": 0.0, "margin_lift": 0.0} for ch in CHANNELS}
            for mtype in MEMORY_TYPES
        }
        gate = interaction_and_gate(flat, min_gap=0.5)
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["verdict"], "Refuted")

    def test_aggregate_uses_none_as_baseline(self):
        rows = [
            {
                "id": "t1",
                "type": "procedural",
                "memory_id": "proc-evidence",
                "behavior": "evidence",
                "channels": {ch: {"margin": 0.0, "success": False} for ch in CHANNELS}
                | {"none": {"margin": 1.0, "success": True}, "activation": {"margin": 3.0, "success": True}},
            }
        ]
        agg = aggregate(rows)
        # activation lift = 3.0 - 1.0 = 2.0
        self.assertAlmostEqual(agg["procedural"]["activation"]["margin_lift"], 2.0)


class SweepRankingTests(unittest.TestCase):
    def _point(self, layer, coeff, gap, adv_proc, proc_act_lift):
        return {
            "layer": layer,
            "coefficient": coeff,
            "aggregate": {"procedural": {"activation": {"margin_lift": proc_act_lift}}},
            "interaction": {"interaction_gap": gap, "adv_proc": adv_proc},
        }

    def test_rank_sweep_prefers_largest_gap(self):
        points = [
            self._point(8, 2, gap=0.2, adv_proc=0.1, proc_act_lift=0.5),
            self._point(12, 8, gap=1.9, adv_proc=1.0, proc_act_lift=2.0),
            self._point(20, 4, gap=1.9, adv_proc=1.5, proc_act_lift=1.0),
        ]
        ranked = rank_sweep(points)
        # Biggest gap wins; ties on gap broken by adv_proc.
        self.assertEqual((ranked[0]["layer"], ranked[0]["coefficient"]), (20, 4))
        self.assertEqual(ranked[-1]["layer"], 8)


if __name__ == "__main__":
    unittest.main()
