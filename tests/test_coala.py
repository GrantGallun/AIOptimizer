import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.brain_runtime.coala import (
    ActionKind,
    CoALAController,
    CognitiveAction,
    CycleLimitExceeded,
    LongTermMemoryKind,
)
from experiments.brain_runtime.session_runtime import PersistentBrainRuntime


class CoALAControllerTests(unittest.TestCase):
    def test_cycle_retrieves_reasons_then_grounds(self):
        memory = PersistentBrainRuntime()
        known = memory.remember(
            "parser",
            "The parser delimiter is pipe.",
            key="delimiter",
            value="pipe",
            long_term=True,
        )

        def reasoner(prompt, context):
            self.assertEqual(prompt, "choose delimiter")
            self.assertIn(known.id, [item.id for item in context.retrieved])
            return "Use the verified pipe delimiter."

        controller = CoALAController(
            memory,
            reasoner=reasoner,
            grounding={"answer": lambda args, context: f"{args['prefix']}: pipe"},
        )

        def policy(context):
            if not context.events:
                return CognitiveAction.retrieve("parser delimiter", limit=1)
            if len(context.events) == 1:
                return CognitiveAction.reason("choose delimiter")
            return CognitiveAction.ground("answer", prefix="result")

        result = controller.run_cycle("Fix parser", "A delimiter is required.", policy)
        self.assertEqual([event.action.kind for event in result.events], [ActionKind.RETRIEVE, ActionKind.REASON, ActionKind.GROUND])
        self.assertEqual(result.output, "result: pipe")
        self.assertEqual(result.working["thoughts"], ["Use the verified pipe delimiter."])
        self.assertEqual(result.retrieved_memory_ids, (known.id,))

    def test_learning_writes_typed_long_term_memory_that_persists(self):
        memory = PersistentBrainRuntime()
        controller = CoALAController(memory)
        action = CognitiveAction.learn(
            LongTermMemoryKind.SEMANTIC,
            "deployment",
            "The stable region is us-central.",
            key="region",
            value="us-central",
            confidence=0.9,
        )
        result = controller.run_cycle("Learn region", "Deployment succeeded.", lambda context: action)
        learned = memory.long_term_memory[result.output]
        self.assertEqual(learned.kind, LongTermMemoryKind.SEMANTIC.value)

        with TemporaryDirectory() as tmp:
            restored = PersistentBrainRuntime.load(memory.save(Path(tmp) / "session.json"))
        self.assertEqual(restored.retrieve("deployment region", limit=1)[0].value, "us-central")

    def test_procedural_learning_requires_explicit_opt_in(self):
        action = CognitiveAction.learn(LongTermMemoryKind.PROCEDURAL, "skill", "Run trusted procedure.")
        with self.assertRaisesRegex(PermissionError, "procedural learning is disabled"):
            CoALAController(PersistentBrainRuntime()).run_cycle("Learn", "Observe", lambda context: action)

        memory = PersistentBrainRuntime()
        result = CoALAController(memory, allow_procedural_learning=True).run_cycle(
            "Learn", "Observe", lambda context: action
        )
        self.assertEqual(memory.long_term_memory[result.output].kind, LongTermMemoryKind.PROCEDURAL.value)

    def test_retrieval_keeps_private_scope_out_of_decision_context(self):
        memory = PersistentBrainRuntime()
        public = memory.publish_cache("public", "Public parser guidance.", scope="project", source="public")
        private = memory.publish_cache("secret", "Private parser token.", scope="worker-b", source="worker-b")
        controller = CoALAController(memory, grounding={"done": lambda args, context: "done"})

        def policy(context):
            if not context.events:
                return CognitiveAction.retrieve("parser", limit=5)
            self.assertNotIn("secret", [item.topic for item in context.retrieved])
            return CognitiveAction.ground("done")

        result = controller.run_cycle("Use parser", "Need guidance", policy, scope="worker-a")
        with self.assertRaisesRegex(PermissionError, "differs from cycle scope"):
            controller.run_cycle(
                "Escalate",
                "Need private data",
                lambda context: CognitiveAction.retrieve("token", scope="worker-b"),
                scope="worker-a",
            )
        with self.assertRaisesRegex(PermissionError, "outside cycle scope"):
            controller.record_feedback(result, reward=1.0, memory_ids=[public.id, private.id])
        self.assertEqual(public.utility, 0.5)  # invalid credit set is atomic

    def test_feedback_records_episode_and_updates_explicit_credit(self):
        memory = PersistentBrainRuntime()
        credited = memory.remember("guide", "Useful procedure.", utility=0.5, long_term=True)
        untouched = memory.remember("other", "Unrelated note.", utility=0.5, long_term=True)
        controller = CoALAController(memory, grounding={"act": lambda args, context: "success"})

        def policy(context):
            if not context.events:
                return CognitiveAction.retrieve("guide", limit=1)
            return CognitiveAction.ground("act")

        result = controller.run_cycle("Use guide", "Start", policy, scope="worker-a")
        episode = controller.record_feedback(result, reward=1.0, memory_ids=[credited.id], learning_rate=0.2)
        self.assertEqual(credited.utility, 0.7)
        self.assertEqual(untouched.utility, 0.5)
        self.assertEqual(episode.kind, LongTermMemoryKind.EPISODIC.value)
        self.assertEqual(episode.scope, "worker-a")
        self.assertIn(credited.id, episode.links)
        self.assertIn("Reward: +1.000", episode.content)

    def test_cycle_limit_stops_unbounded_internal_reasoning(self):
        controller = CoALAController(
            PersistentBrainRuntime(),
            reasoner=lambda prompt, context: "still thinking",
            max_internal_actions=2,
        )
        with self.assertRaisesRegex(CycleLimitExceeded, "exceeded 2"):
            controller.run_cycle("Never finish", "Start", lambda context: CognitiveAction.reason("again"))

    def test_terminal_action_is_allowed_after_maximum_internal_actions(self):
        controller = CoALAController(
            PersistentBrainRuntime(),
            reasoner=lambda prompt, context: "thought",
            grounding={"done": lambda args, context: "complete"},
            max_internal_actions=2,
        )

        def policy(context):
            if len(context.events) < 2:
                return CognitiveAction.reason("think")
            return CognitiveAction.ground("done")

        result = controller.run_cycle("Finish", "Start", policy)
        self.assertEqual(result.output, "complete")
        self.assertEqual(len(result.events), 3)


if __name__ == "__main__":
    unittest.main()
