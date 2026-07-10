#!/usr/bin/env python3
"""Model-agnostic CoALA decision cycles over governed persistent memory.

The controller follows the decomposition in Sumers et al. (2023): retrieval and
reasoning actions support planning, then a learning or grounding action is
selected and executed.  It deliberately accepts callbacks for model reasoning,
policy selection, and environment grounding so architecture mechanics can be
tested without embedding a particular model or research benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from experiments.brain_runtime.runtime import Evidence, MemoryItem, clamp
from experiments.brain_runtime.session_runtime import PersistentBrainRuntime


class ActionKind(str, Enum):
    RETRIEVE = "retrieve"
    REASON = "reason"
    LEARN = "learn"
    GROUND = "ground"


class LongTermMemoryKind(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class CycleLimitExceeded(RuntimeError):
    """The policy did not select a terminal learning/grounding action in time."""


@dataclass(frozen=True)
class CognitiveAction:
    kind: ActionKind
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def retrieve(cls, query: str, *, limit: int = 5, scope: str | None = None) -> "CognitiveAction":
        arguments: dict[str, Any] = {"query": query, "limit": limit}
        if scope is not None:
            arguments["scope"] = scope
        return cls(ActionKind.RETRIEVE, "retrieve", arguments)

    @classmethod
    def reason(cls, prompt: str) -> "CognitiveAction":
        return cls(ActionKind.REASON, "reason", {"prompt": prompt})

    @classmethod
    def learn(
        cls,
        memory_kind: LongTermMemoryKind,
        topic: str,
        content: str,
        **metadata: Any,
    ) -> "CognitiveAction":
        return cls(
            ActionKind.LEARN,
            "learn",
            {"memory_kind": memory_kind.value, "topic": topic, "content": content, **metadata},
        )

    @classmethod
    def ground(cls, name: str, **arguments: Any) -> "CognitiveAction":
        return cls(ActionKind.GROUND, name, arguments)


@dataclass(frozen=True)
class ActionEvent:
    action: CognitiveAction
    output: str
    memory_ids: tuple[str, ...] = ()


@dataclass
class DecisionContext:
    goal: str
    observation: str
    scope: str
    working: dict[str, Any]
    retrieved: list[MemoryItem] = field(default_factory=list)
    events: list[ActionEvent] = field(default_factory=list)


@dataclass(frozen=True)
class CycleResult:
    goal: str
    initial_observation: str
    scope: str
    terminal_action: CognitiveAction
    output: str
    events: tuple[ActionEvent, ...]
    retrieved_memory_ids: tuple[str, ...]
    working: dict[str, Any]


Policy = Callable[[DecisionContext], CognitiveAction]
Reasoner = Callable[[str, DecisionContext], str]
GroundingHandler = Callable[[dict[str, Any], DecisionContext], str]


class CoALAController:
    """Run bounded CoALA planning/execution cycles over a persistent memory."""

    def __init__(
        self,
        memory: PersistentBrainRuntime,
        *,
        reasoner: Reasoner | None = None,
        grounding: dict[str, GroundingHandler] | None = None,
        max_internal_actions: int = 8,
        allow_procedural_learning: bool = False,
    ) -> None:
        if max_internal_actions < 1:
            raise ValueError("max_internal_actions must be positive")
        self.memory = memory
        self.reasoner = reasoner
        self.grounding = dict(grounding or {})
        self.max_internal_actions = max_internal_actions
        self.allow_procedural_learning = allow_procedural_learning

    def run_cycle(
        self,
        goal: str,
        observation: str,
        policy: Policy,
        *,
        scope: str = "project",
    ) -> CycleResult:
        """Plan with retrieval/reasoning, then execute one learning/grounding action."""
        context = DecisionContext(
            goal=goal,
            observation=observation,
            scope=scope,
            working={"goal": goal, "observation": observation, "thoughts": []},
        )
        internal_actions = 0
        for _ in range(self.max_internal_actions + 1):
            action = policy(context)
            if not isinstance(action, CognitiveAction):
                raise TypeError("policy must return CognitiveAction")
            if action.kind in {ActionKind.RETRIEVE, ActionKind.REASON} and internal_actions >= self.max_internal_actions:
                raise CycleLimitExceeded(
                    f"policy exceeded {self.max_internal_actions} internal actions without learning or grounding"
                )
            event = self._execute(action, context)
            context.events.append(event)
            if action.kind in {ActionKind.LEARN, ActionKind.GROUND}:
                return CycleResult(
                    goal=goal,
                    initial_observation=observation,
                    scope=scope,
                    terminal_action=action,
                    output=event.output,
                    events=tuple(context.events),
                    retrieved_memory_ids=tuple(dict.fromkeys(item.id for item in context.retrieved)),
                    working=dict(context.working),
                )
            internal_actions += 1
        raise CycleLimitExceeded(
            f"policy exceeded {self.max_internal_actions} internal actions without learning or grounding"
        )

    def record_feedback(
        self,
        result: CycleResult,
        *,
        reward: float,
        memory_ids: list[str] | tuple[str, ...] | None = None,
        note: str = "",
        learning_rate: float = 0.1,
        source: str = "environment-feedback",
    ) -> MemoryItem:
        """Record an episode and reinforce explicitly credited memories.

        ``reward`` is normalized to ``[-1, 1]``. Memory utility changes only for
        the caller-selected IDs (or the cycle's retrieved memories by default),
        making credit assignment visible rather than silently inferred.
        """
        if not -1.0 <= reward <= 1.0:
            raise ValueError("reward must be between -1 and 1")
        if not 0.0 <= learning_rate <= 1.0:
            raise ValueError("learning_rate must be between 0 and 1")
        credited = tuple(dict.fromkeys(memory_ids if memory_ids is not None else result.retrieved_memory_ids))
        credited_memories: list[MemoryItem] = []
        for memory_id in credited:
            memory = self._memory_by_id(memory_id)
            if memory.scope not in {result.scope, "project", "global"}:
                raise PermissionError(f"memory {memory_id!r} is outside cycle scope {result.scope!r}")
            credited_memories.append(memory)
        for memory in credited_memories:
            memory.utility = clamp(memory.utility + learning_rate * reward)
            memory.evidence.append(
                Evidence(
                    source=source,
                    confidence=(reward + 1.0) / 2.0,
                    observed_at=self.memory.clock,
                    note=note or f"cycle reward {reward:+.3f}",
                )
            )

        episode_content = (
            f"Goal: {result.goal}\n"
            f"Observation: {result.initial_observation}\n"
            f"Action: {result.terminal_action.kind.value}:{result.terminal_action.name}\n"
            f"Outcome: {result.output}\n"
            f"Reward: {reward:+.3f}"
        )
        return self.memory.remember(
            topic=f"episode:{result.terminal_action.name}",
            content=episode_content,
            kind=LongTermMemoryKind.EPISODIC.value,
            scope=result.scope,
            confidence=(reward + 1.0) / 2.0,
            utility=clamp(0.5 + 0.5 * abs(reward)),
            source=source,
            links=set(credited),
            long_term=True,
        )

    def _execute(self, action: CognitiveAction, context: DecisionContext) -> ActionEvent:
        if action.kind is ActionKind.RETRIEVE:
            query = str(action.arguments.get("query") or f"{context.goal} {context.observation}")
            scope = str(action.arguments.get("scope") or context.scope)
            if scope != context.scope:
                raise PermissionError(f"retrieval scope {scope!r} differs from cycle scope {context.scope!r}")
            limit = int(action.arguments.get("limit", 5))
            if limit < 1:
                raise ValueError("retrieval limit must be positive")
            context.retrieved = self.memory.retrieve(query, scope=scope, limit=limit)
            memory_ids = tuple(dict.fromkeys(item.id for item in context.retrieved))
            context.working["retrieved_memory_ids"] = list(memory_ids)
            return ActionEvent(action, f"retrieved {len(memory_ids)} memories", memory_ids)

        if action.kind is ActionKind.REASON:
            if self.reasoner is None:
                raise RuntimeError("reasoning action requested without a reasoner")
            prompt = str(action.arguments.get("prompt") or context.goal)
            output = self.reasoner(prompt, context)
            context.working.setdefault("thoughts", []).append(output)
            return ActionEvent(action, output)

        if action.kind is ActionKind.LEARN:
            memory = self._learn(action, context)
            context.working["learned_memory_id"] = memory.id
            return ActionEvent(action, memory.id, (memory.id,))

        if action.kind is ActionKind.GROUND:
            try:
                handler = self.grounding[action.name]
            except KeyError as exc:
                raise KeyError(f"no grounding handler registered for {action.name!r}") from exc
            output = handler(dict(action.arguments), context)
            context.working["grounding_output"] = output
            return ActionEvent(action, output)

        raise ValueError(f"unsupported action kind: {action.kind!r}")

    def _learn(self, action: CognitiveAction, context: DecisionContext) -> MemoryItem:
        try:
            memory_kind = LongTermMemoryKind(str(action.arguments["memory_kind"]))
            topic = str(action.arguments["topic"])
            content = str(action.arguments["content"])
        except (KeyError, ValueError) as exc:
            raise ValueError("learning action requires valid memory_kind, topic, and content") from exc
        if memory_kind is LongTermMemoryKind.PROCEDURAL and not self.allow_procedural_learning:
            raise PermissionError("procedural learning is disabled; enable it explicitly for trusted code paths")
        metadata = {
            key: value
            for key, value in action.arguments.items()
            if key not in {"memory_kind", "topic", "content"}
        }
        scope = str(metadata.pop("scope", context.scope))
        if scope != context.scope:
            raise PermissionError(f"learning scope {scope!r} differs from cycle scope {context.scope!r}")
        confidence = float(metadata.pop("confidence", 0.5))
        utility = float(metadata.pop("utility", 0.5))
        source = str(metadata.pop("source", "coala-learning"))
        key = metadata.pop("key", None)
        value = metadata.pop("value", None)
        raw_links = metadata.pop("links", set())
        links = {str(raw_links)} if isinstance(raw_links, str) else {str(value) for value in raw_links}
        if metadata:
            raise ValueError(f"unsupported learning metadata: {', '.join(sorted(metadata))}")
        return self.memory.remember(
            topic,
            content,
            kind=memory_kind.value,
            scope=scope,
            confidence=confidence,
            utility=utility,
            source=source,
            key=key,
            value=value,
            links=links,
            long_term=True,
        )

    def _memory_by_id(self, memory_id: str) -> MemoryItem:
        found: MemoryItem | None = None
        for store in (self.memory.working_memory, self.memory.long_term_memory, self.memory.shared_cache):
            candidate = next((item for item in store.values() if item.id == memory_id), None)
            if candidate is None:
                continue
            if found is not None and found is not candidate:
                raise ValueError(f"memory id {memory_id!r} refers to multiple objects")
            found = candidate
        if found is None:
            raise KeyError(f"unknown credited memory id: {memory_id}")
        return found
