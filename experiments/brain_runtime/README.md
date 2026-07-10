# Brain Runtime v0

Brain Runtime v0 is a deterministic external cognition layer for AI workers. It is deliberately model-free: the first question is whether the architecture has useful mechanics before we attach it to a live model.

## Runtime Pieces

- Working memory with bounded capacity and exponential recency decay.
- Long-term memory promoted by repeated use, strong activation, or evidence score.
- Shared cache entries with scope, provenance, confidence, utility, key, and value.
- Contradiction tracking for same-topic same-key facts with different values.
- Evidence scoring from confidence, recency, and utility.
- Task hooks for unresolved hypotheses and contradiction pressure.
- Consolidation/replay that promotes useful memories and forgets weak low-utility ones.

## Cross-session snapshots

`session_runtime.py` adds a versioned persistence boundary without changing the v0 retrieval
or benchmark implementation. It preserves working, long-term, and shared-cache membership;
shared object identity; evidence and links; contradiction/task records; scope; clock; and ID
counters across processes.

```python
from experiments.brain_runtime.session_runtime import PersistentBrainRuntime

runtime = PersistentBrainRuntime()
runtime.remember("parser", "Use structured JSON parsing.", kind="procedure")
runtime.save("state/session.json")

later = PersistentBrainRuntime.load("state/session.json")
```

Snapshots declare the `brain-runtime-session-v1` schema and reject unknown versions or dangling
store references. Run `python -m unittest tests.test_session_runtime` for the persistence contract.

## CoALA decision cycles

`coala.py` supplies a model-agnostic controller following the memory/action/decision decomposition
in [Cognitive Architectures for Language Agents](https://arxiv.org/abs/2309.02427). A policy can
interleave bounded retrieval and reasoning actions before selecting one terminal learning or
grounding action. Reasoning, policy selection, and external grounding remain injected callbacks,
so the same architecture can be tested with deterministic fakes or a real worker.

Long-term writes are typed as episodic, semantic, or procedural. Procedural learning is disabled
unless explicitly enabled, retrieval cannot change the cycle's authorized scope, and feedback
credit is validated atomically before it updates utility or records a scoped episodic memory.

Run `python -m unittest tests.test_coala` for the decision-cycle contract. This is architecture
plumbing, not evidence that the controller improves model reasoning; that requires a separately
specified real-model benchmark and gate.

`coala_ollama.py` attaches that controller to the existing local Ollama worker through strict JSON
actions. It rejects unknown fields, model-supplied scope changes, unavailable grounding actions,
unbounded retrieval, and malformed learning metadata. Retrieved memories are rendered as JSON-line
data and model outputs remain subject to the controller's scope and procedural-write checks.

By default the adapter structurally requires at least one retrieval action before a terminal learn
or ground action. This is reported as `forced_retrievals` in token/latency metrics. The invariant
prevents a model from silently bypassing the memory architecture; it does not guarantee that the
model queried for or used the right memory. Run `python -m unittest tests.test_coala_ollama` for the
adapter contract.

## Backend mechanism comparison

`memory_backends.py` defines a backend-neutral record/search contract with structural scope
filtering. It includes native AIOptimizer and deterministic lexical implementations plus optional
adapters for LangGraph, Mem0, and Graphiti. Third-party packages remain reference mechanisms and
replaceable components; AIOptimizer does not depend on them for its native runtime.

`backend_comparison_v1.py` is an exploratory, model-free cross-session plumbing suite. It holds
records, scopes, queries, and scoring constant while changing the backend. LangGraph is configured
with an independently implemented deterministic signed-hashing embedder, so that baseline requires
no hosted model. Mem0 requires an explicit config file to prevent accidental hosted defaults, and
Graphiti reports unavailable until its package and graph-service environment are configured.

```powershell
python -m experiments.brain_runtime.backend_comparison_v1 `
  --backends native,lexical,langgraph,mem0,graphiti `
  --out results/brain_runtime/backend_comparison_v1.json
```

The v1 suite is exploratory rather than preregistered and does not establish model-quality or
product superiority. Once observed, its generator and result are frozen; mechanism changes require
a versioned successor.

The frozen v1 result exposed cache pollution in native retrieval: repeated use could outweigh a
better scenario-specific match. `memory_backends_v2.py` replaces that weighted blend with an
explicit priority order: authorization, relevance, contradiction validity, evidence, recency, then
frequency as a final tie-breaker. `backend_comparison_v2.py` fixes dev seeds
`11,23,37,41,59` and hidden seeds `101,103,107,109,113`; the policy is selected and committed from
dev before hidden is run. The versioned dev result is `backend_comparison_v2_dev.json`.

## Benchmark

The synthetic benchmark compares `BrainRuntime` against a `VanillaLoop` that uses append-only notes and first-match retrieval. It tests:

- stale fact resolution,
- transfer from a tested hypothesis,
- forgetting low-utility distractors,
- task spawning from contradiction pressure.

Run:

```powershell
python experiments\brain_runtime\benchmark.py --out results\brain_runtime\benchmark_v0.json
python -m unittest tests.test_brain_runtime
```

Current target is not raw model intelligence. The target is whether an external runtime can make ordinary workers less stale, less repetitive, and more evidence-sensitive than a plain context loop.

## Current Result

`results/brain_runtime/benchmark_v0.json`:

| System | Successes | Success rate |
| --- | ---: | ---: |
| Brain Runtime v0 | 4/4 | 1.00 |
| Vanilla loop | 0/4 | 0.00 |

Evidence captured by the run:

- stale fact resolution: Brain selects `pipe`; vanilla selects stale `comma`,
- transfer: Brain retrieves the structured JSON parsing lesson,
- forgetting: Brain avoids the stale scratch/comma note,
- task hook: Brain spawns a contradiction-resolution task,
- replay: consolidation promoted 2 memories and forgot 1 weak memory.

This is a synthetic first-pass benchmark, not a general intelligence claim. The next stronger test is to connect the runtime to real worker transcripts or generated project tasks where memories are not hand-shaped for the benchmark.

## Seeded Multi-Worker Benchmark

`multiworker_benchmark.py` is the next structural layer. It generates multiple
seeded cases for stale facts, cross-worker handoff, private worker facts, and
conflicting evidence. It compares three policies: no memory, append-only shared
notes, and governed Brain Runtime memory.

```powershell
python experiments\brain_runtime\multiworker_benchmark.py --out results\brain_runtime\multiworker_v1.json
```

It reports task success, stale-fact rate, privacy leaks, and provenance coverage.
This remains model-free; its purpose is to establish the memory-policy contract
before submitting the same cases to local or frontier workers.

## Local Worker Evaluation

`local_worker_eval.py` renders those frozen cases to a real Ollama worker using
three policies: no shared notes, append-only notes, and scope-aware governed memory.
It records every model response plus prompt/completion token counts and durations.

```powershell
python experiments\brain_runtime\local_worker_eval.py --model qwen3:8b --out results\brain_runtime\local_worker_v1.json
```

The evaluator does not modify cases. Once a result file exists, make a versioned
successor rather than editing `multiworker_benchmark.py` retroactively.

### First Local Result

On 2026-07-09, `qwen3:8b` ran the frozen five-seed suite (20 tasks per policy):

| Policy | Success | Privacy leaks | Stale errors | Prompt tokens | Model time |
| --- | ---: | ---: | ---: | ---: | ---: |
| No memory | 3/20 (0.15) | 4 | 4 | 1,340 | 3.628 s |
| Append-only | 14/20 (0.70) | 1 | 1 | 1,779 | 3.346 s |
| Governed memory | 11/20 (0.55) | 0 | 0 | 1,627 | 3.296 s |

`results/brain_runtime/local_worker_v1.json` contains every response and local
token/latency field. Governed memory saved 152 prompt tokens (8.5%) relative to
append-only and removed observed leaks/stale errors, but lost task success on this
first real-model run. Treat that as a refuted end-to-end advantage claim, not as a
reason to alter the completed v1 cases.
