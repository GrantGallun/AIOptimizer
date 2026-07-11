# LLM-improvement landscape (through the deterministic-kernel thesis)

Thesis (PROJECT_PLAN §00): *LLMs are probabilistic semantic coprocessors; AIOptimizer is the
deterministic kernel that makes their work reliable.* Strategy: treat mature systems as
**components AND baselines**, reproduce their functional invariant behind a typed contract, keep
only what evidence says our determinism improves. This maps the field so we don't miss a category.

## The map — where each category replaces LLM glue with a deterministic mechanism

| LLM-glue decision | Deterministic-kernel technique | Projects | We have it? |
|---|---|---|---|
| "output valid JSON/schema" | **constrained decoding** (logit masking → only valid tokens) | Outlines, Guidance, XGrammar, Instructor, LMQL, SGLang | ❌ **biggest miss** |
| "here's a hand-tuned prompt" | **prompt/program compilation** (optimize toward a metric) | DSPy, TextGrad | ❌ miss |
| "the LLM says it succeeded" | **verifiers / self-correction** | Reflexion, Self-Refine, generator-verifier | partial (cross-check) |
| "reason harder" | **test-time compute / scaffolds** | Tree-of-Thoughts, ReAct, self-consistency, o1-search | ❌ |
| "search memory if useful" | **mandatory retrieval policy** | (our m0022 forced-retrieval invariant) | ✅ |
| "remember this" | **typed/versioned/scoped memory** | Letta, Mem0, Zep, Graphiti | ✅ (governed) |
| "which model runs" | **routing / cascades** | RouteLLM, vLLM semantic router | ✅ (multicore) |
| "durable workflow state" | **checkpoints / replay** | LangGraph, Burr, AutoGen | partial (board/ROB) |
| "reuse similar answers" | **semantic caching** | GPTCache, SAFE-CACHE | ❌ |
| "is it good?" | **eval / observability** | Ragas, DeepEval, LangFuse, promptfoo | homegrown (prereg + gates) |

## Priorities (thesis-ranked)

1. **Constrained decoding** — the purest token-level determinism; almost nobody frames it as part of
   a unified kernel. Natural first target: a *deterministic action-validity layer* for the CoALA
   controller (Codex's `coala_ollama` emits JSON actions that currently depend on the model
   formatting them right).
2. **DSPy / prompt compilation** — replace hand-prompts with compiled+optimized ones against our gates.
3. **Verifiers / Reflexion** — the "did it succeed" decision. NOTE: Reflexion (persisting
   self-reflective lessons to learn from mistakes) is *literally* the reasoning-memory experiment
   queued for Codex (t0014) — ground it in that literature.

## Scraped repo findings (Haiku scrape, 2026-07-10)

| name | stars | core technique | notable claim | kernel-fit |
|---|---|---|---|---|
| Outlines | 14.5k | token-level constraints (JSON/Pydantic/regex/CFG) | eliminates post-gen parse failures | **High** |
| Guidance | 21.7k | CFG + constrained decoding | quality up, latency/cost down vs prompting | **High** |
| XGrammar | 1.8k | constrained decoding engine (CFG) | near-zero overhead; default backend for vLLM/SGLang/TensorRT | **High** |
| Instructor | 13.5k | Pydantic validation + auto-retry | 100k+ devs; multi-provider | Med (validate-after, not constrain-at-decode) |
| DSPy | 36k | compile/optimize prompts+weights to a metric | comparable to/exceeds RL; multi-stage | **Med-High** |
| TextGrad | 3.6k | "textual gradient" backprop through LLM feedback | Nature 2025 | Med |
| Reflexion | 3.2k | verbal RL: persist self-reflective lessons | NeurIPS 2023; reasoning/coding/decisions | Med (= our t0014) |
| RouteLLM | 5.2k | trained routers on prompt features | ~85% cost cut at 95% GPT-4 quality | Med (we have routing) |
| GPTCache | 8.1k | embedding-similarity cache | 10x cost / 100x speed | Low |
| Tree of Thought | 6k | BFS + thought eval | 69% on Game-of-24 | Med |

**Verdict:** the **constrained-decoding cluster (Outlines / Guidance / XGrammar)** is the clear
highest-value gap — mature (14–22k stars), purest token-level determinism, and directly applicable.
**DSPy** is the optimization layer above it. **Reflexion** is the named ancestor of our queued
reasoning experiment (t0014).

## Context optimization ("which parts of context matter" — a whole subfield)

Motivation: **lost-in-the-middle** / context rot — models degrade as context grows. Two importance
signals dominate: **information/entropy** and **attention**. Scraped repos (Haiku, 2026-07-10):

| repo | stars | technique | kernel-fit |
|---|---|---|---|
| microsoft/LLMLingua | 6.4k | compact LM scores + drops non-essential tokens | High |
| **LLMLingua-2** | (in LLMLingua) | **BERT-scale ENCODER trained on GPT-4-distilled labels → token importance**; task-agnostic, 3–6x | **High** |
| mit-han-lab/streaming-llm | 7.2k | attention sinks: keep initial + recent, discard middle | High |
| FMInference/H2O | 526 | heavy-hitter: recency + salience dual-signal token ranking | High |
| FasterDecoding/SnapKV | 323 | snapshot selection of important KV pairs | High |
| liyucheng09/Selective_Context | 421 | self-information (entropy) ranking, no external model | High |
| parthsarthi03/raptor | 1.7k | recursive summary **tree** (hierarchical retrieval) | Med |

**Key validation:** LLMLingua-2 is *literally an encoder that scores which context matters* — the
user's "attention-encoder" instinct is SOTA, and it sidesteps the FlashAttention-hides-attention
problem that hobbles H2O/SnapKV in production. Most reusable: **LLMLingua-2** (encoder token-scoring
pipeline), **H2O** (recency+salience formulation), **Selective_Context** (entropy, zero extra model).
Our angle: apply encoder importance-scoring at the *memory-chunk/fact* level as a deterministic
selection primitive integrated with governed retrieval — the agent-memory version, not just inference.

## Runnable next experiment (deterministic-kernel test)

**Constrained decoding as an action-validity layer for the CoALA controller.** Codex's
`coala_ollama` emits JSON `CognitiveAction`s that currently depend on the model formatting them —
the same class of failure as m0022 (the model didn't spontaneously do the right structural thing).
Ollama supports structured outputs (`format=<json schema>`). Hypothesis: constraining the action
JSON to the schema eliminates malformed-action failures (a *structural guarantee*) vs. free-form
parsing. Arms: free-form (current) vs. constrained (schema-forced). Metric: malformed-action rate +
task completion. This is the thesis in miniature — a deterministic mechanism replacing "hope it's
valid." Fable-designable, Codex-runnable on the existing substrate.


## LLM gateways / usage optimizers (added 2026-07-11 — the category the product lives in)

The 2026 ecosystem converged on **cache / route / compress** behind an OpenAI-compatible proxy:

| product | model | strengths | gap vs us |
|---|---|---|---|
| LiteLLM | OSS proxy, 100+ providers | budgets/keys, Redis cache, ~10-20ms overhead | no quality evidence for its optimizations |
| Portkey | OSS (Apache 2.0, 2026) | semantic caching, guardrails/PII, audit | same |
| Helicone | OSS observability proxy | best request/cost observability UI | observes cost, not quality deltas |
| OpenRouter | SaaS marketplace | 200+ models, auto-fallback | hosted-only; no local models |
| Cloudflare AI Gateway | edge proxy | mature caching | same gap |

Reported industry wins: prompt caching 45-80% cost cut; routing 60-75%; semantic-cache+routing 47%
in production. **The uniform gap: every gateway reports cost saved; none proves quality preserved.**
Our differentiator (PRODUCT.md): ship each optimization behind a pre-registered quality gate and an
always-on shadow-A/B "receipts" report — the optimizer you can *verify*.

## Usage-based retention / "heat" compaction (added 2026-07-11, post-v11)

Three literature levels validate usage-heat retention — all using OBSERVED usage, never inferred:
- **Token/KV**: H2O (NeurIPS'23) evicts by accumulated ATTENTION (up to 29x throughput);
  Scissorhands' "persistence of importance" = the heat hypothesis, empirically confirmed.
- **Cognitive**: ACT-R base-level activation (Anderson) — retention strength = log-sum of actual
  past uses with power-law decay; ACT-R-inspired LLM memory (HAI 2025) applies it directly.
- **Agent memory (2025-26)**: surveys taxonomize time-/frequency-/importance-driven forgetting
  (LFU on retrieval behavior); AgeMem (2026) learns the policy via RL.
**Lesson vs our v11 failures**: every working system COUNTS usage (attention, retrievals, hits);
our five failed designs INFERRED it from similarity. v12: ACT-R activation over the runtime's
real `uses`/`last_accessed` counters at compaction time.

## "Context rot" framing (Prime Intellect talk, via user, 2026-07-11)

Claim: GPT-5.5 retrieval drops 80% (256k) -> 36% (1M) — "the model accepts the context, it just
can't reason across it"; bigger windows won't save agents; their fix = continual learning +
training on your own traces + real environments (weight-level).

**Where our evidence agrees (measured locally, small scale):** context rot is real and shows up
long before 256k — dump_all 0.83->0.33 on 1,788 real torch functions (HYP-21); raw chronology
0.35 vs organized 1.00 at matched budget (HYP-33). We never needed a million tokens to see it.

**Where we diverge:** their fix is TRAINING (weights); ours is deterministic context engineering
at inference — encoder selection (HYP-20/21), query-ranked organization (HYP-33), ACT-R usage
compaction (v12, running) — local, cheap, receipts-gated, and shippable today as gateway
middlewares. The approaches compose: engineered context is also better TRACE data if you later
train. Honest scope note: our measurements are 8B models at 10^3-10^4-char contexts; the 256k->1M
frontier regime is extrapolation, not our data.
