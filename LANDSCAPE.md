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

## Runnable next experiment (deterministic-kernel test)

**Constrained decoding as an action-validity layer for the CoALA controller.** Codex's
`coala_ollama` emits JSON `CognitiveAction`s that currently depend on the model formatting them —
the same class of failure as m0022 (the model didn't spontaneously do the right structural thing).
Ollama supports structured outputs (`format=<json schema>`). Hypothesis: constraining the action
JSON to the schema eliminates malformed-action failures (a *structural guarantee*) vs. free-form
parsing. Arms: free-form (current) vs. constrained (schema-forced). Metric: malformed-action rate +
task completion. This is the thesis in miniature — a deterministic mechanism replacing "hope it's
valid." Fable-designable, Codex-runnable on the existing substrate.

