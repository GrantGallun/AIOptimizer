# AIOptimizer Gateway — the optimizer with receipts

Owner: Fable (product direction set by the user, 2026-07-11: "I want a tool that I KNOW optimizes
my AI usage"). This document is the build spec; research findings back each stage.

## The gap (LANDSCAPE, gateway category — added 2026-07-11)

The 2026 gateway ecosystem (LiteLLM, Portkey, Helicone, OpenRouter, Cloudflare AI Gateway) has
converged on **cache / route / compress**, self-reporting 45–80% cost cuts. Every one of them
reports *cost saved*; **none proves *quality preserved***. They optimize on faith. Our
differentiator is the project's whole methodology productized: **every optimization ships with a
measured quality delta (shadow A/B + Wilson CIs), not just a token count.**

## Form factor

A thin, dependency-light local proxy exposing an OpenAI-compatible `POST /v1/chat/completions`,
fronting Ollama first (hosted providers later). Point any client (scripts, agents, IDE tools) at
`localhost:<port>` and usage is optimized + measured transparently. Stdlib-only where possible
(house style), JSONL ledger, no external services.

## Pipeline (each stage backed by evidence, or it doesn't ship)

| stage | mechanism | evidence |
|---|---|---|
| 1. exact cache | hash(prompt+params) → replay | plumbing (no gate needed) |
| 2. semantic cache | encoder-similarity hit (all-MiniLM, same encoder as HYP-20) | gate before default-on: hit-quality ≥ exact-answer parity on dev/hidden |
| 3. context compaction | `ContextCompactor.select/compact` on oversized contexts | HYP-20/21 (+ torch external validity): selection ≥ dumping, at oracle on real code |
| 4. model routing | small-vs-large by task class | gate: quality parity on routed class (prereg required) |
| 5. constrained decoding | inject `format=<schema>` when caller declares a schema | HYP-25: malformed 0.055→0.000, completion +12pts |
| 6. deterministic query rewrite (agent mode) | rebuild retrieval queries from task state | v7 (pending hidden): dev 0.765→0.980 |
| 7. **receipts** | shadow A/B: sample N% of requests, run optimized AND raw, encoder-judge the delta; ledger + report with Wilson CIs | the project's methodology, productized |

Stages are independent middlewares; each is OFF until its gate passes. The receipts stage is
always on — it is the product.

## What we deliberately reuse vs build

- Reuse: provider-native prompt caching, Ollama's OpenAI-compatible endpoint; LiteLLM remains the
  escape hatch for multi-provider routing if this outgrows the thin proxy.
- Build: the middleware pipeline, the receipts ledger/report, and the three research-backed stages
  (3/5/6) no gateway has.

## Milestones

- M1 (scaffold, Codex t0024): proxy passthrough + middleware hook interface + JSONL ledger + mock-
  upstream tests. No optimization yet — measurement first.
- M2 (Fable): receipts harness (shadow A/B + encoder judge + report). "It optimizes" becomes a
  dashboard fact.
- M3: stage 1–3 middlewares (exact cache, semantic cache behind its gate, ContextCompactor).
- M4: stages 4–6 behind their gates; first end-to-end report on the user's real usage.

— Fable (Claude Fable 5), 2026-07-11
