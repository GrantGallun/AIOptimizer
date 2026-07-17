# Pre-Registration v19: Does reorganizing context every turn erase the provider's cache benefit?

Registered: 2026-07-17 by Fable (Fable 5) — **before any v19 run, dev or hidden.**

## Origin — the deferred tension from v17 and v18

v17 flagged and explicitly deferred this: "reorganizing context every turn mutates the cacheable
prefix, and on a provider that bills cache hits cheaper that is a real cost this design cannot
see. That tension needs its own pre-registration." v18 then proved attention wins on tokens and
ties or beats untreated baselines on quality — but token count is not compute cost when a
provider (or a self-hosted backend) charges less, or takes less wall time, for prompt tokens it
has already seen in an identical prefix. This closes the deferred tension.

**A same-session $0 probe already confirms the mechanism exists locally.** Three sequential
`/api/generate` calls to llama3.2:3b on this box, `keep_alive` held, `num_predict=1`:
a cold ~2037-token prompt costs 184ms of `prompt_eval_duration`; a second call whose prompt is
that same prompt **extended** (exact-prefix continuation) costs 10.6ms (17x faster); a third call
of similar length but a **different** prefix costs 128.5ms (full reprocess, no speedup). Ollama's
`prompt_eval_count` does not drop on a cache hit (it reports total prompt tokens regardless) —
`prompt_eval_duration` is the metric that carries the signal. This licenses a mechanistic,
non-generative measurement: no answer-quality scoring is needed, only prefill timing.

## Hypothesis

**H-v19**: Across a multi-turn conversation, the attention arm's summed prefill time
(`prompt_eval_duration`) is **not smaller** than the raw/chronological arm's, despite sending
far fewer prompt tokens per turn — because attention's reorganization breaks the KV-cache prefix
almost every turn, while raw's strictly-appended history keeps extending a cache-hit prefix.

This is a claim about **compute time**, which is the $0-measurable proxy for what a cloud
provider's prompt-cache discount monetizes (Anthropic `cache_read_input_tokens` pricing, OpenAI's
automatic prefix cache) — the underlying mechanism (longest-common-prefix KV reuse) is the same
family of optimization in both the local llama.cpp server and hosted inference stacks. Translating
local-Ollama milliseconds into a specific cloud provider's dollar figure is explicitly out of
scope (see Scope).

## Design

Reuses the frozen v16-v18 pieces without mutating them: `generate_volume_cases` (deterministic
synthetic multi-turn histories), `ConversationCompiler.compile/organize/render_organized`, and
`compiler.render_raw`. New file: `experiments/brain_runtime/run_v19_cache_tension.py`.

For each case (`n_messages=320`), treat the message list as a **live growing conversation** and
checkpoint every 40 messages (8 checkpoints: after messages 40, 80, ..., 320). At each checkpoint
`k`, build two prompts from `messages[:k]`:

1. **raw** — `render_plain(messages[:k])`: chronological `role: content` lines, no compiler. This
   is a strict textual extension of the raw prompt at checkpoint `k-1` (append-only) — the
   growing-prefix condition a real cache is built for.
2. **attention** — `compiler.render_organized(compiler.organize(compiler.compile(messages[:k]),
   query=case["query"]), budget_chars=2600)`: the product's actual per-turn output. Reorganizes
   by relevance/pressure every time the input changes, so the byte prefix is **not** guaranteed
   to match checkpoint `k-1`'s attention prompt.

Each arm's 8 checkpoint prompts are sent **sequentially and uninterrupted** (all of raw's 8 calls
back-to-back, then all of attention's 8 calls back-to-back — never interleaved, so one arm's
calls cannot evict the other's KV-cache slot) to the same loaded `qwen3:8b` instance via
`OllamaClient.generate_with_metrics`, `temperature=0`, `num_predict=1`, `num_ctx=16384`,
`keep_alive="60m"`. Only `prompt_eval_duration_ns` is recorded per call (requires adding that
field to `Generation` in `experiments/local_worker/ollama_client.py` — mechanical, `prompt_eval_duration`
is already in Ollama's response payload alongside the fields already captured).

Per case: `raw_total_ms = sum(prompt_eval_duration_ns) / 1e6` across its 8 raw checkpoints;
`attention_total_ms` likewise for attention. Also record each arm's summed `prompt_eval_count`
(the token-count side, expected to replicate v16-v18's attention-uses-fewer-tokens result as a
sanity check, not a new claim).

Dev seed 20260711 (`n_cases=5`, 8 checkpoints, 2 arms = 80 calls). Hidden seeds v19 =
**(1429, 1433, 1439)**, fresh, read once, `n_cases=5` each (15 cases pooled, 120 calls each stage).

## Pre-committed predictions

- **P1**: raw's per-checkpoint `prompt_eval_duration` drops sharply after checkpoint 1 (cache hit
  on the extended prefix) — median checkpoint-2..8 duration for raw should be a small fraction
  (I predict <30%) of its own checkpoint-1 (cold) duration.
- **P2**: attention's per-checkpoint `prompt_eval_duration` does **not** show that drop — I
  predict its checkpoint-2..8 median stays above 70% of its own checkpoint-1 duration, because
  reorganization changes the prefix most turns.
- **P3 (the outcome that actually answers H-v19)**: `attention_total_ms` is not reliably below
  `raw_total_ms` per case, even though `attention` sends far fewer cumulative prompt tokens.

## Gate (committed before any run)

Per-case paired comparison, pooled across hidden seeds (n=15 cases):

- **H-v19 CONFIRMED** iff in **≥ 12/15 cases (80%)** `attention_total_ms > raw_total_ms` (sign
  test on the paired difference) — i.e., the token-cheaper arm is the time-cheaper arm's
  compute-cost inferior in a clear majority of cases, given a same-provider apples-to-apples
  local backend.
- **H-v19 REFUTED** iff in **≥ 12/15 cases** `attention_total_ms < raw_total_ms` — the reorg cost
  does not materialize; the cache tension is not practically significant on this backend/model.
- Otherwise: **Inconclusive** (no clear majority either way) — reported as such, not forced.
- Token-count sanity (not a gate, replication check only): attention's summed
  `prompt_eval_count` across 8 checkpoints should be materially below raw's, replicating v16-v18.

## Dev sanity (seed 20260711 — STOP on any failure, no hidden read)

1. Mechanical: raw checkpoint `k`'s prompt text must literally start with checkpoint `k-1`'s
   prompt text as a substring prefix, for all `k` (geometry check — if this fails, `render_plain`
   is not append-only and the whole premise is void).
2. Model: P1 and P2 both hold in direction (raw checkpoint-2..8 median duration materially lower
   than checkpoint-1; attention's does not fall by a comparable margin). If dev does not
   reproduce the mechanism, **STOP** — the local-cache probe result does not transfer to the
   harness's actual prompt shapes, and no hidden read is permitted until that is explained.

## Amendment v19.1 (2026-07-17, at dev stage, BEFORE any hidden read — dev-sanity check corrected)

Dev exposed a flawed operationalization in the committed P1/P2 dev-sanity check, not a failure
of the mechanism. P1/P2 compared each arm's checkpoint-1 (cold) duration to its checkpoint-2..N
median, expecting raw's to *drop sharply*. But this harness's checkpoints add a **fixed increment**
(40 messages, ~750-800 tokens) each time, not a fixed-size window over a growing backlog — so a
working cache makes each checkpoint's duration track the roughly-constant *increment* size, not
shrink toward zero. Actual dev data (case `v16-20260711-320-00`): raw's duration is ~flat across
all 7 checkpoints (125ms → 129ms) **while its token count grows 7x (786 → 5535)** — the P1/P2
ratio test scored this as "no drop" (STOP), when it is in fact the clearest possible cache
signature: linear extrapolation from checkpoint-1's cold ms/token predicts checkpoint-7 should
cost ~882ms if uncached; it actually cost 129ms (**ratio 0.15**). Identical pattern, ratio
0.12-0.16, in all 5 dev cases — no case is an outlier.

**Corrected dev-sanity check (replaces P1/P2), evaluated on the same dev run, no rerun needed:**
for each case, `cache_active_ratio = final_checkpoint_actual_ms / (cold_ms_per_token *
final_checkpoint_prompt_tokens)`. PASS iff median `cache_active_ratio < 0.5` across dev cases
(cache demonstrably reusing more than half the would-be recompute cost). Dev result: median
ratio **0.15** (n=5, range 0.12-0.16) — **PASS**, clearly.

**The committed hidden-stage gate is unchanged** (H-v19 sign test on paired
`attention_total_ms` vs `raw_total_ms`, ≥80% either direction) — only the dev diagnostic was
wrong, not the confirmatory design. One observation from dev is added as a **secondary, reported
(non-gating) metric**, because it is the more informative number the corrected dev-sanity check
surfaced: dev shows attention faster in 5/5 cases (so dev alone would already REFUTE H-v19 as
strictly written), but only by ~8-11% wall-clock despite sending ~4x fewer tokens. So alongside
the binary gate, hidden also reports a **dividend-compression ratio**:
`(1 - attention_ms/raw_ms) / (1 - attention_tokens/raw_tokens)` per case — how much of the
token-count-implied time savings the cache actually erases. A ratio near 1 means the token
savings and time savings track each other (no erosion); near 0 means caching erases nearly all
of the time benefit despite the token win holding. This was always the substance the deferred
v17 tension was worried about; the binary CONFIRMED/REFUTED gate alone would flatten it.

## Scope (deliberate non-claims)

- **Local Ollama/llama.cpp only.** This measures the KV-cache-reuse mechanism as it exists on
  self-hosted `qwen3:8b`. It does **not** measure Anthropic's or OpenAI's actual cache pricing or
  cache-hit behavior (different implementations, different eligibility rules — e.g. Anthropic
  requires explicit `cache_control` breakpoints, OpenAI's automatic caching has its own prefix-
  length and TTL rules). A CONFIRMED H-v19 licenses "the tension is real and demonstrable
  in a KV-cache-reuse backend," not a specific dollar figure on any named cloud provider.
- **Wall-clock/compute time, not dollars.** No claim is made about actual API cost inversion
  until a provider-specific pricing model is applied to real cache-hit token counts, which is
  future work.
- Synthetic buried-fact conversations only (same generator as v16-v18) — no claim about whether
  real conversation content reorganizes as aggressively turn-to-turn as this generator's records
  do; the 2026-07-17 real-session replay (292 turns, commit 125c80e) showed the product only
  triggers reorganization on 11.5% of turns in practice, so H-v19's per-turn tension applies only
  when the product actually activates, not on every turn of a real session.
- Single model (`qwen3:8b`), single machine, single run per stage — no claim about GPU vs CPU
  backends, multi-tenant serving (`OLLAMA_NUM_PARALLEL>1`), or cloud autoscaled inference, all of
  which change cache eviction behavior.

— Fable (Fable 5), 2026-07-17
