# AIOptimizer — a local LLM gateway with receipts

Every LLM gateway reports the cost it saved. **None of them prove the quality they kept.**
AIOptimizer is a thin, dependency-free proxy for local models (Ollama) whose every optimization
ships behind a pre-registered quality gate — and whose shadow-A/B **receipts** report tells you,
with confidence intervals, what an optimization saved *and* what it cost:

```
receipts: 494/510 requests optimized, 2.1M request chars saved;
quality parity 96.4% (95% CI 91.2%–98.7%, n=55 shadow samples)
```

## Quickstart

```bash
python -m gateway --port 8800            # fronts Ollama at 127.0.0.1:11434
set AIOPT_GATEWAY=http://127.0.0.1:8800  # any OllamaClient / OpenAI-style client routes through
python -m gateway.report results/gateway/ledger.jsonl
```

Proxies `/v1/chat/completions`, `/api/generate`, `/api/chat`, and GET utility endpoints.

## The pipeline — every stage backed by a pre-registered experiment

| stage | mechanism | evidence (all dev/hidden-split, Wilson-gated) |
|---|---|---|
| exact cache | hash replay; **bypasses sampled (temp>0) requests** | plumbing + a correctness bug caught by dogfooding |
| context compaction | encoder top-k selection of oversized contexts | HYP-20/21: selection holds at oracle while dumping degrades 0.83→0.33 — replicated on 1,788 real PyTorch functions |
| attention context (opt-in) | conversation history reorganized into query-ranked semantic clusters, privacy-filtered | HYP-33: **+0.650** vs chronology at matched budget (hidden); HYP-34: honestly scoped — never lost a case (120/120) but the advantage is small when budget pressure is mild |
| constrained decoding | per-kind JSON schema via `format=` | HYP-25: malformed actions 0.055→**0.000**, completion +12pts |
| deterministic query rewrite | rebuild retrieval queries from task state | HYP-31: one line carries ~87% of a full agent-kernel's accuracy edge (0.796→0.974 hidden) |
| receipts (always on) | deterministic shadow sampling + encoder judge + Wilson CIs | the methodology itself, productized |
| serving layer | `keep_alive` residency (+ planned residency-aware routing) | measured: 32.3% of our own GPU time was silent weight reloading |

## What the research actually found (honest version)

34 pre-registered hypotheses (`memory/hypothesis-graveyard.md`), negative results kept:

- **Deterministic structure is a guarantees layer, not a capability multiplier.** A strong prompt
  matches forced invariants on capable models (HYP-27); invariants actively hurt weak models by
  burning their action budget (HYP-29). The accuracy wins live at specific decision boundaries —
  query construction (HYP-31), context selection/organization (HYP-20/21/33) — not in constraining
  the model's reasoning.
- **Fix the input, don't aggregate the outputs**: self-consistency voting failed (errors are
  systematic, not stochastic — HYP-32), but sample *disagreement* predicts wrongness with perfect
  precision on dev, seeding an escalate-on-disagreement router.
- Full arc, verdicts, and every dated amendment: [`FINDINGS.md`](FINDINGS.md).

## How it was built

Two agents over a shared coherent cache (`agent_bus/`): **Fable** (Claude) owns hypotheses, gates,
and verdicts; **Codex** (GPT) implements to spec, reviewed and retired in order on a scoreboard.
Discipline throughout: pre-registration before confirmatory runs, dev-sanity before hidden reads,
fresh hidden seeds per experiment (`experiments/brain_runtime/stats.py`), and a self-audit that
caught and corrected its own confounds mid-project.

## Limitations

Local models only (qwen3 8B/14B, llama3.2 3B) — a frontier-model replication is owed. Task
families are mostly synthetic (real-code and real-transcript replications exist for two findings).
Receipts quality-parity uses an encoder judge (all-MiniLM), not human evaluation.

## Repo map

`gateway/` proxy + middlewares + receipts · `experiments/brain_runtime/` evals + pre-registrations ·
`memory/` graveyard + ideas · `agent_bus/` the two-agent coordination substrate · `intel/` external
signal feeds · [`PRODUCT.md`](PRODUCT.md) roadmap · [`FINDINGS.md`](FINDINGS.md) the evidence.
