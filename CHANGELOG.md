# Changelog

## v0.1.1 — first installable release (pending)

- **Packaging**: `pip install aioptimizer` — stdlib-only core (proxy, exact cache, receipts
  ledger, requirement contracts); `aioptimizer[encoder]` extra enables the encoder-backed stages
  (context compaction, attention organization, semantic cache, shadow quality judge).
- **Restructure**: `gateway/` → `aioptimizer/` with a public `aioptimizer.encoder` surface
  (single embedding/cosine implementation) and `aioptimizer.stats`; research code now depends on
  the product, not vice versa. `gateway/` remains as a deprecation shim.
- Console commands: `aioptimizer` (serve), `aioptimizer-report` (receipts).

## v0.1 — the verified optimizer (tagged 2026-07-11)

- OpenAI/Anthropic/Ollama proxy with streaming (SSE/NDJSON), token-usage normalization,
  bounded upstream timeouts, `/health` + `/status`, persistent JSON config.
- Receipts: deterministic shadow A/B sampling, encoder quality judge, Wilson-CI reports;
  per-request requirement contracts (private envelope, terms never logged, optimization
  pass-delta measured).
- Evidence-gated middleware pipeline (36 pre-registered experiments, hidden splits, negative
  results kept): encoder context compaction; adaptive attention context with relevance-gated
  vague-query routing and privacy filtering; exact cache with sampled-request bypass; ACT-R
  usage-heat memory compaction; `keep_alive` serving fix (32.3% measured reload overhead).
- Flagship finding, cross-vendor: deterministic context compilation beats LLM-rewrite
  preprocessing (attention 1.000 vs rewrite 0.900 hidden at ~0.01% cost); raw 0.350 vs
  attention 1.000 replicated identically on GPT and Claude Sonnet in per-arm isolated sessions.
