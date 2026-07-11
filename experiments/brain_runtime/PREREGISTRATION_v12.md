# Pre-Registration v12: Observed-usage activation (ACT-R) vs recency at compaction time

Registered: 2026-07-11 by Fable (Claude Fable 5)
Status: Registered (no v12 run yet)

## Question (the surviving form of the user's heat idea, literature-grounded)

At compaction time the future query is unknown. Does ACT-R-style base-level activation over
OBSERVED usage (the runtime's real `uses`/`last_accessed` counters) predict which memories will be
needed better than the naive baselines? v11 showed INFERRED usage (lexical/embedding) fails; the
H2O / Scissorhands / ACT-R literature says COUNTED usage works. This tests it on our substrate.

## Protocol (qwen3:8b; model calls only in the test phase)

1. **Learn**: 30 verified operator rules into `PersistentBrainRuntime` (standard protocol).
2. **Usage** (model-free): a seeded retrieval sequence exercises a HOT subset (8 rules × 4
   retrievals each); retrievals increment `uses`/`last_accessed`. Cold rules: 0–1 touches.
3. **Compaction under pressure**: the store is compacted to K=12 items by the arm's policy.
4. **Test** (model): 24 problems per seed — 16 over hot rules, 8 over cold (mix frozen, unknown
   at compaction time). CoALA answer cycle as in prior evals; recurrence accuracy is the metric.

## Arms (compaction policies; pure functions, t0034)

- `newest`: keep the 12 most recently CREATED items (what a naive ring buffer does).
- `random_k`: keep 12 at random (seeded control).
- `actr`: keep top-12 by base-level activation ln(uses + 1) with recency decay on `last_accessed`
  (exact formula frozen in t0034's spec; parameters not tuned after dev).

## Gate (hidden = FRESH_HIDDEN_SEEDS["v12"] = (809, 811, 821))

- **Confirmed** iff `actr` − `newest` overall test accuracy ≥ 0.20 AND Wilson-significant AND
  `actr` ≥ `random_k` significantly (else the win is just "any non-pathological policy").
- Dev stop-rules (seed 20260711): `newest` must genuinely fail on hot problems (< 0.60) — the
  pressure must actually evict hot rules under newest-first — and not all arms at ceiling/floor.
- Null equally reportable: if usage doesn't predict future need on this task, the heat idea is
  scoped to workloads with usage-future correlation (report the hot/cold split either way).

— Fable (Claude Fable 5), 2026-07-11

---

## Dev design notes (2026-07-11, render-only, dated)

1. **Usage = verified application events** (direct counter increments), not retrieval events:
   jaccard retrieval surfaces the same 3 items for every "operator:<name> rule" query (shared
   tokens only) and the runtime's use_bonus then amplifies the winners — a rich-get-richer
   feedback loop. Recorded as a substrate finding (consistent with HYP-23's jaccard-at-scale
   failure); worth its own fix later.
2. **Seed collision caught**: the hot-subset sampler and keep_random consumed the same Mersenne
   stream, making the random control track the treatment (random "kept" exactly the 8 hot rules).
   Policies now get a decorrelated seed.
Post-fix structural separation (render-only): newest 0.50 / random 0.62 / actr 1.00 hot survival;
actr cold survival 0.12 (selective). Model dev run next. — Fable (Claude Fable 5)
