# Pre-Registration v17: Can better-chosen context buy a SMALLER budget? (the cost claim)

Registered: 2026-07-17 by Fable (Claude Opus 4.8) — **before any v17 run, dev or hidden.**
Status: **CONFIRMED** (hidden 1301/1303/1307, read once, n=120/cell). H-v17a passed on both arms of
the committed gate: `wilson_lower(attention@1300)=0.954 > wilson_upper(raw@2600)=0.620`, and 532 vs
916 input tokens. Attention at HALF budget beats raw at FULL budget by +0.459 while spending 42%
fewer input tokens; at QUARTER budget (329 tok) it still scores 1.000. Dev sanity reproduced v16
first (raw 0.575 / attention 0.975). Full verdict + caveats: HYP-20260717-39 in the graveyard.
Honest shape: attention is FLAT at ~1.0 across a 4x budget range, i.e. this task is easy for it
(single-fact retrieval) — the result bounds retrieval-shaped work, not agent builds.

**Amendment 2026-07-17 (post-verdict audit — three corrections, verdict stands but narrower):**
(1) The gate's `AND cheaper` conjunct was true by construction (half the chars cannot cost more
tokens); only the Wilson conjunct was informative. A prereg gate arm that cannot fail is not a gate
arm. (2) Conditioning the frozen rows on `context_expected_present` shows the scorer is clean
(raw succeeded 0/167 times across v16+v17 when the fact was absent) but the raw arm is NOT
transcript truncation: both arms render compiler records with identical pinning through a greedy
first-fit packer, and raw fact-retention *rises* as budget shrinks (0.542 → 0.608 → 0.675;
29/120 same-case flips present@650/absent@2600 — impossible under tail truncation; the small fact
record is crowded out by larger records at bigger budgets). The result is a clean *ordering* A/B;
it does not measure "vs. no product." (3) Follow-up registered as the obvious v18: add true
untreated arms (plain transcript; transcript tail) before any external "vs. baseline" claim.
— Fable (Fable 5)

## Origin — a question our own results cannot currently answer

The user asked the right blunt question: does the product help, hurt, do nothing, reduce cost,
or increase cost? A descriptive post-hoc pass over the **already-read** v16 hidden results
(exploratory, NOT confirmatory — recorded here so the provenance is not laundered later) shows,
at the v16 matched budget of 2600 chars, N=320, qwen3:8b, hidden 1201/1213/1217 pooled (n=60/arm):

| N | raw success | attention success | raw prompt_tok | attention prompt_tok |
|---|---|---|---|---|
| 40 | 1.000 | 1.000 | 925 | 925 |
| 160 | 1.000 | 0.983 | 915 | 934 |
| 320 | 0.567 | 1.000 | 916 | 934 |

So at matched budget the product **does not reduce cost**: both arms spend ~925 input tokens.
It spends the *same* tokens on *better-chosen* context — worth +0.433 success above the cliff,
worth nothing below it, for ~+2% input tokens. That +2% is **inference cost only**; the
encoder/compile cost of the reorganization is NOT in it, and is still unmeasured.

This is a quality claim, not a savings claim. The savings claim is the untested dual:
**if better-chosen context wins at equal budget, does it still win at a smaller one?**

## Design

Fixed: `N_MESSAGES = 320` (above the HYP-38 cliff — the only regime where the treatment does
anything), qwen3:8b local ($0), same frozen `generate_volume_cases` generator, same deterministic
scorer, 40 cases/seed. ONLY `budget_chars` varies.

- Arms: `raw` (chronological truncation) x `attention` (reorganized), as in v16.
- Budgets: **2600** (v16 baseline), **1300** (1/2), **650** (1/4).
- Hidden seeds v17 = **(1301, 1303, 1307)**, fresh, read ONCE.

## Pre-committed prediction + gate

- **H-v17a (cost dividend)**: `attention @ 1300` beats `raw @ 2600` — i.e. the Wilson CI lower
  bound of attention-at-half-budget exceeds the Wilson CI upper bound of raw-at-full-budget,
  AND its mean `prompt_tokens` is lower. That is the only result that licenses a **cost-reduction**
  claim: half the input tokens *and* better quality than the untreated baseline.
- **Null (equally reportable)**: attention degrades with budget at the same rate as raw. Then the
  honest product claim stays "same cost, better quality above the cliff" and we say **the product
  does not save money** — plainly, in the README.
- **Reverse (equally reportable)**: `attention @ 1300` falls below `raw @ 2600`. Then budget cannot
  be traded for organization at all, and the treatment is strictly a quality-at-equal-cost tool.

**Decision rule (committed):** CONFIRMED iff `wilson_lower(attention@1300) > wilson_upper(raw@2600)`
and `mean_prompt_tokens(attention@1300) < mean_prompt_tokens(raw@2600)`. Anything else is Refuted
or Partial, reported as such. `attention @ 650` is exploratory (it maps where the treatment breaks);
it carries no gate and cannot rescue a failed H-v17a.

**Dev sanity first (seed 20260711, no hidden reads):** at budget 2600 / N=320 the harness must
reproduce v16's shape — attention ~1.0 and raw materially below it. If dev does not reproduce,
**STOP**: the harness drifted and no hidden read is permitted until that is explained.

## What this deliberately does not claim

- Local qwen3:8b only. v16's frontier replication was quality-only; nothing here transfers a cost
  claim to a frontier provider.
- The encoder/compile cost is out of scope and remains unmeasured — so even a CONFIRMED H-v17a
  bounds *input-token* savings, not total cost of ownership.
- Provider prompt-cache interaction is out of scope: reorganizing context every turn mutates the
  cacheable prefix, and on a provider that bills cache hits cheaper that is a real cost this
  design cannot see. That tension needs its own pre-registration.

## Separate and prior: production treatment integrity (not a hypothesis — a bug)

The live hook ledger (`.aioptimizer/codex_hook_ledger.jsonl`, 60 real Codex turns) shows that of
**39 real turns carrying the exact condition HYP-38 proved we help (>6k chars), the product treated
4 (10.3%)** — and the last `attention` injection was **2026-07-12**. 13 turns errored (12 `URLError`,
1 15s `TimeoutError`: the sidecar was unreachable and the hook failed **silent**). 27 routed `raw`
with no `route_reason` recorded, so *why* they bypassed is unrecoverable from that data.

This is not a research question, it is a defect: v15's Amendment 15.2 demanded treatment integrity
of the *experiment* and nobody ever applied that standard to *production*. **No usefulness claim may
be made from production data until treatment-applied rate on the pressure subset is ~1.0 and a
sidecar failure is loud rather than silent.** v17 is a bench experiment and is not blocked by this.

— Fable (Claude Opus 4.8), 2026-07-17
