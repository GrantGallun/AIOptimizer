# Pre-Registration: Adaptive Research Search — Stage 0 (controller only, frozen LLM)

Registered: 2026-09-15
Status: Registered (no confirmatory run yet)

Written before any harness code runs, per this repo's discipline (`CLAUDE.md`): the environments,
splits, controls, and gate below are frozen. Any change after the first dev-sanity run must be
logged as a dated amendment, not a silent edit.

## Question

Does a controller that classifies *why* a research branch has stalled (4 plateau types) and
allocates the next unit of search compute accordingly recover a hidden generating mechanism more
efficiently than fixed or naive-random search-diversity strategies, at matched compute, on a
frozen local LLM?

## Primary hypothesis (H-ARS-01)

The **adaptive controller** (arm 5) beats **all four baselines** (arms 1-4) on both:

```
P(discovery | B)   # probability the true mechanism is recovered within a fixed experiment budget B
T_discovery        # experiments needed to recover it, when recovered at all
```

on the **hidden** split of the 3 known-mechanism environment families, and does **not**
under-perform on the adversarial (no-real-structure) family relative to baselines — i.e. it must
not waste materially more budget chasing a phantom breakthrough than the simplest baseline does.

## Design

- Model: qwen3:8b via `experiments/local_worker/ollama_client.py`, **frozen** — no fine-tuning in
  Stage 0. Same model, same prompting scaffold across all five arms; only the search/branch logic
  around it differs.
- Environments (generator families, each with a `known_mechanism()` the harness can check
  exactly, not an LLM judge):
  - `regime_switch`: `y = 2x + eps` for `z < 0.7`, else `y = x^2 + 4w + eps`.
  - `hidden_interaction`: `y` depends on `x1*x2`, not a linear combination — a naive linear fit
    plateaus around a deceptively good R^2.
  - `confounding`: `z -> x` and `z -> y`; a naive agent will treat `x -> y` as causal.
  - `deceptive_correlation` (**adversarial only**): strong surface correlation, no real
    additional structure beyond noise — exists to catch controllers that hallucinate a plateau
    type and burn budget chasing a phantom mechanism.
- Compute budget `B` is identical across all 5 arms per environment instance (same number of
  allowed experiments/LLM calls).
- Arms: (1) single trajectory, (2) high-temperature resampling, (3) fixed diverse seeding,
  (4) random reseed on any plateau, (5) adaptive controller (Q/U/O/N/C priority + 4-way
  taxonomy, per `DESIGN.md`).

## Splits (frozen)

- `data/env_dev_seeds.json` — tuning only: pick lambda/mu/nu/eta and the plateau-detection
  thresholds here.
- `data/env_hidden_seeds.json` — confirmatory; generated, never inspected during tuning.
- `data/env_adversarial_seeds.json` — `deceptive_correlation` family only, stress-tests
  false-plateau detection.

## Controls

- `controller_no_option_value`: same controller with `O_b` fixed at 0 (drops the option-value
  term). Isolates whether `O_b` is doing real work or whether `Q + U` (plain UCB) already
  explains any win.
- `controller_no_taxonomy`: same priority formula, but plateau type is not classified — every
  plateau is treated as type A. Isolates whether the 4-way taxonomy adds anything over one
  reseed-on-any-plateau rule. (This is really baseline 4 restated as a controller ablation, kept
  as a named control to make the comparison explicit in the results table.)

## Decision gate (pre-committed)

Let `P_c` and `T_c` be the controller's hidden-split discovery probability and mean discovery
cost, and `P_i`/`T_i` the same for baseline `i` (`i = 1..4`). Using `gap_significant()` from
`experiments/brain_runtime/stats.py` (`z=1.96`):

H-ARS-01 is **Confirmed** only if, on hidden, all hold:

1. `gap_significant` shows `P_c > P_i` for at least 3 of the 4 baselines (a real Wilson gap, not
   noise), and `P_c` is not below any baseline's `P_i` by a significant margin.
2. Among environments where both the controller and a baseline reach discovery, `T_c <= T_i`
   (median) for the same 3-of-4 baselines.
3. `controller_no_option_value` and `controller_no_taxonomy` each under-perform the full
   controller on `P(discovery)` — i.e. both new terms (option value, taxonomy) are pulling their
   weight, not riding on a plain-UCB effect.
4. On the adversarial family, the controller's mean wasted budget (compute spent past the point a
   baseline-1 agent would have given up) is not more than 25% higher than baseline 3's — i.e. it
   is not badly fooled by phantom structure.

If (1)-(2) hold but (3) fails, record as **Partial: controller wins, mechanism unclear** (the
option-value/taxonomy terms are not shown to be load-bearing — a plain UCB bandit over branches
might already do this). If (1) fails, record **Refuted** — the plateau-aware controller does not
do better than treating exploration as an undifferentiated diversity problem, consistent with the
source conversation's own concern that raw diversity doesn't help.

A tuning config (lambda, mu, nu, eta, thresholds) is selected on `dev` only; hidden and
adversarial are each read once with that single pre-selected config.

## Known limitations to state in any writeup

- Synthetic, fully-known-generator worlds only — this is plumbing/mechanism evidence, not a claim
  about real scientific or ML research (per `CLAUDE.md`'s guardrail: toy passes are plumbing, not
  evidence about real models).
- qwen3:8b only; no claim about frontier models or larger local models.
- 3 dev/hidden environment families plus 1 adversarial family is a small sample of possible
  "hidden mechanism" shapes; a Confirmed verdict licenses Stage 1 design, not a general claim
  about research search.
- No weight training in Stage 0 — a Confirmed result says the *policy* is worth distilling, not
  that distillation will work.
