# Design: Adaptive Research Search — a plateau-aware branch controller

Status: design stage, no run yet. Companion to `PREREGISTRATION.md` (Stage 0, frozen before any
code runs).

## Origin and scope

Condensed from a long external conversation (relayed by the user, 2026-09-15) about whether an
LLM-based research loop can learn *where to point its own search*, not just generate more ideas.
This is a **separate research thread** from this repo's north star (the gateway's determinism
dividend, `memory/ideas.md`). It shares no code path with `gateway/` and does not change any
existing pre-registration. Treat it as a second, independent line — do not fold its results into
`FINDINGS.md`'s product claims.

## The one-line thesis

A small, model-free controller that tracks *why* a research branch stalled (four distinct plateau
types, not one undifferentiated "stuck") and allocates the next unit of search compute
accordingly should recover a hidden generating mechanism faster than either (a) one trajectory,
(b) high-temperature resampling, or (c) fixed diverse seeding with no adaptation.

## Why this is narrower than it sounds

The source conversation's own trajectory converged on "parallel diverse research branches +
hill-climbing", which is close to what's already public: Anthropic's automated-researcher work
already runs parallel seeded agents and reports that raw diversity does **not** cleanly predict
hill-climbing gains, and that seed diversity's benefit decays and reconverges after roughly 20
proposed methods. Building "more seeds in parallel" would mostly reproduce a weaker version of
that. The narrower, still-open question their own result leaves on the table:

> Can a controller detect *what kind* of plateau a branch is in, and choose the matching scale of
> conceptual change, instead of treating every plateau the same way (reseed vs. don't)?

That is what Stage 0 tests. Nothing here proposes to out-scale a frontier lab; the claim is about
a decision policy, testable on a laptop GPU with a toy environment.

## Architecture

**Research Tree.** Each branch node carries:

```
hypothesis_text, seed_vector      # weights over reusable concept tags
Q            # observed value so far (reduction in prediction error attributable to this branch)
U            # uncertainty (spread of recent observations / experiments run)
O            # option value: estimated upside IF this branch's current anomaly resolves
N            # conceptual novelty relative to sibling branches (embedding distance)
C            # compute already spent
plateau_type # None | A_exhausted | B_unresolved | C_convergent | D_representation
history      # every (action, experiment, observation) tuple
```

**Actions**, from any node:

```
a0  continue/refine       — small parameter-level experiment on the current hypothesis
a1  local seed mutation   — Sb + eps  (perturb the concept-weight vector slightly)
a2  assumption swap       — replace one dominant assumption, keep the rest
a3  representation change — re-derive features/variables, same problem
a4  radical reseed        — draw a new seed vector from a distant region
a5  revive dormant branch — reactivate a parked branch whose U or O has since risen
```

**Priority** (the only genuinely new numeric object in this project — what the controller
actually optimizes):

```
P(b) = Q_b + lambda*U_b + mu*O_b + nu*N_b - eta*C_b
```

lambda, mu, nu, eta are tuned on `dev` only and frozen before the hidden read (see
`PREREGISTRATION.md`). This is a UCB-style bonus (Q + uncertainty) plus two terms the plain
multi-armed-bandit form doesn't have: **option value** (a mediocre branch can still be worth
keeping alive at low compute if its upside-if-right is large) and **novelty** (a bonus for
occupying a different region of concept space, capped by a validity floor so it can't reward
incoherent hypotheses).

**Plateau taxonomy** (the classifier the controller conditions its action on):

| Type | Signature | Response |
|---|---|---|
| A — exhausted | delta-Q ~= 0, U low | shift aggressively (a3/a4) |
| B — unresolved | delta-Q ~= 0, U still high, anomalies unexplained | keep branch alive, change experiment type, not direction |
| C — convergent | >= 2 branches' seed vectors are converging | force a distant reseed on one of them (a4) |
| D — representation failure | many hypotheses share the same residual structure | representation change (a3), not a new hypothesis in the same variables |

Collapsing all four into "score stopped improving -> reseed" is exactly the behavior we're trying
to beat (baseline 4, below).

**Possibility vs. probability**, kept explicit rather than collapsed into one score, because the
source conversation's own critique of "make the model more ambitious" was that models silently
multiply these together and then discard low-probability/high-value ideas too early:

```
p_b = P(branch succeeds)
v_b = V(branch | success)
```

Reported separately in every branch record; `O_b` in the priority formula is `p_b * v_b`, but the
harness must print both factors, not just the product, so a `p=.05, v=100` branch is visibly
distinguishable from a `p=.9, v=2` one during analysis. This is the exact mechanism, not a
metaphor, for the "ambition" the source conversation kept circling.

**Regret metric** (the sharpest evaluation of "did the controller waste the breakthrough"):

```
regret = V(best reachable direction) - V(best explored direction)
```

computable only because the environment (below) is synthetic and fully known to the harness.

## Staged roadmap — only Stage 0 is authorized right now

| Stage | What changes | Status |
|---|---|---|
| 0 | Controller only. Base LLM (qwen3:8b, local) is **frozen**. Compares controller vs. baselines on `P(discovery\|budget)`. | **Authorized — see `PREREGISTRATION.md`** |
| 1 | Distill winning controller decisions into LoRA weights (`(state, good_action, bad_action)` preference pairs); reset the research tree; test whether a frozen-then-QLoRA'd model conducts better search with *no* external controller. | Gated behind a Stage 0 win. Not designed yet. |
| 2 | Recursive: M0 -> M1 -> M2 -> M3, each trained on the previous generation's trajectories, measuring whether the per-generation improvement holds or decays to zero. | Gated behind Stage 1. Not designed. |

Do not skip to Stage 1/2. The source conversation's own strongest self-criticism was spending
compute on weight updates before knowing whether the search *policy* even works.

## What already exists in this repo and should be reused, not rebuilt

- **Local worker.** `experiments/local_worker/ollama_client.py` + qwen3:8b already pulled
  (`experiments/local_worker/README.md`) is exactly the "run this on a laptop GPU" lane the
  source conversation asked for. Use it as-is; do not stand up a second local-model adapter.
- **Statistics.** `experiments/brain_runtime/stats.py` (`wilson_interval`, `gap_significant`) is
  the existing Wilson-CI machinery this repo already uses for every gate (HYP-20 through
  HYP-44). The Stage 0 gate reuses it rather than inventing new significance math.
- **Split discipline.** `activation_memory/PREREGISTRATION.md` and `activation_steering`'s
  `data/{eval_dev,eval_hidden,eval_adversarial}.jsonl` convention is copied directly: frozen
  generator seeds for dev/hidden/adversarial synthetic worlds, hidden read once.
- **CoALA controller shape** (`experiments/brain_runtime/coala.py`: `ActionKind`,
  `DecisionContext`, `CycleResult`) is a plausible skeleton for the branch controller's
  decision-cycle plumbing (bounded actions before a terminal choice) — worth reading before
  writing a new one from scratch, but it is currently wired to memory retrieval/grounding
  actions, not to a research-branch tree, so expect to fork it rather than import it directly.
- **`agent_bus/board.py`** looks tempting (it's already a scoreboard with dependency-gated
  dispatch, claim/retire, and a watchdog for stale claims) but **do not force-fit it**: it
  dispatches ready tasks FIFO among satisfied dependencies for *human/agent* coordination; it has
  no concept of a numeric priority score, an uncertainty bonus, or "revive a dormant branch
  because its option value rose." The actual controller — the falsifiable object in this project
  — is a small standalone priority queue over the formula above. Keep it model-free and small
  first, exactly the discipline `brain_runtime/README.md` states for its own v0 ("deliberately
  model-free: the first question is whether the architecture has useful mechanics before we
  attach it to a live model").

## What's new and must be built for Stage 0

1. `environments/` — 3 frozen synthetic generator families with a *known* hidden mechanism the
   harness can check against directly (no LLM judge needed): regime-switch, hidden-interaction,
   confounding. A 4th family (deceptive-correlation-only, no real hidden structure) is held out
   as the adversarial split — it exists to catch a controller that hallucinates structure and
   burns budget chasing a phantom breakthrough.
2. `agent/` — turns an environment's observations into experiment proposals via the local LLM
   (qwen3:8b). This is the only place a model runs; everything else is deterministic.
3. `search/` — the branch tree, plateau classifier, and priority controller described above.
4. `evaluation/` — `P(discovery | budget)` and `T_discovery` against the known generator,
   `wilson_interval`/`gap_significant` reused from `brain_runtime/stats.py`, plus the `regret`
   metric.

## Baselines (Stage 0 gate needs all five, matched compute budget)

1. Single trajectory, no branching.
2. High-temperature resampling only (checks whether the controller beats plain sample diversity).
3. Fixed diverse seeding, no adaptation (checks whether *adaptive* reallocation matters at all).
4. Random reseed on any plateau, no taxonomy (checks whether the 4-way plateau classifier adds
   anything over "reseed when stuck").
5. Adaptive controller (the treatment).

## Explicit non-goals for Stage 0

- No attention-architecture changes, no learned seed tokens, no recurrent-attention mechanism.
  Interesting, but Stage 3+ territory per the source conversation's own sequencing — architecture
  changes before the search-policy claim is established would confound the result.
- No weight training of any kind.
- No real ML/scientific research domain. Synthetic, fully-known-generator worlds only, so
  discovery can be scored exactly instead of by an LLM judge.
- No recursive M0->M3 loop.

## Open research decisions (Fable-owned, not delegable per `CLAUDE.md`)

- The exact lambda/mu/nu/eta weighting and plateau-detection thresholds (tuned on dev, frozen
  before hidden — this is the pre-registration's job, not a worker's).
- Choosing which environment family is "Stage 0 domain A" (already decided here: hidden-mechanism
  discovery, per the source conversation's own recommendation — cheapest ground truth, no LLM
  judge, most conceptual freedom for the plateau taxonomy to matter).
- Interpreting the hidden-split gate into a verdict for `memory/hypothesis-graveyard.md`.

## What could be delegated to a Sonnet worker, once this design is locked

- Implementing the 3+1 environment generators to a fixed interface and frozen seeds (mechanical,
  testable against known ground truth).
- Wiring `wilson_interval`/`gap_significant` into the evaluation script.
- The five-baseline harness runner, given the priority formula and plateau taxonomy already
  specified above (no open judgment calls left once this doc and the pre-registration are
  frozen).

Each of those is a candidate worker spec once `PREREGISTRATION.md` is reviewed and not amended
further — not before.
