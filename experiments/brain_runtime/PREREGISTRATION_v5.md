# Pre-Registration v5: The idiomatic LangGraph agent vs the AIOptimizer kernel

Registered: 2026-07-11 by Fable (Claude Opus 4.8)
Status: Registered (no v5 model run yet)
Thesis link: PROJECT_PLAN §00 / LANDSCAPE.md (LangGraph row = "durable workflow / LLM-driven control
flow"). The Layer-2 product comparison against the framework people actually use.

## Motivation (and the honesty update from HYP-27)

HYP-26 showed the kernel beats a `naive` stack 100-0. But HYP-27's dev run showed a *strong prompt*
alone lifts a no-invariant `prompted` arm to **0.992** recurrence — so most of the 100-0 gap was the
neutral prompt + weak retrieval, not the deterministic invariants. The remaining question is the one
this experiment answers directly: when you build the agent the **idiomatic LangGraph way** — an LLM
router node that DECIDES its own control flow — does it reliably retrieve-then-answer (like the
strongly-prompted arm) or does it skip retrieval (like naive)? And how does it compare to the
kernel's *guarantee*?

## Arms

- **`langgraph_idiomatic`**: a real `langgraph.StateGraph` — `router` (LLM decides retrieve vs
  answer) → conditional edges → `retrieve` (encoder top-1 over the SAME learned-rule store the
  kernel uses) loops back → `answer` (LLM computes). Idiomatic ReAct-style; the LLM chooses whether
  to retrieve. Same model, same 30-operator×5 task, same within-session learning protocol.
- **`full_kernel`** (reference, already committed HYP-26): forces retrieve→reason→ground.

Retrieval quality is held equal (same encoder, same store). The only difference is control-flow
determinism: LangGraph lets the model decide to retrieve; the kernel guarantees it.

## Hypotheses (pre-committed, symmetric)

- **H-v5a (kernel guarantee)**: `full_kernel` recurrence − `langgraph_idiomatic` recurrence ≥ 0.30
  on hidden → the idiomatic LLM-driven agent hits the structural failure (skips retrieval) and the
  kernel's guarantee matters.
- **H-v5b (idiomatic-suffices, equally reportable)**: `langgraph_idiomatic` recurrence ≥
  `full_kernel` − 0.10 → on qwen3:8b the idiomatic agent's router reliably retrieves; the kernel's
  determinism adds little on this model (consistent with HYP-27). Honest, kernel-tempering.
- **Partial** in between.

Primary reported metric alongside recurrence: **`retrieval_participation_rate`** — does the router
actually choose to retrieve? That is the mechanism, whichever way the gap falls.

## Splits

- **Dev**: seed 20260711 (sanity). **Hidden**: 101, 103, 107 (verdict). `full_kernel` hidden numbers
  are the committed HYP-26 runs (comparable by seed). Gate frozen before the model run.

## Integrity notes

- This is NOT "LangGraph the library is bad." The kernel's invariants could be implemented AS a
  LangGraph graph; the contrast is idiomatic LLM-driven control flow vs a deterministic guarantee.
- Same learned-rule store and encoder retrieval as the kernel, so retrieval quality is not the
  confound — control flow is.
- qwen3:8b, one synthetic task, 3 hidden seeds; hosted-model / real-task replication owed.

— Fable (Claude Opus 4.8), 2026-07-11
