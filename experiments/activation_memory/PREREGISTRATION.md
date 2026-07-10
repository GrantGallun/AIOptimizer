# Pre-Registration: Memory-as-Steering vs In-Context Memory

Registered: 2026-07-09
Status: Registered (no confirmatory HF run yet)

This document is written **before** the confirmatory run so the benchmark cannot be
reshaped around the result. Per `memory/ideas.md` IDEA-20260709-06, the predictions,
splits, controls, and decision gate below are frozen. Any change after the first HF
confirmatory run must be logged as an amendment with a date, not a silent edit.

## Question

An external memory system can deliver a retrieved memory to a model in two ways:

- **In-context**: paste the memory text into the prompt (what MemGPT / Generative
  Agents / RAG do).
- **Activation**: turn the memory into a residual-stream steering vector and inject it
  (what representation-engineering / CAA do), never spending prompt tokens.

Nobody has cleanly answered *which memory types each channel is better for*, holding the
memory content fixed. That is the gap this thread targets.

## Primary hypothesis (H-AM-01)

The better delivery channel depends on memory type. Specifically we predict a
**crossover interaction**:

- **Procedural memory** (how to act: verify before claiming, ask when unsure, be
  concise) steers **better via activation injection** than in-context.
- **Factual memory** (a specific project fact: the config delimiter is a pipe) works
  **better in-context** than via activation.

Rationale: a behavior is a diffuse direction in activation space that a single vector can
express, whereas a specific fact requires emitting a discrete token that a mean-pooled
vector cannot reliably carry.

## Design (2x2, content held fixed)

The same memory item is delivered through both channels, and the activation vector is
derived **identically** for both memory types so any measured interaction comes from the
memory *type*, not from a different vector-construction method:

```
memory_vector(item) = unit( mean_hidden(item.text) - mean_hidden(NEUTRAL_REFERENCE) )
inject: hidden[layer] += coefficient * memory_vector(item)
```

Factor A — memory type: `procedural`, `factual`.
Factor B — channel: `none` (baseline), `in_context`, `activation`.

Scoring reuses the steering thread's preference margin: for each task, the log-prob margin
between a memory-consistent `positive_answer` and a memory-violating `negative_answer`,
plus a lexical-generation check. Lift is measured against the `none` baseline.

## Splits (frozen)

- `data/memory_items.jsonl` — the memories (procedural + factual).
- `data/tasks_dev.jsonl` — tuning only (pick layer/coefficient here).
- `data/tasks_hidden.jsonl` — confirmatory; never inspected during tuning.
- `data/tasks_adversarial.jsonl` — stress cases (distractor facts, tempting shortcuts).

## Controls

- `activation_random`: a random vector of matched norm, same layer/coefficient. Guards
  against "any residual perturbation helps."
- `activation_wrong`: the activation vector of a *different* memory of the same type.
  Guards against a generic "some memory direction helps" effect.
- `in_context_wrong`: a different memory of the same type pasted in. Guards against
  "any extra text helps."

## Decision gate (pre-committed)

Let `margin_lift(channel, type)` be the mean steered-minus-baseline preference margin on
the **hidden** split, and define:

```
adv_proc = margin_lift(activation, procedural) - margin_lift(in_context, procedural)
adv_fact = margin_lift(activation, factual)   - margin_lift(in_context, factual)
interaction_gap = adv_proc - adv_fact
```

H-AM-01 is **Confirmed** only if all hold on hidden:

1. `interaction_gap >= 0.5` (the crossover exists and is non-trivial).
2. `adv_proc > 0` (activation actually wins for procedural).
3. `adv_fact < 0` (in-context actually wins for factual).
4. Procedural `activation` beats `activation_random` and `activation_wrong` on margin.
5. Factual `in_context` beats `in_context_wrong` on margin.

If (1) holds but (2)/(3) do not, record as **Partial / interaction without crossover**.
If (1) fails, record **Refuted**. A tuning config is selected on `dev` only; the hidden
numbers reported are from that single pre-selected config.

## Known limitations to state in any writeup

- SmolLM2-135M is tiny; a headline claim needs a 1-3B instruct model too (IDEA-06).
- The task sets are self-authored; a novelty claim needs one external fact/behavior set
  not designed here.
- Mean-pooled memory vectors are the simplest possible derivation; a null result does not
  rule out better derivations (last-token, learned probes, per-layer).
