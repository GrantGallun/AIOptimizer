# Activation Memory: memory-as-steering vs in-context

Does the best way to deliver a retrieved memory to a model depend on the memory *type*?
This thread holds the memory content fixed and varies only the **delivery channel**:

- `in_context` — paste the memory text into the prompt (RAG / MemGPT / Generative Agents).
- `activation` — inject a residual-stream steering vector derived from the memory text
  (representation engineering / CAA), spending zero prompt tokens.

The pre-registered prediction (`PREREGISTRATION.md`, H-AM-01) is a **crossover interaction**:
activation wins for *procedural* memory (how to act), in-context wins for *factual* memory
(a specific project fact). Read the pre-registration before running the confirmatory HF pass.

## Why this is the novel seam

The steering thread and the Brain Runtime thread treat memory as two separate things:
Brain Runtime retrieves *text*; steering manipulates the *residual stream*. This thread is
their intersection — the same memory delivered both ways, with a pre-committed prediction
about which type each channel serves. The activation vector is derived identically for both
memory types, so any measured interaction is attributable to the memory type, not to a
different vector-construction method.

## Design (2x2)

- Factor A — memory type: `procedural`, `factual` (`data/memory_items.jsonl`).
- Factor B — channel: `none`, `in_context`, `activation` (+ controls).
- Vector: `unit(mean_hidden(memory.text) - mean_hidden(NEUTRAL_REFERENCE))`, injected at a
  chosen layer with a coefficient.
- Score: preference log-prob margin between a memory-consistent `positive_answer` and a
  memory-violating `negative_answer`; lift is measured against the `none` baseline. Reuses
  the steering thread's scoring convention.

Controls: `activation_random`, `activation_wrong`, `in_context_wrong` (see pre-registration).

## Backends

- `toy` — deterministic, no dependencies. Proves the harness computes channels, the
  type x channel interaction, and the gate. Its crossover is **built in** by the toy's
  mechanics (a behavior is a continuous axis a vector can move; a fact is a discrete token
  only text can supply). It is not evidence about real LMs.
- `hf` — confirmatory. Requires torch + transformers and a local causal LM.

## Commands

```powershell
python -m unittest tests.test_activation_memory

# Deterministic plumbing + interaction check (built-in crossover, plumbing only):
python experiments\activation_memory\memory_harness.py toy --out results\activation_memory\toy_run.json

# Confirmatory protocol.
# 1. Sweep layer/coefficient on DEV only; the objective is the pre-registered crossover.
python experiments\activation_memory\memory_harness.py hf-sweep --model HuggingFaceTB/SmolLM2-135M-Instruct --tasks experiments\activation_memory\data\tasks_dev.jsonl --layers 8,12,16,20,24 --coefficients 0,2,4,8,12 --out results\activation_memory\hf_sweep_dev.json
# 2. Take best.layer / best.coefficient from the sweep, then run hidden + adversarial ONCE with it.
python experiments\activation_memory\memory_harness.py hf --model HuggingFaceTB/SmolLM2-135M-Instruct --tasks experiments\activation_memory\data\tasks_hidden.jsonl --layer <best> --coefficient <best> --out results\activation_memory\hf_hidden.json
python experiments\activation_memory\memory_harness.py hf --model HuggingFaceTB/SmolLM2-135M-Instruct --tasks experiments\activation_memory\data\tasks_adversarial.jsonl --layer <best> --coefficient <best> --out results\activation_memory\hf_adversarial.json
```

`hf-sweep` ranks dev configs by the pre-registered objective (interaction gap first, then
how much activation wins procedural). It only ever runs on the tasks you pass, so point it at
`tasks_dev.jsonl` and never at hidden. Then confirm the single selected config with `hf`.

## Decision

The gate lives in `interaction_and_gate`: `interaction_gap >= 0.5`, activation wins
procedural, in-context wins factual, and the target beats its controls. `verdict` is one of
`Confirmed`, `Partial: interaction without crossover`, `Partial: crossover but a control
failed`, or `Refuted`. Record the hidden-split verdict in `memory/hypothesis-graveyard.md`.

## Status

Harness + pre-registration + splits landed 2026-07-09; toy gate passes. No confirmatory HF
run yet — the HF numbers are the actual experiment.
