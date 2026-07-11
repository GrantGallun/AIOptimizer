# AIOptimizer — Findings

**Thesis (PROJECT_PLAN §00):** LLMs are probabilistic *semantic coprocessors*; AIOptimizer is the
**deterministic kernel** that makes their work reliable. Where a naive agent lets the LLM decide
everything ("glue"), the kernel replaces each fragile decision with a deterministic mechanism and
measures whether that determinism actually helps.

Everything below is validated with the same discipline: a Fable-owned **pre-registration** (arms,
gate, metric frozen in advance), a **dev split** for sanity and a never-inspected **hidden split**
for the reported verdict, negative controls, and a **hypothesis graveyard** where every claim —
pass or fail — is buried with its evidence. Toy/synthetic passes are labeled as plumbing, not
evidence about real models.

Models are local (Ollama): **qwen3:8b / qwen3:14b / llama3.2:3b**. Encoder is all-MiniLM-L6-v2.

---

## What the kernel provides, and the evidence for each

### 1. Governed memory → within-session *learning* the raw model lacks
An LLM can't remember a verified fact across turns; the kernel gives it typed, scoped, governed
memory. **HYP-19** showed external memory enables genuine within-session learning — but the base
model's rule-application caps the effect. Memory is necessary, not sufficient.

### 2. The right *context*, selected by an encoder, beats dumping everything
As memory grows, stuffing it all into the prompt degrades the model ("lost in the middle").
- **HYP-20** (qwen3:14b): encoder-scored top-k selection holds accuracy as context grows while
  dump-everything degrades.
- **HYP-21** (real code): on functions-as-concepts, both encoder selection *and*
  abstraction-to-signatures beat dumping — and this **replicates on genuine third-party code**
  (1788 functions from PyTorch's `torch/utils`): `dump_all` **0.83 → 0.33** as context grows,
  while selection/abstraction hold at the **oracle** ceiling (0.83). Not an artifact of our own repo.

### 3. Structured reasoning (CoALA) unlocks the learning memory enables
- **HYP-22** (qwen3:8b): a CoALA controller with *mandatory retrieval + a reasoning step* takes
  novel-operator recurrence accuracy from **0.00 → 1.00**, reconciling HYP-19 (the cap was
  structural, not a memory limit).
- **HYP-23** (capstone, scale): at 30 operators, crude jaccard retrieval collapses to **0.125**
  while encoder retrieval holds at **1.00** under the same CoALA reasoning. The kernel needs *both*
  good retrieval **and** structured reasoning: *select the right context, then let the model reason
  over it.*

### 4. Constrained decoding makes the controller's actions structurally valid
The CoALA controller drives itself by emitting action-JSON; free-form, that JSON sometimes fails to
parse (a "malformed action").
- **HYP-24** (dev, Partial): a permissive *union* schema only *halved* malformed actions —
  schema-valid ≠ parse-valid, because the parser enforces per-kind fields the union left optional.
- **HYP-25** (hidden gate, **Confirmed**): a per-kind **conditional** schema (that *is* the parser's
  contract) drives constrained malformed rate to **0.000** (free-form 0.055) via Ollama `format=`,
  and because malformed cycles are wasted cycles, task completion **rises +12 points**
  (0.693 → 0.813). The guarantee is model-independent (holds on llama3.2:3b too).
  *One-liner: "constrain the shape" only equals "guarantee the contract" when the schema is the contract.*

### 5. The whole kernel vs the naive stack — **Confirmed** (prereg v4, HYP-26)
Composing everything (encoder retrieval + constrained actions + forced retrieve→reason→ground) and
letting the model choose its own actions at scale (30 operators, qwen3:8b, hidden seeds).
**naive** recurrence **0.000** vs **full_kernel 1.000** (identical on all three hidden seeds). The
naive failure is **structural**: reasoning-participation **0.000**, retrieval-participation 0.009 —
left free, the model *grounds a null answer immediately*, never retrieving or reasoning. The kernel's
invariants force `retrieve→reason→ground` (participation 1.000) and encoder retrieval feeds the right
rule → 120/120. *The thesis in one number: a capable model left to "decide for itself" blurts null
answers; deterministic structure + good retrieval convert it into reliable learning.*

---

## Two failure modes the kernel eliminates
- **Malformed** ("said it wrong"): invalid/unparseable action JSON. Fixed by **constrained decoding**
  (HYP-25). Common on weak models.
- **Structural** ("did the wrong thing"): valid JSON, bad choice — e.g. answering without retrieving
  or reasoning. Fixed by **forced-structure invariants** (HYP-22, v4). Common even on strong models.

## Honest limitations (carried in every writeup)
- Local models only; small hidden splits (3–5 seeds); mostly synthetic novel-operator tasks.
- HYP-21's context result has real-code external validity; the others still owe a hosted-model and a
  second-task-family replication before any general claim.
- The v4 comparison composes three mechanisms — it shows they compose, not their individual sizes
  (those are HYP-21/23/25).

---
*Full evidence, gates, and buried hypotheses: [`memory/hypothesis-graveyard.md`](memory/hypothesis-graveyard.md).
Pre-registrations: [`experiments/brain_runtime/PREREGISTRATION_v*.md`](experiments/brain_runtime/).*
