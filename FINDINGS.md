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

### 6. Honest stress-test — is the kernel necessary, or just good prompting? *(prereg v4.2/v5)*
Because HYP-26's 100-0 is *so* clean, we attacked it before a skeptic could — two fair baselines on
qwen3:8b (hidden):
- **Strong prompt, no invariants** (HYP-27): recurrence **1.000** — a strong "retrieve→reason→answer"
  prompt *fully matches* the kernel. So the 100-0 was mostly the *neutral prompt + weak retrieval* in
  naive, not the deterministic invariants.
- **Idiomatic LangGraph agent** (HYP-28, real `StateGraph`, LLM router decides control flow):
  recurrence **0.872**, and it **always retrieves** (participation 1.000) — it does *not* collapse
  like naive. Its ~0.13 gap is retrieval/compute brittleness (top-1 retrieval + single-shot answer,
  worsening as memory grows), the kind of thing better retrieval would close.

**The honest headline:** on a *capable, instructable* model, good prompting and even an idiomatic
framework agent get you to ~87–100%. The kernel's determinism is **not** a night-and-day miracle
there — it is a **guarantee** (100% vs prompting's ~97% step-compliance) and **robustness at scale**.
The kernel is a reliability *floor*, not a capability ceiling — and its mechanisms are complementary
to LangGraph, not opposed (they could be implemented as a LangGraph graph).

### 7. The determinism dividend across model capability — ≤ 0 at both ends (HYP-29, Refuted)
We predicted the kernel would rescue *weak* models that can't follow prompts. Wrong — measured on
llama3.2:3b (with the audit-fixed action budget and a repaired invariant-ordering bug the weak model
itself exposed): strong prompt **0.550**, full kernel **0.033**. The dividend is **negative**.
Mechanism: *invariants that correct behavior mid-cycle consume the action budget a weak model needs
to finish; a strong prompt shapes behavior from the first token for free — and a prompt can carry
task knowledge, which a generic invariant cannot.* With qwen3:8b's dividend of 0.000 (HYP-27), the
kernel's forced structure never beat good prompting on single-hop accuracy at either capability end.
Its accuracy case now rests entirely on the task-difficulty axis:

### 8. Multi-hop composition — the decisive test *(prereg v6, HYP-30: significant but modest)*
Depth-2 composed operators (`outer(inner(a,b), c)`: retrieve TWO rules, chain them) finally give
dynamic range instead of 0/1 cliffs. Dev (qwen3:8b, after two dated measurement amendments — reason-
token truncation, then a composition-aware prompt for the prompted arm): **kernel 1.000 vs
best-effort prompting 0.765**. A genuine mechanism appeared: the kernel's *deterministic retrieval
query* (built from the full observation) reliably surfaces both rules; the prompted model's
self-chosen query often missed one. Hidden verdict (fresh seeds 211/223/227): **kernel 0.987 vs prompted 0.884 — gap +0.103,
Wilson-significant but below the pre-registered 0.15 bar** → by the frozen gate, prompting largely
suffices; the kernel keeps a small real edge where tasks stress completion and retrieval breadth.

**The completed determinism-dividend curve:** weak model **−0.52** · capable/single-hop **0.00** ·
capable/multi-hop **+0.10** (significant). The kernel is a guarantees layer with a modest
composition-time edge — not a capability multiplier.

### 9. The distillation — the kernel's accuracy edge is ONE portable line (prereg v7, **Confirmed**)
Three arms on fresh hidden seeds (307/311/313): prompted **0.796** · prompted + *deterministic
query construction only* **0.974** · full kernel **1.000**. Rebuilding every retrieval query from
task state (`goal + observation`) instead of trusting the model to compose it carries **~87% of the
entire kernel advantage** (+0.178, Wilson-significant), with no invariants and no constrained
decoding. Mechanism: model-composed queries intermittently omit one of the needed rule names; the
deterministic query never does. **This is the project's most useful artifact: a one-line mechanism
any agent — LangGraph node, gateway middleware, prompted loop — can adopt.** The full kernel keeps a
last ~2.6% via its completion/ordering guarantees.

**Final thesis (evidence-forced, v1→v3):** *v1: determinism constrains the model's single pass.
v2: the kernel is a guarantees layer, not a capability multiplier. v3: where determinism does buy
accuracy, it buys it at specific decision boundaries — query construction today; sample-aggregation
contracts (self-consistency votes, disagreement-based cascades) are the pre-registered next test.*

### 10. Attention-organized conversation memory — the user's idea, the record's largest effect (prereg v9, **Confirmed**)
At an identical character budget, organizing conversation history into query-ranked semantic
clusters (vs raw chronological truncation) took success from **0.350 → 1.000** on hidden seeds
(gap **+0.650**, perfect source attribution, zero leaks). Mechanism, honestly labeled: **retention**
— recency truncation silently drops the mid-conversation fact ~65% of the time; relevance-ranked
clustering always keeps it. Extends the selection results (§2) to conversation-memory organization.
Carried flag: relevance-ranking also retains *sensitive* content more (privacy filter since landed).
**External-validity scope (v10/v10.1):** on REAL conversation logs the effect is geometry-dependent —
at 45% budget chronology barely fails (stop-rule; no read), at 30% the gap is +0.100 (n.s.). Attention
has **never lost a case** (120/120 hidden, perfect attribution, zero leaks): a strict-improvement
mechanism whose advantage grows with context pressure, not a universal +0.65.
Idea: the user; substrate: Codex; cases/gate: Fable.

### 11. Usage-heat compaction — the user's second confirmed idea (prereg v12, **Confirmed**)
"If context is repeatedly used, don't compact it." Naive similarity-based heat failed five render-
level falsification rounds (v11 — inferred usage confounds reference with resemblance). The
literature-correct form — **ACT-R base-level activation over OBSERVED usage counters**, deciding
what survives memory compaction when the future workload is unknown — passed its hidden gate:
**0.736 vs 0.403** (newest-first) and 0.444 (random), +0.333 Wilson-significant vs both controls,
perfect hot-rule retention with cold-rule eviction. Count usage; never infer it.

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
