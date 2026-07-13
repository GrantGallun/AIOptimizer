# Pre-Registration v14: Two-stage pressure router for live context injection

Registered: 2026-07-12 by Fable (Claude Fable 5)
Status: Registered (no v14 run yet)
Provenance: Codex's live-dogfood proposal (bus m0140) after the first real injection receipt showed
the size-based eligibility flaw: the 12k-char early exit conflates SIZE with salience risk — it
blocked injection for a 7.1k-char thread containing a buried prior decision, while the first
injection that DID fire wasted budget on a superseded conclusion and a raw traceback.

## Design (frozen; Codex implements to this, t0043)

Replace magic-size eligibility with two deterministic stages:
1. **Cheap load signals**: token-like load, redundancy/compressibility (repeated-run detection),
   turn geometry — computed without the encoder.
2. **Cached semantic signals**: peak relevance (existing), PLUS age-of-best-relevant-record and
   recent-tail coverage (is the relevant fact already represented in the last N turns?).
Inject ONLY when a relevant record is old/obscured AND not already represented recently.
Duplicate/repeated-character spam must never qualify. The 12k figure survives only as a
configurable hard COST ceiling. Unchanged invariants: privacy-before-ranking, vague→raw,
additive-only, fail-open, exact provenance, content-free receipts.

## Fixtures (frozen classes; deterministic generators, seeds FRESH_HIDDEN_SEEDS["v14"]=(1109,1117,1123))

(a) dense <12k with a buried OLD relevant decision → MUST inject and select it;
(b) >12k of duplicate/repeated-token spam → must NOT inject;
(c) relevant fact already in the recent tail → must NOT inject (already represented);
(d) vague breadth query → raw route (existing gate);
(e) superseded-conclusion + traceback texture (the live failure) → injected set must EXCLUDE the
    traceback body and prefer the current decision (supersession = later same-topic record wins).

## Gate (hidden fixtures, read once)

Confirmed iff per-class routing/selection correctness ≥ 0.9 for (a)–(d) AND class (e) excludes the
traceback in ≥ 0.9 of cases AND zero privacy regressions. Any class failure reported per class.

— Fable (Claude Fable 5), 2026-07-12
