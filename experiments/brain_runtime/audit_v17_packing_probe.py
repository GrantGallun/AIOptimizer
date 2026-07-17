"""Confirm the crowd-out mechanism: is raw-arm fact presence non-monotone in budget on
the SAME case? First-fit skip-and-continue predicts cases where the fact survives at 650
but is crowded out at 2600 — impossible under any tail/prefix truncation."""
import sys
from collections import Counter

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"c:\Code\AIOptimizer")

from experiments.brain_runtime.make_context_cases import generate_volume_cases
from aioptimizer.context_compiler import ConversationCompiler

compiler = ConversationCompiler()
flip = Counter()
sizes_report = None
for seed in (1301, 1303, 1307):
    for budgeted in (True,):
        cases_hi = generate_volume_cases(seed, n_cases=40, n_messages=320, budget_chars=2600)
        cases_lo = generate_volume_cases(seed, n_cases=40, n_messages=320, budget_chars=650)
        for hi, lo in zip(cases_hi, cases_lo):
            compiled = compiler.compile(hi["messages"])
            raw_hi = compiler.render_raw(compiled, budget_chars=2600)
            raw_lo = compiler.render_raw(compiled, budget_chars=650)
            p_hi = str(hi["expected"]) in raw_hi
            p_lo = str(lo["expected"]) in raw_lo
            flip[(p_hi, p_lo)] += 1
            if sizes_report is None and (not p_hi) and p_lo:
                recs = compiler._public_records(compiled)
                lens = [len(compiler._render_record(r)) for r in recs]
                fact_idx = next((i for i, r in enumerate(recs)
                                 if str(hi["expected"]) in compiler._render_record(r)), None)
                sizes_report = (lens[:12], fact_idx,
                                lens[fact_idx] if fact_idx is not None else None,
                                sum(lens) / len(lens))

print("fact presence on the SAME case, raw arm (hi=2600, lo=650):")
for (p_hi, p_lo), n in sorted(flip.items()):
    marker = "  <-- impossible under tail/prefix truncation" if (p_lo and not p_hi) else ""
    print(f"  present@2600={p_hi!s:<5} present@650={p_lo!s:<5}  {n:>3}{marker}")

if sizes_report:
    lens, idx, fact_len, mean_len = sizes_report
    print(f"\nsample crowded-out case: first 12 record sizes={lens}")
    print(f"  fact record: index {idx}, {fact_len} chars (mean record: {mean_len:.0f} chars)")
