"""Adversarial audit of the v16/v17 verdicts, from the frozen row data.

Two suspicions:
  (1) raw improves as budget shrinks (0.533 -> 0.608 -> 0.642) — anti-mechanism.
  (2) success was never conditioned on context_expected_present. If raw succeeds with
      the fact ABSENT from context, its 'success' is guessing, not retention, and the
      baseline of the H-v17a cost gate means something different than claimed.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

RES = Path(r"c:\Code\AIOptimizer\results\brain_runtime")


def rows_for(pattern):
    out = []
    for p in sorted(RES.glob(pattern)):
        out.extend(json.load(open(p, encoding="utf-8"))["rows"])
    return out


def audit(label, rows):
    print(f"\n{label}  (n={len(rows)} rows)")
    print(f"  {'arm':<11} {'n':>4} {'succ':>6} | {'fact_in_ctx':>11} | {'succ|present':>12} {'succ|ABSENT':>12}")
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
    for arm in sorted(by_arm):
        rs = by_arm[arm]
        n = len(rs)
        succ = sum(1 for r in rs if r.get("success"))
        present = [r for r in rs if r.get("context_expected_present")]
        absent = [r for r in rs if not r.get("context_expected_present")]
        sp = sum(1 for r in present if r.get("success"))
        sa = sum(1 for r in absent if r.get("success"))
        sp_txt = f"{sp}/{len(present)}" if present else "-"
        sa_txt = f"{sa}/{len(absent)}" if absent else "-"
        print(f"  {arm:<11} {n:>4} {succ/n:>6.3f} | {len(present)/n:>11.3f} | {sp_txt:>12} {sa_txt:>12}")


# v17 hidden, per budget
for b in (2600, 1300, 650):
    audit(f"v17 hidden @budget {b} (N=320)", rows_for(f"v17_org_N320_b{b}_h*.json"))

# v16 hidden N320 for comparison
audit("v16 hidden @budget 2600 (N=320)", rows_for("v16_org_N320_h*.json"))

# If raw succeeds while the fact is absent: what is it emitting? Sample a few.
print("\n--- sample raw successes WITHOUT the fact in context (v17 b2600) ---")
shown = 0
for r in rows_for("v17_org_N320_b2600_h*.json"):
    if r["arm"] == "raw" and r.get("success") and not r.get("context_expected_present"):
        print(f"  expected={r['expected']!r:<22} response={r['response'][:60]!r}")
        shown += 1
        if shown >= 5:
            break
if shown == 0:
    print("  (none — every raw success had the fact in context)")
