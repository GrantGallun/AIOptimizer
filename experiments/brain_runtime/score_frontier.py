#!/usr/bin/env python3
"""Score both subscription-frontier response files against the packets (one command)."""
import json, subprocess, sys
from pathlib import Path

PACKETS = "results/brain_runtime/frontier_context_v1_dev_packets.jsonl"
RESPONSES = {
    "codex_gpt": "results/brain_runtime/frontier_responses_codex_gpt.jsonl",
    "claude_sonnet": "results/brain_runtime/frontier_responses_claude_sonnet.jsonl",
}
for model, path in RESPONSES.items():
    if not Path(path).exists():
        print(f"{model}: responses not present yet"); continue
    out = f"results/brain_runtime/frontier_scored_{model}.json"
    r = subprocess.run([sys.executable, "experiments/brain_runtime/frontier_context_eval.py",
                        "score", "--packets", PACKETS, "--responses", path, "--out", out],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"{model} scoring FAILED:", (r.stderr or "").splitlines()[-3:]); continue
    d = json.loads(Path(out).read_text(encoding="utf-8"))
    summaries = d["summaries"] if isinstance(d.get("summaries"), list) else list(d.get("summaries", {}).values())
    print(f"\n=== {model} ===")
    for s in sorted(summaries, key=lambda x: x["arm"]):
        print(f"  {s['arm']:<12} success={s.get('end_to_end_success', s.get('success')):.3f}")
