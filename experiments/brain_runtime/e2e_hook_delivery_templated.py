"""End-to-end treatment-delivery test of the CURRENT hook build.

Drives plugins/aioptimizer-codex/scripts/user_prompt_submit.py exactly as Codex does:
JSON payload on stdin (prompt, transcript_path, cwd), rollout-format JSONL transcript,
history > 6k chars with a fact buried early and never restated. PASS = the hook prints
hookSpecificOutput with an <aioptimizer_context route="attention"> block AND the ledger
row in the test workspace says route=attention, injected=true.

This converts HYP-40's "treatment is not firing, cause unknown" into a reproducible
fires / does-not-fire observation on today's code.
"""
import json
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(r"c:\Code\AIOptimizer")
ENTRY = REPO / "plugins" / "aioptimizer-codex" / "scripts" / "user_prompt_submit.py"


def rollout_row(role: str, text: str) -> str:
    return json.dumps({
        "type": "response_item",
        "payload": {"type": "message", "role": role,
                    "content": [{"type": "output_text", "text": text}]},
    })


def build_transcript(path: Path) -> int:
    """~40 turns, ~14k chars; the deployment port fact is stated once, early, then buried."""
    rows = []
    rows.append(rollout_row("user", "Let's set up the deployment config for the ingest service."))
    rows.append(rollout_row(
        "assistant",
        "Noted. The verified deployment port for the ingest service is 7431. "
        "I will use that in every config we generate from here on.",
    ))
    fillers = [
        ("user", "Now write the retry wrapper for the fetch client, exponential backoff, jitter, cap at 60s. Task {i}."),
        ("assistant", "Here is the retry wrapper for task {i}: it wraps the fetch client with exponential backoff starting at 250ms, full jitter, a 60 second cap, and it re-raises on the final attempt so callers see the real error. It logs attempt counts at debug level and never swallows KeyboardInterrupt."),
        ("user", "Refactor the parser module so the tokenizer is injectable, and add a golden test. Item {i}."),
        ("assistant", "Done for item {i}: the parser now takes a tokenizer factory, defaulting to the whitespace tokenizer; the golden test pins the current output for the sample corpus and the property test checks idempotence of normalization."),
    ]
    i = 0
    chars = sum(len(r) for r in rows)
    while chars < 14_000:
        role, template = fillers[i % len(fillers)]
        text = template.format(i=i)
        rows.append(rollout_row(role, text))
        chars += len(rows[-1])
        i += 1
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return chars


with TemporaryDirectory() as tmp:
    workspace = Path(tmp) / "ws"
    workspace.mkdir()
    transcript = workspace / "rollout.jsonl"
    total = build_transcript(transcript)
    payload = {
        "prompt": "Generate the docker-compose service block for the ingest service. "
                  "Which port must it expose?",
        "transcript_path": str(transcript),
        "cwd": str(workspace),
    }
    print(f"transcript: ~{total:,} chars (budget is 6,000 -> eligible)")
    started = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(ENTRY)],
        input=json.dumps(payload), capture_output=True, text=True,
        timeout=120, cwd=str(REPO),
    )
    elapsed = time.perf_counter() - started
    print(f"hook exit={proc.returncode} in {elapsed:.1f}s")
    if proc.stderr.strip():
        print("stderr:", proc.stderr.strip()[:400])

    injected_fact = False
    if proc.stdout.strip():
        out = json.loads(proc.stdout)
        ctx = out.get("hookSpecificOutput", {}).get("additionalContext", "")
        injected_fact = "7431" in ctx
        print(f"stdout: hookSpecificOutput present, {len(ctx)} chars, "
              f"route=attention tag={'route=\"attention\"' in ctx}, buried fact included={injected_fact}")
    else:
        print("stdout: EMPTY (no injection)")

    ledger = workspace / ".aioptimizer" / "codex_hook_ledger.jsonl"
    if ledger.is_file():
        rows = [json.loads(l) for l in ledger.read_text(encoding="utf-8").splitlines() if l.strip()]
        r = rows[-1]
        keys = ("route", "route_reason", "injected", "history_chars", "sidecar_ready",
                "sidecar_state", "latency_ms", "error_type")
        print("receipt:", {k: r.get(k) for k in keys if k in r})
        verdict = r.get("route") == "attention" and r.get("injected") and injected_fact
    else:
        print("receipt: NO LEDGER WRITTEN")
        verdict = False

    print(f"\n[{'PASS' if verdict else 'FAIL'}] current build treats an eligible turn end-to-end")

    pid_file = workspace / ".aioptimizer" / "sidecar.pid"
    if pid_file.is_file():
        print(f"(sidecar pid file: {pid_file.read_text().strip()} — lazy start engaged in test ws)")
