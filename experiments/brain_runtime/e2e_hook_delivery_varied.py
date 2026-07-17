"""Variant of the e2e test with NON-repetitive filler: every turn gets distinct vocabulary
(deterministic word salad from a seeded RNG), so the repetition detector should not fire.
Compares the two shapes to isolate whether covered_by_recent_tail is a repetition false
negative or a broader routing failure."""
import json
import random
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

WORDS = (
    "ledger gateway compaction receipt threshold packer verdict harness sidecar telemetry "
    "quorum retries jitter backoff parser tokenizer corpus golden idempotent normalization "
    "docker compose ingest scraper crawler robots allowlist sqlite parameterized envelope "
    "contract wilson interval seed frozen registry episode flywheel coverage integrity "
    "attention chronology cliff budget dividend router escalation disagreement provider "
    "prefix cache mutation shadow judge encoder embedding cosine cluster pinned records"
).split()


def rollout_row(role, text):
    return json.dumps({
        "type": "response_item",
        "payload": {"type": "message", "role": role,
                    "content": [{"type": "output_text", "text": text}]},
    })


def build_varied_transcript(path: Path) -> int:
    rng = random.Random(41)
    rows = [
        rollout_row("user", "Let's set up the deployment config for the ingest service."),
        rollout_row("assistant",
                    "Noted. The verified deployment port for the ingest service is 7431. "
                    "I will use that in every config we generate from here on."),
    ]
    chars = sum(len(r) for r in rows)
    turn = 0
    while chars < 14_000:
        role = "user" if turn % 2 == 0 else "assistant"
        sentence = " ".join(rng.sample(WORDS, k=14)).capitalize() + \
            f". Then we adjusted {rng.choice(WORDS)} against {rng.choice(WORDS)} " \
            f"and confirmed {rng.randrange(10_000, 99_999)} rows passed."
        rows.append(rollout_row(role, sentence))
        chars += len(rows[-1])
        turn += 1
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return chars


with TemporaryDirectory() as tmp:
    ws = Path(tmp) / "ws"
    ws.mkdir()
    transcript = ws / "rollout.jsonl"
    total = build_varied_transcript(transcript)
    payload = {
        "prompt": "Generate the docker-compose service block for the ingest service. "
                  "Which port must it expose?",
        "transcript_path": str(transcript),
        "cwd": str(ws),
    }
    print(f"varied transcript: ~{total:,} chars")
    proc = subprocess.run([sys.executable, str(ENTRY)], input=json.dumps(payload),
                          capture_output=True, text=True, timeout=120, cwd=str(REPO))
    injected_fact = False
    if proc.stdout.strip():
        ctx = json.loads(proc.stdout).get("hookSpecificOutput", {}).get("additionalContext", "")
        injected_fact = "7431" in ctx
        print(f"stdout: injection present, {len(ctx)} chars, fact included={injected_fact}")
    else:
        print("stdout: EMPTY (no injection)")
    ledger = ws / ".aioptimizer" / "codex_hook_ledger.jsonl"
    rows = [json.loads(l) for l in ledger.read_text(encoding="utf-8").splitlines() if l.strip()]
    r = rows[-1]
    print("receipt:", {k: r.get(k) for k in
                       ("route", "route_reason", "injected", "history_chars",
                        "load_compression_ratio", "load_duplicate_turn_ratio",
                        "load_unique_token_ratio") if k in r})
    print(f"\n[{'PASS' if (r.get('route') == 'attention' and injected_fact) else 'FAIL'}] "
          f"varied-filler eligible turn is treated")
