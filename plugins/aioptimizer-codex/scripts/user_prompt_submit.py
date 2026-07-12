"""Codex UserPromptSubmit hook entry point; stdout is reserved for hook JSON."""

import json
import os
from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="strict")

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from aioptimizer.codex_hook import append_receipt, process_hook, request_context


def main() -> None:
    payload = json.load(sys.stdin)
    endpoint = os.environ.get("AIOPTIMIZER_CONTEXT_URL", "http://127.0.0.1:8800/optimize/context")
    budget = int(os.environ.get("AIOPTIMIZER_CODEX_CONTEXT_CHARS", "6000"))

    def optimizer(messages, query, output_budget_chars):
        return request_context(
            messages,
            query,
            output_budget_chars,
            endpoint=endpoint,
        )

    output, receipt = process_hook(payload, optimizer=optimizer, output_budget_chars=budget)
    cwd = payload.get("cwd")
    if isinstance(cwd, str):
        try:
            append_receipt(cwd, receipt)
        except OSError:
            pass
    if output is not None:
        print(json.dumps(output, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
