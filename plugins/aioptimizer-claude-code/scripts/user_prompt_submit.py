"""Claude Code UserPromptSubmit entry point; stdout is reserved for hook JSON."""

import json
import os
from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="strict")

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT / "scripts"))
SOURCE_ROOT = PLUGIN_ROOT.parents[1]
if (SOURCE_ROOT / "aioptimizer" / "__init__.py").is_file():
    sys.path.insert(0, str(SOURCE_ROOT))
configured_home = os.environ.get("AIOPTIMIZER_HOME")
if configured_home and (Path(configured_home) / "aioptimizer" / "__init__.py").is_file():
    sys.path.insert(0, str(Path(configured_home).resolve()))

from claude_code_hook_adapter import append_receipt, process_hook, request_context


def handle_payload(payload, *, environ=None, context_request=request_context):
    """Handle one Claude Code payload, returning no output on every failure."""
    environment = os.environ if environ is None else environ
    endpoint = environment.get(
        "AIOPTIMIZER_CONTEXT_URL", "http://127.0.0.1:8800/optimize/context"
    )
    try:
        budget = int(environment.get("AIOPTIMIZER_CODEX_CONTEXT_CHARS", "6000"))
        timeout = float(environment.get("AIOPTIMIZER_CODEX_TIMEOUT_SECONDS", "30"))

        def optimizer(messages, query, output_budget_chars):
            return context_request(
                messages,
                query,
                output_budget_chars,
                endpoint=endpoint,
                timeout_seconds=timeout,
            )

        return process_hook(payload, optimizer=optimizer, output_budget_chars=budget)
    except Exception as error:
        return None, {
            "route": "error",
            "error_type": type(error).__name__,
            "injected": False,
            "latency_ms": 0.0,
        }


def main() -> None:
    try:
        payload = json.load(sys.stdin)
        if not isinstance(payload, dict):
            return
        output, receipt = handle_payload(payload)
        cwd = payload.get("cwd")
        if not isinstance(cwd, str):
            cwd = os.getcwd()
        try:
            append_receipt(cwd, receipt)
        except OSError:
            pass
        if output is not None:
            print(json.dumps(output, ensure_ascii=False, separators=(",", ":")))
    except Exception:
        return


if __name__ == "__main__":
    main()
