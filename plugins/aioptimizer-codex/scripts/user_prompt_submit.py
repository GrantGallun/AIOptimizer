"""Codex UserPromptSubmit hook entry point; stdout is reserved for hook JSON."""

import json
import os
from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="strict")

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT / "scripts"))

from codex_hook_adapter import append_receipt, process_hook, request_context


def discover_aioptimizer_home(workspace, *, environ=None):
    """Find a source checkout without requiring a global package installation."""
    environment = os.environ if environ is None else environ
    candidates = []
    configured = environment.get("AIOPTIMIZER_HOME")
    if configured:
        candidates.append(Path(configured))
    root = Path(workspace).resolve()
    candidates.extend((root, root.parent / "AIOptimizer", PLUGIN_ROOT.parents[1]))
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if (resolved / "aioptimizer" / "__init__.py").is_file():
            return resolved
    return None


def handle_payload(payload, *, environ=None, sidecar_ensurer=None, context_request=request_context):
    """Start the local compiler if needed, then handle one hook payload fail-open."""
    environment = os.environ if environ is None else environ
    endpoint = environment.get(
        "AIOPTIMIZER_CONTEXT_URL", "http://127.0.0.1:8800/optimize/context"
    )
    health_url = environment.get(
        "AIOPTIMIZER_HEALTH_URL", "http://127.0.0.1:8800/health"
    )
    workspace = payload.get("cwd")
    if not isinstance(workspace, str):
        workspace = os.getcwd()

    try:
        source_root = discover_aioptimizer_home(workspace, environ=environment)
        if source_root is not None and str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        if sidecar_ensurer is None:
            from aioptimizer.sidecar import ensure_sidecar

            sidecar_ensurer = ensure_sidecar
        sidecar = sidecar_ensurer(
            workspace,
            health_url=health_url,
            port=int(environment.get("AIOPTIMIZER_SIDECAR_PORT", "8800")),
            source_root=source_root,
            startup_timeout_seconds=float(
                environment.get("AIOPTIMIZER_SIDECAR_STARTUP_SECONDS", "10")
            ),
        )
        sidecar_receipt = sidecar.receipt()
        if not sidecar.ready:
            return None, {"route": "sidecar_error", "injected": False, **sidecar_receipt}
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

        output, receipt = process_hook(payload, optimizer=optimizer, output_budget_chars=budget)
        receipt.update(sidecar_receipt)
        return output, receipt
    except Exception as error:
        return None, {
            "route": "sidecar_error",
            "injected": False,
            "sidecar_ready": False,
            "sidecar_state": "error",
            "sidecar_error_type": type(error).__name__,
        }


def main() -> None:
    payload = json.load(sys.stdin)
    output, receipt = handle_payload(payload)
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
