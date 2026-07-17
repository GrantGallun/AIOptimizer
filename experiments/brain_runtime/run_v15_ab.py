#!/usr/bin/env python3
"""Run one paired, scripted v15 build replicate and invoke the frozen scorer.

Each task/arm receives a fresh copy of its arm's byte-identical seed workspace and a
fresh Codex thread. Prompts are replayed as separate turns in that thread. Produced
files are copied into the exact ``<arm-dir>/<task-id>/<expected-file>`` layout that
``score_v15.py`` consumes. The scorer receives only the task fixture path, two
artifact directory paths, and its output path.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aioptimizer.health import read_rows


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASKS = Path(__file__).with_name("v15_build_tasks.json")
DEFAULT_SCORER = Path(__file__).with_name("score_v15.py")
DEFAULT_PAIR_ROOT = Path(r"C:\Code\TalentTrader\v15-ab-v4")
# Nested Codex processes are launched by a bridge Codex that is itself sandboxed to
# this repository.  A control home under the user's protected ``.codex`` directory is
# readable but not writable from that outer workspace-write sandbox, so child startup
# fails before the first turn while creating ``tmp/arg0``.  Keep the isolated homes in
# ignored workspace runtime state instead.
DEFAULT_CONTROL_ROOT = REPO_ROOT / ".aioptimizer" / "codex-controls" / "ab-v15-v4"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "brain_runtime"
ALLOWED_AST_KINDS = {"defines", "imports", "forbid_bare_except"}

AgentRunner = Callable[..., Mapping[str, Any]]
CHECKPOINT_NAME = "runner_checkpoint.jsonl"
CODEX_HOOK_LEDGER_RELATIVE = Path(".aioptimizer/codex_hook_ledger.jsonl")


#  The router's legitimate content-based decisions (HYP-38/HYP-41's five frozen classes plus
# the low-pressure no-op). Any other route string (sidecar_error, invalid_input, an unrecognized
# value) is a DELIVERY failure, not a judgment call, and disqualifies unconditionally.
KNOWN_ROUTES = {"raw", "attention", "below_threshold"}


def check_treatment_integrity(
    ledger_path: Path,
    *,
    window_start: float,
    window_end: float,
    budget_chars: int = 6000,
) -> dict[str, Any]:
    """Evaluate one arm-A task window per Amendment v15.5.

    Amendment v15.3's original `treated/eligible >= 0.95` gate is WRONG: the 2026-07-17
    real-session replay (commit 125c80e) found ~11.5% treated is what HEALTHY production
    traffic looks like (most size-eligible turns are correctly declined by the router's own
    relevance/tail-coverage judgment -- HYP-38/HYP-41's job, not this gate's). Requiring near-
    100% treated conflated "the router is conservative" with "the pipeline is broken". This
    checks DELIVERY instead: did every turn the router decided to treat actually get injected,
    and did the pipeline avoid delivery-layer failures (sidecar errors, unrecognized routes)?
    How OFTEN the router chooses to treat is a routing-correctness question already covered by
    PREREGISTRATION_v14 and the replay work, not something a v15 treatment-integrity gate
    should re-litigate.
    """
    windowed: list[dict[str, Any]] = []
    for row in read_rows(ledger_path):
        timestamp = row.get("ts")
        if (
            isinstance(timestamp, (int, float))
            and not isinstance(timestamp, bool)
            and window_start <= timestamp <= window_end
        ):
            windowed.append(row)

    errors = sum(row.get("route") not in KNOWN_ROUTES for row in windowed)
    eligible_rows = [
        row
        for row in windowed
        if isinstance(row.get("history_chars"), int)
        and row["history_chars"] > budget_chars
    ]
    attention_rows = [row for row in windowed if row.get("route") == "attention"]
    treated = sum(row.get("injected") is True for row in attention_rows)
    delivery_failures = len(attention_rows) - treated
    eligible = len(eligible_rows)
    rate = treated / eligible if eligible else None
    return {
        "eligible": eligible,
        "treated": treated,
        "errors": errors,
        "rate": rate,
        "delivery_failures": delivery_failures,
        "qualified": errors == 0 and delivery_failures == 0,
    }


def _safe_relative_path(raw: str) -> Path:
    path = Path(raw)
    if not raw or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"expected_files entry must be a safe relative path: {raw!r}")
    return path


def validate_tasks(tasks: Any, *, expected_count: int | None = 12) -> list[Mapping[str, Any]]:
    """Validate the frozen v15 fixture shape without changing scoring semantics."""
    if not isinstance(tasks, list):
        raise ValueError("tasks must be a JSON list")
    if expected_count is not None and len(tasks) != expected_count:
        raise ValueError(f"expected exactly {expected_count} tasks, found {len(tasks)}")

    required = {"id", "domain", "prompts", "expected_files", "requirements", "ast_checks"}
    ids: set[str] = set()
    for index, task in enumerate(tasks):
        if not isinstance(task, Mapping) or not required <= set(task):
            missing = sorted(required - set(task) if isinstance(task, Mapping) else required)
            raise ValueError(f"task {index} is missing required keys: {missing}")
        task_id = task["id"]
        if not isinstance(task_id, str) or not task_id or task_id in ids:
            raise ValueError(f"task id must be a unique non-empty string: {task_id!r}")
        ids.add(task_id)
        if not isinstance(task["domain"], str) or not task["domain"]:
            raise ValueError(f"{task_id}: domain must be a non-empty string")

        prompts = task["prompts"]
        if not isinstance(prompts, list):
            raise ValueError(f"{task_id}: prompts must be a list")
        if not 24 <= len(prompts) <= 40:
            raise ValueError(
                f"{task_id}: prompts must contain between 24 and 40 turns, found {len(prompts)}"
            )
        if not all(isinstance(prompt, str) and prompt.strip() for prompt in prompts):
            raise ValueError(f"{task_id}: prompts must contain non-empty strings")
        for prompt_index, prompt in enumerate(prompts):
            if "reminder" in prompt.casefold():
                raise ValueError(
                    f"{task_id}: prompt {prompt_index} contains disallowed reminder text"
                )

        expected_files = task["expected_files"]
        if not isinstance(expected_files, list) or not expected_files:
            raise ValueError(f"{task_id}: expected_files must be a non-empty list")
        for raw in expected_files:
            if not isinstance(raw, str):
                raise ValueError(f"{task_id}: expected_files entries must be strings")
            _safe_relative_path(raw)

        requirements = task["requirements"]
        if not isinstance(requirements, list) or not requirements:
            raise ValueError(f"{task_id}: requirements must be a non-empty list")
        constraint_count = 0
        requirement_ids: set[str] = set()
        for row in requirements:
            if not isinstance(row, Mapping) or not isinstance(row.get("id"), str):
                raise ValueError(f"{task_id}: every requirement needs a string id")
            if row["id"] in requirement_ids:
                raise ValueError(f"{task_id}: duplicate requirement id {row['id']!r}")
            requirement_ids.add(row["id"])
            includes = row.get("must_include", [])
            excludes = row.get("must_exclude", [])
            if not isinstance(includes, list) or not all(isinstance(v, str) and v for v in includes):
                raise ValueError(f"{task_id}/{row['id']}: must_include must be a string list")
            if not isinstance(excludes, list) or not all(isinstance(v, str) and v for v in excludes):
                raise ValueError(f"{task_id}/{row['id']}: must_exclude must be a string list")
            if not includes and not excludes:
                raise ValueError(f"{task_id}/{row['id']}: requirement has no objective check")
            if "is_constraint" in row and not isinstance(row["is_constraint"], bool):
                raise ValueError(f"{task_id}/{row['id']}: is_constraint must be boolean")
            constraint_count += int(bool(row.get("is_constraint")))
        if not constraint_count:
            raise ValueError(f"{task_id}: at least one requirement must be a constraint")

        ast_checks = task["ast_checks"]
        if not isinstance(ast_checks, list) or not ast_checks:
            raise ValueError(f"{task_id}: ast_checks must be a non-empty list")
        for check in ast_checks:
            if (
                not isinstance(check, Mapping)
                or check.get("kind") not in ALLOWED_AST_KINDS
                or not isinstance(check.get("name"), str)
                or not check["name"]
            ):
                raise ValueError(f"{task_id}: invalid AST check {check!r}")
    return tasks


def load_tasks(path: Path, *, expected_count: int | None = 12) -> list[Mapping[str, Any]]:
    return validate_tasks(json.loads(path.read_text(encoding="utf-8")), expected_count=expected_count)


def prepare_fresh_workspace(seed: Path, destination: Path, task: Mapping[str, Any]) -> Path:
    """Copy an arm seed into a never-before-used, path-opaque task workspace."""
    if not seed.is_dir():
        raise FileNotFoundError(f"seed workspace not found: {seed}")
    if destination.exists():
        raise FileExistsError(f"refusing to reuse task workspace: {destination}")
    for raw in task["expected_files"]:
        if (seed / _safe_relative_path(raw)).exists():
            raise ValueError(f"seed workspace already contains scored artifact {raw!r}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(seed, destination)
    return destination


def collect_task_artifacts(
    workspace: Path,
    arm_dir: Path,
    task: Mapping[str, Any],
) -> dict[str, list[str]]:
    """Copy expected files into the frozen scorer's task directory layout."""
    task_dir = arm_dir / task["id"]
    if task_dir.exists():
        raise FileExistsError(f"refusing to overwrite collected artifacts: {task_dir}")
    task_dir.mkdir(parents=True)
    collected: list[str] = []
    missing: list[str] = []
    for raw in task["expected_files"]:
        relative = _safe_relative_path(raw)
        source = workspace / relative
        if source.is_file():
            destination = task_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            collected.append(relative.as_posix())
        else:
            missing.append(relative.as_posix())
    return {"collected": collected, "missing": missing}


def inspect_task_artifacts(
    arm_dir: Path,
    task: Mapping[str, Any],
) -> dict[str, list[str]]:
    """Describe an already-collected task directory without modifying it."""
    task_dir = arm_dir / task["id"]
    if not task_dir.is_dir():
        raise FileNotFoundError(f"collected task directory not found: {task_dir}")
    collected: list[str] = []
    missing: list[str] = []
    for raw in task["expected_files"]:
        relative = _safe_relative_path(raw)
        if (task_dir / relative).is_file():
            collected.append(relative.as_posix())
        else:
            missing.append(relative.as_posix())
    return {"collected": collected, "missing": missing}


def _append_checkpoint(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_checkpoint(path: Path) -> tuple[Mapping[str, Any] | None, dict[tuple[str, str], dict[str, Any]]]:
    header: Mapping[str, Any] | None = None
    records: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.exists():
        return header, records
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid checkpoint line {line_number}: {path}") from exc
        if not isinstance(event, Mapping):
            raise ValueError(f"checkpoint line {line_number} is not an object: {path}")
        if event.get("type") == "header":
            if header is not None:
                raise ValueError(f"checkpoint contains multiple headers: {path}")
            header = event
        elif event.get("type") == "task_complete":
            arm = event.get("arm")
            task_id = event.get("task_id")
            record = event.get("record")
            if arm not in {"A", "B"} or not isinstance(task_id, str) or not isinstance(record, Mapping):
                raise ValueError(f"invalid task checkpoint line {line_number}: {path}")
            key = (arm, task_id)
            if key in records:
                raise ValueError(f"duplicate task checkpoint {arm}/{task_id}: {path}")
            records[key] = dict(record)
        else:
            raise ValueError(f"unknown checkpoint event on line {line_number}: {path}")
    return header, records


def _json_events(stdout: str) -> list[Mapping[str, Any]]:
    events: list[Mapping[str, Any]] = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, Mapping):
            events.append(event)
    return events


def _thread_id(events: Sequence[Mapping[str, Any]]) -> str | None:
    for event in events:
        direct = event.get("thread_id")
        if isinstance(direct, str) and direct:
            return direct
        thread = event.get("thread")
        if isinstance(thread, Mapping) and isinstance(thread.get("id"), str):
            return thread["id"]
    return None


def _usage(events: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    totals = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
    for event in events:
        if event.get("type") != "turn.completed" or not isinstance(event.get("usage"), Mapping):
            continue
        usage = event["usage"]
        for key in totals:
            value = usage.get(key, 0)
            if isinstance(value, int) and not isinstance(value, bool):
                totals[key] += value
    return totals


def _resolve_codex(explicit: str | None) -> str:
    if explicit:
        resolved = shutil.which(explicit)
        if resolved:
            return resolved
        candidate = Path(explicit)
        if candidate.is_file():
            return str(candidate.resolve())
        raise FileNotFoundError(f"Codex executable not found: {explicit}")
    for name in ("codex.exe", "codex.cmd", "codex"):
        resolved = shutil.which(name)
        if resolved:
            return resolved
    raise FileNotFoundError("Codex executable not found on PATH")


def run_codex_prompt_script(
    *,
    workspace: Path,
    codex_home: Path,
    prompts: Sequence[str],
    model: str,
    codex: str | None = None,
    aioptimizer_home: Path = REPO_ROOT,
    prompt_timeout_seconds: float = 1800.0,
    windows_sandbox: str | None = None,
) -> Mapping[str, Any]:
    """Replay prompts as turns in one fresh, isolated Codex thread."""
    executable = _resolve_codex(codex)
    if not codex_home.is_dir():
        raise FileNotFoundError(f"isolated CODEX_HOME not found: {codex_home}")
    environment = os.environ.copy()
    environment["CODEX_HOME"] = str(codex_home.resolve())
    environment["AIOPTIMIZER_HOME"] = str(aioptimizer_home.resolve())
    if windows_sandbox not in {None, "elevated", "unelevated"}:
        raise ValueError(f"unsupported Windows sandbox mode: {windows_sandbox!r}")
    base = [executable]
    if windows_sandbox is not None:
        base.extend(["--config", f'windows.sandbox="{windows_sandbox}"'])
    base.extend([
        "--sandbox", "workspace-write",
        "--cd", str(workspace.resolve()),
        "--model", model,
    ])
    session_id: str | None = None
    elapsed = 0.0
    totals = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
    for turn_index, prompt in enumerate(prompts):
        if turn_index == 0:
            command = [*base, "exec", "--skip-git-repo-check", "--json", "-"]
        else:
            if session_id is None:
                raise RuntimeError("Codex did not report a thread id for the initial prompt")
            command = [
                *base,
                "exec", "resume", "--skip-git-repo-check", "--json", session_id, "-",
            ]
        started = time.monotonic()
        completed = subprocess.run(
            command,
            cwd=workspace,
            env=environment,
            input=prompt,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=prompt_timeout_seconds,
            check=False,
        )
        elapsed += time.monotonic() - started
        events = _json_events(completed.stdout)
        if turn_index == 0:
            session_id = _thread_id(events)
        usage = _usage(events)
        for key in totals:
            totals[key] += usage[key]
        if completed.returncode:
            detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic output"
            raise RuntimeError(f"Codex turn {turn_index + 1} exited {completed.returncode}: {detail}")
    return {
        "thread_id": session_id,
        "turns": len(prompts),
        "elapsed_seconds": round(elapsed, 3),
        **totals,
    }


def _opaque_workspace(root: Path) -> Path:
    return root / uuid.uuid4().hex / "workspace"


def run_paired_tasks(
    tasks: Sequence[Mapping[str, Any]],
    *,
    pair_root: Path,
    control_root: Path,
    run_root: Path,
    agent_runner: AgentRunner = run_codex_prompt_script,
    model: str = "gpt-5.6-sol",
    codex: str | None = None,
    aioptimizer_home: Path = REPO_ROOT,
    windows_sandbox: str | None = None,
    resume: bool = False,
) -> tuple[Path, Path, dict[str, Any]]:
    """Run both arms and return their artifact dirs plus content-free execution telemetry."""
    if run_root.exists() and not resume:
        raise FileExistsError(f"refusing to reuse run directory: {run_root}")
    if resume and not run_root.is_dir():
        raise FileNotFoundError(f"resume run directory not found: {run_root}")
    run_root.mkdir(parents=True, exist_ok=resume)
    artifact_root = run_root / "artifacts"
    workspace_root = run_root / "workspaces"
    arm_specs = {
        "A": (
            pair_root / "arm-a-plugin-on" / "workspace",
            control_root / "arm-a",
            artifact_root / "A",
        ),
        "B": (
            pair_root / "arm-b-plugin-off" / "workspace",
            control_root / "arm-b",
            artifact_root / "B",
        ),
    }
    checkpoint = run_root / CHECKPOINT_NAME
    expected_header = {
        "type": "header",
        "schema_version": 1,
        "model": model,
        "windows_sandbox": windows_sandbox,
        "task_ids": [task["id"] for task in tasks],
        "pair_root": str(pair_root.resolve()),
        "control_root": str(control_root.resolve()),
    }
    header, checkpoint_records = _load_checkpoint(checkpoint)
    if header is None:
        _append_checkpoint(checkpoint, expected_header)
    elif dict(header) != expected_header:
        raise ValueError("resume checkpoint does not match this runner configuration")

    records: dict[str, Any] = {"A": {}, "B": {}}
    for task in tasks:
        for arm, (seed, codex_home, artifact_dir) in arm_specs.items():
            key = (arm, task["id"])
            existing_task_dir = artifact_dir / task["id"]
            if key in checkpoint_records:
                if not existing_task_dir.is_dir():
                    raise FileNotFoundError(
                        f"checkpoint exists but collected artifacts are missing: {arm}/{task['id']}"
                    )
                records[arm][task["id"]] = checkpoint_records[key]
                continue
            if existing_task_dir.is_dir():
                if not resume:
                    raise FileExistsError(f"unexpected collected task directory: {existing_task_dir}")
                inspection = inspect_task_artifacts(artifact_dir, task)
                if inspection["missing"]:
                    raise ValueError(
                        "cannot recover uncheckpointed task with missing expected artifacts: "
                        f"{arm}/{task['id']}"
                    )
                recovered = {
                    "execution": {},
                    "error": None,
                    "recovered_without_telemetry": True,
                    **inspection,
                }
                records[arm][task["id"]] = recovered
                _append_checkpoint(checkpoint, {
                    "type": "task_complete",
                    "arm": arm,
                    "task_id": task["id"],
                    "record": recovered,
                })
                continue
            workspace = prepare_fresh_workspace(seed, _opaque_workspace(workspace_root), task)
            execution: dict[str, Any]
            turn_window_start: float | None = None
            turn_window_end: float | None = None
            try:
                if arm == "A":
                    turn_window_start = time.time()
                try:
                    runner_result = agent_runner(
                        workspace=workspace,
                        codex_home=codex_home,
                        prompts=tuple(task["prompts"]),
                        model=model,
                        codex=codex,
                        aioptimizer_home=aioptimizer_home,
                        windows_sandbox=windows_sandbox,
                    )
                finally:
                    if arm == "A":
                        turn_window_end = time.time()
                execution = dict(runner_result)
                error = None
            except Exception as exc:  # keep the paired run scoreable; missing artifacts fail closed
                execution = {}
                error = f"{type(exc).__name__}: {exc}"
            treatment_integrity: dict[str, Any] | None = None
            if arm == "A":
                assert turn_window_start is not None and turn_window_end is not None
                # The plugin hook writes its own routing receipts workspace-locally
                # (SidecarPaths.for_workspace(workspace)'s runtime dir), NOT to the
                # gateway's own request ledger and NOT under the shared aioptimizer_home
                # -- each task gets a fresh, disposable workspace, so this is the only
                # ledger that reflects THIS task's actual hook invocations.
                try:
                    treatment_integrity = check_treatment_integrity(
                        workspace / CODEX_HOOK_LEDGER_RELATIVE,
                        window_start=turn_window_start,
                        window_end=turn_window_end,
                    )
                except FileNotFoundError:
                    # A missing ledger is a STRONGER signal than eligible==0 (which means the
                    # hook fired and correctly found nothing to do): it means the hook never
                    # wrote a single row for this workspace's entire run. Record it as a clear
                    # disqualification, not a crash that takes the whole multi-task run down
                    # with it (2026-07-17: exactly this crash killed all 4 tasks of the v15.6
                    # exploratory wave over one workspace's missing ledger).
                    treatment_integrity = {
                        "eligible": 0, "treated": 0, "errors": 0, "rate": None,
                        "delivery_failures": 0, "qualified": False, "ledger_missing": True,
                    }
                    if error is None:
                        error = (
                            "treatment_integrity: codex_hook_ledger.jsonl never appeared "
                            "for this workspace"
                        )
            collection = collect_task_artifacts(workspace, artifact_dir, task)
            record = {
                "execution": execution,
                "error": error,
                **collection,
            }
            if arm == "A":
                record.update({
                    "turn_window_start": turn_window_start,
                    "turn_window_end": turn_window_end,
                    "treatment_integrity": treatment_integrity,
                })
            records[arm][task["id"]] = record
            _append_checkpoint(checkpoint, {
                "type": "task_complete",
                "arm": arm,
                "task_id": task["id"],
                "record": record,
            })
    return arm_specs["A"][2], arm_specs["B"][2], records


def run_frozen_scorer(
    *,
    scorer: Path,
    tasks_path: Path,
    arm_a: Path,
    arm_b: Path,
    out: Path,
) -> tuple[dict[str, Any], str]:
    """Invoke the frozen scorer with no arm metadata beyond its two directory paths."""
    command = [
        sys.executable,
        str(scorer.resolve()),
        "--tasks", str(tasks_path.resolve()),
        "--arm-a", str(arm_a.resolve()),
        "--arm-b", str(arm_b.resolve()),
        "--out", str(out.resolve()),
    ]
    completed = subprocess.run(command, text=True, encoding="utf-8", errors="replace", capture_output=True)
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "scorer failed")
    return json.loads(out.read_text(encoding="utf-8")), completed.stdout.strip()


def next_versioned_result(results_root: Path) -> Path:
    for version in range(1, 10000):
        candidate = results_root / f"v15_ab_result_v{version}.json"
        if not candidate.exists():
            return candidate
    raise RuntimeError("no free v15 result version below 10000")


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(data)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite versioned result: {path}") from exc


def _telemetry_totals(records: Mapping[str, Any]) -> dict[str, Any]:
    totals: dict[str, Any] = {}
    for arm in ("A", "B"):
        rows = records[arm].values()
        totals[arm] = {
            "runs": len(records[arm]),
            "errors": sum(row["error"] is not None for row in rows),
            "recovered_without_telemetry": sum(
                bool(row.get("recovered_without_telemetry")) for row in records[arm].values()
            ),
            "elapsed_seconds": round(sum(row["execution"].get("elapsed_seconds", 0.0) for row in records[arm].values()), 3),
            "input_tokens": sum(row["execution"].get("input_tokens", 0) for row in records[arm].values()),
            "cached_input_tokens": sum(row["execution"].get("cached_input_tokens", 0) for row in records[arm].values()),
            "output_tokens": sum(row["execution"].get("output_tokens", 0) for row in records[arm].values()),
        }
    return totals


def print_treatment_integrity_summary(
    records: Mapping[str, Any], *, expected_count: int
) -> None:
    """Print arm-A treatment integrity without altering frozen scorer input."""
    qualified: list[str] = []
    inconclusive: list[str] = []
    disqualified: list[str] = []
    for task_id, record in records["A"].items():
        integrity = record["treatment_integrity"]
        rate = integrity["rate"]
        rate_text = "n/a" if rate is None else f"{rate:.3f}"
        print(
            f"treatment-integrity {task_id}: eligible={integrity['eligible']} "
            f"treated={integrity['treated']} rate={rate_text} "
            f"errors={integrity['errors']} delivery_failures={integrity['delivery_failures']} "
            f"qualified={integrity['qualified']}"
        )
        if not integrity["qualified"]:
            disqualified.append(task_id)
        elif integrity["eligible"] == 0:
            inconclusive.append(task_id)
        else:
            qualified.append(task_id)
    print(
        f"{len(qualified)}/{expected_count} arm-A tasks qualified, "
        f"{len(inconclusive)}/{expected_count} inconclusive-by-design (eligible==0), "
        f"{len(disqualified)}/{expected_count} DISQUALIFIED"
    )
    if disqualified:
        print(f"STOP: disqualified arm-A task ids: {', '.join(disqualified)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument(
        "--expected-count", type=int, default=12,
        help="Task count validate_tasks() requires. Lower this only for a dev-sanity pilot "
             "against a deliberately smaller --tasks fixture; the frozen confirmatory run must "
             "use the default 12.",
    )
    parser.add_argument("--scorer", type=Path, default=DEFAULT_SCORER)
    parser.add_argument("--pair-root", type=Path, default=DEFAULT_PAIR_ROOT)
    parser.add_argument("--control-root", type=Path, default=DEFAULT_CONTROL_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--codex")
    parser.add_argument("--windows-sandbox", choices=("elevated", "unelevated"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    tasks_path = args.tasks.resolve()
    tasks = load_tasks(tasks_path, expected_count=args.expected_count)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = args.run_root or (args.results_root / "v15_ab_runs" / f"run_{stamp}_{uuid.uuid4().hex[:8]}")
    out = args.out or next_versioned_result(args.results_root)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite result: {out}")

    arm_a, arm_b, records = run_paired_tasks(
        tasks,
        pair_root=args.pair_root.resolve(),
        control_root=args.control_root.resolve(),
        run_root=run_root.resolve(),
        model=args.model,
        codex=args.codex,
        windows_sandbox=args.windows_sandbox,
        resume=args.resume,
    )
    print_treatment_integrity_summary(records, expected_count=len(tasks))
    scorer_out = run_root / "frozen_score.json"
    scored, scorer_summary = run_frozen_scorer(
        scorer=args.scorer,
        tasks_path=tasks_path,
        arm_a=arm_a,
        arm_b=arm_b,
        out=scorer_out,
    )
    scored["runner"] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "windows_sandbox": args.windows_sandbox,
        "task_fixture": str(tasks_path),
        "run_root": str(run_root.resolve()),
        "telemetry": records,
        "totals": _telemetry_totals(records),
    }
    _write_new_json(out.resolve(), scored)
    if scorer_summary:
        print(scorer_summary)
    totals = scored["runner"]["totals"]
    print(
        f"telemetry A {totals['A']['input_tokens'] + totals['A']['output_tokens']} tokens/"
        f"{totals['A']['elapsed_seconds']:.3f}s; B "
        f"{totals['B']['input_tokens'] + totals['B']['output_tokens']} tokens/"
        f"{totals['B']['elapsed_seconds']:.3f}s"
    )
    print(f"result {out.resolve()}")


if __name__ == "__main__":
    main()
