"""Fail-open lifecycle management for the local AIOptimizer sidecar."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Callable, Mapping
import urllib.request


DEFAULT_HEALTH_URL = "http://127.0.0.1:8800/health"
DEFAULT_PORT = 8800
DEFAULT_STARTUP_TIMEOUT_SECONDS = 10.0
DEFAULT_POLL_INTERVAL_SECONDS = 0.05
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)


def compute_build_stamp(package_dir: str | Path | None = None) -> str:
    """Return a cheap, deterministic stamp for the Python package tree."""
    root = (
        Path(package_dir).resolve()
        if package_dir is not None
        else Path(__file__).resolve().parent
    )
    records: list[tuple[str, int, int]] = []
    for path in root.rglob("*.py"):
        stat = path.stat()
        records.append((path.relative_to(root).as_posix(), stat.st_size, stat.st_mtime_ns))
    digest = hashlib.sha256()
    for relative_path, size, mtime_ns in sorted(records):
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(mtime_ns).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True)
class SidecarPaths:
    """Workspace-local runtime paths used by the sidecar."""

    workspace: Path
    runtime_dir: Path
    lock: Path
    pid: Path
    log: Path
    ledger: Path

    @classmethod
    def for_workspace(cls, workspace: str | Path) -> "SidecarPaths":
        root = Path(workspace).resolve()
        runtime = root / ".aioptimizer"
        return cls(
            workspace=root,
            runtime_dir=runtime,
            lock=runtime / "sidecar.lock",
            pid=runtime / "sidecar.pid",
            log=runtime / "sidecar.log",
            ledger=runtime / "gateway_ledger.jsonl",
        )


@dataclass(frozen=True)
class SidecarResult:
    """Content-free outcome safe to merge into a hook receipt."""

    ready: bool
    state: str
    pid_reused: bool = False
    error_type: str | None = None
    latency_ms: float = 0.0

    def receipt(self) -> dict[str, object]:
        receipt: dict[str, object] = {
            "sidecar_ready": self.ready,
            "sidecar_state": self.state,
            "sidecar_pid_reused": self.pid_reused,
            "sidecar_latency_ms": self.latency_ms,
        }
        if self.error_type is not None:
            receipt["sidecar_error_type"] = self.error_type
        return receipt


def health_ready(
    url: str = DEFAULT_HEALTH_URL,
    *,
    timeout_seconds: float = 0.25,
) -> bool:
    """Return whether the local health endpoint reports ready, without raising."""
    return _health_is_ready(_health_payload(url, timeout_seconds=timeout_seconds))


def _health_payload(
    url: str,
    *,
    timeout_seconds: float,
) -> dict[str, object] | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            if getattr(response, "status", 200) != 200:
                return None
            payload = json.loads(response.read())
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def process_is_alive(pid: int) -> bool:
    """Check a PID without signaling or taking ownership of the process."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def terminate_process(pid: int) -> None:
    """Request termination of the process recorded in the workspace PID file."""
    os.kill(pid, signal.SIGTERM)


def launch_sidecar(
    paths: SidecarPaths,
    *,
    port: int = DEFAULT_PORT,
    source_root: str | Path | None = None,
    python_executable: str = sys.executable,
    platform_name: str | None = None,
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
) -> subprocess.Popen:
    """Launch a detached attention-enabled gateway with workspace-local artifacts."""
    paths.runtime_dir.mkdir(parents=True, exist_ok=True)
    command = [
        python_executable,
        "-m",
        "aioptimizer",
        "--port",
        str(port),
        "--attention-context",
        "--shadow-rate",
        "0",
        "--ledger",
        str(paths.ledger),
    ]
    environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
    launch_root = Path(source_root).resolve() if source_root is not None else paths.workspace
    kwargs: dict[str, object] = {
        "cwd": str(launch_root),
        "stdin": subprocess.DEVNULL,
        "stderr": subprocess.STDOUT,
        "env": environment,
        "close_fds": True,
    }
    if (platform_name or os.name) == "nt":
        kwargs["creationflags"] = CREATE_NO_WINDOW
    else:
        kwargs["start_new_session"] = True

    with paths.log.open("ab") as log:
        kwargs["stdout"] = log
        return popen(command, **kwargs)


def ensure_sidecar(
    workspace: str | Path,
    *,
    health_url: str = DEFAULT_HEALTH_URL,
    port: int = DEFAULT_PORT,
    source_root: str | Path | None = None,
    startup_timeout_seconds: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    health_check: Callable[[], object] | None = None,
    launcher: Callable[[SidecarPaths], object] | None = None,
    pid_is_alive: Callable[[int], bool] = process_is_alive,
    process_terminator: Callable[[int], None] = terminate_process,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> SidecarResult:
    """Ensure one local sidecar is healthy; return failure instead of raising."""
    started = monotonic()
    local_build_stamp = compute_build_stamp()
    check: Callable[[], object] = health_check or (
        lambda: _health_payload(health_url, timeout_seconds=0.25)
    )
    start = launcher or (
        lambda paths: launch_sidecar(paths, port=port, source_root=source_root)
    )

    def result(
        ready: bool,
        state: str,
        *,
        pid_reused: bool = False,
        error_type: str | None = None,
    ) -> SidecarResult:
        return SidecarResult(
            ready=ready,
            state=state,
            pid_reused=pid_reused,
            error_type=error_type,
            latency_ms=round((monotonic() - started) * 1000, 3),
        )

    observation = _safe_health_check(check)
    if _health_matches_build(observation, local_build_stamp):
        return result(True, "healthy")
    if startup_timeout_seconds < 0 or poll_interval_seconds <= 0:
        return result(False, "invalid_configuration", error_type="ValueError")

    paths = SidecarPaths.for_workspace(workspace)
    lock = _StartLock(paths.lock)
    try:
        paths.runtime_dir.mkdir(parents=True, exist_ok=True)
        acquired = lock.acquire(pid_is_alive=pid_is_alive)
        if not acquired:
            ready = _wait_for_health(
                check,
                startup_timeout_seconds,
                poll_interval_seconds,
                expected_build=local_build_stamp,
                monotonic=monotonic,
                sleep=sleep,
            )
            return result(ready, "joined_start" if ready else "startup_timeout")

        observation = _safe_health_check(check)
        if _health_matches_build(observation, local_build_stamp):
            return result(True, "healthy_after_lock")

        if _health_is_ready(observation):
            existing_pid = _read_pid(paths.pid)
            if existing_pid is None or not pid_is_alive(existing_pid):
                return result(False, "stale_code")
            try:
                process_terminator(existing_pid)
            except Exception as error:
                return result(False, "stale_code", error_type=type(error).__name__)
            port_freed = _wait_for_port_free(
                check,
                startup_timeout_seconds,
                poll_interval_seconds,
                monotonic=monotonic,
                sleep=sleep,
            )
            if not port_freed:
                return result(False, "stale_code", error_type="TimeoutError")
            paths.pid.unlink(missing_ok=True)

        existing_pid = _read_pid(paths.pid)
        if existing_pid is not None and pid_is_alive(existing_pid):
            ready = _wait_for_health(
                check,
                startup_timeout_seconds,
                poll_interval_seconds,
                expected_build=local_build_stamp,
                monotonic=monotonic,
                sleep=sleep,
            )
            return result(
                ready,
                "pid_reused" if ready else "startup_timeout",
                pid_reused=True,
            )

        if existing_pid is not None:
            paths.pid.unlink(missing_ok=True)
        process = start(paths)
        pid = getattr(process, "pid", process)
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            raise ValueError("sidecar launcher did not return a positive PID")
        _write_pid(paths.pid, pid)
        ready = _wait_for_health(
            check,
            startup_timeout_seconds,
            poll_interval_seconds,
            expected_build=local_build_stamp,
            monotonic=monotonic,
            sleep=sleep,
        )
        return result(ready, "started" if ready else "startup_timeout")
    except Exception as error:
        return result(False, "error", error_type=type(error).__name__)
    finally:
        lock.release()


def _safe_health_check(check: Callable[[], object]) -> object:
    try:
        return check()
    except Exception:
        return False


def _health_is_ready(observation: object) -> bool:
    if isinstance(observation, Mapping):
        return observation.get("ready") is True or observation.get("status") == "ok"
    return bool(observation)


def _health_matches_build(observation: object, expected_build: str) -> bool:
    if not _health_is_ready(observation):
        return False
    if isinstance(observation, Mapping):
        return observation.get("build") == expected_build
    # Boolean fakes predate stamped health payloads and remain useful for lifecycle tests.
    return True


def _wait_for_health(
    check: Callable[[], object],
    timeout_seconds: float,
    poll_interval_seconds: float,
    *,
    expected_build: str,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
) -> bool:
    deadline = monotonic() + timeout_seconds
    while True:
        if _health_matches_build(_safe_health_check(check), expected_build):
            return True
        remaining = deadline - monotonic()
        if remaining <= 0:
            return False
        sleep(min(poll_interval_seconds, remaining))


def _wait_for_port_free(
    check: Callable[[], object],
    timeout_seconds: float,
    poll_interval_seconds: float,
    *,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
) -> bool:
    deadline = monotonic() + timeout_seconds
    while True:
        if not _health_is_ready(_safe_health_check(check)):
            return True
        remaining = deadline - monotonic()
        if remaining <= 0:
            return False
        sleep(min(poll_interval_seconds, remaining))


def _read_pid(path: Path) -> int | None:
    try:
        value = int(path.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return None
    return value if value > 0 else None


def _write_pid(path: Path, pid: int) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(f"{pid}\n", encoding="ascii")
    os.replace(temporary, path)


class _StartLock:
    """Small cross-process lock based on exclusive file creation."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._descriptor: int | None = None

    def acquire(self, *, pid_is_alive: Callable[[int], bool]) -> bool:
        for _ in range(2):
            try:
                descriptor = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
            except FileExistsError:
                owner = _read_pid(self.path)
                if owner is not None and not pid_is_alive(owner):
                    try:
                        self.path.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                return False
            try:
                os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
            except Exception:
                os.close(descriptor)
                self.path.unlink(missing_ok=True)
                raise
            self._descriptor = descriptor
            return True
        return False

    def release(self) -> None:
        if self._descriptor is None:
            return
        os.close(self._descriptor)
        self._descriptor = None
        if _read_pid(self.path) == os.getpid():
            self.path.unlink(missing_ok=True)
