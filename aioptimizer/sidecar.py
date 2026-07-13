"""Fail-open lifecycle management for the local AIOptimizer sidecar."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable
import urllib.request


DEFAULT_HEALTH_URL = "http://127.0.0.1:8800/health"
DEFAULT_PORT = 8800
DEFAULT_STARTUP_TIMEOUT_SECONDS = 10.0
DEFAULT_POLL_INTERVAL_SECONDS = 0.05
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)


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
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            if getattr(response, "status", 200) != 200:
                return False
            payload = json.loads(response.read())
        return isinstance(payload, dict) and payload.get("status") == "ok"
    except Exception:
        return False


def process_is_alive(pid: int) -> bool:
    """Check a PID without signaling or taking ownership of the process."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def launch_sidecar(
    paths: SidecarPaths,
    *,
    port: int = DEFAULT_PORT,
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
    kwargs: dict[str, object] = {
        "cwd": str(paths.workspace),
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
    startup_timeout_seconds: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    health_check: Callable[[], bool] | None = None,
    launcher: Callable[[SidecarPaths], object] | None = None,
    pid_is_alive: Callable[[int], bool] = process_is_alive,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> SidecarResult:
    """Ensure one local sidecar is healthy; return failure instead of raising."""
    started = monotonic()
    check = health_check or (lambda: health_ready(health_url))
    start = launcher or (lambda paths: launch_sidecar(paths, port=port))

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

    if _safe_health_check(check):
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
                monotonic=monotonic,
                sleep=sleep,
            )
            return result(ready, "joined_start" if ready else "startup_timeout")

        if _safe_health_check(check):
            return result(True, "healthy_after_lock")

        existing_pid = _read_pid(paths.pid)
        if existing_pid is not None and pid_is_alive(existing_pid):
            ready = _wait_for_health(
                check,
                startup_timeout_seconds,
                poll_interval_seconds,
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
            monotonic=monotonic,
            sleep=sleep,
        )
        return result(ready, "started" if ready else "startup_timeout")
    except Exception as error:
        return result(False, "error", error_type=type(error).__name__)
    finally:
        lock.release()


def _safe_health_check(check: Callable[[], bool]) -> bool:
    try:
        return bool(check())
    except Exception:
        return False


def _wait_for_health(
    check: Callable[[], bool],
    timeout_seconds: float,
    poll_interval_seconds: float,
    *,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
) -> bool:
    deadline = monotonic() + timeout_seconds
    while True:
        if _safe_health_check(check):
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
