#!/usr/bin/env python3
"""Small dependency-free client for a local Ollama worker.

The adapter intentionally targets Ollama's local HTTP API instead of importing a
vendor SDK. That keeps Brain Runtime model-agnostic and makes failures (missing
service, missing model, low disk space) explicit in experiment output.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


DEFAULT_ENDPOINT = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3:8b"


@dataclass(frozen=True)
class LocalWorkerStatus:
    endpoint: str
    reachable: bool
    installed_models: tuple[str, ...]
    recommended_model_ready: bool
    free_disk_gb: float
    message: str


@dataclass(frozen=True)
class Generation:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_duration_ns: int
    load_duration_ns: int
    prompt_eval_duration_ns: int = 0


class OllamaClient:
    """Ollama-compatible client, routed through ``AIOPT_GATEWAY`` by default.

    Passing ``endpoint`` explicitly takes precedence over the environment.
    """

    def __init__(self, endpoint: str | None = None, timeout_seconds: float = 30.0) -> None:
        resolved_endpoint = endpoint if endpoint is not None else os.environ.get("AIOPT_GATEWAY", DEFAULT_ENDPOINT)
        self.endpoint = resolved_endpoint.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def list_models(self) -> list[dict[str, Any]]:
        payload = self._request("/api/tags")
        models = payload.get("models", [])
        if not isinstance(models, list):
            raise RuntimeError("Ollama returned an invalid model list.")
        return models

    def generate(
        self,
        prompt: str,
        *,
        model: str = DEFAULT_MODEL,
        system: str | None = None,
        format: dict[str, Any] | str | None = None,
    ) -> str:
        return self.generate_with_metrics(prompt, model=model, system=system, format=format).text

    def generate_with_metrics(
        self,
        prompt: str,
        *,
        model: str = DEFAULT_MODEL,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 24,
        format: dict[str, Any] | str | None = None,
        num_ctx: int | None = None,
    ) -> Generation:
        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            # Keep the model resident between calls: 32.3% of all measured model time across
            # 61 arm-runs was weight (re)loading (r/LocalLLaMA-prompted audit, 2026-07-11).
            "keep_alive": "60m",
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if num_ctx is not None:
            # Ollama's default window (4096) silently truncates long prompts; experiments
            # that feed full transcripts must set this explicitly (PREREGISTRATION_v18).
            body["options"]["num_ctx"] = num_ctx
        if system:
            body["system"] = system
        if format is not None:
            body["format"] = format
        payload = self._request("/api/generate", body)
        response = payload.get("response")
        if not isinstance(response, str):
            raise RuntimeError("Ollama returned no text response.")
        return Generation(
            text=response,
            prompt_tokens=int(payload.get("prompt_eval_count", 0)),
            completion_tokens=int(payload.get("eval_count", 0)),
            total_duration_ns=int(payload.get("total_duration", 0)),
            load_duration_ns=int(payload.get("load_duration", 0)),
            prompt_eval_duration_ns=int(payload.get("prompt_eval_duration", 0)),
        )

    def _request(self, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            f"{self.endpoint}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if body is not None else "GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach Ollama at {self.endpoint}: {exc.reason}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Ollama returned an invalid response.")
        return payload


def diagnose(endpoint: str = DEFAULT_ENDPOINT, model: str = DEFAULT_MODEL) -> LocalWorkerStatus:
    free_disk_gb = shutil.disk_usage(".").free / (1024**3)
    try:
        models = OllamaClient(endpoint).list_models()
    except RuntimeError as exc:
        return LocalWorkerStatus(endpoint, False, (), False, free_disk_gb, str(exc))

    names = tuple(str(item.get("name", "")) for item in models if item.get("name"))
    ready = model in names
    message = "Local worker is ready." if ready else f"Ollama is running; pull '{model}' before the first run."
    return LocalWorkerStatus(endpoint, True, names, ready, free_disk_gb, message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["diagnose", "smoke"], help="Diagnostic or one local generation.")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default="Reply with exactly: local worker ready")
    args = parser.parse_args()

    if args.command == "diagnose":
        print(json.dumps(diagnose(args.endpoint, args.model).__dict__, indent=2, sort_keys=True))
        return

    print(OllamaClient(args.endpoint, timeout_seconds=180.0).generate(args.prompt, model=args.model))


if __name__ == "__main__":
    main()
