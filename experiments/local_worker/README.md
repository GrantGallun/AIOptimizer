# Local Worker Lane

This adapter lets experiments use a local Ollama model without making Brain Runtime
depend on Ollama. It calls the local HTTP endpoint directly and can be replaced by
another provider later.

The first target is `qwen3:8b`, a practical Q4-sized worker for the RTX 5080's 16 GB
of VRAM. Use it for repeatable cheap tests; keep frontier hosted models for the
high-value comparison runs.

## Setup

Install Ollama for Windows from <https://ollama.com/download>, then in PowerShell:

```powershell
ollama pull qwen3:8b
python experiments\local_worker\ollama_client.py diagnose
python experiments\local_worker\ollama_client.py smoke
```

The pull is roughly 5 GB. Keep at least 30 GB free before adding more models or
Hugging Face checkpoints.

## Installed state (2026-07-09)

Ollama 0.31.2 is installed at `%LOCALAPPDATA%\Programs\Ollama\ollama.exe` with `qwen3:8b`
already pulled (~4.9 GB). If `ollama` is not on a fresh shell's PATH, call it by full path or add
it for the session:

```powershell
$env:Path += ";$env:LOCALAPPDATA\Programs\Ollama"
ollama list
```

Hardware note: RTX 5080 has 16 GB VRAM, so Q4 models up to ~14B fit and run on GPU; do not pull
models larger than ~14B-Q4 (a 32B-Q4 spills to system RAM and crawls). Qwen3-8B is the pinned
local regression worker; reserve frontier/hosted models for the one high-value comparison after a
local result holds.
