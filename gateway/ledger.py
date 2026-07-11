"""Thread-safe JSONL request ledger."""

import json
import threading
import time
from pathlib import Path


class JsonlLedger:
    """Append gateway request measurements to a JSON Lines file."""

    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def record(self, entry: dict) -> None:
        payload = {"ts": time.time(), **entry}
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
