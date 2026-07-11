"""Thread-safe exact-response caching for the gateway."""

import copy
import hashlib
import json
import threading
import time
from collections import OrderedDict

from gateway.middleware import ShortCircuit


class ExactCacheMiddleware:
    """Cache responses by a stable hash of the complete request body."""

    def __init__(self, max_entries=256, ttl_seconds=None, clock=time.time):
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.clock = clock
        self._entries = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _key(body):
        serialized = json.dumps(body, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def _is_sampled(body):
        """True when the request expects stochastic output — caching would collapse
        independent samples into one (found dogfooding the v8 self-consistency runs,
        whose 5 identical temp-0.7 prompts must each hit the model)."""
        temperature = body.get("temperature")
        if temperature is None:
            temperature = (body.get("options") or {}).get("temperature")
        return bool(temperature)

    def before_request(self, body):
        if self._is_sampled(body):
            return body
        key = self._key(body)
        now = self.clock()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return body
            created_at, response = entry
            if self.ttl_seconds is not None and now - created_at > self.ttl_seconds:
                del self._entries[key]
                return body
            return ShortCircuit(copy.deepcopy(response))

    def after_response(self, body, response):
        if self._is_sampled(body):
            return response
        key = self._key(body)
        with self._lock:
            self._entries[key] = (self.clock(), copy.deepcopy(response))
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        return response
