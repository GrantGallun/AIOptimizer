"""Thread-safe exact-response caching for the gateway."""

import copy
import hashlib
import json
import threading
import time
from collections import OrderedDict

from .middleware import ShortCircuit

_MISSING = object()


def _validated_sampling_default(value):
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or float(value) < 0.0
    ):
        raise ValueError("upstream_sampling_default must be a non-negative number")
    return float(value)


class ExactCacheMiddleware:
    """Cache responses by a stable hash of the complete request body."""

    def __init__(
        self,
        max_entries=256,
        ttl_seconds=None,
        clock=time.time,
        upstream_sampling_default=0.8,
    ):
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.clock = clock
        self.upstream_sampling_default = _validated_sampling_default(
            upstream_sampling_default
        )
        self._entries = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _key(body):
        serialized = json.dumps(body, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def _is_sampled(body, upstream_sampling_default=0.8):
        """Return whether caching could collapse independent samples.

        Sampledness cannot be inferred from the request body when temperature is
        omitted because the provider's default is invisible there. Fail closed by
        using the operator-declared upstream default; streaming always bypasses.
        """
        if body.get("stream") is True:
            return True
        temperature = body.get("temperature", _MISSING)
        if temperature is _MISSING:
            temperature = (body.get("options") or {}).get("temperature", _MISSING)
        if temperature is _MISSING:
            temperature = upstream_sampling_default
        return float(temperature) != 0.0

    def before_request(self, body):
        if self._is_sampled(body, self.upstream_sampling_default):
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
        if self._is_sampled(body, self.upstream_sampling_default):
            return response
        key = self._key(body)
        with self._lock:
            self._entries[key] = (self.clock(), copy.deepcopy(response))
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        return response

    def status_metadata(self):
        with self._lock:
            return {
                "entries": len(self._entries),
                "max_entries": self.max_entries,
                "ttl_seconds": self.ttl_seconds,
                "sampled_requests_bypass": True,
                "upstream_sampling_default": self.upstream_sampling_default,
                "streaming_requests_bypass": True,
            }
