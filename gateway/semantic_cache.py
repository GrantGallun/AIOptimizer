"""Thread-safe semantic response caching for the gateway."""

from __future__ import annotations

import copy
import threading
import time
from collections import OrderedDict
from typing import Any, Callable

from gateway.cache_middleware import ExactCacheMiddleware
from gateway.middleware import ShortCircuit
from gateway.receipts import _as_vector, _cosine

EmbedFn = Callable[[list[str]], Any]


def _request_text(body: dict) -> str | None:
    """Return the last user prompt from OpenAI or Ollama request shapes."""
    messages = body.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                return content

    prompt = body.get("prompt")
    return prompt if isinstance(prompt, str) else None


def _default_embed(texts: list[str]) -> Any:
    """Load the research encoder only when semantic caching is actually used."""
    from agent_bus.context import DEFAULT_ENCODER, _encoder_embed

    return _encoder_embed(texts, model_name=DEFAULT_ENCODER)


class SemanticCacheMiddleware:
    """Replay the response to the most similar cached user request."""

    def __init__(
        self,
        threshold=0.95,
        max_entries=256,
        embed_fn=None,
        clock=time.time,
        ttl_seconds=None,
    ):
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self.threshold = threshold
        self.max_entries = max_entries
        self._embed_fn: EmbedFn | None = embed_fn
        self.clock = clock
        self.ttl_seconds = ttl_seconds
        self._entries = OrderedDict()
        self._lock = threading.Lock()

    def _embed(self, text: str) -> list[float]:
        embed = self._embed_fn or _default_embed
        values = embed([text])
        return _as_vector(values[0])

    def before_request(self, body):
        if ExactCacheMiddleware._is_sampled(body):
            return body
        text = _request_text(body)
        if text is None:
            return body

        embedding = self._embed(text)
        now = self.clock()
        with self._lock:
            expired = []
            best_key = None
            best_similarity = float("-inf")
            for key, (created_at, stored_embedding, _response) in self._entries.items():
                if self.ttl_seconds is not None and now - created_at > self.ttl_seconds:
                    expired.append(key)
                    continue
                similarity = _cosine(embedding, stored_embedding)
                if similarity > best_similarity:
                    best_key = key
                    best_similarity = similarity

            for key in expired:
                del self._entries[key]
            if best_key is None or best_similarity < self.threshold:
                return body

            _created_at, _stored_embedding, response = self._entries[best_key]
            self._entries.move_to_end(best_key)
            return ShortCircuit(copy.deepcopy(response))

    def after_response(self, body, response):
        if ExactCacheMiddleware._is_sampled(body):
            return response
        text = _request_text(body)
        if text is None:
            return response

        embedding = self._embed(text)
        with self._lock:
            self._entries[text] = (self.clock(), embedding, copy.deepcopy(response))
            self._entries.move_to_end(text)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        return response
