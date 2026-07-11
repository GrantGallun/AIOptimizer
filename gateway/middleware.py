"""Middleware interfaces for gateway request and response processing."""

from typing import Protocol


class Middleware(Protocol):
    """Transform a request before forwarding and its corresponding response."""

    def before_request(self, body: dict) -> dict:
        """Return the request body to pass to the next middleware."""

    def after_response(self, body: dict, response: dict) -> dict:
        """Return the response body to pass to the previous middleware."""


class PassthroughMiddleware:
    """A middleware that leaves requests and responses unchanged."""

    def before_request(self, body: dict) -> dict:
        return body

    def after_response(self, body: dict, response: dict) -> dict:
        return response
