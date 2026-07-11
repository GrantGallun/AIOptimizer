"""Middleware interfaces for gateway request and response processing."""

from typing import Protocol, Union


class ShortCircuit:
    """A response produced by middleware without calling the upstream."""

    def __init__(self, response: dict):
        self.response = response


class Middleware(Protocol):
    """Transform a request before forwarding and its corresponding response."""

    def before_request(self, body: dict) -> Union[dict, ShortCircuit]:
        """Return the next request body or a response that skips the upstream."""

    def after_response(self, body: dict, response: dict) -> dict:
        """Return the response body to pass to the previous middleware."""


class PassthroughMiddleware:
    """A middleware that leaves requests and responses unchanged."""

    def before_request(self, body: dict) -> dict:
        return body

    def after_response(self, body: dict, response: dict) -> dict:
        return response
