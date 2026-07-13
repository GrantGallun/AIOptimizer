"""Provider-neutral helpers for request message content shapes."""

from __future__ import annotations

from typing import Any, Mapping
from urllib.parse import urlsplit


def message_text(message: Mapping[str, Any]) -> str:
    """Return the text represented by an OpenAI or Anthropic message."""
    if not isinstance(message, Mapping):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        block["text"]
        for block in content
        if isinstance(block, Mapping)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    )


def set_message_text(message: Mapping[str, Any], new_text: str) -> dict[str, Any]:
    """Return ``message`` with text replaced while retaining its content shape.

    Anthropic content may contain several text blocks interleaved with tool or image
    blocks. The replacement occupies the first valid text block; later text blocks
    are removed while every non-text block keeps its value and relative position.
    Messages without a supported text container are copied unchanged.
    """
    if not isinstance(message, Mapping):
        raise TypeError("message must be a mapping")
    if not isinstance(new_text, str):
        raise TypeError("new_text must be a string")

    rewritten = dict(message)
    content = message.get("content")
    if isinstance(content, str):
        rewritten["content"] = new_text
        return rewritten
    if not isinstance(content, list):
        return rewritten

    blocks: list[Any] = []
    replaced = False
    for block in content:
        is_text = (
            isinstance(block, Mapping)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        )
        if not is_text:
            blocks.append(block)
        elif not replaced:
            blocks.append({**block, "text": new_text})
            replaced = True
    if replaced:
        rewritten["content"] = blocks
    return rewritten


def detect_shape(body: Mapping[str, Any]) -> str:
    """Detect an OpenAI or Anthropic chat request from endpoint hints and keys."""
    if not isinstance(body, Mapping):
        return "unknown"

    for key in ("endpoint", "path", "request_path", "url"):
        value = body.get(key)
        if not isinstance(value, str):
            continue
        path = urlsplit(value).path.rstrip("/")
        if path.endswith("/v1/messages"):
            return "anthropic"
        if path.endswith("/v1/chat/completions"):
            return "openai"

    messages = body.get("messages")
    if not isinstance(messages, list):
        return "unknown"

    # Anthropic's top-level system prompt and content-block lists are both
    # provider-specific request signals. An empty list alone is not enough.
    if "system" in body or any(
        isinstance(message, Mapping) and isinstance(message.get("content"), list)
        for message in messages
    ):
        return "anthropic"

    if messages and all(
        isinstance(message, Mapping) and isinstance(message.get("content"), str)
        for message in messages
    ):
        return "openai"
    return "unknown"
