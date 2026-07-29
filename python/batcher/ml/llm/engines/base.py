"""LLM engine adapters — the pluggable ``list[str] -> list[str]`` backends.

An *engine* is the only thing `generate` needs from an LLM: hand it a batch of
requests, get back a string per request in order. Keeping that contract this narrow is
what lets vLLM, an OpenAI-compatible HTTP endpoint, and a deterministic test double be
interchangeable — and what keeps the columnar machinery in `generate` free of any
model library.

An engine may also report, per request and in request order, the token usage and the
finish reason behind each generation. The channel for both is `llm.channels` — a
thread-local, per-call sink the caller opens a scope around, so nothing is shared
between concurrent calls. The older ``engine.last_usage`` attribute still works for
user-written engines and is still set by the engines here, but it is a shared mutable
read and the sink is what `generate` prefers.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

__all__ = ["Engine", "EngineFactory", "unpack_request"]

Engine = Callable[[list[str]], Sequence[str]]
"""Maps a list of prompts to a list of generated strings (one per prompt, in order)."""

EngineFactory = Callable[[], Engine]
"""Builds an `Engine`, called once per worker so the model loads a single time."""


def unpack_request(request: Any, override_keys: Sequence[str]) -> tuple[str, Any, dict]:
    """Split one request into ``(prompt, image, overrides)``.

    Every served-endpoint engine accepts the same two request shapes and has to take them
    apart the same way: a plain string carries only the prompt, while a dict may also carry
    a vision ``image`` and per-row sampling overrides. Only which sampling keys are
    forwarded differs between providers, so that is the parameter. ``adapter`` is never
    forwarded — a served endpoint selects its model by name, not per request.

    Args:
        request: A prompt string, or a dict with ``prompt`` and optional extras.
        override_keys: The sampling keys this provider accepts per row, such as
            ``("max_tokens", "temperature")``. Anything else in the dict is ignored.

    Returns:
        The prompt, the image (or `None`), and the per-row overrides present in `request`.
    """
    if not isinstance(request, dict):
        return str(request), None, {}
    prompt = str(request.get("prompt", ""))
    image = request.get("image")
    overrides = {k: request[k] for k in override_keys if k in request}
    return prompt, image, overrides
