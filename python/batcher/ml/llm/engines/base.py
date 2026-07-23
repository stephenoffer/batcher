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

__all__ = ["Engine", "EngineFactory"]

Engine = Callable[[list[str]], Sequence[str]]
"""Maps a list of prompts to a list of generated strings (one per prompt, in order)."""

EngineFactory = Callable[[], Engine]
"""Builds an `Engine`, called once per worker so the model loads a single time."""
