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

from batcher.ml.llm.channels import finish_reason_sink, usage_sink

__all__ = ["Engine", "EngineFactory", "batched_engine", "unpack_request"]

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


def batched_engine(
    call_one: Callable[[Any], Any],
    pool: Any,
    concurrency: int,
    *,
    report: Callable[[list], None] | None = None,
) -> Engine:
    """The batch loop every served-endpoint engine runs: overlap the calls, report, unpack.

    Each engine differs in how it *builds* one request and reads one response; none of them
    differ in what happens around a batch of them. That loop carries three properties worth
    stating once rather than four times:

    * **Order is the contract.** ``Executor.map`` yields results in *input* order whatever
      order they complete in, so the returned strings stay aligned with the rows that
      produced them — which everything columnar downstream assumes without checking.
    * **Overlap, but only above one.** A single prompt, or a single slot, runs inline: a
      pool round-trip buys nothing and makes a stack trace harder to read.
    * **Every signal is reported.** Token usage and finish reason go to the per-call sinks
      in `llm.channels`, and `engine.last_usage` is still set for the documented legacy
      channel.

    `report` is the hook for a signal only some providers return — OpenAI's per-token
    logprobs — so an engine that has one does not need its own copy of the loop to report
    it. That copy is what this replaced: the OpenAI factory carried a byte-for-byte second
    version of this function differing only in one extra sink call, which meant a fix to
    ordering or reporting had to be made twice and was one of the two places
    `just lint-duplication` flagged in the tree.

    Args:
        call_one: Sends one request and returns a result object carrying at least ``text``,
            ``usage`` and ``finish_reason``.
        pool: The executor whose slots the batch overlaps across, reused for the worker's
            whole life so connections stay warm.
        concurrency: How many requests may be in flight; 1 runs inline.
        report: Called with the batch's results after the shared sinks, for a provider
            signal the shared loop does not know about.

    Returns:
        An `Engine` over the prompts.
    """

    def engine(prompts: list) -> list[str]:
        if concurrency <= 1 or len(prompts) <= 1:
            replies = [call_one(p) for p in prompts]
        else:
            replies = list(pool.map(call_one, prompts))
        usage = [r.usage for r in replies]
        usage_sink().report(usage)
        finish_reason_sink().report([r.finish_reason for r in replies])
        if report is not None:
            report(replies)
        engine.last_usage = usage  # the documented legacy channel
        return [r.text for r in replies]

    return engine
