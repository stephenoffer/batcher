"""The Anthropic Messages API backend — batch generation against a hosted Claude model.

A *served* engine like `http_engine`, but speaking Anthropic's ``/v1/messages`` wire shape
rather than the OpenAI one: ``x-api-key`` + ``anthropic-version`` headers, a top-level
``system`` field, ``content`` blocks in and out, and ``usage.input_tokens`` /
``output_tokens``. Kept dependency-free (the standard-library `post_json`, no ``anthropic``
SDK) so it stays a leaf of `bc-py`-free control-plane code like every other engine here.

The batch's requests go out concurrently over a worker-lifetime thread pool, so a batch
costs the slowest request rather than their sum, and each request retries with backoff on
the 429s a hosted API returns.
"""

from __future__ import annotations

from typing import Any

from batcher.ml.llm.channels import finish_reason_sink, usage_sink
from batcher.ml.llm.engines.base import Engine, EngineFactory, unpack_request
from batcher.ml.llm.engines.limits import _estimated_tokens, build_limiter

__all__ = ["anthropic_engine"]

#: The Anthropic API version header value the Messages API expects.
_ANTHROPIC_VERSION = "2023-06-01"

#: Anthropic ``stop_reason`` → the ``"stop"``/``"length"`` vocabulary the batcher columns
#: use (see `llm.columns`), so a truncation is detectable the same way across engines.
_FINISH_REASON = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length"}


def anthropic_engine(
    model: str,
    *,
    api_key: str | None = None,
    base_url: str = "https://api.anthropic.com/v1",
    system: str | None = None,
    max_tokens: int = 1024,
    temperature: float | None = None,
    stop_sequences: list[str] | None = None,
    extra_body: dict | None = None,
    on_error: str = "raise",
    timeout: float = 60.0,
    retries: int = 3,
    backoff: float = 0.5,
    concurrency: int = 8,
    requests_per_minute: float | None = None,
    tokens_per_minute: float | None = None,
) -> EngineFactory:
    """An `EngineFactory` calling the Anthropic Messages API — a hosted Claude model.

    Drops into ``ds.ml.generate`` / `llm_generate` exactly like `vllm_engine` or
    `http_engine`; only the backend differs. Each row's prompt becomes a single-user-turn
    message, an optional `system` prompt is sent top-level, and the concatenated text of
    the response's content blocks is the generation. Per-row overrides carried on a request
    dict (``max_tokens``, ``temperature``, a vision ``image``) win over the engine defaults.

    ``max_tokens`` is **required** by the Messages API, so it always has a value here; give
    each row its own with a ``max_tokens_column`` when budgets vary. `temperature` is
    omitted from the body when unset — several current Claude models reject it, so leaving
    it unset is the safe default.

    Examples:
        .. doctest::

            >>> import batcher as bt  # doctest: +SKIP
            >>> from batcher.ml import anthropic_engine  # doctest: +SKIP
            >>> engine = anthropic_engine("claude-haiku-4-5")  # doctest: +SKIP
            >>> ds.ml.generate(engine, prompt_column="question").collect()  # doctest: +SKIP

    Args:
        model: the Claude model id (e.g. ``"claude-haiku-4-5"``).
        api_key: the Anthropic API key; falls back to ``$ANTHROPIC_API_KEY`` when unset.
        base_url: the API root (defaults to Anthropic's; override for a proxy).
        system: an optional system prompt sent with every request.
        max_tokens: tokens to sample per request (a per-row ``max_tokens`` overrides it).
        temperature: sampling temperature; omitted from the body when unset (some models
            reject it). A per-row ``temperature`` overrides it.
        stop_sequences: strings that stop generation; omitted when unset.
        extra_body: extra fields merged into every request body (``top_p``, ``top_k``,
            ``thinking``, ...) — the escape hatch for options without a named parameter.
        on_error: ``"raise"`` (default) to fail the batch on an exhausted/un-retryable
            request, or ``"null"`` to yield an empty generation for that row and continue.
        timeout: per-request timeout in seconds.
        retries: retry attempts per request on a transient failure (429/5xx/connection).
        backoff: base seconds for the jittered exponential backoff between retries.
        concurrency: in-flight requests per batch. Set to 1 to serialize.
        requests_per_minute: client-side cap on requests per minute, **per worker**. Unset
            means unlimited. Waiting for capacity holds the send rate at the quota, where
            retrying a 429 only re-sends the burst that caused it.
        tokens_per_minute: client-side cap on tokens per minute, per worker, counting the
            prompt plus the reply the request reserved. Unset means unlimited.

    Returns:
        A zero-arg factory building the Anthropic-backed `Engine` once per worker.

    Raises:
        PlanError: if `on_error` is not ``"raise"`` or ``"null"``.
    """
    if on_error not in ("raise", "null"):
        from batcher._internal.errors import PlanError

        raise PlanError(f"on_error must be 'raise' or 'null', got {on_error!r}")

    limiter = build_limiter(requests_per_minute, tokens_per_minute)

    def factory() -> Engine:
        import os
        from concurrent.futures import ThreadPoolExecutor

        from batcher.ml.serving.http import post_json

        url = base_url.rstrip("/") + "/messages"
        headers = {"content-type": "application/json", "anthropic-version": _ANTHROPIC_VERSION}
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if key:
            headers["x-api-key"] = key

        def call_one(request: Any) -> tuple[str, tuple[int | None, int | None], str | None]:
            prompt, image, overrides = unpack_request(request, ("max_tokens", "temperature"))
            body = _messages_body(
                model,
                prompt,
                system,
                max_tokens,
                temperature,
                stop_sequences,
                overrides,
                image,
                extra_body,
            )
            if limiter is not None:
                # Charged before the call: waiting for capacity holds the send rate at the
                # quota, where retrying a 429 only re-sends the burst that caused it.
                limiter.acquire(_estimated_tokens(prompt, body))
            try:
                resp = post_json(
                    url, body, headers=headers, timeout=timeout, retries=retries, backoff=backoff
                )
            except Exception:
                if on_error == "raise":
                    raise
                return "", (None, None), None
            return _message_text(resp), _message_usage(resp), _message_finish_reason(resp)

        pool = ThreadPoolExecutor(max_workers=max(1, concurrency))

        def engine(prompts: list) -> list[str]:
            if concurrency <= 1 or len(prompts) <= 1:
                results = [call_one(p) for p in prompts]
            else:
                results = list(pool.map(call_one, prompts))
            usage = [u for _t, u, _r in results]
            usage_sink().report(usage)
            finish_reason_sink().report([r for _t, _u, r in results])
            engine.last_usage = usage  # the documented legacy channel
            return [text for text, _u, _r in results]

        return engine

    return factory


def _messages_body(
    model: str,
    prompt: str,
    system: str | None,
    max_tokens: int,
    temperature: float | None,
    stop_sequences: list[str] | None,
    overrides: dict,
    image: Any,
    extra_body: dict | None,
) -> dict[str, object]:
    """The Messages API request body. Unset optional fields are omitted, not sent null."""
    body: dict[str, object] = {
        "model": model,
        "max_tokens": overrides.get("max_tokens", max_tokens),
        "messages": [{"role": "user", "content": _user_content(prompt, image)}],
    }
    if system is not None:
        body["system"] = system
    temp = overrides.get("temperature", temperature)
    if temp is not None:
        body["temperature"] = temp
    if stop_sequences:
        body["stop_sequences"] = stop_sequences
    if extra_body:
        body.update(extra_body)
    return body


def _user_content(prompt: str, image: Any) -> object:
    """The user message content: a plain string, or a text+image block list for vision.

    Anthropic vision takes a base64 ``image`` source block before the text block.
    """
    if image is None:
        return prompt
    return [
        {"type": "image", "source": _image_source(image)},
        {"type": "text", "text": prompt},
    ]


def _image_source(image: Any) -> dict[str, object]:
    """A base64 image source block for a decoded PIL image."""
    import base64
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {"type": "base64", "media_type": "image/png", "data": data}


def _message_text(response: dict) -> str:
    """The generation: every ``text`` content block concatenated (empty on a refusal).

    A ``stop_reason`` of ``"refusal"`` returns HTTP 200 with an empty ``content`` list, so
    reading ``content[0]`` unconditionally would crash a batch on one refused row.
    """
    blocks = response.get("content") or []
    return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")


def _message_usage(response: dict) -> tuple[int | None, int | None]:
    """The ``(input_tokens, output_tokens)`` Anthropic reports, or ``(None, None)``."""
    usage = response.get("usage") or {}
    return usage.get("input_tokens"), usage.get("output_tokens")


def _message_finish_reason(response: dict) -> str | None:
    """The ``stop_reason`` normalized to the ``"stop"``/``"length"`` vocabulary, else raw.

    ``end_turn``/``stop_sequence`` → ``"stop"`` and ``max_tokens`` → ``"length"``, so a
    truncation is detectable the same way it is for the vLLM and OpenAI engines; anything
    else (``"refusal"``, ``"pause_turn"``) passes through unchanged.
    """
    reason = response.get("stop_reason")
    if reason is None:
        return None
    return _FINISH_REASON.get(reason, reason)
