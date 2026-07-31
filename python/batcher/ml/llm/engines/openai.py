"""The OpenAI-compatible HTTP backend: a *served* model behind a REST endpoint.

Targets vLLM's OpenAI server, llama.cpp, or a hosted API. The batch's requests go out
concurrently over a worker-lifetime thread pool, so a batch costs the slowest request
rather than their sum, and connections are not rebuilt per batch.

An engine request is either a plain prompt string or a ``{"prompt": ..., ...}`` dict when
a per-row column is set (a LoRA ``adapter``, a vision ``image``, a per-row ``max_tokens``
or ``temperature``). This backend honors those dict requests the same way `vllm_engine`
does, so a per-row column produces a correct request against a served endpoint instead of
a malformed one.
"""

from __future__ import annotations

from typing import Any

from batcher.ml.llm.channels import finish_reason_sink, logprob_sink, usage_sink
from batcher.ml.llm.engines.base import Engine, EngineFactory, unpack_request
from batcher.ml.llm.engines.limits import _estimated_tokens, build_limiter

__all__ = ["http_engine"]


def http_engine(
    base_url: str,
    model: str,
    *,
    api_key: str | None = None,
    system: str | None = None,
    chat: bool = True,
    max_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float | None = None,
    stop: list[str] | None = None,
    seed: int | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    logprobs: bool = False,
    response_format: dict | None = None,
    extra_body: dict | None = None,
    on_error: str = "raise",
    timeout: float = 60.0,
    retries: int = 3,
    backoff: float = 0.5,
    concurrency: int = 8,
    requests_per_minute: float | None = None,
    tokens_per_minute: float | None = None,
) -> EngineFactory:
    """An `EngineFactory` calling an OpenAI-compatible HTTP endpoint — a *served* model.

    Targets ``{base_url}/chat/completions`` (the default) or ``/completions``; with chat,
    the **server applies the model's chat template**, so a plain prompt string is wrapped
    as a user message. Works against vLLM's OpenAI server, llama.cpp, or a hosted API.

    The prompts in each batch are sent **concurrently** over up to `concurrency`
    in-flight requests (input order preserved), so a batch's latency is the slowest
    request rather than their sum — the right shape for a network-bound served endpoint
    where one request barely uses the connection. Each request still retries with
    backoff on the 429s a hosted API returns.

    Per-row overrides carried on a request dict (``max_tokens``, ``temperature``, a
    vision ``image``) take precedence over the engine-wide defaults, so one pass can mix a
    16-token classification with a 2000-token summary, or send an image with some rows.

    Examples:
        .. doctest::

            >>> import batcher as bt  # doctest: +SKIP
            >>> engine = bt.ml.http_engine("http://localhost:8000/v1", "my-model")  # doctest: +SKIP
            >>> ds.ml.generate(engine, prompt_column="question").collect()  # doctest: +SKIP

    Args:
        base_url: the OpenAI-compatible API root (e.g. ``http://host:8000/v1``).
        model: the model name the server expects.
        api_key: bearer token, when the endpoint needs one.
        system: a system message prepended to every chat request.
        chat: call ``/chat/completions`` (default) rather than ``/completions``.
        max_tokens: tokens to sample per request (a per-row ``max_tokens`` overrides it).
        temperature: sampling temperature, 0 = greedy (a per-row value overrides it).
        top_p: nucleus-sampling mass; omitted from the body when unset, so a server that
            rejects unknown or null fields still works.
        stop: stop strings; omitted from the body when unset.
        seed: sampling seed for reproducible output, when the server honors it.
        frequency_penalty: OpenAI frequency penalty; omitted when unset.
        presence_penalty: OpenAI presence penalty; omitted when unset.
        logprobs: request per-token log-probabilities and report each generation's summed
            logprob through the logprob channel (so ``generate(logprobs=True)`` works over
            HTTP as it does for vLLM).
        response_format: an OpenAI ``response_format`` (e.g. ``{"type": "json_object"}`` or
            a json-schema block) for structured / guided decoding on a served endpoint.
        extra_body: extra fields merged into every request body (``logit_bias``, vendor
            extensions, ...); an escape hatch for options without a named parameter.
        on_error: ``"raise"`` (default) to fail the batch on an exhausted/un-retryable
            request, or ``"null"`` to yield an empty generation for that row and continue —
            per-row tolerance so one bad row does not lose a batch of good ones.
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
        A zero-arg factory building the HTTP-backed `Engine` once per worker.

    Raises:
        PlanError: if `on_error` is not ``"raise"`` or ``"null"``.
    """
    if on_error not in ("raise", "null"):
        from batcher._internal.errors import PlanError

        raise PlanError(f"on_error must be 'raise' or 'null', got {on_error!r}")
    defaults = _Sampling(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
        seed=seed,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        logprobs=logprobs,
        response_format=response_format,
        extra_body=extra_body,
    )

    limiter = build_limiter(requests_per_minute, tokens_per_minute)

    def factory() -> Engine:
        from concurrent.futures import ThreadPoolExecutor

        from batcher.ml.serving.http import post_json

        url = base_url.rstrip("/") + ("/chat/completions" if chat else "/completions")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        def call_one(request: Any) -> _Result:
            prompt, image, overrides = unpack_request(
                request, ("max_tokens", "temperature", "stop")
            )
            body = _openai_body(model, prompt, chat, system, defaults, overrides, image)
            if limiter is not None:
                # Charged before the call, not after: waiting for capacity is what keeps the
                # send rate at the quota. Retrying a 429 only re-sends the burst that caused it.
                limiter.acquire(_estimated_tokens(prompt, body))
            try:
                # Retries with jittered backoff handle the 429 rate limits hosted APIs return.
                resp = post_json(
                    url, body, headers=headers, timeout=timeout, retries=retries, backoff=backoff
                )
            except Exception:
                if on_error == "raise":
                    raise
                return _Result("", (None, None), None, None)
            return _Result(
                _openai_text(resp, chat),
                _openai_usage(resp),
                _openai_finish_reason(resp),
                _openai_logprob(resp, chat) if defaults.logprobs else None,
            )

        # One pool for the worker's whole life, not one per batch. Building a
        # `ThreadPoolExecutor` per call spawned and tore down `concurrency` threads for
        # every batch, and — because each new thread starts with no connection — forced a
        # fresh TCP connect and TLS handshake per request. At scale the handshakes cost
        # more than the inference.
        pool = ThreadPoolExecutor(max_workers=max(1, concurrency))

        def engine(prompts: list) -> list[str]:
            if concurrency <= 1 or len(prompts) <= 1:
                results = [call_one(p) for p in prompts]
            else:
                # `Executor.map` preserves input order; the calls overlap because each
                # blocks on network I/O (GIL released), bounded to `concurrency` slots.
                results = list(pool.map(call_one, prompts))
            usage = [r.usage for r in results]
            usage_sink().report(usage)
            finish_reason_sink().report([r.finish_reason for r in results])
            logprob_sink().report([r.logprob for r in results])
            engine.last_usage = usage  # the documented legacy channel
            return [r.text for r in results]

        return engine

    return factory


class _Sampling:
    """The engine-wide sampling defaults, merged with any per-row overrides per request."""

    __slots__ = (
        "extra_body",
        "frequency_penalty",
        "logprobs",
        "max_tokens",
        "presence_penalty",
        "response_format",
        "seed",
        "stop",
        "temperature",
        "top_p",
    )

    def __init__(self, **kw: Any) -> None:
        for name in self.__slots__:
            setattr(self, name, kw.get(name))


class _Result:
    """One request's outcome: the text plus its usage, finish reason, and logprob."""

    __slots__ = ("finish_reason", "logprob", "text", "usage")

    def __init__(
        self,
        text: str,
        usage: tuple[int | None, int | None],
        finish_reason: str | None,
        logprob: float | None,
    ) -> None:
        self.text = text
        self.usage = usage
        self.finish_reason = finish_reason
        self.logprob = logprob


def _openai_body(
    model: str,
    prompt: str,
    chat: bool,
    system: str | None,
    defaults: _Sampling,
    overrides: dict,
    image: Any = None,
) -> dict[str, object]:
    """The OpenAI request body. Unset sampling fields are omitted, not sent as null:
    several OpenAI-compatible servers reject a null `stop` or an unknown key. Per-row
    `overrides` win over the engine-wide `defaults`."""
    body: dict[str, object] = {
        "model": model,
        "max_tokens": overrides.get("max_tokens", defaults.max_tokens),
        "temperature": overrides.get("temperature", defaults.temperature),
    }
    stop = overrides.get("stop", defaults.stop)
    _put_optional(
        body,
        top_p=defaults.top_p,
        stop=stop or None,
        seed=defaults.seed,
        frequency_penalty=defaults.frequency_penalty,
        presence_penalty=defaults.presence_penalty,
        response_format=defaults.response_format,
    )
    if defaults.logprobs:
        body["logprobs"] = True
    if defaults.extra_body:
        body.update(defaults.extra_body)
    if not chat:
        body["prompt"] = prompt
        return body
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": _user_content(prompt, image)})
    body["messages"] = messages
    return body


def _put_optional(body: dict[str, object], **fields: object) -> None:
    """Add each field to `body` only when its value is not ``None`` (omit, never null)."""
    for name, value in fields.items():
        if value is not None:
            body[name] = value


def _user_content(prompt: str, image: Any) -> object:
    """The chat user message content: a plain string, or a text+image content list.

    A vision request becomes the OpenAI multimodal shape — a list of a text block and an
    ``image_url`` block whose URL is a base64 ``data:`` URI — so a served vision model
    receives the image inline. A `None` image keeps the plain string.
    """
    if image is None:
        return prompt
    return [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": _image_data_uri(image)}},
    ]


def _image_data_uri(image: Any) -> str:
    """A ``data:image/png;base64,...`` URI for a decoded PIL image (or a passthrough str)."""
    if isinstance(image, str):
        return image  # already a URL or data URI
    import base64
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _openai_text(response: dict, chat: bool) -> str:
    choice = response["choices"][0]
    return choice["message"]["content"] if chat else choice["text"]


def _openai_finish_reason(response: dict) -> str | None:
    """The choice's ``finish_reason`` (``"stop"``/``"length"``), or `None` if absent."""
    return (response.get("choices") or [{}])[0].get("finish_reason")


def _openai_usage(response: dict) -> tuple[int | None, int | None]:
    """The `(prompt_tokens, completion_tokens)` from an OpenAI-style ``usage`` block,
    or `(None, None)` when the server reported none."""
    usage = response.get("usage") or {}
    return usage.get("prompt_tokens"), usage.get("completion_tokens")


def _openai_logprob(response: dict, chat: bool) -> float | None:
    """The generation's summed log-probability, or `None` if the server reported none.

    Chat returns ``logprobs.content[i].logprob``; the completions route returns
    ``logprobs.token_logprobs``. Either way the per-token values are summed, matching the
    cumulative logprob the vLLM path reports.
    """
    choice = (response.get("choices") or [{}])[0]
    block = choice.get("logprobs")
    if not block:
        return None
    if chat:
        content = block.get("content") or []
        values = [tok.get("logprob") for tok in content if tok.get("logprob") is not None]
    else:
        values = [v for v in (block.get("token_logprobs") or []) if v is not None]
    return float(sum(values)) if values else None
