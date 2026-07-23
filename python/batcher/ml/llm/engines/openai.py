"""The OpenAI-compatible HTTP backend: a *served* model behind a REST endpoint.

Targets vLLM's OpenAI server, llama.cpp, or a hosted API. The batch's requests go out
concurrently over a worker-lifetime thread pool, so a batch costs the slowest request
rather than their sum, and connections are not rebuilt per batch.
"""

from __future__ import annotations

from batcher.ml.llm.channels import finish_reason_sink, usage_sink
from batcher.ml.llm.engines.base import Engine, EngineFactory

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
    timeout: float = 60.0,
    concurrency: int = 8,
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
        max_tokens: tokens to sample per request.
        temperature: sampling temperature (0 = greedy).
        top_p: nucleus-sampling mass; omitted from the body when unset, so a server that
            rejects unknown or null fields still works.
        stop: stop strings; omitted from the body when unset.
        timeout: per-request timeout in seconds.
        concurrency: in-flight requests per batch. Set to 1 to serialize.

    Returns:
        A zero-arg factory building the HTTP-backed `Engine` once per worker.
    """

    def factory() -> Engine:
        from concurrent.futures import ThreadPoolExecutor

        from batcher.ml.serving.http import post_json

        url = base_url.rstrip("/") + ("/chat/completions" if chat else "/completions")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        def call_one(prompt: str) -> tuple[str, tuple[int | None, int | None], str | None]:
            body = _openai_body(
                model, prompt, chat, system, max_tokens, temperature, top_p=top_p, stop=stop
            )
            # Retries with jittered backoff handle the 429 rate limits hosted APIs return.
            resp = post_json(url, body, headers=headers, timeout=timeout)
            return _openai_text(resp, chat), _openai_usage(resp), _openai_finish_reason(resp)

        # One pool for the worker's whole life, not one per batch. Building a
        # `ThreadPoolExecutor` per call spawned and tore down `concurrency` threads for
        # every batch, and — because each new thread starts with no connection — forced a
        # fresh TCP connect and TLS handshake per request. At scale the handshakes cost
        # more than the inference.
        pool = ThreadPoolExecutor(max_workers=max(1, concurrency))

        def engine(prompts: list[str]) -> list[str]:
            if concurrency <= 1 or len(prompts) <= 1:
                results = [call_one(p) for p in prompts]
            else:
                # `Executor.map` preserves input order; the calls overlap because each
                # blocks on network I/O (GIL released), bounded to `concurrency` slots.
                results = list(pool.map(call_one, prompts))
            usage = [u for _text, u, _r in results]
            usage_sink().report(usage)
            finish_reason_sink().report([r for _text, _u, r in results])
            engine.last_usage = usage  # the documented legacy channel
            return [text for text, _u, _r in results]

        return engine

    return factory


def _openai_body(
    model: str,
    prompt: str,
    chat: bool,
    system: str | None,
    max_tokens: int,
    temperature: float,
    *,
    top_p: float | None = None,
    stop: list[str] | None = None,
) -> dict[str, object]:
    """The OpenAI request body. Unset sampling fields are omitted, not sent as null:
    several OpenAI-compatible servers reject a null `stop` or an unknown key."""
    common: dict[str, object] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if top_p is not None:
        common["top_p"] = top_p
    if stop:
        common["stop"] = stop
    if not chat:
        return {**common, "prompt": prompt}
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    return {**common, "messages": messages}


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
