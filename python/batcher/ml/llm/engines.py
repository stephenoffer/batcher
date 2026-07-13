"""LLM engine adapters — the pluggable ``list[str] -> list[str]`` backends.

An *engine* is the only thing `generate` needs from an LLM: hand it a batch of
requests, get back a string per request in order. Keeping that contract this narrow is
what lets vLLM, an OpenAI-compatible HTTP endpoint, and a deterministic test double be
interchangeable — and what keeps the columnar machinery in `generate` free of any
model library.

An engine may also set ``last_usage`` (one ``(prompt_tokens, completion_tokens)`` pair
per request, in order) for cost/throughput accounting.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

__all__ = ["Engine", "EngineFactory", "http_engine", "vllm_engine"]

Engine = Callable[[list[str]], Sequence[str]]
"""Maps a list of prompts to a list of generated strings (one per prompt, in order)."""

EngineFactory = Callable[[], Engine]
"""Builds an `Engine`, called once per worker so the model loads a single time."""


def vllm_engine(
    model: str,
    *,
    chat: bool = False,
    system: str | None = None,
    sampling: dict[str, object] | None = None,
    guided_json: dict[str, object] | None = None,
    guided_regex: str | None = None,
    lora_path: str | None = None,
    lora_paths: dict[str, str] | None = None,
    quantization: str | None = "auto",
    **engine_kwargs: object,
) -> EngineFactory:
    """An `EngineFactory` backed by vLLM (requires ``batcher-engine[vllm]`` + a GPU).

    The factory builds a vLLM engine once per worker and exposes it as a
    ``list[str] -> list[str]`` callable.

    Zero-config batch defaults: **prefix caching** and **chunked prefill** are enabled
    unless you set them — both are throughput/TTFT wins for offline batch (a shared
    system prompt is encoded once; long prefills interleave with decode) that Ray Data
    users must turn on by hand. Any value you pass in `engine_kwargs` wins. (`max_model_len`
    is left to vLLM's model default — auto-sizing it to the data needs the worker tokenizer
    and is a follow-on; a char-heuristic could truncate prompts and corrupt output.)

    Examples:
        .. doctest::

            >>> import batcher as bt  # doctest: +SKIP
            >>> engine = bt.ml.vllm_engine("meta-llama/Llama-3-8B", chat=True)  # doctest: +SKIP
            >>> ds.ml.generate(engine, prompt_column="question").collect()  # doctest: +SKIP

    Args:
        model: the model id or path handed to ``vllm.LLM``.
        chat: send each prompt as a **chat conversation** (``LLM.chat``), so vLLM applies
            the model's own chat template. Set this for any instruction-tuned or chat
            model: the completion path (the default, matching a base model) skips the
            template, and the model then answers a prompt in a format it was never tuned
            on — degraded output with nothing to signal it. Not compatible with an image
            column (an image has no place in a text conversation); use the completion
            path for vision.
        system: a system turn prepended to every conversation (with `chat`).
        sampling: `SamplingParams` kwargs (``temperature``, ``top_p``, ``max_tokens``,
            ``stop``, ``n``, ``seed``, ...). Defaults to greedy (``temperature=0``).
        guided_json: constrain generation to this JSON schema via vLLM's guided decoding
            — the reliable way to get parseable output; pair with
            ``llm_generate(parse_json=True)``.
        guided_regex: constrain generation to this regex, same mechanism.
        lora_path: serve a single LoRA adapter on top of the base `model` (applied to
            every row that does not name another adapter).
        lora_paths: a ``{name: path}`` table of adapters to **multiplex**: a request
            tagged with that name (via ``llm_generate(adapter_column=...)``) is routed to
            it, so one engine serves many adapters in one batch. Rows are grouped by
            adapter and each group generated together; output order is preserved. Set the
            vLLM ``enable_lora``/``max_loras``/``max_cpu_loras`` kwargs for the cache.
        quantization: ``"auto"`` (default) picks ``"fp8"`` on GPUs with native FP8 tensor
            cores (NVIDIA Ada L4/L40S, Hopper H100), where it halves weight/KV-cache
            memory at <1% quality loss, and keeps native precision (BF16/FP16) elsewhere
            — the zero-config win Ray Data users must select by hand per GPU. Pass an
            explicit string (``"fp8"``, ``"awq"``, ...) to force it, or ``None`` to disable.
        engine_kwargs: passed to ``vllm.LLM``: ``max_model_len``,
            ``gpu_memory_utilization``, ``tensor_parallel_size`` (for a model larger than
            one GPU), ``speculative_config`` / ``spec_decode_disable_by_queue_size``
            (speculative decoding), ``enable_lora`` / ``max_loras`` / ``max_cpu_loras``
            (multi-adapter serving), ...

    Returns:
        A zero-arg factory building the vLLM-backed `Engine` once per worker.
    """
    engine_kwargs = _vllm_batch_defaults(engine_kwargs)

    def factory() -> Engine:
        from vllm import LLM, SamplingParams

        # Resolve `quantization="auto"` here, on the GPU worker, so it reflects the
        # actual device (the driver may have no GPU). An explicit `engine_kwargs`
        # `quantization` still wins.
        kwargs = _with_auto_quant(quantization, engine_kwargs)
        enable_lora = lora_path is not None or bool(lora_paths)
        llm = LLM(model=model, enable_lora=enable_lora, **kwargs)
        sampling_kwargs = {"temperature": 0.0, **(sampling or {})}
        guided = _guided_decoding(guided_json, guided_regex)
        if guided is not None:
            sampling_kwargs["guided_decoding"] = guided
        params = SamplingParams(**sampling_kwargs)
        lora_table = _lora_table(lora_path, lora_paths)

        def engine(prompts: list) -> list[str]:
            # Route per-row by adapter (a request may carry an "adapter" tag), running
            # each adapter's group together; usage + order are preserved.
            texts, usage = _generate_routed(
                llm, params, prompts, lora_table, chat=chat, system=system
            )
            engine.last_usage = usage
            return texts

        return engine

    return factory


def _group_indices_by_adapter(prompts: list) -> dict[str | None, list[int]]:
    """Group request indices by their ``adapter`` tag (``None`` for an untagged request,
    which uses the base model or the single `lora_path`). Pure, so the routing logic
    tests without vLLM."""
    groups: dict[str | None, list[int]] = {}
    for i, p in enumerate(prompts):
        name = p.get("adapter") if isinstance(p, dict) else None
        groups.setdefault(name, []).append(i)
    return groups


def _generate_routed(
    llm, params, prompts: list, lora_table: dict, *, chat: bool = False, system: str | None = None
):
    """Generate with per-row LoRA routing: group requests by adapter, run each group with
    that adapter's `LoRARequest` (from `lora_table`, `None` key = base/single adapter),
    and reassemble outputs + per-prompt token usage in input order.

    With `chat`, each request is wrapped as a conversation and sent through
    ``llm.chat``, which applies the model's own chat template; otherwise the raw prompt
    goes to ``llm.generate``.

    Pure but for `llm.generate`/`llm.chat`, so it tests with a stub `llm` (no vLLM/GPU)."""
    texts: list[str | None] = [None] * len(prompts)
    usage: list[tuple[int, int] | None] = [None] * len(prompts)
    for name, idxs in _group_indices_by_adapter(prompts).items():
        lora = lora_table.get(name)
        if chat:
            convos = [_chat_messages(prompts[i], system) for i in idxs]
            outputs = llm.chat(convos, params, lora_request=lora)
        else:
            requests = [_vllm_request(prompts[i]) for i in idxs]
            outputs = llm.generate(requests, params, lora_request=lora)
        for j, o in zip(idxs, outputs, strict=True):
            texts[j] = o.outputs[0].text
            usage[j] = (len(o.prompt_token_ids), len(o.outputs[0].token_ids))
    return texts, usage


def _chat_messages(prompt: object, system: str | None) -> list[dict[str, str]]:
    """One request as an OpenAI-style conversation: an optional system turn, then the user turn.

    vLLM's ``LLM.chat`` renders these through the model's own chat template, which is
    what an instruction-tuned model was trained on. Sending the bare prompt to
    ``LLM.generate`` instead skips the template — the model still answers, just far
    worse, with no error to notice.

    Raises:
        ValueError: For a vision request, whose image has no place in a text
            conversation; use the completion path (``chat=False``) for `image_column`.
    """
    if isinstance(prompt, dict):
        if prompt.get("image") is not None:
            raise ValueError(
                "vllm_engine(chat=True) cannot carry an image_column; "
                "use chat=False (the completion path) for vision-language models"
            )
        text = prompt["prompt"]
    else:
        text = prompt
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": str(text)})
    return messages


def _vllm_batch_defaults(engine_kwargs: dict[str, object]) -> dict[str, object]:
    """Apply zero-config batch-inference defaults to vLLM `engine_kwargs`.

    Enables prefix caching (a shared prompt prefix is encoded once — big throughput win
    when many rows share a system prompt) and chunked prefill (interleaves long prefills
    with decode — lower TTFT) unless the user set them. Pure + dict-only, so it unit-tests
    without vLLM or a GPU. An explicit value always wins.
    """
    out = dict(engine_kwargs)
    out.setdefault("enable_prefix_caching", True)
    out.setdefault("enable_chunked_prefill", True)
    return out


def _with_auto_quant(
    quantization: str | None, engine_kwargs: dict[str, object]
) -> dict[str, object]:
    """Resolve ``quantization="auto"`` to a GPU-appropriate vLLM `quantization` kwarg.

    ``"auto"`` consults `ml.gpu.recommend_quantization` (FP8 only on native-FP8 GPUs);
    any other value is used verbatim. An explicit `engine_kwargs["quantization"]` always
    wins (we only `setdefault`). Pure but for the GPU probe, so it tests with that probe
    monkeypatched — no vLLM or GPU needed."""
    if quantization == "auto":
        from batcher.ml.gpu import recommend_quantization

        quantization = recommend_quantization()
    out = dict(engine_kwargs)
    if quantization is not None:
        out.setdefault("quantization", quantization)
    return out


def _vllm_request(prompt: object) -> object:
    """Translate a request to vLLM input: a string passes through; a ``{prompt, image}``
    dict becomes ``{"prompt": ..., "multi_modal_data": {"image": ...}}``."""
    if not isinstance(prompt, dict):
        return prompt
    request: dict = {"prompt": prompt["prompt"]}
    image = prompt.get("image")
    if image is not None:
        request["multi_modal_data"] = {"image": image}
    return request


def _guided_decoding(guided_json: dict | None, guided_regex: str | None) -> object | None:
    """A vLLM `GuidedDecodingParams` for JSON-schema or regex-constrained output."""
    if guided_json is None and guided_regex is None:
        return None
    from vllm.sampling_params import GuidedDecodingParams

    if guided_json is not None:
        return GuidedDecodingParams(json=guided_json)
    return GuidedDecodingParams(regex=guided_regex)


def _lora_table(lora_path: str | None, lora_paths: dict[str, str] | None) -> dict:
    """Build the adapter routing table: ``{None: base/single-adapter request, name:
    that adapter's request}``. The `None` key is the single `lora_path` (or `None` for
    the base model); each named adapter in `lora_paths` gets a distinct integer id."""
    table: dict = {None: _make_lora_request("adapter", 1, lora_path) if lora_path else None}
    for idx, (name, path) in enumerate(sorted((lora_paths or {}).items()), start=2):
        table[name] = _make_lora_request(name, idx, path)
    return table


def _make_lora_request(name: str, idx: int, path: str) -> object:
    """A vLLM `LoRARequest` naming `path` with a unique integer id."""
    from vllm.lora.request import LoRARequest

    return LoRARequest(name, idx, path)


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

        def call_one(prompt: str) -> tuple[str, tuple[int | None, int | None]]:
            body = _openai_body(
                model, prompt, chat, system, max_tokens, temperature, top_p=top_p, stop=stop
            )
            # Retries with backoff handle the 429 rate limits hosted APIs return.
            resp = post_json(url, body, headers=headers, timeout=timeout)
            return _openai_text(resp, chat), _openai_usage(resp)

        def engine(prompts: list[str]) -> list[str]:
            if concurrency <= 1 or len(prompts) <= 1:
                pairs = [call_one(p) for p in prompts]
            else:
                # `ThreadPoolExecutor.map` preserves input order; the calls overlap
                # because each blocks on network I/O (GIL released), bounded to
                # `concurrency` slots.
                with ThreadPoolExecutor(max_workers=min(concurrency, len(prompts))) as pool:
                    pairs = list(pool.map(call_one, prompts))
            engine.last_usage = [u for _text, u in pairs]
            return [text for text, _u in pairs]

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


def _openai_usage(response: dict) -> tuple[int | None, int | None]:
    """The `(prompt_tokens, completion_tokens)` from an OpenAI-style ``usage`` block,
    or `(None, None)` when the server reported none."""
    usage = response.get("usage") or {}
    return usage.get("prompt_tokens"), usage.get("completion_tokens")
