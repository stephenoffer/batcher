"""The vLLM backend: an offline, GPU-resident engine with LoRA multiplexing.

Owns everything specific to running vLLM in a batch worker — the zero-config batch
defaults, FP8 auto-quantization, guided decoding, per-row LoRA routing with adapter
co-batching, and context-window truncation. Split from the HTTP backend so neither
grows into the other; both expose the same `Engine` contract.
"""

from __future__ import annotations

from batcher.ml.llm.channels import finish_reason_sink, logprob_sink, usage_sink
from batcher.ml.llm.engines.base import Engine, EngineFactory
from batcher.ml.llm.engines.footprint import declared_context
from batcher.ml.llm.engines.parallelism import advise_tensor_parallelism
from batcher.ml.llm.engines.templates import warn_if_chat_template_unused
from batcher.ml.llm.sizing import fit_to_window, prompt_window, sized_window

__all__ = ["vllm_engine"]


def vllm_engine(
    model: str,
    *,
    chat: bool | None = None,
    system: str | None = None,
    sampling: dict[str, object] | None = None,
    guided_json: dict[str, object] | None = None,
    guided_regex: str | None = None,
    guided_choice: list[str] | None = None,
    guided_grammar: str | None = None,
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
    users must turn on by hand. Any value you pass in `engine_kwargs` wins.

    `max_model_len` is left to vLLM's model default, or pass ``max_model_len="auto"`` to
    size the context window from the data. The KV cache is reserved from that number, so a
    128K default over a corpus of 2K-token prompts spends almost all of the cache on lengths
    the data never reaches — capacity that would otherwise be concurrent sequences. Auto
    sizing measures the first batch, converts characters to tokens at a rate no tokenizer
    beats, adds the generation budget and headroom, and rounds up; if the model refuses the
    result it falls back to the model's own window with a warning. Because it defers the
    engine build to the first batch, it changes *when* the model loads on that path only.

    Whatever the window ends up being, a prompt that would overflow it is **truncated to
    fit** using the worker's own tokenizer (with a warning naming how many rows were cut),
    rather than failing the whole request over one long row. Truncation is skipped entirely
    when no tokenizer is reachable — a character heuristic would cut in the wrong place and
    corrupt output silently.

    Examples:
        .. doctest::

            >>> import batcher as bt  # doctest: +SKIP
            >>> engine = bt.ml.vllm_engine("meta-llama/Llama-3-8B", chat=True)  # doctest: +SKIP
            >>> ds.ml.generate(engine, prompt_column="question").collect()  # doctest: +SKIP

    Args:
        model: the model id or path handed to ``vllm.LLM``.
        chat: send each prompt as a **chat conversation** (``LLM.chat``), so vLLM applies
            the model's own chat template. Set this for any instruction-tuned or chat
            model: the completion path skips the template, and the model then answers a
            prompt in a format it was never tuned on — degraded output with nothing to
            signal it. Left unset it stays on the completion path (matching a base model)
            but **warns** if the model turns out to ship a chat template, since that
            combination is almost always the mistake above; an explicit ``False`` is taken
            as a decision and is silent, which is what constrained-choice classification
            wants. Not compatible with an image column (an image has no place in a text
            conversation); use the completion path for vision.
        system: a system turn prepended to every conversation (with `chat`).
        sampling: `SamplingParams` kwargs (``temperature``, ``top_p``, ``max_tokens``,
            ``stop``, ``n``, ``seed``, ...). Defaults to greedy (``temperature=0``).
        guided_json: constrain generation to this JSON schema via vLLM's guided decoding
            — the reliable way to get parseable output; pair with
            ``llm_generate(parse_json=True)``.
        guided_regex: constrain generation to this regex, same mechanism.
        guided_choice: constrain generation to **exactly one** of these strings — the
            zero-loss classification path (``ds.ml.classify`` with ``instruct=False``), where
            the model can only emit a declared label, not one buried in a sentence.
        guided_grammar: constrain generation to an EBNF/GBNF grammar — a DSL/SQL or any
            formal language, same mechanism.
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
        engine_kwargs: passed to ``vllm.LLM``: ``max_model_len`` (a token count, or
            ``"auto"`` to size it from the data — see above),
            ``gpu_memory_utilization``, ``tensor_parallel_size`` (for a model larger than
            one GPU), ``speculative_config`` / ``spec_decode_disable_by_queue_size``
            (speculative decoding), ``enable_lora`` / ``max_loras`` / ``max_cpu_loras``
            (multi-adapter serving), ...

    Returns:
        A zero-arg factory building the vLLM-backed `Engine` once per worker.
    """
    _reject_best_of_n(sampling)
    engine_kwargs = _vllm_batch_defaults(engine_kwargs)

    def factory() -> Engine:
        from batcher._internal.optional import require

        LLM = require("vllm", "LLM", feature="vllm_engine", provides="vllm", extra="vllm")
        SamplingParams = require(
            "vllm", "SamplingParams", feature="vllm_engine", provides="vllm", extra="vllm"
        )

        # Resolve `quantization="auto"` here, on the GPU worker, so it reflects the
        # actual device (the driver may have no GPU). An explicit `engine_kwargs`
        # `quantization` still wins.
        kwargs = _with_auto_quant(quantization, engine_kwargs)
        enable_lora = lora_path is not None or bool(lora_paths)
        sampling_kwargs = {"temperature": 0.0, **(sampling or {})}
        gkw = _guided_decoding_kwargs(guided_json, guided_regex, guided_choice, guided_grammar)
        if gkw is not None:
            from vllm.sampling_params import GuidedDecodingParams

            sampling_kwargs["guided_decoding"] = GuidedDecodingParams(**gkw)
        params = SamplingParams(**sampling_kwargs)
        lora_table = _lora_table(lora_path, lora_paths)
        # `max_model_len="auto"` sizes the KV cache from the data, which means the engine
        # cannot exist until a batch does. Everything else builds eagerly, exactly as before,
        # so the opt-in path is the only one whose construction timing changed.
        auto_window = kwargs.get("max_model_len") == "auto"
        if auto_window:
            # The sentinel is Batcher's, not vLLM's — it must not reach the engine.
            del kwargs["max_model_len"]
        built: dict = {}

        def _build(prompts: list) -> None:
            kw = dict(kwargs)
            # On the GPU worker, with the model's real footprint read from repository metadata
            # rather than by downloading it. Without those numbers the branch that matters —
            # "this degree cannot hold this model" — could only fire as an allocation error.
            advise_tensor_parallelism(model, int(kw.get("tensor_parallel_size", 1) or 1))
            if auto_window:
                sized = sized_window(prompts, sampling_kwargs, declared_context(model))
                if sized is not None:
                    kw["max_model_len"] = sized
            built["llm"] = _construct(LLM, model, enable_lora, kw)
            built["tokenizer"] = _worker_tokenizer(built["llm"])
            if chat is None:
                warn_if_chat_template_unused(built["tokenizer"], model)
            built["window"] = prompt_window(built["llm"], kw, sampling_kwargs.get("max_tokens"))

        if not auto_window:
            _build([])

        def engine(prompts: list) -> list[str]:
            if "llm" not in built:
                _build(prompts)
            llm = built["llm"]
            # Route per-row by adapter (a request may carry an "adapter" tag), co-batching
            # the adapters where vLLM supports it; usage + order are preserved.
            prompts = fit_to_window(prompts, built["tokenizer"], built["window"])
            texts, usage, reasons, logprobs = _generate_signals(
                llm, params, prompts, lora_table, chat=bool(chat), system=system
            )
            usage_sink().report(usage)
            finish_reason_sink().report(reasons)
            logprob_sink().report(logprobs)
            engine.last_usage = usage  # the documented legacy channel
            return texts

        return engine

    return factory


def _reject_best_of_n(sampling: dict[str, object] | None) -> None:
    """Refuse ``sampling={"n": k}`` for k > 1 rather than generate k and return one.

    An `Engine` returns exactly one string per request, so there is nowhere for the other
    k-1 candidates to go. Accepting the parameter and dropping them meant paying for k
    times the decode on the GPU and silently receiving one candidate — the expensive kind
    of no-op. Fail at construction, where the caller can still act on it.

    Raises:
        PlanError: If `sampling` requests more than one candidate per prompt.
    """
    n = (sampling or {}).get("n", 1)
    if isinstance(n, int) and n > 1:
        from batcher._internal.errors import PlanError

        raise PlanError(
            f"vllm_engine(sampling={{'n': {n}}}): best-of-n is not supported — an engine "
            "returns one generation per request, so n=" + str(n) + " would generate "
            f"{n} candidates and discard {n - 1} of them. Drop `n`, or generate the "
            "candidates as separate rows."
        )


def _worker_tokenizer(llm: object) -> object | None:
    """The worker's own tokenizer, when vLLM exposes one, else `None`.

    Truncation has to count *tokens*, and the only tokenizer that agrees with the model
    is the model's. A character heuristic would cut prompts in the wrong place and
    corrupt output silently, which is why this returns `None` — disabling truncation —
    rather than guessing when no tokenizer is reachable.
    """
    getter = getattr(llm, "get_tokenizer", None)
    if getter is None:
        return None
    try:
        return getter()
    except Exception:  # pragma: no cover - vLLM version differences
        return None


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
    """`_generate_signals` without the logprobs — the ``(texts, usage, reasons)`` seam.

    Kept because it is what the engine tests drive directly with a stub `llm`. It is a
    projection of `_generate_signals`, not a second implementation, so the two cannot
    disagree about routing or ordering.
    """
    return _generate_signals(llm, params, prompts, lora_table, chat=chat, system=system)[:3]


def _generate_signals(
    llm, params, prompts: list, lora_table: dict, *, chat: bool = False, system: str | None = None
):
    """Generate with per-row LoRA routing, returning every signal the batch produced.

    Groups requests by adapter, runs each group with that adapter's `LoRARequest` (from
    `lora_table`, `None` key = base/single adapter), and reassembles the outputs,
    per-prompt token usage, finish reasons, and cumulative logprobs in input order.

    With `chat`, each request is wrapped as a conversation and sent through
    ``llm.chat``, which applies the model's own chat template; otherwise the raw prompt
    goes to ``llm.generate``.

    Pure but for `llm.generate`/`llm.chat`, so it tests with a stub `llm` (no vLLM/GPU)."""
    texts: list[str | None] = [None] * len(prompts)
    usage: list[tuple[int, int] | None] = [None] * len(prompts)
    reasons: list[str | None] = [None] * len(prompts)
    logprobs: list[float | None] = [None] * len(prompts)
    for idxs, outputs in _dispatch_groups(
        llm, params, prompts, lora_table, chat=chat, system=system
    ):
        for j, o in zip(idxs, outputs, strict=True):
            texts[j] = o.outputs[0].text
            usage[j] = (len(o.prompt_token_ids), len(o.outputs[0].token_ids))
            reasons[j] = getattr(o.outputs[0], "finish_reason", None)
            # vLLM populates this without `SamplingParams(logprobs=...)`, which only asks
            # for the *per-token* table; the cumulative figure is free.
            logprobs[j] = getattr(o.outputs[0], "cumulative_logprob", None)
    return texts, usage, reasons, logprobs


def _dispatch_groups(llm, params, prompts, lora_table, *, chat, system) -> list[tuple[list, list]]:
    """The batch's ``(indices, outputs)`` pairs, co-batching adapters where possible.

    vLLM can hold several LoRA adapters resident at once (``max_loras``) and serve them
    **in a single scheduler step**, which is the entire point of multi-adapter serving.
    Looping the adapters and calling `generate` once per group threw that away: N adapters
    meant N sequential passes, each under-filling the GPU, and the batch took N times as
    long as it needed to.

    Newer vLLM accepts a per-prompt *list* of `LoRARequest`s, so the whole batch goes in
    one call. An older version rejects the list; that raises, and the per-adapter loop
    below is the fallback. The two paths produce identical outputs — only the number of
    scheduler steps differs.
    """
    requests = _vllm_inputs(prompts, chat=chat, system=system)
    loras = [lora_table.get(_adapter_of(p)) for p in prompts]
    if len({id(lora) for lora in loras}) > 1:
        try:
            outputs = _submit(llm, requests, _per_request_params(params, prompts), loras, chat=chat)
        except (TypeError, ValueError, AttributeError):
            # This vLLM wants one adapter per call; fall through to the serial loop.
            pass
        else:
            return [(list(range(len(prompts))), list(outputs))]
    groups = []
    for name, idxs in _group_indices_by_adapter(prompts).items():
        outputs = _submit(
            llm,
            [requests[i] for i in idxs],
            _per_request_params(params, [prompts[i] for i in idxs]),
            lora_table.get(name),
            chat=chat,
        )
        groups.append((idxs, list(outputs)))
    return groups


def _submit(llm, requests, params, lora, *, chat):
    """One vLLM call, through `chat` (template applied) or `generate` (raw completion)."""
    if chat:
        return llm.chat(requests, params, lora_request=lora)
    return llm.generate(requests, params, lora_request=lora)


#: Request-dict keys that override a `SamplingParams` field for one row. Kept in step
#: with `llm.requests._PER_ROW_SAMPLING`, which is what puts them on the request.
_PER_ROW_SAMPLING_FIELDS = ("max_tokens", "temperature")


def _per_request_params(params, prompts: list):
    """`params` as-is, or one cloned `SamplingParams` per prompt when any row overrides.

    vLLM accepts either a single `SamplingParams` for the whole call or a list of one per
    prompt, which is what makes a per-row token budget or temperature possible without
    splitting the batch into one call per distinct setting — that would defeat continuous
    batching entirely.

    A batch where no row overrides anything returns the shared object untouched, so the
    common case allocates nothing. Pure, so it tests without vLLM or a GPU.
    """
    overrides = [
        {k: p[k] for k in _PER_ROW_SAMPLING_FIELDS if isinstance(p, dict) and p.get(k) is not None}
        for p in prompts
    ]
    if not any(overrides):
        return params
    out = []
    for override in overrides:
        if not override:
            out.append(params)
            continue
        clone = params.clone()
        for key, value in override.items():
            setattr(clone, key, value)
        out.append(clone)
    return out


def _vllm_inputs(prompts: list, *, chat: bool, system: str | None) -> list:
    """Every request translated to vLLM input, in prompt order."""
    if chat:
        return [_chat_messages(p, system) for p in prompts]
    return [_vllm_request(p) for p in prompts]


def _adapter_of(prompt: object) -> str | None:
    """The request's ``adapter`` tag, or `None` for an untagged (base-model) request."""
    return prompt.get("adapter") if isinstance(prompt, dict) else None


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


def _construct(LLM: object, model: str, enable_lora: bool, kwargs: dict) -> object:
    """Build the vLLM engine, retrying once without a sized window if that window is refused.

    A `max_model_len` above the model's config is a hard error in vLLM, and the model's true
    maximum is unknown until the engine exists, so a rejected auto-sized window falls back to
    the default: auto-sizing can cost throughput but never a run. The original error is quoted
    rather than described, since the retry cannot tell a refused window from a missing model.
    """
    try:
        return LLM(model=model, enable_lora=enable_lora, **kwargs)  # type: ignore[operator]
    except Exception as first:
        if "max_model_len" not in kwargs:
            raise
        import warnings

        warnings.warn(
            f"vllm_engine(max_model_len='auto') proposed a {kwargs['max_model_len']}-token "
            f"window and the engine failed to start ({type(first).__name__}: {first}); "
            f"retrying with its default. If that fails too, pass an explicit max_model_len.",
            UserWarning,
            stacklevel=2,
        )
        del kwargs["max_model_len"]
        return LLM(model=model, enable_lora=enable_lora, **kwargs)  # type: ignore[operator]


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


def _guided_decoding_kwargs(
    guided_json: dict | None,
    guided_regex: str | None,
    guided_choice: list[str] | None,
    guided_grammar: str | None = None,
) -> dict | None:
    """The single `GuidedDecodingParams` kwarg the constraints select, or `None`.

    Pure/vLLM-free so it is unit-testable without a GPU; precedence json > regex > choice
    > grammar (the engine's own order)."""
    for key, value in (("json", guided_json), ("regex", guided_regex)):
        if value is not None:
            return {key: value}
    if guided_choice is not None:
        return {"choice": list(guided_choice)}
    return {"grammar": guided_grammar} if guided_grammar is not None else None


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
