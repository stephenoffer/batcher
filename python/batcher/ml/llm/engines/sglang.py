"""The SGLang backend: an offline, GPU-resident engine built around prefix reuse.

SGLang is the other production offline LLM engine, and it is not a vLLM clone. Its
scheduler is built on **RadixAttention**: every prompt's KV cache is kept in a radix tree
keyed by token prefix, so two requests that share an opening share its computation, and a
third that shares a longer prefix shares more. vLLM's prefix caching does this for one
configured prefix; SGLang does it for whatever the data happens to share, discovered per
batch.

That is why this is worth a second engine rather than a flag. Offline batch inference is
the workload where it pays most, because the prompts in a batch are usually the *same
template* over different rows — a shared system message, a shared instruction, a shared
few-shot block, and a few hundred varying characters at the end. Under a template like
that most of the prefill is the same tokens over and over, and reusing it is the
difference between prefill dominating the run and decode dominating it.

The engine keeps the same `Engine` contract as every other backend, so
`llm_generate`/`ds.ml.generate` reach it by swapping one factory. Everything columnar —
templating, vision inputs, per-row overrides, JSON parsing, usage — is unchanged.
"""

from __future__ import annotations

from typing import Any

from batcher.ml.llm.channels import finish_reason_sink, logprob_sink, usage_sink
from batcher.ml.llm.engines.base import Engine, EngineFactory

__all__ = ["sglang_engine"]

#: Request-dict keys SGLang accepts as a per-row sampling override, mapped to its own
#: `sampling_params` spelling. Kept in step with `llm.requests._PER_ROW_SAMPLING`, which is
#: what puts them on the request in the first place.
_PER_ROW_SAMPLING = {"max_tokens": "max_new_tokens", "temperature": "temperature"}

#: Structured-output keys, in the order they are checked. SGLang takes these on
#: `sampling_params` rather than in a separate object, so constraining a generation is one
#: dict entry — but only one of them may be set at a time, which is checked at construction.
_CONSTRAINTS = ("json_schema", "regex", "ebnf")


def sglang_engine(
    model: str,
    *,
    chat: bool | None = None,
    system: str | None = None,
    sampling: dict[str, object] | None = None,
    json_schema: dict[str, object] | str | None = None,
    regex: str | None = None,
    ebnf: str | None = None,
    lora_paths: dict[str, str] | None = None,
    **engine_kwargs: object,
) -> EngineFactory:
    """An `EngineFactory` backed by SGLang (requires ``batcher-engine[sglang]`` + a GPU).

    The factory builds an ``sglang.Engine`` once per worker and exposes it as the same
    ``list[str] -> list[str]`` callable every other backend does, so switching from
    `vllm_engine` is a one-line change and nothing columnar moves.

    SGLang's RadixAttention prefix cache is on by default and is the reason to pick this
    backend for templated batch work: the shared opening of every row's prompt is prefilled
    once for the batch rather than once per row. Sorting a batch so that rows sharing a
    prefix are adjacent makes the tree hit more often, which is what
    ``llm_generate(sort_by_length=...)`` already does for a different reason.

    Examples:
        .. doctest::

            >>> import batcher as bt  # doctest: +SKIP
            >>> engine = bt.ml.sglang_engine("meta-llama/Llama-3-8B", chat=True)  # doctest: +SKIP
            >>> ds.ml.generate(engine, prompt_column="question").collect()  # doctest: +SKIP

    Args:
        model: the model id or path handed to ``sglang.Engine`` as ``model_path``.
        chat: send each prompt as a **chat conversation**, so SGLang applies the model's own
            chat template. Set this for any instruction-tuned model: without it the prompt
            goes to the completion path in a format the model was never tuned on, and the
            only symptom is worse output. Left unset it stays on the completion path and
            warns if the model turns out to ship a chat template; an explicit ``False`` is
            taken as a decision and is silent.
        system: a system turn prepended to every conversation (with `chat`).
        sampling: SGLang ``sampling_params`` (``temperature``, ``top_p``, ``max_new_tokens``,
            ``stop``, ...). ``max_tokens`` is accepted as an alias for ``max_new_tokens``, so
            a sampling dict written for `vllm_engine` or `http_engine` transfers unchanged.
            Defaults to greedy (``temperature=0``).
        json_schema: constrain generation to this JSON schema (a dict or a schema string) —
            the reliable way to get parseable output; pair with
            ``llm_generate(parse_json=True)``.
        regex: constrain generation to this regular expression.
        ebnf: constrain generation to this EBNF grammar.
        lora_paths: a ``{name: path}`` table of LoRA adapters to serve; a request tagged
            with that name (via ``llm_generate(adapter_column=...)``) is routed to it.
        engine_kwargs: passed to ``sglang.Engine``: ``tp_size`` (tensor parallelism),
            ``dp_size``, ``mem_fraction_static``, ``context_length``, ``quantization``,
            ``disable_radix_cache`` (turn the prefix tree off), ...

    Returns:
        A zero-arg factory building the SGLang-backed `Engine` once per worker.

    Raises:
        PlanError: if more than one of `json_schema` / `regex` / `ebnf` is set.
    """
    constraint = _one_constraint(json_schema, regex, ebnf)
    defaults = _sampling_defaults(sampling, constraint)
    kwargs = dict(engine_kwargs)
    if lora_paths:
        kwargs.setdefault("lora_paths", dict(lora_paths))

    def factory() -> Engine:
        from batcher._internal.optional import require

        SGLEngine = require(
            "sglang", "Engine", feature="sglang_engine", provides="sglang", extra="sglang"
        )
        llm = SGLEngine(model_path=model, **kwargs)
        tokenizer = _worker_tokenizer(llm)
        if chat is None:
            from batcher.ml.llm.engines.templates import warn_if_chat_template_unused

            warn_if_chat_template_unused(tokenizer, model)

        def engine(prompts: list) -> list[str]:
            requests = [
                _render(p, chat=bool(chat), system=system, tokenizer=tokenizer) for p in prompts
            ]
            params = [_row_params(defaults, p) for p in prompts]
            images = _image_data(prompts)
            # `image_data` is passed only when the batch actually carries images. An SGLang
            # build serving a text model rejects the keyword, and sending it unconditionally
            # would make every text batch fail on a parameter it does not need.
            outputs = (
                llm.generate(requests, params, image_data=images)
                if images is not None
                else llm.generate(requests, params)
            )
            texts, usage, reasons, logprobs = _unpack(outputs, len(prompts))
            usage_sink().report(usage)
            finish_reason_sink().report(reasons)
            logprob_sink().report(logprobs)
            engine.last_usage = usage  # the documented legacy channel
            return texts

        # `close` is the teardown contract `core.udf.lifecycle` and `InferencePool` look for.
        # An SGLang engine owns a scheduler process group and the whole KV cache; leaving it
        # to the garbage collector means two generations of them share one device the next
        # time a script runs two stages back to back.
        engine.close = lambda: _shutdown(llm)
        return engine

    return factory


def _one_constraint(
    json_schema: dict[str, object] | str | None, regex: str | None, ebnf: str | None
) -> dict[str, Any]:
    """The single structured-output constraint, as a `sampling_params` fragment.

    SGLang applies exactly one grammar per request and silently prefers one over another
    when given several, which produces output constrained by a rule the caller did not
    choose. Refusing here is the difference between a clear error at construction and a
    plausible-looking wrong shape a thousand rows later.

    Raises:
        PlanError: If more than one constraint is given.
    """
    import json as _json

    given = {"json_schema": json_schema, "regex": regex, "ebnf": ebnf}
    named = [key for key in _CONSTRAINTS if given[key] is not None]
    if len(named) > 1:
        from batcher._internal.errors import PlanError

        raise PlanError(
            f"sglang_engine accepts one structured-output constraint, got {named}. "
            "A generation is constrained by one grammar; pick the one you mean."
        )
    if not named:
        return {}
    key = named[0]
    value = given[key]
    # SGLang wants the schema as a JSON *string*; a dict is what a caller has (and what
    # `json_schema()` builds), so it is serialized here rather than at every call site.
    if key == "json_schema" and not isinstance(value, str):
        value = _json.dumps(value)
    return {key: value}


def _sampling_defaults(
    sampling: dict[str, object] | None, constraint: dict[str, Any]
) -> dict[str, Any]:
    """The engine-wide `sampling_params`, with the cross-backend aliases normalized.

    ``max_tokens`` is what vLLM, OpenAI and Anthropic all call the generation budget, and
    SGLang calls it ``max_new_tokens``. Accepting both is what lets one `sampling` dict move
    between backends instead of being rewritten — and silently dropping the alias would cap
    every generation at SGLang's own default with nothing to say so.
    """
    params: dict[str, Any] = {"temperature": 0.0, **(sampling or {})}
    if "max_tokens" in params:
        params.setdefault("max_new_tokens", params.pop("max_tokens"))
    params.update(constraint)
    return params


def _row_params(defaults: dict[str, Any], request: Any) -> dict[str, Any]:
    """One request's `sampling_params`: the engine defaults, plus that row's overrides.

    Returns the shared defaults object unchanged when a row overrides nothing, so the common
    case allocates no per-row dict.
    """
    if not isinstance(request, dict):
        return defaults
    overrides = {
        target: request[source] for source, target in _PER_ROW_SAMPLING.items() if source in request
    }
    return {**defaults, **overrides} if overrides else defaults


def _image_data(prompts: list) -> list | None:
    """The batch's per-request images for SGLang's ``image_data``, or `None` if it has none.

    A vision request carries its image beside the prompt rather than in it, so it travels as a
    parallel list aligned with the requests — one entry per request, `None` for the text-only
    rows, which is the shape SGLang expects for a mixed batch.

    Returning `None` for a batch with no images at all is what keeps this off the text path
    entirely: the alternative, dropping the image and generating from the prompt alone, is the
    worst possible outcome here, because the model answers a question about a picture it never
    saw and nothing in the output says so.
    """
    images = [request.get("image") if isinstance(request, dict) else None for request in prompts]
    return images if any(image is not None for image in images) else None


def _render(request: Any, *, chat: bool, system: str | None, tokenizer: Any) -> str:
    """One request as the text SGLang should generate from.

    On the completion path that is the prompt itself. On the chat path the model's own
    template is applied *here*, with the worker's tokenizer, because ``Engine.generate``
    takes text rather than a conversation — so a caller asking for `chat` on a backend that
    has no chat entry point still gets the template applied rather than silently skipped.
    A tokenizer that cannot apply one leaves the prompt alone, which is the completion path
    and is at least not wrong in a way that hides.

    The image, when there is one, is *not* rendered into the text: it travels beside the
    request in `_image_data`.
    """
    prompt, _image, _ = _unpack_request(request)
    if not chat or tokenizer is None:
        return prompt
    apply = getattr(tokenizer, "apply_chat_template", None)
    if apply is None:
        return prompt
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    try:
        return apply(messages, tokenize=False, add_generation_prompt=True)
    except Exception:  # pragma: no cover - a model with no template raises here
        return prompt


def _unpack_request(request: Any) -> tuple[str, Any, dict]:
    """Split one request into ``(prompt, image, overrides)`` using the shared rule."""
    from batcher.ml.llm.engines.base import unpack_request

    return unpack_request(request, tuple(_PER_ROW_SAMPLING))


def _unpack(outputs: Any, count: int) -> tuple[list[str], list, list, list]:
    """SGLang's per-request results as the four aligned signal lists the engine reports.

    A single-request call returns one dict rather than a list of one, which is a shape that
    silently produced ``count`` copies of the same character when indexed. Normalizing here
    is why the engine's return is always one string per request.
    """
    rows = outputs if isinstance(outputs, list) else [outputs]
    texts: list[str] = [""] * count
    usage: list[tuple[int | None, int | None] | None] = [None] * count
    reasons: list[str | None] = [None] * count
    logprobs: list[float | None] = [None] * count
    for index, row in enumerate(rows[:count]):
        if not isinstance(row, dict):
            texts[index] = str(row)
            continue
        texts[index] = str(row.get("text", ""))
        meta = row.get("meta_info") or {}
        usage[index] = (meta.get("prompt_tokens"), meta.get("completion_tokens"))
        reasons[index] = _finish_reason(meta.get("finish_reason"))
        logprobs[index] = _cumulative_logprob(meta)
    return texts, usage, reasons, logprobs


def _finish_reason(reason: Any) -> str | None:
    """SGLang's finish reason as the plain string every other backend reports.

    It is reported as ``{"type": "stop", ...}``; the rest of the engine — and the
    ``finish_reason`` column `generate` writes — expects ``"stop"``/``"length"``, so a raw
    dict would land a struct in a string column.
    """
    if isinstance(reason, dict):
        return reason.get("type")
    return reason if reason is None or isinstance(reason, str) else str(reason)


def _cumulative_logprob(meta: dict) -> float | None:
    """The generation's summed log-probability, when the engine was asked for logprobs.

    SGLang reports per-token pairs rather than a running total, so the total is summed here
    to match what `vllm_engine` and `http_engine` put on the same channel. `None` when
    logprobs were not requested, which is the default.
    """
    tokens = meta.get("output_token_logprobs")
    if not isinstance(tokens, list) or not tokens:
        return None
    total = 0.0
    for entry in tokens:
        value = entry[0] if isinstance(entry, (list, tuple)) and entry else None
        if isinstance(value, (int, float)):
            total += float(value)
    return total


def _worker_tokenizer(llm: Any) -> Any | None:
    """The worker's own tokenizer, when SGLang exposes one, else `None`.

    Used for the chat template and for the same reason `vllm_engine` reaches for it: the only
    tokenizer that agrees with the model is the model's.
    """
    for name in ("get_tokenizer", "tokenizer"):
        attribute = getattr(llm, name, None)
        if attribute is None:
            continue
        try:
            return attribute() if callable(attribute) else attribute
        except Exception:  # pragma: no cover - version differences
            continue
    return None


def _shutdown(llm: Any) -> None:
    """Release the engine's scheduler processes and KV cache, best effort."""
    import contextlib

    for name in ("shutdown", "close"):
        stop = getattr(llm, name, None)
        if callable(stop):
            with contextlib.suppress(Exception):
                stop()
            return
