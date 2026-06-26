"""LLM batch inference — the Ray Data LLM competitor (offline text generation).

Run a text-generation engine (vLLM, or any callable) over millions of rows. The
engine is built **once per worker** (the load-once pattern) and does its *own*
continuous batching, so Batcher feeds it each input batch as a whole request list
and imposes **no** outer fixed batch size — an outer batch-size PID would fight
vLLM's scheduler (the distinct LLM-stage contract, vs the embeddings stage which
*does* adapt its batch size). On top of that the operator handles the surrounding
columnar work: build each prompt from row columns via a `template`, and optionally
parse each structured/JSON output into typed columns.

The engine is injected as a factory, so this works with vLLM, an OpenAI client, or a
deterministic test double — the engine never depends on a specific library. A vLLM
adapter (`vllm_engine`) lives behind the optional ``batcher-engine[vllm]`` extra.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import TYPE_CHECKING

from batcher.ml.inference import InferencePool

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = ["Engine", "EngineFactory", "http_engine", "llm_generate", "vllm_engine"]

# Maps a list of prompts to a list of generated strings (one per prompt, in order).
Engine = Callable[[list[str]], Sequence[str]]
EngineFactory = Callable[[], Engine]


def _render(template: str | None, column: str, batch: pa.RecordBatch) -> list[str]:
    """The prompt for each row: ``column`` verbatim, or `template` formatted with the
    row's columns (``"{system} Q: {question}"``-style ``str.format`` placeholders)."""
    if template is None:
        return [str(v) for v in batch.column(column).to_pylist()]
    rows = batch.to_pylist()
    return [template.format(**row) for row in rows]


def _build_requests(
    template: str | None,
    prompt_column: str,
    image_column: str | None,
    adapter_column: str | None,
    batch: pa.RecordBatch,
) -> list:
    """Per-row engine requests: plain prompt strings, or ``{prompt, image?, adapter?}``
    dicts when an `image_column` (vision-language) or `adapter_column` (per-row LoRA) is
    given. A null image/adapter for a row drops that key (text-only / base model)."""
    prompts = _render(template, prompt_column, batch)
    if image_column is None and adapter_column is None:
        return prompts
    n = len(prompts)
    images = _decode_image_inputs(batch.column(image_column)) if image_column else [None] * n
    adapters = batch.column(adapter_column).to_pylist() if adapter_column else [None] * n
    requests = []
    for prompt, image, adapter in zip(prompts, images, adapters, strict=True):
        request: dict = {"prompt": prompt}
        if image is not None:
            request["image"] = image
        if adapter is not None:
            request["adapter"] = adapter
        requests.append(request)
    return requests


def _decode_image_inputs(column: pa.Array) -> list:
    """A list of PIL images for a column of raw image bytes or decoded pixel tensors.

    Bytes → ``PIL.Image.open``; a fixed-shape-tensor ``(H, W, 3)`` → ``Image.fromarray``.
    Null rows yield ``None`` (the model sees a text-only request for that row)."""
    import io as _io

    from batcher.io.formats.ml.tensor import is_tensor_column

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional extra
        from batcher._internal.errors import BackendError

        msg = "vision LLM input needs Pillow: pip install 'batcher-engine[image]'"
        raise BackendError(msg) from exc

    if is_tensor_column(column):
        if hasattr(column, "combine_chunks"):
            column = column.combine_chunks()
        return [Image.fromarray(row) for row in column.to_numpy_ndarray()]
    return [None if b is None else Image.open(_io.BytesIO(b)) for b in column.to_pylist()]


def llm_generate(
    batches: Iterable[pa.RecordBatch],
    engine_factory: EngineFactory,
    *,
    prompt_column: str,
    output_column: str = "response",
    template: str | None = None,
    image_column: str | None = None,
    adapter_column: str | None = None,
    parse_json: bool = False,
    usage: bool = False,
    num_workers: int = 2,
    target_batch_rows: int = 256,
) -> Iterator[pa.RecordBatch]:
    """Append an LLM-generated `output_column` to each batch.

    Args:
        batches: an iterable of `pyarrow.RecordBatch`.
        engine_factory: zero-arg callable returning an engine (``list[str]`` →
            sequence of strings); called once per worker so the model loads once.
        prompt_column: the text column to send (ignored if `template` is set, which
            builds prompts from any columns).
        output_column: name of the appended generated column.
        template: optional ``str.format`` template over the row's columns.
        image_column: optional image column (raw bytes, or a decoded ``(H, W, 3)``
            tensor) for **vision-language** models. Each request becomes
            ``{"prompt": text, "image": PIL.Image}``; the engine must be vision-capable
            (`vllm_engine` on a multimodal model handles it).
        adapter_column: optional column naming the **LoRA adapter** to use per row
            (multi-adapter serving). The engine routes each row to that adapter; a null
            uses the base model. Pair with ``vllm_engine(lora_paths={name: path})``.
        parse_json: parse each output as JSON into a struct column (guided/structured
            decoding); on a parse error the row's value is null.
        usage: also append integer ``prompt_tokens`` and ``completion_tokens`` columns
            (the per-row token counts the engine reported — `vllm_engine` and
            `http_engine` do). Aggregate them to track cost (tokens * price) or
            throughput. Null for an engine that does not report usage.
        num_workers / target_batch_rows: forwarded to `InferencePool` (no latency
            controller — the engine owns its own batching).

    Yields:
        Each input batch with `output_column` appended, in order.
    """
    import pyarrow as pa

    def make_worker() -> Callable[[pa.RecordBatch], pa.RecordBatch]:
        engine = engine_factory()  # built once per worker

        def worker(batch: pa.RecordBatch) -> pa.RecordBatch:
            requests = _build_requests(template, prompt_column, image_column, adapter_column, batch)
            outputs = list(engine(requests))
            if parse_json:
                col = pa.array([_safe_json(o) for o in outputs])
            else:
                col = pa.array([str(o) for o in outputs], type=pa.string())
            arrays = [batch.column(i) for i in range(batch.num_columns)] + [col]
            names = [*batch.schema.names, output_column]
            if usage:
                prompt_toks, completion_toks = _usage_columns(engine, len(outputs))
                arrays += [prompt_toks, completion_toks]
                names += ["prompt_tokens", "completion_tokens"]
            return pa.RecordBatch.from_arrays(arrays, names=names)

        return worker

    pool = InferencePool(make_worker, num_workers=num_workers, target_batch_rows=target_batch_rows)
    yield from pool.run(batches)


def _safe_json(text: str) -> object | None:
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


def _usage_columns(engine: object, n: int):
    """Per-row `(prompt_tokens, completion_tokens)` Int64 arrays from the engine's
    `last_usage` (set on its most recent call), or all-null when it reports none.

    `last_usage` is `n` `(prompt_tokens, completion_tokens)` pairs in prompt order; a
    `None` pair (a request whose usage the engine couldn't report) yields nulls for that
    row."""
    import pyarrow as pa

    reported = getattr(engine, "last_usage", None)
    pairs = list(reported) if reported is not None else [None] * n
    prompt = [p[0] if p else None for p in pairs]
    completion = [p[1] if p else None for p in pairs]
    return pa.array(prompt, type=pa.int64()), pa.array(completion, type=pa.int64())


def vllm_engine(
    model: str,
    *,
    sampling: dict[str, object] | None = None,
    guided_json: dict[str, object] | None = None,
    guided_regex: str | None = None,
    lora_path: str | None = None,
    lora_paths: dict[str, str] | None = None,
    quantization: str | None = "auto",
    **engine_kwargs: object,
) -> EngineFactory:
    """An `EngineFactory` backed by vLLM (requires ``batcher-engine[vllm]`` + a GPU).

    Returns a factory that builds a vLLM engine once per worker and exposes it as a
    ``list[str] -> list[str]`` callable, with full control over modern batch-inference
    knobs:

    * `sampling` — `SamplingParams` kwargs (``temperature``, ``top_p``, ``max_tokens``,
      ``stop``, ``n``, ``seed``, ...). Defaults to greedy (``temperature=0``).
    * `guided_json` / `guided_regex` — **structured output**: constrain generation to a
      JSON schema or regex via vLLM's guided decoding (the reliable way to get parseable
      output; pair with ``llm_generate(parse_json=True)``).
    * `lora_path` — serve a single LoRA adapter on top of the base `model` (applied to
      every row that does not name another adapter).
    * `lora_paths` — a ``{name: path}`` table of adapters to **multiplex**: a request
      tagged with that name (via ``llm_generate(adapter_column=...)``) is routed to it,
      so one engine serves many adapters in one batch. Rows are grouped by adapter and
      each group is generated together; output order is preserved. Set the vLLM
      ``enable_lora``/``max_loras``/``max_cpu_loras`` engine kwargs for the adapter cache.
    * `quantization` — ``"auto"`` (default) picks ``"fp8"`` on GPUs with native FP8
      tensor cores (NVIDIA Ada L4/L40S, Hopper H100), where it halves weight/KV-cache
      memory at <1% quality loss, and keeps native precision (BF16/FP16) elsewhere —
      the zero-config win Ray Data users must select by hand per GPU. Pass an explicit
      string (``"fp8"``, ``"awq"``, ...) to force it, or ``None`` to disable.
    * `engine_kwargs` — pass through to ``vllm.LLM``: ``max_model_len``,
      ``gpu_memory_utilization``, ``tensor_parallel_size`` (tensor parallelism for a
      model larger than one GPU), ``speculative_config`` /
      ``spec_decode_disable_by_queue_size`` (speculative decoding), ``enable_lora`` /
      ``max_loras`` / ``max_cpu_loras`` (multi-adapter serving), ...

    Zero-config batch defaults: **prefix caching** and **chunked prefill** are enabled
    unless you set them — both are throughput/TTFT wins for offline batch (a shared
    system prompt is encoded once; long prefills interleave with decode) that Ray Data
    users must turn on by hand. Any value you pass in `engine_kwargs` wins. (`max_model_len`
    is left to vLLM's model default — auto-sizing it to the data needs the worker tokenizer
    and is a follow-on; a char-heuristic could truncate prompts and corrupt output.)
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
            texts, usage = _generate_routed(llm, params, prompts, lora_table)
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


def _generate_routed(llm, params, prompts: list, lora_table: dict):
    """Generate with per-row LoRA routing: group requests by adapter, run each group with
    that adapter's `LoRARequest` (from `lora_table`, `None` key = base/single adapter),
    and reassemble outputs + per-prompt token usage in input order.

    Pure but for `llm.generate`, so it tests with a stub `llm` (no vLLM/GPU)."""
    texts: list[str | None] = [None] * len(prompts)
    usage: list[tuple[int, int] | None] = [None] * len(prompts)
    for name, idxs in _group_indices_by_adapter(prompts).items():
        requests = [_vllm_request(prompts[i]) for i in idxs]
        outputs = llm.generate(requests, params, lora_request=lora_table.get(name))
        for j, o in zip(idxs, outputs, strict=True):
            texts[j] = o.outputs[0].text
            usage[j] = (len(o.prompt_token_ids), len(o.outputs[0].token_ids))
    return texts, usage


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
    timeout: float = 60.0,
    concurrency: int = 8,
) -> EngineFactory:
    """An `EngineFactory` calling an OpenAI-compatible HTTP endpoint (the Ray Data
    ``HttpRequestProcessor`` analog) — batch inference against a *served* model.

    Targets ``{base_url}/chat/completions`` (``chat=True``, the default) or
    ``/completions``; with chat, the **server applies the model's chat template**, so a
    plain prompt string is wrapped as a user message (with an optional `system`
    message). Works against vLLM's OpenAI server, llama.cpp, or a hosted API. `api_key`
    sets the bearer token; `max_tokens`/`temperature` control decoding.

    The prompts in each batch are sent **concurrently** over up to `concurrency`
    in-flight requests (input order preserved), so a batch's latency is the slowest
    request rather than their sum — the right shape for a network-bound served endpoint
    where one request barely uses the connection. Each request still retries with
    backoff on the 429s a hosted API returns. Set `concurrency=1` to serialize.
    """

    def factory() -> Engine:
        from concurrent.futures import ThreadPoolExecutor

        from batcher.ml.serving.http import post_json

        url = base_url.rstrip("/") + ("/chat/completions" if chat else "/completions")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        def call_one(prompt: str) -> tuple[str, tuple[int | None, int | None]]:
            body = _openai_body(model, prompt, chat, system, max_tokens, temperature)
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
    model: str, prompt: str, chat: bool, system: str | None, max_tokens: int, temperature: float
) -> dict[str, object]:
    common = {"model": model, "max_tokens": max_tokens, "temperature": temperature}
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
