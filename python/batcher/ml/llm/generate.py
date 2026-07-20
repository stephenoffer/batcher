"""LLM batch generation — the columnar half of offline text generation.

Builds each row's request from its columns (a `template`, an image column for
vision-language models, a per-row LoRA adapter tag), hands the batch to an `Engine`,
and appends the generated text — optionally parsed as JSON into a struct column, and
with per-row token usage. The engine is built **once per worker** (the load-once
pattern) and does its *own* continuous batching, so no outer fixed batch size is
imposed: an outer batch-size PID would fight vLLM's scheduler.

Two entry points over the same core: `llm_generate` streams `RecordBatch`es (the
library form), and `llm_udf` wraps it as a load-once class UDF so `ds.ml.generate`
can schedule it on GPU actors through `map_batches`.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING

from batcher.ml.inference import InferencePool
from batcher.ml.llm.engines import Engine, EngineFactory

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = ["llm_generate", "llm_udf"]


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

    Examples:
        .. doctest::

            >>> import batcher as bt  # doctest: +SKIP
            >>> from batcher.ml import llm_generate, vllm_engine  # doctest: +SKIP
            >>> engine = vllm_engine("meta-llama/Llama-3-8B")  # doctest: +SKIP
            >>> batches = ds.iter_batches()  # doctest: +SKIP
            >>> out = list(llm_generate(batches, engine, prompt_column="q"))  # doctest: +SKIP

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
        num_workers: pool size — engines built, and batches generated, in parallel.
        target_batch_rows: rows per batch handed to an engine (no latency controller —
            the engine owns its own batching).

    Yields:
        Each input batch with `output_column` appended, in order.
    """

    def make_worker() -> Callable[[pa.RecordBatch], pa.RecordBatch]:
        engine = engine_factory()  # built once per worker
        return lambda batch: _generate_batch(
            engine,
            batch,
            prompt_column=prompt_column,
            output_column=output_column,
            template=template,
            image_column=image_column,
            adapter_column=adapter_column,
            parse_json=parse_json,
            usage=usage,
        )

    # Offline bulk generation is the throughput objective: engage the hill-climbing
    # autobatcher (as `embed` does) so more prompts reach the engine's scheduler per step —
    # under the default "latency" objective, with no `target_latency_ms`, no controller ran
    # and the batch size was fixed.
    pool = InferencePool(
        make_worker,
        num_workers=num_workers,
        target_batch_rows=target_batch_rows,
        objective="throughput",
    )
    yield from pool.run(batches)


def _generate_batch(
    engine: Engine,
    batch: pa.RecordBatch,
    *,
    prompt_column: str,
    output_column: str,
    template: str | None,
    image_column: str | None,
    adapter_column: str | None,
    parse_json: bool,
    usage: bool,
) -> pa.RecordBatch:
    """One batch through the engine: build requests, generate, append the columns.

    The single place the columnar work lives, shared by the streaming `llm_generate`
    and the `llm_udf` class UDF, so the two entry points cannot drift.
    """
    import pyarrow as pa

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


def llm_udf(
    engine_factory: EngineFactory,
    *,
    prompt_column: str,
    output_column: str = "response",
    template: str | None = None,
    image_column: str | None = None,
    adapter_column: str | None = None,
    parse_json: bool = False,
    usage: bool = False,
) -> type:
    """A **load-once class UDF** that appends an LLM-generated column to each batch.

    The same generation as `llm_generate`, packaged as a class so `map_batches` builds
    the engine once per worker and schedules it on GPU actors — which is what lets
    `ds.ml.generate` reuse the whole `num_gpus`/`concurrency`/`accelerator_type`
    machinery instead of owning a second scheduler. A plain function would rebuild the
    engine (reloading the model) on every batch.

    Examples:
        .. doctest::

            >>> from batcher.ml import llm_udf, vllm_engine  # doctest: +SKIP
            >>> engine = vllm_engine("meta-llama/Llama-3-8B")  # doctest: +SKIP
            >>> udf = llm_udf(engine, prompt_column="question")  # doctest: +SKIP
            >>> ds.ml.map_batches(udf, num_gpus=1).collect()  # doctest: +SKIP

    Args:
        engine_factory: zero-arg callable returning an `Engine`; called once per worker.
        prompt_column: the text column to send (ignored when `template` is set).
        output_column: name of the appended generated column.
        template: optional ``str.format`` template over the row's columns.
        image_column: optional image column for vision-language models.
        adapter_column: optional per-row LoRA adapter name.
        parse_json: parse each output as JSON into a struct column (null on error).
        usage: also append ``prompt_tokens`` / ``completion_tokens``.

    Returns:
        A class whose instances map a `pyarrow.RecordBatch` to the batch plus the
        generated column(s).
    """

    class _LlmGenerate:
        """Holds one engine for the worker's lifetime; called once per batch."""

        def __init__(self) -> None:
            self._engine = engine_factory()

        def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
            return _generate_batch(
                self._engine,
                batch,
                prompt_column=prompt_column,
                output_column=output_column,
                template=template,
                image_column=image_column,
                adapter_column=adapter_column,
                parse_json=parse_json,
                usage=usage,
            )

    return _LlmGenerate


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
    if len(pairs) != n:
        # Otherwise this surfaces far from its cause, as an opaque "arrays must all be
        # the same length" from `RecordBatch.from_arrays`, with nothing naming the
        # engine. A mismatch means the engine reported usage for a different number of
        # requests than it returned outputs for, which would silently misalign every
        # token count against its row.
        from batcher._internal.errors import BackendError

        msg = (
            f"{type(engine).__name__}.last_usage reported {len(pairs)} usage pairs for "
            f"{n} generated outputs; they must correspond one-to-one and in prompt "
            "order. Pass usage=False to skip token accounting for this engine."
        )
        raise BackendError(msg)
    prompt = [p[0] if p else None for p in pairs]
    completion = [p[1] if p else None for p in pairs]
    return pa.array(prompt, type=pa.int64()), pa.array(completion, type=pa.int64())
