"""LLM batch generation — the columnar half of offline text generation.

Builds each row's request from its columns (a `template`, an image column for
vision-language models, a per-row LoRA adapter tag, per-row sampling overrides), hands
the batch to an `Engine`, and appends the generated text — optionally parsed as JSON
into a struct column, and with per-row token usage, finish reason, and logprob.

Two entry points over **one** implementation: `llm_udf` packages the work as a
load-once class UDF (what `ds.ml.generate` schedules on GPU actors), and `llm_generate`
streams `RecordBatch`es for library use by instantiating that same class. Both describe
what they want with a single `GenerateSpec`, so an option added to one reaches the other
by construction rather than by remembering to.

The engine is built **once per worker** and does its *own* continuous batching, so no
outer batch size is imposed by default: an outer fixed batch would fight vLLM's
scheduler, and re-chunking the caller's stream would change its memory shape. Both are
available through `llm_generate(target_batch_rows=...)` when a caller wants them.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Any

from batcher.ml.llm.columns import _usage_columns as _usage_columns  # test seam (relocated)
from batcher.ml.llm.engines import Engine, EngineFactory
from batcher.ml.llm.requests import (
    GenerateSpec,
    _build_requests,
    _length_sorted_order,
    _restore_order,
)
from batcher.ml.llm.requests import _decode_image_inputs as _decode_image_inputs  # test seam

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = ["llm_generate", "llm_udf"]


def llm_udf(
    engine_factory: EngineFactory,
    *,
    prompt_column: str,
    output_column: str = "response",
    template: str | None = None,
    image_column: str | None = None,
    adapter_column: str | None = None,
    max_tokens_column: str | None = None,
    temperature_column: str | None = None,
    parse_json: bool = False,
    usage: bool = False,
    finish_reason: bool = False,
    logprobs: bool = False,
) -> type:
    """A **load-once class UDF** that appends an LLM-generated column to each batch.

    The implementation both entry points share, packaged as a class so `map_batches`
    builds the engine once per worker and schedules it on GPU actors — which is what lets
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
        max_tokens_column: optional column giving each row its own token budget, so one
            batch can mix a 16-token classification with a 2000-token summary instead of
            paying the longest budget for every row. A null uses the engine's default.
        temperature_column: optional column giving each row its own sampling temperature
            (a null uses the engine's default), so a factual extraction and a creative
            rewrite can share one pass over the data.
        parse_json: parse each output as JSON into a struct column (null on error).
        usage: also append ``prompt_tokens`` / ``completion_tokens``.
        finish_reason: also append a ``finish_reason`` column, so a generation truncated
            at ``max_tokens`` is detectable rather than silently corrupting a parse.
        logprobs: also append a ``logprob`` column holding the generation's cumulative
            log-probability — the model's own confidence, for routing the least certain
            rows to review. Null for an engine that does not report one.

    Returns:
        A class whose instances map a `pyarrow.RecordBatch` to the batch plus the
        generated column(s).
    """
    spec = GenerateSpec(
        prompt_column=prompt_column,
        output_column=output_column,
        template=template,
        image_column=image_column,
        adapter_column=adapter_column,
        max_tokens_column=max_tokens_column,
        temperature_column=temperature_column,
        parse_json=parse_json,
        usage=usage,
        finish_reason=finish_reason,
        logprobs=logprobs,
    )

    class _LlmGenerate:
        """Holds one engine for the worker's lifetime; called once per batch."""

        def __init__(self) -> None:
            self._engine = engine_factory()

        def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
            return _generate_batch(self._engine, batch, spec)

    return _LlmGenerate


def llm_generate(
    batches: Iterable[pa.RecordBatch],
    engine_factory: EngineFactory,
    *,
    prompt_column: str,
    output_column: str = "response",
    template: str | None = None,
    image_column: str | None = None,
    adapter_column: str | None = None,
    max_tokens_column: str | None = None,
    temperature_column: str | None = None,
    parse_json: bool = False,
    usage: bool = False,
    finish_reason: bool = False,
    logprobs: bool = False,
    num_workers: int = 1,
    target_batch_rows: int | None = None,
) -> Iterator[pa.RecordBatch]:
    """Append an LLM-generated `output_column` to each batch.

    The streaming form of `llm_udf`, and built from it, so the two cannot produce
    different columns for the same input. By default each input batch comes back with the
    generated column appended and its **row boundaries unchanged**, generated by a single
    engine — the same shape `ds.ml.generate` produces. `num_workers` and
    `target_batch_rows` opt into extra scheduling on top of that; see their descriptions
    for what each costs.

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
        max_tokens_column: optional per-row token budget; a null uses the engine default.
        temperature_column: optional per-row sampling temperature; a null uses the
            engine default.
        parse_json: parse each output as JSON into a struct column (guided/structured
            decoding); on a parse error the row's value is null.
        usage: also append integer ``prompt_tokens`` and ``completion_tokens`` columns
            (the per-row token counts the engine reported — `vllm_engine` and
            `http_engine` do). Aggregate them to track cost (tokens * price) or
            throughput. Null for an engine that does not report usage.
        finish_reason: also append a string ``finish_reason`` column (``"stop"`` when the
            model finished, ``"length"`` when it was cut off at ``max_tokens``). Without
            it a truncated generation is indistinguishable from a complete one, and it
            silently corrupts a downstream `parse_json`. Null for an engine that does not
            report one.
        logprobs: also append a float ``logprob`` column (the generation's cumulative
            log-probability). Null for an engine that does not report one.
        num_workers: how many engines to build **in this process** and run batches
            across. Leave at ``1`` for a GPU-resident engine: each worker calls
            `engine_factory` again, so ``2`` loads two full copies of the weights onto
            the same device. Raise it for a network-bound engine such as `http_engine`,
            where the workers are waiting on sockets rather than holding a model.
        target_batch_rows: re-chunk the stream to about this many rows per engine call,
            hill-climbing the size for throughput. Unset (the default) preserves the
            caller's batch boundaries. Setting it helps when the caller's batches are far
            smaller than the engine's scheduler can fill, and costs the memory of holding
            that many rows.

    Yields:
        Each input batch with `output_column` appended, in order. With
        `target_batch_rows` set the boundaries are the re-chunked ones instead.
    """
    worker_class = llm_udf(
        engine_factory,
        prompt_column=prompt_column,
        output_column=output_column,
        template=template,
        image_column=image_column,
        adapter_column=adapter_column,
        max_tokens_column=max_tokens_column,
        temperature_column=temperature_column,
        parse_json=parse_json,
        usage=usage,
        finish_reason=finish_reason,
        logprobs=logprobs,
    )
    if num_workers <= 1 and target_batch_rows is None:
        # The documented default: one engine, the caller's batches, in order. No pool,
        # because a pool here would re-chunk the stream and build a second engine — both
        # of which this function's contract says it does not do.
        worker = worker_class()
        for batch in batches:
            yield worker(batch)
        return
    yield from _pooled(batches, worker_class, num_workers, target_batch_rows)


def _pooled(
    batches: Iterable[pa.RecordBatch],
    worker_class: type,
    num_workers: int,
    target_batch_rows: int | None,
) -> Iterator[pa.RecordBatch]:
    """Run the same worker across an `InferencePool`, re-chunking toward throughput.

    Only reached when the caller explicitly asks for more than one worker or a target
    batch size, since both change the shape `llm_generate` otherwise promises.
    """
    from batcher.ml.inference import InferencePool

    pool = InferencePool(
        worker_class,
        num_workers=num_workers,
        target_batch_rows=target_batch_rows or 256,
        objective="throughput",
    )
    yield from pool.run(batches)


def _generate_batch(
    engine: Engine, batch: pa.RecordBatch, spec: GenerateSpec | None = None, **spec_kwargs: Any
) -> pa.RecordBatch:
    """One batch through the engine: build requests, generate, append the columns.

    The single place the columnar work lives, reached by both entry points, so the two
    cannot drift. Pass a `GenerateSpec`, or the spec's fields as keywords — the latter is
    the seam the engine tests drive, so a caller need not construct a spec to exercise
    one batch.

    Requests are dispatched **length-sorted** and the results un-permuted (see
    `_length_sorted_order`), so the appended columns line up with the caller's rows
    exactly as they did before.
    """
    import pyarrow as pa

    if spec is None:
        spec = GenerateSpec(**spec_kwargs)

    from batcher.ml.llm.channels import finish_reason_sink, logprob_sink, usage_sink

    requests = _build_requests(spec, batch)
    order = _length_sorted_order(requests)
    with usage_sink().capture(), finish_reason_sink().capture(), logprob_sink().capture():
        generated = list(engine([requests[i] for i in order]))
        reported = _Reported(
            usage=usage_sink().collected(),
            reasons=finish_reason_sink().collected(),
            logprobs=logprob_sink().collected(),
        )
    if len(generated) != len(order):
        raise _count_mismatch(engine, len(generated), len(order))
    outputs = _restore_order(generated, order)

    arrays = [batch.column(i) for i in range(batch.num_columns)]
    arrays.append(_output_column(outputs, spec))
    arrays += _reported_columns(engine, outputs, order, reported, spec)
    return pa.RecordBatch.from_arrays(arrays, names=[*batch.schema.names, *spec.appended_columns])


class _Reported:
    """What the engine pushed into the per-call channels, in dispatch order."""

    __slots__ = ("logprobs", "reasons", "usage")

    def __init__(self, usage: list | None, reasons: list | None, logprobs: list | None) -> None:
        self.usage = usage
        self.reasons = reasons
        self.logprobs = logprobs


def _output_column(outputs: list, spec: GenerateSpec) -> Any:
    """The generated column: parsed JSON structs, or plain strings."""
    import pyarrow as pa

    from batcher.ml.llm.columns import _safe_json

    if spec.parse_json:
        return pa.array([_safe_json(o) for o in outputs])
    return pa.array([str(o) for o in outputs], type=pa.string())


def _reported_columns(
    engine: Engine, outputs: list, order: list[int], reported: _Reported, spec: GenerateSpec
) -> list:
    """The optional side-channel columns, in the order `GenerateSpec.appended_columns`
    names them — the one place that ordering is decided for both."""
    from batcher.ml.llm.columns import _finish_reason_column, _logprob_column, _usage_columns

    n = len(outputs)
    arrays: list = []
    if spec.usage:
        arrays.extend(_usage_columns(engine, n, reported.usage, order))
    if spec.finish_reason:
        arrays.append(_finish_reason_column(reported.reasons, n, order))
    if spec.logprobs:
        arrays.append(_logprob_column(reported.logprobs, n, order))
    return arrays


def _count_mismatch(engine: object, got: int, expected: int) -> Exception:
    from batcher._internal.errors import BackendError

    return BackendError(
        f"{type(engine).__name__} returned {got} generations for {expected} requests; "
        "an engine must return exactly one string per request, in request order."
    )
