"""The `Dataset.ml` namespace — batch inference / embedding / model UDFs.

Breadth on `Dataset` lives on accessors, not new methods (the Polars pattern, and
the v2 maintainability contract). This is the ML/multimodal surface: apply a model
over Arrow batches, optionally loading it once per worker and scheduling it on GPU
actors while preprocessing stays on CPU — the heterogeneous pipeline Ray Data
specializes in. Reached as `ds.ml.infer(...)` / `ds.ml.embed(...)`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from batcher.plan.logical import MapBatches

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = ["DatasetML"]


def _validate_concurrency(concurrency: int | tuple[int, int] | None) -> None:
    """Validate the `map_batches` actor-pool size (an int or a ``(min, max)`` tuple)."""
    if concurrency is None:
        return
    from batcher._internal.errors import PlanError

    if isinstance(concurrency, tuple):
        if len(concurrency) != 2 or not (0 < concurrency[0] <= concurrency[1]):
            raise PlanError(
                f"concurrency tuple must be (min, max) with 0 < min <= max, got {concurrency}"
            )
    elif not (isinstance(concurrency, int) and concurrency > 0):
        raise PlanError(
            f"concurrency must be a positive int or (min, max) tuple, got {concurrency}"
        )


def _warn_if_model_reloads(fn: object, num_gpus: float) -> None:
    """Warn when a GPU stage gets a plain function (rebuilt per batch → model reload).

    Passing a class/factory instead loads the model once per worker (the GPU-inference
    pattern); a plain function is re-created on every batch — the most common Ray Data
    inference foot-gun.
    """
    if num_gpus > 0 and not isinstance(fn, type):
        import warnings

        from batcher._internal.errors import PerformanceWarning

        warnings.warn(
            "map_batches got a plain function with num_gpus > 0; the model will be "
            "re-created on every batch (reloaded each time). Pass a class/factory "
            "instead so it loads once per worker (the GPU-inference pattern).",
            PerformanceWarning,
            stacklevel=3,
        )


class DatasetML:
    """Accessor for ML/multimodal operations over a `Dataset` (`ds.ml`)."""

    __slots__ = ("_ds",)

    def __init__(self, ds: Dataset) -> None:
        self._ds = ds

    def map_batches(
        self,
        fn: Callable | type,
        *,
        batch_size: int | None = None,
        output_columns: list[str] | None = None,
        num_workers: int = 1,
        num_gpus: float = 0.0,
        concurrency: int | tuple[int, int] | None = None,
        batch_format: str = "pyarrow",
        accelerator_type: str | None = None,
        model_memory_gb: float = 0.0,
    ) -> Dataset:
        """Apply a Python function to each batch.

        `fn` receives one batch and returns the transformed batch — the building
        block for batch inference, embeddings, and custom preprocessing. Pass a
        **class** instead of a function to load a model *once per worker* (it is
        instantiated once; the callable instance handles each batch) — the stateful
        GPU-inference pattern.

        `batch_format` chooses what `fn` sees and returns: ``"pyarrow"`` (a
        `pyarrow.RecordBatch`, zero-copy, the default), ``"numpy"`` (a
        ``{column: ndarray}`` dict), ``"pandas"`` (a `DataFrame`), or ``"torch"`` (a
        ``{column: tensor}`` dict over numeric columns). Conversion happens only
        around the call; the engine boundary stays Arrow. A `pyarrow`/`numpy` `fn`
        may also return a Table or column dict.

        `batch_size` rebatches before calling `fn` (e.g. to a model's GPU batch
        size). `output_columns` declares the result schema. `num_workers > 1` runs
        the per-batch calls concurrently within a worker (overlapping GIL-releasing
        inference). `num_gpus` reserves GPUs per distributed worker; `concurrency`
        sizes the distributed actor pool — an `int` for a fixed pool, or a
        ``(min, max)`` tuple to autoscale the pool to the workload within those bounds.
        `accelerator_type` pins GPU actors to a model (a `ray.util.accelerators` name
        like ``"NVIDIA_A100"``). `model_memory_gb` (the model's GB footprint) lets the
        resource layer budget host RAM per worker (OOM protection) and VRAM-pack small
        models onto a shared GPU, and lets Kyber cost inference by model size. Together
        they schedule a heterogeneous CPU+GPU pipeline across Ray (`distributed=True`).

        Warns (`PerformanceWarning`) when a GPU stage (`num_gpus > 0`) is given a
        plain function rather than a class/factory: a function is rebuilt on every
        batch, reloading the model each time — the single most common Ray Data
        inference foot-gun. Pass a class so the model loads once per worker.

        Raises:
            PlanError: if `batch_format` or `concurrency` is invalid.
        """
        from batcher.ml.batch_format import FORMATS

        if batch_format not in FORMATS:
            from batcher._internal.errors import PlanError

            raise PlanError(f"batch_format must be one of {FORMATS}, got {batch_format!r}")
        _validate_concurrency(concurrency)
        _warn_if_model_reloads(fn, num_gpus)
        cols = tuple(output_columns) if output_columns is not None else None
        return self._ds._derive(
            MapBatches(
                self._ds._plan,
                fn,
                batch_size,
                cols,
                num_workers=max(1, num_workers),
                num_gpus=num_gpus,
                concurrency=concurrency,
                batch_format=batch_format,
                accelerator_type=accelerator_type,
                model_memory_gb=model_memory_gb,
            )
        )

    def infer(
        self,
        model: Callable | type,
        *,
        batch_size: int | None = None,
        output_columns: list[str] | None = None,
        num_gpus: float = 0.0,
        concurrency: int | tuple[int, int] | None = None,
        batch_format: str = "pyarrow",
        accelerator_type: str | None = None,
        model_memory_gb: float = 0.0,
    ) -> Dataset:
        """Run batch model inference over the dataset (ML/multimodal path).

        Sugar for `map_batches` with inference defaults: pass a callable model, or a
        class that loads the model once per worker. `num_gpus`/`concurrency`/
        `accelerator_type`/`model_memory_gb` place and size the model on GPU actors
        while upstream preprocessing stays on CPU workers — the heterogeneous pipeline
        Ray Data specializes in. `batch_format` selects what the model receives/returns;
        see `map_batches`.
        """
        return self.map_batches(
            model,
            batch_size=batch_size,
            output_columns=output_columns,
            num_gpus=num_gpus,
            concurrency=concurrency,
            batch_format=batch_format,
            accelerator_type=accelerator_type,
            model_memory_gb=model_memory_gb,
        )

    def stream_loader(
        self,
        *,
        batch_size: int,
        world_size: int = 1,
        rank: int = 0,
        epoch: int = 0,
        seed: int = 0,
        shuffle: bool = True,
        drop_last: bool = True,
        columns: list[str] | None = None,
        global_consumed: int = 0,
    ):
        """A `torch.utils.data.IterableDataset` feeding this dataset to one training
        rank — deterministic, balanced across ranks, elastic, and resumable.

        The streaming-training-ingest path for PyTorch DDP/FSDP/DeepSpeed (the
        MosaicML-Streaming / Ray Train role): every rank yields the same number of
        ``{column: tensor}`` batches in a seed-reproducible global order that is
        independent of `world_size`, so a job can resume on a differently-sized
        cluster (pass `global_consumed` from a checkpoint) with no repeated or skipped
        samples. Disable any framework auto-sharding — this is the single shard
        authority. Requires `torch`. See `batcher.ml.stream_loader`.
        """
        from batcher.ml.loader import stream_loader

        return stream_loader(
            self._ds,
            batch_size=batch_size,
            world_size=world_size,
            rank=rank,
            epoch=epoch,
            seed=seed,
            shuffle=shuffle,
            drop_last=drop_last,
            columns=columns,
            global_consumed=global_consumed,
        )

    def iter_torch_batches(
        self,
        *,
        batch_size: int | None = None,
        columns: list[str] | None = None,
        device: object = "auto",
        collate_fn: object = None,
        prefetch_batches: int = 1,
        pin_memory: bool = False,
        local_shuffle_buffer_size: int | None = None,
        seed: int = 0,
    ):
        """Stream this dataset to PyTorch as ``{column: tensor}`` batches (lazy).

        The bounded-memory training-iteration path (Ray Data's ``iter_torch_batches``):
        consumes `iter_batches()` incrementally with `device` transfer (``"auto"``
        picks the best accelerator — CUDA/ROCm/Intel/Apple — or CPU), optional
        `pin_memory` for fast host→device copies, background `prefetch_batches`, a
        `local_shuffle_buffer_size` window, and a custom `collate_fn`. For a
        deterministic, balanced, resumable *distributed* split over a bounded corpus
        use `stream_loader`. Requires `torch`. See `batcher.ml.iter_torch_batches`.
        """
        from batcher.ml.loader import iter_torch_batches

        return iter_torch_batches(
            self._ds,
            batch_size=batch_size,
            columns=columns,
            device=device,
            collate_fn=collate_fn,
            prefetch_batches=prefetch_batches,
            pin_memory=pin_memory,
            local_shuffle_buffer_size=local_shuffle_buffer_size,
            seed=seed,
        )

    def embed(
        self,
        model: Callable | type,
        *,
        batch_size: int | None = None,
        output_columns: list[str] | None = None,
        num_gpus: float = 0.0,
        concurrency: int | tuple[int, int] | None = None,
        batch_format: str = "pyarrow",
        accelerator_type: str | None = None,
        model_memory_gb: float = 0.0,
    ) -> Dataset:
        """Compute embeddings over the dataset — `infer` specialized for embedding
        models (text/image → vector column). Same GPU/actor scheduling as `infer`;
        the distinct name documents the intent at the call site. `batch_format`
        selects what the model receives/returns; see `map_batches`.
        """
        return self.map_batches(
            model,
            batch_size=batch_size,
            output_columns=output_columns,
            num_gpus=num_gpus,
            concurrency=concurrency,
            batch_format=batch_format,
            accelerator_type=accelerator_type,
            model_memory_gb=model_memory_gb,
        )
