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
        """Bind the ML accessor to its `Dataset`; reached as `ds.ml`, not constructed directly."""
        self._ds = ds

    def map_batches(
        self,
        fn: Callable | type,
        *,
        batch_size: int | None = None,
        output_columns: list[str] | None = None,
        num_workers: int | str = "auto",
        num_gpus: float = 0.0,
        concurrency: int | tuple[int, int] | None = None,
        batch_format: str = "pyarrow",
        accelerator_type: str | None = None,
        model_memory_gb: float = 0.0,
        multiprocessing: bool = False,
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

        `batch_size` rebatches before calling `fn` (e.g. to a model's GPU batch size).
        `output_columns` declares the result schema. `num_workers` (default ``"auto"``:
        all local cores for a CPU stage, one model/CUDA context for a GPU stage) runs the
        per-batch calls concurrently within a worker — parallel by default, not
        single-threaded; an explicit int wins. `multiprocessing=True` runs them across
        *processes* (a CPU-bound pure-Python `fn`); it falls back to threads for a
        class/factory or GPU `fn` or a non-pyarrow `batch_format`. `num_gpus` reserves
        GPUs per distributed worker; `concurrency` sizes the distributed actor pool
        (default ``"auto"``: one actor per GPU) — an `int`, or a ``(min, max)`` tuple.
        `accelerator_type` pins GPU actors to a model (a `ray.util.accelerators` name
        like ``"NVIDIA_A100"``). `model_memory_gb` (the model's GB footprint) lets the
        resource layer budget host RAM per worker (OOM protection) and VRAM-pack small
        models onto a shared GPU, and lets Kyber cost inference by model size. Together
        they schedule a heterogeneous CPU+GPU pipeline across Ray (`distributed=True`).

        Warns (`PerformanceWarning`) when a GPU stage (`num_gpus > 0`) is given a
        plain function rather than a class/factory: a function is rebuilt on every
        batch, reloading the model each time — the single most common Ray Data
        inference foot-gun. Pass a class so the model loads once per worker.

        `multiprocessing=True` uses a `spawn`-based process pool, so the `fn` must be
        importable (picklable) and the **calling code must be import-safe** — a script
        that runs the pipeline at module top level needs an ``if __name__ ==
        "__main__":`` guard, or each spawned worker re-imports and re-runs it. A
        non-picklable `fn` (lambda/closure), a class/factory `fn`, a GPU `fn`, or a
        non-``pyarrow`` `batch_format` silently falls back to threads.

        Under `distributed=True`, a partition whose worker is **preempted** (a spot
        node reclaimed mid-batch) is reassigned and **recomputed** from its durable
        input — so `fn` must be *idempotent*: a pure transform is safe, but a `fn` with
        external side effects (a vector-DB upsert, a REST POST, an external counter)
        may apply that effect more than once on a retry. Make such a sink idempotent
        (upsert on a stable key derived from the row, not a blind insert) so recompute
        is exactly-once at the sink.

        Raises:
            PlanError: if `batch_format` or `concurrency` is invalid.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import pyarrow.compute as pc
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> def add_ten(batch):
                ...     return batch.set_column(0, "x", pc.add(batch.column("x"), 10))
                >>> ds.ml.map_batches(add_ten).to_pydict()
                {'x': [11, 12, 13]}
        """
        from batcher.ml.batch_format import FORMATS
        from batcher.ml.gpu import resolve_num_workers

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
                num_workers=resolve_num_workers(num_workers, num_gpus),
                num_gpus=num_gpus,
                concurrency=concurrency,
                batch_format=batch_format,
                accelerator_type=accelerator_type,
                model_memory_gb=model_memory_gb,
                multiprocessing=multiprocessing,
            )
        )

    def map(
        self,
        fn: Callable,
        *,
        batch_size: int | None = None,
        output_columns: list[str] | None = None,
        num_workers: int | str = "auto",
        concurrency: int | tuple[int, int] | None = None,
    ) -> Dataset:
        """Apply a per-row Python function ``fn(row_dict) -> row_dict`` (Ray Data
        ``map``).

        Each row is passed to `fn` as a ``{column: value}`` dict **inside the worker**
        (never the driver), so the hot-path rule holds; the per-row cost is yours.
        Prefer the vectorized `map_batches` (whole Arrow batch) when you can express
        the work over columns — it is far faster. `output_columns` declares the result
        schema. Returns a new lazy `Dataset`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> ds.ml.map(lambda row: {"x": row["x"] * 10}).to_pydict()
                {'x': [10, 20, 30]}
        """
        from batcher.api.dataset.callbacks import _RowMap

        return self.map_batches(
            _RowMap(fn),
            batch_size=batch_size,
            output_columns=output_columns,
            num_workers=num_workers,
            concurrency=concurrency,
        )

    def flat_map(
        self,
        fn: Callable,
        *,
        batch_size: int | None = None,
        output_columns: list[str] | None = None,
        num_workers: int | str = "auto",
        concurrency: int | tuple[int, int] | None = None,
    ) -> Dataset:
        """Apply a per-row function ``fn(row_dict) -> iterable[row_dict]`` and flatten
        the results (Ray Data ``flat_map``) — a one-to-many row transform.

        Like `map`, `fn` runs per row inside the worker. Each call returns zero or more
        output rows (dicts), all concatenated. `output_columns` declares the result
        schema. Returns a new lazy `Dataset`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> ds.ml.flat_map(lambda row: [{"x": row["x"]}, {"x": row["x"]}]).to_pydict()
                {'x': [1, 1, 2, 2, 3, 3]}
        """
        from batcher.api.dataset.callbacks import _RowFlatMap

        return self.map_batches(
            _RowFlatMap(fn),
            batch_size=batch_size,
            output_columns=output_columns,
            num_workers=num_workers,
            concurrency=concurrency,
        )

    def infer(
        self,
        model: str | Callable | type,
        *,
        column: str | None = None,
        output_column: str = "prediction",
        output_columns: list[str] | None = None,
        task: str | None = None,
        batch_size: int | None = None,
        num_gpus: float = 0.0,
        concurrency: int | tuple[int, int] | None = None,
        batch_format: str = "pyarrow",
        accelerator_type: str | None = None,
        model_memory_gb: float = 0.0,
    ) -> Dataset:
        """Run batch model inference over the dataset (ML/multimodal path).

        Pass a **model identifier** (a HuggingFace ``transformers`` model id) and the
        `column` to score: the model loads once per worker and its prediction is
        appended as `output_column`. `task` selects the pipeline kind
        (``"sentiment-analysis"``, ``"text-classification"``, …; inferred from the model
        when omitted). Needs ``transformers`` (``batcher-engine[transformers]``).

        Pass a **callable or class** instead for full control (a class loads the model
        once per worker — the GPU-inference pattern); the call then mirrors
        `map_batches`, with `output_columns` declaring the result schema.

        Either way `num_gpus`/`concurrency`/`accelerator_type`/`model_memory_gb` place
        and size the model on GPU actors while upstream preprocessing stays on CPU
        workers — the heterogeneous pipeline Ray Data specializes in. For arbitrary
        batch work that is not model inference, use `map_batches` directly.

        Raises:
            PlanError: if a model id is given without `column`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"text": ["great!", "awful."]})
                >>> scored = ds.ml.infer(  # doctest: +SKIP
                ...     "distilbert-base-uncased-finetuned-sst-2-english", column="text"
                ... )
        """
        if isinstance(model, str):
            if column is None:
                from batcher._internal.errors import PlanError

                raise PlanError("ds.ml.infer(<model id>) requires column= (the input column)")
            from batcher.ml.inference import transformers_pipeline_encoder

            encoder = transformers_pipeline_encoder(
                model, column, output_column=output_column, task=task
            )
            cols = (
                [*self._ds.columns, output_column]
                if output_column not in self._ds.columns
                else None
            )
            return self.map_batches(
                encoder,
                output_columns=cols,
                batch_size=batch_size,
                num_gpus=num_gpus,
                concurrency=concurrency,
                accelerator_type=accelerator_type,
                model_memory_gb=model_memory_gb,
            )
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

        Examples:
            .. code-block:: python

                import batcher as bt
                from torch.utils.data import DataLoader

                ds = bt.read.parquet("s3://bucket/train/*.parquet")
                iterable = ds.ml.stream_loader(
                    batch_size=256, world_size=8, rank=rank, columns=["image", "label"]
                )
                for batch in DataLoader(iterable, batch_size=None):
                    train_step(batch["image"], batch["label"])
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
        zero_copy: bool = False,
        local_shuffle_buffer_size: int | None = None,
        seed: int = 0,
    ):
        """Stream this dataset to PyTorch as ``{column: tensor}`` batches (lazy).

        The bounded-memory training-iteration path (Ray Data's ``iter_torch_batches``):
        consumes `iter_batches()` incrementally with `device` transfer (``"auto"``
        picks the best accelerator — CUDA/ROCm/Intel/Apple — or CPU), optional
        `pin_memory` for fast host→device copies, `zero_copy` DLPack views for
        read-only inference, background `prefetch_batches`, a `local_shuffle_buffer_size`
        window, and a custom `collate_fn`. For a deterministic, balanced, resumable
        *distributed* split over a bounded corpus use `stream_loader`. Requires `torch`.
        See `batcher.ml.iter_torch_batches`.

        Examples:
            .. code-block:: python

                import batcher as bt

                ds = bt.read.parquet("s3://bucket/train/*.parquet")
                for batch in ds.ml.iter_torch_batches(batch_size=256, device="auto"):
                    train_step(batch["image"], batch["label"])
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
            zero_copy=zero_copy,
            local_shuffle_buffer_size=local_shuffle_buffer_size,
            seed=seed,
        )

    def embed(
        self,
        model: str | Callable | type,
        *,
        column: str | None = None,
        output_column: str = "embedding",
        output_columns: list[str] | None = None,
        batch_size: int | None = None,
        num_gpus: float = 0.0,
        concurrency: int | tuple[int, int] | None = None,
        batch_format: str = "pyarrow",
        accelerator_type: str | None = None,
        model_memory_gb: float = 0.0,
    ) -> Dataset:
        """Compute embeddings over the dataset — `infer` shaped for embedding models.

        Pass a **model identifier** (a sentence-transformers model id) and the text
        `column` to embed: the model loads once per worker and the vector is appended
        as a tensor `output_column`. The provider-pluggable, distributed, GPU-aware
        text-embedding path (cf. Daft's ``embed_text``). Needs ``sentence-transformers``
        (``batcher-engine[st]``).

        Pass a **callable or class** instead for any other embedding model (text or
        image → vector); the call then mirrors `map_batches`, with `output_columns`
        declaring the result schema.

        `num_gpus`/`concurrency`/`accelerator_type`/`model_memory_gb` place and size
        the model on GPU actors, the same scheduling as `infer`.

        Raises:
            PlanError: if a model id is given without `column`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"text": ["a sentence", "another"]})
                >>> vectors = ds.ml.embed(  # doctest: +SKIP
                ...     "sentence-transformers/all-MiniLM-L6-v2", column="text"
                ... )
        """
        if isinstance(model, str):
            if column is None:
                from batcher._internal.errors import PlanError

                raise PlanError("ds.ml.embed(<model id>) requires column= (the text column)")
            from batcher.ml.embed import sentence_transformer_encoder

            encoder = sentence_transformer_encoder(model, column, output_column=output_column)
            cols = (
                [*self._ds.columns, output_column]
                if output_column not in self._ds.columns
                else None
            )
            return self.map_batches(
                encoder,
                output_columns=cols,
                batch_size=batch_size,
                num_gpus=num_gpus,
                concurrency=concurrency,
                accelerator_type=accelerator_type,
                model_memory_gb=model_memory_gb,
            )
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

    def download(
        self,
        url_column: str,
        *,
        output_column: str = "bytes",
        max_concurrency: int = 16,
        on_error: str = "raise",
    ) -> Dataset:
        """Fetch the bytes at each URL/path into ``output_column`` (the multimodal
        ingestion entry point — URL table → bytes → decode → model).

        Reads ``s3://``/``gs://``/``az://``/``http(s)://``/local paths through the shared
        filesystem resolver, fetching each batch's rows concurrently and parallelizing
        across the cluster (a `map_batches` stage). ``on_error="null"`` makes a failed
        fetch a null instead of raising. See `batcher.ml.download_dataset`.

        Examples:
            .. code-block:: python

                import batcher as bt

                urls = bt.from_pydict({"url": ["s3://bucket/cat.jpg", "s3://bucket/dog.jpg"]})
                images = urls.ml.download("url", output_column="bytes")
        """
        from batcher.ml.decode import download_dataset

        return download_dataset(
            self._ds,
            url_column=url_column,
            output_column=output_column,
            max_concurrency=max_concurrency,
            on_error=on_error,
        )

    def upload(
        self,
        data_column: str,
        directory: str,
        *,
        output_column: str = "path",
        name_column: str | None = None,
        extension: str = "",
        max_concurrency: int = 16,
    ) -> Dataset:
        """Write each row's bytes to a file under `directory`, appending the path.

        The counterpart to `download` — write transformed media back to
        ``s3://``/``gs://``/``az://``/local storage, parallelized across the cluster.
        Names come from `name_column` (+ `extension`) or a content hash. See
        `batcher.ml.decode.upload_dataset`.

        Examples:
            .. code-block:: python

                import batcher as bt

                thumbs = ds.ml.map_batches(make_thumbnails)  # bytes in "thumb"
                written = thumbs.ml.upload(
                    "thumb", "s3://bucket/thumbs", extension=".jpg"
                )
        """
        from batcher.ml.decode import upload_dataset

        return upload_dataset(
            self._ds,
            data_column=data_column,
            directory=directory,
            output_column=output_column,
            name_column=name_column,
            extension=extension,
            max_concurrency=max_concurrency,
        )
