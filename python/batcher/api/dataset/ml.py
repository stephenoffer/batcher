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

from batcher.api.dataset._build import build_random_split, build_train_test_split
from batcher.api.dataset._dedup import (
    build_drop_near_duplicates,
    build_near_duplicates,
    build_similarity_join,
)
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


def _as_key_columns(key: str | list[str] | None) -> list[str] | None:
    """Normalize a split key: a single column name, several, or `None` (hash every column)."""
    if key is None:
        return None
    return [key] if isinstance(key, str) else list(key)


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
    """Accessor for ML/multimodal operations over a `Dataset` (`ds.ml`).

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1, 2, 3]})
            >>> ds.ml.map(lambda row: {"x": row["x"] + 1}).to_pydict()
            {'x': [2, 3, 4]}
    """

    __slots__ = ("_ds",)

    def __init__(self, ds: Dataset) -> None:
        """Bind the ML accessor to its `Dataset`; reached as `ds.ml`, not constructed directly."""
        self._ds = ds

    def map_batches(
        self,
        fn: Callable | type,
        *,
        batch_size: int | None = None,
        input_columns: list[str] | None = None,
        output_columns: list[str] | None = None,
        num_workers: int | str = "auto",
        num_gpus: float = 0.0,
        concurrency: int | tuple[int, int] | None = None,
        batch_format: str = "pyarrow",
        accelerator_type: str | None = None,
        model_memory_gb: float = 0.0,
        multiprocessing: bool = False,
        max_errored_rows: int = 0,
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

        `max_errored_rows` gives dirty-data tolerance: with it set (default 0 = strict), a
        batch whose `fn` raises is bisected to isolate the offending rows, and a failing row
        is *dropped* (up to this many, per worker) so a corrupt image / malformed record
        doesn't kill a long inference job — the guides' ``max_errored_blocks`` need. Beyond
        the budget the error propagates, so a real bug on clean data still fails fast.

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

        Args:
            fn: A function (or class/factory) applied to each batch.
            batch_size: Rebatch to this many rows before each call.
            input_columns: The columns `fn` reads. Declaring them lets the optimizer prune
                everything else out of the scan — an embedding stage over one column of a
                41-column Parquet file otherwise reads all 41, because the engine cannot see
                inside a Python function and must assume it reads anything. Omitting a column
                the `fn` actually reads is a correctness bug, not a slow path: it will be
                pruned out from under the function. Leave unset (the default) if unsure.
            output_columns: The result schema when `fn` changes the columns.
            num_workers: Concurrent per-batch calls within a worker (``"auto"``
                sizes to the stage), or an explicit int.
            num_gpus: GPUs to reserve per distributed worker.
            concurrency: Size of the distributed actor pool; an int or ``(min, max)``.
            batch_format: What `fn` sees — ``"pyarrow"``, ``"numpy"``, ``"pandas"``,
                or ``"torch"``.
            accelerator_type: Pin GPU actors to a model (e.g. ``"NVIDIA_A100"``).
            model_memory_gb: The model's footprint, for memory budgeting.
            multiprocessing: Run CPU-bound pure-Python calls across processes.
            max_errored_rows: Rows a raising `fn` may drop per worker before failing.

        Returns:
            A new lazy `Dataset` with `fn` applied to every batch.

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
                input_columns=tuple(input_columns) if input_columns is not None else None,
                num_workers=resolve_num_workers(num_workers, num_gpus),
                num_gpus=num_gpus,
                concurrency=concurrency,
                batch_format=batch_format,
                accelerator_type=accelerator_type,
                model_memory_gb=model_memory_gb,
                multiprocessing=multiprocessing,
                max_errored_rows=max_errored_rows,
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
        """Apply a per-row Python function ``fn(row_dict) -> row_dict`` (Ray Data ``map``).

        Each row is passed to `fn` as a ``{column: value}`` dict **inside the worker**
        (never the driver), so the hot-path rule holds; the per-row cost is yours.
        Prefer the vectorized `map_batches` (whole Arrow batch) when you can express
        the work over columns — it is far faster.

        Args:
            fn: A ``row_dict -> row_dict`` function applied per row.
            batch_size: Rebatch to this many rows before processing.
            output_columns: The result schema when `fn` changes the columns.
            num_workers: Concurrent calls within a worker (``"auto"`` sizes it).
            concurrency: Size of the distributed actor pool; an int or ``(min, max)``.

        Returns:
            A new lazy `Dataset` with `fn` applied to every row.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> ds.ml.map(lambda row: {"x": row["x"] * 10}).to_pydict()
                {'x': [10, 20, 30]}
        """
        from batcher.api.dataset.callbacks import _RowMap

        cols = tuple(output_columns) if output_columns is not None else None
        return self.map_batches(
            _RowMap(fn, cols),
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
        """Apply ``fn(row_dict) -> iterable[row_dict]`` and flatten (Ray Data ``flat_map``).

        A one-to-many row transform. Like `map`, `fn` runs per row inside the worker;
        each call returns zero or more output rows (dicts), all concatenated.

        Args:
            fn: A ``row_dict -> iterable[row_dict]`` function applied per row.
            batch_size: Rebatch to this many rows before processing.
            output_columns: The result schema when `fn` changes the columns.
            num_workers: Concurrent calls within a worker (``"auto"`` sizes it).
            concurrency: Size of the distributed actor pool; an int or ``(min, max)``.

        Returns:
            A new lazy `Dataset` with the flattened per-row outputs.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> ds.ml.flat_map(lambda row: [{"x": row["x"]}, {"x": row["x"]}]).to_pydict()
                {'x': [1, 1, 2, 2, 3, 3]}
        """
        from batcher.api.dataset.callbacks import _RowFlatMap

        cols = tuple(output_columns) if output_columns is not None else None
        return self.map_batches(
            _RowFlatMap(fn, cols),
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

        Args:
            model: A HuggingFace model id, or a callable/class scoring a batch.
            column: The input column to score (required for a model id).
            output_column: Name of the appended prediction column (model-id path).
            output_columns: The result schema when a callable/class is passed.
            task: The pipeline kind for a model id (inferred when omitted).
            batch_size: Rebatch to this many rows before each call.
            num_gpus: GPUs to reserve per distributed worker.
            concurrency: Size of the distributed actor pool; an int or ``(min, max)``.
            batch_format: What a callable `model` sees (``"pyarrow"`` by default).
            accelerator_type: Pin GPU actors to a model (e.g. ``"NVIDIA_A100"``).
            model_memory_gb: The model's footprint, for memory budgeting.

        Returns:
            A new lazy `Dataset` with the prediction column(s) appended.

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

    def train_test_split(
        self, test_size: float = 0.25, *, seed: int = 0, key: str | list[str] | None = None
    ) -> tuple[Dataset, Dataset]:
        """Split the rows into a disjoint train and test `Dataset`.

        Each row is assigned by a reproducible hash of its **own values** and `seed`, so
        the parts are disjoint, together cover every row, and are identical however the
        data is partitioned — single-node, multi-core, distributed, or streaming. Each
        part is a plain row-wise filter, so nothing is materialized, nothing is shuffled,
        and both parts stay lazy until a terminal op. Sizes are binomial around
        ``test_size * n`` rather than exact, as with any hash-keyed split.

        Args:
            test_size: The fraction of rows to place in the test part, in ``(0, 1)``.
            seed: Seed for the row assignment; the same seed reproduces the split.
            key: The column(s) identifying a row. Prefer it on a real corpus: hashing
                only these keeps the split stable when the *other* columns change
                (recompute a feature and the same rows stay in train), costs one hash
                per key column instead of one per column, and does not depend on how
                floats render as text. The default hashes every column — correct and
                reproducible, but re-splits whenever any value or the schema changes.

        Returns:
            The ``(train, test)`` pair.

        Raises:
            PlanError: If `test_size` is not strictly between 0 and 1, or `key` names a
                column the dataset does not have.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.range(0, 1000)
                >>> train, test = ds.ml.train_test_split(0.2, seed=42, key="value")
                >>> train.count() + test.count()
                1000
        """
        return build_train_test_split(self._ds, test_size, seed=seed, key=_as_key_columns(key))

    def random_split(
        self, fractions: list[float], *, seed: int = 0, key: str | list[str] | None = None
    ) -> list[Dataset]:
        """Split the rows into disjoint random parts sized by `fractions`.

        The generalization of :meth:`train_test_split` to a train/validation/test
        three-way (or n-way) split, with the same content-hash assignment.

        Args:
            fractions: The share of rows per part; must be positive and sum to 1.0.
            seed: Seed for the row assignment; the same seed reproduces the split.
            key: The column(s) identifying a row — see :meth:`train_test_split`.

        Returns:
            One `Dataset` per entry in `fractions`, in order.

        Raises:
            PlanError: If `fractions` is empty, holds a non-positive value, does not
                sum to 1.0, or `key` names a column the dataset does not have.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.range(0, 1000)
                >>> train, val, test = ds.ml.random_split([0.7, 0.15, 0.15], seed=42)
                >>> train.count() + val.count() + test.count()
                1000
        """
        return build_random_split(self._ds, fractions, seed=seed, key=_as_key_columns(key))

    def near_duplicates(
        self,
        column: str,
        *,
        threshold: float = 0.8,
        num_perm: int = 128,
        ngram: int = 5,
        bands: int = 16,
        key: str | None = None,
    ) -> Dataset:
        """Find near-duplicate document pairs by MinHash + LSH — the fuzzy-dedup join.

        Returns the pairs whose estimated Jaccard similarity over character
        `ngram`-shingles is at least `threshold`, as ``(key_a, key_b, jaccard)`` with
        ``key_a < key_b``. Exact duplication is `distinct()`; this finds the same
        article behind a different header, which is what actually dominates a crawl.

        Candidate pairs come from LSH banding and are then **verified** against
        `threshold` by their signature agreement, so every returned pair clears it.
        Recall is not total: a similar pair can miss every band. `bands` is the dial —
        more bands mean more candidates (higher recall, more work), and the S-curve's
        knee sits near ``(1 / bands) ** (bands / num_perm)``.

        Args:
            column: The text column to compare.
            threshold: Minimum estimated Jaccard similarity, in ``(0, 1]``.
            num_perm: MinHash permutations. Standard error is ``1 / sqrt(num_perm)``.
            ngram: Shingle width in characters; larger is stricter.
            bands: LSH bands; must divide `num_perm`.
            key: The column identifying a row. Defaults to a hash of `column`, so
                byte-identical documents collapse to one key before any comparison.

        Returns:
            A lazy `Dataset` of ``key_a``, ``key_b``, ``jaccard``.

        Raises:
            PlanError: On an unknown column, a `threshold` outside ``(0, 1]``, or
                `bands` that does not divide `num_perm`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"t": ["a b c d e f g", "a b c d e f g!", "zzz"]})
                >>> ds.ml.near_duplicates("t", threshold=0.5).count()
                1
        """
        return build_near_duplicates(
            self._ds,
            column,
            threshold=threshold,
            num_perm=num_perm,
            ngram=ngram,
            bands=bands,
            key=key,
        )

    def drop_near_duplicates(
        self,
        column: str,
        *,
        threshold: float = 0.8,
        num_perm: int = 128,
        ngram: int = 5,
        bands: int = 16,
        key: str | None = None,
    ) -> Dataset:
        """Remove near-duplicate rows, keeping one representative — the dedup pass.

        Drops every row that has a near-duplicate (see :meth:`near_duplicates`) with a
        smaller key. The survivors are the rows minimal among their near-duplicates: for
        a duplicate cluster where every member matches every other — the usual shape —
        that is exactly one row.

        Args:
            column: The text column to compare.
            threshold: Minimum estimated Jaccard similarity to count as a duplicate.
            num_perm: MinHash permutations.
            ngram: Shingle width in characters.
            bands: LSH bands; must divide `num_perm`.
            key: The column identifying a row; defaults to a hash of `column`.

        Returns:
            A lazy `Dataset` with the near-duplicates removed and the input schema kept.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"t": ["a b c d e f g", "a b c d e f g!", "zzz"]})
                >>> sorted(ds.ml.drop_near_duplicates("t", threshold=0.5).to_pydict()["t"])
                ['a b c d e f g', 'zzz']
        """
        return build_drop_near_duplicates(
            self._ds,
            column,
            threshold=threshold,
            num_perm=num_perm,
            ngram=ngram,
            bands=bands,
            key=key,
        )

    def similarity_join(
        self,
        other: Dataset,
        *,
        left_on: str,
        right_on: str | None = None,
        threshold: float = 0.8,
        num_bits: int = 64,
        bands: int = 8,
        seed: int = 0,
        left_key: str | None = None,
        right_key: str | None = None,
    ) -> Dataset:
        """Join two datasets on **embedding similarity** rather than on equality.

        Semantic entity resolution: match a product catalogue to a supplier feed, a CRM
        to a billing system, or a query set to a document corpus — wherever the join key
        is "means the same thing" rather than "is the same string". Every returned pair
        has cosine similarity at least `threshold`.

        Comparing every pair of embeddings is ``O(n * m)`` and impossible at scale. This
        is the standard two-stage escape, expressed in operators the engine already has:
        `.list.simhash` reduces each vector to a bit signature, the bits are split into
        `bands` bands, each band is hashed, and two rows become **candidates** only if
        they collide in some band. The candidates are then scored with the *exact*
        `.list.cosine_similarity` over the original vectors.

        So banding controls **recall, never precision**: no pair below `threshold` is
        ever returned, but a pair above it can miss every band and be lost. More `bands`
        means more candidates, higher recall, and more work — that is the dial. A pair
        with similarity `s` survives banding with probability
        ``1 - (1 - s^(num_bits/bands))^bands``.

        Rows whose vector is null or empty have no direction, cannot clear any threshold,
        and are dropped rather than banded (left in, they would all collide and blow the
        candidate set up quadratically). Rows sharing a key collapse to one first.

        Nothing is materialized on the driver — this is a projection, an `explode`, a
        join, and a filter — so it runs wherever a join runs, including distributed.

        Args:
            other: The right-hand `Dataset`.
            left_on: The left embedding column (a `List<Float64>`).
            right_on: The right embedding column; defaults to `left_on`.
            threshold: Minimum cosine similarity of a returned pair, in ``[-1, 1]``.
            num_bits: SimHash signature length. Must be divisible by `bands`.
            bands: Number of LSH bands. The recall/cost dial.
            seed: Selects the hyperplanes; both sides necessarily share it.
            left_key: Column identifying a left row; defaults to a digest of its vector.
            right_key: Column identifying a right row; defaults to a digest of its vector.

        Returns:
            A new lazy `Dataset` of ``key_a``, ``key_b``, ``similarity`` — one row per
            matching pair.

        Raises:
            PlanError: If a column is unknown, `threshold` is outside ``[-1, 1]``, or
                `bands` does not divide `num_bits`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> catalog = bt.from_pydict({"sku": [1, 2], "v": [[1.0, 0.0], [0.0, 1.0]]})
                >>> feed = bt.from_pydict({"ref": [10], "v": [[1.0, 0.02]]})
                >>> pairs = catalog.ml.similarity_join(
                ...     feed, left_on="v", threshold=0.9, left_key="sku", right_key="ref"
                ... )
                >>> pairs.select("key_a", "key_b").to_pydict()
                {'key_a': [1], 'key_b': [10]}
        """
        return build_similarity_join(
            self._ds,
            other,
            left_on=left_on,
            right_on=right_on,
            threshold=threshold,
            num_bits=num_bits,
            bands=bands,
            seed=seed,
            left_key=left_key,
            right_key=right_key,
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
        """Feed this dataset to one training rank as a `torch` ``IterableDataset``.

        The streaming-training-ingest path for PyTorch DDP/FSDP/DeepSpeed (the
        MosaicML-Streaming / Ray Train role): deterministic, balanced across ranks,
        elastic, and resumable. Every rank yields the same number of
        ``{column: tensor}`` batches in a seed-reproducible global order that is
        independent of `world_size`, so a job can resume on a differently-sized
        cluster (pass `global_consumed` from a checkpoint) with no repeated or skipped
        samples. Disable any framework auto-sharding — this is the single shard
        authority. Requires `torch`. See `batcher.ml.stream_loader`.

        Args:
            batch_size: Rows per yielded ``{column: tensor}`` batch.
            world_size: Total number of training ranks.
            rank: This rank's index in ``[0, world_size)``.
            epoch: Epoch number, folded into the shuffle so passes differ.
            seed: Seed for the global order; the same seed reproduces it.
            shuffle: Shuffle the global order before sharding.
            drop_last: Drop the final partial global batch so ranks stay balanced.
            columns: The columns to yield as tensors; defaults to all.
            global_consumed: Samples already consumed, to resume from a checkpoint.

        Returns:
            A `torch.utils.data.IterableDataset` yielding this rank's batches.

        Examples:
            .. doctest::

                >>> import batcher as bt  # doctest: +SKIP
                >>> from torch.utils.data import DataLoader  # doctest: +SKIP
                >>> ds = bt.read.parquet("s3://bucket/train/*.parquet")  # doctest: +SKIP
                >>> iterable = ds.ml.stream_loader(  # doctest: +SKIP
                ...     batch_size=256, world_size=8, rank=0, columns=["image", "label"]
                ... )
                >>> for batch in DataLoader(iterable, batch_size=None):  # doctest: +SKIP
                ...     train_step(batch["image"], batch["label"])
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

        Args:
            batch_size: Rows per yielded ``{column: tensor}`` batch.
            columns: The columns to yield as tensors; defaults to all.
            device: Target device (``"auto"`` picks the best accelerator, else CPU).
            collate_fn: Custom collation applied to each batch before yielding.
            prefetch_batches: Batches to prepare ahead in the background.
            pin_memory: Pin host buffers for faster host-to-device copies.
            zero_copy: Yield read-only DLPack views instead of copying.
            local_shuffle_buffer_size: Window size for local shuffling; None disables.
            seed: Seed for the local shuffle.

        Yields:
            ``{column: tensor}`` batches on `device`.

        Examples:
            .. doctest::

                >>> import batcher as bt  # doctest: +SKIP
                >>> ds = bt.read.parquet("s3://bucket/train/*.parquet")  # doctest: +SKIP
                >>> loader = ds.ml.iter_torch_batches(batch_size=256)  # doctest: +SKIP
                >>> for batch in loader:  # doctest: +SKIP
                ...     train_step(batch["image"], batch["label"])
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

    def generate(
        self,
        engine: Callable,
        *,
        prompt_column: str,
        output_column: str = "response",
        template: str | None = None,
        image_column: str | None = None,
        adapter_column: str | None = None,
        parse_json: bool = False,
        usage: bool = False,
        batch_size: int | None = None,
        num_gpus: float = 0.0,
        concurrency: int | tuple[int, int] | None = None,
        accelerator_type: str | None = None,
        model_memory_gb: float = 0.0,
    ) -> Dataset:
        """Run offline LLM text generation over the dataset, appending `output_column`.

        `engine` is an `EngineFactory` — a zero-arg callable returning a
        ``list[str] -> list[str]`` engine — so the model loads **once per worker** and
        the row work stays columnar. Use `batcher.ml.vllm_engine` for local GPU
        inference (structured JSON output, multi-LoRA, vision) or
        `batcher.ml.http_engine` for an OpenAI-compatible endpoint. Any callable of the
        same shape works, which is what makes this testable without a GPU.

        No outer batch size is imposed by default: vLLM does its own continuous
        batching, and a fixed outer batch would fight its scheduler. `num_gpus`,
        `concurrency`, `accelerator_type` and `model_memory_gb` place and size the
        engine on GPU actors exactly as `infer`/`embed` do.

        Args:
            engine: The `EngineFactory` to build once per worker.
            prompt_column: The text column to send (ignored when `template` is set).
            output_column: Name of the appended generated column.
            template: A ``str.format`` template over the row's columns, e.g.
                ``"Summarize: {body}"``. Overrides `prompt_column` for prompt building.
            image_column: An image column (raw bytes or an ``(H, W, 3)`` tensor) for
                vision-language models.
            adapter_column: A column naming the per-row LoRA adapter; pair with
                ``vllm_engine(lora_paths=...)``.
            parse_json: Parse each output as JSON into a struct column (null on a parse
                error). Pair with ``vllm_engine(guided_json=...)`` for reliable output.
            usage: Also append ``prompt_tokens`` / ``completion_tokens`` columns.
            batch_size: Rebatch before each engine call; leave unset for the engine's own.
            num_gpus: GPUs to reserve per worker.
            concurrency: Size of the distributed actor pool.
            accelerator_type: Pin GPU actors to a model (e.g. ``"NVIDIA_A100"``).
            model_memory_gb: The model's footprint, for memory budgeting.

        Returns:
            A new `Dataset` with the generated column(s) appended.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"q": ["2+2?", "capital of France?"]})
                >>> shout = lambda: (lambda prompts: [p.upper() for p in prompts])
                >>> ds.ml.generate(shout, prompt_column="q").to_pydict()
                {'q': ['2+2?', 'capital of France?'], 'response': ['2+2?', 'CAPITAL OF FRANCE?']}
        """
        from batcher.ml.llm import llm_udf

        udf = llm_udf(
            engine,
            prompt_column=prompt_column,
            output_column=output_column,
            template=template,
            image_column=image_column,
            adapter_column=adapter_column,
            parse_json=parse_json,
            usage=usage,
        )
        appended = [output_column, *(["prompt_tokens", "completion_tokens"] if usage else [])]
        new = [c for c in appended if c not in self._ds.columns]
        return self.map_batches(
            udf,
            output_columns=[*self._ds.columns, *new] if new else None,
            batch_size=batch_size,
            num_gpus=num_gpus,
            concurrency=concurrency,
            accelerator_type=accelerator_type,
            model_memory_gb=model_memory_gb,
        )

    def extract(
        self,
        engine: Callable,
        *,
        schema: dict[str, str],
        prompt_column: str | None = None,
        template: str | None = None,
        instruct: bool = True,
        batch_size: int | None = None,
        num_gpus: float = 0.0,
        concurrency: int | tuple[int, int] | None = None,
        accelerator_type: str | None = None,
        model_memory_gb: float = 0.0,
    ) -> Dataset:
        """Extract declared, **typed** columns from unstructured text with an LLM.

        The AI-powered-ETL step: a support email becomes ``{customer, severity, refund}``;
        an invoice PDF's text becomes ``{vendor, total, due_date}``. The result is a
        normal Arrow column an analyst can filter, join, and aggregate — not a JSON blob.

        `schema` maps each output column to a Batcher dtype, and that declaration — not
        whatever the model emitted in a given batch — decides the Arrow type. This is the
        difference from ``generate(parse_json=True)``, whose struct type is *inferred per
        batch*: ask for ``{label, score}``, have the model omit ``score`` on one batch,
        and the scan fails at concat time with the GPU work already paid for.

        Degradation is per row: an unparseable response, a missing key, or a value that
        will not coerce becomes null in that column. One bad generation over a million
        rows costs you one row, and ``ds.filter(col("total").is_null()).count()`` tells
        you how many.

        Pair with ``vllm_engine(guided_json=json_schema(schema))`` to constrain decoding
        so that every row parses in the first place.

        Args:
            engine: The `EngineFactory` to build once per worker.
            schema: Output column name → Batcher dtype (``"string"``, ``"int64"``,
                ``"float64"``, ``"bool"``, …).
            prompt_column: The text column to send (ignored when `template` is set).
            template: A ``str.format`` template over the row's columns.
            instruct: Append a "reply with JSON having exactly these keys" instruction to
                each prompt. Turn it off when the engine already constrains decoding.
            batch_size: Rebatch before each engine call; leave unset for the engine's own.
            num_gpus: GPUs to reserve per worker.
            concurrency: Size of the distributed actor pool.
            accelerator_type: Pin GPU actors to a model (e.g. ``"NVIDIA_A100"``).
            model_memory_gb: The model's footprint, for memory budgeting.

        Returns:
            A new lazy `Dataset` with one typed column appended per `schema` field.

        Raises:
            PlanError: If `schema` is empty or names an unknown dtype.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"note": ["Paid 42 USD to Acme"]})
                >>> stub = lambda: (lambda ps: ['{"vendor": "Acme", "total": "42"}'] * len(ps))
                >>> out = ds.ml.extract(
                ...     stub, schema={"vendor": "string", "total": "float64"}, prompt_column="note"
                ... )
                >>> out.to_pydict()
                {'note': ['Paid 42 USD to Acme'], 'vendor': ['Acme'], 'total': [42.0]}
        """
        from batcher.ml.llm import llm_extract_udf

        udf = llm_extract_udf(
            engine,
            schema=schema,
            prompt_column=prompt_column,
            template=template,
            instruct=instruct,
        )
        new = [c for c in schema if c not in self._ds.columns]
        return self.map_batches(
            udf,
            output_columns=[*self._ds.columns, *new] if new else None,
            batch_size=batch_size,
            num_gpus=num_gpus,
            concurrency=concurrency,
            accelerator_type=accelerator_type,
            model_memory_gb=model_memory_gb,
        )

    def classify(
        self,
        engine: Callable,
        *,
        labels: list[str],
        prompt_column: str | None = None,
        output_column: str = "label",
        template: str | None = None,
        instruct: bool = True,
        batch_size: int | None = None,
        num_gpus: float = 0.0,
        concurrency: int | tuple[int, int] | None = None,
        accelerator_type: str | None = None,
        model_memory_gb: float = 0.0,
    ) -> Dataset:
        """Label each row with exactly one of `labels`, using an LLM.

        Zero-shot categorization as an ETL step — routing tickets, tagging sentiment,
        flagging policy violations — with the guarantee that the resulting column's
        domain is exactly `labels`.

        A model asked for ``"positive"`` will happily answer ``"Positive."`` or ``"The
        sentiment is positive."``; taking those verbatim gives a category column with a
        long tail of near-duplicate values that never group together. Here the output is
        matched against the declared labels case-insensitively (and the label is found
        inside a short sentence), and **anything that does not resolve to exactly one
        label becomes null** — so bad rows are countable rather than silently wrong.

        Args:
            engine: The `EngineFactory` to build once per worker.
            labels: The permitted labels; must be distinct ignoring case.
            prompt_column: The text column to classify (ignored when `template` is set).
            output_column: Name of the appended label column.
            template: A ``str.format`` template over the row's columns.
            instruct: Append the "answer with one of these labels" instruction to each
                prompt. Turn it off when the template says it itself.
            batch_size: Rebatch before each engine call; leave unset for the engine's own.
            num_gpus: GPUs to reserve per worker.
            concurrency: Size of the distributed actor pool.
            accelerator_type: Pin GPU actors to a model (e.g. ``"NVIDIA_A100"``).
            model_memory_gb: The model's footprint, for memory budgeting.

        Returns:
            A new lazy `Dataset` with the label column appended.

        Raises:
            PlanError: If `labels` is empty or has case-insensitive duplicates.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"review": ["loved it", "awful"]})
                >>> def stub():
                ...     return lambda ps: ["Positive." if "loved" in p else "negative" for p in ps]
                >>> labelled = ds.ml.classify(stub, labels=["positive", "negative"],
                ...                           prompt_column="review")
                >>> labelled.to_pydict()
                {'review': ['loved it', 'awful'], 'label': ['positive', 'negative']}
        """
        from batcher.ml.llm import llm_classify_udf

        udf = llm_classify_udf(
            engine,
            labels=labels,
            prompt_column=prompt_column,
            output_column=output_column,
            template=template,
            instruct=instruct,
        )
        new = [] if output_column in self._ds.columns else [output_column]
        return self.map_batches(
            udf,
            output_columns=[*self._ds.columns, *new] if new else None,
            batch_size=batch_size,
            num_gpus=num_gpus,
            concurrency=concurrency,
            accelerator_type=accelerator_type,
            model_memory_gb=model_memory_gb,
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

        Args:
            model: A sentence-transformers model id, or a callable/class → vector.
            column: The text column to embed (required for a model id).
            output_column: Name of the appended embedding column (model-id path).
            output_columns: The result schema when a callable/class is passed.
            batch_size: Rebatch to this many rows before each call.
            num_gpus: GPUs to reserve per distributed worker.
            concurrency: Size of the distributed actor pool; an int or ``(min, max)``.
            batch_format: What a callable `model` sees (``"pyarrow"`` by default).
            accelerator_type: Pin GPU actors to a model (e.g. ``"NVIDIA_A100"``).
            model_memory_gb: The model's footprint, for memory budgeting.

        Returns:
            A new lazy `Dataset` with the embedding column(s) appended.

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
        """Fetch the bytes at each URL/path into ``output_column`` (multimodal ingestion).

        The entry point of the URL table → bytes → decode → model pipeline. Reads
        ``s3://``/``gs://``/``az://``/``http(s)://``/local paths through the shared
        filesystem resolver, fetching each batch's rows concurrently and parallelizing
        across the cluster (a `map_batches` stage). See `batcher.ml.download_dataset`.

        Args:
            url_column: The column of URLs/paths to fetch.
            output_column: Name of the appended bytes column.
            max_concurrency: Concurrent fetches per batch.
            on_error: ``"raise"`` (default) or ``"null"`` to null a failed fetch.

        Returns:
            A new lazy `Dataset` with the fetched bytes appended.

        Examples:
            .. doctest::

                >>> import batcher as bt  # doctest: +SKIP
                >>> urls = bt.from_pydict(  # doctest: +SKIP
                ...     {"url": ["s3://bucket/cat.jpg", "s3://bucket/dog.jpg"]}
                ... )
                >>> images = urls.ml.download("url", output_column="bytes")  # doctest: +SKIP
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

        Args:
            data_column: The column of bytes to write.
            directory: The destination directory (any supported filesystem).
            output_column: Name of the appended written-path column.
            name_column: Column supplying each file's name; a content hash if omitted.
            extension: Suffix appended to each file name (e.g. ``".jpg"``).
            max_concurrency: Concurrent writes per batch.

        Returns:
            A new lazy `Dataset` with the written path appended.

        Examples:
            .. doctest::

                >>> import batcher as bt  # doctest: +SKIP
                >>> thumbs = ds.ml.map_batches(make_thumbnails)  # doctest: +SKIP
                >>> written = thumbs.ml.upload(  # doctest: +SKIP
                ...     "thumb", "s3://bucket/thumbs", extension=".jpg"
                ... )
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
