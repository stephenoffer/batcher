"""The `Dataset.ml` namespace — batch inference / embedding / model UDFs.

Breadth on `Dataset` lives on accessors, not new methods (the Polars pattern, and
the v2 maintainability contract). This is the ML/multimodal surface: apply a model
over Arrow batches, optionally loading it once per worker and scheduling it on GPU
actors while preprocessing stays on CPU — the heterogeneous pipeline Ray Data
specializes in. Reached as `ds.ml.infer(...)` / `ds.ml.embed(...)`.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import TYPE_CHECKING

from batcher.api.dataset._build import build_random_split, build_train_test_split
from batcher.api.dataset._dedup import (
    build_drop_near_duplicates,
    build_near_duplicates,
    build_similarity_join,
)
from batcher.ml.stats._shared import require_columns

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = ["DatasetML"]


def _public_operations() -> list[str]:
    """The public method names on `DatasetML`, for `__repr__`/`__dir__`/did-you-mean."""
    return [n for n, v in vars(DatasetML).items() if not n.startswith("_") and callable(v)]


def _as_key_columns(key: str | list[str] | None) -> list[str] | None:
    """Normalize a split key: a single column name, several, or `None` (hash every column)."""
    if key is None:
        return None
    return [key] if isinstance(key, str) else list(key)


def _require_column(ds: Dataset, column: str, *, param: str) -> None:
    """Raise a `ColumnNotFoundError` naming the closest real column when `column` is absent.

    Turns the deferred, opaque failure a wrong column name would otherwise cause deep in the
    engine into an eager, actionable one at the API edge (``did you mean 'text'?``).
    """
    require_columns(ds, column, hint=f"Pass an existing column to {param}.")


def _require_query_vector(ds: Dataset, query: list[float], column: str, *, method: str) -> None:
    """Reject a query vector that cannot match `column`, before the engine tries to.

    A query of the wrong length is *the* vector-search mistake — an embedding produced by a
    different model, or by the same model at a different Matryoshka dimension — and the
    engine's answer to it was a raw ``RuntimeError: string function list.CosineSimilarity:
    list dimensions must be equal, got left length 2 and right length 3`` from inside the
    kernel. An empty query was worse: ``ValueError: array() requires at least one element``,
    which names nothing at all.

    The width can only be checked when the column is a fixed-size list, which is what an
    embedding column normally is. A variable-length list column still gets the empty and
    non-numeric checks, and its width mismatch is left to the engine as before.
    """
    from batcher._internal.errors import PlanError

    # The column has to be an embedding column before its width is worth discussing: a text
    # column here is the mistake one step earlier (embed it first), and reporting a width
    # mismatch for it would point at the query rather than at the column.
    _require_vector_column(ds, column, method=method)
    if not query:
        raise PlanError(
            f"{method} got an empty query vector; pass the embedding to search for, with the "
            f"same number of dimensions as {column!r}."
        )
    width = _fixed_width(ds, column)
    if width is not None and width != len(query):
        raise PlanError(
            f"{method} got a {len(query)}-dimensional query but {column!r} holds "
            f"{width}-dimensional vectors. They must match — this usually means the query was "
            f"embedded by a different model, or at a different Matryoshka dimension."
        )


def _fixed_width(ds: Dataset, column: str) -> int | None:
    """The per-row width of `column` when the schema fixes one, else ``None``."""
    import pyarrow as pa

    schema = ds.schema
    field = schema.field(column) if column in schema.names else None
    if field is None:
        return None
    if pa.types.is_fixed_size_list(field.type):
        return field.type.list_size
    from batcher.io.formats.ml.tensor import is_tensor_column

    if is_tensor_column(field.type) and len(field.type.shape) == 1:
        return int(field.type.shape[0])
    return None


def _column_type(ds: Dataset, column: str):
    """`column`'s Arrow type, or ``None`` when the schema cannot be read for it.

    Every type check below is advisory in exactly one direction: it may only *reject*, and
    only on a type it positively recognizes as wrong. A schema that cannot be resolved
    (an unbound source, a plan the binder has not typed yet) returns ``None`` and the check
    passes, leaving the engine to decide as it did before.
    """
    with contextlib.suppress(Exception):
        schema = ds.schema
        if column in schema.names:
            return schema.field(column).type
    return None


def _is_vector_type(dtype) -> bool:
    """Whether `dtype` is an embedding column — a list/tensor of numbers."""
    import pyarrow as pa

    if dtype is None:
        return False
    from batcher.io.formats.ml.tensor import is_tensor_column

    if is_tensor_column(dtype):
        return True
    if not (
        pa.types.is_list(dtype)
        or pa.types.is_large_list(dtype)
        or pa.types.is_fixed_size_list(dtype)
    ):
        return False
    return pa.types.is_floating(dtype.value_type) or pa.types.is_integer(dtype.value_type)


def _is_text_type(dtype) -> bool:
    """Whether `dtype` is a text column the string kernels can read."""
    import pyarrow as pa

    if dtype is None:
        return False
    if pa.types.is_dictionary(dtype):
        return _is_text_type(dtype.value_type)
    return pa.types.is_string(dtype) or pa.types.is_large_string(dtype)


def _is_scalar_type(dtype) -> bool:
    """Whether `dtype` is a per-row scalar — wrong for both a text and a vector operator.

    Deliberately excludes Arrow's ``null``. A lazily-derived column has no resolved type until
    the plan is bound, and a projection's output reads as ``null`` in the schema long before it
    is one: `binarize_embeddings` writes a perfectly good bit code that the schema still calls
    ``null``, and rejecting that would refuse the pipeline the docs recommend. These checks may
    only reject a type they positively recognize as wrong, never one they cannot resolve.
    """
    import pyarrow as pa

    if dtype is None or pa.types.is_null(dtype):
        return False
    return bool(
        pa.types.is_boolean(dtype)
        or pa.types.is_integer(dtype)
        or pa.types.is_floating(dtype)
        or pa.types.is_decimal(dtype)
        or pa.types.is_temporal(dtype)
    )


def _require_text_column(ds: Dataset, column: str, *, method: str, param: str = "column") -> None:
    """Reject a non-text `column` here, where the message can name the vector alternative.

    The text and embedding halves of this namespace take the same-shaped argument and sit
    beside each other in the docs, so passing an embedding column to a text operator is the
    natural mistake — `similarity_join` takes a vector, `near_duplicates` takes text. The
    engine's answer to it was a `RuntimeError` raised inside a distributed worker, after the
    whole fleet had spun up: ``string function MinHash expected a Utf8 argument, got
    List(Field { name: "item", data_type: Float64, ... })``, which names an internal kernel
    and no way forward.
    """
    from batcher._internal.errors import PlanError

    _require_column(ds, column, param=param)
    dtype = _column_type(ds, column)
    if not _is_vector_type(dtype) and not _is_scalar_type(dtype):
        return  # text, or a type this check does not positively recognize as wrong
    hint = ""
    if _is_vector_type(dtype):
        hint = (
            " That looks like an embedding column: the vector equivalents are "
            "`similarity_join` (matching pairs across two datasets), "
            "`batched_nearest_neighbors`, and `near_duplicates` over a text column instead."
        )
    raise PlanError(
        f"{method} needs a text column, but {column!r} is {dtype}.{hint}"
        if hint
        else f"{method} needs a text column, but {column!r} is {dtype}. Cast it with "
        f"`bt.col({column!r}).cast('string')` if it holds text in another type."
    )


def _require_vector_column(ds: Dataset, column: str, *, method: str, param: str = "column") -> None:
    """Reject a non-embedding `column` here, where the message can name the text alternative.

    The mirror of `_require_text_column`, and the same failure without it: a cosine/simhash
    kernel raising from inside a worker about a type the user never named.
    """
    from batcher._internal.errors import PlanError

    _require_column(ds, column, param=param)
    dtype = _column_type(ds, column)
    if not _is_text_type(dtype) and not _is_scalar_type(dtype):
        return  # a vector, or a type this check does not positively recognize as wrong
    hint = ""
    if _is_text_type(dtype):
        hint = (
            " That looks like a text column: embed it first with `ds.ml.embed(...)`, or use "
            "the text-similarity operators (`near_duplicates`, `drop_near_duplicates`)."
        )
    raise PlanError(
        f"{method} needs an embedding column (a list of numbers), but {column!r} is {dtype}.{hint}"
    )


def _warn_extract_overwrites(ds: Dataset, schema: dict, prompt_column: str | None) -> None:
    """Warn when an `extract` schema field replaces a column that already exists.

    Extracted fields are written with `with_columns` semantics, so a schema field named
    after an existing column silently replaces it. Naming one after the **prompt column** is
    the version that bites: the extraction eats its own input, and the prompts are simply
    gone from the result with nothing raised. Nobody asks for that on purpose, and the
    general case — quietly overwriting a column the pipeline still needs — is worth a word
    too. A warning rather than an error, because replacing a column is a legitimate thing to
    ask an extraction to do once you know you are asking for it.
    """
    clashes = sorted(set(schema) & set(ds.columns))
    if not clashes:
        return
    import warnings

    from batcher._internal.errors import DataWarning

    prompt = f" — including the prompt column {prompt_column!r}" if prompt_column in clashes else ""
    warnings.warn(
        f"extract() schema field(s) {clashes} already exist on this dataset and will be "
        f"replaced by the extracted values{prompt}. Rename the schema field if you meant to "
        f"keep the original column.",
        DataWarning,
        stacklevel=3,
    )


def _require_llm_columns(
    ds: Dataset,
    *,
    method: str,
    prompt_column: str | None,
    template: str | None,
    image_column: str | None,
) -> None:
    """Check the column arguments of `generate`/`extract`/`classify` at the API edge.

    `template` was already checked — it produces a clear "references column(s) not in the
    data" error naming the available columns — but the plainer `prompt_column` and
    `image_column` next to it were not, and failed with a bare pyarrow
    ``KeyError: 'Field "nope" does not exist in schema'`` from inside the UDF. Two arguments
    of the same call reporting the same mistake two different ways is the kind of gap that
    makes an API feel arbitrary, and the bare one is the harder to act on.

    A `template` supersedes `prompt_column` for prompt building, so when one is given the
    prompt column is not read and is not required to exist — checking it there would reject a
    call that works.
    """
    if prompt_column is not None and template is None:
        _require_column(ds, prompt_column, param=f"{method}(prompt_column=)")
    if image_column is not None:
        _require_column(ds, image_column, param=f"{method}(image_column=)")


def _require_columns(ds: Dataset, columns: list[str] | None, *, param: str) -> None:
    """`_require_column` for a whole selection, checked eagerly and left alone when ``None``.

    The loaders (`to_numpy_batches`, `iter_torch_batches`, `to_torch`, `to_tf`,
    `stream_loader`) are **generators**, so a mistyped column name did not raise where it was
    written. It raised a bare pyarrow ``KeyError: 'Field "NOPE" does not exist in schema'`` on
    the first pull — which, for a training loader, is inside the first step of the training
    loop, with no mention of the parameter, the method, or the columns that do exist.
    """
    if not columns:
        return
    for column in columns:
        _require_column(ds, column, param=param)


def _reject_model_id_only(
    method: str, model: object, given: dict[str, tuple[object, object]]
) -> None:
    """Reject options that only apply when `model` is a model *identifier*, not a callable.

    `infer` and `embed` have two shapes. Given a HuggingFace model id they build the encoder
    themselves, so `column`, `output_column`, `device`, `normalize` and friends are theirs to
    honor. Given a callable or class they forward straight to `map_batches`, where those
    arguments have no meaning at all — and every one of them was being dropped in silence.

    That is the worst kind of silence: ``embed(MyEncoder, normalize=True)`` returned
    unnormalized vectors, and ``infer(Model, output_column="pred")`` wrote to whatever the
    callable happened to name its column. Both look like they worked. Naming the ignored
    arguments and refusing is the only version a user can act on.

    Args:
        method: The method name, for the message (``"infer"`` / ``"embed"``).
        model: The `model` argument, which decides which shape this call is.
        given: ``{name: (value, default)}`` for each identifier-only option.

    Raises:
        PlanError: if `model` is not a model identifier and any option differs from default.
    """
    if isinstance(model, str):
        return
    ignored = sorted(name for name, (value, default) in given.items() if value != default)
    if not ignored:
        return
    from batcher._internal.errors import PlanError

    raise PlanError(
        f"ds.ml.{method}() got {ignored}, which only apply when the first argument is a model "
        f"identifier — with a callable or class the call forwards to map_batches, where they "
        f"have no effect. Move that behavior inside the callable, and use output_columns=[...] "
        f"to declare the schema it produces."
    )


def _warn_if_training_on_sorted_data(plan: object, shuffle: bool, window: object) -> None:
    """Warn when a training loader is built over an explicitly sorted plan without shuffling.

    Training on ordered data is the classic silent convergence bug: the model sees all of one
    class, then all of the next, and the loss curve looks merely disappointing rather than
    wrong. The guides file it under "training loss not decreasing" and "non-deterministic
    results", where the listed causes are the learning rate and the data — never the ordering.

    `shuffle=False` is the right default here because it is `torch.utils.data.DataLoader`'s,
    and a user reaching for this method is coming from torch. So instead of changing the
    default, the *plan* is inspected: a `sort` the user wrote themselves, feeding a loader
    with no shuffling, is a combination almost nobody wants. Pure plan shape — no scan, no
    cost, and silent for a corpus that was never sorted.
    """
    if shuffle or window:
        return
    from batcher.plan.logical import Sort
    from batcher.plan.profile import logical_preorder

    if not any(isinstance(node, Sort) for _depth, node in logical_preorder(plan)):
        return
    import warnings

    from batcher._internal.errors import PerformanceWarning

    warnings.warn(
        "this dataset is explicitly sorted and the loader was built without shuffling, so "
        "each training step sees a contiguous run of the sort key — the model learns the "
        "ordering, and the only symptom is a loss curve that looks merely disappointing. "
        "Pass shuffle=True (or local_shuffle_buffer_size=N) unless the order is the point, "
        "as it is for a sequence model.",
        PerformanceWarning,
        stacklevel=3,
    )


class DatasetML:
    """Accessor for ML/multimodal operations over a `Dataset` (`ds.ml`).

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, 2.0], "y": [0, 1]})
            >>> train, test = ds.ml.train_test_split(0.5, seed=0)
            >>> train.count() + test.count()
            2
    """

    __slots__ = ("_ds",)

    def __init__(self, ds: Dataset) -> None:
        """Bind the ML accessor to its `Dataset`; reached as `ds.ml`, not constructed directly."""
        self._ds = ds

    def __repr__(self) -> str:
        """``<ds.ml accessor: infer, embed, map_batches, ...>`` — a discoverable summary."""
        ops = ", ".join(_public_operations())
        return f"<ds.ml accessor: {ops}>"

    def __dir__(self) -> list[str]:
        """Expose the ML operations to ``dir()`` and editor autocompletion."""
        return sorted(set(_public_operations()) | set(object.__dir__(self)))

    def __getattr__(self, name: str) -> object:
        """Raise an `AttributeError` naming the spelling to use for an unknown operation.

        Only reached when normal attribute lookup fails, so a real method never gets here. A
        UDF verb that moved onto `Dataset` (``ds.ml.map_batches``) names its new home, and a
        near-miss (``ds.ml.embedd``) is caught with a suggestion rather than a bare error.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        from batcher._internal.errors import absent_error
        from batcher.api.dataset.compat.guidance._dataset_naming import DATASET_ML_MOVED

        raise absent_error(
            "ds.ml", name, DATASET_ML_MOVED, _public_operations(), receiver="Dataset.ml"
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
        device: str | None = None,
        dtype: str | None = None,
        num_workers: int | str = "auto",
        num_gpus: float = 0.0,
        concurrency: int | tuple[int, int] | None = None,
        batch_format: str = "pyarrow",
        model_kwargs: dict | None = None,
        accelerator_type: str | None = None,
        model_memory_gb: float = 0.0,
        max_errored_rows: int = 0,
        timeout: float = 0.0,
        max_retries: int = 0,
        retry_backoff: float = 0.5,
        retry_on: type[BaseException] | tuple[type[BaseException], ...] | None = None,
    ) -> Dataset:
        """Run batch model inference over the dataset (ML/multimodal path).

        Pass a **model identifier** (a HuggingFace ``transformers`` model id) and the
        `column` to score: the model loads once per worker and its prediction is
        appended as `output_column`. `task` selects the pipeline kind
        (``"sentiment-analysis"``, ``"text-classification"``,
        ``"automatic-speech-recognition"``, …; inferred from the model when omitted).
        Needs ``transformers`` (``batcher-engine[transformers]``).

        `column` may be text, a binary column of encoded bytes, or a decoded waveform or
        image-tensor column — the column's type decides what the pipeline is fed, so an
        audio or vision model reads the output of the native decode expressions directly.

        For a model that was **exported** rather than loaded from the hub — an ONNX graph,
        a TorchScript archive, an OpenVINO IR — use
        `map_batches` with `bt.ml.onnx_predictor` / `bt.ml.torch_predictor` /
        `bt.ml.openvino_predictor`, which run the exported form in the worker with no
        framework load at all.

        Pass a **callable or class** instead for full control (a class loads the model
        once per worker — the GPU-inference pattern); the call then mirrors
        `map_batches`, with `output_columns` declaring the result schema. The options
        that exist only to configure the identifier path — `column`, `output_column`,
        `task`, `device`, `dtype`, `model_kwargs` — are **rejected** in that shape rather
        than ignored, because the callable is where that behavior now lives.

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
            device: Where the model runs (model-id path): ``"auto"``/``None`` detect the
                accelerator, or force ``"cuda"``/``"cpu"``/``"mps"``.
            dtype: Model precision (model-id path): ``"float16"``, ``"bfloat16"``,
                ``"float32"``, or an abbreviation such as ``"fp16"``; auto when omitted.
            num_workers: Concurrent per-batch calls within a worker (``"auto"`` sizes to
                the stage), or an explicit int.
            num_gpus: GPUs to reserve per distributed worker.
            concurrency: Size of the distributed actor pool; an int or ``(min, max)``.
            batch_format: What a callable `model` sees (``"pyarrow"`` by default).
            model_kwargs: Extra keyword arguments for the model load (model-id path),
                e.g. ``{"trust_remote_code": True}``.
            accelerator_type: Pin GPU actors to a model (e.g. ``"NVIDIA_A100"``).
            model_memory_gb: The model's footprint, for memory budgeting.
            max_errored_rows: Rows a raising model may drop per worker before failing.
            timeout: Wall-clock ceiling (seconds) for one model call; 0 = no timeout.
            max_retries: Times to retry a batch whose model raises a retryable error.
            retry_backoff: Base backoff (seconds); attempt `k` waits `retry_backoff * 2**k`.
            retry_on: Exception type(s) worth retrying; ``None`` retries any `Exception`.

        Returns:
            A new lazy `Dataset` with the prediction column(s) appended.

        Raises:
            PlanError: if a model id is given without `column`.
            ColumnNotFoundError: if `column` is not in the dataset (model-id path).

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
            _require_column(self._ds, column, param="column")
            from batcher.ml.inference import transformers_pipeline_encoder

            encoder = transformers_pipeline_encoder(
                model,
                column,
                output_column=output_column,
                task=task,
                device=device,
                dtype=dtype,
                model_kwargs=tuple(sorted((model_kwargs or {}).items())),
            )
            cols = (
                [*self._ds.columns, output_column]
                if output_column not in self._ds.columns
                else None
            )
            return self._ds.map_batches(
                encoder,
                output_columns=cols,
                batch_size=batch_size,
                num_workers=num_workers,
                num_gpus=num_gpus,
                concurrency=concurrency,
                accelerator_type=accelerator_type,
                model_memory_gb=model_memory_gb,
                max_errored_rows=max_errored_rows,
                timeout=timeout,
                max_retries=max_retries,
                retry_backoff=retry_backoff,
                retry_on=retry_on,
            )
        _reject_model_id_only(
            "infer",
            model,
            {
                "column": (column, None),
                "output_column": (output_column, "prediction"),
                "task": (task, None),
                "device": (device, None),
                "dtype": (dtype, None),
                "model_kwargs": (model_kwargs, None),
            },
        )
        return self._ds.map_batches(
            model,
            batch_size=batch_size,
            output_columns=output_columns,
            num_workers=num_workers,
            num_gpus=num_gpus,
            concurrency=concurrency,
            batch_format=batch_format,
            accelerator_type=accelerator_type,
            model_memory_gb=model_memory_gb,
            max_errored_rows=max_errored_rows,
            timeout=timeout,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            retry_on=retry_on,
        )

    def predict(
        self,
        model: object,
        *,
        features: list[str] | None = None,
        framework: str | None = None,
        method: str = "predict",
        output_column: str = "prediction",
        output_columns: list[str] | None = None,
        as_list: bool = False,
        missing: float = float("nan"),
        dtype: str | None = None,
        threads: int | None = None,
        batch_size: int | None = None,
        num_workers: int | str = "auto",
        num_gpus: float = 0.0,
        concurrency: int | tuple[int, int] | None = None,
        accelerator_type: str | None = None,
        model_memory_gb: float = 0.0,
        options: dict[str, object] | None = None,
        max_errored_rows: int = 0,
        timeout: float = 0.0,
        max_retries: int = 0,
        retry_backoff: float = 0.5,
        retry_on: type[BaseException] | tuple[type[BaseException], ...] | None = None,
    ) -> Dataset:
        """Score a fitted **tabular** model over the dataset (XGBoost, LightGBM, sklearn, …).

        The classical-ML counterpart of `infer`. Pass a fitted model object or a path to a
        saved one, name the feature columns in the order the model was trained on, and the
        prediction is appended as `output_column`. The model loads **once per worker** and
        each batch's features are assembled into one dense matrix, so nothing crosses the
        boundary a row at a time.

        The framework is detected from the model (or the file extension) and covers
        ``xgboost``, ``lightgbm``, ``catboost``, ``sklearn`` (any fitted estimator or
        ``Pipeline``), and ``onnx``. `method` is uniform across all of them:

        - ``"predict"`` — the model's natural output.
        - ``"predict_proba"`` — class probabilities, one column per class.
        - ``"raw"`` — the untransformed margin / decision function.
        - ``"leaf"`` — the leaf index per tree (boosters only).
        - ``"contrib"`` — per-feature SHAP contributions (boosters only).

        A null feature becomes NaN, which is what XGBoost and LightGBM treat as missing;
        pass `missing` when the model was trained with a different sentinel. Feature order
        is checked against the model's own recorded feature names where it has them,
        because a re-ordered feature list silently changes every prediction rather than
        raising anywhere.

        Args:
            model: A fitted model object, or a path/URI to a saved model.
            features: The feature columns in model order; every column when omitted.
            framework: Force the framework instead of detecting it.
            method: What to compute — see the list above.
            output_column: Base name of the appended prediction column(s).
            output_columns: Explicit names for a multi-output model's columns.
            as_list: Emit one `List<Float64>` column instead of one column per output.
            missing: The value a null feature takes in the matrix (NaN by default).
            dtype: Feature-matrix dtype, ``"float32"`` or ``"float64"``. Defaults to the
                framework's own precision: float32 for the boosters, which compute in it
                anyway, and float64 for scikit-learn, which does not — feeding a float64
                estimator float32 shifts the last digits of every prediction against what
                the same estimator returns in-process.
            threads: The model's own thread pool size per worker; auto-capped when unset.
            batch_size: Rows per scoring call; larger amortizes the per-call overhead.
            num_workers: Concurrent per-batch calls within a worker.
            num_gpus: GPUs to reserve per distributed worker.
            concurrency: Size of the distributed actor pool.
            accelerator_type: Pin GPU actors to a device model.
            model_memory_gb: The model's footprint, for memory budgeting.
            options: Extra framework keywords, e.g. ``{"iteration_range": (0, 50)}``.
            max_errored_rows: Rows a raising model may drop per worker before failing.
            timeout: Wall-clock ceiling (seconds) for one model call; 0 = no timeout.
            max_retries: Times to retry a batch whose model raises a retryable error.
            retry_backoff: Base backoff (seconds); attempt `k` waits `retry_backoff * 2**k`.
            retry_on: Exception type(s) worth retrying; ``None`` retries any `Exception`.

        Returns:
            A new lazy `Dataset` with the prediction column(s) appended.

        Raises:
            PlanError: On an unknown framework or method, an empty feature list, or a
                feature order that contradicts the model's own.
            ColumnNotFoundError: If a named feature column is not in the dataset.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from sklearn.linear_model import LinearRegression
                >>> model = LinearRegression().fit([[0.0], [1.0], [2.0]], [0.0, 2.0, 4.0])
                >>> ds = bt.from_pydict({"x": [3.0, 4.0]})
                >>> scored = ds.ml.predict(model, features=["x"])
                >>> [round(v, 6) for v in scored.to_pydict()["prediction"]]
                [6.0, 8.0]
        """
        from batcher.ml.tabular import (
            detect_framework,
            predicted_column_names,
            resolve_features,
            tabular_predictor,
        )

        feature_list = resolve_features(features, self._ds.columns)
        resolved = framework or detect_framework(model)
        appended = predicted_column_names(
            model,
            framework=resolved,
            method=method,
            features=feature_list,
            output_column=output_column,
            output_columns=output_columns,
            as_list=as_list,
        )
        udf = tabular_predictor(
            model,
            tuple(feature_list),
            framework=resolved,
            method=method,
            output_column=output_column,
            output_columns=tuple(appended),
            as_list=as_list,
            missing=missing,
            dtype=dtype,
            threads=threads,
            options=tuple(sorted((options or {}).items())),
        )
        new = [c for c in appended if c not in self._ds.columns]
        return self._ds.map_batches(
            udf,
            output_columns=[*self._ds.columns, *new],
            batch_size=batch_size,
            num_workers=num_workers,
            num_gpus=num_gpus,
            concurrency=concurrency,
            accelerator_type=accelerator_type,
            model_memory_gb=model_memory_gb,
            max_errored_rows=max_errored_rows,
            timeout=timeout,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            retry_on=retry_on,
        )

    def evaluate(
        self,
        y_true: str,
        *,
        y_pred: str | None = None,
        y_score: str | None = None,
        task: str = "auto",
        metrics: list[str] | None = None,
        positive: object = 1,
        threshold: float = 0.5,
        by: str | list[str] | None = None,
    ) -> dict[str, float] | Dataset:
        """Score predictions against labels, returning the task's whole metric set.

        The evaluation counterpart of `predict`. Every aggregate metric is computed in one
        pass over the predictions, so a ten-metric report costs what one costs; the
        rank-based metrics (``roc_auc``, ``average_precision``, ``ks``, ``gini``) each add
        a sort and are computed only when requested.

        `by` is the reason this is a query rather than a function call: it reports the same
        metrics per segment, per day, or per cohort over the full dataset, which is the
        question a model review actually asks and the one a driver-side
        ``sklearn.metrics`` call cannot answer at scale.

        For a binary task, `y_score` alone is enough: the hard predictions are derived at
        `threshold`, so the threshold metrics and the ranking metrics come from one column.

        Args:
            y_true: The label column.
            y_pred: The hard-prediction column (a label, or a value for regression).
            y_score: The predicted probability of the positive class, for a binary task.
            task: ``"binary"``, ``"multiclass"``, ``"regression"``, or ``"auto"``.
                ``"auto"`` reads it off `y_true`: a float label is a regression, and an
                integer, string or boolean one is a classification with as many classes as
                it has distinct values.
            metrics: The metric names to compute; the task's default set when omitted.
            positive: The label value that counts as the positive class.
            threshold: The cutoff turning `y_score` into a hard prediction.
            by: Column(s) to report a separate row of metrics for.

        Returns:
            A ``{metric: value}`` dict, or a `Dataset` of one row per group when `by` is
            given.

        Raises:
            PlanError: On an unknown task or metric name, or when neither `y_pred` nor
                `y_score` is given.
            ColumnNotFoundError: If a named column is not in the dataset.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"y": [1, 0, 1, 0], "s": [0.9, 0.2, 0.8, 0.4]})
                >>> ds.ml.evaluate("y", y_score="s")["accuracy"]
                1.0
        """
        from batcher.ml.metrics import evaluate

        return evaluate(
            self._ds,
            y_true,
            y_pred=y_pred,
            y_score=y_score,
            task=task,
            metrics=metrics,
            positive=positive,
            threshold=threshold,
            by=by,
        )

    def train_test_split(
        self,
        test_size: float = 0.25,
        *,
        seed: int = 0,
        key: str | list[str] | None = None,
        stratify: str | None = None,
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
            stratify: A column whose distribution to hold constant across both halves.
                Without it the split is proportional only in expectation, so a rare class can
                land almost entirely on one side and every metric computed on the other
                becomes meaningless. Pass the label column on an imbalanced problem.
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
        if stratify is not None:
            from batcher.ml.splitting import stratified_split

            return stratified_split(self._ds, stratify, test_size=test_size, seed=seed, key=key)
        return build_train_test_split(self._ds, test_size, seed=seed, key=_as_key_columns(key))

    def drift(
        self,
        reference: Dataset,
        columns: list[str] | None = None,
        *,
        buckets: int = 10,
    ) -> Dataset:
        """Compare this dataset's feature distributions against a `reference` one.

        The check a deployed model needs before its labels arrive: the code is unchanged,
        the accuracy is unmeasurable, and the only observable thing is whether the *inputs*
        still look like the ones the model was trained on.

        Bin edges always come from `reference`, then apply unchanged here, so a shift shows
        up as mass moving between bins rather than as the bins themselves moving. Deriving
        the edges separately for each side would make two very different distributions look
        identical.

        Read the PSI with the conventional thresholds: below 0.1 is stable, 0.1 to 0.25 is
        moderate, above 0.25 warrants retraining. The result is a `Dataset` rather than a
        dict so it appends to a monitoring table with a timestamp — a single PSI says much
        less than its history.

        Args:
            reference: The baseline dataset, usually the training data.
            columns: The numeric columns to check; every numeric column when omitted.
            buckets: How many quantile bins to build from the reference per column.

        Returns:
            A `Dataset` of ``column``, ``psi``, ``js_divergence``, ``mean_shift``,
            ``null_rate_shift``, ordered by descending PSI.

        Raises:
            PlanError: If `columns` is empty, or a reference column is constant.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> train = bt.from_pydict({"x": [float(i) for i in range(100)]})
                >>> today = bt.from_pydict({"x": [float(i) + 60 for i in range(100)]})
                >>> today.ml.drift(train, ["x"], buckets=4).to_pydict()["psi"][0] > 0.25
                True
        """
        from batcher.ml.stats import drift_report

        names = columns if columns is not None else list(self._ds.select_dtypes("number").columns)
        return drift_report(reference, self._ds, names, buckets=buckets)

    def kfold(
        self,
        k: int = 5,
        *,
        seed: int = 0,
        key: str | list[str] | None = None,
        stratify: str | None = None,
        group: str | None = None,
    ) -> list[tuple[Dataset, Dataset]]:
        """Split into `k` ``(train, validation)`` pairs for cross-validation.

        Every row validates exactly once. Each pair is two lazy `Dataset`s over the same
        deterministic fold assignment, so an unused fold costs nothing and the assignment
        is identical however the data is partitioned — single-node, distributed, or
        streaming. No shuffle and no materialized index array.

        `stratify` and `group` select the variant the data needs, and picking the right one
        is usually the difference between a trustworthy score and a misleading one:

        - `stratify` keeps each label's proportion the same in every fold. Use it whenever
          the label is imbalanced, or the fold-to-fold variance measures the split rather
          than the model.
        - `group` keeps every row of a group in the same fold. Use it whenever rows repeat
          an entity — a user, a patient, a session. Without it the model memorizes the
          entity, cross-validation looks excellent, and production does not.

        Args:
            k: How many folds.
            seed: Seed for the fold assignment; the same seed reproduces it.
            key: The column(s) identifying a row. Prefer it on a real corpus: hashing only
                these keeps the assignment stable when other columns change.
            stratify: A label column whose distribution every fold must preserve.
            group: A column that must not span folds.

        Returns:
            `k` ``(train, validation)`` pairs, in fold order.

        Raises:
            PlanError: If `k` is less than 2, or both `stratify` and `group` are given.
            ColumnNotFoundError: If a named column is not in the dataset.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.range(0, 100)
                >>> folds = ds.ml.kfold(4, key="value")
                >>> sum(validate.count() for _, validate in folds)
                100
        """
        from batcher.ml.splitting import group_kfold, kfold, stratified_kfold

        if stratify is not None and group is not None:
            from batcher._internal.errors import PlanError

            raise PlanError(
                "kfold() takes stratify= or group=, not both: one spreads a label evenly "
                "across folds and the other keeps a group inside one, and they cannot both "
                "hold. Pick the constraint that matters for this dataset."
            )
        if stratify is not None:
            return stratified_kfold(self._ds, stratify, k, seed=seed, key=key)
        if group is not None:
            return group_kfold(self._ds, group, k, seed=seed)
        return kfold(self._ds, k, seed=seed, key=key)

    def time_series_split(
        self, time_column: str, n_splits: int = 5, *, expanding: bool = True
    ) -> list[tuple[Dataset, Dataset]]:
        """Split chronologically into ``(train, validation)`` pairs — never train on the future.

        The only correct cross-validation for a time series, and the one `kfold` silently
        breaks: a random fold puts next week's rows in the training set, so the model sees
        the future and the validation score is one no deployment will reproduce.

        Split *i* trains on everything before the *i*-th time cut and validates on the
        window that follows it. `expanding` grows the training window with each split,
        matching a model retrained on all history; ``expanding=False`` slides a
        fixed-width window instead, matching one that deliberately forgets.

        Args:
            time_column: The column defining chronological order.
            n_splits: How many train/validation pairs to produce.
            expanding: Grow the training window (default) rather than sliding it.

        Returns:
            `n_splits` ``(train, validation)`` pairs, earliest first.

        Raises:
            PlanError: If `n_splits` is less than 1.
            ColumnNotFoundError: If `time_column` is not in the dataset.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"t": list(range(100)), "x": list(range(100))})
                >>> [(tr.count(), va.count()) for tr, va in ds.ml.time_series_split("t", 4)]
                [(20, 20), (40, 20), (60, 20), (80, 19)]

        Note:
            Each validation window is half-open, ``[cut_i, cut_i+1)``, and the last cut is
            the maximum of `time_column`, so the single latest row falls outside every
            validation fold. That is why the final pair above validates on 19 rows rather
            than 20.
        """
        from batcher.ml.splitting import time_series_split

        return time_series_split(self._ds, time_column, n_splits, expanding=expanding)

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
        _require_text_column(self._ds, column, method="near_duplicates")
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
        _require_text_column(self._ds, column, method="drop_near_duplicates")
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
        _require_vector_column(self._ds, left_on, method="similarity_join", param="left_on")
        _require_vector_column(
            other,
            right_on if right_on is not None else left_on,
            method="similarity_join",
            param="right_on",
        )
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

    def nearest_neighbors(
        self,
        query: list[float],
        *,
        column: str = "embedding",
        k: int = 10,
        metric: str = "cosine",
        distance_column: str = "distance",
    ) -> Dataset:
        """Return the `k` rows whose embedding is nearest to `query` (exact brute force).

        The retrieval primitive for RAG / similarity lookup when there is **no index**:
        every row's embedding is scored against the one `query` vector, and the `k` nearest
        are kept, with their distance in `distance_column`. It is exact (unlike
        `similarity_join`'s LSH recall trade-off) and composes the operators the engine
        already has — a projected distance, a sort, and a limit — so it runs wherever a
        sort runs, including distributed, and nothing is materialized on the driver.

        For a large corpus queried repeatedly, build a real ANN index instead (see
        `batcher.ml.build_vector_index` / `vector_search`); brute force is `O(n)` per query.

        Args:
            query: The query embedding as a list of floats. Its length must match the
                stored vectors' dimension.
            column: The embedding column to search (a list/tensor of floats).
            k: How many nearest rows to return.
            metric: ``"cosine"`` (default), ``"l2"`` (Euclidean), ``"l1"`` (Manhattan),
                ``"hamming"`` (for binary/quantized codes), or ``"dot"`` (inner product).
                All but ``dot`` rank by smallest distance; ``dot`` ranks by largest.
            distance_column: Name of the appended score column.

        Returns:
            A new `Dataset` of the `k` nearest rows, nearest first, with `distance_column`
            appended.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict(
                ...     {"id": [1, 2, 3], "embedding": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]}
                ... )
                >>> hits = ds.ml.nearest_neighbors([1.0, 0.0], column="embedding", k=2)
                >>> hits.to_pydict()["id"]
                [1, 3]
        """
        from batcher._internal.errors import PlanError
        from batcher.plan.expr_ir import array, col, lit

        if k < 1:
            raise PlanError(f"nearest_neighbors k must be >= 1, got {k}")
        _require_query_vector(self._ds, query, column, method="nearest_neighbors")
        q = array(*[lit(float(x)) for x in query])
        vec = col(column)
        # cosine/l2/l1/hamming are distances (smaller = nearer); dot is a similarity.
        if metric == "cosine":
            score, descending = vec.list.cosine_distance(q), False
        elif metric == "l2":
            score, descending = vec.list.l2_distance(q), False
        elif metric == "l1":
            score, descending = vec.list.l1_distance(q), False
        elif metric == "hamming":
            score, descending = vec.list.hamming_distance(q), False
        elif metric == "dot":
            score, descending = vec.list.dot(q), True
        else:
            raise PlanError(
                f"metric must be 'cosine', 'l2', 'l1', 'hamming', or 'dot', got {metric!r}"
            )
        return (
            self._ds.with_columns(**{distance_column: score})
            .sort(distance_column, descending=descending)
            .limit(k)
        )

    def normalize_embeddings(self, column: str, *, output_column: str | None = None) -> Dataset:
        """Unit-normalize an embedding column (L2 norm = 1), in the data plane.

        The standard preprocessing before dot-product retrieval: on unit vectors the inner
        product ranks identically to cosine similarity but skips the per-query norm, so it
        is the cheap, index-friendly form. A zero vector stays zero (no divide-by-zero).
        Runs as a native `.list.normalize()` projection — no per-row Python.

        Args:
            column: The embedding column (a list/tensor of floats).
            output_column: Where to write the normalized vector; defaults to `column`
                (in place).

        Returns:
            A new `Dataset` with the normalized embedding column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"emb": [[3.0, 4.0]]})
                >>> ds.ml.normalize_embeddings("emb").to_pydict()
                {'emb': [[0.6, 0.8]]}
        """
        _require_vector_column(self._ds, column, method="normalize_embeddings")
        from batcher.plan.expr_ir import col

        out = output_column or column
        return self._ds.with_columns(**{out: col(column).list.normalize()})

    def truncate_embeddings(
        self,
        column: str,
        dim: int,
        *,
        output_column: str | None = None,
        normalize: bool = True,
    ) -> Dataset:
        """Shorten a Matryoshka embedding to its first `dim` dimensions, re-normalized.

        Matryoshka-trained models (``text-embedding-3-*``, Nomic, mxbai) pack the most
        information into the leading dimensions, so keeping a prefix trades a little recall
        for a smaller, faster index — 256 of 1536 dims is a common 6x storage win. The
        catch is that a raw prefix is no longer unit length, and a cosine index assumes it
        is; `normalize` re-normalizes by default so the truncated vectors stay searchable.
        A plain ``.list.slice`` without it is the silent-wrong-results version of this.

        Runs as a native `.list.slice` (+ `.list.normalize`) projection — no per-row Python.

        Args:
            column: The embedding column (a list of floats).
            dim: How many leading dimensions to keep; must be at least 1.
            output_column: Where to write the truncated vector; defaults to `column`.
            normalize: Re-L2-normalize the prefix (the default, correct for cosine search).

        Returns:
            A new `Dataset` with the truncated embedding column.

        Raises:
            PlanError: if `dim` is less than 1.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"emb": [[3.0, 4.0, 1.0, 1.0]]})
                >>> ds.ml.truncate_embeddings("emb", 2).to_pydict()
                {'emb': [[0.6, 0.8]]}
        """
        from batcher._internal.errors import PlanError
        from batcher.plan.expr_ir import col

        if dim < 1:
            raise PlanError(f"truncate_embeddings dim must be >= 1, got {dim}")
        _require_vector_column(self._ds, column, method="truncate_embeddings")
        out = output_column or column
        vec = col(column).list.slice(0, dim)
        if normalize:
            vec = vec.list.normalize()
        return self._ds.with_columns(**{out: vec})

    def drop_degenerate_embeddings(self, column: str) -> Dataset:
        """Drop rows whose embedding is null or the zero vector — the pre-index filter.

        A zero vector has no direction, so its cosine similarity to everything is undefined
        and an index returns it as a garbage neighbor; it usually means an empty input or a
        failed encode. A null vector is worse. Removing both before building an index is the
        difference between clean retrieval and silent holes in it. Runs as a native
        `.list.is_zero_vector` filter — no per-row Python — and keeps the input schema.

        Args:
            column: The embedding column to check (a list of floats).

        Returns:
            A new `Dataset` with the degenerate rows removed.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict(
                ...     {"id": [1, 2, 3], "emb": [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]}
                ... )
                >>> ds.ml.drop_degenerate_embeddings("emb").to_pydict()["id"]
                [1, 3]
        """
        _require_vector_column(self._ds, column, method="drop_degenerate_embeddings")
        from batcher.plan.expr_ir import col, lit

        # is_zero_vector is null for a null vector, so `== False` drops both the zero and
        # the null rows (a null predicate row is filtered out) — exactly the degenerate set.
        return self._ds.filter(col(column).list.is_zero_vector() == lit(False))

    def binarize_embeddings(self, column: str, *, output_column: str | None = None) -> Dataset:
        """Turn a float embedding into a 0/1 **sign code** for Hamming-distance retrieval.

        Binary embeddings trade a little recall for a much cheaper distance: each dimension
        becomes a single bit by its sign, and search ranks by Hamming distance (the count of
        differing bits) instead of a float dot product. Pair with
        ``nearest_neighbors(..., metric="hamming")`` or ``.list.hamming_distance`` — the
        code composes a native `.list.transform`, so there is no per-row Python.

        Args:
            column: The float embedding column (a list of floats).
            output_column: Where to write the code; defaults to `column` (in place).

        Returns:
            A new `Dataset` with the binary code column (each element ``0`` or ``1``).

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"emb": [[0.5, -0.2, 0.9, -0.1]]})
                >>> ds.ml.binarize_embeddings("emb", output_column="code").to_pydict()["code"]
                [[1, 0, 1, 0]]
        """
        _require_vector_column(self._ds, column, method="binarize_embeddings")
        from batcher.plan.expr_ir import col, lit
        from batcher.plan.functions.collection import element

        out = output_column or column
        code = col(column).list.transform((element() > lit(0.0)).cast("int"))
        return self._ds.with_columns(**{out: code})

    def similarity_to(
        self,
        query: list[float],
        *,
        column: str = "embedding",
        metric: str = "cosine",
        output_column: str = "score",
    ) -> Dataset:
        """Score every row's embedding against a fixed `query` vector (→ a new column).

        The retrieval-scoring step without the top-k cut: use it to threshold, rerank, or
        combine the score with other predicates before selecting. `nearest_neighbors` is
        this plus a sort and a limit. Composes the native `.list` distance kernels, so it
        runs wherever a projection runs, including distributed.

        Args:
            query: The query embedding as a list of floats (length must match the column).
            column: The embedding column to score (a list/tensor of floats).
            metric: ``"cosine"`` similarity (default), ``"dot"`` inner product, ``"l2"``
                (negative Euclidean distance, so larger is still nearer), or ``"l1"``
                (negative Manhattan distance).
            output_column: Name of the appended score column.

        Returns:
            A new `Dataset` with `output_column` appended (larger = more similar).

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"id": [1, 2], "emb": [[1.0, 0.0], [0.0, 1.0]]})
                >>> out = ds.ml.similarity_to([1.0, 0.0], column="emb").to_pydict()
                >>> round(out["score"][0], 4), round(out["score"][1], 4)
                (1.0, 0.0)
        """
        from batcher._internal.errors import PlanError
        from batcher.plan.expr_ir import array, col, lit

        _require_query_vector(self._ds, query, column, method="similarity_to")
        q = array(*[lit(float(x)) for x in query])
        vec = col(column)
        if metric == "cosine":
            score = vec.list.cosine_similarity(q)
        elif metric == "dot":
            score = vec.list.dot(q)
        elif metric == "l2":
            score = vec.list.l2_distance(q) * -1.0  # negate so "larger = nearer" holds
        elif metric == "l1":
            score = vec.list.l1_distance(q) * -1.0
        else:
            raise PlanError(f"metric must be 'cosine', 'dot', 'l2', or 'l1', got {metric!r}")
        return self._ds.with_columns(**{output_column: score})

    def batched_nearest_neighbors(
        self,
        queries: Dataset,
        *,
        query_key: str,
        query_column: str,
        corpus_key: str,
        column: str = "embedding",
        k: int = 10,
        metric: str = "cosine",
        distance_column: str = "distance",
        rank_column: str | None = None,
    ) -> Dataset:
        """The `k` nearest corpus rows for **each** query row — brute-force, exact, no index.

        The multi-query form of `nearest_neighbors`, for retrieval evaluation: score a whole
        query set against this corpus and keep each query's top `k`. It composes a cross
        join, a distance projection, and a per-query windowed rank, so it runs wherever those
        run, including distributed, and nothing materializes on the driver.

        It is ``O(queries x corpus)`` by construction — right for an eval set against a
        modest corpus, wrong for production serving. For a large corpus queried repeatedly,
        build an ANN index (`batcher.ml.build_vector_index` / `vector_search`) instead.

        Args:
            queries: A `Dataset` of query rows, each with `query_key` and `query_column`.
            query_key: The column identifying a query row.
            query_column: The query embedding column (a list of floats).
            corpus_key: The column identifying a corpus row (on this dataset).
            column: The corpus embedding column to search.
            k: How many nearest corpus rows to keep per query.
            metric: ``"cosine"`` (default), ``"l2"``, ``"l1"``, ``"hamming"``, or ``"dot"``.
            distance_column: Name of the appended distance/score column.
            rank_column: If set, also append each row's 1-based rank within its query — the
                hook a rank-based metric (MRR, NDCG) needs on top of the retrieved set.

        Returns:
            A `Dataset` of ``query_key``, ``corpus_key``, `distance_column` (and `rank_column`
            when set) — up to `k` rows per query, nearest first within each query.

        Raises:
            PlanError: if `k` is less than 1 or `metric` is unknown.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> corpus = bt.from_pydict(
                ...     {"cid": [1, 2, 3], "emb": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]}
                ... )
                >>> queries = bt.from_pydict({"qid": [10], "qv": [[1.0, 0.05]]})
                >>> hits = corpus.ml.batched_nearest_neighbors(
                ...     queries, query_key="qid", query_column="qv",
                ...     corpus_key="cid", column="emb", k=1,
                ... )
                >>> hits.to_pydict()["cid"]
                [1]
        """
        from batcher._internal.errors import PlanError
        from batcher.plan.expr_ir import col, lit

        if k < 1:
            raise PlanError(f"batched_nearest_neighbors k must be >= 1, got {k}")
        _require_vector_column(self._ds, column, method="batched_nearest_neighbors")
        _require_vector_column(
            queries, query_column, method="batched_nearest_neighbors", param="query_column"
        )
        vec, qvec = col(column), col("_bnn_qvec")
        # All but dot are distances (smaller = nearer, ascending rank); dot is a similarity.
        distances = {
            "cosine": (vec.list.cosine_distance(qvec), False),
            "l2": (vec.list.l2_distance(qvec), False),
            "l1": (vec.list.l1_distance(qvec), False),
            "hamming": (vec.list.hamming_distance(qvec), False),
            "dot": (vec.list.dot(qvec), True),
        }
        if metric not in distances:
            raise PlanError(f"metric must be one of {sorted(distances)}, got {metric!r}")
        score, descending = distances[metric]
        # Rename the query vector to a private name so the cross join can't collide with the
        # corpus embedding column even when both are called "embedding".
        prepared = queries.select(**{query_key: col(query_key), "_bnn_qvec": col(query_column)})
        joined = self._ds.join(prepared, how="cross").with_columns(**{distance_column: score})
        ranked = joined.window(
            partition_by=[query_key],
            order_by=[(distance_column, descending)],
            functions={"_bnn_rank": "row_number"},
        )
        kept = ranked.filter(col("_bnn_rank") <= lit(k))
        if rank_column is not None:
            kept = kept.with_columns(**{rank_column: col("_bnn_rank")})
            return kept.select(query_key, corpus_key, distance_column, rank_column)
        return kept.select(query_key, corpus_key, distance_column)

    def recall_at_k(self, relevant: Dataset, *, query_key: str, corpus_key: str) -> float:
        """Mean recall of these retrieved neighbors against ground-truth `relevant` pairs.

        The headline retrieval-quality number, and the natural follow-on to
        `batched_nearest_neighbors`: of the documents that *should* have been retrieved for
        each query, what fraction actually appear in the retrieved set. Averaged over
        queries. Since the retrieved set is already the top `k` neighbors, this is recall@k.

        This dataset supplies the retrieved ``(query_key, corpus_key)`` pairs (any extra
        columns, such as a distance, are ignored); `relevant` supplies the ground-truth
        pairs. A query with no relevant documents is not counted. Composes a join and two
        grouped counts — no per-row Python — then collapses to one number.

        Args:
            relevant: A `Dataset` of ground-truth ``(query_key, corpus_key)`` relevant pairs.
            query_key: The column identifying a query, on both datasets.
            corpus_key: The column identifying a retrieved/relevant document, on both.

        Returns:
            The mean per-query recall in ``[0, 1]`` (``0.0`` when there are no relevant pairs).

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> retrieved = bt.from_pydict({"qid": [1, 1, 2, 2], "cid": [10, 11, 20, 21]})
                >>> relevant = bt.from_pydict({"qid": [1, 2, 2], "cid": [10, 20, 22]})
                >>> round(retrieved.ml.recall_at_k(relevant, query_key="qid", corpus_key="cid"), 3)
                0.75
        """
        from batcher.plan.expr_ir import coalesce, col, lit

        rel = relevant.group_by(query_key).agg(_recall_nrel=col(corpus_key).count())
        hits = (
            self._ds.join(relevant, on=[query_key, corpus_key], how="inner")
            .group_by(query_key)
            .agg(_recall_nhit=col(corpus_key).count())
        )
        per_query = rel.join(hits, on=query_key, how="left").with_columns(
            _recall=coalesce(col("_recall_nhit"), lit(0)) / col("_recall_nrel")
        )
        collected = per_query.agg(recall=col("_recall").mean()).to_pydict()["recall"]
        return float(collected[0]) if collected and collected[0] is not None else 0.0

    def mrr(
        self, relevant: Dataset, *, query_key: str, corpus_key: str, rank_column: str = "rank"
    ) -> float:
        """Mean Reciprocal Rank of these ranked neighbors against `relevant` ground truth.

        The other headline retrieval number: how high the *first* relevant document sits in
        each query's ranking, averaged over queries as ``1 / rank``. A first-relevant at
        rank 1 scores 1, at rank 2 scores 0.5; a query with no relevant document in the
        ranking scores 0. It rewards getting *a* good answer to the top, where recall
        rewards getting *all* of them back.

        This dataset must carry a per-query `rank_column` — pass ``rank_column=`` to
        `batched_nearest_neighbors` to produce it. Composes a join, a grouped min, and a
        mean — no per-row Python.

        Args:
            relevant: A `Dataset` of ground-truth ``(query_key, corpus_key)`` relevant pairs.
            query_key: The column identifying a query, on both datasets.
            corpus_key: The column identifying a document, on both datasets.
            rank_column: The 1-based per-query rank column on this dataset.

        Returns:
            The mean reciprocal rank in ``[0, 1]`` (``0.0`` when nothing matched).

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ranked = bt.from_pydict(
                ...     {"qid": [1, 1, 2, 2], "cid": [10, 11, 20, 21], "rank": [1, 2, 1, 2]}
                ... )
                >>> relevant = bt.from_pydict({"qid": [1, 2], "cid": [10, 21]})
                >>> ranked.ml.mrr(relevant, query_key="qid", corpus_key="cid")
                0.75
        """
        from batcher.plan.expr_ir import coalesce, col, lit

        queries = self._ds.select(query_key).distinct()
        matched = (
            self._ds.join(relevant, on=[query_key, corpus_key], how="inner")
            .group_by(query_key)
            .agg(_mrr_rank=col(rank_column).min())
        )
        per_query = queries.join(matched, on=query_key, how="left").with_columns(
            _mrr_rr=coalesce(lit(1.0) / col("_mrr_rank"), lit(0.0))
        )
        collected = per_query.agg(mrr=col("_mrr_rr").mean()).to_pydict()["mrr"]
        return float(collected[0]) if collected and collected[0] is not None else 0.0

    def reciprocal_rank_fusion(
        self,
        *others: Dataset,
        key: str,
        score: str,
        k: int = 60,
        output_column: str = "rrf_score",
    ) -> Dataset:
        """Fuse this ranked result with `others` by Reciprocal Rank Fusion — hybrid search.

        The standard way to combine a dense (embedding) ranking with a lexical (BM25) one,
        or any set of ranked result lists, without having to calibrate their scores onto a
        common scale. Each list contributes ``1 / (k + rank)`` per key, and the fused score
        is the sum — so a row ranked highly by *either* retriever floats up, and one ranked
        highly by *both* wins. `k` damps the tail (60 is the common default). Each input is
        ranked by its own `score` column independently, so the two need no shared units.

        Every input must carry the same `key` and `score` columns. A key present in only one
        list still contributes its single term. Composes a window rank, a union, and a
        grouped sum — no per-row Python — so it runs wherever those run, including distributed.

        Args:
            others: The other ranked `Dataset`s to fuse with this one.
            key: The column identifying a result row across the lists (a doc id).
            score: The per-list score column each list is ranked by (higher is better).
            k: The RRF damping constant; must be positive.
            output_column: Name of the appended fused-score column.

        Returns:
            A `Dataset` of ``key`` and `output_column`, ordered by descending fused score.

        Raises:
            PlanError: if `k` is not positive.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> dense = bt.from_pydict({"id": [1, 2, 3], "score": [0.9, 0.5, 0.1]})
                >>> lexical = bt.from_pydict({"id": [2, 3, 4], "score": [0.8, 0.7, 0.6]})
                >>> fused = dense.ml.reciprocal_rank_fusion(lexical, key="id", score="score")
                >>> fused.to_pydict()["id"]
                [2, 3, 1, 4]
        """
        from batcher._internal.errors import PlanError
        from batcher.plan.expr_ir import col, lit

        if k <= 0:
            raise PlanError(f"reciprocal_rank_fusion k must be positive, got {k}")
        contributions = []
        for result in (self._ds, *others):
            ranked = result.window(order_by=[(score, True)], functions={"_rrf_rank": "row_number"})
            term = lit(1.0) / (lit(float(k)) + col("_rrf_rank"))
            contributions.append(ranked.select(**{key: col(key), "_rrf_term": term}))
        combined = contributions[0]
        for contribution in contributions[1:]:
            combined = combined.union(contribution)
        fused = combined.group_by(key).agg(**{output_column: col("_rrf_term").sum()})
        return fused.sort(output_column, descending=True)

    def stream_loader(
        self,
        *,
        batch_size: int = 256,
        world_size: int = 1,
        rank: int = 0,
        epoch: int = 0,
        seed: int = 0,
        shuffle: bool = True,
        drop_last: bool = True,
        columns: list[str] | None = None,
        global_consumed: int = 0,
        collate_fn: object = None,
        shuffle_block_size: int | None = None,
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
            batch_size: Rows per yielded ``{column: tensor}`` batch (256 by default).
            world_size: Total number of training ranks.
            rank: This rank's index in ``[0, world_size)``.
            epoch: Epoch number, folded into the shuffle so passes differ.
            seed: Seed for the global order; the same seed reproduces it.
            shuffle: Shuffle the global order before sharding.
            drop_last: Drop the final partial global batch so ranks stay balanced.
            columns: The columns to yield as tensors; defaults to all.
            global_consumed: Samples already consumed, to resume from a checkpoint.
            collate_fn: Custom collation over each rank batch's `pyarrow.Table`. Also the
                escape hatch for columns that do not tensorize (string labels or ids),
                which are otherwise dropped from the yielded batch.
            shuffle_block_size: Shuffle within blocks of this many samples rather than
                across the whole corpus, matching the order `shard_stream_loader` gives the
                same corpus written with that shard size.

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
        _require_columns(self._ds, columns, param="columns")
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
            collate_fn=collate_fn,
            shuffle_block_size=shuffle_block_size,
        )

    def write_shards(
        self,
        directory: str,
        *,
        rows_per_shard: int = 65_536,
        batch_size: int | None = None,
        distributed: bool | str = False,
        resume: bool = False,
        write_concurrency: int | None = None,
    ):
        """Write this dataset as a training-shard corpus `shard_stream_loader` can stream.

        The on-ramp to the larger-than-memory training path (the MosaicML ``MDSWriter`` /
        WebDataset ``ShardWriter`` role). `stream_loader` keeps one rank's whole shard of the
        corpus resident, which stops scaling once a rank's share exceeds RAM; a shard corpus
        removes that ceiling, and this is how one gets written. The rows stream out of the
        engine and into fixed-size Arrow-IPC shards plus an ``index.json``, so writing a
        corpus larger than memory needs no more memory than one shard.

        Shards are plain Arrow IPC files, so the same corpus reads back relationally with
        ``bt.read.arrow(f"{directory}/*.arrow")`` — this is a layout, not a private format.

        Args:
            directory: Where to write the shards and their index.
            rows_per_shard: Rows in every shard but the last. Larger shards mean fewer, more
                sequential reads and a coarser default shuffle block; smaller ones let a
                bounded shard cache hold more of the corpus.
            batch_size: Rows per batch read out of the engine (engine default when None).
                Independent of `rows_per_shard`, which the writer repacks to exactly.
            distributed: Produce the rows on a Ray cluster rather than in this process.
                The *writing* stays here and stays sequential, because shard boundaries have
                to be deterministic for a global row index to mean anything — but the read,
                the decode and every transform ahead of it fan out.
            resume: Continue a previous, interrupted write rather than starting over. The
                manifest is republished as the write proceeds, so a corpus left behind by a
                crash is both readable and resumable, and the rows already written are
                skipped from this dataset rather than re-encoded. Requires the dataset to
                produce the same rows in the same order, which a shuffled global order
                already assumes of it.
            write_concurrency: Shards to publish at once. Defaults to a size derived from
                the destination and the measured shard size — cores for local disk, requests
                in flight for an object store, where publishing one shard at a time makes a
                large corpus spend most of its time waiting on round trips.

        Returns:
            The `batcher.io.formats.ml.ShardIndex` describing what was written.

        Examples:
            .. doctest::

                >>> import batcher as bt, os, tempfile
                >>> ds = bt.from_pydict({"x": list(range(10)), "y": [1.0] * 10})
                >>> out = os.path.join(tempfile.mkdtemp(), "corpus")
                >>> index = ds.ml.write_shards(out, rows_per_shard=4)
                >>> index.total_rows, len(index.shard_paths)
                (10, 3)
        """
        from batcher.io.formats.ml.shards import write_shards

        return write_shards(
            self._ds.iter_batches(batch_size, distributed=distributed),
            directory,
            rows_per_shard=rows_per_shard,
            resume=resume,
            write_concurrency=write_concurrency,
        )

    def iter_torch_batches(
        self,
        *,
        batch_size: int | None = None,
        columns: list[str] | None = None,
        device: object = "auto",
        dtypes: dict[str, str] | str | None = None,
        collate_fn: object = None,
        prefetch_batches: int = 2,
        pin_memory: bool = False,
        zero_copy: bool = False,
        local_shuffle_buffer_size: int | None = None,
        seed: int = 0,
        epoch: int = 0,
        drop_last: bool = False,
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
            dtypes: Cast the tensors to a torch dtype — one name for all columns, or a
                ``{column: dtype}`` mapping. Abbreviations (``"fp16"``) are accepted.
            collate_fn: Custom collation applied to each batch before yielding.
            prefetch_batches: Batches to prepare ahead in the background. The default of
                2 overlaps the host-to-device copy with compute; past 4 the prefetched
                batches cost more memory than the overlap is worth.
            pin_memory: Pin host buffers for faster host-to-device copies.
            zero_copy: Yield read-only DLPack views instead of copying.
            local_shuffle_buffer_size: Window size for local shuffling; None disables.
            seed: Seed for the local shuffle.
            epoch: Epoch number, folded into the shuffle seed so successive passes over
                the same dataset see different orders. Without it every epoch replays
                one order, which silently degrades convergence.
            drop_last: Drop a final short batch so every step sees `batch_size` rows.

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
        _require_columns(self._ds, columns, param="columns")
        from batcher.ml.loader import iter_torch_batches

        return iter_torch_batches(
            self._ds,
            batch_size=batch_size,
            columns=columns,
            device=device,
            dtypes=dtypes,
            collate_fn=collate_fn,
            prefetch_batches=prefetch_batches,
            pin_memory=pin_memory,
            zero_copy=zero_copy,
            local_shuffle_buffer_size=local_shuffle_buffer_size,
            seed=seed,
            epoch=epoch,
            drop_last=drop_last,
        )

    def to_torch_dataloader(
        self,
        *,
        batch_size: int | None = None,
        columns: list[str] | None = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        shuffle: bool = False,
        collate_fn: object = None,
        local_shuffle_buffer_size: int | None = None,
        **dataloader_kwargs: object,
    ):
        """Return a ready-to-iterate ``torch.utils.data.DataLoader`` over this dataset.

        Batching happens in the engine (one Arrow batch is one training batch), so the
        loader is built with ``batch_size=None`` and streams ready ``{column: tensor}``
        dicts. `num_workers`, `pin_memory`, and further ``DataLoader`` keywords pass
        straight through; `shuffle` sets a `local_shuffle_buffer_size` window when one is
        not given (a streaming approximation of a full shuffle). Requires `torch`.

        Args:
            batch_size: Rows per training batch (engine default when None).
            columns: The columns to yield as tensors; defaults to all.
            num_workers: PyTorch DataLoader worker processes (0 loads in-process).
            pin_memory: Page-lock host buffers for faster host-to-device copies.
            drop_last: Drop a final short batch so every step sees `batch_size` rows.
            shuffle: Locally shuffle rows before batching when no explicit
                `local_shuffle_buffer_size` is given.
            collate_fn: Custom collation over each ``{column: ndarray}`` batch.
            local_shuffle_buffer_size: Explicit local-shuffle window; overrides `shuffle`.
            **dataloader_kwargs: Further ``DataLoader`` keyword arguments.

        Returns:
            A ``torch.utils.data.DataLoader`` yielding ``{column: torch.Tensor}`` batches.

        Examples:
            .. doctest::

                >>> import batcher as bt  # doctest: +SKIP
                >>> loader = ds.ml.to_torch_dataloader(  # doctest: +SKIP
                ...     batch_size=256, num_workers=4, pin_memory=True
                ... )
                >>> for batch in loader:  # doctest: +SKIP
                ...     train_step(batch)
        """
        _require_columns(self._ds, columns, param="columns")
        from torch.utils.data import DataLoader, IterableDataset

        from batcher.ml.converters import _worker_stride

        window = local_shuffle_buffer_size
        _warn_if_training_on_sorted_data(self._ds._plan, shuffle, window)
        if window is None and shuffle:
            window = 8192
        accessor = self

        def _make_stream() -> object:
            return accessor.iter_torch_batches(
                batch_size=batch_size,
                columns=columns,
                device="cpu",
                collate_fn=collate_fn,
                drop_last=drop_last,
                local_shuffle_buffer_size=window,
            )

        class _StreamDataset(IterableDataset):  # type: ignore[misc]
            # Re-iterable (one fresh engine stream per epoch) and worker-strided, so
            # DataLoader(num_workers=k) partitions the batches instead of replicating the
            # whole stream into every worker — the classic IterableDataset duplication bug.
            def __iter__(self) -> object:
                offset, stride = _worker_stride()
                for i, batch in enumerate(_make_stream()):
                    if i % stride == offset:
                        yield batch

        return DataLoader(
            _StreamDataset(),
            batch_size=None,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **dataloader_kwargs,
        )

    def to_tf(
        self,
        *,
        batch_size: int | None = None,
        columns: list[str] | None = None,
        dtypes: dict[str, str] | str | None = None,
        local_shuffle_buffer_size: int | None = None,
        seed: int = 0,
        epoch: int = 0,
        drop_last: bool = False,
    ):
        """Stream this dataset to a ``tf.data.Dataset`` of ``{column: tensor}`` batches.

        The TensorFlow counterpart of `to_torch`, built over the public batch iterator so
        nothing materializes. It takes the same shuffle, batch-width and dtype options,
        because it prepares its stream with the same code the PyTorch loader does — this
        was the one loader in the package with none of them. Non-numeric columns are
        dropped. Requires `tensorflow`.

        Args:
            batch_size: Rows per yielded batch (engine default when None).
            columns: The columns to keep; defaults to all numeric columns.
            dtypes: Cast the yielded tensors — one dtype name for every column, or a
                ``{column: dtype}`` mapping. Abbreviations (``"fp16"``) are accepted.
            local_shuffle_buffer_size: Shuffle within blocks of this many rows before
                batching, a streaming approximation of a global shuffle. None disables it.
            seed: Seed for that shuffle, combined with `epoch`.
            epoch: Reshuffles the local-shuffle order so successive passes over the same
                dataset differ. Without it every epoch replays one order.
            drop_last: Drop a final batch narrower than `batch_size`, so a ragged tail never
                reaches a fixed-shape graph. Requires `batch_size`.

        Returns:
            A ``tf.data.Dataset`` yielding one ``{column: tensor}`` dict per batch.

        Examples:
            .. doctest::

                >>> import batcher as bt  # doctest: +SKIP
                >>> tf_ds = ds.ml.to_tf(  # doctest: +SKIP
                ...     batch_size=256, local_shuffle_buffer_size=8192
                ... )
                >>> model.fit(tf_ds)  # doctest: +SKIP
        """
        _require_columns(self._ds, columns, param="columns")
        from batcher.ml.converters import tf_dataset_from_arrays
        from batcher.ml.loader.lazy import cast_arrays, numpy_batch_stream

        stream = numpy_batch_stream(
            self._ds,
            batch_size=batch_size,
            columns=columns,
            local_shuffle_buffer_size=local_shuffle_buffer_size,
            seed=seed,
            epoch=epoch,
            drop_last=drop_last,
        )
        return tf_dataset_from_arrays(cast_arrays(stream, dtypes))

    def to_numpy_batches(
        self,
        *,
        batch_size: int | None = None,
        columns: list[str] | None = None,
    ):
        """Stream this dataset as ``{column: np.ndarray}`` dicts, one per batch (lazy).

        Numeric non-null columns convert zero-copy. The framework-agnostic loader for a
        custom training loop or a NumPy pipeline.

        Args:
            batch_size: Rows per yielded batch (engine default when None).
            columns: The columns to keep; defaults to all.

        Yields:
            One ``{column: np.ndarray}`` dict per batch.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> next(ds.ml.to_numpy_batches(batch_size=2))
                {'x': array([1, 2])}
        """
        _require_columns(self._ds, columns, param="columns")
        from batcher.ml.converters import to_numpy_batches

        return to_numpy_batches(self._ds.iter_batches(batch_size), columns=columns)

    def generate(
        self,
        engine: Callable,
        *,
        prompt_column: str,
        output_column: str = "response",
        template: str | None = None,
        image_column: str | None = None,
        adapter_column: str | None = None,
        max_tokens_column: str | None = None,
        temperature_column: str | None = None,
        few_shot: list[tuple[str, str]] | None = None,
        parse_json: bool = False,
        usage: bool = False,
        finish_reason: bool = False,
        logprobs: bool = False,
        dedup: bool = False,
        skip_null_prompts: bool = False,
        batch_size: int | None = None,
        num_gpus: float = 0.0,
        concurrency: int | tuple[int, int] | None = None,
        accelerator_type: str | None = None,
        model_memory_gb: float = 0.0,
        max_errored_rows: int = 0,
        timeout: float = 0.0,
        max_retries: int = 0,
        retry_backoff: float = 0.5,
        retry_on: type[BaseException] | tuple[type[BaseException], ...] | None = None,
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
            max_tokens_column: A column giving each row its own token budget, so one pass
                can mix a 16-token classification with a 2000-token summary. A null uses
                the engine's default.
            temperature_column: A column giving each row its own sampling temperature (a
                null uses the engine's default), so a factual extraction and a creative
                rewrite share one pass.
            few_shot: Fixed ``(input, output)`` demonstration pairs prepended to every
                prompt, so the task format is shown once rather than baked into a template.
            parse_json: Parse each output as JSON into a struct column (null on a parse
                error). Pair with ``vllm_engine(guided_json=...)`` for reliable output.
            usage: Also append ``prompt_tokens`` / ``completion_tokens`` columns.
            finish_reason: Also append a ``finish_reason`` column, so a generation cut off
                at ``max_tokens`` (``"length"``) is detectable rather than silently
                corrupting a downstream `parse_json`.
            logprobs: Also append a ``logprob`` column holding each generation's cumulative
                log-probability — the model's own confidence, for routing the least certain
                rows to review. Null for an engine that does not report one.
            dedup: Send each distinct prompt to the engine once and copy its result to every
                row that repeats it — a throughput win for deterministic decoding over a
                corpus with duplicate prompts. Leave off when sampling and an independent
                draw per row is wanted.
            skip_null_prompts: Leave a row whose `prompt_column` is null out of the request
                and give it a null output, instead of sending it as ``""``. Off by default;
                turn it on to stop spending a decode slot (GPU engine) or a billed request
                (hosted engine) on a row that has no prompt. Ignored when `template` or
                `image_column` is set.
            batch_size: Rebatch before each engine call; leave unset for the engine's own.
            num_gpus: GPUs to reserve per worker.
            concurrency: Size of the distributed actor pool.
            accelerator_type: Pin GPU actors to a model (e.g. ``"NVIDIA_A100"``).
            model_memory_gb: The model's footprint, for memory budgeting.
            max_errored_rows: Rows a raising engine may drop per worker before failing.
            timeout: Wall-clock ceiling (seconds) for one engine call; 0 = no timeout.
            max_retries: Times to retry a batch whose engine call raises a retryable error.
            retry_backoff: Base backoff (seconds); attempt `k` waits `retry_backoff * 2**k`.
            retry_on: Exception type(s) worth retrying; ``None`` retries any `Exception`.

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
        _require_llm_columns(
            self._ds,
            method="generate",
            prompt_column=prompt_column,
            template=template,
            image_column=image_column,
        )
        from batcher.ml.llm import llm_udf

        udf = llm_udf(
            engine,
            prompt_column=prompt_column,
            output_column=output_column,
            template=template,
            image_column=image_column,
            adapter_column=adapter_column,
            max_tokens_column=max_tokens_column,
            temperature_column=temperature_column,
            few_shot=few_shot,
            parse_json=parse_json,
            usage=usage,
            finish_reason=finish_reason,
            logprobs=logprobs,
            dedup=dedup,
            skip_null_prompts=skip_null_prompts,
        )
        # Order must match GenerateSpec.appended_columns: output, usage, finish_reason, logprob.
        appended = [
            output_column,
            *(["prompt_tokens", "completion_tokens"] if usage else []),
            *(["finish_reason"] if finish_reason else []),
            *(["logprob"] if logprobs else []),
        ]
        new = [c for c in appended if c not in self._ds.columns]
        return self._ds.map_batches(
            udf,
            output_columns=[*self._ds.columns, *new] if new else None,
            batch_size=batch_size,
            num_gpus=num_gpus,
            concurrency=concurrency,
            accelerator_type=accelerator_type,
            model_memory_gb=model_memory_gb,
            max_errored_rows=max_errored_rows,
            timeout=timeout,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            retry_on=retry_on,
        )

    def extract(
        self,
        engine: Callable,
        *,
        schema: dict[str, str],
        prompt_column: str | None = None,
        template: str | None = None,
        instruct: bool = True,
        image_column: str | None = None,
        batch_size: int | None = None,
        num_gpus: float = 0.0,
        concurrency: int | tuple[int, int] | None = None,
        accelerator_type: str | None = None,
        model_memory_gb: float = 0.0,
        max_errored_rows: int = 0,
        timeout: float = 0.0,
        max_retries: int = 0,
        retry_backoff: float = 0.5,
        retry_on: type[BaseException] | tuple[type[BaseException], ...] | None = None,
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
            image_column: An image column (bytes or an ``(H, W, 3)`` tensor) for a vision
                model, so fields are extracted from an image (an invoice photo →
                ``{vendor, total}``). The engine must be vision-capable.
            batch_size: Rebatch before each engine call; leave unset for the engine's own.
            num_gpus: GPUs to reserve per worker.
            concurrency: Size of the distributed actor pool.
            accelerator_type: Pin GPU actors to a model (e.g. ``"NVIDIA_A100"``).
            model_memory_gb: The model's footprint, for memory budgeting.
            max_errored_rows: Rows a raising engine may drop per worker before failing.
            timeout: Wall-clock ceiling (seconds) for one engine call; 0 = no timeout.
            max_retries: Times to retry a batch whose engine raises a retryable error.
            retry_backoff: Base backoff (seconds); attempt `k` waits `retry_backoff * 2**k`.
            retry_on: Exception type(s) worth retrying; ``None`` retries any `Exception`.

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
        _require_llm_columns(
            self._ds,
            method="extract",
            prompt_column=prompt_column,
            template=template,
            image_column=image_column,
        )
        from batcher.ml.llm import llm_extract_udf

        udf = llm_extract_udf(
            engine,
            schema=schema,
            prompt_column=prompt_column,
            template=template,
            instruct=instruct,
            image_column=image_column,
        )
        new = [c for c in schema if c not in self._ds.columns]
        _warn_extract_overwrites(self._ds, schema, prompt_column)
        return self._ds.map_batches(
            udf,
            output_columns=[*self._ds.columns, *new] if new else None,
            batch_size=batch_size,
            num_gpus=num_gpus,
            concurrency=concurrency,
            accelerator_type=accelerator_type,
            model_memory_gb=model_memory_gb,
            max_errored_rows=max_errored_rows,
            timeout=timeout,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            retry_on=retry_on,
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
        image_column: str | None = None,
        batch_size: int | None = None,
        num_gpus: float = 0.0,
        concurrency: int | tuple[int, int] | None = None,
        accelerator_type: str | None = None,
        model_memory_gb: float = 0.0,
        max_errored_rows: int = 0,
        timeout: float = 0.0,
        max_retries: int = 0,
        retry_backoff: float = 0.5,
        retry_on: type[BaseException] | tuple[type[BaseException], ...] | None = None,
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
            image_column: An image column for a vision model, so a row is classified from an
                image rather than text. The engine must be vision-capable.
            batch_size: Rebatch before each engine call; leave unset for the engine's own.
            num_gpus: GPUs to reserve per worker.
            concurrency: Size of the distributed actor pool.
            accelerator_type: Pin GPU actors to a model (e.g. ``"NVIDIA_A100"``).
            model_memory_gb: The model's footprint, for memory budgeting.
            max_errored_rows: Rows a raising engine may drop per worker before failing.
            timeout: Wall-clock ceiling (seconds) for one engine call; 0 = no timeout.
            max_retries: Times to retry a batch whose engine raises a retryable error.
            retry_backoff: Base backoff (seconds); attempt `k` waits `retry_backoff * 2**k`.
            retry_on: Exception type(s) worth retrying; ``None`` retries any `Exception`.

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
        _require_llm_columns(
            self._ds,
            method="classify",
            prompt_column=prompt_column,
            template=template,
            image_column=image_column,
        )
        from batcher.ml.llm import llm_classify_udf

        udf = llm_classify_udf(
            engine,
            labels=labels,
            prompt_column=prompt_column,
            output_column=output_column,
            template=template,
            instruct=instruct,
            image_column=image_column,
        )
        new = [] if output_column in self._ds.columns else [output_column]
        return self._ds.map_batches(
            udf,
            output_columns=[*self._ds.columns, *new] if new else None,
            batch_size=batch_size,
            num_gpus=num_gpus,
            concurrency=concurrency,
            accelerator_type=accelerator_type,
            model_memory_gb=model_memory_gb,
            max_errored_rows=max_errored_rows,
            timeout=timeout,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            retry_on=retry_on,
        )

    def embed(
        self,
        model: str | Callable | type,
        *,
        column: str | None = None,
        output_column: str = "embedding",
        output_columns: list[str] | None = None,
        batch_size: int | None = None,
        device: str | None = None,
        num_workers: int | str = "auto",
        num_gpus: float = 0.0,
        concurrency: int | tuple[int, int] | None = None,
        batch_format: str = "pyarrow",
        accelerator_type: str | None = None,
        model_memory_gb: float = 0.0,
        normalize: bool = False,
        fp16: bool = False,
        output_type: str = "tensor",
        max_errored_rows: int = 0,
        timeout: float = 0.0,
        max_retries: int = 0,
        retry_backoff: float = 0.5,
        retry_on: type[BaseException] | tuple[type[BaseException], ...] | None = None,
    ) -> Dataset:
        """Compute embeddings over the dataset — `infer` shaped for embedding models.

        Pass a **model identifier** (a sentence-transformers model id) and the text
        `column` to embed: the model loads once per worker and the vector is appended
        as a tensor `output_column`. The provider-pluggable, distributed, GPU-aware
        text-embedding path (cf. Daft's ``embed_text``). Needs ``sentence-transformers``
        (``batcher-engine[st]``).

        Pass a **callable or class** instead for any other embedding model (text or
        image → vector); the call then mirrors `map_batches`, with `output_columns`
        declaring the result schema. The options that exist only to configure the
        identifier path — `column`, `output_column`, `device`, `normalize`, `fp16`,
        `output_type` — are **rejected** in that shape rather than ignored, because the
        callable is where that behavior now lives.

        `num_gpus`/`concurrency`/`accelerator_type`/`model_memory_gb` place and size
        the model on GPU actors, the same scheduling as `infer`.

        Args:
            model: A sentence-transformers model id, or a callable/class → vector.
            column: The text column to embed (required for a model id).
            output_column: Name of the appended embedding column (model-id path).
            output_columns: The result schema when a callable/class is passed.
            batch_size: Rebatch to this many rows before each call.
            device: Where the encoder runs (model-id path): ``"auto"``/``None`` detect the
                accelerator, or force ``"cuda"``/``"cpu"``/``"mps"``.
            num_workers: Concurrent per-batch calls within a worker (``"auto"`` sizes to
                the stage), or an explicit int.
            num_gpus: GPUs to reserve per distributed worker.
            concurrency: Size of the distributed actor pool; an int or ``(min, max)``.
            batch_format: What a callable `model` sees (``"pyarrow"`` by default).
            accelerator_type: Pin GPU actors to a model (e.g. ``"NVIDIA_A100"``).
            model_memory_gb: The model's footprint, for memory budgeting.
            normalize: L2-normalize each vector in the producing pass (model-id path).
                Cosine search needs normalized vectors, and doing it here avoids a second
                full scan over the embedding column.
            fp16: Run the encoder in half precision on GPU; ignored on CPU.
            output_type: ``"tensor"`` (default) or ``"fixed_size_list"``, which is what
                Lance ANN indexing expects.
            max_errored_rows: Rows a raising encoder may drop per worker before failing.
            timeout: Wall-clock ceiling (seconds) for one encoder call; 0 = no timeout.
            max_retries: Times to retry a batch whose encoder raises a retryable error.
            retry_backoff: Base backoff (seconds); attempt `k` waits `retry_backoff * 2**k`.
            retry_on: Exception type(s) worth retrying; ``None`` retries any `Exception`.

        Returns:
            A new lazy `Dataset` with the embedding column(s) appended.

        Raises:
            PlanError: if a model id is given without `column`.
            ColumnNotFoundError: if `column` is not in the dataset (model-id path).

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
            _require_column(self._ds, column, param="column")
            from batcher.ml.embed import sentence_transformer_encoder

            encoder = sentence_transformer_encoder(
                model,
                column,
                output_column=output_column,
                device=device,
                batch_size=batch_size,
                normalize=normalize,
                fp16=fp16,
                output_type=output_type,
            )
            cols = (
                [*self._ds.columns, output_column]
                if output_column not in self._ds.columns
                else None
            )
            return self._ds.map_batches(
                encoder,
                output_columns=cols,
                batch_size=batch_size,
                num_workers=num_workers,
                num_gpus=num_gpus,
                concurrency=concurrency,
                accelerator_type=accelerator_type,
                model_memory_gb=model_memory_gb,
                max_errored_rows=max_errored_rows,
                timeout=timeout,
                max_retries=max_retries,
                retry_backoff=retry_backoff,
                retry_on=retry_on,
            )
        _reject_model_id_only(
            "embed",
            model,
            {
                "column": (column, None),
                "output_column": (output_column, "embedding"),
                "device": (device, None),
                "normalize": (normalize, False),
                "fp16": (fp16, False),
                "output_type": (output_type, "tensor"),
            },
        )
        return self._ds.map_batches(
            model,
            batch_size=batch_size,
            output_columns=output_columns,
            num_workers=num_workers,
            num_gpus=num_gpus,
            concurrency=concurrency,
            batch_format=batch_format,
            accelerator_type=accelerator_type,
            model_memory_gb=model_memory_gb,
            max_errored_rows=max_errored_rows,
            timeout=timeout,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            retry_on=retry_on,
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
        _require_column(self._ds, url_column, param="url_column")
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
                >>> thumbs = ds.map_batches(make_thumbnails)  # doctest: +SKIP
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
