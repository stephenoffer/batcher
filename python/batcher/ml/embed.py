"""Embeddings — compute them (`embed`) and retrieve over them (`vector_search`).

`embed` turns a text column into an embedding column via [`InferencePool`]
[batcher.ml.InferencePool]: the model loads once per worker (not once per batch) and
runs over whole batches, injected as a factory so it works with sentence-transformers,
a custom model, or a test double. The retrieval half — `vector_search` /
`build_vector_index` — runs approximate-nearest-neighbor search over a Lance vector
store, completing the RAG loop (embed → write Lance → ANN search).
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any

from batcher.ml._embed_dedup import embed_unique
from batcher.ml.inference import InferencePool

if TYPE_CHECKING:
    import pyarrow as pa

    from batcher.api.dataset import Dataset

__all__ = [
    "EncoderFactory",
    "build_vector_index",
    "embed",
    "sentence_transformer_encoder",
    "vector_search",
]


def sentence_transformer_encoder(
    model: str,
    text_column: str,
    *,
    output_column: str = "embedding",
    device: str | None = None,
    batch_size: int | None = None,
    normalize: bool = False,
    fp16: bool = False,
    output_type: str = "tensor",
) -> type:
    """A load-once class UDF that embeds `text_column` with a sentence-transformers model.

    Drops into ``ds.ml.map_batches`` / ``ds.ml.embed`` (instantiate-once-per-worker), so
    text embedding runs **distributed and GPU-aware** — the provider-pluggable
    model-id form of ``ds.ml.embed`` (cf. Daft's ``embed_text``). Needs
    ``sentence-transformers`` (``batcher-engine[st]``).

    The defaults are chosen for bulk embedding rather than for a single ``encode`` call.
    `device` defaults to the detected accelerator (`batcher.ml.gpu.torch_device`) instead
    of letting the library pick, and `batch_size` defaults to *the whole Arrow batch*
    rather than the library's internal default of 32, which otherwise caps GPU occupancy
    at 32 rows however large a batch the pipeline hands over.

    Args:
        model: the sentence-transformers model id to load.
        text_column: the string column to embed.
        output_column: the appended (or replaced) embedding column.
        device: torch device string; defaults to the detected accelerator.
        batch_size: rows per forward pass; defaults to the whole incoming Arrow batch.
        normalize: L2-normalize each vector, so a dot product is cosine similarity.
        fp16: run the model in half precision. Ignored on CPU, where fp16 is
            typically slower than fp32 rather than faster.
        output_type: ``"tensor"`` for a fixed-shape-tensor column, or
            ``"fixed_size_list"`` for the plain ``fixed_size_list<float32>`` that
            Lance ANN indexing expects.

    Returns:
        A class to instantiate once per worker, callable on a `pyarrow.RecordBatch`.
    """
    _check_output_type(output_type)

    class _STEncoder:
        def __init__(self) -> None:
            from batcher._internal.optional import require

            SentenceTransformer = require(
                "sentence_transformers",
                "SentenceTransformer",
                feature="ds.ml.embed(<model id>)",
                provides="sentence-transformers",
                extra="st",
            )
            from batcher.ml.gpu import torch_device

            self._device = device or torch_device()
            self._model = SentenceTransformer(model, device=self._device)
            if fp16 and self._device != "cpu":
                self._model.half()

        def __call__(self, batch: Any) -> Any:
            # A null cell reaches `encode` as `None`, which the tokenizer rejects — one null
            # text failed the whole batch. `""` is the same rendering the streaming `embed`
            # path and the prompt renderers use.
            texts = [_text_cell(v) for v in batch.column(text_column).to_pylist()]
            vectors = self._model.encode(
                texts,
                convert_to_numpy=True,
                batch_size=batch_size or max(len(texts), 1),
                normalize_embeddings=normalize,
            )
            col = _to_embedding_column(vectors, output_type=output_type)
            return _append_embedding_column(batch, output_column, col)

    return _STEncoder


Encoder = Callable[[list[str]], Sequence[Sequence[float]]]
"""Encodes a list of strings into a sequence of equal-length numeric vectors."""


_POOLINGS = ("none", "mean", "cls", "last")
_OUTPUT_TYPES = ("tensor", "fixed_size_list")


def _check_output_type(output_type: str) -> None:
    if output_type not in _OUTPUT_TYPES:
        from batcher._internal.errors import PlanError

        raise PlanError(f"output_type must be one of {_OUTPUT_TYPES}, got {output_type!r}")


def _to_numpy(vectors: Any) -> Any:
    """Materialize an encoder result as a NumPy array, torch tensors included.

    A torch tensor still on the accelerator, or attached to the autograd graph, cannot
    go through `numpy.asarray`; left to it, the failure reads as a confusing dtype/ndim
    complaint about the *shape* rather than about where the tensor lives.
    """
    import numpy as np

    if hasattr(vectors, "detach"):  # a torch.Tensor, possibly on CUDA and/or requiring grad
        vectors = _tensor_to_numpy(vectors)
    elif isinstance(vectors, (list, tuple)) and vectors and hasattr(vectors[0], "detach"):
        vectors = [_tensor_to_numpy(v) for v in vectors]
    return np.asarray(vectors)


def _tensor_to_numpy(tensor: Any) -> Any:
    host = tensor.detach().cpu()
    if str(host.dtype) == "torch.bfloat16":
        # NumPy has no bfloat16; widening to f32 is the only lossless landing type.
        host = host.float()
    return host.numpy()


def _pool(arr: Any, pooling: str) -> Any:
    """Reduce a ``(rows, tokens, dims)`` encoder output to ``(rows, dims)``.

    Mean pooling averages every token position. It does not consult an attention mask,
    because an encoder handed to `embed` returns vectors and not a mask, so a caller
    whose model pads within a batch should mask inside the encoder and pool there.
    """
    if arr.ndim != 3 or pooling == "none":
        return arr
    if pooling == "mean":
        return arr.mean(axis=1)
    if pooling == "cls":
        return arr[:, 0, :]
    return arr[:, -1, :]


def _l2_normalize(arr: Any) -> Any:
    """L2-normalize each row, leaving all-zero rows at zero rather than NaN."""
    import numpy as np

    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    out = np.zeros_like(arr)
    np.divide(arr, norms, out=out, where=norms > 0)
    return out


def _as_matrix(vectors: Any, pooling: str) -> Any:
    """Validate an encoder result and return it as a 2-D ``(rows, dims)`` array."""
    arr = _pool(_to_numpy(vectors), pooling)
    if arr.ndim != 2 or arr.dtype == object:
        from batcher._internal.errors import BackendError

        msg = (
            "embed() expects the encoder to return equal-length numeric vectors "
            f"(a 2-D array of shape (rows, dims)); got {arr.ndim}-D with dtype "
            f"{arr.dtype}. A 3-D (rows, tokens, dims) result is per-token output — "
            "pass pooling='mean' (or 'cls'/'last') to reduce it. A ragged result "
            "means variable-length output, which must be pooled inside the encoder."
        )
        raise BackendError(msg)
    return arr


def _to_embedding_column(
    vectors: Sequence[Sequence[float]],
    *,
    pooling: str = "none",
    normalize: bool = False,
    output_type: str = "tensor",
) -> pa.Array:
    """Convert an encoder's output into an embedding column.

    Goes through NumPy so the conversion happens once at C speed over the whole block.
    The obvious spelling — a nested comprehension calling ``float()`` per element — is
    ``O(rows x dims)`` Python-level work in the hot path (a 256-row batch of 1024-dim
    vectors is a quarter-million ``float()`` calls), which is exactly the per-tuple work
    the control plane must never do.

    The default result is a fixed-shape-tensor column (``FixedSizeList`` storage) rather
    than a ``list<float64>``: it keeps the encoder's native width (``f32`` for essentially
    every embedding model, so half the bytes) and carries its dimensionality in the type.
    ``output_type="fixed_size_list"`` drops the extension wrapper for Lance ANN indexing.
    """
    arr = _as_matrix(vectors, pooling)
    if normalize:
        arr = _l2_normalize(arr)
    return _embedding_column(arr, output_type)


def _embedding_column(arr: Any, output_type: str) -> pa.Array:
    import numpy as np
    import pyarrow as pa

    if output_type == "tensor":
        from batcher.io.formats.ml.tensor import to_tensor_column

        return to_tensor_column(arr)
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    return pa.FixedSizeListArray.from_arrays(pa.array(arr.reshape(-1)), arr.shape[1])


def _text_cell(value: Any) -> str:
    """One cell as encoder input — a null reads as ``""``, never the string ``"None"``."""
    return "" if value is None else str(value)


def _append_embedding_column(batch: Any, name: str, column: Any) -> Any:
    """`batch` with `name` set to `column`, replacing it in place when it already exists.

    The one write-back step every encoder in this package shares — the local
    sentence-transformers one, the streaming `embed` pool, and the served-endpoint encoders
    in `embed_api`, which import it from here rather than keeping a second copy.

    Replacing matters because Arrow permits duplicate field names: appending unconditionally
    left a batch carrying two columns of one name, where `to_pydict()` keeps the last,
    expressions resolve the first, and nothing raises. Re-embedding into the column you read
    is the ordinary way to hit it.
    """
    if name in batch.schema.names:
        return batch.set_column(batch.schema.get_field_index(name), name, column)
    return batch.append_column(name, column)


def _chunk_texts(texts: list[Any], size: int, overlap: int) -> tuple[list[str], list[int]]:
    """Split each text into overlapping windows, with the row each window came from."""
    step = max(size - overlap, 1)
    chunks: list[str] = []
    owners: list[int] = []
    for row, text in enumerate(texts):
        body = _text_cell(text)
        starts = range(0, max(len(body) - overlap, 1), step) if len(body) > size else (0,)
        for start in starts:
            chunks.append(body[start : start + size])
            owners.append(row)
    return chunks, owners


def _mean_by_row(arr: Any, owners: list[int], rows: int) -> Any:
    """Average the chunk vectors belonging to each row back down to one vector per row."""
    import numpy as np

    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32)
    out = np.zeros((rows, arr.shape[1]), dtype=arr.dtype)
    counts = np.zeros((rows, 1), dtype=arr.dtype)
    index = np.asarray(owners, dtype=np.intp)
    np.add.at(out, index, arr)
    np.add.at(counts, index, 1)
    return out / np.maximum(counts, 1)


def _embed_matrix(
    encoder: Encoder, texts: list[Any], *, pooling: str, chunk_size: int | None, overlap: int
) -> Any:
    """Encode a batch's texts into one 2-D ``(rows, dims)`` matrix."""
    if not chunk_size:
        return _as_matrix(encoder(texts), pooling)
    chunks, owners = _chunk_texts(texts, chunk_size, overlap)
    if not chunks:
        return _as_matrix(encoder(texts), pooling)
    return _mean_by_row(_as_matrix(encoder(chunks), pooling), owners, len(texts))


EncoderFactory = Callable[[], Encoder]
"""Builds an `Encoder`, called once per worker so the model loads a single time."""


def embed(
    batches: Iterable[pa.RecordBatch],
    encoder_factory: EncoderFactory,
    *,
    text_column: str,
    output_column: str = "embedding",
    num_workers: int = 2,
    target_batch_rows: int = 256,
    normalize: bool = False,
    pooling: str = "none",
    chunk_size: int | None = None,
    chunk_overlap: int = 0,
    output_type: str = "tensor",
    dedup: bool = False,
    **pool_kwargs: object,
) -> Iterator[pa.RecordBatch]:
    """Append an embedding column produced from `text_column`.

    `normalize` L2-normalizes in the same pass that produces the vectors, so a dot
    product is cosine similarity. Doing it here rather than through a following
    ``ds.ml.normalize_embeddings`` saves a second full scan of the embedding column,
    and it is the easiest correctness detail to forget before a cosine search.

    `pooling` reduces a per-token ``(rows, tokens, dims)`` encoder output to one vector
    per row; without it such a result is an error, because there is no single defensible
    way to collapse it. `chunk_size` handles documents longer than the model's context:
    each text is split into overlapping character windows, every window is encoded in
    the same batched call, and the windows of one row are averaged back into one vector
    (re-normalized afterwards when `normalize` is set). Chunking trades a longer encoder
    call for not silently truncating the tail of a long document.

    Examples:
        .. doctest::

            >>> import batcher as bt  # doctest: +SKIP
            >>> from batcher.ml import embed, sentence_transformer_encoder  # doctest: +SKIP
            >>> ds = bt.from_pydict({"text": ["a cat", "a dog"]})  # doctest: +SKIP
            >>> factory = lambda: SentenceTransformer("all-MiniLM-L6-v2").encode  # doctest: +SKIP
            >>> list(embed(ds.iter_batches(), factory, text_column="text"))  # doctest: +SKIP

    Args:
        batches: an iterable of `pyarrow.RecordBatch`.
        encoder_factory: zero-arg callable returning an encoder
            (`list[str] -> sequence of vectors`); called once per worker so the
            model loads once.
        text_column: the string column to embed.
        output_column: name of the appended embedding column, whose element width
            follows the encoder's own (``f32`` for most models).
        num_workers: pool size — encoders built, and batches embedded, in parallel.
        target_batch_rows: rows per batch handed to an encoder.
        normalize: L2-normalize each vector (zero vectors stay zero, not NaN).
        pooling: how to reduce a 3-D per-token result — ``"none"`` (default, an error
            on 3-D output), ``"mean"``, ``"cls"``, or ``"last"``.
        chunk_size: split each text into windows of this many characters before
            encoding, then average a row's window vectors. ``None`` disables chunking.
        chunk_overlap: characters shared between consecutive windows.
        output_type: ``"tensor"`` for a fixed-shape-tensor column, or
            ``"fixed_size_list"`` for the ``fixed_size_list<float32>`` Lance indexes.
        dedup: encode each repeated text once per batch and reuse its vector (same result).
        pool_kwargs: further `InferencePool` options (e.g. ``target_latency_ms``).

    Raises:
        PlanError: if `pooling`, `output_type`, or the chunking options are invalid.

    Yields:
        Each input batch with `output_column` appended, in order.
    """
    from batcher._internal.errors import PlanError

    if pooling not in _POOLINGS:
        raise PlanError(f"pooling must be one of {_POOLINGS}, got {pooling!r}")
    _check_output_type(output_type)
    if chunk_size is not None and chunk_size <= 0:
        raise PlanError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_size is not None and not 0 <= chunk_overlap < chunk_size:
        raise PlanError(f"chunk_overlap must be in [0, chunk_size), got {chunk_overlap}")

    def make_worker() -> Callable[[pa.RecordBatch], pa.RecordBatch]:
        encoder = encoder_factory()

        encode = functools.partial(
            _embed_matrix, encoder, pooling=pooling, chunk_size=chunk_size, overlap=chunk_overlap
        )

        def worker(batch: pa.RecordBatch) -> pa.RecordBatch:
            # A null cell reaches the encoder as `None`, which sentence-transformers and every
            # tokenizer behind it reject — so one null text failed the whole batch, on the one
            # column type most likely to have them. Rendered as "" here (what `_chunk_texts`
            # already did on the chunked path, and what the prompt renderers do), *before*
            # `embed_unique`, so nulls and empties dedupe to the one forward pass they share.
            texts = [_text_cell(v) for v in batch.column(text_column).to_pylist()]
            matrix = embed_unique(texts, encode) if dedup else encode(texts)
            if normalize:
                matrix = _l2_normalize(matrix)
            embeddings = _embedding_column(matrix, output_type)
            return _append_embedding_column(batch, output_column, embeddings)

        return worker

    pool = InferencePool(
        make_worker,
        num_workers=num_workers,
        target_batch_rows=target_batch_rows,
        # Offline bulk embedding is the throughput objective, not the latency one: there
        # is no request to be timely for, only rows/sec to maximize under a VRAM cap.
        # Left at the pool's "latency" default this ran at a *fixed* batch size, because
        # that objective only engages when a `target_latency_ms` is supplied — so no
        # controller ran at all. Overridable via `pool_kwargs` for a latency-bound caller.
        **{"objective": "throughput", **pool_kwargs},  # type: ignore[arg-type]
    )
    yield from pool.run(batches)


def vector_search(
    uri: str,
    query: Any,
    *,
    column: str = "embedding",
    k: int = 10,
    columns: list[str] | None = None,
    filter: str | None = None,
    nprobes: int | None = None,
    refine_factor: int | None = None,
) -> Dataset:
    """Approximate-nearest-neighbor search over a Lance vector store → a `Dataset`.

    Returns the `k` rows nearest to `query` (a 1-D embedding), with a ``_distance``
    column — the retrieval step for RAG / similarity lookup. Uses the column's ANN
    index when one exists (build it with `build_vector_index`), else a brute-force
    scan. Needs ``batcher-engine[lance]``.

    Examples:
        .. doctest::

            >>> from batcher.ml import vector_search  # doctest: +SKIP
            >>> hits = vector_search("s3://bucket/docs.lance", query_vector, k=5)  # doctest: +SKIP
            >>> hits.collect()  # doctest: +SKIP

    Args:
        uri: the Lance dataset to search.
        query: the query embedding (a 1-D sequence of floats).
        column: the vector column to search.
        k: how many nearest rows to return.
        columns: subset of columns to return (default: all).
        filter: optional SQL predicate applied with the search.
        nprobes: index partitions to probe — higher is more recall, more latency.
        refine_factor: re-rank ``k * refine_factor`` candidates with exact distances.

    Returns:
        A `Dataset` of the `k` nearest rows, with a ``_distance`` column appended.
    """
    import batcher as bt
    from batcher.io.formats.structured.lance import lance_vector_search

    table = lance_vector_search(
        uri,
        query,
        column=column,
        k=k,
        columns=columns,
        filter=filter,
        nprobes=nprobes,
        refine_factor=refine_factor,
    )
    return bt.from_arrow(table)


def _validate_vector_field(schema: Any, column: str) -> None:
    """Raise before index construction if `column` cannot carry a Lance ANN index.

    Lance indexes a ``fixed_size_list`` of floats. An embedding column written straight
    out of `embed` is a fixed-shape-tensor *extension* column over exactly that storage,
    which Lance does not unwrap — so the index build fails, and it fails only after the
    entire GPU embedding cost has already been paid and written. Checking the schema
    first turns that into an actionable message that names the fix.
    """
    import pyarrow as pa

    from batcher._internal.errors import BackendError

    if schema.get_field_index(column) < 0:
        names = ", ".join(map(str, schema.names))
        raise BackendError(f"build_vector_index: no column {column!r} in the dataset ({names})")
    dtype = schema.field(column).type
    if isinstance(dtype, pa.FixedShapeTensorType):
        raise BackendError(
            f"build_vector_index: column {column!r} is a fixed-shape-tensor column, but "
            "Lance ANN indexing needs fixed_size_list<float32>. Re-write it with "
            "embed(..., output_type='fixed_size_list')."
        )
    if not pa.types.is_fixed_size_list(dtype) or not pa.types.is_floating(dtype.value_type):
        raise BackendError(
            f"build_vector_index: column {column!r} is {dtype}, but Lance ANN indexing "
            "needs fixed_size_list<float32> (every row the same, fixed dimensionality)."
        )


def build_vector_index(uri: str, column: str = "embedding", **index_kwargs: Any) -> None:
    """Build an ANN index on a Lance vector `column` so `vector_search` scales.

    The column's type is checked against what Lance can index before any work starts,
    because the alternative is discovering the mismatch at the end of an index build
    over an already-embedded dataset. Needs ``batcher-engine[lance]``.

    Examples:
        .. doctest::

            >>> from batcher.ml import build_vector_index  # doctest: +SKIP
            >>> build_vector_index("s3://bucket/docs.lance", "embedding")  # doctest: +SKIP

    Args:
        uri: the Lance dataset holding the vectors.
        column: the vector column to index.
        index_kwargs: passed to Lance (``index_type``, ``metric``, ``num_partitions``,
            ``num_sub_vectors``, ...).

    Raises:
        BackendError: if `column` is missing or is not an indexable float vector column.
    """
    from batcher._internal.optional import require
    from batcher.io.formats.structured.lance import lance_create_vector_index

    lance = require("lance", feature="Lance", provides="pylance", extra="lance")
    _validate_vector_field(lance.LanceDataset(uri).schema, column)
    lance_create_vector_index(uri, column, **index_kwargs)
