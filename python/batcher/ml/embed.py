"""Embeddings — compute them (`embed`) and retrieve over them (`vector_search`).

`embed` turns a text column into an embedding column via [`InferencePool`]
[batcher.ml.InferencePool]: the model loads once per worker (not once per batch) and
runs over whole batches, injected as a factory so it works with sentence-transformers,
a custom model, or a test double. The retrieval half — `vector_search` /
`build_vector_index` — runs approximate-nearest-neighbor search over a Lance vector
store, completing the RAG loop (embed → write Lance → ANN search).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any

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
    model: str, text_column: str, *, output_column: str = "embedding", device: str | None = None
) -> type:
    """A load-once class UDF that embeds `text_column` with a sentence-transformers model.

    Drops into ``ds.ml.map_batches`` / ``ds.ml.embed`` (instantiate-once-per-worker), so
    text embedding runs **distributed and GPU-aware** — the provider-pluggable
    model-id form of ``ds.ml.embed`` (cf. Daft's ``embed_text``). The embedding is
    appended as a fixed-shape-tensor column. Needs ``sentence-transformers``
    (``batcher-engine[st]``).
    """

    class _STEncoder:
        def __init__(self) -> None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - optional extra
                from batcher._internal.errors import BackendError

                msg = "ds.ml.embed(<model id>) needs: pip install 'batcher-engine[st]'"
                raise BackendError(msg) from exc
            self._model = SentenceTransformer(model, device=device)

        def __call__(self, batch: Any) -> Any:
            from batcher.io.formats.ml.tensor import to_tensor_column

            texts = batch.column(text_column).to_pylist()
            vectors = self._model.encode(texts, convert_to_numpy=True)
            col = to_tensor_column(vectors)
            if output_column in batch.schema.names:
                idx = batch.schema.get_field_index(output_column)
                return batch.set_column(idx, output_column, col)
            return batch.append_column(output_column, col)

    return _STEncoder


Encoder = Callable[[list[str]], Sequence[Sequence[float]]]
"""Encodes a list of strings into a sequence of equal-length numeric vectors."""

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
    **pool_kwargs: object,
) -> Iterator[pa.RecordBatch]:
    """Append an embedding column produced from `text_column`.

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
        output_column: name of the appended `list<float64>` column.
        num_workers: pool size — encoders built, and batches embedded, in parallel.
        target_batch_rows: rows per batch handed to an encoder.
        pool_kwargs: further `InferencePool` options (e.g. ``target_latency_ms``).

    Yields:
        Each input batch with `output_column` appended, in order.
    """
    import pyarrow as pa

    def make_worker() -> Callable[[pa.RecordBatch], pa.RecordBatch]:
        encoder = encoder_factory()

        def worker(batch: pa.RecordBatch) -> pa.RecordBatch:
            texts = batch.column(text_column).to_pylist()
            vectors = encoder(texts)
            embeddings = pa.array(
                [[float(x) for x in vector] for vector in vectors],
                type=pa.list_(pa.float64()),
            )
            arrays = [batch.column(i) for i in range(batch.num_columns)] + [embeddings]
            names = [*batch.schema.names, output_column]
            return pa.RecordBatch.from_arrays(arrays, names=names)

        return worker

    pool = InferencePool(
        make_worker,
        num_workers=num_workers,
        target_batch_rows=target_batch_rows,
        **pool_kwargs,  # type: ignore[arg-type]
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


def build_vector_index(uri: str, column: str = "embedding", **index_kwargs: Any) -> None:
    """Build an ANN index on a Lance vector `column` so `vector_search` scales.

    Needs ``batcher-engine[lance]``.

    Examples:
        .. doctest::

            >>> from batcher.ml import build_vector_index  # doctest: +SKIP
            >>> build_vector_index("s3://bucket/docs.lance", "embedding")  # doctest: +SKIP

    Args:
        uri: the Lance dataset holding the vectors.
        column: the vector column to index.
        index_kwargs: passed to Lance (``index_type``, ``metric``, ``num_partitions``,
            ``num_sub_vectors``, ...).
    """
    from batcher.io.formats.structured.lance import lance_create_vector_index

    lance_create_vector_index(uri, column, **index_kwargs)
