"""`embed` — the headline ML workload: turn a text column into an embedding column.

A thin, convenient wrapper over [`InferencePool`][batcher.ml.InferencePool]: the
embedding model loads once per worker (not once per batch) and runs over whole
batches. The encoder is injected as a factory, so this works with
sentence-transformers, a custom model, or a test double — the engine never depends
on a specific ML library.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import TYPE_CHECKING

from batcher.ml.inference import InferencePool

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = ["EncoderFactory", "embed"]

# Encodes a list of strings into a sequence of equal-length numeric vectors.
Encoder = Callable[[list[str]], Sequence[Sequence[float]]]
EncoderFactory = Callable[[], Encoder]


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

    Args:
        batches: an iterable of `pyarrow.RecordBatch`.
        encoder_factory: zero-arg callable returning an encoder
            (`list[str] -> sequence of vectors`); called once per worker so the
            model loads once.
        text_column: the string column to embed.
        output_column: name of the appended `list<float64>` column.
        num_workers / target_batch_rows / **pool_kwargs: forwarded to `InferencePool`.

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
