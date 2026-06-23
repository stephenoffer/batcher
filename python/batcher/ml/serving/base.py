"""Serving clients + the load-once `map_batches` adapter they share.

A serving backend (Triton, TorchServe, an HTTP endpoint) is reached through a
`ServingClient`: ``predict({column: ndarray}) -> {column: ndarray}``. `serving_udf`
wraps a *connect* function (run once per worker) into a class UDF for
``ds.ml.map_batches`` — it extracts the input columns as NumPy (tensor columns keep
their shape), calls the server, and appends the outputs. Because it returns a
*class*, ``map_batches`` instantiates it once per worker (connection + warm model),
the load-once pattern; only `batches` cross the wire, never per-row Python.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy as np
    import pyarrow as pa

__all__ = ["ServingClient", "serving_udf"]


@runtime_checkable
class ServingClient(Protocol):
    """A connected inference backend: a batch of named arrays in, named arrays out."""

    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference on one batch of input arrays, returning output arrays."""
        ...


def _column_to_numpy(column: pa.Array) -> np.ndarray:
    """A batch column as NumPy — tensor columns keep their ``(N, *shape)`` form."""
    from batcher.io.formats.ml.tensor import is_tensor_column

    if is_tensor_column(column):
        return column.to_numpy_ndarray()
    return column.to_numpy(zero_copy_only=False)


def _array_from_numpy(values: np.ndarray) -> pa.Array:
    """An output array → Arrow: 1-D stays scalar, higher-rank becomes a tensor column."""
    import pyarrow as pa

    if values.ndim <= 1:
        return pa.array(values)
    from batcher.io.formats.ml.tensor import to_tensor_column

    return to_tensor_column(values)


def serving_udf(
    connect: Callable[[], ServingClient],
    *,
    input_columns: Sequence[str],
    output_columns: Sequence[str] | None = None,
) -> type:
    """Build a load-once class UDF that runs `input_columns` through a serving backend.

    Args:
        connect: a zero-arg callable returning a connected `ServingClient`; run once
            per worker (the model/connection is reused across batches).
        input_columns: the columns sent to the server, in order.
        output_columns: the appended result columns (defaults to the server's keys).

    Returns:
        A class for ``ds.ml.map_batches(...)`` — instantiate-once-per-worker inference.
    """
    inputs = list(input_columns)
    outputs = None if output_columns is None else list(output_columns)

    class _ServingUDF:
        def __init__(self) -> None:
            self._client = connect()

        def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
            import pyarrow as pa

            feed = {name: _column_to_numpy(batch.column(name)) for name in inputs}
            result = self._client.predict(feed)
            keep = [batch.column(i) for i in range(batch.num_columns)]
            names = list(batch.schema.names)
            for name in outputs if outputs is not None else list(result):
                keep.append(_array_from_numpy(result[name]))
                names.append(name)
            return pa.RecordBatch.from_arrays(keep, names=names)

    return _ServingUDF
