"""TorchServe adapter — batch inference against a TorchServe model endpoint.

TorchServe exposes models at ``{base_url}/predictions/{model}``. This builds that URL
and reuses the columnar-JSON HTTP client, so a TorchServe handler that accepts and
returns ``{column: [values...]}`` works with no extra glue. (Handlers that speak a
different wire format can use `batcher.ml.serving.http_client` directly.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher.ml.serving.http import http_client

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["torchserve_client"]


def torchserve_client(
    base_url: str,
    model: str,
    *,
    input_columns: Sequence[str],
    output_columns: Sequence[str],
    timeout: float = 30.0,
    tensor_encoding: str = "json",
    max_batch_size: int | None = None,
    pipeline_depth: int = 1,
) -> type:
    """A `map_batches` class UDF posting each batch to a TorchServe model.

    Examples:
        .. doctest::

            >>> from batcher.ml import torchserve_client  # doctest: +SKIP
            >>> udf = torchserve_client(  # doctest: +SKIP
            ...     "http://host:8080",
            ...     "resnet50",
            ...     input_columns=["image"],
            ...     output_columns=["class"],
            ... )
            >>> ds.ml.map_batches(udf, concurrency=4).collect()  # doctest: +SKIP

    Args:
        base_url: the TorchServe inference base (e.g. ``http://host:8080``).
        model: the registered model name (the URL becomes ``/predictions/{model}``).
        input_columns: the columns sent to the handler, in order.
        output_columns: the result columns appended to each batch.
        timeout: per-request timeout in seconds.
        tensor_encoding: how tensor inputs are encoded. Defaults to ``"json"``, the
            nested-list shape a stock TorchServe handler expects. Pass ``"auto"`` for
            the compact binary envelope when your handler decodes it.
        max_batch_size: rows per request. Set it to the model's registered
            ``batch_size``: TorchServe answers a request above that window with an error
            rather than with predictions, and an engine batch is thousands of rows.
        pipeline_depth: how many requests to keep in flight, so the server keeps working
            while this worker encodes and decodes. Results stay in input order.

    Returns:
        A class for ``ds.ml.map_batches(...)`` — the client connects once per worker.
    """
    url = f"{base_url.rstrip('/')}/predictions/{model}"
    return http_client(
        url,
        input_columns=input_columns,
        output_columns=output_columns,
        timeout=timeout,
        tensor_encoding=tensor_encoding,
        max_batch_size=max_batch_size,
        pipeline_depth=pipeline_depth,
    )
