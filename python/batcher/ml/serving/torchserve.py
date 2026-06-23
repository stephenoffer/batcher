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
) -> type:
    """A `map_batches` class UDF posting each batch to a TorchServe model.

    Args:
        base_url: the TorchServe inference base (e.g. ``http://host:8080``).
        model: the registered model name (the URL becomes ``/predictions/{model}``).
        input_columns / output_columns: columns sent and appended.
        timeout: per-request timeout in seconds.
    """
    url = f"{base_url.rstrip('/')}/predictions/{model}"
    return http_client(
        url, input_columns=input_columns, output_columns=output_columns, timeout=timeout
    )
