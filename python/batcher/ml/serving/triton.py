"""NVIDIA Triton Inference Server adapter — batch inference over HTTP or gRPC.

Sends the input columns as named Triton tensors and reads the named output tensors
back, one request per Arrow batch. The Triton client (``tritonclient``) is built once
per worker. Tensor input columns (e.g. decoded images from `read.images(decode=True)`)
pass through with their ``(N, *shape)`` form intact.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.errors import BackendError
from batcher.ml.serving.base import serving_udf

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

__all__ = ["triton_client"]


def triton_client(
    url: str,
    model: str,
    *,
    input_columns: Sequence[str],
    output_columns: Sequence[str],
    protocol: str = "http",
    model_version: str = "",
) -> type:
    """A `map_batches` class UDF running each batch through a Triton model.

    Args:
        url: the Triton endpoint (``host:8000`` for http, ``host:8001`` for grpc).
        model: the Triton model name.
        input_columns / output_columns: the model's input/output tensor names.
        protocol: ``"http"`` or ``"grpc"``.
        model_version: optional model version (default: server-chosen).

    Needs ``tritonclient`` (``pip install 'batcher-engine[triton]'``).
    """
    if protocol not in ("http", "grpc"):
        raise BackendError(f"triton protocol must be 'http' or 'grpc', got {protocol!r}")

    def connect() -> _TritonServingClient:
        return _TritonServingClient(url, model, list(output_columns), protocol, model_version)

    return serving_udf(connect, input_columns=input_columns, output_columns=output_columns)


class _TritonServingClient:
    """Wraps a `tritonclient` http/grpc connection as a `ServingClient`."""

    def __init__(
        self, url: str, model: str, outputs: list[str], protocol: str, version: str
    ) -> None:
        try:
            if protocol == "grpc":
                import tritonclient.grpc as tc
            else:
                import tritonclient.http as tc
        except ImportError as exc:  # pragma: no cover - optional extra
            raise BackendError("Triton needs: pip install 'batcher-engine[triton]'") from exc
        self._tc = tc
        self._client = tc.InferenceServerClient(url=url)
        self._model = model
        self._outputs = outputs
        self._version = version

    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        infer_inputs = []
        for name, arr in inputs.items():
            triton_in = self._tc.InferInput(name, list(arr.shape), _triton_dtype(arr))
            triton_in.set_data_from_numpy(arr)
            infer_inputs.append(triton_in)
        requested = [self._tc.InferRequestedOutput(name) for name in self._outputs]
        response = self._client.infer(
            self._model, infer_inputs, model_version=self._version, outputs=requested
        )
        return {name: response.as_numpy(name) for name in self._outputs}


# NumPy dtype *name* → Triton dtype string. Keyed by name (not `np.dtype` object) so
# extension dtypes from `ml_dtypes` (bfloat16, the fp8 variants modern LLMs serve in)
# map without importing the package — Triton's KServe-v2 dtype vocabulary.
_TRITON_DTYPES = {
    "bool": "BOOL",
    "uint8": "UINT8",
    "uint16": "UINT16",
    "uint32": "UINT32",
    "uint64": "UINT64",
    "int8": "INT8",
    "int16": "INT16",
    "int32": "INT32",
    "int64": "INT64",
    "float16": "FP16",
    "float32": "FP32",
    "float64": "FP64",
    "bfloat16": "BF16",
    "float8_e4m3fn": "FP8",
    "float8_e5m2": "FP8",
}


def _triton_dtype(arr: np.ndarray) -> str:
    """The Triton dtype string for a NumPy array (``FP32``/``BF16``/``UINT8``/…).

    Covers the bf16 and fp8 dtypes (via `ml_dtypes`) modern transformer serving uses.
    """
    dt = _TRITON_DTYPES.get(arr.dtype.name)
    if dt is None:
        raise BackendError(
            f"unsupported Triton input dtype {arr.dtype.name}; supported: "
            f"{sorted(set(_TRITON_DTYPES.values()))}"
        )
    return dt
