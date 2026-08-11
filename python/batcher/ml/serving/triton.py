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
    max_batch_size: int | None = None,
    pipeline_depth: int = 1,
    retries: int = 2,
    timeout: float | None = None,
) -> type:
    """A `map_batches` class UDF running each batch through a Triton model.

    Needs ``tritonclient`` (``pip install 'batcher-engine[triton]'``).

    Examples:
        .. doctest::

            >>> from batcher.ml import triton_client  # doctest: +SKIP
            >>> udf = triton_client(  # doctest: +SKIP
            ...     "localhost:8000",
            ...     "resnet50",
            ...     input_columns=["input__0"],
            ...     output_columns=["output__0"],
            ... )
            >>> ds.ml.map_batches(udf, concurrency=4).collect()  # doctest: +SKIP

    Args:
        url: the Triton endpoint (``host:8000`` for http, ``host:8001`` for grpc).
        model: the Triton model name.
        input_columns: the model's input tensor names (the batch columns sent).
        output_columns: the model's output tensor names (appended to each batch).
        protocol: ``"http"`` or ``"grpc"``.
        model_version: optional model version (default: server-chosen).
        max_batch_size: the model's configured Triton batch window. A larger Arrow batch
            is split into requests of at most this many rows. Left unset, the window is
            read from the model's own configuration on the server, which is where it is
            declared and where it can be right — an engine batch is far larger than a
            typical `max_batch_size`, and Triton rejects an oversized request whole.
        pipeline_depth: how many requests to keep in flight, so Triton is not idle while
            the client encodes and decodes. Results stay in input order.
        retries: retry attempts, with jittered backoff, for a transient Triton failure
            (a restarting server, a dropped connection).
        timeout: per-request timeout in seconds. Without one, a wedged Triton replica
            blocks the worker forever and the retry never fires (the call never returns);
            with one, a timed-out request raises and is retried like any transient failure.

    Returns:
        A class for ``ds.ml.map_batches(...)`` — the client connects once per worker.
    """
    if protocol not in ("http", "grpc"):
        raise BackendError(f"triton protocol must be 'http' or 'grpc', got {protocol!r}")

    def connect() -> _TritonServingClient:
        return _TritonServingClient(
            url, model, list(output_columns), protocol, model_version, timeout
        )

    return serving_udf(
        connect,
        input_columns=input_columns,
        output_columns=output_columns,
        max_batch_size=max_batch_size,
        pipeline_depth=pipeline_depth,
        retries=retries,
    )


class _TritonServingClient:
    """Wraps a `tritonclient` http/grpc connection as a `ServingClient`."""

    def __init__(
        self,
        url: str,
        model: str,
        outputs: list[str],
        protocol: str,
        version: str,
        timeout: float | None = None,
    ) -> None:
        try:
            if protocol == "grpc":
                import tritonclient.grpc as tc
            else:
                import tritonclient.http as tc
        except ImportError as exc:  # pragma: no cover - optional extra
            from batcher._internal.errors import MissingDependencyError

            raise MissingDependencyError.of(
                feature="Triton serving", provides="tritonclient", extra="triton"
            ) from exc
        self._tc = tc
        self._client = tc.InferenceServerClient(url=url)
        self._model = model
        self._outputs = outputs
        self._version = version
        self._timeout = timeout

    def warmup(self) -> None:
        """Check the model is loaded and serving, before the first real batch.

        `serving_udf` calls this once per worker under its retry, so a worker starting
        while Triton is still loading waits for it instead of failing the job — and a
        genuinely absent model is reported now, not after the read stage has run.
        """
        ready = getattr(self._client, "is_model_ready", None)
        if ready is None:  # pragma: no cover - older tritonclient without the probe
            return
        if not ready(self._model, model_version=self._version):
            raise BackendError(f"triton model {self._model!r} is not ready")

    def close(self) -> None:
        """Release the Triton connection (an HTTP pool, or a gRPC channel).

        `serving_udf` calls this when the worker is done, through the same optional-`close`
        teardown contract the rest of the engine uses. Without it the pool or channel was
        reclaimed only whenever the garbage collector reached it, so a script running two
        serving stages back to back held both generations of connections at once.
        """
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    def batch_window(self) -> int | None:
        """The `max_batch_size` this model declares, or `None` when it declares none.

        Read from the model configuration the server publishes, once per worker. A Triton
        model config almost always names a window, and it is almost always far below an
        engine batch — `max_batch_size: 8` against sixteen thousand rows — so sending the
        batch whole is rejected outright by the server. Reading it here is what lets the
        connector work against a real model without the caller restating a number the
        server already knows.

        A model with `max_batch_size: 0` does not batch at all in Triton's sense: its
        inputs carry their own leading dimension. That is `None` here, not zero, because
        zero would read as "split into empty requests".

        Returns:
            The declared window, or `None` when the model declares none, the field is
            absent, or the config cannot be read.
        """
        get_config = getattr(self._client, "get_model_config", None)
        if get_config is None:  # pragma: no cover - an older tritonclient
            return None
        config = get_config(self._model, model_version=self._version)
        # The HTTP client returns the config dict itself; the gRPC client wraps it in a
        # response message whose `config` field holds it.
        config = getattr(config, "config", config)
        declared = (
            config.get("max_batch_size")
            if isinstance(config, dict)
            else getattr(config, "max_batch_size", None)
        )
        return int(declared) if isinstance(declared, int) and declared > 0 else None

    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        infer_inputs = []
        for name, arr in inputs.items():
            triton_in = self._tc.InferInput(name, list(arr.shape), _triton_dtype(arr))
            triton_in.set_data_from_numpy(arr)
            infer_inputs.append(triton_in)
        requested = [self._tc.InferRequestedOutput(name) for name in self._outputs]
        # `client_timeout` bounds the wait so a wedged replica raises (and is retried)
        # instead of blocking the worker forever; both the http and grpc clients accept it.
        response = self._client.infer(
            self._model,
            infer_inputs,
            model_version=self._version,
            outputs=requested,
            client_timeout=self._timeout,
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
