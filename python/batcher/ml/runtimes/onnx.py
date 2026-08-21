"""ONNX Runtime as a local `ServingClient` — deep-model batch inference, no server.

ONNX is how a trained model leaves its training framework: a PyTorch/TensorFlow/sklearn
model is exported once and then runs anywhere ONNX Runtime does, at a fraction of the
memory and startup cost of loading the original framework. It is the standard production
inference runtime, and the standard way to reach TensorRT and OpenVINO without writing
against either directly (both are execution providers).

Batcher already had ONNX in the *tabular* path (`ml.tabular`), where a model is one
feature matrix in and one prediction out. A deep model is not that shape: it takes several
named inputs (``input_ids``, ``attention_mask``, ``pixel_values``), returns several named
outputs, and cares which of them you fetch. This module is that shape, expressed as a
`ServingClient` so it inherits the whole `serving_udf` apparatus — batch splitting,
in-flight pipelining, retry, output alignment, worker teardown — instead of restating it.

Two behaviors here exist because an exported graph is stricter than a framework model:

* **Declared dtypes are honored exactly.** A graph input typed ``tensor(float16)`` gets
  float16, ``tensor(int64)`` gets int64. Feeding the wrong width is the single most common
  ONNX failure, and the runtime reports it as an opaque shape mismatch far from the cause.
* **A fixed batch dimension is a batch window.** Exporters bake the tracing batch size into
  the graph unless told not to, so a 1024-row Arrow batch meets a model that only accepts
  32 and the run fails. The static dimension is read at load and reported through
  `ServingClient.batch_window`, which `serving_udf` already splits against.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher.ml.runtimes.providers import (
    RenamedPorts,
    onnx_providers,
    port_mapping,
    runtime_thread_target,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

__all__ = ["ONNX_TO_NUMPY", "OnnxSession", "onnx_predictor"]

#: ONNX tensor element types mapped to the NumPy dtype the runtime demands for them.
#: A graph declares its own precision and does not coerce; feeding float32 to a float16
#: input raises `INVALID_ARGUMENT` with a message about the *shape*, which sends everyone
#: looking in the wrong place. Coercing against this table is what makes an fp16 export
#: usable at all.
ONNX_TO_NUMPY: dict[str, str] = {
    "tensor(float)": "float32",
    "tensor(float16)": "float16",
    "tensor(bfloat16)": "float32",  # NumPy has no bfloat16; the runtime casts on ingest
    "tensor(double)": "float64",
    "tensor(int64)": "int64",
    "tensor(int32)": "int32",
    "tensor(int16)": "int16",
    "tensor(int8)": "int8",
    "tensor(uint64)": "uint64",
    "tensor(uint32)": "uint32",
    "tensor(uint16)": "uint16",
    "tensor(uint8)": "uint8",
    "tensor(bool)": "bool",
}

#: Graph-optimization levels, friendly name to ONNX Runtime enum member.
_GRAPH_LEVELS = {
    "disabled": "ORT_DISABLE_ALL",
    "basic": "ORT_ENABLE_BASIC",
    "extended": "ORT_ENABLE_EXTENDED",
    "all": "ORT_ENABLE_ALL",
}


class OnnxSession:
    """A loaded ONNX Runtime session presented as a `ServingClient`.

    Built once per worker by `onnx_predictor`; usable directly with
    `batcher.ml.serving_udf` when you want a different column wiring than the predictor's.

    Examples:
        .. doctest::

            >>> from batcher.ml import OnnxSession  # doctest: +SKIP
            >>> session = OnnxSession("model.onnx", providers=["cuda"])  # doctest: +SKIP
            >>> session.predict({"input": features})  # doctest: +SKIP
            {'logits': array([[0.1, 0.9]], dtype=float32)}

    Args:
        model_path: filesystem path to the ``.onnx`` graph.
        providers: execution providers, friendly (``"cuda"``, ``"tensorrt"``,
            ``"openvino"``) or in ONNX Runtime's own spelling. `None` auto-selects the
            accelerated providers this build offers when an accelerator is visible.
        provider_options: extra per-provider option dicts, keyed by provider name.
        device_id: accelerator ordinal to pin GPU providers to.
        output_names: graph outputs to fetch; `None` fetches every output.
        intra_op_threads: threads inside one operator; `None` sizes to the worker's
            usable cores rather than the host's.
        inter_op_threads: threads across independent operators; `None` leaves the default.
        graph_optimization: ``"disabled"``/``"basic"``/``"extended"``/``"all"``.
        session_options: extra attributes set on ``onnxruntime.SessionOptions``.
    """

    def __init__(
        self,
        model_path: str,
        *,
        providers: object = None,
        provider_options: dict[str, dict[str, Any]] | None = None,
        device_id: int | None = None,
        output_names: Sequence[str] | None = None,
        intra_op_threads: int | None = None,
        inter_op_threads: int | None = None,
        graph_optimization: str = "all",
        session_options: dict[str, Any] | None = None,
    ) -> None:
        from batcher._internal.optional import require

        ort = require(
            "onnxruntime",
            feature="ONNX batch inference",
            provides="onnxruntime",
            extra="onnx",
        )
        options = _session_options(
            ort, intra_op_threads, inter_op_threads, graph_optimization, session_options
        )
        resolved = onnx_providers(
            providers,
            list(ort.get_available_providers()),
            device_id=device_id,
            provider_options=provider_options,
        )
        self._session = ort.InferenceSession(
            model_path, sess_options=options, providers=resolved or None
        )
        self._inputs = {i.name: i for i in self._session.get_inputs()}
        self._outputs = [o.name for o in self._session.get_outputs()]
        self._fetch = list(output_names) if output_names is not None else None
        self._window = _static_batch_size(self._session.get_inputs())

    @property
    def input_names(self) -> list[str]:
        """The graph's input names, in declaration order."""
        return list(self._inputs)

    @property
    def output_names(self) -> list[str]:
        """The graph's output names, in declaration order."""
        return list(self._outputs)

    @property
    def providers(self) -> list[str]:
        """The execution providers the session actually bound to, most preferred first."""
        return list(self._session.get_providers())

    def batch_window(self) -> int | None:
        """The graph's fixed batch size, or `None` when its batch dimension is dynamic.

        `serving_udf` reads this and splits a larger Arrow batch into requests the graph
        accepts, which is the difference between a static export running and failing.
        """
        return self._window

    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run the graph over one batch of named arrays.

        Args:
            inputs: arrays keyed by graph input name. Each is cast to the dtype the graph
                declares and reshaped to the declared rank when a trailing feature axis is
                missing (a 1-D column feeding a ``(batch, 1)`` input).

        Returns:
            The fetched outputs keyed by graph output name.
        """
        feed = {name: self._coerce(name, array) for name, array in inputs.items()}
        results = self._session.run(self._fetch, feed)
        names = self._fetch if self._fetch is not None else self._outputs
        return dict(zip(names, results, strict=False))

    def close(self) -> None:
        """Drop the session so its device memory is released with the worker, not with the GC."""
        self._session = None  # type: ignore[assignment]

    def _coerce(self, name: str, array: np.ndarray) -> np.ndarray:
        """One input array in the dtype and rank the graph declared for `name`."""
        import numpy as np

        spec = self._inputs.get(name)
        if spec is None:
            from batcher._internal.errors import BackendError

            raise BackendError(
                f"ONNX graph has no input named {name!r}; it declares {sorted(self._inputs)}."
            )
        target = ONNX_TO_NUMPY.get(spec.type)
        out = np.asarray(array)
        if target is not None and out.dtype != np.dtype(target):
            out = out.astype(target, copy=False)
        rank = len(spec.shape or ())
        # An exported model whose feature axis is size 1 declares rank 2 while the column
        # arrives rank 1. Adding *one* trailing axis is unambiguous — there is exactly one
        # place it can go — and turns a shape error into a working call. Two or more missing
        # axes is not: (N,) into rank 4 could be any of several reshapes, and picking one
        # would be reshaping the caller's data on a guess. That is left for the runtime to
        # report, where the message names the shapes.
        if rank and out.ndim == rank - 1:
            out = out[..., None]
        return np.ascontiguousarray(out)


def _session_options(
    ort: Any,
    intra_op_threads: int | None,
    inter_op_threads: int | None,
    graph_optimization: str,
    extra: dict[str, Any] | None,
) -> Any:
    """Build ``SessionOptions`` with the thread caps and optimization level applied."""
    from batcher._internal.errors import PlanError

    if graph_optimization not in _GRAPH_LEVELS:
        raise PlanError(
            f"graph_optimization must be one of {sorted(_GRAPH_LEVELS)}, got {graph_optimization!r}"
        )
    options = ort.SessionOptions()
    options.intra_op_num_threads = runtime_thread_target(intra_op_threads)
    if inter_op_threads is not None and inter_op_threads > 0:
        options.inter_op_num_threads = int(inter_op_threads)
    options.graph_optimization_level = getattr(
        ort.GraphOptimizationLevel, _GRAPH_LEVELS[graph_optimization]
    )
    for key, value in (extra or {}).items():
        setattr(options, key, value)
    return options


def _static_batch_size(inputs: Sequence[Any]) -> int | None:
    """The fixed leading dimension every input agrees on, or `None` if any is dynamic.

    A dynamic axis appears as a string (the symbolic name) or `None`, so an int is the only
    thing that pins the batch. A leading dimension of 1 is *not* treated as a window: that
    is what a single-example export looks like, and splitting a batch into 1-row requests
    would be catastrophically slow rather than merely correct — the caller who genuinely
    wants that passes ``max_batch_size=1``.
    """
    sizes = set()
    for spec in inputs:
        shape = getattr(spec, "shape", None) or ()
        if not shape:
            return None
        head = shape[0]
        if not isinstance(head, int):
            return None
        sizes.add(head)
    if len(sizes) != 1:
        return None
    only = sizes.pop()
    return only if only > 1 else None


def onnx_predictor(
    model_path: str,
    *,
    input_columns: Sequence[str],
    output_columns: Sequence[str] | None = None,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    providers: object = None,
    provider_options: dict[str, dict[str, Any]] | None = None,
    device_id: int | None = None,
    intra_op_threads: int | None = None,
    inter_op_threads: int | None = None,
    graph_optimization: str = "all",
    session_options: dict[str, Any] | None = None,
    max_batch_size: int | None = None,
    pipeline_depth: int = 1,
) -> type:
    """A load-once class UDF running an ONNX graph over each batch.

    The session is created once per worker (ONNX Runtime holds the weights and, on GPU, a
    device context — rebuilding it per batch would dominate the run) and every batch is fed
    through it as whole arrays, never per row.

    Reach TensorRT or OpenVINO through `providers` rather than a separate integration:
    ``providers=["tensorrt", "cuda"]`` compiles the supported subgraphs with TensorRT and
    leaves the rest on CUDA, and ``providers=["openvino"]`` targets an Intel CPU/GPU/NPU.

    Examples:
        .. doctest::

            >>> from batcher.ml import onnx_predictor  # doctest: +SKIP
            >>> udf = onnx_predictor(  # doctest: +SKIP
            ...     "resnet50.onnx",
            ...     input_columns=["pixel_values"],
            ...     output_columns=["logits"],
            ...     providers=["cuda"],
            ... )
            >>> ds.ml.map_batches(udf, num_gpus=1).collect()  # doctest: +SKIP

    Args:
        model_path: filesystem path to the ``.onnx`` graph.
        input_columns: dataset columns to feed, in the graph's input order.
        output_columns: names for the appended result columns; defaults to the graph's
            own output names.
        input_names: graph input names for `input_columns`, when the column names differ
            from the graph's. Defaults to the graph's declaration order.
        output_names: graph outputs to fetch; `None` fetches every output.
        providers: execution providers, friendly (``"cuda"``, ``"tensorrt"``,
            ``"openvino"``, ``"rocm"``, ``"coreml"``) or ONNX Runtime's own spelling.
            `None` auto-selects the accelerated providers available on this worker.
        provider_options: extra per-provider option dicts, keyed by provider name.
        device_id: accelerator ordinal to pin GPU providers to.
        intra_op_threads: threads within one operator; sized to the worker's cores by default.
        inter_op_threads: threads across independent operators.
        graph_optimization: ``"disabled"``/``"basic"``/``"extended"``/``"all"``.
        session_options: extra attributes set on ``onnxruntime.SessionOptions``.
        max_batch_size: rows per graph invocation. Defaults to the graph's own fixed batch
            dimension when it has one, and to the whole batch when it does not.
        pipeline_depth: graph invocations to keep in flight, so the device keeps working
            while this worker converts the next sub-batch. Results stay in input order.

    Returns:
        A class for ``ds.ml.map_batches(...)`` — the session loads once per worker.
    """
    columns = list(input_columns)
    graph_inputs = list(input_names) if input_names is not None else None
    from batcher.ml.serving import serving_udf

    def connect() -> Any:
        session = OnnxSession(
            model_path,
            providers=providers,
            provider_options=provider_options,
            device_id=device_id,
            output_names=output_names,
            intra_op_threads=intra_op_threads,
            inter_op_threads=inter_op_threads,
            graph_optimization=graph_optimization,
            session_options=session_options,
        )
        names = graph_inputs if graph_inputs is not None else session.input_names[: len(columns)]
        fetched = list(output_names) if output_names is not None else session.output_names
        return RenamedPorts(
            session,
            dict(zip(columns, names, strict=False)),
            port_mapping(fetched, output_columns),
        )

    return serving_udf(
        connect,
        input_columns=columns,
        output_columns=output_columns,
        max_batch_size=max_batch_size,
        pipeline_depth=pipeline_depth,
        retries=0,  # a local graph that raises will raise again; retrying only hides the cause
    )
