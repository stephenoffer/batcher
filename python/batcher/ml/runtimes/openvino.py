"""OpenVINO as a local `ServingClient` — the CPU-first inference runtime.

Most batch inference does not run on a GPU. It runs on whatever CPU the cluster already
has, because the model is small, the fleet is large, and accelerators are the scarce
resource. On Intel hardware OpenVINO is the runtime that makes that case fast: it fuses and
re-lays-out the graph for the host's vector units and — the part that matters for *batch*
rather than *serving* — runs several inference streams over the cores at once, which is how
a CPU reaches its throughput ceiling rather than its latency floor.

That last point is why this is a separate module rather than an ONNX Runtime execution
provider. Through the provider you get OpenVINO's kernels but ONNX Runtime's scheduling, one
inference at a time. Here the runtime owns the scheduling, so ``performance_hint`` and
``num_streams`` mean what OpenVINO's own documentation says they mean.

The model may be an OpenVINO IR (``.xml``), an ONNX graph, or a saved TensorFlow/PaddlePaddle
model — the runtime reads all of them, so nothing needs converting first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

__all__ = ["OpenVinoModel", "openvino_predictor"]

#: What to optimize the compiled model for. ``THROUGHPUT`` is the default here and
#: ``LATENCY`` is OpenVINO's, which is the right default for a request-response server and
#: the wrong one for scoring a table: latency mode runs one inference across every core,
#: leaving the cores idle at each stage boundary, where throughput mode keeps several
#: inferences in flight and saturates them.
_HINTS = ("THROUGHPUT", "LATENCY", "CUMULATIVE_THROUGHPUT", "NONE")


class OpenVinoModel:
    """A compiled OpenVINO model presented as a `ServingClient`.

    Examples:
        .. doctest::

            >>> from batcher.ml import OpenVinoModel  # doctest: +SKIP
            >>> model = OpenVinoModel("model.xml", device="CPU")  # doctest: +SKIP
            >>> model.predict({"input": features})  # doctest: +SKIP
            {'logits': array([[0.3, 0.7]], dtype=float32)}

    Args:
        model_path: an OpenVINO IR ``.xml``, an ``.onnx`` graph, or a saved model directory.
        device: OpenVINO device name — ``"CPU"``, ``"GPU"``, ``"NPU"``, ``"AUTO"``, or a
            multi-device string such as ``"MULTI:CPU,GPU"``.
        performance_hint: ``"THROUGHPUT"`` (default), ``"LATENCY"``,
            ``"CUMULATIVE_THROUGHPUT"``, or ``"NONE"`` to leave it unset.
        num_streams: independent inference streams. `None` lets the hint choose, which is
            what OpenVINO's own tuning does better than a fixed number.
        num_threads: cap on inference threads; sized to the worker's usable cores by default
            so co-located actors do not each claim the whole host.
        cache_dir: directory for the compiled-model cache. Compilation is seconds to minutes
            per worker, and every worker in a fleet compiles the same model, so a shared
            cache turns that into a one-time cost for the whole run.
        config: extra properties passed to ``compile_model``.
    """

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "AUTO",
        performance_hint: str = "THROUGHPUT",
        num_streams: int | None = None,
        num_threads: int | None = None,
        cache_dir: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        from batcher._internal.errors import PlanError
        from batcher._internal.optional import require

        if performance_hint not in _HINTS:
            raise PlanError(
                f"performance_hint must be one of {list(_HINTS)}, got {performance_hint!r}"
            )
        ov = require(
            "openvino",
            feature="OpenVINO batch inference",
            provides="openvino",
            extra="openvino",
        )
        core = ov.Core()
        if cache_dir:
            core.set_property({"CACHE_DIR": cache_dir})
        self._compiled = core.compile_model(
            model_path,
            device,
            _properties(performance_hint, num_streams, num_threads, config),
        )
        self._inputs = [_port_name(port, index) for index, port in enumerate(self._compiled.inputs)]
        self._outputs = [
            _port_name(port, index) for index, port in enumerate(self._compiled.outputs)
        ]
        # One reusable request per model. Creating one per batch re-allocates the runtime's
        # internal tensors on every call, which on a small model costs more than the
        # inference does.
        self._request = self._compiled.create_infer_request()

    @property
    def input_names(self) -> list[str]:
        """The compiled model's input names, in port order."""
        return list(self._inputs)

    @property
    def output_names(self) -> list[str]:
        """The compiled model's output names, in port order."""
        return list(self._outputs)

    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run one batch of named arrays through the compiled model.

        Args:
            inputs: arrays keyed by model input name (or by column name, when
                `openvino_predictor` mapped them).

        Returns:
            The model's outputs as NumPy arrays, keyed by output name.
        """
        import numpy as np

        feed = {name: np.ascontiguousarray(array) for name, array in inputs.items()}
        results = self._request.infer(feed)
        return {
            self._outputs[index]: np.asarray(results[port])
            for index, port in enumerate(self._compiled.outputs)
        }

    def close(self) -> None:
        """Drop the request and the compiled model so the runtime frees its arenas."""
        self._request = None
        self._compiled = None


def _properties(
    performance_hint: str,
    num_streams: int | None,
    num_threads: int | None,
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    """The ``compile_model`` property dict for these tuning options."""
    properties: dict[str, Any] = {}
    if performance_hint != "NONE":
        properties["PERFORMANCE_HINT"] = performance_hint
    if num_streams is not None and num_streams > 0:
        properties["NUM_STREAMS"] = str(int(num_streams))
    from batcher.ml.runtimes.providers import runtime_thread_target

    properties["INFERENCE_NUM_THREADS"] = runtime_thread_target(num_threads)
    properties.update(config or {})
    return properties


def _port_name(port: Any, index: int) -> str:
    """A port's tensor name, or a positional fallback when the graph left it unnamed.

    An ONNX graph carries names through; an IR converted from some frameworks does not, and
    `get_any_name` raises rather than returning empty when there is none. A stable positional
    name keeps the column wiring working either way.
    """
    try:
        return str(port.get_any_name())
    except Exception:
        return f"output_{index}" if index else "output"


def openvino_predictor(
    model_path: str,
    *,
    input_columns: Sequence[str],
    output_columns: Sequence[str] | None = None,
    input_names: Sequence[str] | None = None,
    device: str = "AUTO",
    performance_hint: str = "THROUGHPUT",
    num_streams: int | None = None,
    num_threads: int | None = None,
    cache_dir: str | None = None,
    config: dict[str, Any] | None = None,
    max_batch_size: int | None = None,
    pipeline_depth: int = 1,
) -> type:
    """A load-once class UDF running an OpenVINO model over each batch.

    The model compiles once per worker — compilation is the expensive part, and `cache_dir`
    makes it a one-time cost for a whole fleet — and every batch runs through it as whole
    arrays. `performance_hint` defaults to ``THROUGHPUT`` rather than OpenVINO's own
    ``LATENCY``, because scoring a table wants every core busy on several inferences, not one
    inference spread thin across all of them.

    Examples:
        .. doctest::

            >>> from batcher.ml import openvino_predictor  # doctest: +SKIP
            >>> udf = openvino_predictor(  # doctest: +SKIP
            ...     "model.xml",
            ...     input_columns=["features"],
            ...     output_columns=["score"],
            ...     cache_dir="/tmp/ov-cache",
            ... )
            >>> ds.ml.map_batches(udf, concurrency=8).collect()  # doctest: +SKIP

    Args:
        model_path: an OpenVINO IR ``.xml``, an ``.onnx`` graph, or a saved model directory.
        input_columns: dataset columns to feed, in the model's input order.
        output_columns: names for the appended result columns; defaults to the model's own
            output names.
        input_names: the model's input names for `input_columns`, when they differ.
        device: ``"CPU"``, ``"GPU"``, ``"NPU"``, ``"AUTO"``, or ``"MULTI:CPU,GPU"``.
        performance_hint: ``"THROUGHPUT"``, ``"LATENCY"``, ``"CUMULATIVE_THROUGHPUT"``,
            or ``"NONE"``.
        num_streams: independent inference streams; the hint chooses when unset.
        num_threads: inference thread cap; the worker's usable cores when unset.
        cache_dir: directory for the compiled-model cache, shared across workers.
        config: extra properties passed to ``compile_model``.
        max_batch_size: rows per inference; the whole batch by default.
        pipeline_depth: inferences to keep in flight, so the runtime is never idle waiting
            on this worker's array conversion.

    Returns:
        A class for ``ds.ml.map_batches(...)`` — the model compiles once per worker.
    """
    columns = list(input_columns)
    model_inputs = list(input_names) if input_names is not None else None
    from batcher.ml.serving import serving_udf

    def connect() -> Any:
        model = OpenVinoModel(
            model_path,
            device=device,
            performance_hint=performance_hint,
            num_streams=num_streams,
            num_threads=num_threads,
            cache_dir=cache_dir,
            config=config,
        )
        names = model_inputs if model_inputs is not None else model.input_names[: len(columns)]
        from batcher.ml.runtimes.providers import RenamedPorts, port_mapping

        return RenamedPorts(
            model,
            dict(zip(columns, names, strict=False)),
            port_mapping(model.output_names, output_columns),
        )

    return serving_udf(
        connect,
        input_columns=columns,
        output_columns=output_columns,
        max_batch_size=max_batch_size,
        pipeline_depth=pipeline_depth,
        retries=0,  # a local inference that raises will raise again; a retry only hides why
    )
