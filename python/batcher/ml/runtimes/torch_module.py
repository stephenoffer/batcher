"""A PyTorch module as a local `ServingClient` — TorchScript, a checkpoint, or a factory.

The framework path, next to the exported-graph path in `runtimes.onnx`. A model that was
never exported — a custom architecture, a research checkpoint, anything with Python control
flow in its forward — still has to run over a hundred million rows, and the thing standing
between "a `.pt` file" and "a distributed inference stage" is a handful of details that are
easy to get individually wrong and expensive to get wrong together:

* the model left in **train mode**, so dropout randomizes predictions and batch-norm updates
  its running statistics from the inference data;
* the forward run **with autograd on**, building a backward graph nobody reads and holding
  every activation alive, which caps the batch size at a fraction of what the card could do;
* a NCHW convolution left in **contiguous** layout on a tensor-core GPU, which declines the
  channels-last kernels the hardware is fastest at;
* the tensors built on the **CPU** and moved per call rather than pinned, so the copy is
  synchronous and the device idles through it.

All four are handled here, once, so the caller writes a path and a column list. `eval()` and
`inference_mode` are unconditional because neither changes what a correct inference computes;
`channels_last` and `torch.compile` are opt-in because both can regress a model they do not
suit, exactly as `ml.inference` documents for the transformers path.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy as np

__all__ = ["TorchModule", "torch_predictor"]


class TorchModule:
    """A loaded ``torch.nn.Module`` presented as a `ServingClient`.

    Built once per worker by `torch_predictor`; usable directly with
    `batcher.ml.serving_udf` for a different column wiring.

    Examples:
        .. doctest::

            >>> from batcher.ml import TorchModule  # doctest: +SKIP
            >>> module = TorchModule("scripted.pt", device="cuda")  # doctest: +SKIP
            >>> module.predict({"x": features})  # doctest: +SKIP
            {'output': array([[0.2, 0.8]], dtype=float32)}

    Args:
        model: a TorchScript file path, or a zero-arg callable returning an ``nn.Module``.
        device: torch device string; defaults to the detected accelerator.
        dtype: ``"float16"``/``"bfloat16"``/``"float32"`` (or an abbreviation). `None`
            keeps the checkpoint's own precision — half precision is a numerical change,
            so it is never applied without being asked for. Float inputs are cast to the
            module's weight precision either way.
        output_names: names for the forward's outputs, in order. A model returning a dict
            keeps its own keys; otherwise these name the tuple's elements, defaulting to
            ``output``, ``output_1``, ...
        channels_last: put 4-D inputs and the model in ``channels_last`` memory format,
            which is what NVIDIA tensor cores want for convolutions.
        compile: ``torch.compile`` the module once per worker. A real win on a fixed-shape
            vision model and a regression on a dynamic-shape text model, so it is opt-in.
    """

    def __init__(
        self,
        model: str | Callable[[], Any],
        *,
        device: str | None = None,
        dtype: str | None = None,
        output_names: Sequence[str] | None = None,
        channels_last: bool = False,
        compile: bool = False,
    ) -> None:
        from batcher._internal.optional import require

        torch = require("torch", feature="torch batch inference", provides="torch", extra="torch")
        self._torch = torch
        from batcher.ml.devices import resolve_device

        self._device = resolve_device(device)
        self._module = _load_module(torch, model, self._device)
        self._dtype = _torch_dtype(torch, dtype)
        if self._dtype is not None:
            self._module = self._module.to(dtype=self._dtype)
        # The precision the module's own weights are in, which is what its inputs must match.
        # This is not an optimization: the engine normalizes Float32 to Float64 at the FFI
        # boundary, so an ordinary numeric column arrives as float64 and *every* float32
        # checkpoint — which is nearly all of them — refused it with "expected scalar type
        # Float but found Double" from inside the forward. Feeding a model the precision it
        # declares is the same rule `runtimes.onnx` applies to a graph's declared input type.
        self._input_dtype = self._dtype or _parameter_dtype(self._module)
        self._channels_last = channels_last
        if channels_last:
            self._module = self._module.to(memory_format=torch.channels_last)
        # `eval()` is not optional and not a performance tweak: a module left in train mode
        # applies dropout and lets batch-norm update its running statistics from the data it
        # is scoring, so the same row scores differently depending on what shared a batch
        # with it. Nothing in the output says so.
        self._module.eval()
        if compile:
            self._module = _compiled(torch, self._module)
        self._names = list(output_names) if output_names is not None else None

    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run the module's forward over one batch of named arrays.

        Args:
            inputs: arrays keyed by the forward's keyword-argument names. A single-input
                module is called positionally, so the key does not have to match.

        Returns:
            The forward's outputs as NumPy arrays, keyed by `output_names` (or the model's
            own dict keys).
        """
        torch = self._torch
        tensors = {name: self._to_tensor(array) for name, array in inputs.items()}
        guard = getattr(torch, "inference_mode", None) or torch.no_grad
        with guard():
            result = (
                self._module(next(iter(tensors.values())))
                if len(tensors) == 1
                else self._module(**tensors)
            )
        return _outputs_to_numpy(result, self._names)

    def close(self) -> None:
        """Drop the module and release its cached device blocks with the worker."""
        self._module = None
        from batcher._internal.hardware.devices import release_device_cache

        release_device_cache()

    def _to_tensor(self, array: np.ndarray) -> Any:
        """One input array as a device tensor in the module's precision and memory format.

        `from_numpy` shares the buffer rather than copying it, so the only copy is the one
        `.to(device)` genuinely needs. Integer inputs keep their dtype — an index tensor cast
        to float is not an index any more — so the precision cast applies to floats only.
        """
        torch = self._torch
        import numpy as np

        contiguous = np.ascontiguousarray(array)
        # An Arrow-backed array is read-only, and `from_numpy` warns once per call that
        # torch cannot write through it. Inference never does, and the alternative — copying
        # every input to get a writable buffer — pays a full batch copy per call to silence a
        # warning about something this path does not do. Suppressed narrowly, by message, so
        # a genuine non-writable problem elsewhere still surfaces.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*non-writable.*", category=UserWarning)
            tensor = torch.from_numpy(contiguous)
        # Floats only. An integer input is an index — a token id, a segment id, a class — and
        # casting one to a float is not a precision change, it is destroying the thing the
        # embedding table is looked up by.
        if self._input_dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(dtype=self._input_dtype)
        if self._channels_last and tensor.dim() == 4:
            tensor = tensor.to(memory_format=torch.channels_last)
        return tensor.to(self._device) if self._device != "cpu" else tensor


def _parameter_dtype(module: Any) -> Any | None:
    """The floating-point dtype the module's weights are in, or `None` when it has none.

    Read from the first floating parameter rather than from a buffer or an integer one: a
    module's integer buffers (position ids, a token-type table) say nothing about the
    precision its matmuls run in. `None` for a module with no parameters at all — a pure
    functional transform — where there is nothing to match and the input is left alone.
    """
    try:
        for parameter in module.parameters():
            if parameter.is_floating_point():
                return parameter.dtype
    except Exception:  # pragma: no cover - a scripted module without `parameters()`
        return None
    return None


def _load_module(torch: Any, model: str | Callable[[], Any], device: str) -> Any:
    """The ``nn.Module`` behind a TorchScript path, a pickled module, or a factory.

    A path is tried as TorchScript first and as a pickled module second, because
    ``torch.jit.load`` raises a clear error on a non-script archive while ``torch.load``
    would happily return a bare ``state_dict`` — a dict, not a module — and fail later with
    ``'dict' object is not callable`` from inside the forward. A state dict has no
    architecture in it, so it needs a factory; saying that here is more useful than the
    failure it otherwise produces.
    """
    if not isinstance(model, str):
        loaded = model()
        return loaded.to(device) if hasattr(loaded, "to") else loaded
    try:
        return torch.jit.load(model, map_location=device)
    except Exception:
        loaded = torch.load(model, map_location=device, weights_only=False)
    if isinstance(loaded, dict):
        from batcher._internal.errors import BackendError

        raise BackendError(
            f"{model!r} holds a state_dict, which has no model architecture in it. Pass a "
            "zero-arg factory that builds the module and loads this file into it, e.g. "
            "torch_predictor(lambda: build_model(weights=path), ...)."
        )
    return loaded.to(device) if hasattr(loaded, "to") else loaded


def _torch_dtype(torch: Any, dtype: str | None) -> Any | None:
    """The ``torch.dtype`` for a dtype name, or `None` to keep the checkpoint's precision."""
    if dtype is None:
        return None
    from batcher.ml.devices import resolve_dtype

    return getattr(torch, resolve_dtype(dtype))


def _compiled(torch: Any, module: Any) -> Any:
    """``torch.compile`` the module, falling back to eager on any failure.

    Eager is always correct, so a compilation that the installed torch/inductor cannot do —
    an unsupported operator, no compiler toolchain on the image — must degrade rather than
    fail a job that would otherwise have run.
    """
    from batcher._internal.logging import note_suppressed

    try:
        return torch.compile(module)
    except Exception as exc:
        note_suppressed("ml", "torch.compile the inference module", exc)
        return module


def _outputs_to_numpy(result: Any, names: Sequence[str] | None) -> dict[str, np.ndarray]:
    """The forward's return value as ``{name: ndarray}``, whatever shape it came back in.

    A model returns a tensor, a tuple of tensors, a dict, or a HuggingFace-style output
    object with a ``to_tuple``. All four are normalized here so `serving_udf` sees one shape,
    and so a caller never has to know which of them their checkpoint happens to use.
    """
    if hasattr(result, "to_tuple") and not isinstance(result, dict | tuple | list):
        result = result.to_tuple()
    if isinstance(result, dict):
        return {name: _tensor_to_numpy(value) for name, value in result.items()}
    values = list(result) if isinstance(result, tuple | list) else [result]
    labels = list(names) if names is not None else _default_names(len(values))
    return {
        label: _tensor_to_numpy(value)
        for label, value in zip(labels, values, strict=False)
        if value is not None
    }


def _default_names(count: int) -> list[str]:
    """``["output"]``, then ``output_1``, ``output_2``, ... for a multi-output forward."""
    return ["output" if i == 0 else f"output_{i}" for i in range(count)]


def _tensor_to_numpy(value: Any) -> np.ndarray:
    """One output tensor as a host NumPy array.

    `detach` before `cpu` because a tensor produced outside `inference_mode` (a model that
    opens its own autograd scope) still carries `requires_grad`, and `.numpy()` refuses one.
    Half precision is widened to float32: Arrow has no float16 tensor type that the rest of
    the engine reads, so keeping it would produce a column nothing downstream can use.
    """
    import numpy as np

    if not hasattr(value, "detach"):
        return np.asarray(value)
    tensor = value.detach()
    if hasattr(tensor, "is_floating_point") and tensor.is_floating_point():
        tensor = tensor.float()
    return tensor.cpu().numpy()


def torch_predictor(
    model: str | Callable[[], Any],
    *,
    input_columns: Sequence[str],
    output_columns: Sequence[str] | None = None,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    device: str | None = None,
    dtype: str | None = None,
    channels_last: bool = False,
    compile: bool = False,
    max_batch_size: int | None = None,
    pipeline_depth: int = 1,
) -> type:
    """A load-once class UDF running a PyTorch module's forward over each batch.

    The module loads once per worker and every batch crosses to the device as whole tensors,
    never per row. `eval()` and `torch.inference_mode()` are applied for you, which is the
    pair most hand-written inference UDFs forget.

    Examples:
        .. doctest::

            >>> from batcher.ml import torch_predictor  # doctest: +SKIP
            >>> udf = torch_predictor(  # doctest: +SKIP
            ...     "resnet50_scripted.pt",
            ...     input_columns=["image"],
            ...     output_columns=["logits"],
            ...     channels_last=True,
            ... )
            >>> ds.ml.map_batches(udf, num_gpus=1).collect()  # doctest: +SKIP

    Args:
        model: a TorchScript path, a pickled module path, or a zero-arg factory returning
            an ``nn.Module`` (the shape a checkpoint needs, since a ``state_dict`` carries
            no architecture).
        input_columns: dataset columns to feed, in the forward's argument order.
        output_columns: names for the appended result columns; defaults to the forward's
            output names.
        input_names: the forward's keyword names for `input_columns`, when they differ from
            the column names. A single-input module is called positionally regardless.
        output_names: names for a tuple/tensor return, in order.
        device: torch device string; defaults to the detected accelerator.
        dtype: run the module in this precision (``"float16"``, ``"bfloat16"``). Left unset
            the checkpoint's own precision is kept, because half precision is a numerical
            change and this path does not make one on the caller's behalf. Either way the
            *inputs* are cast to whatever precision the weights are in, since a float32
            module cannot consume the float64 an ordinary numeric column arrives as.
        channels_last: use ``channels_last`` memory format for the module and its 4-D
            inputs, which is the layout convolution tensor cores are fastest in.
        compile: ``torch.compile`` the module once per worker. Measured as a win on
            fixed-shape vision models and a regression on dynamic-shape text models, so it
            is opt-in rather than automatic.
        max_batch_size: rows per forward pass; the whole batch by default.
        pipeline_depth: forwards to keep in flight so the device is not idle between them.

    Returns:
        A class for ``ds.ml.map_batches(...)`` — the module loads once per worker.
    """
    columns = list(input_columns)
    forward_names = list(input_names) if input_names is not None else None
    from batcher.ml.serving import serving_udf

    # A forward returns tensors, not named ports: unlike an ONNX graph there is nothing to
    # preserve, so the caller's `output_columns` ARE the names, applied in return order.
    # An explicit `output_names` still wins, for the model that returns a dict and whose
    # keys the caller wants mapped rather than replaced.
    labels = list(output_names) if output_names is not None else output_columns

    def connect() -> Any:
        module = TorchModule(
            model,
            device=device,
            dtype=dtype,
            output_names=labels,
            channels_last=channels_last,
            compile=compile,
        )
        from batcher.ml.runtimes.providers import RenamedPorts, port_mapping

        renamed = port_mapping(list(labels), output_columns) if output_names is not None else {}
        if forward_names is None and not renamed:
            return module
        inputs = dict(zip(columns, forward_names or columns, strict=False))
        return RenamedPorts(module, inputs, renamed)

    return serving_udf(
        connect,
        input_columns=columns,
        output_columns=output_columns,
        max_batch_size=max_batch_size,
        pipeline_depth=pipeline_depth,
        retries=0,  # a local forward that raises will raise again; a retry only hides why
    )
