"""CPU batch inference — the parts a GPU stage got and a CPU stage did not.

Most batch inference runs on whatever CPU the cluster already has. The engine treats that as
a first-class path (`apply._apply_udf_autobatch` documents the load-once class UDF as "the CPU
batch-inference pattern"), but two of the things it does for a model stage were gated behind
`num_gpus > 0`. One of them belongs on CPU as well, and one measurably does not.
"""

from __future__ import annotations

import gc
import os

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.unit


def _rss_bytes() -> int:
    with open("/proc/self/statm") as handle:
        return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


class _RecordsGrad:
    """A CPU torch UDF that reports whether its forward ran with autograd on."""

    saw_grad: bool | None = None

    def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        import torch

        x = torch.ones(4, 4, requires_grad=True)
        out = (x * 2).sum()
        type(self).saw_grad = out.requires_grad
        return batch


def test_a_cpu_stage_runs_its_forward_without_autograd():
    # The gate was `num_gpus > 0`, so every CPU inference stage built a backward graph nobody
    # reads and held each layer's activations alive for the whole call.
    pytest.importorskip("torch", reason="torch not installed")
    _RecordsGrad.saw_grad = None
    bt.from_pydict({"a": [1, 2, 3]}).ml.map_batches(_RecordsGrad).collect()
    assert _RecordsGrad.saw_grad is False


class _OptsOut:
    """A UDF whose *output* is a gradient, so it must keep autograd."""

    batcher_inference_mode = False
    saw_grad: bool | None = None

    def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        import torch

        x = torch.ones(4, 4, requires_grad=True)
        type(self).saw_grad = (x * 2).sum().requires_grad
        return batch


def test_a_udf_that_needs_gradients_can_decline():
    # A saliency map, an adversarial perturbation and an influence score all compute a gradient
    # as their result; the opt-out is what keeps this from breaking them.
    pytest.importorskip("torch", reason="torch not installed")
    _OptsOut.saw_grad = None
    bt.from_pydict({"a": [1, 2, 3]}).ml.map_batches(_OptsOut).collect()
    assert _OptsOut.saw_grad is True


def _deep_model():
    import torch

    return torch.nn.Sequential(
        *[m for _ in range(8) for m in (torch.nn.Linear(512, 512), torch.nn.ReLU())]
    ).eval()


def test_dropping_the_backward_graph_is_what_bounds_a_cpu_stages_memory():
    """The claim behind the change, measured rather than asserted.

    Speed is *not* the reason: the same forward timed within noise either way. Peak memory is,
    and peak memory is what caps the batch size and what kills a CPU worker.
    """
    torch = pytest.importorskip("torch", reason="torch not installed")
    torch.set_num_threads(2)
    model = _deep_model()
    rows = torch.randn(2048, 512)

    def peak(guard) -> int:
        gc.collect()
        base = high = _rss_bytes()
        with guard():
            h = rows
            for layer in model:
                h = layer(h)
                high = max(high, _rss_bytes())
        return high - base

    class _Null:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    with_autograd = peak(_Null)
    without = peak(torch.inference_mode)
    # Deliberately a weak bound: the exact ratio is machine- and allocator-dependent (10.3x
    # measured on the box this was written on). What must hold is the direction and that it is
    # not marginal, because a marginal difference would not be worth a behaviour change.
    assert without * 2 < with_autograd, f"autograd {with_autograd} vs inference_mode {without}"


class _NoTorch:
    """The common CPU UDF: numpy, no torch anywhere."""

    def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        import numpy as np

        doubled = np.asarray(batch.column("a")) * 2
        return batch.append_column("b", pa.array(doubled))


def test_a_non_torch_cpu_udf_is_unaffected():
    # The wrap costs a `sys.modules` lookup when torch was never imported, so a numpy or
    # scikit-learn stage must be untouched in behaviour.
    out = bt.from_pydict({"a": [1, 2, 3]}).ml.map_batches(_NoTorch).to_pydict()
    assert out["b"] == [2, 4, 6]


def test_a_cpu_stage_is_not_autocast(monkeypatch):
    """Half precision stays GPU-only, and that is a decision rather than an oversight.

    It changes the numbers, it needs tensor cores to pay for that, and its probe re-executes
    the model — which for a UDF that bills a request means paying twice.
    """
    from batcher.core.udf import apply

    _ = apply  # the module under test; the patches below are on what it imports
    wrapped: list[str] = []

    import batcher.ml.gpu as gpu

    monkeypatch.setattr(gpu, "autocast_call", lambda call: (wrapped.append("autocast"), call)[1])
    monkeypatch.setattr(
        gpu,
        "inference_mode_call",
        lambda call: (wrapped.append("inference_mode"), call)[1],
    )
    bt.from_pydict({"a": [1, 2, 3]}).ml.map_batches(_NoTorch).collect()
    assert wrapped == ["inference_mode"], wrapped
