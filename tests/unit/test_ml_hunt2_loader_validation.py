"""Bug-hunt (wave 2) regressions for loader batch_size validation (ml/loader).

The defect: `stream_loader` / `shard_stream_loader` computed ``bs = max(1, batch_size)``,
silently coercing a typo'd ``batch_size=0`` (or a negative) into an epoch of one-row
batches — correct output, catastrophically slow, no error. `iter_torch_batches` instead
let ``batch_size=0`` fall through to the engine and surface as a bare
``ValueError: range() arg 3 must not be zero``. The sampler primitives these wrap already
reject ``batch_size < 1``; the loaders now do too, with a typed `PlanError`.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.ml.loader.indexed import stream_loader
from batcher.ml.loader.lazy import iter_torch_batches

torch = pytest.importorskip("torch")


def _ds() -> bt.Dataset:
    return bt.from_pydict({"x": list(range(6))})


@pytest.mark.parametrize("bad", [0, -1, -3])
def test_stream_loader_rejects_bad_batch_size(bad: int) -> None:
    # Without the fix this silently builds one-row batches instead of raising.
    with pytest.raises(PlanError):
        stream_loader(_ds(), batch_size=bad)


@pytest.mark.parametrize("bad", [0, -1, -3])
def test_iter_torch_batches_rejects_bad_batch_size(bad: int) -> None:
    with pytest.raises(PlanError):
        list(iter_torch_batches(_ds(), batch_size=bad, device=None))


def test_valid_batch_size_still_works() -> None:
    loader = stream_loader(_ds(), batch_size=2, shuffle=False, drop_last=False)
    assert len(loader) == 3
    # None stays valid for the lazy loader (engine default batch size).
    assert sum(1 for _ in iter_torch_batches(_ds(), batch_size=None, device=None)) >= 1
