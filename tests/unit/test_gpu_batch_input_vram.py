"""A GPU batch occupies the device twice over: the input rows, and their activations.

`decide_gpu_map_params` budgeted only the second, at a flat 64 KiB/row activation prior,
and treated the input tensor as free. That is a rounding error on a numeric feature row and
the whole budget on the data the rule exists for: a decoded 224x224x3 `uint8` image is
147 KiB per row *before* a single activation, and one 1080p RGB frame is 5.9 MiB. The seeded
batch then asked the device for far more VRAM than it has — an OOM on the first dispatch,
not a slow start the throughput controller could recover from.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher.kyber.gpu.policy import decide_gpu_map_params

pytestmark = pytest.mark.unit

_DEVICE_GB = 24.0
_MODEL_GB = 3.0

# Widths of the columns a real inference stage reads.
_NUMERIC = 64.0
_EMBEDDING = 768 * 4.0
_IMAGE = 224.0 * 224 * 3
_FRAME = 1920.0 * 1080 * 3


def _seed(width: float) -> int:
    params = decide_gpu_map_params(
        _MODEL_GB, 0.0, None, gpu_memory_gb=_DEVICE_GB, input_row_bytes=width
    )
    return params.batch_size


def test_the_default_reproduces_the_activation_only_budget():
    # The safety property: a caller with no estimator (`input_row_bytes` unset) gets exactly
    # the seed it got before.
    before = decide_gpu_map_params(_MODEL_GB, 0.0, None, gpu_memory_gb=_DEVICE_GB)
    after = decide_gpu_map_params(
        _MODEL_GB, 0.0, None, gpu_memory_gb=_DEVICE_GB, input_row_bytes=0.0
    )
    assert before.batch_size == after.batch_size


def test_a_narrow_row_barely_moves_the_seed():
    # 64 B against a 64 KiB activation prior — the input is genuinely negligible here, and
    # the change must not disturb the numeric case it was already right for.
    assert _seed(_NUMERIC) == pytest.approx(_seed(0.0), rel=0.01)


@pytest.mark.parametrize(("label", "width"), [("image", _IMAGE), ("1080p frame", _FRAME)])
def test_a_wide_input_batch_fits_the_device(label, width):
    # The property that matters: the batch the seed proposes must not, on its own, demand
    # more VRAM than the device has left after the model.
    headroom_bytes = (_DEVICE_GB * 0.85 - _MODEL_GB) * 1e9
    assert _seed(width) * width <= headroom_bytes, label


def test_the_seed_shrinks_as_the_input_widens():
    assert _seed(_NUMERIC) > _seed(_EMBEDDING) > _seed(_IMAGE) > _seed(_FRAME)


def test_a_video_frame_stage_no_longer_asks_for_hundreds_of_gigabytes():
    # Concretely: at the previous activation-only seed a 1080p stage's inputs alone came to
    # roughly 200 GB on a 24 GB device.
    old = decide_gpu_map_params(_MODEL_GB, 0.0, None, gpu_memory_gb=_DEVICE_GB).batch_size
    assert old * _FRAME > 100e9
    assert _seed(_FRAME) * _FRAME < _DEVICE_GB * 1e9


def test_a_user_pinned_batch_size_is_still_honored():
    params = decide_gpu_map_params(
        _MODEL_GB, 0.0, 512, gpu_memory_gb=_DEVICE_GB, input_row_bytes=_FRAME
    )
    assert params.batch_size == 512


def test_the_rule_reads_the_width_from_the_plan():
    # End to end: two inference stages differing only in their input column's width must not
    # be seeded with the same batch size.
    from batcher.config import active_config
    from batcher.kyber.cardinality import CardinalityEstimator
    from batcher.kyber.gpu.sizing import _input_row_bytes
    from batcher.kyber.pass_base import OptimizerContext

    rows = 64
    narrow = bt.from_pydict({"x": list(range(rows))})
    wide = bt.from_arrow(
        pa.table(
            {
                "t": pa.FixedShapeTensorArray.from_numpy_ndarray(
                    np.zeros((rows, 64, 64, 3), dtype="uint8")
                )
            }
        )
    )

    def width_of(ds):
        staged = ds.ml.map_batches(lambda b: b, num_gpus=1, model_memory_gb=1.0)
        ctx = OptimizerContext(
            config=active_config(),
            sources=staged._sources,
            hub=None,
            estimator=CardinalityEstimator(staged._sources),
        )
        return _input_row_bytes(staged._plan, ctx)

    assert width_of(wide) > width_of(narrow)


# --- the size gate that decides whether the GPU is used at all -------------------


def _gpu_decision(ds, min_rows: int):
    """`decide_gpu_backend` under a `gpu_min_rows` floor of `min_rows`, one GPU visible."""
    import dataclasses

    from batcher.config import active_config, config_context
    from batcher.kyber.gpu.policy import decide_gpu_backend

    cfg = active_config()
    scoped = cfg.replace(distributed=dataclasses.replace(cfg.distributed, gpu_min_rows=min_rows))
    with config_context(scoped):
        return decide_gpu_backend(ds._plan, ds._sources, None, gpu_count=1)


def _tensor_frame(rows: int):
    arr = pa.FixedShapeTensorArray.from_numpy_ndarray(np.zeros((rows, 224, 224, 3), dtype="uint8"))
    return bt.from_arrow(pa.table({"t": arr}))


def test_the_gpu_size_gate_reads_bytes_as_well_as_rows():
    """The GPU's fixed overhead is amortized by *work*, and rows proxy for work only while
    a row is the ~64 bytes `optimizer.row_bytes` assumes.

    At the shipped 10M-row floor a narrow relation clears the gate at 0.64 GB of input while
    a decoded 224x224x3 image column needs 1,505 GB — so a 100 GB image query, the workload
    this path exists for, was refused the GPU for being "too small to amortize the overhead".
    Two relations of the *same row count* must now separate on their bytes.
    """
    rows = 1000
    narrow = bt.from_pydict({"k": list(range(rows))})
    wide = _tensor_frame(rows)
    assert _gpu_decision(narrow, 1_000_000).use_gpu is False
    assert _gpu_decision(wide, 1_000_000).use_gpu is True


def test_a_narrow_query_below_both_floors_is_unchanged():
    """The safety property: nothing that stayed on the CPU may be moved onto the GPU."""
    ds = bt.from_pydict({"k": list(range(1000))})
    decision = _gpu_decision(ds, 1_000_000)
    assert decision.use_gpu is False
    assert "CPU wins on overhead" in decision.reason


def test_the_byte_floor_is_derived_from_the_row_floor():
    """One knob says how big "big" is, rather than two that can drift apart."""
    from batcher.config import active_config

    ds = _tensor_frame(64)
    min_rows = 1_000_000
    expected_gb = min_rows * active_config().optimizer.row_bytes / 1e9
    reason = _gpu_decision(ds, min_rows).reason
    # Far under both floors at 64 rows, so the refusal names both.
    assert f"{expected_gb:.2f}GB" in reason
