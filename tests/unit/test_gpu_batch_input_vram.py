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
