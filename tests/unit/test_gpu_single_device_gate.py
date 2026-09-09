"""When the fan-out declines, running the whole query on one device is sometimes right.

`_translated` has three rungs: fan out across the fleet, run on one worker that reads for
itself, and ship the table from the driver. The second is right for a small plan — the tree
dispatcher says why, one device reading four small relations beats sixteen each reading three —
and it is one board against forty-eight cores for anything else.

The rule is that one device may do the work of **one wave of the fan-out**: at most
`gpu_min_shard_bytes` per device, times the devices. It needs no new constant and it says
something coherent, because the rung is a substitute for a fan-out.

Holding everything else constant on a six-T4 fleet, with the rung's projection fixed:

| query | one device reads | on one device | declined to the CPU |
|---|---:|---:|---:|
| ClickBench q08 | 0.10 GB | **2.21x** | 0.89x |
| ClickBench q10 | 0.58 GB | **1.37x** | 0.87x |
| ClickBench q13 | 0.58 GB | 0.88x | 1.00x |
| ClickBench q22 | 1.60 GB | **0.04x** | 0.86x |
| TPC-H q16 | (tree) | **2.95x** | 1.23x |
| TPC-H q20 | (tree) | **1.48x** | 0.95x |

An earlier version of this gate keyed on "the plan is not shardable", on the strength of
ClickBench q08 through q14 measuring 10.5 s on one device. That was the right observation and
the wrong cause: those queries were slow because the rung read all 105 columns of `hits` to
answer a two-column question. Once that was fixed the same queries were **wins** on one device,
and the shardability rule was left declining four profitable runs to catch one bad one.
"""

from __future__ import annotations

import pytest

from batcher.api.terminal.gpu_backend.translate import _one_device_is_enough, _one_wave_bytes
from batcher.kyber.gpu.policy import GpuDecision

pytestmark = pytest.mark.unit


def _decision(*, distributed: bool = False, desired: int = 1) -> GpuDecision:
    return GpuDecision(
        use_gpu=True,
        distributed=distributed,
        reason="test",
        est_rows=1_000,
        desired_gpus=desired,
    )


# --- what Kyber already decided ----------------------------------------------


def test_a_plan_kyber_wanted_to_spread_may_not_run_on_one_device():
    """Running it on one board is knowingly a fraction of what it was routed to the accelerator
    for, while the CPU engine still gets the whole cluster."""
    assert _one_device_is_enough(_decision(distributed=True, desired=6), 1.0) is False


def test_a_plan_that_wants_several_devices_may_not_either():
    assert _one_device_is_enough(_decision(desired=4), 1.0) is False


# --- what one device would actually read -------------------------------------


def test_a_read_within_one_wave_is_allowed():
    assert _one_device_is_enough(_decision(), _one_wave_bytes() * 0.5) is True


def test_a_read_at_exactly_one_wave_is_allowed():
    assert _one_device_is_enough(_decision(), _one_wave_bytes()) is True


def test_a_read_past_one_wave_is_refused():
    """ClickBench q22 is this case: 1.6 GB onto one board, measured at 0.04x."""
    assert _one_device_is_enough(_decision(), _one_wave_bytes() * 1.01) is False


def test_an_unmeasurable_read_keeps_the_previous_behaviour():
    """`0.0` means the source would not say how big it is. Refusing on no evidence would close
    the rung for every source that cannot report a row count."""
    assert _one_device_is_enough(_decision(), 0.0) is True


def test_the_wave_scales_with_the_fleet(monkeypatch):
    """One wave is the shard floor times the devices, so a wider fleet allows a larger read."""
    import batcher.api.terminal.gpu_backend.translate as translate

    monkeypatch.setattr(translate, "_cluster_gpu_count", lambda: 1)
    narrow = _one_wave_bytes()
    monkeypatch.setattr(translate, "_cluster_gpu_count", lambda: 8)
    assert _one_wave_bytes() == pytest.approx(narrow * 8)


def test_the_wave_is_the_configured_shard_floor():
    """Not a constant of its own: the floor already means "below this, the dispatch costs more
    than the shard", which is exactly the comparison being made."""
    from batcher.config import active_config

    floor = float(active_config().distributed.gpu_min_shard_bytes)
    assert _one_wave_bytes() % floor == 0
