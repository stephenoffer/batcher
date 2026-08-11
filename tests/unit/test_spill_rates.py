"""The measured spill rate corrects an over-pessimistic device class — and only that way.

`device_class` errs toward pessimism by construction (it resolves a composite device to the
*slowest* class beneath it, and calls an NVMe-oF namespace remote on a transport string), so
the correction this module makes is one-directional on purpose. These pin both halves: that a
device proven faster than its class has its factor brought down, and that a device that merely
*measures* slow is left alone, because the only clock available makes a low reading
inconclusive.
"""

from __future__ import annotations

import pytest

from batcher._internal.hardware import fingerprint
from batcher._internal.hardware.storage import FLASH_SPILL_MBPS
from batcher.kyber.spill_rates import learned_spill_factor, measured_spill_mbps
from batcher.metadata import MetadataHub
from batcher.metadata.backends.in_process import InProcessBackend
from batcher.plan.feedback import OperatorFeedback

pytestmark = pytest.mark.unit

_FP = fingerprint()
# Comfortably above the configured `cost_calibration_min_samples` gate.
_ENOUGH = 40


def _hub(samples: int, *, spill_mb: float, ms: float) -> MetadataHub:
    """A hub whose history holds `samples` spilling sorts of the given size and duration."""
    hub = MetadataHub(InProcessBackend())
    for i in range(samples):
        hub.record(
            OperatorFeedback(
                op_id=i,
                kind="sort",
                n_actual=1000,
                t_op_ms=ms,
                m_peak_bytes=int(spill_mb * 1e6),
                selectivity=1.0,
                batch_size=1024,
                spill_bytes=int(spill_mb * 1e6),
            )
        )
    return hub


# --- the measurement ----------------------------------------------------------------------


def test_the_measured_rate_is_bytes_over_the_operators_wall_time():
    # 500 MB in 1000 ms is 500 MB/s.
    assert measured_spill_mbps(_hub(_ENOUGH, spill_mb=500, ms=1000.0), _FP) == pytest.approx(500.0)


def test_no_hub_measures_nothing():
    assert measured_spill_mbps(None, _FP) is None


def test_too_few_samples_measure_nothing():
    assert measured_spill_mbps(_hub(3, spill_mb=500, ms=1000.0), _FP, min_samples=20) is None


def test_a_tiny_spill_is_not_a_throughput_sample():
    """Under the byte floor the time is mostly fixed overhead, so the implied rate is noise."""
    assert measured_spill_mbps(_hub(_ENOUGH, spill_mb=0.1, ms=5.0), _FP, min_samples=1) is None


def test_an_operator_with_no_recorded_time_is_skipped():
    assert measured_spill_mbps(_hub(_ENOUGH, spill_mb=500, ms=0.0), _FP, min_samples=1) is None


# --- the correction -----------------------------------------------------------------------


def test_a_device_proven_faster_than_its_class_has_its_factor_reduced():
    """A `network` class claims 10x; sustaining half of flash proves no worse than 2x."""
    hub = _hub(_ENOUGH, spill_mb=FLASH_SPILL_MBPS / 2, ms=1000.0)
    corrected = learned_spill_factor(hub, "network", None, _FP)
    assert corrected is not None
    assert 1.0 <= corrected < 10.0


def test_the_correction_is_shrunk_toward_the_class_not_snapped_to_the_measurement():
    """Evidence moves the factor part of the way; it never replaces the reading outright."""
    hub = _hub(_ENOUGH, spill_mb=FLASH_SPILL_MBPS / 2, ms=1000.0)
    corrected = learned_spill_factor(hub, "network", None, _FP)
    implied = 2.0  # flash / (flash / 2)
    assert corrected is not None
    assert implied < corrected < 10.0


def test_more_evidence_moves_the_factor_further():
    fast = {"spill_mb": FLASH_SPILL_MBPS / 2, "ms": 1000.0}
    few = learned_spill_factor(_hub(_ENOUGH, **fast), "network", None, _FP)
    many = learned_spill_factor(_hub(_ENOUGH * 10, **fast), "network", None, _FP)
    assert few is not None and many is not None
    assert many < few  # converging on the measurement as the prior is outweighed


def test_a_device_that_measures_slow_is_left_alone():
    """A low reading is inconclusive: the operator may simply have been compute-bound.

    This is the half that must never fire. Raising the factor on this evidence would price
    every spill on a busy machine as though its disk were the bottleneck.
    """
    hub = _hub(_ENOUGH, spill_mb=FLASH_SPILL_MBPS / 50, ms=1000.0)
    assert learned_spill_factor(hub, "network", None, _FP) is None


def test_a_class_already_at_the_floor_has_nothing_to_correct():
    hub = _hub(_ENOUGH, spill_mb=FLASH_SPILL_MBPS * 4, ms=1000.0)
    assert learned_spill_factor(hub, "nvme", None, _FP) is None


def test_a_cold_store_keeps_the_class_factor():
    assert learned_spill_factor(MetadataHub(InProcessBackend()), "network", None, _FP) is None
    assert learned_spill_factor(None, "network", None, _FP) is None


def test_the_correction_never_falls_below_the_flash_floor():
    """However fast the device measures, a spilled byte is never cheaper than a flash byte."""
    hub = _hub(_ENOUGH * 20, spill_mb=FLASH_SPILL_MBPS * 100, ms=1000.0)
    corrected = learned_spill_factor(hub, "rotational", None, _FP)
    assert corrected is not None and corrected >= 1.0


def test_a_rotational_claim_is_corrected_further_than_a_network_one():
    """Both are corrected toward the same measured evidence, from different starting points."""
    hub = _hub(_ENOUGH, spill_mb=FLASH_SPILL_MBPS / 2, ms=1000.0)
    net = learned_spill_factor(hub, "network", None, _FP)
    rot = learned_spill_factor(hub, "rotational", None, _FP)
    assert net is not None and rot is not None
    assert net < rot  # 30x had further to travel than 10x, and neither reached the measurement


def test_history_from_another_machine_class_is_not_read():
    """The spill device is a property of the machine, so a different class's spills say
    nothing about this one."""
    hub = _hub(_ENOUGH, spill_mb=FLASH_SPILL_MBPS / 2, ms=1000.0)
    assert learned_spill_factor(hub, "network", None, "some-other-machine") is None


# --- what the cost model does with it -------------------------------------------------------


def test_the_corrected_factor_scales_the_spill_term():
    from batcher.kyber.cost.terms import memory_budget, spill_io

    budget = memory_budget(64 << 20)
    state = budget * 20
    claimed = spill_io(state, budget, "network")
    corrected = spill_io(state, budget, "network", 2.0)
    assert corrected == pytest.approx(claimed / 5.0)


def test_an_unsupplied_factor_leaves_every_existing_call_site_unchanged():
    from batcher.kyber.cost.terms import memory_budget, merge_io, spill_io

    budget = memory_budget(64 << 20)
    state = budget * 20
    assert spill_io(state, budget, "network", None) == spill_io(state, budget, "network")
    assert merge_io(state, budget, "network", None) == merge_io(state, budget, "network")
