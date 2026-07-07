"""Learned warm-start for the throughput batch-size controller.

The `ThroughputController` hill-climbs the inference batch size to the throughput plateau. When a
hub + model signature is supplied it warm-starts from the plateau a prior run learned and persists
each new best back, so a recurring job converges in a few batches instead of re-climbing from the
cold default. The batch size only shards rows, so the *result* is unchanged — only the convergence
speed differs; the controller reaches the same plateau from any start.
"""

from __future__ import annotations

import pytest

from batcher.metadata import MetadataHub
from batcher.metadata.backends import InProcessBackend
from batcher.ml.autobatch import ThroughputController, learned_batch_size, record_batch_size

pytestmark = pytest.mark.unit


def _hub() -> MetadataHub:
    return MetadataHub(InProcessBackend())


def _sig() -> str:
    return "mypkg.MyModel"


# --- persistence round-trip --------------------------------------------------------------


def test_record_and_read_round_trip():
    hub = _hub()
    assert learned_batch_size(hub, _sig()) is None  # cold
    record_batch_size(hub, _sig(), 1024)
    assert learned_batch_size(hub, _sig()) == 1024


def test_cold_or_missing_signature_reads_none():
    assert learned_batch_size(None, _sig()) is None
    assert learned_batch_size(_hub(), None) is None
    assert learned_batch_size(_hub(), _sig()) is None


# --- the controller warm-starts from the learned plateau ---------------------------------


def test_controller_seeds_initial_from_learned_size():
    hub = _hub()
    record_batch_size(hub, _sig(), 2000)
    # A cold controller starts at `initial`; a warm one starts at the learned plateau instead.
    cold = ThroughputController(min_rows=16, max_rows=8192, initial=64)
    warm = ThroughputController(min_rows=16, max_rows=8192, initial=64, hub=hub, signature=_sig())
    assert cold.current() == 64
    assert warm.current() == 2000


def test_controller_ignores_learned_when_no_signature():
    hub = _hub()
    record_batch_size(hub, _sig(), 2000)
    c = ThroughputController(min_rows=16, max_rows=8192, initial=64, hub=hub)  # no signature
    assert c.current() == 64  # cold default start


def test_controller_persists_its_best_size():
    hub = _hub()
    c = ThroughputController(min_rows=16, max_rows=8192, initial=64, hub=hub, signature=_sig())
    for _ in range(40):
        c.update(min(c.current(), 1024) * 1.0, vram_fraction=0.5)  # climb to the 1024 plateau
    learned = learned_batch_size(hub, _sig())
    assert learned is not None and learned >= 700  # the plateau was written back
    assert c.best_size() >= 700


# --- result-invariance: the plateau is the same from a cold or warm start ----------------


def _run_to_plateau(c: ThroughputController) -> int:
    for _ in range(60):
        c.update(min(c.current(), 1024) * 1.0, vram_fraction=0.5)
    return c.best_size()


def test_warm_start_reaches_the_same_plateau_as_cold():
    hub = _hub()
    record_batch_size(hub, _sig(), 900)  # seed near the plateau
    cold = _run_to_plateau(ThroughputController(min_rows=16, max_rows=8192, initial=64))
    warm = _run_to_plateau(
        ThroughputController(min_rows=16, max_rows=8192, initial=64, hub=hub, signature=_sig())
    )
    # Both settle in the same high-throughput plateau region — the seed changes how fast the
    # climb converges, not where it lands (and the inference result is invariant to the size).
    assert cold >= 700 and warm >= 700


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
