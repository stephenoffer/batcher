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


def test_the_learned_store_never_records_a_size_above_the_oom_ceiling():
    """A warm start must not hand the next run a size this run watched fail.

    After an out-of-memory the internal target and the effective size diverge: `current()`
    clamps to the ceiling while `_cur` was multiplied by `grow` on every improving
    observation, so it ran away above a size already proven to fail. The persisted value
    came from that runaway target, so a ceiling of 700 (set by an OOM at 1000 rows) still
    wrote 5316 to the store, and the following run warm-started five times above the
    failing size. `best_size` documents exactly this trap; the record path bypassed it.
    """
    hub = MetadataHub(InProcessBackend())
    controller = ThroughputController(
        min_rows=1, max_rows=65_536, initial=1000, hub=hub, signature=_sig()
    )
    controller.note_oom(rows=1000)
    ceiling = controller.current()
    for step in range(6):
        controller.update(100.0 * (step + 2))  # every observation "improves"

    assert controller.current() <= ceiling
    assert controller.best_size() <= ceiling
    assert learned_batch_size(hub, _sig()) <= ceiling


def test_the_climb_stays_bounded_by_the_oom_ceiling():
    """The effective size, the settle-back target, and the raw target all respect it."""
    controller = ThroughputController(min_rows=1, max_rows=65_536, initial=1000)
    controller.note_oom(rows=1000)
    ceiling = controller.current()
    for step in range(10):
        controller.update(100.0 * (step + 2))
    assert controller.current() <= ceiling
    # A plateau then settles back to a size that was actually run, not to a runaway target.
    for _ in range(4):
        controller.update(1.0)
    assert controller.current() <= ceiling


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
