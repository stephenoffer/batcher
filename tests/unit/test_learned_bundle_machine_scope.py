"""The learned bundle is keyed by the machine class it was assembled for.

`load_learned_stats` memoizes on `(hub.version, hub.params_version)` and folds in
`load_udf_row_seconds_table`, which reads a hardware-`scoped` namespace — so the bundle's
*content* depends on the enclosing `planning_for` scope while its *key* did not. This is the
third place in the learning loop where a machine class went missing from a cache key, after
`kyber.cpu_shares` and `kyber.spill_rates`.
"""

from __future__ import annotations

import pytest

from batcher.metadata import MetadataHub
from batcher.metadata.backends.in_process import InProcessBackend

pytestmark = pytest.mark.unit


def test_the_learned_bundle_is_keyed_by_the_machine_class_it_was_built_for() -> None:
    """One component of the bundle is machine-scoped, so the bundle's cache key must be too.

    `load_learned_stats` memoizes on `(hub.version, hub.params_version)`, and folds in
    `load_udf_row_seconds_table`, which reads a `scoped("udf.row_seconds")` namespace — so its
    content depends on the enclosing `planning_for` while its key did not. A driver that plans
    a distributed run and then a local one leaves both counters untouched, so the stale bundle
    reads as current and each scope is served the other's measured per-row UDF cost. With one
    `fn` at 1 ms/row on the worker and 1 ns/row on the driver that is a millionfold error in
    the figure deciding whether a `map_batches` is priced as a trivial column map.
    """
    from batcher.kyber.column_tables import UDF_ROW_SECONDS_KEY
    from batcher.kyber.learning import load_learned_stats
    from batcher.metadata.hardware_scope import planning_for
    from batcher.metadata.udf_stats import record_udf_row_seconds

    hub = MetadataHub(InProcessBackend())
    with planning_for("worker-class"):
        record_udf_row_seconds(hub, "mod.fn", 1e-3)
    record_udf_row_seconds(hub, "mod.fn", 1e-9)

    assert load_learned_stats(hub)[UDF_ROW_SECONDS_KEY] == {"mod.fn": 1e-9}
    with planning_for("worker-class"):
        assert load_learned_stats(hub)[UDF_ROW_SECONDS_KEY] == {"mod.fn": 1e-3}
    # ...and switching back is not served the worker's entry either.
    assert load_learned_stats(hub)[UDF_ROW_SECONDS_KEY] == {"mod.fn": 1e-9}
