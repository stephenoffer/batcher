"""The warm session fleet must not hold the cluster after the query that borrowed it.

A fleet reserves the cluster's whole CPU capacity — one worker per node holding that node's
cores. That is right while a query shuffles over it, and it is why the fleet is kept warm only
under a *lease*: the idle timer may fire, and free those cores, once no operator holds one.

`execute_aggregate_flight` took a lease and never gave it back. It was the one Flight operator
that hand-rolled its teardown instead of calling `release_fleet`, and the hand-rolled version
covered only the fleet it had *spawned*. So the lease count never returned to zero, the idle
timer was never armed, and the fleet held all 32 cores of a 4-node cluster for the life of the
process. The next query that ran plain Ray tasks — a filtered scan, the commonest shape there
is — found no schedulable core anywhere and hung forever. That is ClickBench stalling on
`SELECT UserID FROM hits WHERE UserID = ...` with `CPU 32/32 in use` and 20 tasks pending.

The two tests below pin the two halves: the lease comes back, and a scan behind a warm fleet
still runs.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, count

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _need_ray():
    pytest.importorskip("ray")


@pytest.fixture
def _table() -> pa.Table:
    rng = np.random.default_rng(0)
    return pa.table(
        {
            "k": rng.integers(0, 50, 4000).astype("int64"),
            "v": rng.integers(0, 100, 4000).astype("int64"),
        }
    )


def test_a_flight_aggregate_hands_its_session_fleet_lease_back(_table):
    """Leases must return to their starting count, or the fleet never goes idle again."""
    from batcher.dist.fleet import _fleet

    before = _fleet._SESSION_LEASES
    bt.from_arrow(_table).group_by("k").agg(n=count()).collect(distributed=True, num_workers=2)
    assert before == _fleet._SESSION_LEASES


def test_a_scan_runs_behind_a_warm_fleet(_table):
    """The regression: a shuffle query leaves the fleet warm, and the plain-task query after
    it must still be able to schedule. This hung indefinitely."""
    ds = bt.from_arrow(_table)
    ds.group_by("k").agg(n=count()).collect(distributed=True, num_workers=2)
    scanned = ds.filter(col("v") > 90).select("k", "v").collect(distributed=True, num_workers=2)
    expected = ds.filter(col("v") > 90).select("k", "v").collect()
    assert scanned.num_rows == expected.num_rows
