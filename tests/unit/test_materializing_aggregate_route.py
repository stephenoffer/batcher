"""Which executor a grouped aggregate is routed to, and what the plan root may hide.

`prefer_materializing_aggregate` is Kyber's verdict that a join-free grouped aggregate is
cheaper on the materializing executor than on the streaming one. The engine re-checks the
shape itself (`bc_interp::materializing_aggregate_is_faster`) and ANDs in its own memory
test, so the two must agree about which operators sit *above* the aggregate — a hint the
engine's guard rejects is silently discarded, which is a routing bug that looks like nothing.

The peel used to admit a `Project` only. That left out `GROUP BY k ORDER BY <agg> DESC LIMIT
n` — the shape every leaderboard query in ClickBench has — so five of them stayed on the
streaming executor. Measured on the 1 M-row ClickBench mirror, engine rebuilt with only this
routing changed (best of five, milliseconds):

| query | before | after |
|---|---:|---:|
| `cb-q32` (`GROUP BY WatchID, ClientIP`) | 61.3 | **25.2** |
| `cb-q39` (five keys, filtered)          | 91.0 | **69.4** |
| `cb-q36` (`GROUP BY URL`, filtered)     | 29.3 | **28.5** |
| `cb-q33` (`GROUP BY URL`)               | 34.4 | 36.4 |
| `cb-q34` (`GROUP BY 1, URL`)            | 35.1 | 37.2 |

The two that lose are the unfiltered single wide-string key, by ~6%, against 36 ms and 22 ms
won on the two that gain.

The second half of the fix is that the group count is read off the **`Aggregate`** rather
than off the plan root: a `LIMIT 10` makes the root's estimate 10, which is under
`MATERIALIZE_AGG_MIN_GROUPS` (4,000), so estimating the root would have refused every query
the peel exists to admit while looking like it had asked the right question.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher import kyber

pytestmark = pytest.mark.unit

# The group count the verdict reads is an *estimate*, and on a cold in-memory source with no
# learned ndv it is the optimizer's flat `0.1 x rows` default (`provenance=default`) whatever
# the key's real cardinality is. So the fixtures below are sized by that: 50,000 rows estimate
# 5,000 groups and clear `MATERIALIZE_AGG_MIN_GROUPS` (4,000); 20,000 rows estimate 2,000 and
# do not. Choosing the row counts rather than the key cardinality is what makes these tests
# say something — a fixture picked by its real group count would pass or fail by coincidence.
ROWS = 50_000
ROWS_BELOW_FLOOR = 20_000
GROUPS = 20_000


def _grouped() -> bt.Dataset:
    rng = np.random.default_rng(0)
    table = pa.table(
        {
            "k": rng.integers(0, GROUPS, ROWS).astype("int64"),
            "v": rng.random(ROWS),
        }
    )
    return bt.from_arrow(table).group_by("k").agg(c=bt.col("v").count())


def _prefers(ds: bt.Dataset) -> bool:
    return kyber.optimize(ds._plan, sources=ds._sources).prefer_materializing_aggregate


def test_a_bare_grouped_aggregate_prefers_materializing():
    assert _prefers(_grouped())


@pytest.mark.parametrize(
    "wrap",
    [
        pytest.param(lambda g: g.select("k", "c"), id="project"),
        pytest.param(lambda g: g.sort("c", descending=True), id="sort"),
        pytest.param(lambda g: g.sort("c", descending=True).limit(10), id="top-n"),
        pytest.param(lambda g: g.limit(10), id="limit"),
        pytest.param(lambda g: g.select("k", "c").sort("c").limit(10), id="project-sort-limit"),
    ],
)
def test_the_operators_that_read_one_row_per_group_are_peeled(wrap):
    """Each of these consumes the aggregate's output, so none of them changes the verdict."""
    assert _prefers(wrap(_grouped()))


def test_a_limit_does_not_shrink_the_group_count_the_verdict_reads():
    """The regression this guards: `LIMIT 10` at the root, 20,000 groups underneath.

    Estimating the root gives 10 — below the 4,000-group floor — so a verdict taken there
    refuses the query. Estimating the `Aggregate` gives the group count, which is the number
    the floor was measured against.
    """
    ds = _grouped().sort("c", descending=True).limit(10)
    assert kyber.optimize(ds._plan, sources=ds._sources).prefer_materializing_aggregate
    # ... and the root really does estimate to the limit, so the test above is not vacuous.
    from batcher.kyber.cardinality import CardinalityEstimator

    est = CardinalityEstimator(ds._sources)
    assert est.estimate(ds._plan).rows <= 10


def test_a_global_aggregate_and_a_joined_one_are_still_refused():
    """The two shapes the verdict has always excluded, now re-checked under a top-N."""
    rng = np.random.default_rng(0)
    left = pa.table({"k": rng.integers(0, GROUPS, ROWS).astype("int64"), "v": rng.random(ROWS)})
    right = pa.table({"k": np.arange(GROUPS, dtype="int64")})

    glob = bt.from_arrow(left).agg(c=bt.col("v").count()).limit(10)
    assert not _prefers(glob)

    joined = (
        bt.from_arrow(left)
        .join(bt.from_arrow(right), on="k")
        .group_by("k")
        .agg(c=bt.col("v").count())
        .sort("c", descending=True)
        .limit(10)
    )
    assert not _prefers(joined)


def test_a_group_count_under_the_floor_is_still_left_on_the_streaming_executor():
    """The floor is the half the peel does not change, under a top-N as much as anywhere."""
    rng = np.random.default_rng(0)
    table = pa.table(
        {
            "k": rng.integers(0, 100, ROWS_BELOW_FLOOR).astype("int64"),
            "v": rng.random(ROWS_BELOW_FLOOR),
        }
    )
    small = bt.from_arrow(table).group_by("k").agg(c=bt.col("v").count())
    assert not _prefers(small)
    assert not _prefers(small.sort("c", descending=True).limit(10))
