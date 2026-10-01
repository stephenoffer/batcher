"""Differential tests for windows whose ORDER BY is a *composite* key, against DuckDB.

`PARTITION BY s ORDER BY a, b` used to reach the window's row-encoded sort, whose inline
prefix is the partition key and so settled almost no comparison. It now sorts one packed
`u64` per row when the key tuple's measured ranges fit (`bc_runtime::window::packed_order`),
with the partition as a measured range or a dense group id. These cases hold that path to
DuckDB on what it encodes by hand: each key's direction and null placement, a nullable and a
narrow (`INT32`, `DATE`) key, a string partition, ties on the leading order key, and a fixture
large enough that the window runs bucket-parallel (> 65,536 rows).

They also pin the change that a `ROWS` frame no longer builds the peer encoding: a moving
aggregate is compared beside a `RANGE` one and a `rank`, which still need it.

Every query here returns one value per row with no outermost ORDER BY, so the multiset
comparison is the right one; `row_number` orders by a unique trailing key so its answer is
defined.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential

N = 120_000


def _table() -> pa.Table:
    """Composite-order fixture: a string partition, nullable/narrow/temporal order keys."""
    base = dt.date(2020, 1, 1)
    return pa.table(
        {
            "id": pa.array(range(N), pa.int64()),
            "s": pa.array([None if i % 211 == 0 else f"part-{(i * 37) % 97}" for i in range(N)]),
            "p": pa.array([(i * 7919) % 400 for i in range(N)], pa.int64()),
            "a": pa.array(
                [None if i % 53 == 0 else (i * 31) % 50 - 25 for i in range(N)], pa.int32()
            ),
            "d": pa.array([base + dt.timedelta(days=(i * 13) % 900) for i in range(N)]),
            "v": pa.array([((i * 104_729) % 10_000) / 8.0 - 300.0 for i in range(N)]),
        }
    )


def _run(duck, sql: str) -> None:
    t = _table()
    duck.register("t", t)
    session = bt.Session()
    session.register("t", t)
    assert_same(session.sql(sql).collect(), duck.sql(sql))


@pytest.mark.parametrize(
    "a_dir", ["ASC NULLS LAST", "ASC NULLS FIRST", "DESC NULLS LAST", "DESC NULLS FIRST"]
)
@pytest.mark.parametrize("d_dir", ["ASC", "DESC"])
def test_moving_average_over_a_composite_order(duck, a_dir, d_dir):
    """A `ROWS` frame over `(a, d, id)`: the frame is physical, so the order must be exact."""
    _run(
        duck,
        "SELECT id, avg(v) OVER (PARTITION BY p ORDER BY a "
        f"{a_dir}, d {d_dir}, id ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) AS m FROM t",
    )


@pytest.mark.parametrize("a_dir", ["ASC", "DESC NULLS FIRST"])
def test_string_partition_and_peer_functions(duck, a_dir):
    """A string partition takes the grouper's ids; `rank` and `RANGE` still read peers."""
    _run(
        duck,
        "SELECT id, "
        f"rank() OVER (PARTITION BY s ORDER BY a {a_dir}, d) AS r, "
        f"dense_rank() OVER (PARTITION BY s ORDER BY a {a_dir}, d) AS dr, "
        f"sum(v) OVER (PARTITION BY s ORDER BY a {a_dir}, d "
        "RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS rs, "
        f"row_number() OVER (PARTITION BY s ORDER BY a {a_dir}, d, id) AS rn, "
        f"lag(v) OVER (PARTITION BY s ORDER BY a {a_dir}, d, id) AS lg "
        "FROM t",
    )


def test_rows_frames_of_every_aggregate_and_two_partition_keys(duck):
    """`min`/`max`/`count`/`sum` over a centred `ROWS` frame, partitioned by two keys."""
    _run(
        duck,
        "SELECT id, "
        "min(v) OVER w AS lo, max(v) OVER w AS hi, count(a) OVER w AS n, sum(a) OVER w AS sa "
        "FROM t WINDOW w AS (PARTITION BY p, s ORDER BY d DESC, a, id "
        "ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING)",
    )


def test_unpartitioned_composite_order(duck):
    """No PARTITION BY: one partition, the packed word is the order keys alone."""
    _run(
        duck,
        "SELECT id, sum(v) OVER (ORDER BY d, a DESC NULLS FIRST, id "
        "ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS m FROM t",
    )


def test_float_order_key_signed_zero_and_nan(duck):
    """A float order key with `-0.0`/`0.0` (peers) and NaN (last), which packs only alone."""
    t = pa.table(
        {
            "id": pa.array(range(N), pa.int64()),
            "f": pa.array(
                [
                    None
                    if i % 17 == 0
                    else (-0.0 if i % 5 == 0 else (float("nan") if i % 23 == 0 else (i % 9) - 4.0))
                    for i in range(N)
                ]
            ),
            "v": pa.array([float(i % 101) for i in range(N)]),
        }
    )
    duck.register("t", t)
    session = bt.Session()
    session.register("t", t)
    for sql in (
        "SELECT id, rank() OVER (ORDER BY f DESC NULLS FIRST) AS r FROM t",
        "SELECT id, sum(v) OVER (ORDER BY f) AS rs FROM t",
    ):
        assert_same(session.sql(sql).collect(), duck.sql(sql))
