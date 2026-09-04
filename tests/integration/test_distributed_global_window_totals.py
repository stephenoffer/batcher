"""Global window functions that need the relation's *total* — distributed, and equal.

`percent_rank` and `cume_dist` divide by the row count of the whole relation, `ntile` cuts
that same total into tiles, and `last_value` reads the partition's final row. None of those
is a number an ordered bucket knows, so the ordered-bucket algebra used to decline all four —
and a global window is not a `_split_at` pass-through, so declining did not mean "run it
slower", it meant `PlanError` on distributed data. ``ORDER BY <col>`` with a `percent_rank`
simply could not be distributed.

They are offsettable after all, one pass later. Each rides a helper the kernel computes
beside it — a running `rank`, a running row count, a running `row_number` — which the
ordinary per-bucket offset shifts during the walk; when the walk ends, the total row count is
just how many rows went past, and `OrderedBucketOffsets.finalize` closes the four out over
the assembled result. A driver that concatenates its buckets can do that (both distributed
ones); the single-node streaming driver yields as it goes and cannot, which is what
`supports_ordered_bucket_offsets(..., assembled=...)` distinguishes.

`var` and `stddev` are here too, for a different reason: they need no second pass but no
constant shift either, combining by Chan's formula over `(count, mean, M2)`.

The source is a real multi-file Parquet directory rather than an in-memory table on purpose.
`dist.executor._unsupported` runs an in-memory source on one node by design — correct, since
there is no distributed data — so a missing route over `bt.from_arrow` is indistinguishable
from the right answer. On a splittable source it raises, and the test would fail rather than
quietly pass.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from _ray_cluster import init_test_ray, shutdown_test_ray

pytestmark = pytest.mark.integration

pytest.importorskip("ray", reason="ray not installed")

_N = 400
_WORKERS = 2


@pytest.fixture(scope="module", autouse=True)
def _ray_session():
    started = init_test_ray(4)
    yield
    shutdown_test_ray(started)


@pytest.fixture(scope="module")
def splittable(cluster_scratch) -> str:
    """Three Parquet files of the same rows — genuinely splittable, with duplicate keys.

    The duplicates matter: `percent_rank` and `cume_dist` are defined over peer groups, and a
    key that never repeats would let a per-row approximation pass.
    """
    table = pa.table(
        {
            "rid": pa.array(range(_N), pa.int64()),
            "x": pa.array([(i * 37) % 91 for i in range(_N)], pa.int64()),
            "v": pa.array(
                [None if i % 13 == 0 else float(i % 17) for i in range(_N)], pa.float64()
            ),
        }
    )
    directory = cluster_scratch("global_window_totals")
    for part in range(3):
        pq.write_table(table, directory / f"p{part}.parquet")
    return str(directory)


#: One entry per function this route gained. `mixed` is not redundant: the four finalized
#: functions share one `finalize` pass and one helper namespace, so a window carrying several
#: at once is the case where a helper alias collision or a dropped column would show.
_CASES: dict[str, dict] = {
    "percent_rank": {"r": "percent_rank"},
    "cume_dist": {"r": "cume_dist"},
    "ntile_3": {"r": ("ntile", 3)},
    "ntile_7": {"r": ("ntile", 7)},
    "last_value": {"r": ("last_value", bt.col("v"))},
    "var": {"r": ("var", bt.col("v"))},
    "stddev": {"r": ("stddev", bt.col("v"))},
    "mixed": {
        "a": "row_number",
        "b": "percent_rank",
        "c": ("var", bt.col("v")),
        "d": ("last_value", bt.col("v")),
        "e": ("avg", bt.col("v")),
        "f": ("ntile", 4),
        "g": "cume_dist",
        "h": ("stddev", bt.col("v")),
    },
}


def _by_rid(table: pa.Table) -> dict:
    """`table` in `rid` order — a window result is an unordered relation, so sort it here."""
    return table.sort_by([("rid", "ascending")]).to_pydict()


@pytest.mark.parametrize("case", sorted(_CASES))
def test_the_distributed_answer_is_the_single_node_answer(splittable, case):
    functions = _CASES[case]
    windowed = bt.read.parquet(splittable).window(order_by=["x"], functions=functions)
    single = _by_rid(windowed.collect())
    distributed = _by_rid(
        bt.read.parquet(splittable)
        .window(order_by=["x"], functions=functions)
        .collect(distributed=True, num_workers=_WORKERS)
    )

    assert sorted(distributed) == sorted(single)
    for column in single:
        got, want = distributed[column], single[column]
        assert [x is None for x in got] == [x is None for x in want], column
        pairs = [(a, b) for a, b in zip(got, want, strict=True) if b is not None]
        if pairs and isinstance(pairs[0][1], float):
            # `var`/`stddev` are float reductions the two paths associate differently; the
            # rest are exact and pass this comparison at any tolerance.
            assert [a for a, _ in pairs] == pytest.approx([b for _, b in pairs], rel=1e-12), column
        else:
            assert [a for a, _ in pairs] == [b for _, b in pairs], column


def test_the_streaming_driver_still_declines_the_finalized_four(splittable):
    """The control. `finalize` is what makes these four correct, and the streaming driver
    yields each bucket before the total is known — so it must keep the materializing kernel.

    Without this, a change that dropped the `assembled` distinction would leave every
    assertion above green while `collect(spill=True)` returned a `percent_rank` divided by
    one bucket's row count: a number in `[0, 1]`, on the right rows, and wrong.
    """
    from batcher.dist.global_window import supports_ordered_bucket_offsets

    plan = bt.read.parquet(splittable).window(order_by=["x"], functions={"r": "percent_rank"})._plan
    assert supports_ordered_bucket_offsets(plan) is False
    assert supports_ordered_bucket_offsets(plan, assembled=True) is True
