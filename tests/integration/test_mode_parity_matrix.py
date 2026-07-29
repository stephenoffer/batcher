"""Every operator distributes over a SPLITTABLE source, and does so repeatably.

This is the distributed half of invariant #7 (single-node == distributed) applied as a
*matrix* rather than one test per operator, and it is deliberately built on a Parquet
source rather than `bt.from_arrow`. That distinction is the whole point: `_unsupported`
in `dist/executor.py` only refuses a shape when the plan reads a **splittable** source —
an in-memory source has no distributed data, so it legitimately runs on one node. A
parity test over an in-memory table therefore passes for any shape whose distributed path
is missing entirely, which is exactly how an operator ends up single-node-only without
anyone noticing.

Two properties are asserted per shape, because each catches a different class of bug:

1. The distributed result equals the single-node one. A missing path raises `PlanError`
   here instead of quietly falling back.
2. Running the *same* distributed query twice in one process gives the same valid result
   both times. Cross-query state (a reused worker fleet, a cached shuffle artifact,
   learned statistics changing the plan on the second run) is invisible to a
   single-execution test, and a nested-type column is where it shows up first — a
   structurally invalid `ListArray` passes `num_rows` and column-name checks and only
   aborts later, when something reads a value.

Both runs validate the Arrow result with `validate(full=True)`: a corrupt child array is
not visible in a value comparison that never reaches the bad offset.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from _harness import assert_tables_equal
from _ray_cluster import init_test_ray, shutdown_test_ray
from batcher import col, count

pytest.importorskip("ray", reason="ray not installed")
pytest.importorskip("batcher._native", reason="native engine not built")

WORKERS = 2
ROWS = 12_000


@pytest.fixture(scope="module", autouse=True)
def _ray_session():
    started = init_test_ray(4)
    yield
    shutdown_test_ray(started)


@pytest.fixture(scope="module")
def path(tmp_path_factory) -> str:
    """A multi-row-group Parquet file — a splittable source, so the dispatcher must
    either distribute a shape or refuse it, never silently run it on one node.

    Carries one column of every family the shuffle has to move intact: integer keys, a
    string, a float, a timestamp, and a **list**. The nested column is not decoration —
    it is the only one whose corruption is structural rather than value-level.
    """
    n = ROWS
    rng = np.random.default_rng(7)
    table = pa.table(
        {
            # A unique, source-ordered key, so every ordered shape (row_number, sort,
            # limit) has a total order and cannot differ merely by tie-breaking.
            "id": np.arange(n, dtype="int64"),
            "k": rng.integers(0, 16, n).astype("int64"),
            "g": pa.array([f"s{i % 5}" for i in range(n)]),
            "v": rng.integers(0, 100, n).astype("int64"),
            "w": rng.random(n),
            "ts": pa.array(np.arange(n, dtype="int64") * 60_000_000, type=pa.timestamp("us")),
            "lst": pa.array([[i % 3, i % 5] for i in range(n)]),
        }
    )
    out = str(tmp_path_factory.mktemp("mode_parity") / "t.parquet")
    pq.write_table(table, out, row_group_size=1_000)
    return out


#: One entry per relational operator reachable from the public API, as
#: `(id, builder, ordered)`. `ordered` selects a row-order-sensitive comparison, which an
#: order-independent one cannot make: a distributed sort that returns the right *set* of
#: rows in the wrong order is precisely the bug that hides behind a multiset compare.
SHAPES: list[tuple[str, object, bool]] = [
    ("scan_filter_project", lambda d: d.filter(col("v") > 50).select("k", "v", "lst"), False),
    ("limit", lambda d: d.filter(col("v") > 10).select("id", "v").limit(20), True),
    ("row_id", lambda d: d.select("id", "k", "lst").with_row_index("i"), True),
    ("sample_fraction", lambda d: d.sample(fraction=0.1, seed=3), False),
    ("unnest", lambda d: d.select("k", "lst").explode("lst"), False),
    ("unpivot", lambda d: d.select("k", "v", "w").unpivot(index=["k"], on=["v", "w"]), False),
    ("aggregate_grouped", lambda d: d.group_by("k").agg(s=col("v").sum(), n=count()), False),
    ("aggregate_global", lambda d: d.group_by().agg(s=col("v").sum(), n=count()), False),
    ("aggregate_expr_key", lambda d: d.group_by(b=col("v") % 5).agg(n=count()), False),
    ("distinct", lambda d: d.select("k", "g").distinct(), False),
    ("distinct_then_agg", lambda d: d.select("k").distinct().group_by().agg(n=count()), False),
    ("sort_column", lambda d: d.select("id", "k", "lst").sort("id"), True),
    (
        # The multiplier makes the computed key UNIQUE (`v < 100`, `id < ROWS`). A sort is
        # not stable across partitioning, so a shape with tied keys would differ between
        # the two modes purely in tie order — a false failure that says nothing about the
        # rewrite under test (`_hoist_computed_sort_key`).
        "sort_computed_key",
        lambda d: d.select("id", "k", "v").sort(col("v") * (ROWS * 10) + col("id")),
        True,
    ),
    ("sort_limit_topn", lambda d: d.select("id", "v").sort("id", descending=True).limit(10), True),
    (
        "window_partition_column",
        lambda d: d.window(
            partition_by=["k"], order_by=[("id", False)], functions={"rn": "row_number"}
        ),
        False,
    ),
    (
        "window_partition_computed",
        lambda d: d.window(
            partition_by=[col("v") % 4], order_by=[("id", False)], functions={"rn": "row_number"}
        ),
        False,
    ),
    ("window_global_unordered", lambda d: d.window(functions={"tot": ("sum", "v")}), False),
    ("union", lambda d: d.select("k", "lst").union(d.select("k", "lst")), False),
    ("union_distinct", lambda d: d.select("k").union(d.select("k"), distinct=True), False),
    (
        "join",
        lambda d: d.select("id", "k", "lst").join(d.select("id", "v"), on="id"),
        False,
    ),
]

IDS = [s[0] for s in SHAPES]


def _valid(table: pa.Table, label: str) -> pa.Table:
    """Assert `table` is structurally sound, not merely the right shape.

    `validate(full=True)` walks child arrays and offset buffers. Without it a mangled
    nested column reads as a perfectly good result until a consumer touches the bad
    offset — at which point Arrow aborts the process rather than raising, so the failure
    surfaces nowhere near the query that produced it.
    """
    try:
        table.validate(full=True)
    except Exception as exc:
        pytest.fail(f"{label} produced a structurally invalid Arrow table: {exc}")
    return table


@pytest.mark.integration
@pytest.mark.parametrize(("name", "build", "ordered"), SHAPES, ids=IDS)
def test_distributed_equals_single_node(path, name, build, ordered):
    """Each operator's distributed result equals its single-node result, or the shape
    raises rather than silently running the whole job on one node."""
    single = _valid(build(bt.read.parquet(path)).collect(), f"{name} single-node")
    dist = _valid(
        build(bt.read.parquet(path)).collect(distributed=True, num_workers=WORKERS),
        f"{name} distributed",
    )
    assert_tables_equal(dist, single, ordered=ordered)


@pytest.mark.integration
@pytest.mark.parametrize(("name", "build", "ordered"), SHAPES, ids=IDS)
def test_distributed_is_repeatable(path, name, build, ordered):
    """The same distributed query, run twice in one process, gives the same valid result.

    The second execution is the one that sees every piece of cross-query state the first
    left behind — a reused fleet, a shuffle scratch artifact, learned statistics that
    re-plan the query. Asserting only the first run's correctness cannot see any of it.
    """
    first = _valid(
        build(bt.read.parquet(path)).collect(distributed=True, num_workers=WORKERS),
        f"{name} distributed run 1",
    )
    second = _valid(
        build(bt.read.parquet(path)).collect(distributed=True, num_workers=WORKERS),
        f"{name} distributed run 2",
    )
    assert_tables_equal(second, first, ordered=ordered)
