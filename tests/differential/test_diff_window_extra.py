"""`window_extra` rewrites preserve results vs DuckDB.

Every rule runs end to end through the full optimizer and is compared with DuckDB's canonical
window form. The metadata-driven rules need *proven* statistics, so their input is a Parquet
file (EXACT footer min/max/null-count/row count) or a source declaring an EXACT unique key;
`_optimized` re-uses the conductor's own statistics collection, so "the rule fired" is an
assertion about the plan `.collect()` actually runs.

Coverage is what makes a window fragile: **ties** (checked with `rank`/`dense_rank`, whose
values are order-independent under ties, so the comparison cannot be fooled), NULLs in a
partition and in an order key, duplicate rows, a single-row relation, and an empty one.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher import col
from batcher.api.dataset import Dataset
from batcher.api.orchestration import collect_source_stats
from batcher.api.source_stats import column_bounds_needed
from batcher.io.source import InMemorySource
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.rules.extra import window_extra as _window_extra  # noqa: F401  (registers)
from batcher.plan.logical import Limit, Scan, Window
from batcher.plan.schema import SchemaRef
from batcher.plan.source_stats import SourceStatistics
from batcher.plan.stats import ColumnStat, Provenance
from batcher.plan.visitor import walk
from conftest import assert_same

# `id` unique, `dept` a partition with a NULL, `k` a non-null constant, `sal` with ties and a
# NULL — every edge a window rule can trip over.
_W = pa.table(
    {
        "id": [1, 2, 3, 4, 5, 6, 7],
        "dept": ["a", "a", "a", "b", "b", None, None],
        "k": [7, 7, 7, 7, 7, 7, 7],
        "sal": [100, 300, 200, 150, 250, 250, None],
    }
)
# Distinct salaries per partition — safe for `row_number`, whose tie-breaking is unspecified.
_NOTIES = pa.table(
    {
        "id": [1, 2, 3, 4, 5, 6],
        "dept": ["a", "a", "a", "b", "b", "b"],
        "k": [7, 7, 7, 7, 7, 7],
        "sal": [100, 300, 200, 150, 250, 270],
    }
)
_ONE = _NOTIES.slice(0, 1)
_EMPTY = _NOTIES.slice(0, 0)
_TABLES = {"w": _W, "nt": _NOTIES, "one": _ONE, "e": _EMPTY}


class _KeyedSource(InMemorySource):
    """An in-memory source declaring an EXACT unique key (a catalog primary key).

    `resident = False` so the conductor takes the full `statistics()` path — the resident fast
    path reports only a row count and the bounds a predicate needs, so the declaration would
    never reach the optimizer.
    """

    __slots__ = ("_key",)
    resident = False

    def __init__(self, batches, key: str) -> None:
        super().__init__(batches)
        self._key = key

    def statistics(self) -> SourceStatistics:
        base = InMemorySource.statistics(self)
        columns = dict(base.columns)
        columns[self._key] = ColumnStat(
            null_count=0, ndv=self.row_count(), provenance=Provenance.EXACT
        )
        return SourceStatistics(row_count=self.row_count(), columns=columns)


@pytest.fixture(scope="module")
def pq_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("window_extra")
    for name, table in _TABLES.items():
        pq.write_table(table, str(path / f"{name}.parquet"))
    return path


@pytest.fixture(autouse=True)
def _register(duck):
    for name, table in _TABLES.items():
        duck.register(name, table)


@pytest.fixture
def w(pq_dir):
    return bt.read.parquet(str(pq_dir / "w.parquet"))


@pytest.fixture
def nt(pq_dir):
    return bt.read.parquet(str(pq_dir / "nt.parquet"))


@pytest.fixture
def one(pq_dir):
    return bt.read.parquet(str(pq_dir / "one.parquet"))


@pytest.fixture
def empty(pq_dir):
    return bt.read.parquet(str(pq_dir / "e.parquet"))


def _keyed(table, key="id"):
    src = _KeyedSource(table.to_batches(), key)
    return Dataset(Scan(source_id=0, schema=SchemaRef.from_arrow(src.schema())), [src])


def _optimized(ds):
    """The optimized plan — over exactly the statistics the conductor collects for it."""
    stats = collect_source_stats(ds._sources, None, need_columns=column_bounds_needed(ds._plan))
    return Optimizer(sources=ds._sources, source_stats=stats).logical_rewrite(ds._plan)


def _windows(ds) -> list[Window]:
    return [n for n in walk(_optimized(ds)) if isinstance(n, Window)]


# --- drop_partition_key_proven_constant ----------------------------------------


def test_constant_partition_key_dropped(duck, w):
    ds = w.window(partition_by=["k", "dept"], order_by=["sal"], functions={"r": "rank"})
    assert [len(x.partition_keys) for x in _windows(ds)] == [1]  # `k` never splits a partition
    assert_same(
        ds.collect(),
        duck.sql("SELECT *, rank() OVER (PARTITION BY k, dept ORDER BY sal) AS r FROM w"),
    )


def test_constant_partition_key_dropped_with_ties_and_nulls(duck, w):
    # Ties (sal = 250) and a NULL partition (dept IS NULL) and a NULL order key (sal IS NULL).
    ds = w.window(partition_by=["k", "dept"], order_by=["sal"], functions={"d": "dense_rank"})
    assert_same(
        ds.collect(),
        duck.sql("SELECT *, dense_rank() OVER (PARTITION BY k, dept ORDER BY sal) AS d FROM w"),
    )


def test_non_constant_partition_key_is_kept(duck, w):
    ds = w.window(partition_by=["dept", "sal"], order_by=["id"], functions={"r": "rank"})
    assert [len(x.partition_keys) for x in _windows(ds)] == [2]
    assert_same(
        ds.collect(),
        duck.sql("SELECT *, rank() OVER (PARTITION BY dept, sal ORDER BY id) AS r FROM w"),
    )


# --- drop_order_key_proven_constant / drop_order_key_equal_to_partition_key -----


def test_constant_order_key_dropped(duck, w):
    ds = w.window(partition_by=["dept"], order_by=["k", "sal"], functions={"r": "rank"})
    assert [[k.expr.name for k in x.order_keys] for x in _windows(ds)] == [["sal"]]
    assert_same(
        ds.collect(),
        duck.sql("SELECT *, rank() OVER (PARTITION BY dept ORDER BY k, sal) AS r FROM w"),
    )


def test_order_key_equal_to_partition_key_dropped(duck, w):
    ds = w.window(partition_by=["dept"], order_by=["dept", "sal"], functions={"r": "rank"})
    assert [[k.expr.name for k in x.order_keys] for x in _windows(ds)] == [["sal"]]
    assert_same(
        ds.collect(),
        duck.sql("SELECT *, rank() OVER (PARTITION BY dept ORDER BY dept, sal) AS r FROM w"),
    )


def test_lone_constant_order_key_is_kept(duck, w):
    # Dropping it would leave a ranking function with no ORDER BY — every row would tie.
    ds = w.window(partition_by=["dept"], order_by=["k"], functions={"r": "rank"})
    assert [[k.expr.name for k in x.order_keys] for x in _windows(ds)] == [["k"]]
    assert_same(
        ds.collect(), duck.sql("SELECT *, rank() OVER (PARTITION BY dept ORDER BY k) AS r FROM w")
    )


# --- drop_order_key_after_unique_key_in_window_order ----------------------------


def test_order_keys_after_a_unique_key_dropped(duck):
    ds = _keyed(_W).window(order_by=["id", "sal"], functions={"rn": "row_number"})
    assert [[k.expr.name for k in x.order_keys] for x in _windows(ds)] == [["id"]]
    assert_same(
        ds.collect(), duck.sql("SELECT *, row_number() OVER (ORDER BY id, sal) AS rn FROM w")
    )


def test_order_keys_after_a_non_unique_key_are_kept(duck, w):
    # `dept` is not unique, so `sal` still breaks its ties — both keys must survive.
    ds = w.window(order_by=["dept", "sal"], functions={"r": "rank"})
    assert [len(x.order_keys) for x in _windows(ds)] == [2]
    assert_same(ds.collect(), duck.sql("SELECT *, rank() OVER (ORDER BY dept, sal) AS r FROM w"))


# --- frames ---------------------------------------------------------------------


def test_unbounded_frame_without_order_dropped(duck, w):
    ds = w.window(partition_by=["dept"], functions={"s": ("sum", "sal")}, frame=(None, None))
    assert [fn.frame for x in _windows(ds) for fn in x.functions] == [None]
    assert_same(
        ds.collect(),
        duck.sql(
            "SELECT *, sum(sal) OVER (PARTITION BY dept ROWS BETWEEN UNBOUNDED PRECEDING "
            "AND UNBOUNDED FOLLOWING) AS s FROM w"
        ),
    )


def test_order_keys_dropped_under_unbounded_frames(duck, w):
    ds = w.window(
        partition_by=["dept"], order_by=["sal"], functions={"s": ("sum", "sal")}, frame=(None, None)
    )
    assert [x.order_keys for x in _windows(ds)] == [()]  # the sort is unobservable → dead
    assert_same(
        ds.collect(),
        duck.sql(
            "SELECT *, sum(sal) OVER (PARTITION BY dept ORDER BY sal ROWS BETWEEN UNBOUNDED "
            "PRECEDING AND UNBOUNDED FOLLOWING) AS s FROM w"
        ),
    )


def test_running_frame_keeps_its_order(duck, w):
    ds = w.window(
        partition_by=["dept"], order_by=["sal"], functions={"s": ("sum", "sal")}, frame=(None, 0)
    )
    assert [len(x.order_keys) for x in _windows(ds)] == [1]  # a running frame reads the order
    assert_same(
        ds.collect(),
        duck.sql(
            "SELECT *, sum(sal) OVER (PARTITION BY dept ORDER BY sal ROWS BETWEEN UNBOUNDED "
            "PRECEDING AND CURRENT ROW) AS s FROM w"
        ),
    )


# --- dedupe_window_functions -----------------------------------------------------


def test_duplicate_window_functions_computed_once(duck, w):
    ds = w.window(
        partition_by=["dept"],
        order_by=["sal"],
        functions={"a": ("sum", "sal"), "b": ("sum", "sal"), "c": ("min", "sal")},
    )
    assert [len(x.functions) for x in _windows(ds)] == [2]  # `b` re-derived from `a`
    assert_same(
        ds.collect(),
        duck.sql(
            "SELECT *, sum(sal) OVER p AS a, sum(sal) OVER p AS b, min(sal) OVER p AS c "
            "FROM w WINDOW p AS (PARTITION BY dept ORDER BY sal)"
        ),
    )


# --- constant_window_function_folding ---------------------------------------------


def test_window_min_of_constant_column_folded(duck, w):
    ds = w.window(partition_by=["dept"], order_by=["sal"], functions={"lo": ("min", "k")})
    assert not _windows(ds)  # the only function folds → the whole window disappears
    assert_same(
        ds.collect(),
        duck.sql("SELECT *, min(k) OVER (PARTITION BY dept ORDER BY sal) AS lo FROM w"),
    )


def test_window_count_of_constant_column_is_untouched(duck, w):
    # COUNT depends on how many rows the (running) frame holds — not only on the value.
    ds = w.window(partition_by=["dept"], order_by=["sal"], functions={"n": ("count", "k")})
    assert len(_windows(ds)) == 1
    assert_same(
        ds.collect(),
        duck.sql("SELECT *, count(k) OVER (PARTITION BY dept ORDER BY sal) AS n FROM w"),
    )


def test_window_min_of_constant_column_with_explicit_frame_is_untouched(duck, w):
    # A following-only frame can be *empty* at the last rows, where MIN is NULL, not 7.
    ds = w.window(
        partition_by=["dept"], order_by=["sal"], functions={"lo": ("min", "k")}, frame=(1, 2)
    )
    assert len(_windows(ds)) == 1
    assert_same(
        ds.collect(),
        duck.sql(
            "SELECT *, min(k) OVER (PARTITION BY dept ORDER BY sal ROWS BETWEEN 1 FOLLOWING "
            "AND 2 FOLLOWING) AS lo FROM w"
        ),
    )


# --- simplify_window_over_single_row_partition -------------------------------------


def test_window_over_single_row_input(duck, one):
    ds = one.window(
        partition_by=["dept"],
        order_by=["sal"],
        functions={"rn": "row_number", "r": "rank", "lo": ("min", "sal"), "n": ("count", "sal")},
    )
    assert not _windows(ds)  # ≤ 1 row → no partition to build, no sort to run
    assert_same(
        ds.collect(),
        duck.sql(
            "SELECT *, row_number() OVER p AS rn, rank() OVER p AS r, min(sal) OVER p AS lo, "
            "count(sal) OVER p AS n FROM one WINDOW p AS (PARTITION BY dept ORDER BY sal)"
        ),
    )


def test_window_over_empty_input(duck, empty):
    ds = empty.window(partition_by=["dept"], order_by=["sal"], functions={"rn": "row_number"})
    assert_same(
        ds.collect(),
        duck.sql("SELECT *, row_number() OVER (PARTITION BY dept ORDER BY sal) AS rn FROM e"),
    )


def test_window_over_many_rows_is_untouched(duck, nt):
    ds = nt.window(partition_by=["dept"], order_by=["sal"], functions={"rn": "row_number"})
    assert len(_windows(ds)) == 1
    assert_same(
        ds.collect(),
        duck.sql("SELECT *, row_number() OVER (PARTITION BY dept ORDER BY sal) AS rn FROM nt"),
    )


# --- rank_limit_zero_to_empty --------------------------------------------------------


def test_qualify_rank_below_one_is_empty(duck, nt):
    ds = nt.window(partition_by=["dept"], order_by=["sal"], functions={"rn": "row_number"}).filter(
        col("rn") < 1
    )
    # `QUALIFY rn < 1` fuses to a per-partition top-0: the window is run over the empty
    # relation rather than ranking every row and then discarding it.
    assert any(isinstance(n, Limit) and n.n == 0 for n in walk(_optimized(ds)))
    assert_same(
        ds.collect(),
        duck.sql(
            "SELECT * FROM (SELECT *, row_number() OVER (PARTITION BY dept ORDER BY sal) AS rn "
            "FROM nt) WHERE rn < 1"
        ),
    )


def test_qualify_rank_one_still_returns_the_top_row(duck, nt):
    ds = nt.window(partition_by=["dept"], order_by=["sal"], functions={"rn": "row_number"}).filter(
        col("rn") <= 1
    )
    assert_same(
        ds.collect(),
        duck.sql(
            "SELECT * FROM (SELECT *, row_number() OVER (PARTITION BY dept ORDER BY sal) AS rn "
            "FROM nt) WHERE rn <= 1"
        ),
    )
