"""Join reordering keeps one key per implied equality and sinks the key range it derives.

Plan-shape contracts the JOB workload exposed, each pinned with the rows it must not change:

* a region written with every pairwise equality on one key (JOB's `t.id = mi.movie_id AND
  mi.movie_id = mk.movie_id AND ...`) is rebuilt with **one** key pair per join, because the
  other pairs are implied by equalities a side already enforces;
* the key range `runtime_join_filter` derives sinks beneath the side's own filters when it is
  selective, so the cheap integer compare runs first, and is not attached when it would keep
  nearly every row;
* a keyless aggregate plans with the bounds of its filter and join columns.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher.config import active_config
from batcher.kyber.optimizer import optimize_logical
from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.rules.joins.order_search import _crossing_keys
from batcher.kyber.rules.joins.runtime_range import _place_key_range
from batcher.kyber.stats.estimator import StatsEstimator
from batcher.plan.expr_ir import Col, Lit
from batcher.plan.logical import Filter, Join, JoinOutputCol, Project, Projection, Scan
from batcher.plan.schema import SchemaRef
from batcher.plan.source_stats import SourceStatistics
from batcher.plan.stats import ColumnStat, Provenance
from batcher.plan.visitor import walk

_N = 4_000


def _movie_tables() -> dict[str, pa.Table]:
    """A title table and three per-movie fact tables, all keyed on the movie id."""
    ids = list(range(_N))
    return {
        "t": pa.table({"id": ids, "year": [1950 + i % 70 for i in ids]}),
        "mi": pa.table({"movie_id": [i % _N for i in range(3 * _N)], "info": ["x"] * (3 * _N)}),
        "mk": pa.table({"movie_id": [i % 500 for i in range(2 * _N)], "kw": list(range(2 * _N))}),
        "mc": pa.table({"movie_id": [i % 50 for i in range(_N)], "cid": list(range(_N))}),
    }


_CLIQUE = """
SELECT MIN(t.year) AS y, COUNT(*) AS n
FROM t, mi, mk, mc
WHERE t.id = mi.movie_id AND t.id = mk.movie_id AND t.id = mc.movie_id
  AND mi.movie_id = mk.movie_id AND mi.movie_id = mc.movie_id AND mk.movie_id = mc.movie_id
  AND t.year > 1990
"""


def _session() -> bt.Session:
    s = bt.Session()
    for name, table in _movie_tables().items():
        s.register(name, table)
    return s


def test_a_key_clique_is_rebuilt_with_one_key_per_join():
    ds = _session().sql(_CLIQUE)
    opt = optimize_logical(ds._plan, sources=ds._sources)
    joins = [n for n in walk(opt) if isinstance(n, Join) and n.join_type == "inner"]
    assert len(joins) == 3  # the four tables were reordered into three joins...
    # ...and every one of them joins on a single key: the other pairs are implied.
    assert [len(j.left_keys) for j in joins] == [1, 1, 1]


def test_crossing_keys_keeps_an_equality_only_one_side_cannot_imply():
    # Left holds leaves 0 and 1 joined on something else; both carry a column equal to the
    # right's `x`. Inside the left, `a` and `b` are NOT known equal, so both pairs stay.
    left = {(0, "a"): "a", (1, "b"): "b", (0, "k"): "k", (1, "k"): "k_r"}
    right = {(2, "x"): "x"}
    edges = [((0, "k"), (1, "k")), ((0, "a"), (2, "x")), ((1, "b"), (2, "x"))]
    assert _crossing_keys(left, right, edges) == (["a", "b"], ["x", "x"])
    # Once `a = b` is an edge inside the left, the second pair is implied and dropped.
    edges_eq = [((0, "a"), (1, "b")), ((0, "a"), (2, "x")), ((1, "b"), (2, "x"))]
    assert _crossing_keys(left, right, edges_eq) == (["a"], ["x"])
    # A repeated pair is the degenerate case of the same thing.
    assert _crossing_keys(left, right, [((0, "a"), (2, "x"))] * 2) == (["a"], ["x"])
    assert _crossing_keys(left, right, [((0, "k"), (1, "k"))]) == ([], [])


def _scan(sid: int, names: list[str]) -> Scan:
    fields = [pa.field(n, pa.string() if n == "s" else pa.int64()) for n in names]
    return Scan(sid, SchemaRef(pa.schema(fields)))


def _ctx(*stats: SourceStatistics) -> OptimizerContext:
    est = StatsEstimator([None] * len(stats), source_stats=list(stats))
    return OptimizerContext(
        config=active_config(), sources=[None] * len(stats), hub=None, estimator=est
    )


def _fact(lo: int = 0, hi: int = 1_000_000) -> SourceStatistics:
    return SourceStatistics(
        row_count=1_000_000,
        columns={"k": ColumnStat(min=lo, max=hi, ndv=hi - lo, provenance=Provenance.EXACT)},
    )


def _leaf() -> Project:
    """`Project(Filter(s LIKE ..., Scan))`, the shape a pruned, filtered reorder leaf has."""
    scan = _scan(0, ["k", "s"])
    filtered = Filter(scan, Col("s").str.contains("abc"))
    return Project(filtered, (Projection("key", Col("k")), Projection("s", Col("s"))))


def _place(side, pred):
    ctx = _ctx(_fact())
    return _place_key_range(side, pred, ctx.estimator.estimate(side), ctx)


def test_a_selective_key_range_sinks_beneath_the_leaf_filters():
    pred = (Col("key") >= Lit(10)) & (Col("key") <= Lit(20))
    out = _place(_leaf(), pred)
    # Project -> Filter(contains) -> Filter(range, phrased on the scan's `k`) -> Scan
    assert isinstance(out, Project)
    assert isinstance(out.input, Filter) and "contains" in str(out.input.predicate.to_ir())
    sunk = out.input.input
    assert isinstance(sunk, Filter) and isinstance(sunk.input, Scan)
    assert sunk.predicate.to_ir() == ((Col("k") >= Lit(10)) & (Col("k") <= Lit(20))).to_ir()


def test_a_moderately_selective_key_range_stays_on_top():
    # Keeps ~70%: worth filtering the join input, not worth a pass over every scanned row.
    pred = (Col("key") >= Lit(0)) & (Col("key") <= Lit(700_000))
    leaf = _leaf()
    out = _place(leaf, pred)
    assert isinstance(out, Filter) and out.input is leaf


def test_a_key_range_that_keeps_nearly_everything_is_not_attached():
    # One key short of the whole domain: `_narrows` accepts it, and it would cost a full pass
    # plus a compaction to remove almost nothing (JOB q13a's `movie_id >= 2`).
    pred = (Col("key") >= Lit(1)) & (Col("key") <= Lit(1_000_000))
    assert _place(_leaf(), pred) is None


def test_a_range_with_no_filter_to_pass_wraps_the_side():
    scan = _scan(0, ["k", "s"])
    pred = (Col("k") >= Lit(10)) & (Col("k") <= Lit(20))
    out = _place(scan, pred)
    assert isinstance(out, Filter) and out.input is scan


def test_a_range_does_not_pass_a_computed_key():
    scan = _scan(0, ["k", "s"])
    leaf = Project(
        Filter(scan, Col("s").str.contains("abc")), (Projection("key", Col("k") + Lit(1)),)
    )
    pred = (Col("key") >= Lit(10)) & (Col("key") <= Lit(20))
    out = _place(leaf, pred)
    assert isinstance(out, Filter) and out.input is leaf


def test_a_keyless_aggregate_plans_with_its_filter_and_join_bounds():
    from batcher.api.terminal.core import _keyless_aggregate_bound_columns

    ds = _session().sql(_CLIQUE)
    need = _keyless_aggregate_bound_columns(ds._plan)
    # The MIN's own column, plus every column a filter or a join of the plan reads: the
    # execution plans with these same statistics when the metadata answer misses.
    assert need is not None
    assert {"year", "id", "movie_id"} <= need


def test_a_key_frequency_table_a_filter_made_stale_does_not_zero_the_join():
    def scan(sid: int, names: list[str]) -> Scan:
        fields = [pa.field(n, pa.string() if n == "info" else pa.int64()) for n in names]
        return Scan(sid, SchemaRef(pa.schema(fields)))

    # `info_type`: 113 ids, a frequency table listing eight of them; filtered on `info` to a row.
    it = SourceStatistics(
        row_count=113,
        columns={
            "id": ColumnStat(
                min=1, max=113, ndv=113, mcv={str(v): 1 / 113 for v in range(106, 114)}
            ),
            "info": ColumnStat(ndv=113),
        },
    )
    # `movie_info_idx`: its type ids are three values the other table's list does not name.
    mi = SourceStatistics(
        row_count=806_365,
        columns={
            "info_type_id": ColumnStat(
                min=99, max=113, ndv=5, mcv={"99": 1 / 3, "100": 1 / 3, "101": 1 / 3}
            )
        },
    )
    left = Filter(scan(0, ["id", "info"]), Col("info") == Lit("rating"))
    out = (JoinOutputCol("left", "id", "id"), JoinOutputCol("right", "info_type_id", "t"))
    join = Join(left, scan(1, ["info_type_id"]), ("id",), ("info_type_id",), "inner", out)
    rows = StatsEstimator([None, None], source_stats=[it, mi]).estimate(join).rows
    # Containment: the surviving row's id meets ~1/5 of the 806,365 rows. The decomposition
    # over the stale table said zero (floored to one row).
    assert rows > 1_000
