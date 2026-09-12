"""Plan-level common-subplan elimination returns what the plan as written returns.

The rewrite executes a repeated subtree once and reads the result back through a scan, so
what it must prove is that the answer never moves: against DuckDB, and against the same
query with the rewrite switched off — which is the sharper of the two, since it isolates
the rewrite from every other difference between the two engines.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col, count
from batcher.config import active_config, set_config


@pytest.fixture
def cse_off():
    """Run the block with the rewrite disabled, for the A/B half of each test."""
    import contextlib

    @contextlib.contextmanager
    def _off():
        cfg = active_config()
        set_config(
            cfg.replace(optimizer=dataclasses.replace(cfg.optimizer, common_subplan_max_bytes=0))
        )
        try:
            yield
        finally:
            set_config(cfg)

    return _off


@pytest.fixture
def fact(duck):
    rng = np.random.default_rng(7)
    t = pa.table(
        {
            "k": rng.integers(0, 50, 20_000).astype("int64"),
            "g": rng.integers(0, 4, 20_000).astype("int64"),
            "v": rng.integers(0, 100, 20_000).astype("int64"),
        }
    )
    duck.register("f", t)
    return t


def _rows(t):
    return sorted(tuple(r.values()) for r in t.to_pylist())


def _shared_agg_join(fact):
    """`agg ⋈ project(filter(agg))` — one `GROUP BY` feeding both operands."""
    agg = bt.from_arrow(fact).group_by("k").agg(total=col("v").sum(), n=count())
    hot = agg.filter(col("total") > 20_000).select(col("k").alias("hk"))
    return agg.join(hot, left_on="k", right_on="hk", how="inner").select("k", "total", "n")


def test_shared_aggregate_matches_duckdb(duck, fact):
    assert_same(
        _shared_agg_join(fact).collect(),
        duck.sql(
            "WITH a AS (SELECT k, SUM(v) total, COUNT(*) n FROM f GROUP BY k) "
            "SELECT a.k, a.total, a.n FROM a JOIN "
            "(SELECT k hk FROM a WHERE total > 20000) h ON a.k = h.hk"
        ),
    )


def test_shared_aggregate_matches_the_unrewritten_plan(fact, cse_off):
    with cse_off():
        want = _rows(_shared_agg_join(fact).collect())
    assert _rows(_shared_agg_join(fact).collect()) == want


def test_the_rewrite_actually_fired(fact):
    """Guard against the tests above passing because nothing happened.

    Every other assertion here is satisfied by a rewrite that declines, so without this one
    the suite would stay green if the analysis silently stopped finding anything — which is
    exactly how it behaved before source ids were canonicalized.
    """
    from batcher import core
    from batcher.api.subplan_reuse import reuse_common_subplans

    q = _shared_agg_join(fact)
    ctx = core.ExecutionContext(columns=q.columns, hub=core.default_hub())
    plan, sources = reuse_common_subplans(q._plan, q._sources, ctx)
    assert len(sources) > len(q._sources), "no subplan was materialized"
    assert plan is not q._plan


def test_a_shared_subplan_under_three_appearances(duck, fact):
    """Three references to one derived relation, the SQL `WITH`-used-thrice shape."""
    agg = bt.from_arrow(fact).group_by("k").agg(total=col("v").sum())
    a = agg.filter(col("total") > 20_000).select(col("k").alias("a"))
    b = agg.filter(col("total") > 21_000).select(col("k").alias("b"))
    q = agg.join(a, left_on="k", right_on="a").join(b, left_on="k", right_on="b").select("k")
    assert_same(
        q.collect(),
        duck.sql(
            "WITH agg AS (SELECT k, SUM(v) total FROM f GROUP BY k) "
            "SELECT agg.k FROM agg "
            "JOIN (SELECT k a FROM agg WHERE total > 20000) x ON agg.k = x.a "
            "JOIN (SELECT k b FROM agg WHERE total > 21000) y ON agg.k = y.b"
        ),
    )


def test_a_shared_subplan_with_nulls_and_an_empty_result(duck):
    """Nulls in the group key, and a predicate that keeps nothing."""
    t = pa.table({"k": [1, None, 2, None, 2], "v": [1, 2, 3, 4, 5]})
    duck.register("n", t)
    agg = bt.from_arrow(t).group_by("k").agg(total=col("v").sum())
    hot = agg.filter(col("total") > 1_000).select(col("k").alias("hk"))
    q = agg.join(hot, left_on="k", right_on="hk", how="inner").select("k", "total")
    assert q.collect().num_rows == 0
    assert_same(
        q.collect(),
        duck.sql(
            "WITH a AS (SELECT k, SUM(v) total FROM n GROUP BY k) "
            "SELECT a.k, a.total FROM a JOIN (SELECT k hk FROM a WHERE total > 1000) h "
            "ON a.k = h.hk"
        ),
    )


def test_a_shared_sort_is_reused(duck, fact):
    """The candidate need not be an aggregate — any breaker qualifies."""
    ordered = bt.from_arrow(fact).sort("v").limit(500)
    q = ordered.join(ordered.select(col("k").alias("hk")), left_on="k", right_on="hk").select("k")
    with_cse = _rows(q.collect())
    cfg = active_config()
    set_config(
        cfg.replace(optimizer=dataclasses.replace(cfg.optimizer, common_subplan_max_bytes=0))
    )
    try:
        assert _rows(q.collect()) == with_cse
    finally:
        set_config(cfg)


def test_a_map_batches_subtree_is_never_collapsed(fact):
    """Opaque user code may be non-deterministic or have side effects: it must run per use."""
    calls = []

    def tag(batch):
        calls.append(1)
        return batch

    mapped = bt.from_arrow(fact).map_batches(tag)
    agg = mapped.group_by("k").agg(total=col("v").sum())
    q = agg.join(agg.select(col("k").alias("hk")), left_on="k", right_on="hk").select("k")
    q.collect()
    assert calls, "the UDF never ran"


def test_two_scans_of_one_source_are_filtered_independently(tmp_path, cse_off):
    """A source read by two differently-filtered scans is not pre-filtered by one of them.

    Source-side predicate pushdown is recorded per `source_id`, but a predicate belongs to a
    *scan*. When the reuse pass points both bindings of one source at a single index — which
    is exactly what makes a repeated subplan visible — two scans share that key, and the last
    one recorded used to overwrite the first. The source was then read pre-filtered by one
    branch's predicate, and the rows the other branch needed were gone before any operator
    ran. The `Filter` the engine keeps above the scan cannot restore them.

    Spelled as a semi join unioned with an anti join over one table, because that is the
    shape `MERGE INTO` composes and it is what the bug cost there: the anti branch is the
    unmatched target rows, so an upsert silently dropped every row the change set did not
    mention.
    """
    path = str(tmp_path / "t.parquet")
    bt.from_arrow(pa.table({"id": [1, 2, 3, 4], "v": [10, 20, 30, 40]})).write.parquet(path)

    def build():
        t = bt.read.parquet(path)
        keys = bt.from_arrow(pa.table({"id": [2, 3]})).distinct()
        semi = t.join(keys, left_on="id", right_on="id", how="semi").select("id", "v")
        anti = t.join(keys, left_on="id", right_on="id", how="anti").select("id", "v")
        return semi.union(anti)

    with_cse = _rows(build().collect())
    with cse_off():
        without_cse = _rows(build().collect())

    # Every row survives exactly once: the two branches partition the table.
    assert with_cse == [(1, 10), (2, 20), (3, 30), (4, 40)]
    assert with_cse == without_cse


def test_a_shared_source_under_disjoint_filters_keeps_both_halves(tmp_path, cse_off):
    """The same collision with plain filters rather than joins, and no reuse pass involved.

    `_one_id_per_source` is one way two scans come to share a key; it is not the only one, so
    the property is pinned on the shape itself.
    """
    path = str(tmp_path / "t.parquet")
    bt.from_arrow(pa.table({"id": [1, 2, 3, 4], "v": [10, 20, 30, 40]})).write.parquet(path)

    def build():
        t = bt.read.parquet(path)
        return t.filter(col("id") <= 2).union(t.filter(col("id") >= 3))

    assert _rows(build().collect()) == [(1, 10), (2, 20), (3, 30), (4, 40)]
    with cse_off():
        assert _rows(build().collect()) == [(1, 10), (2, 20), (3, 30), (4, 40)]


def _sess_with_star_schema():
    """Four tables joined on one key, the shape a `WHERE` above a join tree is written in."""
    rng = np.random.default_rng(7)
    n = 4000
    sess = bt.Session()
    sess.register(
        "f",
        bt.from_arrow(pa.table({"k": rng.integers(0, 400, n), "v": rng.integers(0, 100, n)})),
    )
    for name in ("d1", "d2", "d3"):
        sess.register(
            "" + name,
            bt.from_arrow(pa.table({"k": np.arange(400), name: np.arange(400) % 11})),
        )
    return sess


_SHARED = (
    "SELECT f.k AS k, f.v AS v FROM f, d1, d2, d3 "
    "WHERE f.k = d1.k AND f.k = d2.k AND f.k = d3.k AND d1.d1 < 3"
)
_REPEATED = (
    f"SELECT l.k AS k, sum(l.v) AS total FROM ({_SHARED}) l, ({_SHARED}) r "
    "WHERE l.k = r.k GROUP BY l.k"
)


def test_a_predicate_above_a_join_tree_is_reused_and_still_matches_duckdb(duck):
    """The shape the size gate used to refuse, because it read the unpushed estimate.

    Every join is keyed here; what the optimizer moves is the predicate. Before the gates
    read the optimized form, the repeated subtree estimated orders of magnitude too large
    and the rewrite declined.
    """
    sess = _sess_with_star_schema()
    for name in ("f", "d1", "d2", "d3"):
        duck.register(name, sess.sql(f"SELECT * FROM {name}").collect())
    assert_same(sess.sql(_REPEATED).collect(), duck.sql(_REPEATED))


def test_a_predicate_above_a_join_tree_matches_the_unrewritten_plan(cse_off):
    sess = _sess_with_star_schema()
    with cse_off():
        want = _rows(sess.sql(_REPEATED).collect())
    assert _rows(sess.sql(_REPEATED).collect()) == want
    assert want, "the fixture must return rows, or neither side proves anything"


def test_the_rewrite_fires_on_a_predicate_above_a_join_tree():
    """The positive control for the two above, which a declining rewrite would satisfy."""
    from batcher import core
    from batcher.api.subplan_reuse import reuse_common_subplans

    q = _sess_with_star_schema().sql(_REPEATED)
    ctx = core.ExecutionContext(columns=q.columns, hub=core.default_hub())
    rewritten, sources = reuse_common_subplans(q._plan, list(q._sources), ctx)
    assert len(sources) > len(q._sources), "no subplan was materialized"
    assert rewritten is not q._plan, "the plan must be rewritten to read the materialization"


def test_a_rollup_over_a_join_matches_duckdb(duck):
    """ROLLUP lowers to one GROUP BY per level over the same input, so the input subtree is
    repeated once per level -- the shape that makes this rewrite worth the most."""
    sess = _sess_with_star_schema()
    for name in ("f", "d1", "d2", "d3"):
        duck.register(name, sess.sql(f"SELECT * FROM {name}").collect())
    sql = (
        "SELECT d1.d1 AS a, d2.d2 AS b, sum(f.v) AS total "
        "FROM f, d1, d2 WHERE f.k = d1.k AND f.k = d2.k "
        "GROUP BY ROLLUP(d1.d1, d2.d2)"
    )
    assert_same(sess.sql(sql).collect(), duck.sql(sql))


def test_a_materialized_subtree_carries_only_the_columns_the_plan_reads(duck):
    """Materializing forfeits the fusion each appearance had with its parent, and what that
    costs is a width: embedded, a subtree's consumers prune it; standalone there is no
    consumer to prune it. Measured on TPC-DS q18 the chosen subtree carried 133 columns
    where the plan read 11, and on q86 84 where it read 3 -- and those were exactly the two
    queries reuse made slower.

    The answer must not move, which is the half that matters: dropping a column the plan
    still needs is a wrong result, not a slow one.
    """
    from batcher import core
    from batcher.api.subplan_reuse import reuse_common_subplans

    sess = _sess_with_star_schema()
    q = sess.sql(_REPEATED)
    ctx = core.ExecutionContext(columns=q.columns, hub=core.default_hub())
    _, sources = reuse_common_subplans(q._plan, list(q._sources), ctx)
    assert len(sources) > len(q._sources), "no subplan was materialized"
    materialized = sources[len(q._sources) :][0]
    carried = set(materialized.schema().names)
    # The shared SELECT projects k and v; the enclosing query reads both, so nothing is
    # dropped here -- what this pins is that narrowing never drops a column that is read.
    assert {"k", "v"} <= carried, f"a needed column was dropped; kept {sorted(carried)}"

    for name in ("f", "d1", "d2", "d3"):
        duck.register(name, sess.sql(f"SELECT * FROM {name}").collect())
    assert_same(sess.sql(_REPEATED).collect(), duck.sql(_REPEATED))


def _sess_with_wide_star_schema():
    """The same star schema with several columns per table, so width is visible."""
    rng = np.random.default_rng(7)
    n = 4000
    sess = bt.Session()
    sess.register(
        "f",
        bt.from_arrow(
            pa.table(
                {
                    "k": rng.integers(0, 400, n),
                    "v": rng.integers(0, 100, n),
                    "w": rng.integers(0, 100, n),
                    "x": rng.integers(0, 100, n),
                }
            )
        ),
    )
    for name in ("d1", "d2", "d3"):
        sess.register(
            name,
            bt.from_arrow(
                pa.table(
                    {
                        "k": np.arange(400),
                        name: np.arange(400) % 11,
                        name + "b": np.arange(400) % 7,
                    }
                )
            ),
        )
    return sess


_WIDE_ROLLUP = (
    "SELECT d1.d1 AS a, d2.d2 AS b, sum(f.v) AS total "
    "FROM f, d1, d2, d3 WHERE f.k = d1.k AND f.k = d2.k AND f.k = d3.k AND d1.d1 < 3 "
    "GROUP BY ROLLUP(d1.d1, d2.d2)"
)


def test_narrowing_drops_the_columns_the_levels_never_read(duck):
    """The positive control: a wide subtree shared by several *different* consumers.

    A `ROLLUP`'s levels are exactly that -- one group-by each over one common input -- so
    the largest repeated subtree is the join's `Filter`, which carries every column of all
    four tables while the levels read three. This is TPC-DS q18's shape, where the chosen
    subtree carried 133 columns against the 11 the plan read, and q86's, where it carried 84
    against 3; those were the two queries reuse made slower before this cut them down.

    Both halves are asserted, and the second is the one that matters: narrowing must drop
    columns, and the answer must not move -- dropping one the plan still needs is a wrong
    result, not a slow one.
    """
    from batcher import core
    from batcher.api.subplan_reuse import reuse_common_subplans

    sess = _sess_with_wide_star_schema()
    q = sess.sql(_WIDE_ROLLUP)
    ctx = core.ExecutionContext(columns=q.columns, hub=core.default_hub())
    _, sources = reuse_common_subplans(q._plan, list(q._sources), ctx)
    assert len(sources) > len(q._sources), "no subplan was materialized"
    kept = sorted(sources[len(q._sources) :][0].schema().names)
    assert len(kept) <= 4, f"the wide subtree was materialized at full width; kept {kept}"

    for name in ("f", "d1", "d2", "d3"):
        duck.register(name, sess.sql(f"SELECT * FROM {name}").collect())
    assert_same(q.collect(), duck.sql(_WIDE_ROLLUP))


def test_narrowing_keeps_a_column_a_later_level_still_reads(duck):
    """Guard on the other side: the union of what *every* appearance reads, not the first.

    Each `ROLLUP` level reads a different subset of the keys, so narrowing to one level's
    need would drop a column another level groups by -- a wrong answer, and one the row
    count would not reveal.
    """
    sess = _sess_with_wide_star_schema()
    for name in ("f", "d1", "d2", "d3"):
        duck.register(name, sess.sql(f"SELECT * FROM {name}").collect())
    got = sess.sql(_WIDE_ROLLUP).collect()
    assert_same(got, duck.sql(_WIDE_ROLLUP))
    # One row per (a, b), one per a, and one grand total -- so more rows than the finest
    # level alone, which is what proves every level survived.
    finest = sess.sql(_WIDE_ROLLUP.replace("ROLLUP(d1.d1, d2.d2)", "d1.d1, d2.d2")).collect()
    assert got.num_rows > finest.num_rows > 0, (
        f"levels went missing: {got.num_rows} rows against {finest.num_rows} for the finest"
    )
