"""The hard invariant: a metadata answer always equals the executed answer.

For every plan where `count()` / `is_empty()` / a global aggregate is answered
from metadata, that answer MUST equal a full execution — bit for bit. These
tests also pin the provenance firewall: a filtered count is never answered from
metadata, and `count_distinct` is answered only from an exact distinct count.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher import col, count
from batcher.api.terminal import _collect
from batcher.api.terminal.metadata_answer import metadata_count as _answer_count


@pytest.fixture
def pq_path(tmp_path):
    table = pa.table({"x": list(range(100)), "g": [i % 7 for i in range(100)]})
    path = str(tmp_path / "t.parquet")
    pq.write_table(table, path)
    return path


def _ds(pq_path):
    return bt.read.parquet(pq_path)


# --- count(): metadata answer == execution, whenever an answer is produced ---


@pytest.mark.parametrize(
    "build",
    [
        lambda d: d,
        lambda d: d.limit(10),
        lambda d: d.limit(10, offset=5),
        lambda d: d.limit(500),  # n > rows
        lambda d: d.select(a=col("x")),
        lambda d: d.sort("x"),
        lambda d: d.sort("x").limit(3),
        lambda d: d.agg(c=count()),
        lambda d: d.limit(0),
    ],
)
def test_metadata_count_matches_execution(pq_path, build):
    ds = build(_ds(pq_path))
    answer = _answer_count(ds._plan, ds._sources)
    executed = _collect(ds._plan, ds._sources, ds.columns).num_rows
    if answer is not None:
        assert answer == executed
    # The plain count() API must agree with execution regardless of which path it took.
    assert ds.count() == executed


def test_filter_count_not_answered_from_metadata(pq_path):
    # The firewall: a filtered row count is never EXACT, so no metadata answer.
    ds = _ds(pq_path).filter(col("x") > 50)
    assert _answer_count(ds._plan, ds._sources) is None
    assert ds.count() == 49  # but execution is correct


def test_union_count_is_exact_and_matches(pq_path):
    ds = _ds(pq_path).union(_ds(pq_path))
    answer = _answer_count(ds._plan, ds._sources)
    assert answer == 200
    assert ds.count() == 200


# --- global aggregate: metadata answer == execution ---


def test_global_aggregate_min_max_matches_execution(pq_path):
    ds = _ds(pq_path).agg(mn=col("x").min(), mx=col("x").max(), c=count())
    meta = ds.to_pydict()
    assert meta == {"mn": [0], "mx": [99], "c": [100]}


def test_count_distinct_executes_without_exact_ndv(pq_path):
    # Parquet footers don't give an exact distinct count → must execute, correctly.
    ds = _ds(pq_path).agg(n=col("g").n_unique())
    assert ds.to_pydict() == {"n": [7]}


def test_is_empty_matches_execution(pq_path):
    ds = _ds(pq_path)
    assert ds.is_empty() is False
    assert ds.limit(0).is_empty() is True
    assert ds.filter(col("x") > 1000).is_empty() == (
        _collect(ds.filter(col("x") > 1000)._plan, ds._sources, ds.columns).num_rows == 0
    )


def test_schema_matches_execution(pq_path):
    ds = _ds(pq_path).select(a=col("x"), b=col("g"))
    meta_schema = ds.schema
    executed_schema = _collect(ds._plan, ds._sources, ds.columns).schema
    assert meta_schema.names == executed_schema.names


# --- provably-empty joins: count()/is_empty() answer 0 from metadata ---


def test_inner_join_empty_side_counts_zero_from_metadata(pq_path):
    # An EXACT-empty side makes an inner join EXACT-empty, so count() answers 0
    # from metadata (and equals execution) without running the join.
    ds = _ds(pq_path).limit(0).join(_ds(pq_path), on="x")
    answer = _answer_count(ds._plan, ds._sources)
    executed = _collect(ds._plan, ds._sources, ds.columns).num_rows
    assert executed == 0
    assert answer == 0  # not None → the metadata shortcut fired
    assert ds.is_empty() is True


def test_left_join_empty_left_counts_zero_from_metadata(pq_path):
    # LEFT join is left-driven: an empty left → empty result, answered from metadata.
    ds = _ds(pq_path).limit(0).join(_ds(pq_path), on="x", how="left")
    answer = _answer_count(ds._plan, ds._sources)
    assert answer == 0
    assert _collect(ds._plan, ds._sources, ds.columns).num_rows == 0


def test_left_join_empty_right_not_claimed_empty(pq_path):
    # The firewall: a LEFT join with an empty *right* keeps every left row (null-
    # extended), so it is NOT provably empty — no metadata answer, execution correct.
    ds = _ds(pq_path).join(_ds(pq_path).limit(0), on="x", how="left")
    assert _answer_count(ds._plan, ds._sources) is None
    assert ds.count() == _collect(ds._plan, ds._sources, ds.columns).num_rows


def test_asof_join_empty_left_counts_zero_from_metadata(pq_path):
    # ASOF is left-style: an empty left → empty result, answered from metadata.
    ds = _ds(pq_path).limit(0).join_asof(_ds(pq_path), on="x")
    answer = _answer_count(ds._plan, ds._sources)
    assert answer == 0
    assert _collect(ds._plan, ds._sources, ds.columns).num_rows == 0


# --- the in-memory moment facets: sum / mean / count_distinct answered without a scan ---
#
# An immutable in-memory relation computes its own sum, average and distinct count on
# demand, and `metadata_answer.enrich` lifts them into the statistics so a keyless
# aggregate is answered without touching a row. The conductor collects the surrounding
# bundle for the columns a `MIN`/`MAX` needs — none, for a `SUM` — so the bundle is
# `DEFAULT`, and every one of these used to be refused for sitting inside it and executed
# in full (`sum` 2.60 ms against `min`'s 0.24 ms over 6M rows).
#
# These assert the two halves that matter together: the answer is *produced* (so a
# regression to "execute everything" fails the test rather than passing it quietly), and it
# equals what Arrow computes over the same values.

_MOMENT_RELATIONS = {
    "plain": {"a": [1.5, 2.5, 3.0, 4.0], "i": [1, 2, 2, 3]},
    "with-nulls": {"a": [1.5, None, 3.0, None], "i": [1, None, 1, 3]},
    "all-null": {"a": [None, None], "i": [None, None]},
    "one-row": {"a": [7.5], "i": [7]},
    "signed-zero": {"a": [0.0, -0.0, 1.0], "i": [0, 0, 1]},
    "nan": {"a": [1.0, float("nan"), 3.0], "i": [1, 2, 3]},
    "big-ints": {"a": [1.0], "i": [2**60, 2**60, 2**60]},
    # SQL `sum`/`avg` over no rows is NULL, not 0 — the one answer a recorded total is most
    # likely to get wrong, since "no rows" and "a total of zero" look identical in the stats.
    "empty": {"a": [], "i": []},
}


def _same_scalar(got, want) -> bool:
    """Equality that treats two NaNs as equal, the way the comparison gate here needs."""
    if got is None or want is None:
        return got is want or (got is None and want is None)
    if isinstance(got, float) and isinstance(want, float):
        import math

        return (math.isnan(got) and math.isnan(want)) or got == want
    return got == want


@pytest.mark.parametrize("name", sorted(_MOMENT_RELATIONS))
@pytest.mark.parametrize("agg", ["sum_a", "mean_a", "nunique_i", "sum_i", "min_a", "count_a"])
def test_in_memory_global_aggregate_matches_arrow(name, agg):
    import pyarrow.compute as pc

    data = _MOMENT_RELATIONS[name]
    # `i` and `a` may differ in length across the fixtures above; hold each aggregate to the
    # column it names, so a relation is only ever asked about a column it has.
    column = "i" if agg.endswith("_i") else "a"
    # Typed explicitly: an all-null Python list infers Arrow's `null` type, which has no
    # `count_distinct` kernel — so the oracle, not the engine, would be the thing that failed.
    dtype = pa.int64() if column == "i" else pa.float64()
    table = pa.table({column: pa.array(data[column], type=dtype)})
    ds = bt.from_arrow(table)
    build, oracle = {
        "sum_a": (lambda d: d.agg(v=col("a").sum()), pc.sum),
        "mean_a": (lambda d: d.agg(v=col("a").mean()), pc.mean),
        "nunique_i": (lambda d: d.agg(v=col("i").n_unique()), pc.count_distinct),
        "sum_i": (lambda d: d.agg(v=col("i").sum()), pc.sum),
        "min_a": (lambda d: d.agg(v=col("a").min()), pc.min),
        "count_a": (lambda d: d.agg(v=col("a").count()), lambda c: pc.count(c, mode="only_valid")),
    }[agg]
    got = build(ds).to_pydict()["v"][0]
    assert _same_scalar(got, oracle(table[column]).as_py())


def test_the_moment_facets_are_actually_answered_from_metadata():
    """The other half: these shapes must not reach the engine at all.

    Without this the equivalence test above passes just as happily when every answer comes
    from a full execution, which is the state this fixed.
    """
    from batcher.api.terminal.metadata_answer.aggregate import metadata_aggregate_table

    ds = bt.from_arrow(pa.table({"a": [1.5, 2.5, 3.0], "i": [1, 2, 2]}))
    for build in (
        lambda d: d.agg(v=col("a").sum()),
        lambda d: d.agg(v=col("a").mean()),
        lambda d: d.agg(v=col("i").n_unique()),
        lambda d: d.agg(v=col("i").sum()),
    ):
        q = build(ds)
        assert metadata_aggregate_table(q._plan, q._sources, None) is not None


def test_a_filtered_sum_is_never_answered_from_the_whole_relation_total():
    """A recorded total describes the whole relation, so a filter must send it to the engine.

    The dangerous direction of this change: the enrichment attaches an exact sum of *every*
    row, and answering a `WHERE`-restricted query from it would be a wrong answer produced by
    an optimization.
    """
    from batcher.api.terminal.metadata_answer.aggregate import metadata_aggregate_table

    table = pa.table({"a": [1.0, 2.0, 3.0, 4.0]})
    ds = bt.from_arrow(table)
    q = ds.filter(col("a") > 2.0).agg(v=col("a").sum())
    assert metadata_aggregate_table(q._plan, q._sources, None) is None
    assert q.to_pydict()["v"][0] == 7.0


@pytest.mark.parametrize(
    ("build", "expected"),
    [
        (lambda d: d.select(a=col("b") * 2).agg(v=col("a").sum()), 120.0),
        (lambda d: d.select(a=col("b")).agg(v=col("a").sum()), 60.0),
        (lambda d: d.select(a=col("b")).agg(v=col("a").mean()), 20.0),
        (lambda d: d.select(a=col("b")).agg(v=col("a").n_unique()), 3),
        (lambda d: d.select(a=col("a") * 10).agg(v=col("a").sum()), 60.0),
    ],
)
def test_a_projection_that_rebinds_a_source_column_name_is_not_answered_from_the_source(
    build, expected
):
    """The dangerous shape for a facet lifted onto a *source* column: name shadowing.

    `enrich` fills a column's exact sum by asking the source for it **by name**, and the
    aggregate below sits above a `Project` that has rebound that same name to a different
    expression. Answering the projected `sum(a)` from the source's `a` would be a wrong
    answer produced by an optimization — the worst kind, and invisible to a row-multiset
    comparison because the shape returns one row either way.

    It does not happen: the enrichment writes onto the *scan*'s statistics and the estimator
    derives the projected column from the expression that produces it. Pinned because
    nothing else here would notice if that stopped being true.
    """
    ds = bt.from_arrow(pa.table({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]}))
    assert build(ds).to_pydict()["v"][0] == expected
