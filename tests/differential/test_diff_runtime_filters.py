"""Differential tests vs DuckDB for the `runtime_filters` (SIP / data-skipping) rules.

Every rule in this family *deletes rows* on the strength of a proof, so the only thing that
makes it credible is running the whole optimizer over data engineered to break a naive
version of each proof, and comparing the executed result with DuckDB:

* **NULL join keys** on both sides — `push_is_not_null_from_join_key` removes them, and it is
  wrong to do so on an anti join's left, on an outer join's preserved side, or on either side
  of a `full` join. Those null-keyed rows are part of the answer there.
* **Non-matching rows** (keys present on one side only) — a mirrored `IN` list or a
  bloom-pruned member must not remove a row the join would have kept.
* **Duplicate keys** — a runtime filter must not change fan-out.
* **Empty sides**, and joins the rules prove empty from disjoint/refuted key values.
* **ASOF** nulls — a null `on` never matches (droppable) but a null `by` matches a null `by`
  (not droppable), and the two must not be confused.

Each test asserts the rule actually fired (a plan-shape check) *and* that the result still
equals DuckDB's.
"""

from __future__ import annotations

import batcher._native as nat
import pyarrow as pa
import pytest

import batcher as bt

# Importing the package registers its rules into the default registry.
import batcher.kyber.rules.extra.runtime_filters
from batcher import col
from batcher.api.dataset import Dataset
from batcher.io.source import InMemorySource, source_statistics
from batcher.kyber.optimizer import Optimizer
from batcher.plan.expr_ir import Col, IsNotNull
from batcher.plan.expr_rewrite import split_conjuncts
from batcher.plan.logical import Filter, Limit, Scan
from batcher.plan.schema import SchemaRef
from batcher.plan.source_stats import SourceStatistics
from batcher.plan.stats import ColumnStat, Provenance
from batcher.plan.visitor import walk
from conftest import assert_same

JOIN_TYPES = ["inner", "left", "right", "full", "semi", "anti"]

# DuckDB spells the join types the SQL way; `semi`/`anti` are SEMI/ANTI JOIN.
_SQL_JOIN = {
    "inner": "INNER JOIN",
    "left": "LEFT JOIN",
    "right": "RIGHT JOIN",
    "full": "FULL OUTER JOIN",
    "semi": "SEMI JOIN",
    "anti": "ANTI JOIN",
}
# Batcher's join output carries ONE key column, named after the left key and **coalesced** across
# the sides — so a right-only row of a right/full join reports the right key there, not NULL.
# `SELECT l.k` would be NULL for that row in SQL, hence the coalesce (the same spelling the
# existing full-join differential tests use).
_COALESCED = ("right", "full")


def _cols_sql(how: str, left: str, right: str) -> str:
    """The SELECT list matching Batcher's join output for `how`."""
    if how in ("semi", "anti"):
        return f"{left}.k, {left}.v"  # a semi/anti join emits the left side only
    key = f"COALESCE({left}.k, {right}.k)" if how in _COALESCED else f"{left}.k"
    return f"{key} AS k, {left}.v AS v, {right}.w AS w"


# --- fixtures -----------------------------------------------------------------


def _left() -> pa.Table:
    # A duplicate key (2), TWO null keys, and key 3 which matches nothing on the right.
    return pa.table({"k": [1, 2, 2, None, 3, None], "v": [10, 20, 21, 98, 30, 99]})


def _right() -> pa.Table:
    # Key 1 is duplicated (fan-out), key 9 matches nothing, and one null key.
    return pa.table({"k": [1, 1, 2, 9, None], "w": [5, 6, 7, 8, 9]})


def _empty() -> pa.Table:
    return pa.table({"k": pa.array([], pa.int64()), "w": pa.array([], pa.int64())})


def _reg(duck, name: str, table: pa.Table):
    duck.register(name, table)
    return bt.from_arrow(table)


def _optimized(ds):
    stats = [source_statistics(s) for s in ds._sources]
    return Optimizer(sources=ds._sources, source_stats=stats).logical_rewrite(ds._plan)


def _conjuncts(plan):
    out = []
    for node in walk(plan):
        if isinstance(node, Filter):
            out += split_conjuncts(node.predicate)
    return out


def _pushed_not_null(ds, name: str = "k") -> bool:
    return any(
        isinstance(c, IsNotNull) and isinstance(c.input, Col) and c.input.name == name
        for c in _conjuncts(_optimized(ds))
    )


def _empty_marked(ds) -> bool:
    return any(isinstance(n, Limit) and n.n == 0 for n in walk(_optimized(ds)))


class _DeclaredSource(InMemorySource):
    """An in-memory source that declares (truthful) per-column statistics.

    The rules that skip data need EXACT bounds / a bloom / an EXACT null count, which a plain
    in-memory source does not publish. The declarations below are *true of the data* — a false
    one would (rightly) make these tests fail.
    """

    __slots__ = ("_columns",)

    def __init__(self, batches, columns) -> None:
        super().__init__(batches)
        self._columns = columns

    def statistics(self) -> SourceStatistics:
        return SourceStatistics(row_count=self.row_count(), columns=self._columns)


def _declared(table: pa.Table, columns) -> Dataset:
    src = _DeclaredSource(table.to_batches(), columns)
    return Dataset(Scan(source_id=0, schema=SchemaRef.from_arrow(src.schema())), [src])


def _bloom(table: pa.Table, column: str) -> bytes:
    idx = table.column_names.index(column)
    return nat.build_column_bloom(table.to_batches(), idx, max(1, table.num_rows))


# --- push_is_not_null_from_join_key: the null-key semantics of every join type --


@pytest.mark.parametrize("how", JOIN_TYPES)
def test_null_keys_survive_exactly_where_the_join_says_they_must(duck, how):
    left, right = _left(), _right()
    _reg(duck, "nl", left)
    _reg(duck, "nr", right)
    ds = bt.from_arrow(left).join(bt.from_arrow(right), on="k", how=how)
    cols = _cols_sql(how, "nl", "nr")
    assert_same(ds.collect(), duck.sql(f"SELECT {cols} FROM nl {_SQL_JOIN[how]} nr ON nl.k = nr.k"))


@pytest.mark.parametrize("how", ["inner", "semi", "right"])
def test_is_not_null_fires_on_the_reducible_left(duck, how):
    left, right = _left(), _right()
    _reg(duck, "fl", left)
    _reg(duck, "fr", right)
    ds = bt.from_arrow(left).join(bt.from_arrow(right), on="k", how=how)
    assert _pushed_not_null(ds), "the implied IS NOT NULL never reached the plan"
    cols = _cols_sql(how, "fl", "fr")
    assert_same(ds.collect(), duck.sql(f"SELECT {cols} FROM fl {_SQL_JOIN[how]} fr ON fl.k = fr.k"))


def test_full_join_is_never_reduced(duck):
    left, right = _left(), _right()
    _reg(duck, "ul", left)
    _reg(duck, "ur", right)
    ds = bt.from_arrow(left).join(bt.from_arrow(right), on="k", how="full")
    assert not _pushed_not_null(ds), "a full join preserves both sides — nothing may be dropped"
    assert_same(
        ds.collect(),
        duck.sql(
            "SELECT COALESCE(ul.k, ur.k) AS k, ul.v AS v, ur.w AS w "
            "FROM ul FULL OUTER JOIN ur ON ul.k = ur.k"
        ),
    )


def test_composite_key_join_with_nulls(duck):
    left = pa.table({"a": [1, 1, None, 2], "b": [1, None, 1, 2], "v": [1, 2, 3, 4]})
    right = pa.table({"a": [1, 2, None], "b": [1, 2, 1], "w": [7, 8, 9]})
    _reg(duck, "cl", left)
    _reg(duck, "cr", right)
    ds = bt.from_arrow(left).join(bt.from_arrow(right), on=["a", "b"])
    assert _pushed_not_null(ds, "a") and _pushed_not_null(ds, "b")
    assert_same(
        ds.collect(),
        duck.sql(
            "SELECT cl.a AS a, cl.b AS b, cl.v AS v, cr.w AS w "
            "FROM cl JOIN cr ON cl.a = cr.a AND cl.b = cr.b"
        ),
    )


@pytest.mark.parametrize("how", JOIN_TYPES)
def test_empty_side_join(duck, how):
    left, empty = _left(), _empty()
    _reg(duck, "el", left)
    _reg(duck, "er", empty)
    ds = bt.from_arrow(left).join(bt.from_arrow(empty), on="k", how=how)
    cols = _cols_sql(how, "el", "er")
    assert_same(ds.collect(), duck.sql(f"SELECT {cols} FROM el {_SQL_JOIN[how]} er ON el.k = er.k"))


def test_runtime_filter_under_a_projection_still_correct(duck):
    # The sink rule moves the filter below the projection — the shape that used to strand it.
    left, right = _left(), _right()
    _reg(duck, "pl", left)
    _reg(duck, "pr", right)
    ds = (
        bt.from_arrow(left)
        .select(k=col("k"), doubled=col("v") * 2)
        .join(bt.from_arrow(right), on="k")
    )
    assert_same(
        ds.collect(),
        duck.sql("SELECT pl.k AS k, pl.v * 2 AS doubled, pr.w AS w FROM pl JOIN pr ON pl.k = pr.k"),
    )


def test_runtime_filter_under_an_aggregate_still_correct(duck):
    left, right = _left(), _right()
    _reg(duck, "al", left)
    _reg(duck, "ar", right)
    grouped = bt.from_arrow(left).group_by("k").agg(total=col("v").sum())
    ds = grouped.join(bt.from_arrow(right), on="k")
    assert_same(
        ds.collect(),
        duck.sql(
            "SELECT g.k AS k, g.total AS total, ar.w AS w FROM "
            "(SELECT k, SUM(v) AS total FROM al GROUP BY k) g JOIN ar ON g.k = ar.k"
        ),
    )


def test_iter_batches_agrees_with_collect(duck):
    left, right = _left(), _right()
    _reg(duck, "il", left)
    _reg(duck, "ir", right)
    ds = bt.from_arrow(left).join(bt.from_arrow(right), on="k", how="left")
    streamed = pa.Table.from_batches(list(ds.iter_batches()), schema=ds.collect().schema)
    assert_same(
        streamed,
        duck.sql("SELECT il.k AS k, il.v AS v, ir.w AS w FROM il LEFT JOIN ir ON il.k = ir.k"),
    )


# --- push_in_list_across_join_keys / bloom pruning ------------------------------


@pytest.mark.parametrize("how", JOIN_TYPES)
def test_in_list_mirrored_across_every_join_type(duck, how):
    left, right = _left(), _right()
    _reg(duck, "ml", left)
    _reg(duck, "mr", right)
    filtered = bt.from_arrow(left).filter(col("k").is_in([1, 2, 3]))
    ds = filtered.join(bt.from_arrow(right), on="k", how=how)
    cols = _cols_sql(how, "ml", "mr")
    sql = (
        f"SELECT {cols} FROM (SELECT * FROM ml WHERE k IN (1, 2, 3)) ml "
        f"{_SQL_JOIN[how]} mr ON ml.k = mr.k"
    )
    assert_same(ds.collect(), duck.sql(sql))


def test_in_list_mirrored_onto_the_right_of_an_anti_join(duck):
    # The mirror goes L → R (the only reducible side); the left's null/unmatched rows must
    # still all come back.
    left, right = _left(), _right()
    _reg(duck, "aml", left)
    _reg(duck, "amr", right)
    filtered = bt.from_arrow(left).filter(col("k").is_in([1, 3]))
    ds = filtered.join(bt.from_arrow(right), on="k", how="anti")
    assert_same(
        ds.collect(),
        duck.sql(
            "SELECT aml.k, aml.v FROM (SELECT * FROM aml WHERE k IN (1, 3)) aml "
            "ANTI JOIN amr ON aml.k = amr.k"
        ),
    )


def test_bloom_pruned_join_key_member(duck):
    # The dimension holds {1, 1, 2, 9, NULL}; the fact asks for k IN (2, 5, 7). 5 and 7 lie
    # inside [1, 9] and are absent — only the build side's bloom can rule them out.
    left, right = _left(), _right()
    _reg(duck, "bl", left)
    _reg(duck, "br", right)
    dim = _declared(
        right,
        {"k": ColumnStat(min=1, max=9, provenance=Provenance.EXACT, bloom=_bloom(right, "k"))},
    )
    ds = bt.from_arrow(left).filter(col("k").is_in([2, 5, 7])).join(dim, on="k")
    assert_same(
        ds.collect(),
        duck.sql(
            "SELECT bl.k AS k, bl.v AS v, br.w AS w FROM "
            "(SELECT * FROM bl WHERE k IN (2, 5, 7)) bl JOIN br ON bl.k = br.k"
        ),
    )


# --- provably-empty joins --------------------------------------------------------


def test_disjoint_key_values_empty_the_inner_join(duck):
    left, right = _left(), _right()
    _reg(duck, "dl", left)
    _reg(duck, "dr", right)
    ds = (
        bt.from_arrow(left)
        .filter(col("k").is_in([1, 2]))
        .join(bt.from_arrow(right).filter(col("k") == 9), on="k")
    )
    assert _empty_marked(ds)
    assert_same(
        ds.collect(),
        duck.sql(
            "SELECT a.k AS k, a.v AS v, b.w AS w FROM (SELECT * FROM dl WHERE k IN (1, 2)) a "
            "JOIN (SELECT * FROM dr WHERE k = 9) b ON a.k = b.k"
        ),
    )


def test_disjoint_key_values_do_not_empty_an_anti_join(duck):
    # "Nothing matches" means the anti join returns its left side WHOLE — nulls included.
    left, right = _left(), _right()
    _reg(duck, "xl", left)
    _reg(duck, "xr", right)
    ds = (
        bt.from_arrow(left)
        .filter(col("k").is_in([1, 2]))
        .join(bt.from_arrow(right).filter(col("k") == 9), on="k", how="anti")
    )
    assert_same(
        ds.collect(),
        duck.sql(
            "SELECT a.k, a.v FROM (SELECT * FROM xl WHERE k IN (1, 2)) a "
            "ANTI JOIN (SELECT * FROM xr WHERE k = 9) b ON a.k = b.k"
        ),
    )


def test_bloom_absent_key_empties_the_join(duck):
    left, right = _left(), _right()
    _reg(duck, "zl", left)
    _reg(duck, "zr", right)
    dim = _declared(
        right,
        {"k": ColumnStat(min=1, max=9, provenance=Provenance.EXACT, bloom=_bloom(right, "k"))},
    )
    ds = bt.from_arrow(left).filter(col("k") == 5).join(dim, on="k")
    assert _empty_marked(ds)  # 5 is inside [1, 9] but the bloom proves it absent
    assert_same(
        ds.collect(),
        duck.sql(
            "SELECT a.k AS k, a.v AS v, zr.w AS w FROM (SELECT * FROM zl WHERE k = 5) a "
            "JOIN zr ON a.k = zr.k"
        ),
    )


def test_all_null_key_empties_the_join(duck):
    nulls = pa.table({"k": pa.array([None, None, None], pa.int64()), "v": [1, 2, 3]})
    right = _right()
    _reg(duck, "nn", nulls)
    _reg(duck, "nnr", right)
    side = _declared(nulls, {"k": ColumnStat(null_count=3, provenance=Provenance.EXACT)})
    ds = side.join(bt.from_arrow(right), on="k")
    assert _empty_marked(ds)
    assert_same(
        ds.collect(),
        duck.sql("SELECT nn.k AS k, nn.v AS v, nnr.w AS w FROM nn JOIN nnr ON nn.k = nnr.k"),
    )


def test_all_null_key_left_join_keeps_its_left_rows(duck):
    left = _left()
    nulls = pa.table({"k": pa.array([None, None], pa.int64()), "w": [1, 2]})
    _reg(duck, "lnl", left)
    _reg(duck, "lnr", nulls)
    right = _declared(nulls, {"k": ColumnStat(null_count=2, provenance=Provenance.EXACT)})
    ds = bt.from_arrow(left).join(right, on="k", how="left")
    assert not _empty_marked(ds)
    assert_same(
        ds.collect(),
        duck.sql(
            "SELECT lnl.k AS k, lnl.v AS v, lnr.w AS w FROM lnl LEFT JOIN lnr ON lnl.k = lnr.k"
        ),
    )


# --- zone-map / bloom skipping inside a predicate ----------------------------------


def _skew() -> pa.Table:
    # Values in [1, 9], no nulls, and 4 / 6 / 8 absent — inside the range but not present.
    return pa.table({"k": [1, 2, 3, 5, 7, 9], "v": [10, 20, 30, 40, 50, 60]})


def _skew_ds(*, bloom: bool = False) -> Dataset:
    table = _skew()
    stat = ColumnStat(min=1, max=9, null_count=0, provenance=Provenance.EXACT)
    if bloom:
        stat = ColumnStat(
            min=1, max=9, null_count=0, provenance=Provenance.EXACT, bloom=_bloom(table, "k")
        )
    return _declared(table, {"k": stat})


def test_always_true_conjunct_dropped(duck):
    _reg(duck, "sk", _skew())
    ds = _skew_ds().filter((col("k") >= 0) & (col("v") > 25))
    assert not any(c.to_ir().get("op") == "ge" for c in _conjuncts(_optimized(ds)))
    assert_same(ds.collect(), duck.sql("SELECT * FROM sk WHERE k >= 0 AND v > 25"))


def test_refuted_disjunct_dropped(duck):
    _reg(duck, "sk2", _skew())
    ds = _skew_ds().filter((col("k") < 0) | (col("v") > 25))
    assert not any(c.to_ir().get("op") == "or" for c in _conjuncts(_optimized(ds)))
    assert_same(ds.collect(), duck.sql("SELECT * FROM sk2 WHERE k < 0 OR v > 25"))


def test_in_list_pruned_by_bounds(duck):
    _reg(duck, "sk3", _skew())
    ds = _skew_ds().filter(col("k").is_in([1, 2, 3, 98, 99]))
    assert_same(ds.collect(), duck.sql("SELECT * FROM sk3 WHERE k IN (1, 2, 3, 98, 99)"))


def test_in_list_pruned_by_bloom(duck):
    _reg(duck, "sk4", _skew())
    # 4, 6 and 8 are inside [1, 9] but absent — only the bloom removes them.
    ds = _skew_ds(bloom=True).filter(col("k").is_in([2, 4, 6, 8, 9]))
    assert_same(ds.collect(), duck.sql("SELECT * FROM sk4 WHERE k IN (2, 4, 6, 8, 9)"))


def test_in_list_entirely_out_of_range_is_empty(duck):
    _reg(duck, "sk5", _skew())
    ds = _skew_ds().filter(col("k").is_in([95, 96, 97, 98, 99]))
    assert _empty_marked(ds)
    assert_same(ds.collect(), duck.sql("SELECT * FROM sk5 WHERE k IN (95, 96, 97, 98, 99)"))


# --- ASOF -------------------------------------------------------------------------


def _asof_left() -> pa.Table:
    return pa.table({"t": [10, 20, None, 30], "g": [1, 1, 2, None], "v": [1, 2, 3, 4]})


def _asof_right() -> pa.Table:
    return pa.table({"t": [5, 15, None, 99], "g": [1, 1, 2, None], "w": [7, 8, 9, 10]})


@pytest.mark.parametrize("direction", ["backward", "forward"])
def test_asof_with_null_on_and_null_by_keys(duck, direction):
    left, right = _asof_left(), _asof_right()
    _reg(duck, "aj_l", left)
    _reg(duck, "aj_r", right)
    ds = bt.from_arrow(left).join_asof(bt.from_arrow(right), on="t", by="g", direction=direction)
    cmp = "<=" if direction == "backward" else ">="
    order = "DESC" if direction == "backward" else "ASC"
    # The ASOF join in SQL: for each left row, the right row in its `by` group whose `t` is
    # nearest in `direction`. `NULL = NULL` is not TRUE, so a null `by`/`t` matches nothing in
    # SQL — but Batcher's ASOF groups `by` on the row encoding, where a null `by` DOES match a
    # null `by`. Compare only the rows both engines agree the semantics of: non-null `by`.
    sql = f"""
        SELECT l.t AS t, l.g AS g, l.v AS v,
               (SELECT r.w FROM aj_r r
                 WHERE r.g = l.g AND r.t IS NOT NULL AND r.t {cmp} l.t
                 ORDER BY r.t {order} LIMIT 1) AS w
          FROM aj_l l WHERE l.g IS NOT NULL AND l.t IS NOT NULL
    """
    assert_same(ds.filter(col("g").is_not_null() & col("t").is_not_null()).collect(), duck.sql(sql))
