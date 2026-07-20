"""``MERGE INTO`` vs DuckDB — every clause type, on every target layout.

DuckDB implements the full statement (including ``WHEN NOT MATCHED BY SOURCE``), so it is
a real oracle here and not just for the simple upsert. Each test states a merge once, runs
it through Batcher and through DuckDB, and compares the resulting *table*.

The layout cross-product is the point of `target`. A merge's result must not depend on how
many files the target happens to be stored in — but its *code path* very much does:

* a **single-file** target rewrites in place, and can skip nothing;
* a **directory** target rewrites only the files whose key statistics prove they could
  match, and swaps them in beside the ones it skipped;
* a directory target whose source keys reach **every** file falls back to a full rewrite;
* and any merge carrying a ``NOT MATCHED BY SOURCE`` clause must rewrite everything,
  because that clause is *about* the rows pruning would have skipped.

Running every case over each layout is what catches a pruning bug — which by construction
only ever shows up as *missing* rows in the files that were skipped, and which a
single-file test can never see.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import lit, source_col, target_col

pytestmark = pytest.mark.differential

# The layouts a target can physically take. `rows_per_file=None` writes one file.
_LAYOUTS = ["single_file", "many_files"]


@pytest.fixture(params=_LAYOUTS)
def target(request, tmp_path):
    """A factory that writes the target table in one of the physical layouts."""

    def _write(table: pa.Table) -> str:
        if request.param == "single_file":
            path = f"{tmp_path}/t.parquet"
            bt.from_arrow(table).write.parquet(path)
            return path
        path = f"{tmp_path}/t"
        # Two rows per file: enough files that pruning has something to skip, and small
        # enough that a key lands in exactly one of them.
        bt.from_arrow(table).write.parquet(path, max_rows_per_file=2)
        return path

    return _write


def _duck(con, target_tbl: pa.Table, source_tbl: pa.Table, merge_sql: str):
    """Run `merge_sql` in DuckDB against the same two tables and return the result."""
    con.register("_src", source_tbl)
    con.execute("DROP TABLE IF EXISTS t")
    con.execute("CREATE TABLE t AS SELECT * FROM _tgt")
    con.execute("DROP TABLE IF EXISTS s")
    con.execute("CREATE TABLE s AS SELECT * FROM _src")
    con.execute(merge_sql)
    return con.sql("SELECT * FROM t")


@pytest.fixture
def con(duck):
    return duck


@pytest.fixture
def duck():
    import duckdb

    return duckdb.connect()


def _run(con, target, target_tbl, source_tbl, merge_sql, build):
    """Apply the merge both ways and return `(batcher_result, duckdb_relation)`."""
    con.register("_tgt", target_tbl)
    path = target(target_tbl)
    build(bt.from_arrow(source_tbl), path).execute()
    got = bt.read.parquet(path).collect()
    expected = _duck(con, target_tbl, source_tbl, merge_sql)
    return got, expected


# --------------------------------------------------------------------------------------
# The classic upsert, and its three keyword variants.
# --------------------------------------------------------------------------------------

_T = pa.table({"id": [1, 2, 3, 4, 5, 6, 7], "v": [10, 20, 30, 40, 50, 60, 70]})
_S = pa.table({"id": [2, 5, 99], "v": [222, 555, 999]})


def test_upsert_update_and_insert(con, target):
    got, expected = _run(
        con,
        target,
        _T,
        _S,
        """MERGE INTO t USING s ON t.id = s.id
           WHEN MATCHED THEN UPDATE SET v = s.v
           WHEN NOT MATCHED THEN INSERT VALUES (s.id, s.v)""",
        lambda src, p: (
            src.write.merge_into(p, on="id")
            .when_matched()
            .update_all()
            .when_not_matched()
            .insert_all()
        ),
    )
    assert_same(got, expected)


def test_update_only_no_insert(con, target):
    got, expected = _run(
        con,
        target,
        _T,
        _S,
        """MERGE INTO t USING s ON t.id = s.id
           WHEN MATCHED THEN UPDATE SET v = s.v""",
        lambda src, p: src.write.merge_into(p, on="id").when_matched().update_all(),
    )
    assert_same(got, expected)


def test_delete_when_matched(con, target):
    got, expected = _run(
        con,
        target,
        _T,
        _S,
        """MERGE INTO t USING s ON t.id = s.id
           WHEN MATCHED THEN DELETE
           WHEN NOT MATCHED THEN INSERT VALUES (s.id, s.v)""",
        lambda src, p: (
            src.write.merge_into(p, on="id").when_matched().delete().when_not_matched().insert_all()
        ),
    )
    assert_same(got, expected)


# --------------------------------------------------------------------------------------
# Conditional clauses, tried in order — the part a two-keyword upsert cannot express.
# --------------------------------------------------------------------------------------


def test_conditional_clauses_are_ordered_first_match_wins(con, target):
    """A guarded DELETE before an unguarded UPDATE: the guard must win where it holds."""
    got, expected = _run(
        con,
        target,
        _T,
        _S,
        """MERGE INTO t USING s ON t.id = s.id
           WHEN MATCHED AND s.v > 500 THEN DELETE
           WHEN MATCHED THEN UPDATE SET v = s.v
           WHEN NOT MATCHED THEN INSERT VALUES (s.id, s.v)""",
        lambda src, p: (
            src.write.merge_into(p, on="id")
            .when_matched(source_col("v") > lit(500))
            .delete()
            .when_matched()
            .update_all()
            .when_not_matched()
            .insert_all()
        ),
    )
    assert_same(got, expected)


def test_clause_condition_reading_both_sides(con, target):
    """``source.v > target.v`` — an update that only applies when the change is newer."""
    got, expected = _run(
        con,
        target,
        _T,
        _S,
        """MERGE INTO t USING s ON t.id = s.id
           WHEN MATCHED AND s.v > t.v THEN UPDATE SET v = s.v
           WHEN NOT MATCHED THEN INSERT VALUES (s.id, s.v)""",
        lambda src, p: (
            src.write.merge_into(p, on="id")
            .when_matched(source_col("v") > target_col("v"))
            .update_all()
            .when_not_matched()
            .insert_all()
        ),
    )
    assert_same(got, expected)


def test_matched_rows_with_no_firing_clause_are_left_untouched(con, target):
    """Every matched clause is guarded and none holds ⇒ the target row survives as it was."""
    got, expected = _run(
        con,
        target,
        _T,
        _S,
        """MERGE INTO t USING s ON t.id = s.id
           WHEN MATCHED AND s.v < 0 THEN DELETE""",
        lambda src, p: (
            src.write.merge_into(p, on="id").when_matched(source_col("v") < lit(0)).delete()
        ),
    )
    assert_same(got, expected)


# --------------------------------------------------------------------------------------
# Partial column writes.
# --------------------------------------------------------------------------------------

_T3 = pa.table({"id": [1, 2, 3, 4], "v": [10, 20, 30, 40], "w": [1.5, 2.5, 3.5, 4.5]})
_S3 = pa.table({"id": [2, 9], "v": [222, 999], "w": [9.5, 9.9]})


def test_update_writes_only_the_named_column(con, target):
    """``UPDATE SET v = …`` must leave ``w`` at its existing target value."""
    got, expected = _run(
        con,
        target,
        _T3,
        _S3,
        """MERGE INTO t USING s ON t.id = s.id
           WHEN MATCHED THEN UPDATE SET v = s.v""",
        lambda src, p: (
            src.write.merge_into(p, on="id").when_matched().update({"v": source_col("v")})
        ),
    )
    assert_same(got, expected)


def test_update_with_a_computed_expression(con, target):
    got, expected = _run(
        con,
        target,
        _T3,
        _S3,
        """MERGE INTO t USING s ON t.id = s.id
           WHEN MATCHED THEN UPDATE SET v = t.v + s.v""",
        lambda src, p: (
            src.write.merge_into(p, on="id")
            .when_matched()
            .update({"v": target_col("v") + source_col("v")})
        ),
    )
    assert_same(got, expected)


def test_insert_with_a_column_list_nulls_the_rest(con, target):
    """SQL leaves an unlisted column NULL — not zero, and not the source's value."""
    got, expected = _run(
        con,
        target,
        _T3,
        _S3,
        """MERGE INTO t USING s ON t.id = s.id
           WHEN NOT MATCHED THEN INSERT (id, v) VALUES (s.id, s.v)""",
        lambda src, p: (
            src.write.merge_into(p, on="id")
            .when_not_matched()
            .insert({"id": source_col("id"), "v": source_col("v")})
        ),
    )
    assert_same(got, expected)


# --------------------------------------------------------------------------------------
# NOT MATCHED BY SOURCE — the population a plain upsert forgets.
# --------------------------------------------------------------------------------------


def test_not_matched_by_source_delete(con, target):
    """Reconcile to a snapshot: rows the source dropped are removed."""
    got, expected = _run(
        con,
        target,
        _T,
        _S,
        """MERGE INTO t USING s ON t.id = s.id
           WHEN MATCHED THEN UPDATE SET v = s.v
           WHEN NOT MATCHED THEN INSERT VALUES (s.id, s.v)
           WHEN NOT MATCHED BY SOURCE THEN DELETE""",
        lambda src, p: (
            src.write.merge_into(p, on="id")
            .when_matched()
            .update_all()
            .when_not_matched()
            .insert_all()
            .when_not_matched_by_source()
            .delete()
        ),
    )
    assert_same(got, expected)


def test_not_matched_by_source_update(con, target):
    """The soft-delete / SCD-2 expiry shape: mark, don't remove."""
    got, expected = _run(
        con,
        target,
        _T,
        _S,
        """MERGE INTO t USING s ON t.id = s.id
           WHEN MATCHED THEN UPDATE SET v = s.v
           WHEN NOT MATCHED BY SOURCE THEN UPDATE SET v = -1""",
        lambda src, p: (
            src.write.merge_into(p, on="id")
            .when_matched()
            .update_all()
            .when_not_matched_by_source()
            .update({"v": lit(-1)})
        ),
    )
    assert_same(got, expected)


def test_not_matched_by_source_conditional(con, target):
    got, expected = _run(
        con,
        target,
        _T,
        _S,
        """MERGE INTO t USING s ON t.id = s.id
           WHEN MATCHED THEN UPDATE SET v = s.v
           WHEN NOT MATCHED BY SOURCE AND t.v > 40 THEN DELETE""",
        lambda src, p: (
            src.write.merge_into(p, on="id")
            .when_matched()
            .update_all()
            .when_not_matched_by_source(target_col("v") > lit(40))
            .delete()
        ),
    )
    assert_same(got, expected)


def test_all_three_populations_at_once(con, target):
    """The full statement: every population, guarded, in one merge."""
    got, expected = _run(
        con,
        target,
        _T,
        _S,
        """MERGE INTO t USING s ON t.id = s.id
           WHEN MATCHED AND s.v > 900 THEN DELETE
           WHEN MATCHED THEN UPDATE SET v = s.v
           WHEN NOT MATCHED THEN INSERT VALUES (s.id, s.v)
           WHEN NOT MATCHED BY SOURCE AND t.v < 35 THEN UPDATE SET v = 0
           WHEN NOT MATCHED BY SOURCE THEN DELETE""",
        lambda src, p: (
            src.write.merge_into(p, on="id")
            .when_matched(source_col("v") > lit(900))
            .delete()
            .when_matched()
            .update_all()
            .when_not_matched()
            .insert_all()
            .when_not_matched_by_source(target_col("v") < lit(35))
            .update({"v": lit(0)})
            .when_not_matched_by_source()
            .delete()
        ),
    )
    assert_same(got, expected)


# --------------------------------------------------------------------------------------
# Edges: keys, nulls, empties, types.
# --------------------------------------------------------------------------------------


def test_multi_column_key(con, target):
    t = pa.table({"a": [1, 1, 2, 2], "b": ["x", "y", "x", "y"], "v": [1, 2, 3, 4]})
    s = pa.table({"a": [1, 2, 3], "b": ["y", "x", "z"], "v": [99, 88, 77]})
    got, expected = _run(
        con,
        target,
        t,
        s,
        """MERGE INTO t USING s ON t.a = s.a AND t.b = s.b
           WHEN MATCHED THEN UPDATE SET v = s.v
           WHEN NOT MATCHED THEN INSERT VALUES (s.a, s.b, s.v)""",
        lambda src, p: (
            src.write.merge_into(p, on=["a", "b"])
            .when_matched()
            .update_all()
            .when_not_matched()
            .insert_all()
        ),
    )
    assert_same(got, expected)


def test_string_key(con, target):
    t = pa.table({"k": ["a", "b", "c", "d"], "v": [1, 2, 3, 4]})
    s = pa.table({"k": ["b", "z"], "v": [99, 88]})
    got, expected = _run(
        con,
        target,
        t,
        s,
        """MERGE INTO t USING s ON t.k = s.k
           WHEN MATCHED THEN UPDATE SET v = s.v
           WHEN NOT MATCHED THEN INSERT VALUES (s.k, s.v)""",
        lambda src, p: (
            src.write.merge_into(p, on="k")
            .when_matched()
            .update_all()
            .when_not_matched()
            .insert_all()
        ),
    )
    assert_same(got, expected)


def test_null_payload_values(con, target):
    t = pa.table({"id": [1, 2, 3, 4], "v": [10, None, 30, None]})
    s = pa.table({"id": [2, 3, 5], "v": [None, 333, None]})
    got, expected = _run(
        con,
        target,
        t,
        s,
        """MERGE INTO t USING s ON t.id = s.id
           WHEN MATCHED THEN UPDATE SET v = s.v
           WHEN NOT MATCHED THEN INSERT VALUES (s.id, s.v)""",
        lambda src, p: (
            src.write.merge_into(p, on="id")
            .when_matched()
            .update_all()
            .when_not_matched()
            .insert_all()
        ),
    )
    assert_same(got, expected)


def test_null_keys_never_match(con, target):
    """SQL's ``=`` is never true for NULL, so a NULL-keyed source row matches nothing."""
    t = pa.table({"id": [1, None, 3], "v": [10, 20, 30]})
    s = pa.table({"id": [None, 3], "v": [999, 333]})
    got, expected = _run(
        con,
        target,
        t,
        s,
        """MERGE INTO t USING s ON t.id = s.id
           WHEN MATCHED THEN UPDATE SET v = s.v
           WHEN NOT MATCHED THEN INSERT VALUES (s.id, s.v)""",
        lambda src, p: (
            src.write.merge_into(p, on="id")
            .when_matched()
            .update_all()
            .when_not_matched()
            .insert_all()
        ),
    )
    assert_same(got, expected)


def test_empty_source_changes_nothing(con, target):
    empty = pa.table({"id": pa.array([], pa.int64()), "v": pa.array([], pa.int64())})
    got, expected = _run(
        con,
        target,
        _T,
        empty,
        """MERGE INTO t USING s ON t.id = s.id
           WHEN MATCHED THEN UPDATE SET v = s.v
           WHEN NOT MATCHED THEN INSERT VALUES (s.id, s.v)""",
        lambda src, p: (
            src.write.merge_into(p, on="id")
            .when_matched()
            .update_all()
            .when_not_matched()
            .insert_all()
        ),
    )
    assert_same(got, expected)


def test_empty_source_with_not_matched_by_source_empties_the_table(con, target):
    """The dangerous pairing: an empty change set + a by-source DELETE removes everything.

    It is also the one that proves pruning cannot fire here — a merge that skipped any file
    would leave that file's rows behind.
    """
    empty = pa.table({"id": pa.array([], pa.int64()), "v": pa.array([], pa.int64())})
    got, expected = _run(
        con,
        target,
        _T,
        empty,
        """MERGE INTO t USING s ON t.id = s.id
           WHEN MATCHED THEN UPDATE SET v = s.v
           WHEN NOT MATCHED BY SOURCE THEN DELETE""",
        lambda src, p: (
            src.write.merge_into(p, on="id")
            .when_matched()
            .update_all()
            .when_not_matched_by_source()
            .delete()
        ),
    )
    assert got.num_rows == 0
    assert_same(got, expected)


def test_source_matching_every_target_row(con, target):
    """No file can be skipped — the fallback-to-full-rewrite path."""
    s = pa.table({"id": [1, 2, 3, 4, 5, 6, 7], "v": [-1, -2, -3, -4, -5, -6, -7]})
    got, expected = _run(
        con,
        target,
        _T,
        s,
        """MERGE INTO t USING s ON t.id = s.id
           WHEN MATCHED THEN UPDATE SET v = s.v""",
        lambda src, p: src.write.merge_into(p, on="id").when_matched().update_all(),
    )
    assert_same(got, expected)


def test_source_matching_nothing_is_pure_insert(con, target):
    """Every key is new, so no target file can match — the merge reads none of them."""
    s = pa.table({"id": [100, 200], "v": [1, 2]})
    got, expected = _run(
        con,
        target,
        _T,
        s,
        """MERGE INTO t USING s ON t.id = s.id
           WHEN MATCHED THEN UPDATE SET v = s.v
           WHEN NOT MATCHED THEN INSERT VALUES (s.id, s.v)""",
        lambda src, p: (
            src.write.merge_into(p, on="id")
            .when_matched()
            .update_all()
            .when_not_matched()
            .insert_all()
        ),
    )
    assert_same(got, expected)


def test_float_key_with_negative_zero(con, target):
    """``-0.0 == 0.0`` in SQL, so they are the same key. A hash that disagrees splits them."""
    t = pa.table({"k": [0.0, 1.5, 2.5], "v": [1, 2, 3]})
    s = pa.table({"k": [-0.0, 9.5], "v": [99, 88]})
    got, expected = _run(
        con,
        target,
        t,
        s,
        """MERGE INTO t USING s ON t.k = s.k
           WHEN MATCHED THEN UPDATE SET v = s.v
           WHEN NOT MATCHED THEN INSERT VALUES (s.k, s.v)""",
        lambda src, p: (
            src.write.merge_into(p, on="k")
            .when_matched()
            .update_all()
            .when_not_matched()
            .insert_all()
        ),
    )
    assert_same(got, expected)


def test_merge_into_a_table_that_does_not_exist_yet(con, tmp_path):
    """No target ⇒ only the insert clauses can fire, against an empty table."""
    path = f"{tmp_path}/new.parquet"
    bt.from_arrow(_S).write.merge_into(
        path, on="id"
    ).when_matched().update_all().when_not_matched().insert_all().execute()
    got = bt.read.parquet(path).collect()

    con.register("_src", _S)
    con.execute("CREATE TABLE t AS SELECT * FROM _src WHERE false")
    con.execute("CREATE TABLE s AS SELECT * FROM _src")
    con.execute(
        """MERGE INTO t USING s ON t.id = s.id
           WHEN MATCHED THEN UPDATE SET v = s.v
           WHEN NOT MATCHED THEN INSERT VALUES (s.id, s.v)"""
    )
    assert_same(got, con.sql("SELECT * FROM t"))


# --------------------------------------------------------------------------------------
# Repeated merges: the layout must survive, or later merges stop being able to prune.
# --------------------------------------------------------------------------------------


def test_repeated_merges_converge_and_keep_the_file_layout(tmp_path):
    path = f"{tmp_path}/t"
    bt.from_arrow(_T).write.parquet(path, max_rows_per_file=2)
    before = bt.read.parquet(path).collect().num_rows

    for i in range(5):
        changes = pa.table({"id": [2, 100 + i], "v": [1000 + i, i]})
        (
            bt.from_arrow(changes)
            .write.merge_into(path, on="id")
            .when_matched()
            .update_all()
            .when_not_matched()
            .insert_all()
            .execute()
        )

    out = bt.read.parquet(path).collect().to_pydict()
    rows = dict(zip(out["id"], out["v"], strict=True))
    assert len(out["id"]) == len(rows) == before + 5  # 5 new keys, no duplicates
    assert rows[2] == 1004  # the last update to key 2 won
    assert rows[1] == 10 and rows[7] == 70  # untouched rows survived every merge

    import glob

    files = glob.glob(f"{path}/*.parquet")
    assert len(files) > 1, "the table collapsed to one file; pruning would be dead"
