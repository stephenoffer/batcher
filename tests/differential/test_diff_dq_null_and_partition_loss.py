"""Two silent narrowings of a result, both pinned against SQL's own answer.

**A NULL foreign key was reported as a referential-integrity failure.** `dq.foreign_key`
lowers to an anti-join, and NULL matches nothing, so every row with an *unset* optional
key came back as an orphan. SQL disagrees in the strongest available way: a real
``FOREIGN KEY`` constraint accepts a NULL and rejects only a non-null key with no
referent, dbt's ``relationships`` test excludes nulls explicitly, and the `ds.dq`
accessor already documents that same convention for its value constraints ("NULL passes;
forbid it with `not_null`"). On a table where the key is genuinely optional the check
reported every such row as broken, which is the kind of false positive that gets a
data-quality gate switched off.

**Reading a partitioned directory dropped the partition columns without a word.**
``to_parquet(dir, partition_by=["k"])`` writes ``k`` into the directory *names*, and a
plain file reader only ever reads files — so the obvious read-back returned every row and
no ``k``. Nothing failed, the row count was right, and the result was indistinguishable
from a correct read of a table that never had the column. That was first made *loud* (a
`DataWarning` naming the missing columns) and is now simply fixed: `read.parquet` detects
the layout and recovers the column, as DuckDB does with ``hive_partitioning=true`` and as
the explicit `read.parquet_dataset` entry point already did.

The tests below therefore pin the recovered *values* against DuckDB rather than the
warning. The remaining warning tests are the negative ones — a flat directory, and a
column that is present in the files as well as in the path, neither of which loses
anything and neither of which may announce that it did.
"""

from __future__ import annotations

import warnings

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher._internal.errors import DataWarning

pytestmark = pytest.mark.differential


# ------------------------------------------------------------------- foreign keys
def test_null_foreign_key_is_not_an_orphan() -> None:
    """A NULL key means "no reference", which is what SQL's FK constraint accepts."""
    customers = bt.from_pydict({"cid": [1, 2]})
    orders = bt.from_pydict({"cid": [1, None, 9]})
    assert orders.dq.foreign_key("cid", references=customers).to_pydict() == {"cid": [9]}


def test_null_foreign_key_matches_a_real_sql_fk_constraint() -> None:
    """The oracle, stated as the constraint itself: DuckDB takes the NULL, rejects the 9."""
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.execute("create table c(cid integer primary key)")
    con.execute("insert into c values (1), (2)")
    con.execute("create table o(cid integer references c(cid))")

    con.execute("insert into o values (1), (NULL)")  # accepted: NULL is not a violation
    with pytest.raises(duckdb.ConstraintException):
        con.execute("insert into o values (9)")  # rejected: a real dangling reference

    # Batcher's check must flag exactly what the constraint rejects, and nothing else.
    customers = bt.from_pydict({"cid": [1, 2]})
    orders = bt.from_pydict({"cid": [1, None, 9]})
    assert orders.dq.foreign_key("cid", references=customers).to_pydict() == {"cid": [9]}


def test_composite_foreign_key_with_a_null_part_is_not_an_orphan() -> None:
    """A composite key with a NULL in any part is equally "no reference"."""
    ref = bt.from_pydict({"a": [1, 1], "b": [1, 2]})
    rows = bt.from_pydict({"a": [1, 1, None], "b": [1, 9, 1]})
    got = rows.dq.foreign_key(["a", "b"], references=ref).to_pydict()
    assert got == {"a": [1], "b": [9]}


def test_foreign_key_still_finds_every_real_orphan() -> None:
    """The null exemption must not become a way for a genuine dangling key to pass."""
    ref = bt.from_pydict({"k": [1]})
    rows = bt.from_pydict({"k": [1, 2, 3, None]})
    assert sorted(rows.dq.foreign_key("k", references=ref).to_pydict()["k"]) == [2, 3]


# -------------------------------------------------------------- partition columns
def test_partitioned_read_back_recovers_the_column_rather_than_losing_it(tmp_path) -> None:
    """Write with `partition_by`, read it back the obvious way, and get the column back.

    This used to assert a `DataWarning`: `bt.read.parquet` returned every row *without* the
    partition column, because that column lives in the directory names rather than in any
    file, and being told was better than losing it silently. The plain reader is now
    partition-aware, so there is nothing to warn about — and what is worth pinning is the
    recovery, not the notice. A test that waits for a warning passes just as happily when the
    column is lost as when it is recovered, so it stops meaning anything once the gap closes.

    Held against DuckDB reading the same layout with `hive_partitioning=true`, which is the
    same claim `test_the_partition_aware_reader_recovers_the_column_like_duckdb` makes for
    the explicit `read.parquet_dataset` entry point.
    """
    duckdb = pytest.importorskip("duckdb")
    bt.from_pydict({"k": ["a", "b"], "v": [1, 2]}).to_parquet(str(tmp_path), partition_by=["k"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", DataWarning)  # nothing is lost, so nothing is announced
        got = bt.read.parquet(str(tmp_path)).to_pydict()
    expected = (
        duckdb.connect()
        .execute(
            f"select v, k from read_parquet('{tmp_path}/**/*.parquet', hive_partitioning=true)"
        )
        .fetchall()
    )
    assert sorted(zip(got["v"], got["k"], strict=True)) == sorted(expected)


def test_the_partition_aware_reader_recovers_the_column_like_duckdb(tmp_path) -> None:
    """The fix the warning names has to actually work, and agree with DuckDB."""
    duckdb = pytest.importorskip("duckdb")
    pq.write_to_dataset(
        pa.table({"x": [1, 2], "part": ["a", "b"]}), str(tmp_path), partition_cols=["part"]
    )
    got = bt.read.parquet_dataset(str(tmp_path)).to_pydict()
    expected = (
        duckdb.connect()
        .execute(f"select x, part from read_parquet('{tmp_path}/**/*.parquet')")
        .fetchall()
    )
    assert sorted(zip(got["x"], got["part"], strict=True)) == sorted(expected)


def test_a_flat_directory_does_not_warn(tmp_path) -> None:
    """The warning must fire on the layout, not on every directory read."""
    pq.write_table(pa.table({"x": [1]}), f"{tmp_path}/a.parquet")
    with warnings.catch_warnings():
        warnings.simplefilter("error", DataWarning)
        assert bt.read.parquet(str(tmp_path)).to_pydict() == {"x": [1]}


def test_no_warning_when_the_partition_column_is_also_in_the_files(tmp_path) -> None:
    """A column present in the files is not lost, whatever the directory is called."""
    d = tmp_path / "k=a"
    d.mkdir()
    pq.write_table(pa.table({"k": ["a"], "v": [1]}), f"{d}/part.parquet")
    with warnings.catch_warnings():
        warnings.simplefilter("error", DataWarning)
        got = bt.read.parquet(str(tmp_path)).to_pydict()
    assert got == {"k": ["a"], "v": [1]}
