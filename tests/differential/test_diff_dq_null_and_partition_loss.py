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
from a correct read of a table that never had the column. DuckDB detects the layout and
recovers the column; `read.parquet_dataset` is the reader here that does. The plain
reader now warns rather than staying silent, because rerouting it would quietly discard
the caller's `on_error`/`schema_mode`/`columns`/`n_rows`, which that reader accepts and
the dataset reader does not — a second silent change to cover the first.
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
def test_partitioned_read_back_warns_instead_of_silently_losing_the_column(tmp_path) -> None:
    """Write with `partition_by`, read it back the obvious way, and be told what is missing."""
    bt.from_pydict({"k": ["a", "b"], "v": [1, 2]}).to_parquet(str(tmp_path), partition_by=["k"])
    with pytest.warns(DataWarning, match="partition columns"):
        got = bt.read.parquet(str(tmp_path)).to_pydict()
    # The warning describes what actually happened: every row, without `k`.
    assert sorted(got["v"]) == [1, 2]
    assert "k" not in got


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
