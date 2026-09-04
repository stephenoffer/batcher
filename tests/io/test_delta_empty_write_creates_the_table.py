"""An empty result written to Delta is a readable, correctly-typed empty table.

An empty write is an ordinary outcome — a filter that matched nothing, a day with no
events — and it used to leave a directory that was not a Delta table at all. The commit
short-circuited on "no add-actions and mode is append", which is right for an append to
an *existing* table and wrong when the table does not exist yet: what an empty write to a
new table has to say is the schema. The writer's own zero-row part file was left sitting
in the root with no `_delta_log` beside it, `write()` returned a `WriteManifest` as though
it had succeeded, and `bt.read(..., format="delta")` then refused the tree Batcher had
just produced with "No files in log segment".

`deltalake.write_deltalake` creates the log for the same input, so this was Batcher-side.

These pin both halves: the table is now created, and the no-op that motivated the
short-circuit — appending zero rows to a table that already exists — still writes no
commit.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt

pytest.importorskip("deltalake")

_EMPTY = pa.table({"i": pa.array([], pa.int64()), "s": pa.array([], pa.string())})
_ONE = pa.table({"i": pa.array([1], pa.int64()), "s": pa.array(["x"], pa.string())})


def _commits(root) -> int:
    return len(list((root / "_delta_log").glob("*.json")))


@pytest.mark.io
def test_an_empty_write_creates_a_readable_table(tmp_path):
    root = tmp_path / "t"
    bt.from_arrow(_EMPTY).write.delta(str(root))

    assert (root / "_delta_log").is_dir(), "no transaction log: this is not a Delta table"
    back = bt.read(str(root), format="delta").collect()
    assert back.num_rows == 0


@pytest.mark.io
def test_the_empty_table_keeps_its_declared_column_types(tmp_path):
    # The schema is the whole point of committing an empty write, so a `null`-typed
    # readback would be as useless as no table at all.
    root = tmp_path / "t"
    bt.from_arrow(_EMPTY).write.delta(str(root))

    back = bt.read(str(root), format="delta").collect()
    assert {f.name: str(f.type) for f in back.schema} == {"i": "int64", "s": "string"}


@pytest.mark.io
def test_a_later_append_lands_in_the_table_the_empty_write_created(tmp_path):
    # The day-two case: the first run matched nothing, the second has rows.
    root = tmp_path / "t"
    bt.from_arrow(_EMPTY).write.delta(str(root))
    bt.from_arrow(_ONE).write.delta(str(root), mode="append")

    assert bt.read(str(root), format="delta").collect().to_pydict() == {"i": [1], "s": ["x"]}


@pytest.mark.io
def test_appending_no_rows_to_an_existing_table_still_writes_no_commit(tmp_path):
    # The optimization the short-circuit existed for. A table that already says what it is
    # has nothing to add when a write brings no rows, and a commit per empty micro-batch
    # would grow the log without bound.
    root = tmp_path / "t"
    bt.from_arrow(_ONE).write.delta(str(root))
    before = _commits(root)

    bt.from_arrow(_EMPTY).write.delta(str(root), mode="append")

    assert _commits(root) == before
    assert bt.read(str(root), format="delta").collect().num_rows == 1


@pytest.mark.io
def test_an_empty_overwrite_empties_an_existing_table(tmp_path):
    # Unlike an append, an empty *overwrite* has something to say: the rows are gone.
    root = tmp_path / "t"
    bt.from_arrow(_ONE).write.delta(str(root))
    bt.from_arrow(_EMPTY).write.delta(str(root), mode="overwrite")

    assert bt.read(str(root), format="delta").collect().num_rows == 0


@pytest.mark.io
def test_an_empty_write_of_a_filtered_query_round_trips(tmp_path):
    # The shape this is actually reached by: a query, not a hand-built empty table.
    root = tmp_path / "t"
    source = bt.from_pydict({"i": [1, 2, 3], "s": ["a", "b", "c"]})
    source.filter(bt.col("i") > 100).write.delta(str(root))

    back = bt.read(str(root), format="delta").collect()
    assert back.num_rows == 0
    assert set(back.column_names) == {"i", "s"}
