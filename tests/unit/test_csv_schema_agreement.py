"""Every CSV read path agrees with `schema()` and with the others.

CSV types are inferred from the file's first block — pyarrow's streaming reader commits
to that, and it is what DuckDB and Polars sample. So a column that is integral for a
million rows and then holds ``"N/A"`` was inferred wrong, and the three read paths each
handled it differently:

* `schema()` said ``int64``;
* `read()` re-inferred over the *whole* file and silently returned ``string`` —
  contradicting the schema the engine had already planned and typed its operators against;
* `iter_batches()` raised a bare pyarrow conversion error.

One file, three answers. The silent one is the dangerous one: an engine told a column is
`int64` and handed `string` batches has had its contract broken, and nothing says so.

Every path is now pinned to the advertised schema, so they agree; when inference cannot
reach the truth, `schema=` declares it and the error says so.
"""

from __future__ import annotations

import pickle

import pyarrow as pa
import pytest

from batcher._internal.errors import SchemaError
from batcher.io.formats.structured.csv import CSVSource

pytestmark = pytest.mark.unit

# Long enough that the offending value lands well past the first inference block.
_ROWS = 200_000


@pytest.fixture
def late_string(tmp_path):
    """A column that looks integral for 200k rows and then is not."""
    path = tmp_path / "t.csv"
    with open(path, "w") as fh:
        fh.write("k,v\n")
        for i in range(_ROWS):
            fh.write(f"{i},{i}\n")
        fh.write("999999,not_a_number\n")
    return str(path)


def _table(batches) -> pa.Table:
    return pa.Table.from_batches(batches)


def test_read_no_longer_silently_contradicts_the_advertised_schema(late_string) -> None:
    """It used to return `string` for a column `schema()` called `int64`."""
    source = CSVSource(late_string)
    assert source.schema().field("v").type == pa.int64()

    with pytest.raises(SchemaError):
        source.read()


def test_both_read_paths_fail_the_same_way(late_string) -> None:
    """Consistency is the point: the same file must not read three different ways."""
    source = CSVSource(late_string)

    with pytest.raises(SchemaError):
        source.read()
    with pytest.raises(SchemaError):
        list(source.iter_batches())


def test_the_error_names_the_escape_hatch(late_string) -> None:
    """A user hitting this needs to be told the type is declarable, which the raw
    pyarrow error does not say."""
    with pytest.raises(SchemaError, match="schema="):
        CSVSource(late_string).read()


def test_a_declared_schema_makes_every_path_agree(late_string) -> None:
    declared = pa.schema([("k", pa.int64()), ("v", pa.string())])
    source = CSVSource(late_string, schema=declared)

    from_read = _table(source.read())
    from_iter = _table(list(source.iter_batches()))

    assert source.schema() == declared
    assert from_read.schema.field("v").type == pa.string()
    assert from_read.num_rows == _ROWS + 1
    assert from_read.equals(from_iter)


def test_a_declared_schema_survives_the_split_round_trip(late_string) -> None:
    """A worker rebuilds the reader from the split alone. Without the schema, a range
    holding only integers types the column `int64` while its sibling ranges type it
    `string` — the ranges of one file disagreeing with each other."""
    declared = pa.schema([("k", pa.int64()), ("v", pa.string())])
    source = CSVSource(late_string, schema=declared)

    splits = source.splits(target_size=512 * 1024)
    shipped = [pickle.loads(pickle.dumps(s)) for s in splits]
    rows = sum(sum(b.num_rows for b in s.read()) for s in shipped)
    types = {b.schema.field("v").type for s in shipped for b in s.read()}

    assert len(splits) > 1, "expected the file to split into byte ranges"
    assert rows == _ROWS + 1
    assert types == {pa.string()}, "the ranges disagreed on the column type"


def test_an_ordinary_file_is_unaffected(tmp_path) -> None:
    """Pinning must not change a file whose inference was right."""
    path = tmp_path / "ok.csv"
    path.write_text("a,b\n1,2\n3,4\n")
    source = CSVSource(str(path))

    assert _table(source.read()).equals(_table(list(source.iter_batches())))
    assert source.schema().field("a").type == pa.int64()


def test_projection_still_works_with_a_declared_schema(late_string) -> None:
    declared = pa.schema([("k", pa.int64()), ("v", pa.string())])
    source = CSVSource(late_string, schema=declared)

    batch = next(iter(source.iter_batches(["v"])))

    assert batch.schema.names == ["v"]


def test_the_public_reader_accepts_a_schema(late_string) -> None:
    import batcher as bt

    declared = pa.schema([("k", pa.int64()), ("v", pa.string())])
    ds = bt.read.csv(late_string, schema=declared)

    assert ds.count() == _ROWS + 1
    assert ds.schema.field("v").type == pa.string()
