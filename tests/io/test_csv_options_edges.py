"""CSV reader and writer options at their edges, each against DuckDB or pyarrow.

Covers the defects this file was written for, and the options no test exercised:

- a quoted field holding a newline must never be cut by a byte-range split (a range that
  started inside it parsed the field's tail as invented rows);
- ``null_values="MISSING"`` is one token, not seven one-character tokens;
- a ``null_value=`` write must read back through Batcher's own reader, types and nulls
  included (Arrow's writer quoted every cell, the token too);
- a ``schema_mode="union"`` split must read its own file's columns, not the union's;
- the option errors say what is wrong with the option, not that a file is unreadable.
"""

from __future__ import annotations

import pickle

import duckdb
import pyarrow as pa
import pyarrow.csv as pacsv
import pytest

import batcher as bt
from batcher._internal.errors import FormatError, SchemaError
from batcher.io.formats.structured.csv import CSVRangeSplit, CSVSource
from batcher.io.splits import FileSplit

pytestmark = pytest.mark.differential


def _write(tmp_path, name: str, content: str | bytes) -> str:
    path = tmp_path / name
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, newline="")
    return str(path)


def _rows(table: pa.Table) -> list[tuple]:
    return sorted(tuple(row.values()) for row in table.to_pylist())


def _duck(sql: str) -> pa.Table:
    return duckdb.sql(sql).to_arrow_table()


def _read_splits(splits) -> pa.Table:
    """Every split read after a pickle round trip, as a worker would read it."""
    batches = [b for s in splits for b in pickle.loads(pickle.dumps(s)).read(None)]
    return pa.Table.from_batches(batches)


# --- byte-range splits and quoted newlines ----------------------------------------------


def test_a_quoted_newline_is_never_cut_by_a_byte_range(tmp_path):
    """The field ``"x\\n{i+1000},fake"`` looks like a record when cut after its newline."""
    rows = "".join(f'{i},"x\n{i + 1000},fake"\n' for i in range(200))
    path = _write(tmp_path, "quoted.csv", "a,b\n" + rows)
    splits = CSVSource(path)._file_splits(path, 300)
    assert len(splits) > 1, "the file must still be cut, or this proves nothing"
    ours = _read_splits(splits)
    oracle = _duck(f"select * from read_csv('{path}')")
    assert ours.num_rows == oracle.num_rows == 200
    assert sorted(ours.column("a").to_pylist()) == sorted(oracle.column("a").to_pylist())
    assert ours.column("b").to_pylist() == pacsv.read_csv(path).column("b").to_pylist()


def test_a_file_with_no_quoted_newline_keeps_its_byte_ranges(tmp_path):
    """The fast path survives: quotes alone, without a newline inside, cut as before."""
    path = _write(tmp_path, "plain.csv", "a,b\n" + "".join(f'{i},"x,y"\n' for i in range(400)))
    splits = CSVSource(path)._file_splits(path, 300)
    assert len(splits) > 5 and all(isinstance(s, CSVRangeSplit) for s in splits)
    assert sorted(_read_splits(splits).column("a").to_pylist()) == list(range(400))


def test_a_file_using_an_escape_character_is_read_whole(tmp_path):
    """An escaped quote does not toggle quoting, so the parity proof does not hold."""
    body = "".join(f'{i},"q\\"x"\n' for i in range(300))
    path = _write(tmp_path, "escaped.csv", "a,b\n" + body)
    splits = CSVSource(path, escape_char="\\")._file_splits(path, 300)
    assert [type(s) for s in splits] == [FileSplit]
    ours = _read_splits(splits)
    oracle = _duck(f"select * from read_csv('{path}', escape='\\')")
    assert _rows(ours) == _rows(oracle)


# --- option vocabulary ------------------------------------------------------------------


def test_a_single_null_token_string_is_one_token(tmp_path):
    path = _write(tmp_path, "m.csv", "a,b\n1,MISSING\n2,M\n3,I\n")
    as_str = bt.read.csv(path, null_values="MISSING").collect(distributed=False)
    as_list = bt.read.csv(path, na_values=["MISSING"]).collect(distributed=False)
    oracle = _duck(f"select * from read_csv('{path}', nullstr='MISSING')")
    assert as_str.equals(as_list)
    assert as_str.column("b").to_pylist() == oracle.column("b").to_pylist() == [None, "M", "I"]


_OPTIONS = {
    "quote_char": ("a,b\n1,'x,y'\n", {"quote_char": "'"}, "quote=''''"),
    "escape_char": ('a,b\n1,"he said \\"hi\\""\n', {"escape_char": "\\"}, "escape='\\'"),
    "skip_rows": ("junk1\njunk2\na,b\n1,x\n", {"skip_rows": 2}, "skip=2"),
    "header_none": ("1,x\n2,y\n", {"header": None}, "header=false"),
    "names": ("1,x\n2,y\n", {"header": None, "names": ["p", "q"]}, "header=false, names=['p','q']"),
    "true_false_values": (
        "a,b\n1,yes\n2,no\n",
        {"true_values": ["yes"], "false_values": ["no"]},
        "types={'b': 'BOOLEAN'}",
    ),
    "decimal_point": (
        "a;b\n1;1,5\n",
        {"sep": ";", "decimal_point": ","},
        "delim=';', decimal_separator=','",
    ),
    "semicolon": ("a;b\n1;x\n", {"delimiter": ";"}, "delim=';'"),
}


@pytest.mark.parametrize("case", sorted(_OPTIONS))
def test_an_option_reads_what_duckdb_reads(tmp_path, case):
    content, ours_kw, duck_kw = _OPTIONS[case]
    path = _write(tmp_path, f"{case}.csv", content)
    ours = bt.read.csv(path, **ours_kw).collect(distributed=False)
    oracle = _duck(f"select * from read_csv('{path}', {duck_kw})")
    assert _rows(ours) == _rows(oracle)
    if "names" in ours_kw:
        assert ours.column_names == ours_kw["names"]


def test_parse_dates_types_a_listed_column_as_a_timestamp(tmp_path):
    path = _write(tmp_path, "d.csv", "a,d\n1,2024-01-02\n2,2024-03-04\n")
    ours = bt.read.csv(path, parse_dates=["d"]).collect(distributed=False)
    assert pa.types.is_timestamp(ours.schema.field("d").type)
    oracle = _duck(f"select a, d::timestamp as d from read_csv('{path}')")
    assert _rows(ours) == _rows(oracle)


def test_a_latin1_file_reads_with_its_encoding(tmp_path):
    path = _write(tmp_path, "latin.csv", "a,b\n1,caf\xe9\n".encode("latin-1"))
    ours = bt.read.csv(path, encoding="latin-1").collect(distributed=False)
    oracle = _duck(f"select * from read_csv('{path}', encoding='latin-1')")
    assert _rows(ours) == _rows(oracle) == [(1, "caf\xe9")]


# --- null_value round trip --------------------------------------------------------------


def test_a_null_value_write_reads_back_through_batchers_own_reader(tmp_path):
    source = {
        "a": [1, None, 3],
        "b": ["x,y", None, ""],
        "c": ['q"uote', "NULL", "plain"],
        "d": [1.5, None, 2.0],
        "e": [True, None, False],
    }
    out = str(tmp_path / "out.csv")
    bt.from_pydict(source).write.csv(out, null_value="NULL")
    back = bt.read.csv(out, null_values="NULL").collect(distributed=False)
    assert back.to_pydict() == source
    assert back.schema.types == [pa.int64(), pa.string(), pa.string(), pa.float64(), pa.bool_()]
    # DuckDB reads the same file with the same token, apart from the literal string "NULL",
    # which it turns into a null (its quoted-null default) and Batcher keeps as written.
    oracle = _duck(f"select * from read_csv('{out}', nullstr='NULL')")
    assert oracle.column("a").to_pylist() == source["a"]
    assert oracle.column("b").to_pylist() == source["b"]


def test_a_null_value_stream_write_keeps_the_header_and_the_token(tmp_path):
    from batcher.io.formats.structured.csv import CSVSink

    path = str(tmp_path / "s.csv")
    batches = [pa.record_batch({"a": [1, None]}), pa.record_batch({"a": [None, 4]})]
    sink = CSVSink(null_value="NA")
    writer_fh = open(path, "wb")  # noqa: SIM115 (the sink's incremental writer owns it)
    writer = sink._open_stream_writer(writer_fh, batches[0].schema)
    for batch in batches:
        sink._write_batch(writer, batch)
    sink._close_stream_writer(writer)
    writer_fh.close()
    with open(path) as fh:
        assert fh.read() == '"a"\n1\nNA\nNA\n4\n'


# --- union across files with different headers ------------------------------------------


def test_union_splits_read_their_own_files_columns(tmp_path):
    """Each split reads its own header and is filled to the union, as DuckDB does."""
    _write(tmp_path, "f0.csv", "a,b\n1,x\n2,y\n")
    _write(tmp_path, "f1.csv", "b,a,c\nz,3.5,q\n")
    source = CSVSource(str(tmp_path), schema_mode="union")
    ours = _read_splits(source.splits())
    oracle = _duck(f"select * from read_csv('{tmp_path}/*.csv', union_by_name=true)")
    assert ours.schema == source.schema()
    assert sorted(map(str, ours.to_pylist())) == sorted(
        map(
            str,
            bt.read.csv(str(tmp_path), schema_mode="union").collect(distributed=False).to_pylist(),
        )
    )
    assert len(_rows(ours)) == oracle.num_rows == 3


def test_strict_names_a_column_a_later_header_lacks(tmp_path):
    _write(tmp_path, "f0.csv", "a,b\n1,x\n")
    _write(tmp_path, "f1.csv", "a,c\n2,y\n")
    with pytest.raises(SchemaError, match="schema_mode='union'"):
        bt.read.csv(str(tmp_path)).select("a", "b").collect(distributed=False)


# --- option and file errors -------------------------------------------------------------


@pytest.mark.parametrize("option", ["delimiter", "quote_char", "escape_char"])
def test_a_multi_character_separator_is_named_as_the_option(tmp_path, option):
    path = _write(tmp_path, "m.csv", "a||b\n1||x\n")
    with pytest.raises(FormatError, match=f"{option}='\\|\\|' must be a single character"):
        bt.read.csv(path, **{option: "||"})


def test_a_negative_skip_rows_is_named_as_the_option(tmp_path):
    path = _write(tmp_path, "m.csv", "a,b\n1,2\n")
    with pytest.raises(FormatError, match="skip_rows=-1 is negative"):
        bt.read.csv(path, skip_rows=-1)


def test_an_empty_file_says_it_is_empty(tmp_path):
    path = _write(tmp_path, "empty.csv", "")
    with pytest.raises(FormatError, match="is empty"):
        bt.read.csv(path).collect(distributed=False)


def test_a_duplicate_header_names_the_duplicate_and_duckdbs_renaming_works(tmp_path):
    path = _write(tmp_path, "dup.csv", "a,b,a,a\n1,2,3,4\n")
    with pytest.raises(FormatError) as info:
        bt.read.csv(path).collect(distributed=False)
    oracle = _duck(f"select * from read_csv('{path}')")
    assert str(oracle.column_names) in str(info.value)
    fixed = bt.read.csv(path, names=oracle.column_names, skip_rows=1).collect(distributed=False)
    assert fixed.column_names == oracle.column_names
    assert _rows(fixed) == _rows(oracle)
