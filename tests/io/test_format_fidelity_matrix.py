"""Every writable file format x every Arrow type, through a real file on disk.

The existing IO tests are per-format "hunts" — the shapes somebody thought to check for
one reader or writer. What no test states is the *contract*: which types survive a
write->read intact, and where a format converts one silently. That matters more here than
a missing edge case, because a lossy conversion is data corruption that produces no error
and no wrong row count. A `date32` read back as an `int64` of epoch milliseconds still
joins, still sorts, still counts.

So this is a matrix, and it is written from measurement, not from what the formats ought
to do. Two claims, each of which fails loudly if it stops holding:

* the binary sinks — parquet, arrow (IPC), avro, orc, lance, delta — round-trip **every**
  type class here exactly, including the values that normally get canonicalized away:
  `-0.0` distinct from `0.0`, NaN, +-inf, and `2**53 +- 1` (the integers a float64 detour
  would collapse). That is a strong guarantee and the reason these are the formats to
  reach for; a regression in it is silent corruption.
* the text sinks — csv, json, msgpack — convert some types, and each conversion is pinned
  below with what it becomes. Pinning is the point: these are not bugs, they are the
  formats' limits, and a reader needs to know that a `decimal` written to CSV comes back
  as a `double`. An *unpinned* change to any of them shows up here as a failure.

`test_every_writable_file_sink_is_in_the_matrix` keeps the table honest: a sink added to
the registry without a fidelity entry fails, so this cannot quietly fall behind the
engine.

Comparison is on a NaN- and signed-zero-aware key. A plain `==` reports NaN != NaN as a
difference on every format (a false alarm) while silently accepting `-0.0` folded to `0.0`
(a real one), so it is wrong in both directions at once.
"""

from __future__ import annotations

import datetime
import decimal
import math
import pathlib

import pyarrow as pa
import pytest

import batcher as bt
from batcher.api.io_namespace import Writer

pytestmark = pytest.mark.io

#: format -> file extension. `""` means the sink writes a directory, not a file.
LOCAL_FILE_SINKS: dict[str, str] = {
    "parquet": ".parquet",
    "arrow": ".arrow",
    "avro": ".avro",
    "orc": ".orc",
    "lance": ".lance",
    "delta": "",
    "csv": ".csv",
    "json": ".json",
    "msgpack": ".msgpack",
}

#: The sinks that preserve every type class below.
EXACT_SINKS = ("parquet", "arrow", "avro", "orc", "lance", "delta")

#: ...and the ones that convert. Each cell is pinned in `LOSSY` or `TYPE_ONLY` below.
TEXT_SINKS = ("csv", "json", "msgpack")

#: Registry sinks this matrix cannot cover locally, and why. Keyed so the guard below can
#: tell "not covered because it needs a server" from "not covered because nobody added it".
NOT_LOCALLY_WRITABLE: dict[str, str] = {
    "iceberg": "needs a catalog URI; covered by tests/io/test_lakehouse_iceberg.py",
    "hudi": "Batcher supports Hudi reads only — the writer raises BackendError by design",
    "mongo": "needs a running MongoDB server",
    "snowflake": "needs Snowflake credentials",
    "adbc": "needs a live ADBC driver/database",
    "ipc": "the same writer as `arrow`, reached through a second registry name",
    "feather": "the same writer as `arrow`, reached through a second registry name",
    # The bioinformatics sinks take a **fixed schema**, not an arbitrary relation: a FASTA
    # record is (id, description, sequence) and a BED line is (chrom, start, end, ...), so
    # there is nowhere to put this matrix's `int64_precision` or `timestamp_us` column and
    # nothing it would mean if there were. Their fidelity is pinned on the schema they do
    # accept, by the `*_round_trips` tests in the two suites named here.
    "fasta": "fixed record schema; round-tripped by tests/io/test_io_fasta_fastq.py",
    "fastq": "fixed record schema; round-tripped by tests/io/test_io_fasta_fastq.py",
    "bed": "fixed record schema; round-tripped by tests/io/test_io_bed_gff_vcf.py",
    "gff": "fixed record schema; round-tripped by tests/io/test_io_bed_gff_vcf.py",
}

#: Type class -> a 4-row column carrying that class's hard values.
TYPE_COLUMNS: dict[str, pa.Array] = {
    "int64": pa.array([1, -2, None, 2**62], pa.int64()),
    # 2**53 and 2**53+1 share a float64 image, so a format that detours through a double
    # collapses them — the off-by-one no other assertion would show.
    "int64_precision": pa.array([2**53, 2**53 + 1, None, -(2**53) - 1], pa.int64()),
    # `-0.0` and NaN are the values the comparison helpers elsewhere in this suite
    # deliberately canonicalize; here they are the payload.
    "float64": pa.array([1.5, -0.0, float("nan"), None], pa.float64()),
    "float_infinities": pa.array([float("inf"), float("-inf"), 0.0, None], pa.float64()),
    # An empty string and a null must stay distinguishable, and the delimiters/quotes must
    # survive a text encoder.
    "string": pa.array(["a", "", None, "ünïcødé\n\t,\"'"]),
    "bool": pa.array([True, False, None, True], pa.bool_()),
    "date32": pa.array(
        [datetime.date(2020, 1, 1), None, datetime.date(1999, 12, 31), datetime.date(2024, 2, 29)],
        pa.date32(),
    ),
    "timestamp_us": pa.array(
        [
            datetime.datetime(2020, 1, 1, 12, 30, 15, 123456),
            None,
            datetime.datetime(1970, 1, 1),
            datetime.datetime(2038, 1, 19, 3, 14, 7),
        ],
        pa.timestamp("us"),
    ),
    "decimal": pa.array(
        [decimal.Decimal("1.23"), decimal.Decimal("-4.56"), None, decimal.Decimal("0.00")],
        pa.decimal128(10, 2),
    ),
    "binary": pa.array([b"\x00\x01", b"", None, b"\xff" * 4], pa.binary()),
    "list_int": pa.array([[1, 2], [], None, [3]], pa.list_(pa.int64())),
    "struct": pa.array(
        [{"a": 1, "b": "x"}, {"a": None, "b": None}, None, {"a": 2, "b": "y"}],
        pa.struct([("a", pa.int64()), ("b", pa.string())]),
    ),
}

#: (format, type) -> the type it comes back as, where the *values* survive but the Arrow
#: type is renamed or widened. Not data loss, but still a contract a reader depends on.
TYPE_ONLY: dict[tuple[str, str], str] = {
    # Arrow's own field-naming convention for list elements; the values are identical.
    ("parquet", "list_int"): "list<element: int64>",
    ("delta", "list_int"): "list<element: int64>",
    # ORC stores timestamps at nanosecond resolution.
    ("orc", "timestamp_us"): "timestamp[ns]",
    ("csv", "timestamp_us"): "timestamp[ns]",
    ("json", "struct"): "struct<a: double, b: string>",
}

#: (format, type) -> what the format does to the values. These are format limits, not bugs.
LOSSY: dict[tuple[str, str], str] = {
    ("csv", "float64"): "CSV has no NaN literal — NaN is written empty and reads back null",
    ("csv", "string"): "CSV cannot distinguish an empty string from a null — null reads as ''",
    ("csv", "decimal"): "CSV is untyped text — a decimal is inferred back as a double",
    ("json", "float64"): "JSON has no NaN literal — NaN reads back null",
    ("json", "float_infinities"): "JSON has no Infinity literal — +-inf reads back null",
    # ISO-8601, as `msgpack` below writes and as DuckDB/Spark do. Arrow's JSON reader infers
    # a bare date back as `timestamp[s]` and declines a sub-second instant entirely, leaving
    # it a string -- a reader limit, not a writer one. These said "epoch milliseconds" while
    # the writer actually emitted a *wrong number*: the pandas encoder it fell through to
    # reads every timestamp column's raw integers as nanoseconds, so a `timestamp[us]` was
    # divided by a million. The values are now exact at the column's own resolution.
    ("json", "date32"): "a date is written as an ISO-8601 string and reads back timestamp[s]",
    ("json", "timestamp_us"): "a timestamp is written as an ISO-8601 string and reads back string",
    ("json", "decimal"): "JSON numbers are doubles — a decimal loses its exact type",
    ("msgpack", "date32"): "a date is written as an ISO-8601 string and reads back string",
    ("msgpack", "timestamp_us"): "a timestamp is written as an ISO-8601 string",
}

#: (format, type) -> the format cannot represent this type at all and raises on write.
UNSUPPORTED: dict[tuple[str, str], str] = {
    ("csv", "binary"): "CSV is text — arbitrary bytes have no encoding",
    ("csv", "list_int"): "CSV is flat — a list has no column representation",
    ("csv", "struct"): "CSV is flat — a struct has no column representation",
    ("json", "binary"): "JSON has no byte-string type",
    ("msgpack", "decimal"): "the msgpack encoder has no decimal type",
}


def _key(value):
    """A comparison key that sees NaN as equal to NaN and keeps `-0.0` distinct from `0.0`.

    Both halves matter and they pull in opposite directions: `nan != nan` makes a plain
    `==` report a difference every format actually preserved, while `-0.0 == 0.0` makes it
    accept a signed zero that was folded. Reading the sign bit explicitly settles both.
    """
    if isinstance(value, float):
        if math.isnan(value):
            return "<nan>"
        if value == 0.0:
            return "-0.0" if math.copysign(1.0, value) < 0 else "0.0"
    if isinstance(value, list):
        return [_key(v) for v in value]
    if isinstance(value, dict):
        return {k: _key(v) for k, v in value.items()}
    return value


def _table(type_name: str) -> pa.Table:
    """A two-column table: a stable id, plus the type class under test."""
    return pa.table({"id": pa.array([0, 1, 2, 3], pa.int64()), type_name: TYPE_COLUMNS[type_name]})


def _round_trip(tmp_path: pathlib.Path, fmt: str, type_name: str) -> pa.Table:
    """Write the column to a real file with `fmt`, read it back, return what came back."""
    extension = LOCAL_FILE_SINKS[fmt]
    target = str(tmp_path / (f"data{extension}" if extension else "table"))
    getattr(Writer(bt.from_arrow(_table(type_name))), fmt)(target)
    return bt.read(target).collect()


def _values(table: pa.Table, column: str) -> list:
    return [_key(v) for v in table.column(column).to_pylist()]


@pytest.mark.parametrize("type_name", sorted(TYPE_COLUMNS))
@pytest.mark.parametrize("fmt", sorted(EXACT_SINKS))
def test_a_binary_format_round_trips_the_type_exactly(tmp_path, fmt, type_name):
    """Values and Arrow type both survive — the guarantee these formats exist for."""
    back = _round_trip(tmp_path, fmt, type_name)
    assert type_name in back.column_names, f"{fmt} dropped the {type_name} column entirely"
    assert _values(back, type_name) == _values(_table(type_name), type_name), (
        f"{fmt} changed the values of a {type_name} column on a write->read round trip"
    )
    expected_type = TYPE_ONLY.get((fmt, type_name), str(TYPE_COLUMNS[type_name].type))
    assert str(back.schema.field(type_name).type) == expected_type, (
        f"{fmt} read {type_name} back as {back.schema.field(type_name).type}, expected "
        f"{expected_type}. If this is an intended change, update TYPE_ONLY and say why."
    )


@pytest.mark.parametrize("type_name", sorted(TYPE_COLUMNS))
@pytest.mark.parametrize("fmt", sorted(TEXT_SINKS))
def test_a_text_format_matches_its_pinned_fidelity(tmp_path, fmt, type_name):
    """Each text-format conversion is exactly the one recorded above — no more, no less.

    A cell that is neither in `LOSSY` nor `UNSUPPORTED` must round-trip its values intact;
    one that is must behave as described. Either direction of drift fails: a format that
    starts losing a type it used to keep, and one that starts keeping a type recorded as
    lossy (in which case delete the entry — that is a fix worth noticing).
    """
    key = (fmt, type_name)
    if key in UNSUPPORTED:
        with pytest.raises(Exception):  # noqa: B017 — the encoders raise varied types
            _round_trip(tmp_path, fmt, type_name)
        return

    back = _round_trip(tmp_path, fmt, type_name)
    assert type_name in back.column_names, f"{fmt} dropped the {type_name} column"
    same_values = _values(back, type_name) == _values(_table(type_name), type_name)
    if key in LOSSY:
        assert not same_values, (
            f"{fmt} now round-trips {type_name} intact, but it is pinned as lossy "
            f"({LOSSY[key]}). Delete the LOSSY entry — the format got better."
        )
    else:
        assert same_values, (
            f"{fmt} changed the values of a {type_name} column, which is not a pinned "
            f"conversion. Either this is a regression, or add it to LOSSY with the reason."
        )


@pytest.mark.parametrize("type_name", sorted(TYPE_COLUMNS))
@pytest.mark.parametrize("fmt", sorted(LOCAL_FILE_SINKS))
def test_the_row_count_and_column_set_always_survive(tmp_path, fmt, type_name):
    """Whatever a format does to types, it must not lose or invent a row or a column.

    Asserted for the lossy formats too, so a writer that silently drops a null row — the
    failure a per-type value check misses when the type is also being converted — is
    caught everywhere rather than only where the values are comparable.
    """
    if (fmt, type_name) in UNSUPPORTED:
        pytest.skip(f"{fmt} cannot represent {type_name}: {UNSUPPORTED[(fmt, type_name)]}")
    back = _round_trip(tmp_path, fmt, type_name)
    assert back.num_rows == 4, f"{fmt}/{type_name}: {back.num_rows} rows, wrote 4"
    assert set(back.column_names) == {"id", type_name}, (
        f"{fmt}/{type_name}: columns {back.column_names}"
    )
    assert back.column("id").to_pylist() == [0, 1, 2, 3], (
        f"{fmt}/{type_name}: the id column did not survive intact"
    )


def test_every_writable_file_sink_is_in_the_matrix():
    """A sink added to the registry must declare its fidelity, or say why it cannot.

    Without this the matrix silently falls behind the engine: a new format lands, nothing
    round-trips it, and the table still looks complete.
    """
    from batcher.io.formats import SINKS

    uncovered = set(SINKS) - set(LOCAL_FILE_SINKS) - set(NOT_LOCALLY_WRITABLE)
    assert not uncovered, (
        f"sinks with no round-trip fidelity coverage: {sorted(uncovered)}. Add each to "
        f"LOCAL_FILE_SINKS with its extension, or to NOT_LOCALLY_WRITABLE with the reason "
        f"it cannot be exercised against a local file."
    )


def test_the_exemption_list_stays_honest():
    """An exemption for a sink that no longer exists, or that the matrix does cover."""
    from batcher.io.formats import SINKS

    for fmt, reason in NOT_LOCALLY_WRITABLE.items():
        assert fmt in SINKS, f"NOT_LOCALLY_WRITABLE names {fmt!r}, which is not a sink"
        assert reason.strip(), f"NOT_LOCALLY_WRITABLE[{fmt!r}] needs a reason"
    overlap = set(NOT_LOCALLY_WRITABLE) & set(LOCAL_FILE_SINKS)
    assert not overlap, f"exempted but also in the matrix: {sorted(overlap)}"


def test_the_comparison_key_sees_what_a_plain_equality_cannot():
    """The matrix rests on `_key`; if it collapsed these, every cell above would be blind.

    `==` gets both of these wrong in opposite directions, which is exactly why the helper
    exists rather than a direct comparison.
    """
    assert _key(float("nan")) == _key(float("nan")), "NaN must compare equal to NaN"
    assert _key(-0.0) != _key(0.0), "-0.0 must stay distinct from 0.0"
    assert _key([-0.0, float("nan")]) == ["-0.0", "<nan>"]
    assert _key({"x": -0.0}) == {"x": "-0.0"}
    # ...and it must not disturb ordinary values.
    assert _key(1.5) == 1.5 and _key(None) is None and _key("a") == "a"
