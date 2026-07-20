"""Warehouse type fidelity: what a database column *becomes* when it crosses into Arrow.

A wrong Arrow type at the IO boundary is the silent-failure shape `CLAUDE.md` calls out by
name — "a `null` where an `int64` belongs". `schema()` is what the plan types every operator
against (`core/scan_only.py::_as_table` builds an empty relation straight from it), so a
column mistyped here is mistyped for the whole query, and no differential test downstream can
see it.

These tests drive `DBAPISource` with stdlib `sqlite3`, so they need no optional dependency and
always run. sqlite3 is also the *worst case* for this path and therefore the right probe: it
reports ``None`` for every `cursor.description` type code, so `_arrow_type` can resolve
nothing and the source falls back to inferring Arrow types from real Python values — the code
path every driver without PEP 249 type objects takes.

Each test asserts the **actual** Arrow type and the **round-tripped value**. Where today's
behavior is wrong, the test states the correct behavior and is marked `xfail` — baking a bug
in as "expected" would make this file worse than nothing.

A sqlite caveat worth stating once, because it is sqlite's bug and not Batcher's: a column
declared ``NUMERIC``/``DECIMAL`` has NUMERIC *affinity*, so sqlite itself coerces a stored
decimal string to REAL and loses the digits before any driver sees it. The decimal tests below
use a ``TEXT``-affinity declared type so they measure the Arrow boundary rather than sqlite.
"""

from __future__ import annotations

import datetime
import math
import sqlite3
from decimal import Decimal

import pyarrow as pa
import pytest

from batcher._internal.errors import BackendError
from batcher.io.formats.sql.dbapi import DBAPISource

pytestmark = pytest.mark.unit

# sqlite3's converter registry is process-global, so these decltypes are prefixed to keep
# them from colliding with (or being clobbered by) any other test that registers converters.
sqlite3.register_converter("BCDECTEXT", lambda b: Decimal(b.decode()))
sqlite3.register_converter("BCDATE", lambda b: datetime.date.fromisoformat(b.decode()))
sqlite3.register_converter("BCTIMESTAMP", lambda b: datetime.datetime.fromisoformat(b.decode()))
sqlite3.register_converter("BCBOOL", lambda b: b != b"0")


def _db(tmp_path, name: str, script: str, rows: list[tuple] | None = None, insert: str = "") -> str:
    """Build a throwaway sqlite database and return its path."""
    path = str(tmp_path / f"{name}.db")
    conn = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES)
    try:
        conn.executescript(script)
        for row in rows or []:
            conn.execute(insert, row)
        conn.commit()
    finally:
        conn.close()
    return path


def _source(path: str, *, typed: bool = False, **kwargs) -> DBAPISource:
    """A `DBAPISource` over `path`; `typed` turns on sqlite's declared-type converters."""
    connect_kwargs: dict = {"database": path}
    if typed:
        connect_kwargs["detect_types"] = sqlite3.PARSE_DECLTYPES
    return DBAPISource(module="sqlite3", connect_kwargs=connect_kwargs, **kwargs)


def _one(source: DBAPISource) -> dict:
    """The single batch of a small read, as a plain dict of columns."""
    batches = source.read()
    assert len(batches) == 1
    return batches[0].to_pydict()


# --------------------------------------------------------------------------------------
# Types that survive the boundary intact
# --------------------------------------------------------------------------------------


def test_integers_are_int64_across_the_full_signed_range(tmp_path):
    """INTEGER → int64, with both int64 bounds round-tripping exactly."""
    path = _db(
        tmp_path,
        "ints",
        "CREATE TABLE t (lo INTEGER, hi INTEGER)",
        [(-(2**63), 2**63 - 1)],
        "INSERT INTO t VALUES (?, ?)",
    )
    src = _source(path, table="t")
    assert src.schema() == pa.schema([("lo", pa.int64()), ("hi", pa.int64())])
    assert _one(src) == {"lo": [-(2**63)], "hi": [2**63 - 1]}


def test_real_is_double_with_full_float64_precision(tmp_path):
    """REAL → double, keeping all 17 significant digits.

    A detour through float32 — a plausible "narrow types are normalized anyway" mistake —
    would round the first value in the 8th digit and be invisible in a casual assertion.

    NaN comes back as NULL, and that is sqlite, not Batcher: SQLite has no NaN storage
    class and writes ``typeof(v) = 'null'`` for one (verified directly against the driver).
    A NULL is therefore the faithful reading of what the database actually holds.
    """
    precise = 0.1234567890123456789
    path = _db(
        tmp_path,
        "reals",
        "CREATE TABLE t (v REAL)",
        [(precise,), (1.5,), (float("nan"),)],
        "INSERT INTO t VALUES (?)",
    )
    src = _source(path, table="t")
    assert src.schema() == pa.schema([("v", pa.float64())])
    values = _one(src)["v"]
    assert values[0] == precise
    assert values[1] == 1.5
    assert values[2] is None
    assert not any(v is not None and math.isnan(v) for v in values)


def test_text_and_blob_keep_their_bytes(tmp_path):
    """TEXT → string and BLOB → binary, with non-UTF8 and NUL bytes preserved."""
    path = _db(
        tmp_path,
        "bytes",
        "CREATE TABLE t (s TEXT, b BLOB)",
        [("héllo", b"\x00\xde\xad\xbe\xef")],
        "INSERT INTO t VALUES (?, ?)",
    )
    src = _source(path, table="t")
    assert src.schema() == pa.schema([("s", pa.string()), ("b", pa.binary())])
    assert _one(src) == {"s": ["héllo"], "b": [b"\x00\xde\xad\xbe\xef"]}


def test_decimal_keeps_precision_and_scale(tmp_path):
    """A driver that yields `Decimal` lands as decimal128 — no float64 in between.

    This is the fidelity question that matters most for money columns. 29 significant digits
    is well past float64's 15 to 17, so a detour through double would be visible in the value.
    """
    value = Decimal("12345678901234567890.123456789")
    path = _db(
        tmp_path,
        "dec",
        "CREATE TABLE t (v BCDECTEXT)",
        [(str(value),)],
        "INSERT INTO t VALUES (?)",
    )
    src = _source(path, typed=True, table="t")
    assert src.schema() == pa.schema([("v", pa.decimal128(29, 9))])
    assert _one(src) == {"v": [value]}


def test_decimals_of_mixed_scale_widen_rather_than_truncate(tmp_path):
    """Two scales in one column widen to the larger scale; neither value is rounded away."""
    path = _db(
        tmp_path,
        "dec2",
        "CREATE TABLE t (v BCDECTEXT)",
        [("1.5",), ("1.25",)],
        "INSERT INTO t VALUES (?)",
    )
    src = _source(path, typed=True, table="t")
    assert src.schema() == pa.schema([("v", pa.decimal128(3, 2))])
    assert _one(src) == {"v": [Decimal("1.50"), Decimal("1.25")]}


def test_date_timestamp_and_timezone_are_distinct_and_the_offset_survives(tmp_path):
    """DATE → date32, TIMESTAMP → timestamp[us], and a tz-aware value keeps its offset.

    Collapsing a DATE to a TIMESTAMP, or silently dropping a ``+05:30``, is the classic
    warehouse-boundary correctness bug: both are wrong by hours and neither raises.
    """
    path = _db(
        tmp_path,
        "times",
        "CREATE TABLE t (d BCDATE, ts BCTIMESTAMP, tz BCTIMESTAMP)",
        [("2024-01-02", "2024-01-02 03:04:05.000006", "2024-01-02 03:04:05+05:30")],
        "INSERT INTO t VALUES (?, ?, ?)",
    )
    src = _source(path, typed=True, table="t")
    schema = src.schema()
    assert schema.field("d").type == pa.date32()
    assert schema.field("ts").type == pa.timestamp("us")
    tz_type = schema.field("tz").type
    assert pa.types.is_timestamp(tz_type)
    assert tz_type.tz is not None, "the timezone must not be dropped at the Arrow boundary"

    row = _one(src)
    assert row["d"] == [datetime.date(2024, 1, 2)]
    assert row["ts"] == [datetime.datetime(2024, 1, 2, 3, 4, 5, 6)]
    # Compared as an instant: the offset is what makes this 21:34:05 UTC on the 1st.
    assert row["tz"][0].utctimetuple() == datetime.datetime(2024, 1, 1, 21, 34, 5).utctimetuple()


def test_untyped_date_column_is_string_not_a_silently_wrong_timestamp(tmp_path):
    """Without sqlite's converters a date is TEXT — and must stay `string`, never a guess.

    Inferring `timestamp` from a string that merely *looks* like a date is how a locale- or
    format-dependent misparse becomes a wrong answer. `string` is honest.
    """
    path = _db(
        tmp_path,
        "textdate",
        "CREATE TABLE t (d DATE)",
        [("2024-01-02",)],
        "INSERT INTO t VALUES (?)",
    )
    src = _source(path, table="t")
    assert src.schema() == pa.schema([("d", pa.string())])
    assert _one(src) == {"d": ["2024-01-02"]}


def test_boolean_without_a_converter_is_int64(tmp_path):
    """sqlite has no boolean storage class, so BOOLEAN arrives as int64 1/0.

    This is the driver's semantics, not a Batcher mistype: sqlite genuinely stored integers.
    Pinned so a future change to `_arrow_type` that starts guessing `bool` from a column
    *name* or declared type — and would mistype a 0/1/2 status column — fails here.
    """
    path = _db(
        tmp_path,
        "boolint",
        "CREATE TABLE t (flag BOOLEAN)",
        [(1,), (0,)],
        "INSERT INTO t VALUES (?)",
    )
    src = _source(path, table="t")
    assert src.schema() == pa.schema([("flag", pa.int64())])
    assert _one(src) == {"flag": [1, 0]}


def test_boolean_with_a_converter_is_arrow_bool(tmp_path):
    """When the driver yields real `bool`s, Arrow gets `bool` — not int64."""
    path = _db(
        tmp_path,
        "boolreal",
        "CREATE TABLE t (flag BCBOOL)",
        [("1",), ("0",)],
        "INSERT INTO t VALUES (?)",
    )
    src = _source(path, typed=True, table="t")
    assert src.schema() == pa.schema([("flag", pa.bool_())])
    assert _one(src) == {"flag": [True, False]}


def test_schema_is_identical_across_every_batch_of_a_multi_batch_read(tmp_path):
    """`batch_size` must not be observable in the types — only in the batching."""
    path = _db(
        tmp_path,
        "multi",
        "CREATE TABLE t (i INTEGER, s TEXT)",
        [(n, f"r{n}") for n in range(7)],
        "INSERT INTO t VALUES (?, ?)",
    )
    src = _source(path, table="t", batch_size=2)
    expected = pa.schema([("i", pa.int64()), ("s", pa.string())])
    batches = list(src.iter_batches())
    assert len(batches) == 4
    assert all(b.schema == expected for b in batches)
    assert [v for b in batches for v in b.column(0).to_pylist()] == list(range(7))


def test_pushdown_does_not_change_the_types_it_returns(tmp_path):
    """A pushed projection and predicate narrow the rows, never the column types."""
    path = _db(
        tmp_path,
        "push",
        "CREATE TABLE t (i INTEGER, f REAL, s TEXT)",
        [(1, 1.5, "a"), (2, 2.5, "b")],
        "INSERT INTO t VALUES (?, ?, ?)",
    )
    src = _source(path, table="t")
    predicate = {
        "e": "binary",
        "op": "gt",
        "left": {"e": "col", "name": "i"},
        "right": {"e": "lit", "value": {"int": 1}},
    }
    batches = src.read(projection=["i", "f"], predicate=predicate)
    assert len(batches) == 1
    assert batches[0].schema == pa.schema([("i", pa.int64()), ("f", pa.float64())])
    assert batches[0].to_pydict() == {"i": [2], "f": [2.5]}


# --------------------------------------------------------------------------------------
# Where the boundary is wrong today
# --------------------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "BUG: an all-NULL column is typed pa.null(). dbapi.py:191 returns null-typed fields "
        "when nothing can be inferred, and schema() is what the plan types operators against "
        "(core/scan_only.py::_as_table). The declared INTEGER is available from sqlite's "
        "PRAGMA/declared type and is thrown away. This is exactly the 'null where an int64 "
        "belongs' failure CLAUDE.md names. schema_override= is the documented workaround, "
        "which the next test pins."
    ),
)
def test_null_only_column_keeps_its_declared_type(tmp_path):
    """A column whose every value is NULL must still be typed by its declaration."""
    path = _db(
        tmp_path,
        "nullonly",
        "CREATE TABLE t (i INTEGER, n INTEGER)",
        [(1, None), (2, None)],
        "INSERT INTO t VALUES (?, ?)",
    )
    src = _source(path, table="t")
    assert src.schema() == pa.schema([("i", pa.int64()), ("n", pa.int64())])


def test_null_only_column_is_typed_null_today(tmp_path):
    """The behavior as it actually is, so the bug above cannot regress further unnoticed."""
    path = _db(
        tmp_path,
        "nullonly2",
        "CREATE TABLE t (i INTEGER, n INTEGER)",
        [(1, None)],
        "INSERT INTO t VALUES (?, ?)",
    )
    src = _source(path, table="t")
    assert src.schema() == pa.schema([("i", pa.int64()), ("n", pa.null())])


def test_schema_override_recovers_a_null_only_column(tmp_path):
    """`schema_override=` is the stated escape hatch, and it must type the values too."""
    path = _db(
        tmp_path,
        "override",
        "CREATE TABLE t (i INTEGER, n INTEGER)",
        [(1, None)],
        "INSERT INTO t VALUES (?, ?)",
    )
    declared = pa.schema([("i", pa.int64()), ("n", pa.int64())])
    src = _source(path, table="t", schema_override=declared)
    assert src.schema() == declared
    batches = src.read()
    assert batches[0].schema == declared
    assert batches[0].to_pydict() == {"i": [1], "n": [None]}


@pytest.mark.xfail(
    strict=True,
    reason=(
        "BUG: an empty result set is typed all-null. The WHERE 1 = 0 probe returns no rows, "
        "probe_is_typed() correctly rejects it, and the full-read fallback then also sees no "
        "rows — so dbapi.py:191 types every column null. sqlite reports the column names in "
        "cursor.description and their declared types via PRAGMA table_info, so the types are "
        "recoverable. An empty relation with the right schema is a normal, correct answer; an "
        "empty relation with null columns poisons every operator typed against it."
    ),
)
def test_empty_result_keeps_the_column_types(tmp_path):
    """Zero rows is not zero type information."""
    path = _db(
        tmp_path,
        "empty",
        "CREATE TABLE t (i INTEGER, s TEXT)",
        [(1, "a")],
        "INSERT INTO t VALUES (?, ?)",
    )
    src = _source(path, query="SELECT i, s FROM t WHERE 1 = 0")
    assert src.schema() == pa.schema([("i", pa.int64()), ("s", pa.string())])


def test_empty_result_returns_no_batches(tmp_path):
    """Whatever the schema says, an empty read yields no batches rather than raising."""
    path = _db(
        tmp_path,
        "empty2",
        "CREATE TABLE t (i INTEGER)",
        [(1,)],
        "INSERT INTO t VALUES (?)",
    )
    assert _source(path, query="SELECT i FROM t WHERE 1 = 0").read() == []


def test_null_in_the_first_batch_does_not_break_the_rest_of_the_read(tmp_path):
    """NULLs arriving before real values must not make the column unreadable."""
    path = _db(
        tmp_path,
        "nullfirst",
        "CREATE TABLE t (x INTEGER)",
        [(None,), (7,)],
        "INSERT INTO t VALUES (?)",
    )
    src = _source(path, table="t", batch_size=1)
    values = [v for b in src.iter_batches() for v in b.column(0).to_pylist()]
    assert values == [None, 7]


def test_a_float_arriving_after_an_int_batch_raises_instead_of_truncating(tmp_path):
    """A value that cannot fit the established type must fail loudly, not round.

    This used to return ``2`` for ``2.5``: batch one inferred `int64` and every later
    batch was force-cast to it, so the fraction vanished with no error at all.

    Widening the column retroactively is the intuitive fix and is not available. Batch one
    has already been handed downstream as `int64`, and the engine concatenates a source's
    batches with `pa.Table.from_batches` (`core/scan_only.py:96`), which requires every
    batch to share one schema. Re-typing after the fact would therefore mean buffering the
    entire relation before emitting anything — which is exactly the streaming property
    this source exists to provide. So the honest outcome is a typed error naming the
    column and pointing at `schema_override=`, which resolves it in one line.
    """
    path = _db(
        tmp_path,
        "widen",
        "CREATE TABLE t (x)",
        [(1,), (2.5,)],
        "INSERT INTO t VALUES (?)",
    )
    src = _source(path, table="t", batch_size=1)
    with pytest.raises(BackendError, match="changed type mid-read"):
        [v for b in src.iter_batches() for v in b.column(0).to_pylist()]


def test_schema_override_resolves_a_mid_read_type_change(tmp_path):
    """The escape hatch the error names must actually work."""
    path = _db(
        tmp_path,
        "widen_override",
        "CREATE TABLE t (x)",
        [(1,), (2.5,)],
        "INSERT INTO t VALUES (?)",
    )
    src = _source(
        path,
        table="t",
        batch_size=1,
        schema_override=pa.schema([("x", pa.float64())]),
    )
    assert [v for b in src.iter_batches() for v in b.column(0).to_pylist()] == [1.0, 2.5]


def test_an_int_arriving_after_a_float_batch_widens_correctly(tmp_path):
    """The benign direction of the same mechanism: int under a double schema is exact."""
    path = _db(
        tmp_path,
        "widen2",
        "CREATE TABLE t (x)",
        [(2.5,), (1,)],
        "INSERT INTO t VALUES (?)",
    )
    src = _source(path, table="t", batch_size=1)
    batches = list(src.iter_batches())
    assert all(b.schema.field(0).type == pa.float64() for b in batches)
    assert [v for b in batches for v in b.column(0).to_pylist()] == [2.5, 1.0]


def test_a_genuinely_mixed_column_raises_rather_than_coercing(tmp_path):
    """int and str in one sqlite column is unrepresentable — it must fail loudly, and does.

    This is the *correct* outcome and the counterpoint to the truncation bug above: Arrow has
    no type holding both, so raising beats inventing one. Pinned so a future "be permissive"
    change cannot quietly start stringifying the integer.
    """
    path = _db(
        tmp_path,
        "mixed",
        "CREATE TABLE t (x)",
        [(1,), ("str",)],
        "INSERT INTO t VALUES (?)",
    )
    src = _source(path, table="t")
    with pytest.raises(pa.ArrowInvalid):
        src.read()
