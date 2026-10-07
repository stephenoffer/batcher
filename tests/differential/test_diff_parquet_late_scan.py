"""The engine's own Parquet read with its `Filter` decoded first returns what DuckDB returns.

When the engine reads a Parquet source row group by row group (`execute_plan_parquet`), the
`Filter` directly over the scan is handed to the reader as a late-materialization filter
(`bc_py::chunked::late_filter`, `bc_io::late`): the predicate's columns are decoded first, the
engine's own predicate is evaluated on them, and every other column is decoded only for the
rows it keeps. The predicate's native translation (`to_native_predicate`) also prunes row
groups on their footers, now keeping the conjuncts it can express when another one -- a date
-- cannot be.

Both remove rows before the engine sees them, so both are held to DuckDB here over files built
to break them: several row groups per file and several pages per row group, dictionary- and
plain-encoded files side by side, nulls in every column, NaN and -0.0, a dictionary-typed
column, a time-zoned timestamp, a decimal, a struct and a list carried through as deferred
columns, a predicate that keeps nothing, a conjunction split into stages, one that is not, and
the negations and disjunctions a partial translation must not narrow.
"""

from __future__ import annotations

import datetime as dt
import decimal

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from _harness import assert_same, assert_same_for_query

pytestmark = pytest.mark.differential

_FILES = 4
_ROWS_PER_FILE = 6_000
_ROW_GROUP = 1_000


def _part(part: int) -> pa.Table:
    base = part * _ROWS_PER_FILE
    ids = list(range(base, base + _ROWS_PER_FILE))
    floats = [float("nan"), -0.0, 0.0, None, 0.75, 0.25, 2.5]
    return pa.table(
        {
            "id": pa.array(ids, pa.int64()),
            "k": pa.array([None if i % 17 == 0 else i % 7 for i in ids], pa.int32()),
            "f": pa.array([floats[i % len(floats)] for i in ids], pa.float64()),
            "cat": pa.array(
                [None if i % 19 == 0 else "abcd"[i % 4] for i in ids]
            ).dictionary_encode(),
            "txt": pa.array(
                [
                    None
                    if i % 13 == 0
                    else f"{'needle' if i % 173 == 0 else 'hay'}-{i}-" + "z" * (i % 50)
                    for i in ids
                ]
            ),
            # A low-cardinality plain string: read as a `Dictionary` by a stage over it alone.
            "mode": pa.array(
                [
                    None if i % 11 == 0 else ("AIR", "MAIL", "RAIL", "SHIP", "AIR REG")[i % 5]
                    for i in ids
                ]
            ),
            "d": pa.array([dt.date(2020, 1, 1) + dt.timedelta(days=i // 100) for i in ids]),
            "ts": pa.array(
                [None if i % 23 == 0 else i * 1_000_003 for i in ids],
                pa.timestamp("us", tz="America/New_York"),
            ),
            "dec": pa.array([decimal.Decimal(i % 10_000) / 100 for i in ids], pa.decimal128(12, 2)),
            "st": pa.array(
                [None if i % 29 == 0 else {"x": i, "y": f"s{i % 11}"} for i in ids],
                pa.struct([("x", pa.int64()), ("y", pa.string())]),
            ),
            "lst": pa.array(
                [None if i % 31 == 0 else list(range(i % 4)) for i in ids], pa.list_(pa.int64())
            ),
        }
    )


def _whole() -> pa.Table:
    return pa.concat_tables(_part(p) for p in range(_FILES))


@pytest.fixture(scope="module")
def hits_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("late_scan")
    for part in range(_FILES):
        pq.write_table(
            _part(part),
            root / f"part-{part}.parquet",
            row_group_size=_ROW_GROUP,
            data_page_size=2_048,
            # Alternate files are plain-encoded, so both decoders serve a selective read.
            use_dictionary=part % 2 == 0,
        )
    return root


@pytest.fixture
def engine_read(monkeypatch):
    """Force the engine's row-group read on these small inputs and count its calls."""
    import batcher.core as core
    from batcher.api.orchestration import chunked

    monkeypatch.setattr(chunked, "units_worthy", lambda _bytes: True)
    calls: list[int] = []
    real = core.execute_local_parquet

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(core, "execute_local_parquet", spy)
    return calls


_QUERIES = [
    # A selective `LIKE` (no native translation) with every column deferred behind it.
    "SELECT * FROM t WHERE txt LIKE '%needle%' ORDER BY id LIMIT 25",
    "SELECT id, cat, d, ts, dec FROM t WHERE txt LIKE '%needle%' ORDER BY id",
    # Keeps nothing: every row group's filter empties it.
    "SELECT * FROM t WHERE txt LIKE '%no such text%'",
    # A conjunction of infallible conjuncts, split into stages cheapest first.
    "SELECT id, txt, dec FROM t WHERE k = 3 AND cat <> 'b' ORDER BY id",
    "SELECT id, txt FROM t WHERE k IS NULL AND f < 0.5 ORDER BY id",
    # -0.0 equals 0.0 in both engines; the mask must agree.
    "SELECT count(*) AS n, sum(dec) AS s FROM t WHERE f = 0.0",
    # A dictionary-typed predicate column, decoded to its values before the mask.
    "SELECT id, cat, txt FROM t WHERE cat = 'c' AND id % 10 = 1 ORDER BY id",
    "SELECT cat, count(*) AS n, sum(f) AS s FROM t WHERE txt LIKE '%needle%' OR k IS NULL "
    "GROUP BY cat",
    # A stage over one low-cardinality string, its mask computed once per dictionary value:
    # membership, a pattern, a null test, a value no row holds, and the column in the output.
    "SELECT id, mode, txt FROM t WHERE mode IN ('MAIL', 'SHIP') ORDER BY id",
    "SELECT mode, count(*) AS n, sum(dec) AS s FROM t WHERE mode = 'AIR' OR mode IS NULL "
    "GROUP BY mode",
    "SELECT id, txt FROM t WHERE mode LIKE 'AIR%' AND k = 3 ORDER BY id",
    "SELECT count(*) AS n, sum(dec) AS s FROM t WHERE mode <> 'RAIL' AND txt LIKE '%needle%'",
    "SELECT count(*) AS n FROM t WHERE mode IN ('NOPE')",
    # A date the native translation cannot express beside a key it can: the key still prunes.
    "SELECT count(*) AS n, sum(dec) AS s, max(d) AS hi FROM t "
    "WHERE id >= 17000 AND d >= DATE '2020-07-01'",
    "SELECT id, ts FROM t WHERE d < DATE '2020-01-15' AND id > 700 ORDER BY id",
    # TPC-H q12's shape: a date range, column-to-column comparisons and a low-cardinality
    # string IN, grouped into one stage for the rest and a dictionary stage for the string;
    # then the same split across a subquery's filter and the outer one, which can reach the
    # scan as two stacked filters.
    "SELECT mode, count(*) AS n, sum(dec) AS s FROM t WHERE mode IN ('MAIL', 'SHIP') "
    "AND id < k * 4000 AND d >= DATE '2020-02-01' AND d < DATE '2020-06-01' GROUP BY mode",
    "SELECT count(*) AS n, sum(f) AS s FROM (SELECT * FROM t WHERE d >= DATE '2020-03-01') s "
    "WHERE mode IN ('AIR', 'SHIP') AND k < id % 7",
    # A top-N over the scan: the narrow sort columns are read and sorted first, and only the
    # winners are fetched whole -- descending keys, an offset, deferred nested and decimal
    # columns, and a filter that keeps every row. (Ties the sort breaks by input order are
    # left to the engine's own test, `stream::chunked::top_n`: DuckDB breaks them freely.)
    "SELECT * FROM t WHERE cat = 'b' AND k = 3 ORDER BY ts DESC, id LIMIT 5",
    "SELECT id, k, txt FROM t WHERE f = 0.25 ORDER BY k, d DESC, id LIMIT 8",
    "SELECT id, st, lst, dec FROM t WHERE k = 2 ORDER BY dec DESC, id LIMIT 6",
    "SELECT * FROM t WHERE dec <> 0.5 ORDER BY d DESC, id LIMIT 3",
    "SELECT id, cat, ts, txt FROM t WHERE cat <> 'a' ORDER BY id DESC LIMIT 4 OFFSET 2",
    "SELECT * FROM t WHERE dec = 0.005 ORDER BY id LIMIT 3",
    # Neither a negation nor a disjunction may be narrowed by the partial translation.
    "SELECT count(*) AS n FROM t WHERE NOT (id >= 17000 AND d >= DATE '2020-07-01')",
    "SELECT count(*) AS n FROM t WHERE id < 3000 OR d >= DATE '2020-08-01'",
    "SELECT count(*) AS n FROM t WHERE NOT (id < 3000 OR d >= DATE '2020-08-01')",
]


@pytest.mark.parametrize("query", _QUERIES)
def test_late_scan_matches_duckdb(duck, hits_dir, engine_read, query):
    duck.register("t", _whole())
    s = bt.Session()
    s.register("t", bt.read.parquet(str(hits_dir / "*.parquet")))
    got = s.sql(query).collect()
    assert engine_read, "the query did not take the engine's row-group read"
    assert_same_for_query(got, duck.sql(query), query)


#: Held to the in-memory engine rather than to DuckDB: Batcher orders NaN above every float, so
#: ``NaN > 0.5`` keeps the row, where DuckDB over the same Arrow data drops it. That divergence
#: predates this read path (the in-memory engine gives the same answer) and is not what these
#: test; what they test is that the mask computed in the decode is the engine's own.
_NAN_QUERIES = [
    "SELECT id, f, txt FROM t WHERE f > 0.5 ORDER BY id LIMIT 60",
    "SELECT id, txt FROM t WHERE k IS NULL AND f > 0.5 ORDER BY id",
    "SELECT count(*) AS n FROM t WHERE f >= 2.5 AND txt LIKE '%zz%'",
]


@pytest.mark.parametrize("query", _NAN_QUERIES)
def test_late_scan_matches_the_in_memory_engine_on_nan(hits_dir, engine_read, query):
    from _harness import assert_tables_equal

    s = bt.Session()
    s.register("t", bt.read.parquet(str(hits_dir / "*.parquet")))
    got = s.sql(query).collect()
    assert engine_read, "the query did not take the engine's row-group read"
    resident = bt.Session()
    resident.register("t", bt.from_arrow(_whole()))
    assert_tables_equal(got, resident.sql(query).collect(), ordered="ORDER BY" in query)


def test_nested_columns_deferred_behind_the_filter(duck, hits_dir, engine_read):
    """A struct and a list decoded only for the rows the `LIKE` keeps come back intact."""
    got = (
        bt.read.parquet(str(hits_dir / "*.parquet"))
        .filter(bt.col("txt").str.contains("needle"))
        .select("id", "st", "lst")
        .collect()
    )
    assert engine_read, "the query did not take the engine's row-group read"
    duck.register("t", _whole())
    assert_same(got, duck.sql("SELECT id, st, lst FROM t WHERE contains(txt, 'needle')"))
