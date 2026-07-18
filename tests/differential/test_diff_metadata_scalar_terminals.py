"""Scalar metadata terminals equal DuckDB — over a real Parquet footer and in memory.

`min` / `max` / `n_unique` / `n_null` / `has_nulls` / `all_null` are answered from
metadata when provably exact and otherwise executed; either way the returned scalar
MUST equal DuckDB's executed answer. Each case runs twice — once over a Parquet file
(real footer stats drive the shortcut) and once over an in-memory source (the shortcut
falls back to execution) — so both the metadata path and the fallback are proven equal
to the oracle. Covers NULLs, empty input, a single row, an all-null column, and the
int/float/str/date/bool type edges.
"""

from __future__ import annotations

import datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt

# One table exercising every type edge plus nulls, an all-null column, and a
# constant column (whose exact stats are the strongest case for the shortcut).
_TABLE = pa.table(
    {
        "i": pa.array([3, 1, 2, None, 5], type=pa.int64()),
        "f": pa.array([1.5, None, 2.5, 2.5, 4.0], type=pa.float64()),
        "s": pa.array(["b", "a", None, "a", "c"], type=pa.string()),
        "d": pa.array(
            [
                datetime.date(2024, 1, 3),
                datetime.date(2024, 1, 1),
                None,
                None,
                datetime.date(2024, 1, 5),
            ],
            type=pa.date32(),
        ),
        "b": pa.array([True, False, True, None, True], type=pa.bool_()),
        "allnull": pa.array([None] * 5, type=pa.int64()),
        "k": pa.array([7, 7, 7, 7, 7], type=pa.int64()),
    }
)

# Types the engine's MIN/MAX aggregate can execute (so the in-memory fallback runs). The
# engine now reduces temporal (Date32) and Boolean columns too — previously it could only
# answer those from the Parquet footer, a metadata-vs-engine disagreement — so `d`/`b` are
# exercised through real execution here, not only the footer shortcut below.
_ORDERABLE = ["i", "f", "s", "d", "b", "k"]
_FOOTER_ORDERABLE = ["i", "f", "d", "b", "k"]
_ALL_COLS = ["i", "f", "s", "d", "b", "allnull", "k"]


@pytest.fixture
def pq_path(tmp_path):
    path = str(tmp_path / "t.parquet")
    pq.write_table(_TABLE, path)
    return path


def _duck(con):
    con.register("t", _TABLE)
    return con


def _sources(pq_path):
    """The two sources every case is checked over: real footer, then in-memory."""
    return {"parquet": bt.read.parquet(pq_path), "memory": bt.from_arrow(_TABLE)}


@pytest.mark.differential
@pytest.mark.parametrize("col", _ORDERABLE)
def test_min_max_match_duckdb(pq_path, duck, col):
    _duck(duck)
    dmin, dmax = duck.execute(f"select min({col}), max({col}) from t").fetchone()
    for ds in _sources(pq_path).values():
        assert ds.min(col) == dmin
        assert ds.max(col) == dmax


@pytest.mark.differential
@pytest.mark.parametrize("col", _FOOTER_ORDERABLE)
def test_min_max_from_footer_match_duckdb(pq_path, duck, col):
    """min/max match DuckDB — and the footer shortcut fires wherever it is *sound*.

    `min` is answered straight from the EXACT footer bound for every orderable type, with no
    scan. `max` is too — **except on a float column**, where the bound cannot represent the
    answer: Parquet omits NaN from its statistics (as does our KLL sketch), but SQL's total
    order makes NaN the *greatest* value, so `max(f)` over a column containing NaN is NaN. The
    shortcut used to return the largest non-NaN value and silently disagree with executing the
    same query. It now declines, and the engine runs it. See `kyber/stats/columns.py`.

    Both answers must still equal DuckDB — the last two assertions — whichever path produced
    them. That is the property; the shortcut is only an optimization.
    """
    from batcher.api.terminal.metadata_answer import metadata_max, metadata_min

    _duck(duck)
    dmin, dmax = duck.execute(f"select min({col}), max({col}) from t").fetchone()
    ds = bt.read.parquet(pq_path)
    assert metadata_min(ds._plan, ds._sources, col) == dmin  # fired, no execution

    if col == "f":  # float: `max` cannot be answered from a NaN-blind bound
        assert metadata_max(ds._plan, ds._sources, col) is None
    else:
        assert metadata_max(ds._plan, ds._sources, col) == dmax  # fired, no execution

    # Whichever path answers, the answer is the same one DuckDB gives.
    assert ds.min(col) == dmin
    assert ds.max(col) == dmax


@pytest.mark.differential
@pytest.mark.parametrize("col", _ALL_COLS)
def test_n_null_and_has_nulls_match_duckdb(pq_path, duck, col):
    _duck(duck)
    n_null = duck.execute(f"select count(*) - count({col}) from t").fetchone()[0]
    for ds in _sources(pq_path).values():
        assert ds.n_null(col) == n_null
        assert ds.has_nulls(col) == (n_null > 0)


@pytest.mark.differential
@pytest.mark.parametrize("col", _ALL_COLS)
def test_all_null_matches_duckdb(pq_path, duck, col):
    _duck(duck)
    # all-null ⇔ non-empty and every value null ⇔ count(col) == 0 and count(*) > 0.
    total, nonnull = duck.execute(f"select count(*), count({col}) from t").fetchone()
    expected = total > 0 and nonnull == 0
    for ds in _sources(pq_path).values():
        assert ds.all_null(col) is expected


@pytest.mark.differential
@pytest.mark.parametrize("col", _ALL_COLS)
def test_n_unique_matches_duckdb(pq_path, duck, col):
    _duck(duck)
    expected = duck.execute(f"select count(distinct {col}) from t").fetchone()[0]
    for ds in _sources(pq_path).values():
        assert ds.n_unique(col) == expected


@pytest.mark.differential
def test_scalar_terminals_on_empty_match_duckdb(pq_path, duck):
    _duck(duck)
    # An empty relation: min/max NULL, counts 0, all_null False.
    empty = bt.read.parquet(pq_path).filter(bt.col("i") > 1000)
    assert empty.min("i") is None
    assert empty.max("i") is None
    assert empty.n_null("i") == 0
    assert empty.has_nulls("i") is False
    assert empty.all_null("i") is False
    assert empty.n_unique("i") == 0
    assert empty.has_rows is False


@pytest.mark.differential
def test_scalar_terminals_on_single_row(pq_path):
    one = bt.from_arrow(_TABLE).limit(1)  # {i: 3, ...}
    assert one.min("i") == 3
    assert one.max("i") == 3
    assert one.n_unique("i") == 1
    assert one.n_null("i") == 0
    assert one.has_nulls("i") is False
    assert one.has_rows is True


@pytest.mark.differential
def test_has_rows_matches_count(pq_path):
    for ds in _sources(pq_path).values():
        assert ds.has_rows is True
        assert ds.is_empty() is False


@pytest.mark.differential
def test_approx_terminals_answer(pq_path):
    # Explicitly approximate: assert they answer (a float / a positive count) and that
    # the streaming quantile (no learned grid) lands within the true value range. The
    # learned-grid path's exact interpolation is covered in the unit tests; here we
    # avoid asserting tight bounds on it (a process-wide learned grid keyed by bare
    # column name could otherwise flake under a full-suite run).
    for ds in _sources(pq_path).values():
        assert isinstance(ds.approx_median("f"), float)
        assert isinstance(ds.approx_percentile("f", 90), float)
        assert isinstance(ds.approx_quantile("f", 0.5), float)
        au = ds.approx_n_unique("i")
        assert au is not None and au >= 1
    # The streaming path (projected column through the TDigest) stays within [min, max].
    from batcher.api.orchestration import approx_quantile as _stream_q

    ds = bt.from_arrow(_TABLE)
    streamed = _stream_q(ds.select("f").iter_batches(), "f", 0.5)
    assert streamed is not None and 1.5 <= streamed <= 4.0


@pytest.mark.differential
def test_constant_column_n_unique_from_metadata(pq_path):
    # A literal projection carries an EXACT ndv=1 — the shortcut answers with no scan.
    from batcher.api.terminal.metadata_answer import metadata_n_unique
    from batcher.plan.expr_ir import lit

    ds = bt.read.parquet(pq_path).select(c=lit(9))
    assert metadata_n_unique(ds._plan, ds._sources, "c") == 1  # fired
    assert ds.n_unique("c") == 1


@pytest.mark.differential
def test_unknown_column_raises(pq_path):
    from batcher._internal.errors import PlanError

    ds = bt.read.parquet(pq_path)
    for op in (ds.min, ds.max, ds.n_unique, ds.n_null, ds.has_nulls, ds.all_null):
        with pytest.raises(PlanError):
            op("nope")
