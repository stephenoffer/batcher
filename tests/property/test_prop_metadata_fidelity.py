"""Property: a metadata-answered terminal equals the executed answer and DuckDB.

Batcher answers ``count`` / ``is_empty`` / ``min`` / ``max`` / ``n_unique`` / ``n_null``
from metadata (footer bounds, exact row counts, per-column null counts) *without
scanning* when the answer is provably exact — the ~100 metadata shortcuts. Two things
must hold, and Hypothesis checks both over random typed data (ints, floats, strings,
dates, bools; with nulls, all-null, empty, and single-row draws all reachable):

1. **Fidelity** — when the shortcut fires (returns non-``None``), its answer equals both
   the engine-executed answer and the DuckDB oracle. A wrong footer-derived answer is a
   silent correctness bug that never scans a row to get caught.
2. **The EXACT firewall** — after a ``filter`` the source's bounds/counts are no longer
   exact for the *result*, so the shortcut MUST decline (return ``None``) and fall back
   to execution. The test asserts the shortcut goes to ``None`` *and* that the executed
   fallback still matches DuckDB on the filtered relation. A shortcut that keeps
   answering past a filter would return the pre-filter bound — a correctness bug.

Data is written to Parquet (whose footer carries exact per-column min/max/null-count/row
count) and read back, so the metadata paths that need exact source statistics actually
fire — an in-memory source exposes only the exact row count.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import batcher as bt
from batcher import col, lit

pytest.importorskip("batcher._native", reason="native engine not built")
duckdb = pytest.importorskip("duckdb")

from _optmeta_common import coerce  # noqa: E402  (test-dir helper)

from batcher.api.terminal.metadata_answer import (  # noqa: E402
    metadata_count,
    metadata_is_empty,
    metadata_max,
    metadata_min,
    metadata_n_unique,
    metadata_null_count,
)

pytestmark = [pytest.mark.property, pytest.mark.integration]


@pytest.fixture(autouse=True)
def _stable_morsel_tuning():
    """Pin adaptive morsel-size tuning off (a result-invariant Carbonite concern)."""
    from batcher.config import active_config, set_config

    prev = active_config()
    set_config(
        prev.replace(execution=dataclasses.replace(prev.execution, adaptive_morsel_sizing=False))
    )
    yield
    set_config(prev)


# `k` is a non-null int key (the firewall filter ranges over it and keeps every row when
# the bound is wide). The rest are nullable across every tested type.
_SCHEMA = pa.schema(
    [
        ("k", pa.int64()),
        ("v", pa.int64()),
        ("f", pa.float64()),
        ("s", pa.string()),
        ("d", pa.date32()),
        ("bo", pa.bool_()),
    ]
)
_DATES = [dt.date(2019, 1, 1), dt.date(2020, 6, 15), dt.date(2021, 12, 31), dt.date(2022, 3, 3)]


@st.composite
def _table(draw: st.DrawFn) -> pa.Table:
    """A random table on `_SCHEMA` — 0..25 rows, every column (but `k`) nullable."""
    n = draw(st.integers(min_value=0, max_value=25))
    return pa.table(
        {
            "k": draw(st.lists(st.integers(-20, 20), min_size=n, max_size=n)),
            "v": draw(st.lists(st.one_of(st.none(), st.integers(-20, 20)), min_size=n, max_size=n)),
            "f": draw(
                st.lists(
                    st.one_of(
                        st.none(),
                        st.floats(
                            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
                        ),
                    ),
                    min_size=n,
                    max_size=n,
                )
            ),
            "s": draw(
                st.lists(
                    st.one_of(st.none(), st.sampled_from(["a", "bb", "ccc"])),
                    min_size=n,
                    max_size=n,
                )
            ),
            "d": draw(
                st.lists(st.one_of(st.none(), st.sampled_from(_DATES)), min_size=n, max_size=n)
            ),
            "bo": draw(st.lists(st.one_of(st.none(), st.booleans()), min_size=n, max_size=n)),
        },
        schema=_SCHEMA,
    )


def _duck_scalar(con, sql: str):
    """Fetch a single scalar from DuckDB (``None`` for an empty/all-null aggregate)."""
    return con.execute(sql).fetchone()[0]


_PROP = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.function_scoped_fixture,
    ],
)


@_PROP
# `orderable` is restricted to the types the engine's min/max *execution* supports
# (int/float/string); Date32/Boolean min/max are execution-unsupported (a declared engine
# limitation) and answered only from metadata — the date footer bound is checked
# separately, metadata-only, inside the test.
@given(_table(), st.sampled_from(["k", "v", "f", "s"]), st.integers(-20, 20))
def test_metadata_terminal_equals_execution_and_duckdb(
    table: pa.Table, orderable: str, thr: int
) -> None:
    """count/is_empty/min/max/n_unique/n_null: metadata == executed == DuckDB, + firewall."""
    con = duckdb.connect()
    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/t.parquet"
        pq.write_table(table, path)
        ds = bt.read(path)
        plan, sources = ds._plan, ds._sources
        con.register("t", table)
        _run_checks(con, ds, plan, sources, table, orderable, thr)
    con.close()


def _run_checks(con, ds, plan, sources, table, orderable, thr) -> None:
    # --- count / is_empty (answered from the exact row count) ---------------
    oracle_count = int(_duck_scalar(con, "SELECT count(*) FROM t"))
    m_count = metadata_count(plan, sources, None)
    assert m_count is not None, "row-count shortcut should fire on an exact source"
    assert m_count == oracle_count == ds.count() == table.num_rows
    m_empty = metadata_is_empty(plan, sources, None)
    assert m_empty is not None and m_empty == (oracle_count == 0) == ds.is_empty()

    # Scalar aggregates over a *genuinely empty* relation raise a declared
    # "aggregation over empty input is not yet supported" engine limitation (an explicit
    # error, not a wrong answer), so the min/max/n_unique/n_null fidelity checks — which
    # must execute — are exercised on non-empty draws (empty is still covered by
    # count/is_empty above and the firewall below). Single-row and all-null non-empty
    # draws (where min/max legitimately return NULL) are included.
    if table.num_rows > 0:
        # --- min / max (footer bounds) --------------------------------------
        _check_scalar(
            con,
            ds,
            plan,
            sources,
            orderable,
            meta=metadata_min(plan, sources, orderable),
            terminal=ds.min(orderable),
            executed=_executed_agg(ds, col(orderable).min()),
            oracle=_duck_scalar(con, f"SELECT min({orderable}) FROM t"),
            label="min",
        )
        _check_scalar(
            con,
            ds,
            plan,
            sources,
            orderable,
            meta=metadata_max(plan, sources, orderable),
            terminal=ds.max(orderable),
            executed=_executed_agg(ds, col(orderable).max()),
            oracle=_duck_scalar(con, f"SELECT max({orderable}) FROM t"),
            label="max",
        )

        # --- n_unique / n_null (over every tested column, incl. str/bool) ---
        for c in ("k", "v", "f", "s", "d", "bo"):
            nu_oracle = int(_duck_scalar(con, f"SELECT count(DISTINCT {c}) FROM t"))
            nu_meta = metadata_n_unique(plan, sources, c)
            assert ds.n_unique(c) == nu_oracle, f"n_unique({c}) terminal != DuckDB"
            assert int(_executed_agg(ds, col(c).n_unique()) or 0) == nu_oracle
            if nu_meta is not None:
                assert nu_meta == nu_oracle, f"n_unique({c}) metadata != DuckDB"

            nn_oracle = int(_duck_scalar(con, f"SELECT count(*) - count({c}) FROM t"))
            nn_meta = metadata_null_count(plan, sources, c)
            assert ds.n_null(c) == nn_oracle, f"n_null({c}) terminal != DuckDB"
            if nn_meta is not None:
                assert nn_meta == nn_oracle, f"n_null({c}) metadata (footer) != DuckDB"

        # --- date min/max: metadata-only (engine execution is unsupported for Date32) ---
        # The terminal answers these from the footer with no scan; when it fires the bound
        # must be exact. (A wrong footer date bound is exactly the silent bug to catch.)
        d_min_meta = metadata_min(plan, sources, "d")
        d_max_meta = metadata_max(plan, sources, "d")
        if d_min_meta is not None:
            assert coerce(d_min_meta) == coerce(_duck_scalar(con, "SELECT min(d) FROM t"))
        if d_max_meta is not None:
            assert coerce(d_max_meta) == coerce(_duck_scalar(con, "SELECT max(d) FROM t"))

    # --- THE EXACT FIREWALL ------------------------------------------------
    # The load-bearing invariant is: past a filter a shortcut must never return a *stale*
    # (pre-filter) answer. It may still answer when the answer is provably exact — the
    # min/max path is a sound zone-map rule (``min(filter(k > c))`` == footer_min iff
    # footer_min > c, i.e. the extreme row provably survives; else it declines), and the
    # count path answers a provably-total/empty filter — but on a genuinely partial filter
    # the count is not provable and MUST decline. In every case: if it answers at all, that
    # answer equals the executed answer and DuckDB; and the executed fallback is correct.
    fds = ds.filter(col("k") > thr)
    fplan, fsrc = fds._plan, fds._sources
    f_count = int(_duck_scalar(con, f"SELECT count(*) FROM t WHERE k > {thr}"))
    assert fds.count() == f_count, "executed count past a filter != DuckDB"

    m_fcount = metadata_count(fplan, fsrc, None)
    if 0 < f_count < table.num_rows:  # genuinely partial → not provable → must decline
        assert m_fcount is None, "count shortcut returned a partial-filter count it can't prove"
    if m_fcount is not None:  # if it answers, it must be the true post-filter count
        assert m_fcount == f_count, "count shortcut returned a STALE (pre-filter) count"

    if f_count > 0:  # a filtered-to-empty relation hits the empty-agg limitation
        f_min = _duck_scalar(con, f"SELECT min({orderable}) FROM t WHERE k > {thr}")
        f_max = _duck_scalar(con, f"SELECT max({orderable}) FROM t WHERE k > {thr}")
        assert coerce(fds.min(orderable)) == coerce(f_min), "executed min past a filter != DuckDB"
        assert coerce(fds.max(orderable)) == coerce(f_max), "executed max past a filter != DuckDB"
        # If the sound zone-map rule chooses to answer past the filter, it must be exact,
        # never the stale pre-filter bound.
        mn = metadata_min(fplan, fsrc, orderable)
        mx = metadata_max(fplan, fsrc, orderable)
        if mn is not None:
            assert coerce(mn) == coerce(f_min), "min shortcut returned a STALE bound past a filter"
        if mx is not None:
            assert coerce(mx) == coerce(f_max), "max shortcut returned a STALE bound past a filter"


def _executed_agg(ds: bt.Dataset, agg_expr):
    """Force an engine-executed global aggregate (bypassing the metadata shortcut).

    A wide-open ``filter`` over the non-null key keeps every row but makes the plan a
    ``Filter``-over-``Aggregate`` rather than a bare global aggregate over a scan, so the
    metadata-aggregate fast path declines and the value is genuinely computed by the engine.
    """
    res = ds.filter(col("k") >= lit(-1_000_000)).agg(__m__=agg_expr).to_pydict()["__m__"]
    return res[0] if res else None


def _check_scalar(con, ds, plan, sources, column, *, meta, terminal, executed, oracle, label):
    """Assert ``terminal == executed == oracle`` and, when it fired, ``meta == oracle``."""
    assert coerce(terminal) == coerce(oracle), (
        f"{label}({column}) terminal != DuckDB: {terminal!r} vs {oracle!r}"
    )
    assert coerce(executed) == coerce(oracle), (
        f"{label}({column}) engine-executed != DuckDB: {executed!r} vs {oracle!r}"
    )
    if meta is not None:
        assert coerce(meta) == coerce(oracle), (
            f"{label}({column}) METADATA shortcut != DuckDB (footer-derived answer is "
            f"wrong): {meta!r} vs {oracle!r}"
        )
