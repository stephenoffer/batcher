"""Numeric-stability differential coverage for the moment aggregates.

`var`/`stddev` moved to Welford (B9) to stop the ``Sum(x^2) - Sum(x)^2/n`` cancellation, but
`covar_pop`/`covar_samp`/`corr`/`skewness`/`kurtosis` kept a sum-of-powers state and
the same cancelling finalize. On a column with a large offset (timestamps, ids), that
subtraction of two nearly-equal large numbers catastrophically cancels:
`covar_pop([1e9+1, 1e9+2, …])` came back as exactly `0` (true `2`), and `corr` as
`NULL`. These pin the central-moment (co-moment) rewrite against DuckDB — the values
DuckDB itself returns are the oracle.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, corr, covar_pop, covar_samp
from conftest import assert_same

pytestmark = pytest.mark.differential

# Values at a 1e9 offset: exactly representable in f64, so the true covariance/correlation
# is exact, but the sum-of-powers formula cancelled it to 0 / NULL.
_OFF = 1e9
_X = [_OFF + 1, _OFF + 2, _OFF + 3, _OFF + 4, _OFF + 5]
_Y = [_OFF + 1, _OFF + 3, _OFF + 2, _OFF + 7, _OFF + 4]


def _tbl():
    return pa.table({"x": _X, "y": _Y})


def _approx_matches(out, duck, cols):
    # Two-pass central moments and DuckDB's algorithm agree to f64 precision but not
    # bit-for-bit, so compare with a relative tolerance. This still catches the old
    # cancelling formula, which returned 0.0 or NULL — off by ~100%, not ~1e-9.
    got = out.to_pydict()
    want = {k: v for k, v in zip(cols, duck.fetchone(), strict=True)}
    for k in cols:
        assert got[k][0] == pytest.approx(want[k], rel=1e-7), f"{k}: {got[k][0]} vs {want[k]}"


def test_covar_corr_stable_at_large_offset(duck):
    duck.register("t", _tbl())
    out = (
        bt.from_arrow(_tbl())
        .agg(
            cp=covar_pop(col("x"), col("y")),
            cs=covar_samp(col("x"), col("y")),
            c=corr(col("x"), col("y")),
        )
        .collect()
    )
    rel = duck.sql("SELECT covar_pop(x,y) AS cp, covar_samp(x,y) AS cs, corr(x,y) AS c FROM t")
    _approx_matches(out, rel, ["cp", "cs", "c"])


def test_skewness_kurtosis_stable_at_large_offset(duck):
    # Enough points (asymmetric) that kurtosis is defined (n >= 4) and non-trivial.
    base = (1.0, 2.0, 3.0, 4.0, 10.0, 1.0, 2.0)
    xs = [_OFF + v for v in base]
    out = bt.from_arrow(pa.table({"x": xs})).agg(s=col("x").skewness(), k=col("x").kurtosis()).collect()
    # Skewness/kurtosis are translation-invariant, so the oracle is DuckDB on the
    # *un-offset* data — where DuckDB is stable. (At `_OFF` DuckDB's own sum-of-powers
    # formula catastrophically cancels and returns NaN, so it cannot be the oracle there;
    # the point of the fix is that Batcher's two-pass state stays exact at the offset.)
    duck.register("b", pa.table({"x": list(base)}))
    rel = duck.sql("SELECT skewness(x) AS s, kurtosis(x) AS k FROM b")
    _approx_matches(out, rel, ["s", "k"])


def test_covar_corr_grouped_single_node_equals_distributed():
    # Same offset, grouped — the mergeable central-moment state must give the identical
    # result single-node and across workers (the single-node == distributed invariant).
    g = {
        "g": ["a", "a", "a", "a", "b", "b", "b", "b"],
        "x": [_OFF + v for v in (1, 2, 3, 4, 1, 5, 2, 8)],
        "y": [_OFF + v for v in (1, 3, 2, 7, 1, 6, 3, 9)],
    }
    ds = (
        bt.from_pydict(g)
        .group_by("g")
        .agg(
            c=corr(col("x"), col("y")),
            s=col("x").skewness(),
            cp=covar_pop(col("x"), col("y")),
        )
    )
    single = {
        k: (round(c, 6), round(s, 6), round(cp, 6))
        for k, c, s, cp in zip(
            *[ds.collect().to_pydict()[col_] for col_ in ("g", "c", "s", "cp")], strict=True
        )
    }
    dd = ds.collect(distributed=True, num_workers=3).to_pydict()
    multi = {
        k: (round(c, 6), round(s, 6), round(cp, 6))
        for k, c, s, cp in zip(dd["g"], dd["c"], dd["s"], dd["cp"], strict=True)
    }
    assert single == multi
