"""approx_count_distinct (HLL) — within error of the exact distinct count.

Approximate, so it is checked against the *exact* distinct count within a relative
tolerance (HLL's standard error), not by exact-match against DuckDB. The point is
bounded memory under skew, with a small, bounded error.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.differential

_TOL = 0.05  # HLL relative error budget at default precision


def test_approx_n_unique_grouped_within_tolerance(duck):
    rng = np.random.default_rng(7)
    n = 60_000
    t = pa.table({"g": (np.arange(n) % 4), "v": rng.integers(0, 3000, n)})
    duck.register("t", t)

    approx = bt.from_arrow(t).group_by("g").agg(a=col("v").approx_n_unique()).collect()
    exact = duck.sql("SELECT g, COUNT(DISTINCT v) e FROM t GROUP BY g").fetchall()

    approx_by_g = dict(
        zip(approx.column("g").to_pylist(), approx.column("a").to_pylist(), strict=True)
    )
    for g, e in exact:
        a = approx_by_g[g]
        assert abs(a - e) / e < _TOL, f"group {g}: approx {a} vs exact {e}"


def test_approx_n_unique_global_within_tolerance(duck):
    rng = np.random.default_rng(11)
    n = 100_000
    t = pa.table({"v": rng.integers(0, 5000, n)})
    duck.register("t", t)

    approx = bt.from_arrow(t).agg(a=col("v").approx_n_unique()).collect().column("a")[0].as_py()
    (exact,) = duck.sql("SELECT COUNT(DISTINCT v) FROM t").fetchone()
    assert abs(approx - exact) / exact < _TOL, f"approx {approx} vs exact {exact}"


def test_approx_quantile_within_tolerance(duck):
    rng = np.random.default_rng(5)
    n = 100_000
    t = pa.table({"g": (np.arange(n) % 3), "v": rng.normal(100.0, 20.0, n)})
    duck.register("t", t)

    approx = (
        bt.from_arrow(t)
        .group_by("g")
        .agg(m=col("v").approx_median(), q9=col("v").approx_quantile(0.9))
        .collect()
    )
    exact = duck.sql("SELECT g, median(v) m, quantile_cont(v, 0.9) q9 FROM t GROUP BY g").fetchall()

    am = dict(zip(approx.column("g").to_pylist(), approx.column("m").to_pylist(), strict=True))
    aq = dict(zip(approx.column("g").to_pylist(), approx.column("q9").to_pylist(), strict=True))
    # KLL rank error translates to a small *relative* value error (larger in the
    # low-density q90 tail than at the median, but still a few percent).
    for g, m, q9 in exact:
        assert abs(am[g] - m) / abs(m) < 0.03, f"g{g} median {am[g]} vs {m}"
        assert abs(aq[g] - q9) / abs(q9) < 0.03, f"g{g} q90 {aq[g]} vs {q9}"
