"""`list.entropy` and `list.log_softmax` against DuckDB.

Both have a closed form DuckDB can express over an unnested list, so the engine's kernels are
checked against an independent implementation rather than against remembered numbers. That
matters most for `log_softmax`, whose whole reason to exist is that it is *not*
`ln(softmax(x))` — an implementation that quietly took the easy route would still pass a test
comparing the two on well-conditioned input, and would differ from DuckDB nowhere except the
underflow case this pins separately.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential

# Rows chosen for the edges: a certain distribution, a uniform one, unnormalized counts, an
# asymmetric spread, and a single-element row.
_ROWS = pa.table(
    {
        "id": [1, 2, 3, 4, 5],
        "p": [
            [1.0, 0.0, 0.0],
            [0.25, 0.25, 0.25, 0.25],
            [2.0, 2.0],
            [0.7, 0.2, 0.1],
            [5.0],
        ],
    }
)


def test_entropy_matches_a_duckdb_reduction(duck):
    duck.register("rows", _ROWS)
    # Normalize by the row's own positive total, then sum -p*ln(p) over the positive elements,
    # which is the definition the kernel implements.
    expected = duck.sql(
        """
        WITH flat AS (
          SELECT id, unnest(p) AS v FROM rows
        ),
        totals AS (
          SELECT id, sum(v) FILTER (WHERE v > 0) AS total FROM flat GROUP BY id
        )
        SELECT f.id,
               sum(CASE WHEN f.v > 0
                        THEN -(f.v / t.total) * ln(f.v / t.total)
                        ELSE 0 END)::DOUBLE AS h
        FROM flat f JOIN totals t USING (id)
        GROUP BY f.id
        """
    )
    got = bt.from_arrow(_ROWS).select("id", h=bt.col("p").list.entropy()).collect()
    assert_same(got, expected)


def test_log_softmax_matches_a_duckdb_reduction(duck):
    duck.register("rows", _ROWS)
    # `x - max - ln(sum(exp(x - max)))`, elementwise, gathered back into a list in the original
    # order. `generate_subscripts` keeps the position so the ordering is pinned, not incidental.
    expected = duck.sql(
        """
        WITH flat AS (
          SELECT id, v, i FROM rows, unnest(p) WITH ORDINALITY AS u(v, i)
        ),
        stats AS (
          SELECT id, max(v) AS mx FROM flat GROUP BY id
        ),
        denom AS (
          SELECT f.id, ln(sum(exp(f.v - s.mx))) AS lse
          FROM flat f JOIN stats s USING (id) GROUP BY f.id
        )
        SELECT f.id, list(round(f.v - s.mx - d.lse, 9) ORDER BY f.i) AS lg
        FROM flat f JOIN stats s USING (id) JOIN denom d USING (id)
        GROUP BY f.id
        """
    )
    got = (
        bt.from_arrow(_ROWS)
        .select(
            "id",
            lg=bt.col("p").list.log_softmax().list.transform(bt.element().round(9)),
        )
        .collect()
    )
    assert_same(got, expected)


def test_log_softmax_exponentiates_back_to_the_engines_own_softmax():
    """The defining relationship, checked inside the engine where DuckDB has no softmax."""
    out = (
        bt.from_arrow(_ROWS)
        .select(lg=bt.col("p").list.log_softmax(), sm=bt.col("p").list.softmax())
        .to_pydict()
    )
    for logs, probs in zip(out["lg"], out["sm"], strict=True):
        assert [math.exp(v) for v in logs] == pytest.approx(probs)


def test_log_softmax_stays_finite_where_the_linear_form_underflows():
    """The reason it is a kernel rather than `softmax().ln()`."""
    extreme = pa.table({"p": [[0.0, -900.0]]})
    out = (
        bt.from_arrow(extreme)
        .select(lg=bt.col("p").list.log_softmax(), sm=bt.col("p").list.softmax())
        .to_pydict()
    )
    assert all(math.isfinite(v) for v in out["lg"][0])
    assert out["sm"][0][1] == 0.0  # the linear form has already lost it
    assert out["lg"][0][1] < -800


def test_entropy_is_bounded_by_the_log_of_the_row_length():
    """The maximum a distribution over n outcomes can reach, checked per row."""
    out = (
        bt.from_arrow(_ROWS)
        .select(h=bt.col("p").list.entropy(), n=bt.col("p").list.len())
        .to_pydict()
    )
    for h, n in zip(out["h"], out["n"], strict=True):
        assert h <= math.log(n) + 1e-12
