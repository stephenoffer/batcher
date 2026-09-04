"""`WHERE a.f = b.f` over NaN on both sides: Batcher keeps the row, DuckDB does not.

A recorded divergence, not a defect in either engine, and worth pinning because the two are
self-inconsistent in *opposite* directions and nothing else in the suite says so:

* As a **scalar** comparison the two agree, and this file checks that rather than assuming
  it: `'nan'::DOUBLE = 'nan'::DOUBLE` is TRUE in DuckDB, and a `col(a) == col(b)` filter over
  two NaNs keeps the row in Batcher.
* Across **two relations** DuckDB decorrelates the predicate into a hash join, whose key
  comparison is IEEE, so NaN stops matching NaN and the row disappears. Batcher applies the
  same predicate as a filter and keeps its scalar answer.

So the row counts differ while neither engine changed what `=` means on its own. This is the
reason a rewrite that turns such a predicate into a join key is **not** semantics-preserving
for a floating-point column: the engine's key identity folds every NaN to one bit pattern
(`bc_arrow::canon_f64_bits`, deliberately, so `GROUP BY` and `=` agree on the two zeros), and
that is the opposite of what an IEEE key comparison does. Any future join-key derivation must
keep refusing floats for this reason.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import col

_NAN = float("nan")


def _tables():
    left = pa.table({"k": [1, 2, 3], "lf": [_NAN, 2.0, 3.0]})
    right = pa.table({"k": [1, 2, 3], "rf": [_NAN, 2.0, 3.0]})
    return left, right


def test_the_scalar_comparison_agrees(duck):
    """Both engines call `NaN = NaN` true as a plain predicate. The divergence is not here."""
    t = pa.table({"a": [_NAN, 1.0, 2.0], "b": [_NAN, 1.0, 3.0]})
    duck.register("t", t)
    got = bt.from_arrow(t).filter(col("a") == col("b")).collect()
    assert got.num_rows == len(duck.sql("SELECT a, b FROM t WHERE a = b").fetchall()) == 2


def test_across_two_relations_duckdb_drops_the_nan_pairing(duck):
    """Batcher keeps the NaN row; DuckDB's hash join drops exactly it."""
    left, right = _tables()
    duck.register("l", left)
    duck.register("r", right)
    got = (
        bt.from_arrow(left)
        .join(bt.from_arrow(right), on="k")
        .filter(col("lf") == col("rf"))
        .select("k", "lf", "rf")
        .collect()
    )
    duck_rows = duck.sql(
        "SELECT l.k, l.lf, r.rf FROM l JOIN r USING (k) WHERE l.lf = r.rf"
    ).fetchall()
    nan_rows = [row for row in got.to_pylist() if row["lf"] != row["lf"]]
    assert nan_rows, "Batcher should keep the NaN=NaN pairing, matching its scalar answer"
    assert len(duck_rows) == got.num_rows - len(nan_rows), (
        "expected DuckDB to drop exactly the NaN pairings its hash join cannot match"
    )
