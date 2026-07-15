"""Differential tests: float distinct-identity (-0.0/0.0, NaN) in mode/histogram.

`GROUP BY`, `count(distinct)`, and DuckDB all treat `-0.0 == 0.0` and every NaN bit
pattern as one value. `mode`/`histogram` encoded the raw child through Arrow's row
format, which is NOT canonical for floats, so they split `-0.0` from `0.0` and split
NaN payloads — disagreeing with DuckDB and with the engine's own COUNT(DISTINCT).
See docs/internals/bug_hunt_ledger.md.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from conftest import assert_same

pytestmark = pytest.mark.differential


def test_mode_folds_signed_zero(duck):
    """mode must fold -0.0/0.0 into one value before racing frequencies.

    With `-0.0`×2 and `0.0`×2 counted as one value (total 4) they beat `7.0`×3, so the
    mode is a zero. Before the fix the two zero signs were separate (2 and 2), `7.0`
    won, and the mode was 7.0 — disagreeing with DuckDB (which returns a zero).
    """
    t = pa.table(
        {
            "g": ["z"] * 7,
            "f": pa.array([-0.0, -0.0, 0.0, 0.0, 7.0, 7.0, 7.0], pa.float64()),
        }
    )
    out = bt.from_arrow(t).group_by("g").agg(m=bt.col("f").mode()).collect()
    duck.register("t", t)
    assert_same(out, duck.sql("SELECT g, mode(f) AS m FROM t GROUP BY g"))


def test_histogram_folds_signed_zero(duck):
    """histogram must fold -0.0/0.0 into ONE key with the summed count.

    Before the fix `[-0.0, 0.0, 5.0]` produced three keys of count 1; DuckDB (and the
    fixed engine) produce two: the folded zero key with count 2, and 5.0 with count 1.
    """
    t = pa.table(
        {
            "g": ["z"] * 3,
            "f": pa.array([-0.0, 0.0, 5.0], pa.float64()),
        }
    )
    out = bt.from_arrow(t).group_by("g").agg(h=bt.col("f").histogram()).collect()
    duck.register("t", t)
    assert_same(out, duck.sql("SELECT g, histogram(f) AS h FROM t GROUP BY g"))
