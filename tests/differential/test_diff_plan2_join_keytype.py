"""Join key-type validation: a build-time `PlanError`, not a runtime crash.

The engine's row encoder requires paired join keys to share an Arrow type — it
does not coerce ``Int64`` against ``Float64`` or ``Utf8``. Before this fix a
mismatch surfaced only at ``collect()`` as an opaque
``RowConverter column schema mismatch`` ``RuntimeError``; now it is rejected at
plan-build time with an actionable message. A same-type join still matches DuckDB.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from conftest import assert_same

_LEFT = pa.table(
    {
        "k_i": pa.array([1, 2, 3], pa.int64()),
        "k_f": pa.array([1.0, 2.0, 3.0], pa.float64()),
        "k_s": pa.array(["1", "2", "3"], pa.string()),
        "lv": pa.array([10, 20, 30], pa.int64()),
    }
)
_RIGHT = pa.table(
    {
        "k_i": pa.array([1, 2, 3], pa.int64()),
        "k_f": pa.array([1.0, 2.0, 3.0], pa.float64()),
        "k_s": pa.array(["1", "2", "3"], pa.string()),
        "rv": pa.array([100, 200, 300], pa.int64()),
    }
)


@pytest.mark.differential
@pytest.mark.parametrize(
    "lk,rk",
    [("k_i", "k_f"), ("k_f", "k_i"), ("k_i", "k_s"), ("k_s", "k_i"), ("k_f", "k_s")],
)
def test_join_incompatible_key_types_raise_planerror(lk: str, rk: str) -> None:
    left = bt.from_arrow(_LEFT)
    right = bt.from_arrow(_RIGHT)
    with pytest.raises(PlanError, match="join key type mismatch"):
        left.join(right, left_on=lk, right_on=rk)


@pytest.mark.differential
def test_join_matching_key_types_still_work(duck) -> None:
    duck.register("l", _LEFT)
    duck.register("r", _RIGHT)
    got = (
        bt.from_arrow(_LEFT)
        .join(bt.from_arrow(_RIGHT), on="k_i")
        .select("k_i", "lv", "rv")
        .collect()
    )
    want = duck.execute("SELECT l.k_i, l.lv, r.rv FROM l JOIN r USING (k_i)")
    assert_same(got, want)
