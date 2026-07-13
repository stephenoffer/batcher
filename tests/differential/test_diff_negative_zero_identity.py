"""`x + 0` is not an identity on a float — it erases the sign of negative zero.

IEEE-754 defines `-0.0 + 0.0` as `+0.0`, so simplifying `x + 0` away preserves `-0.0`
where the engine would have produced `+0.0`. The simplification rule guarded the *literal*
(insisting the `0` be an integer) rather than the *column*, so it fired on float columns
and Batcher returned `-0.0` where DuckDB returns `+0.0`.

No existing differential test could catch this: `assert_same` canonicalizes `±0.0` to one
sentinel precisely so NaN/zero-sign noise does not make results incomparable. So these
compare the **sign bit** directly, via `copysign`.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.differential


def _signs(values) -> list[float]:
    """The sign bit of each value — the thing `assert_same` deliberately cannot see."""
    return [math.copysign(1.0, v) for v in values]


@pytest.fixture
def t(duck):
    table = pa.table(
        {
            "f": pa.array([-0.0, 0.0, 1.5, -2.5], type=pa.float64()),
            "i": pa.array([-1, 0, 3, 7], type=pa.int64()),
        }
    )
    duck.register("t", table)
    return bt.from_arrow(table)


def test_float_plus_zero_keeps_duckdbs_sign(t, duck):
    got = t.select(x=col("f") + 0).collect().to_pydict()["x"]
    want = duck.sql("SELECT f + 0 AS x FROM t").to_arrow_table().to_pydict()["x"]
    assert _signs(got) == _signs(want)


def test_zero_plus_float_keeps_duckdbs_sign(t, duck):
    """The mirrored operand order goes through the other arm of the rule."""
    got = t.select(x=0 + col("f")).collect().to_pydict()["x"]
    want = duck.sql("SELECT 0 + f AS x FROM t").to_arrow_table().to_pydict()["x"]
    assert _signs(got) == _signs(want)


def test_float_minus_zero_is_still_an_identity(t, duck):
    """`-0.0 - 0.0` is `-0.0`, so this one needs no guard and must keep simplifying."""
    got = t.select(x=col("f") - 0).collect().to_pydict()["x"]
    want = duck.sql("SELECT f - 0 AS x FROM t").to_arrow_table().to_pydict()["x"]
    assert _signs(got) == _signs(want)


def test_float_times_one_is_still_an_identity(t, duck):
    got = t.select(x=col("f") * 1).collect().to_pydict()["x"]
    want = duck.sql("SELECT f * 1 AS x FROM t").to_arrow_table().to_pydict()["x"]
    assert _signs(got) == _signs(want)


def test_integer_plus_zero_is_still_simplified_away(t):
    """The identity is sound for integers and must not have been lost to the guard."""
    from batcher.kyber.optimizer import Optimizer

    ds = t.select(y=col("i") + 0)
    ir = str(Optimizer(sources=ds._sources).optimize_full(ds._plan)[1].to_ir())
    assert "add" not in ir, "the integer identity should still be simplified away"
