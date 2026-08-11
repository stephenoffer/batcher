"""Join key types: promotable pairs are widened, incompatible pairs raise at plan time.

The engine's row encoder requires paired join keys to share an Arrow type — it compares
them byte-for-byte and does not coerce. Two things follow, and they used to be conflated
into one rule that rejected everything:

- A pair the promotion lattice **can** reconcile (``Int64`` against ``Float64``,
  ``decimal(10,2)`` against ``decimal(12,4)``, an all-null column against a typed one) is
  widened on both sides before the encoder runs. Widening cannot change a key's value, so
  no match is gained or lost, and the result matches DuckDB — which joins these pairs
  without comment. This file used to assert that ``k_i`` against ``k_f`` *raised*, which
  pinned Batcher's limitation rather than DuckDB's semantics; the oracle was never
  consulted for that case.
- A pair with **no** lossless common type (``Int64`` against ``Utf8``) is a data-contract
  error and still raises at plan-build time with an actionable message, rather than
  surfacing at ``collect()`` as an opaque ``RowConverter column schema mismatch``.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher._internal.errors import PlanError

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
        "k_f": pa.array([1.0, 2.0, 3.5], pa.float64()),
        "k_s": pa.array(["1", "2", "3"], pa.string()),
        "rv": pa.array([100, 200, 300], pa.int64()),
    }
)


@pytest.mark.differential
@pytest.mark.parametrize("lk,rk", [("k_i", "k_s"), ("k_s", "k_i"), ("k_f", "k_s")])
def test_join_keys_with_no_common_type_raise_planerror(lk: str, rk: str) -> None:
    """A number against a string has no lossless common type — that is a bug, not a cast."""
    left = bt.from_arrow(_LEFT)
    right = bt.from_arrow(_RIGHT)
    with pytest.raises(PlanError, match="join key type mismatch"):
        left.join(right, left_on=lk, right_on=rk)


@pytest.mark.differential
@pytest.mark.parametrize("lk,rk", [("k_i", "k_f"), ("k_f", "k_i")])
def test_join_keys_that_promote_are_widened_and_match_duckdb(duck, lk: str, rk: str) -> None:
    """An int key against a float key joins, on the rows where the values are equal.

    `k_f` on the right ends in ``3.5``, which equals no integer, so this also proves the
    widening is a *widening*: the third row must not match, where narrowing the float side
    to `int64` would have rounded it into a match that does not exist.
    """
    duck.register("l", _LEFT)
    duck.register("r", _RIGHT)
    got = (
        bt.from_arrow(_LEFT)
        .join(bt.from_arrow(_RIGHT), left_on=lk, right_on=rk)
        .select("lv", "rv")
        .collect()
    )
    want = duck.execute(f"SELECT l.lv, r.rv FROM l JOIN r ON l.{lk} = r.{rk}")
    assert_same(got, want)


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
