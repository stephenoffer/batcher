"""Float comparison identity vs DuckDB — the `-0.0` / NaN sweep for ledger B26.

Two questions decide every float-valued operation: *are these the same value?* and *which
is greater?* SQL fixes both in a way raw IEEE and raw bits each get wrong, and DuckDB is
the oracle for it:

* `-0.0` and `0.0` are **one value** — `WHERE f = 0.0` returns both;
* every NaN is **one value**, **greater than every number** — `'nan' = 'nan'` is true and
  `'nan' > 1` is true, regardless of NaN sign or payload.

The engine compares floats through Arrow's `cmp` kernels, which use `f64::total_cmp` plus
**raw-bit** equality. That is a different relation on exactly these values, and it made the
scalar comparison the one path disagreeing with the engine's own `GROUP BY` / `DISTINCT` /
join keys (which canonicalize via `bc_arrow::canon_f64`). Measured before the fix:
`WHERE f = 0.0` **dropped** the `-0.0` row, `WHERE f < 0` **returned** it, a *negative* NaN
ranked below `-inf` while a positive one ranked above `+inf`, and two NaNs of differing
payload compared unequal. Both tiers agreed with each other — the interpreter and the JIT
were aligned on the same wrong relation — so only this oracle catches it.

The fix canonicalizes both operands before the kernel, in the interpreter and in the JIT
(scalar + SIMD) together; `bc_arrow::float_ident` proves the two formulations are the same
relation, and `bc-codegen`'s parity tests hold the tiers to it.

NOTE ON THE ORACLE: these use `duck_materialize`, not `duck.register`. A registered Arrow
table is scanned with the filter pushed *into* the Arrow scan, where DuckDB evaluates it
with IEEE semantics — contradicting its own executor on NaN. See `conftest.duck_materialize`.
"""

from __future__ import annotations

import struct

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same, assert_same_ordered, duck_materialize

pytestmark = pytest.mark.differential


def _f64(bits: int) -> float:
    """The f64 with exactly these bits — the only way to name a NaN's sign and payload."""
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


#: NaNs that the raw-bit order treats as distinct values ranked in different places.
POS_NAN = _f64(0x7FF8000000000000)  # the canonical quiet NaN
NEG_NAN = _f64(0xFFF8000000000000)  # sign bit set — raw total order ranks it below -inf
POS_NAN_PAYLOAD = _f64(0x7FF8000000000001)  # a different payload — raw bit equality says !=
NEG_NAN_PAYLOAD = _f64(0xFFF8000000000007)

#: Every awkward float, plus a NULL and ordinary values to prove nothing else moved.
_VALUES = [
    -0.0,
    0.0,
    1.5,
    -1.5,
    float("inf"),
    float("-inf"),
    POS_NAN,
    NEG_NAN,
    POS_NAN_PAYLOAD,
    None,
]

_OPS = {
    "=": lambda c, v: c == v,
    "<>": lambda c, v: c != v,
    "<": lambda c, v: c < v,
    "<=": lambda c, v: c <= v,
    ">": lambda c, v: c > v,
    ">=": lambda c, v: c >= v,
}

#: Literals worth comparing against: the zeros, an ordinary value, and the infinities.
_LITERALS = {"0.0": 0.0, "-0.0": -0.0, "1.0": 1.0, "'inf'::DOUBLE": float("inf")}


@pytest.fixture
def floats(duck):
    """A float column spanning both zeros, three NaN bit-patterns, the infinities, a NULL."""
    table = pa.table({"r": list(range(len(_VALUES))), "f": pa.array(_VALUES, pa.float64())})
    duck_materialize(duck, "t", table)
    return table


@pytest.mark.parametrize("op", list(_OPS))
@pytest.mark.parametrize("lit_sql,lit_py", list(_LITERALS.items()))
def test_float_column_vs_literal_matches_duckdb(duck, floats, op, lit_sql, lit_py):
    """`f <op> <literal>` over every awkward float. 24 combinations, all vs the oracle."""
    out = bt.from_arrow(floats).filter(_OPS[op](bt.col("f"), lit_py)).select("r").collect()
    assert_same(out, duck.sql(f"SELECT r FROM t WHERE f {op} {lit_sql}"))


@pytest.mark.parametrize("op", list(_OPS))
def test_negative_zero_is_the_same_value_as_zero(duck, op):
    """The half of B26 both DuckDB paths agree on: `-0.0` and `0.0` are one value."""
    table = pa.table({"r": [0, 1], "f": pa.array([-0.0, 0.0], pa.float64())})
    duck_materialize(duck, "z", table)
    out = bt.from_arrow(table).filter(_OPS[op](bt.col("f"), 0.0)).select("r").collect()
    assert_same(out, duck.sql(f"SELECT r FROM z WHERE f {op} 0.0"))


@pytest.mark.parametrize("op", list(_OPS))
def test_column_vs_column_matches_duckdb(duck, op):
    """`a <op> b` — the two-column path, which broadcasts no literal and so takes the
    array kernel rather than the scalar fast path."""
    a = [-0.0, 0.0, POS_NAN, POS_NAN, NEG_NAN, 1.5, POS_NAN, None]
    b = [0.0, -0.0, POS_NAN, NEG_NAN, POS_NAN_PAYLOAD, POS_NAN, 1.5, 0.0]
    table = pa.table(
        {"r": list(range(len(a))), "a": pa.array(a, pa.float64()), "b": pa.array(b, pa.float64())}
    )
    duck_materialize(duck, "p", table)
    out = bt.from_arrow(table).filter(_OPS[op](bt.col("a"), bt.col("b"))).select("r").collect()
    assert_same(out, duck.sql(f"SELECT r FROM p WHERE a {op} b"))


def test_every_nan_is_one_value(duck):
    """All NaN sign/payload combinations compare equal — raw-bit equality says otherwise."""
    nans = [POS_NAN, NEG_NAN, POS_NAN_PAYLOAD, NEG_NAN_PAYLOAD]
    table = pa.table({"r": list(range(len(nans))), "f": pa.array(nans, pa.float64())})
    duck_materialize(duck, "n", table)
    out = bt.from_arrow(table).filter(bt.col("f") == POS_NAN).select("r").collect()
    assert_same(out, duck.sql("SELECT r FROM n WHERE f = 'nan'::DOUBLE"))
    assert out.num_rows == len(nans), "every NaN must equal every other NaN"


def test_nan_is_greater_than_every_number(duck):
    """Including a *negative* NaN, which the raw total order ranks below `-inf`."""
    table = pa.table(
        {
            "r": [0, 1, 2, 3],
            "f": pa.array([NEG_NAN, POS_NAN, float("inf"), 1.5], pa.float64()),
        }
    )
    duck_materialize(duck, "g", table)
    out = bt.from_arrow(table).filter(bt.col("f") > float("inf")).select("r").collect()
    assert_same(out, duck.sql("SELECT r FROM g WHERE f > 'inf'::DOUBLE"))
    assert sorted(out.to_pydict()["r"]) == [0, 1], "both NaNs outrank +inf"


def test_comparison_agrees_with_group_by_and_join_on_float_identity(duck):
    """The invariant the fix restores: `=` means what `GROUP BY`/`DISTINCT`/join mean.

    A scalar `=` that split the zeros while `GROUP BY` folded them made one column mean two
    things depending on which operator read it. Kept alongside the DuckDB check because it
    is the *internal* consistency statement — it fails even if DuckDB changed its mind.
    """
    zeros = pa.table({"k": pa.array([0.0, -0.0, POS_NAN, NEG_NAN], pa.float64())})
    ds = bt.from_arrow(zeros)
    # `=` folds both pairs...
    eq = ds.filter(bt.col("k") == 0.0).collect().num_rows
    eq_nan = ds.filter(bt.col("k") == POS_NAN).collect().num_rows
    assert (eq, eq_nan) == (2, 2)
    # ...and so do GROUP BY and DISTINCT: two groups, {zero, NaN}.
    assert ds.group_by("k").agg(c=bt.col("k").count()).collect().num_rows == 2
    assert ds.select(bt.col("k")).distinct().collect().num_rows == 2


def test_arithmetic_on_floats_stays_ieee(duck):
    """Canonicalization is for *comparison* only — it must not touch arithmetic.

    `-0.0 + 0.0` is `0.0` and `1/-0.0` is `-inf` under IEEE; folding `-0.0` into `0.0`
    before an arithmetic kernel would silently change results.
    """
    table = pa.table({"f": pa.array([-0.0], pa.float64())})
    got = bt.from_arrow(table).select(neg=bt.col("f") * 1.0, div=1.0 / bt.col("f")).collect()
    # `-0.0 * 1.0` keeps its sign, so `1/x` is still `-inf` — not `+inf`.
    assert struct.pack("<d", got.column("neg").to_pylist()[0]) == struct.pack("<d", -0.0)
    assert got.column("div").to_pylist()[0] == float("-inf")


def test_sort_order_matches_comparison(duck, floats):
    """`ORDER BY` must rank floats the way `=`/`<` compare them — NaN last, zeros together."""
    out = bt.from_arrow(floats).filter(bt.col("f").is_not_null()).sort("f").collect()
    assert_same_ordered(
        out.select(["f"]),
        duck.sql("SELECT f FROM t WHERE f IS NOT NULL ORDER BY f"),
    )
