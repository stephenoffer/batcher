"""`plan.types.promote` — the control plane's mirror of `bc_expr::common_supertype`.

`promote` is what `Dataset.schema` predicts for a union, a `coalesce`, and a `case`
expression; `bc_expr::common_supertype` is what the engine actually produces. The two are
written to agree, and when they did not the control plane advertised `int64` for a union
the engine then refused to run. These cases pin the control-plane half — the engine half
lives in `crates/bc-expr/src/supertype.rs`, and `tests/differential/`
`test_diff_type_promotion_lattice.py` pins that they agree end to end against DuckDB.

Algebraic properties get their own cases: the lattice must be commutative (two callers
that happen to pass the pair the other way round cannot get different schemas) and
idempotent (a no-op union must not rewrite the schema).
"""

from __future__ import annotations

from decimal import Decimal
from itertools import product

import pyarrow as pa
import pytest

from batcher.plan.types import promote

# Every type in this list is one the lattice has a rule about, so the algebraic
# properties below exercise the real arms rather than the identity fast path.
LATTICE_TYPES = [
    pa.null(),
    pa.bool_(),
    pa.int32(),
    pa.int64(),
    pa.uint8(),
    pa.float32(),
    pa.float64(),
    pa.decimal128(10, 2),
    pa.decimal128(12, 4),
    pa.string(),
    pa.large_string(),
    pa.binary(),
    pa.large_binary(),
    pa.date32(),
    pa.date64(),
    pa.timestamp("ms"),
    pa.timestamp("us"),
    pa.timestamp("us", "UTC"),
    pa.time32("s"),
    pa.time64("ns"),
    pa.duration("s"),
    pa.duration("ns"),
]


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        # `null` carries no values to lose, so it adopts whatever it meets. Without this
        # an all-null column could not be unioned, coalesced, or compared with anything.
        (pa.null(), pa.int64(), pa.int64()),
        (pa.null(), pa.string(), pa.string()),
        (pa.null(), pa.null(), pa.null()),
        # Numerics, as DuckDB promotes them: a float on either side wins, two integers of
        # any width meet at int64 (which is what the FFI boundary widens them to anyway).
        (pa.int32(), pa.uint8(), pa.int64()),
        (pa.int64(), pa.float64(), pa.float64()),
        (pa.float32(), pa.float64(), pa.float64()),
        # Boolean widens into a number (`SELECT true UNION SELECT 1` is INTEGER in DuckDB,
        # with `true` reading as 1) and into nothing else.
        (pa.bool_(), pa.int64(), pa.int64()),
        (pa.bool_(), pa.string(), None),
        # Decimals keep the finer scale and the wider integer part: (10,2) and (12,4) both
        # carry 8 integer digits, so the result is 8 + scale 4 = (12,4), as DuckDB returns.
        (pa.decimal128(10, 2), pa.decimal128(12, 4), pa.decimal128(12, 4)),
        (pa.decimal128(14, 4), pa.decimal128(10, 0), pa.decimal128(14, 4)),
        # An integer widens *into* the decimal so a money column keeps its cents; a float
        # dominates it instead (DuckDB casts DECIMAL up to DOUBLE).
        (pa.decimal128(10, 2), pa.int64(), pa.decimal128(21, 2)),
        (pa.decimal128(10, 2), pa.float64(), pa.float64()),
        # Past 38 digits no decimal128 holds both, and the honest answer is to decline
        # rather than truncate the scale.
        (pa.decimal128(38, 0), pa.decimal128(38, 10), None),
        # Temporal types widen to the finer resolution; a date is midnight, so widening it
        # into a timestamp is exact (DuckDB returns TIMESTAMP for DATE UNION TIMESTAMP).
        (pa.timestamp("ms"), pa.timestamp("us"), pa.timestamp("us")),
        (pa.date32(), pa.timestamp("us"), pa.timestamp("us")),
        (pa.date32(), pa.date64(), pa.date64()),
        (pa.time32("s"), pa.time32("ms"), pa.time32("ms")),
        (pa.time32("s"), pa.time64("us"), pa.time64("us")),
        (pa.duration("s"), pa.duration("ns"), pa.duration("ns")),
        # A differing timezone is a genuine disagreement about which instant a stored
        # value denotes, so it is declined rather than guessed at.
        (pa.timestamp("us", "UTC"), pa.timestamp("us"), None),
        # Wider offsets are lossless, exactly as int32/int64 widen.
        (pa.string(), pa.large_string(), pa.large_string()),
        (pa.binary(), pa.large_binary(), pa.large_binary()),
        # A dictionary is an encoding of its value type, not a distinct logical type.
        (pa.dictionary(pa.int32(), pa.string()), pa.string(), pa.string()),
        (pa.dictionary(pa.int8(), pa.int64()), pa.float64(), pa.float64()),
        # Genuinely incompatible pairs decline, so the caller raises a typed error instead
        # of a lenient cast silently nulling the non-conforming side.
        (pa.int64(), pa.string(), None),
        (pa.date32(), pa.int64(), None),
        (pa.string(), pa.timestamp("us"), None),
    ],
)
def test_promote_matches_the_engine_lattice(a, b, expected):
    assert promote(a, b) == expected


@pytest.mark.parametrize(("a", "b"), list(product(LATTICE_TYPES, LATTICE_TYPES)))
def test_promote_is_commutative(a, b):
    """The answer cannot depend on operand order, or two callers would disagree."""
    assert promote(a, b) == promote(b, a)


@pytest.mark.parametrize("dtype", LATTICE_TYPES)
def test_promote_is_idempotent(dtype):
    """A type paired with itself is itself — a no-op union must not rewrite the schema."""
    assert promote(dtype, dtype) == dtype


@pytest.mark.parametrize(("a", "b"), list(product(LATTICE_TYPES, LATTICE_TYPES)))
def test_promote_never_narrows_either_side(a, b):
    """Whatever the lattice returns, both inputs must cast into it without an error.

    This is the property the lattice exists to guarantee, and it is the one a table of
    hand-written expectations cannot check: a rule that returned a *narrower* type would
    look plausible in isolation and corrupt data at the cast. Casting a one-element array
    of each input type to the result proves arrow accepts the widening in both directions.
    """
    common = promote(a, b)
    if common is None:
        return
    for side in (a, b):
        pa.nulls(1, type=side).cast(common)


def test_the_lattice_is_not_the_arithmetic_result_type():
    """`promote` answers a union's question, and arithmetic must not borrow the answer.

    A `decimal(10,2)` and an `int64` *union* to a decimal wide enough for either
    (`decimal(21,2)`), but they *add* to `decimal(11,2)` — one carry digit past the widest
    operand. Wiring the lattice into the arithmetic inference path made `Dataset.schema`
    advertise the union type for a sum, which is exactly the lying schema the type
    inference module exists to prevent. Inference stays silent for operand types whose
    arithmetic result it cannot reproduce.
    """
    import batcher as bt
    from batcher.plan.expr_ir import Binary, Col
    from batcher.plan.types.infer import infer_type

    table = pa.table(
        {
            "d": pa.array([Decimal("1.50")], pa.decimal128(10, 2)),
            "i": pa.array([3], pa.int64()),
            "f": pa.array([1.5], pa.float64()),
        }
    )
    schema = bt.from_arrow(table)._plan.available_schema()
    assert schema is not None
    # The lattice has an answer for the pair...
    assert promote(pa.decimal128(10, 2), pa.int64()) == pa.decimal128(21, 2)
    # ...and arithmetic inference must not use it.
    assert infer_type(Binary("add", Col("d"), Col("i")), schema) is None
    # The numeric operands, where the two answers do agree, still infer.
    assert infer_type(Binary("add", Col("i"), Col("f")), schema) == pa.float64()
