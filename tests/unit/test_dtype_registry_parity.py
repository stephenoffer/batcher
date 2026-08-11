"""The dtype-name vocabulary stays in lockstep across the tiers.

`Expr.cast` carries the target dtype as a raw string on the JSON IR wire, so the set of
accepted names is part of the contract with the Rust engine. The canonical implementation
lives in `bc_arrow::dtype_name`; `plan.types.registry` mirrors it on the Python side.

The vocabulary has two halves and they need two different proofs:

- The **fixed** names are a set, so they are pinned by comparing the two sets
  (`bc-py::supported_cast_dtypes` against `CAST_DTYPES`).
- The **parametrized** names are a grammar — ``decimal(12,4)``, ``timestamp(us, UTC)``,
  ``time64(ns)`` — with more legal spellings than can be listed. A name list cannot pin
  those, so each spelling is resolved through *both* implementations and the resulting
  Arrow types compared. That checks the property that actually matters: not that both
  sides accept the name, but that both sides build the same type from it.
"""

from __future__ import annotations

import batcher._native as _native
import pyarrow as pa
import pytest

from batcher.plan.types import (
    CAST_DTYPES,
    DTYPE_REGISTRY,
    canonical_dtype_name,
    resolve_dtype,
)

# Spellings that exercise every arm of the parametrized grammar, plus the malformed ones
# both sides must reject. Resolved through Rust and Python and compared type-for-type.
GRAMMAR_SPELLINGS = [
    # Decimals: explicit scale, defaulted scale, the aliases, the wide variant.
    "decimal(12,4)",
    "decimal(12, 4)",
    "decimal128(38,10)",
    "numeric(9)",
    "decimal256(50,10)",
    # Timestamps: bare, with a unit, with a unit and a zone (including a named zone).
    "timestamp",
    "timestamp(s)",
    "timestamp(ms)",
    "timestamp(us)",
    "timestamp(ns)",
    "timestamp(us, UTC)",
    "timestamp(ms, America/New_York)",
    "datetime(ns)",
    # Time of day: the unqualified form picks its width, the explicit forms are honored.
    "time(us)",
    "time(ms)",
    "time32(s)",
    "time32(ms)",
    "time64(us)",
    "time64(ns)",
    # Durations, in the Arrow spelling and the SQL word.
    "duration(s)",
    "duration(nanosecond)",
    "interval(ms)",
    # Rejected: out of range, impossible width, unparseable, unknown.
    "decimal(39,2)",
    "decimal(0,0)",
    "decimal(4,6)",
    "decimal256(77,2)",
    "decimal(x,2)",
    "decimal",
    "decimal(",
    "time32(us)",
    "time64(s)",
    "timestamp()",
    "duration(fortnight)",
    "int64(4)",
    "not_a_type",
    "",
]


def test_python_fixed_cast_dtypes_match_engine_vocabulary():
    engine = set(_native.supported_cast_dtypes())
    assert engine == set(CAST_DTYPES), (
        "plan.types.CAST_DTYPES drifted from bc_arrow::dtype_name; "
        f"python-only={set(CAST_DTYPES) - engine}, engine-only={engine - set(CAST_DTYPES)}"
    )


def test_registry_keys_equal_cast_dtypes():
    assert set(DTYPE_REGISTRY) == set(CAST_DTYPES)


@pytest.mark.parametrize("name", GRAMMAR_SPELLINGS)
def test_parametrized_grammar_resolves_identically_in_both_tiers(name):
    """Both implementations must build the same Arrow type, or both must reject it."""
    engine = _native.resolve_cast_dtype(name)
    ours = resolve_dtype(name)
    if engine is None or ours is None:
        assert engine is None and ours is None, (
            f"{name!r}: engine={engine}, python={ours} — one side accepts what the other "
            "rejects, so a cast that plans will fail in the engine (or vice versa)"
        )
        return
    assert ours == engine, f"{name!r}: python builds {ours}, engine builds {engine}"


@pytest.mark.parametrize("name", sorted(CAST_DTYPES))
def test_every_fixed_name_resolves_identically_in_both_tiers(name):
    assert resolve_dtype(name) == _native.resolve_cast_dtype(name)


def test_registry_aliases_collapse_to_one_type():
    assert DTYPE_REGISTRY["long"] == DTYPE_REGISTRY["int64"] == pa.int64()
    assert DTYPE_REGISTRY["bigint"] == pa.int64()
    assert DTYPE_REGISTRY["double"] == DTYPE_REGISTRY["float64"] == pa.float64()
    assert DTYPE_REGISTRY["int"] == DTYPE_REGISTRY["int32"] == pa.int32()
    assert DTYPE_REGISTRY["utf8"] == DTYPE_REGISTRY["string"] == pa.string()
    assert DTYPE_REGISTRY["varchar"] == pa.string()
    assert DTYPE_REGISTRY["datetime"] == DTYPE_REGISTRY["timestamp"] == pa.timestamp("us")


def test_bare_names_keep_their_historical_types():
    """A plan already on disk must lower to exactly the type it always did."""
    assert resolve_dtype("timestamp") == pa.timestamp("us")
    assert resolve_dtype("date") == pa.date32()
    assert resolve_dtype("int") == pa.int32()
    assert resolve_dtype("float") == pa.float32()


@pytest.mark.parametrize(
    ("written", "canonical"),
    [
        ("Int64", "int64"),
        ("BIGINT", "bigint"),
        ("DECIMAL(12,4)", "decimal(12, 4)"),
        ("Timestamp(NS)", "timestamp(ns)"),
        # The zone is the one part that keeps its case: Arrow compares it byte-wise, so
        # folding it would silently build a type the user did not ask for.
        ("TIMESTAMP(US, America/New_York)", "timestamp(us, America/New_York)"),
        ("TIMESTAMP(US, UTC)", "timestamp(us, UTC)"),
    ],
)
def test_names_are_canonicalized_without_folding_a_timezone(written, canonical):
    assert canonical_dtype_name(written) == canonical
    assert resolve_dtype(canonical) is not None


def test_a_folded_timezone_would_have_been_a_different_type():
    """The reason the canonicalizer spares the zone, stated as an assertion."""
    assert resolve_dtype("timestamp(us, UTC)") != resolve_dtype("timestamp(us, utc)")
