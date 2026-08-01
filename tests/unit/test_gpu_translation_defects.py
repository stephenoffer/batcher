"""Translations that returned a wrong answer, a wrong column, or a false backend defect.

Every case here was found by replaying the *whole* public expression surface through the
translator and comparing against the engine, and every one of them passed the existing suite
while being wrong. They are grouped by how they failed, because the three failure modes need
different things from a reader:

* a **wrong answer** — `strip_chars` ignored the characters it was given and stripped
  whitespace instead, so a value with no leading space came back untouched;
* a **wrong column** — `is_leap_year` and the bit operators over a boolean returned the right
  values in the wrong type. That is not cosmetic here: a fan-out concatenates its shards, and a
  shard that fell back to the CPU engine contributes the engine's type beside this one's;
* a **false defect** — `substr` with no length and the shift operators raised `KeyError` and
  `TypeError`, neither of which is an `Unsupported`. The query still got its answer from the
  CPU engine, but the backend was reported as *broken* (`gpu_backend.failure`) for what is an
  ordinary expression, which is exactly the signal that is supposed to mean a real outage.

The oracle is the native engine, and the backend under test is pandas, which models the device
faithfully (`gpu_plan.backend`) — a GPU is only *where* a translated plan runs, never *what* it
computes.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from batcher.core.gpu_plan import DfBackend, gpu_plan_ops
from batcher.core.gpu_plan.execute import run_chain

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def be():
    import pandas as pd

    return DfBackend(pd)


TEXT = pa.table(
    {
        # `xxaybxx` strips on both ends, `  ab  ` has no stripped character at either end (the
        # case the whitespace default silently got right for the wrong reason), `aaa` strips to
        # nothing, and `abcba` strips one character from each end.
        "s": pa.array(["xxaybxx", "  ab  ", "aaa", None, "abcba"], pa.string()),
    }
)

NUMBERS = pa.table(
    {
        "i": pa.array([1, -8, 3, None, 7], pa.int64()),
        "b": pa.array([True, False, True, None, False], pa.bool_()),
    }
)

INSTANTS = pa.table(
    {
        "t": pa.array(
            [
                dt.datetime(2024, 2, 29),  # a leap year
                dt.datetime(2023, 3, 1),  # a common year
                dt.datetime(1900, 1, 1),  # divisible by 100 and NOT a leap year
                None,
                dt.datetime(2000, 6, 15),  # divisible by 400 and a leap year
            ],
            pa.timestamp("us"),
        )
    }
)


def _rows(table: pa.Table) -> list[tuple]:
    return [tuple(r) for r in zip(*table.to_pydict().values(), strict=True)]


def _assert_matches_engine(ds, table: pa.Table, be) -> None:
    """The translated result equals the engine's, column types included."""
    spec = gpu_plan_ops(ds._plan)
    assert spec is not None, "the translator declined a plan it is supposed to match"
    expected = ds.collect()
    got = be.to_arrow(run_chain(table, spec[1], be)).select(expected.column_names)
    assert _rows(got) == _rows(expected)
    assert got.schema.types == expected.schema.types


# --- a wrong answer: the strip functions ignored their character set ------------------


@pytest.mark.parametrize("fn", ["strip_chars", "strip_chars_start", "strip_chars_end"])
def test_stripping_a_character_set_strips_those_characters(be, fn):
    """`strip_chars("ax")` removes `a` and `x`, not whitespace.

    The pattern was dropped on the way through, so the translation stripped whitespace — which
    is the same answer on a value that happens to start with a space, and a silently different
    one on every other value.
    """
    ds = bt.from_arrow(TEXT).select(out=getattr(col("s").str, fn)("ax"))
    _assert_matches_engine(ds, TEXT, be)


def test_stripping_nothing_still_means_whitespace(be):
    """`strip()` with no character set is the whitespace default on both sides."""
    ds = bt.from_arrow(TEXT).select(out=col("s").str.strip())
    _assert_matches_engine(ds, TEXT, be)


# --- a false defect: `substr` with no length ------------------------------------------


def test_substr_runs_to_the_end_of_the_string_without_a_length(be):
    """`substr(s, 2)` has no `length` field at all, and reading one raised `KeyError`.

    Not an `Unsupported`, so it did not read as a decline: it reached the caller as a backend
    defect and was logged as "the GPU backend is not usable".
    """
    ds = bt.from_arrow(TEXT).select(out=col("s").str.substr(2))
    _assert_matches_engine(ds, TEXT, be)


def test_capitalize_reaches_the_device(be):
    """`capitalize` lowers to an open-ended `substr`, which is how the defect above shipped."""
    ds = bt.from_arrow(TEXT).select(out=col("s").str.capitalize())
    _assert_matches_engine(ds, TEXT, be)


# --- a wrong column: booleans through the bit operators --------------------------------


@pytest.mark.parametrize("fn", ["bitwise_and", "bitwise_or", "bitwise_xor", "xor"])
def test_a_bit_operator_over_booleans_answers_in_an_integer(be, fn):
    """The engine (and DuckDB) return an integer; both libraries return a boolean.

    The values agree and the column does not, which a fan-out cannot concatenate.
    """
    ds = bt.from_arrow(NUMBERS).select(out=getattr(col("b"), fn)(col("b")))
    _assert_matches_engine(ds, NUMBERS, be)


def test_a_bit_operator_over_integers_stays_an_integer(be):
    """The integer case must not be disturbed by the boolean correction above."""
    ds = bt.from_arrow(NUMBERS).select(out=col("i").bitwise_and(col("i")))
    _assert_matches_engine(ds, NUMBERS, be)


# --- a false defect: the shift operators ------------------------------------------------


@pytest.mark.parametrize("shift", [0, 1, 2, 10, 30])
def test_shifting_left_is_multiplication_by_a_power_of_two(be, shift):
    ds = bt.from_arrow(NUMBERS).select(out=col("i").bitwise_left_shift(shift))
    _assert_matches_engine(ds, NUMBERS, be)


def test_a_left_shift_that_would_overflow_declines(be):
    """The engine wraps; one backend raises and the other wraps, so neither runs it.

    Declined on the *data* rather than on the type, so an ordinary shift still reaches the
    device — the check is one reduction beside a multiplication over the whole column.
    """
    from batcher.core.gpu_plan.backend import Unsupported

    ds = bt.from_arrow(NUMBERS).select(out=col("i").bitwise_left_shift(62))
    spec = gpu_plan_ops(ds._plan)
    assert spec is not None
    with pytest.raises(Unsupported):
        run_chain(NUMBERS, spec[1], be)


@pytest.mark.parametrize("shift", [0, 1, 2, 10])
def test_shifting_right_is_floor_division_including_for_negatives(be, shift):
    """An arithmetic right shift *is* floor division on two's complement — `-8 >> 1` is `-4`,
    and so is `floor(-8 / 2)`. The negative row is the one a truncating division gets wrong."""
    ds = bt.from_arrow(NUMBERS).select(out=col("i").bitwise_right_shift(shift))
    _assert_matches_engine(ds, NUMBERS, be)


def test_a_shift_past_the_word_width_declines_rather_than_overflowing(be):
    """No portable answer exists for it, so the chain goes to the CPU engine."""
    from batcher.core.gpu_plan.backend import Unsupported

    ds = bt.from_arrow(NUMBERS).select(out=col("i").bitwise_left_shift(64))
    spec = gpu_plan_ops(ds._plan)
    assert spec is not None
    with pytest.raises(Unsupported):
        run_chain(NUMBERS, spec[1], be)


# --- a wrong column: `is_leap_year` ------------------------------------------------------


def test_is_leap_year_is_a_boolean(be):
    """It rode the int64 cast every other `.dt` attribute takes and came back as `1`/`0`."""
    ds = bt.from_arrow(INSTANTS).select(out=col("t").dt.is_leap_year())
    _assert_matches_engine(ds, INSTANTS, be)


def test_days_in_year_is_the_case_built_on_it(be):
    """`days_in_year` lowers to a `CASE` over `is_leap_year`, and an integer condition made the
    branch selection raise rather than return a number."""
    ds = bt.from_arrow(INSTANTS).select(out=col("t").dt.days_in_year())
    _assert_matches_engine(ds, INSTANTS, be)


# --- a wrong column: the grouped product -------------------------------------------------


def test_a_grouped_product_answers_in_a_double(be):
    """The engine answers `product` in double whatever the input type, because a running
    product leaves a bigint's range almost immediately. Both libraries keep the integer."""
    table = pa.table(
        {
            "k": ["a", "b", "a", "b", "c"],
            "i": pa.array([2, 3, 5, 7, None], pa.int64()),
        }
    )
    ds = bt.from_arrow(table).group_by("k").agg(r=col("i").product())
    spec = gpu_plan_ops(ds._plan)
    assert spec is not None
    expected = ds.collect()
    got = be.to_arrow(run_chain(table, spec[1], be)).select(expected.column_names)
    assert got.schema.types == expected.schema.types
    assert sorted(_rows(got), key=str) == sorted(_rows(expected), key=str)
