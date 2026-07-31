"""The torch scatter-reduce group-by kernel, checked against the CPU engine on CPU torch.

The kernel is device-parameterized on purpose, so the densify-and-scatter algorithm is
verifiable without a GPU — the device is only *where* it runs. That seam had four tests and
none of them carried a null, which is how every reduction over a null-bearing column came to
return `NaN`: a dense tensor has no null mask, so reading the column through NumPy turns each
null into `NaN` and one of those makes the whole group's `scatter_add` `NaN`.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from batcher._internal.errors import BackendError
from batcher.core.gpu_transform import _torch_groupby_agg

pytestmark = pytest.mark.unit

pytest.importorskip("torch", reason="the kernel under test is a torch kernel")

NAN = float("nan")
FUNCS = {"sum": "sum", "mean": "mean", "min": "min", "max": "max", "count": "count"}


def _engine(table: pa.Table, red: str) -> dict:
    out = bt.from_arrow(table).group_by("k").agg(r=getattr(col("v"), FUNCS[red])()).collect()
    return {row["k"]: row["r"] for row in out.to_pylist()}


def _kernel(table: pa.Table, red: str) -> dict:
    out = _torch_groupby_agg(table, "k", {"r": ("v", red)}, device="cpu")
    return {row["k"]: row["r"] for row in out.to_pylist()}


def _same(got: dict, want: dict) -> bool:
    if got.keys() != want.keys():
        return False
    for key, expected in want.items():
        actual = got[key]
        if expected is None or actual is None:
            if expected is not actual:
                return False
        elif actual != actual or expected != expected:  # NaN on either side
            if not (actual != actual and expected != expected):
                return False
        elif abs(actual - expected) > 1e-9:
            return False
    return True


TABLES = {
    "clean": pa.table({"k": [1, 1, 2, 2, 3], "v": [1.0, 2.0, 3.0, 4.0, 5.0]}),
    "some nulls": pa.table({"k": [1, 1, 2, 2, 3], "v": [1.0, None, 3.0, None, None]}),
    "a group of only nulls": pa.table({"k": [1, 1, 2], "v": [None, None, 3.0]}),
    "a nan": pa.table({"k": [1, 1, 2], "v": [1.0, NAN, 3.0]}),
    "only nans": pa.table({"k": [1, 1, 2], "v": [NAN, NAN, 3.0]}),
    "integers": pa.table({"k": [1, 1, 2], "v": [1, 2, 3]}),
    "negatives": pa.table({"k": [1, 1, 2], "v": [-1.0, -2.0, -3.0]}),
    "one row": pa.table({"k": [7], "v": [1.5]}),
}


@pytest.mark.parametrize("red", list(FUNCS))
@pytest.mark.parametrize("case", list(TABLES))
def test_the_torch_kernel_matches_the_engine(case, red):
    table = TABLES[case]
    assert _same(_kernel(table, red), _engine(table, red))


def test_a_group_with_no_non_null_value_reduces_to_null():
    """Not to the accumulator's seed, which would read as a measurement of zero."""
    table = pa.table({"k": [1, 1, 2], "v": [None, None, 3.0]})
    for red in ("sum", "mean", "min", "max"):
        assert _kernel(table, red)[1] is None, red


def test_count_counts_values_not_rows():
    """The mirror of the same bug: the group's rows include the ones that held nothing."""
    table = pa.table({"k": [1, 1, 1, 2], "v": [1.0, None, None, 2.0]})
    assert _kernel(table, "count") == {1: 1, 2: 1}


def test_nan_loses_a_minimum_and_wins_a_maximum():
    """The engine orders `NaN` above every number; `amin` propagates it instead."""
    table = pa.table({"k": [1, 1], "v": [1.0, NAN]})
    assert _kernel(table, "min")[1] == 1.0
    assert _kernel(table, "max")[1] != _kernel(table, "max")[1]  # NaN


def test_a_group_of_only_nans_has_nan_for_its_minimum():
    table = pa.table({"k": [1, 1], "v": [NAN, NAN]})
    got = _kernel(table, "min")[1]
    assert got != got


@pytest.mark.parametrize(
    ("case", "table"),
    [
        ("a string key", pa.table({"k": ["a", "b"], "v": [1.0, 2.0]})),
        ("a null in the key", pa.table({"k": [1, None], "v": [1.0, 2.0]})),
    ],
)
def test_a_key_the_kernel_cannot_densify_is_refused_by_type(case, table):
    """A bare `TypeError` reached the caller, which has no handler for one.

    A string key has no tensor form, and a key carrying nulls comes back from NumPy as a float
    column whose null has become `NaN` — a different key and a different type from the one the
    engine groups by. Both are for the caller to route around, and a typed refusal is what
    lets it.
    """
    with pytest.raises(BackendError):
        _torch_groupby_agg(table, "k", {"r": ("v", "sum")}, device="cpu")
