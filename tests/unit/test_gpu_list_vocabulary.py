"""The GPU translator's list and vector vocabulary, checked against the CPU engine.

An embedding column is the single most common reason to own a device, and every reduction over
one — `sum`, the norms, the dot product, cosine similarity, every vector distance — used to
send the whole chain to the host, because neither dataframe library exposes a list reduction
both of them have. `vocab.lists` builds them from `explode` plus `groupby` instead; this module
is the evidence that the construction is *exact* rather than close.

The cases are chosen for the four things the naive form gets wrong, each of which returns a
plausible number rather than an error:

* an **empty list** reduces to null for a measurement and to zero for a count — and the two
  families disagree: `sum([])` is null while `dot([], [])` is `0.0`;
* a **null inside a list** is skipped, so a list of nothing but nulls reduces to null;
* a **pairwise operation masks by both sides**, so `cosine_similarity([1, null], [2, 3])` is
  `1.0` — which only holds if the second position leaves both norms too;
* the **element type is kept** by `sum`, `min` and `max` and widened by everything else, and a
  shard that returns the right number in the wrong column cannot be concatenated with its peers.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from batcher.core.gpu_plan import DfBackend, gpu_plan_ops
from batcher.core.gpu_plan.backend import Unsupported
from batcher.core.gpu_plan.execute import run_chain

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def be():
    import pandas as pd

    return DfBackend(pd)


def _lists(kind: pa.DataType) -> pa.Table:
    """One list column covering every shape a reduction has to answer differently.

    Ordinary, single-element, empty, missing, all-null, null-bearing, and negative-valued —
    and a repeat, so `n_unique` and `median` have something to distinguish.
    """
    values = [
        [1, 2, 3],
        [4],
        [],
        None,
        [None, None],
        [5, None, 7],
        [-2, 6, -2],
    ]
    return pa.table({"v": pa.array(values, pa.list_(kind))})


FLOATS = _lists(pa.float64())
INTS = _lists(pa.int64())

#: Two aligned vector columns: ordinary, orthogonal, empty, missing, a zero vector (no
#: direction, so no similarity), a null-bearing pair, and a pair that is identical.
PAIRS = pa.table(
    {
        "a": pa.array(
            [[1.0, 2.0], [1.0, 0.0], [], None, [0.0, 0.0], [1.0, None], [3.0, 4.0]],
            pa.list_(pa.float64()),
        ),
        "b": pa.array(
            [[3.0, 4.0], [0.0, 1.0], [], None, [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
            pa.list_(pa.float64()),
        ),
    }
)


def _rows(table: pa.Table) -> list[tuple]:
    def canon(v):
        return float(f"{v:.12e}") if isinstance(v, float) and v == v else v

    return [tuple(canon(v) for v in r) for r in zip(*table.to_pydict().values(), strict=True)]


def _assert_matches_engine(ds, table: pa.Table, be) -> None:
    spec = gpu_plan_ops(ds._plan)
    assert spec is not None, "the translator declined a plan it is supposed to match"
    expected = ds.collect()
    got = be.to_arrow(run_chain(table, spec[1], be)).select(expected.column_names)
    assert _rows(got) == _rows(expected)
    assert got.schema.types == expected.schema.types


# --- one-list reductions -----------------------------------------------------------------

REDUCTIONS = [
    "sum", "mean", "min", "max", "product", "std", "var", "median",
    "n_unique", "len", "l1_norm", "l2_norm", "max_abs", "arg_max", "arg_min",
]  # fmt: skip


@pytest.mark.parametrize("fn", REDUCTIONS)
@pytest.mark.parametrize("table", [FLOATS, INTS], ids=["float", "int"])
def test_a_list_reduction_matches_the_engine(be, fn, table):
    """Including the column type: `sum` of a bigint list is a bigint, `mean` is a double."""
    ds = bt.from_arrow(table).select(out=getattr(col("v").list, fn)())
    _assert_matches_engine(ds, table, be)


@pytest.mark.parametrize("fn", ["magnitude", "mean_pool", "max_pool", "is_unit_norm",
                                "is_zero_vector", "sum_squares", "dim"])  # fmt: skip
def test_the_vector_spellings_reach_the_device(be, fn):
    """The `.list` namespace's vector-shaped aliases lower onto the same reductions."""
    ds = bt.from_arrow(FLOATS).select(out=getattr(col("v").list, fn)())
    _assert_matches_engine(ds, FLOATS, be)


def test_an_empty_list_reduces_to_null_and_not_to_the_identity(be):
    """`sum([])` is null, not `0.0` — the libraries return the fold's identity, which reads as
    a real measurement."""
    table = pa.table({"v": pa.array([[], [1.0]], pa.list_(pa.float64()))})
    ds = bt.from_arrow(table).select(out=col("v").list.sum())
    _assert_matches_engine(ds, table, be)


def test_a_list_of_nothing_but_nulls_reduces_to_null(be):
    table = pa.table({"v": pa.array([[None, None], [1.0]], pa.list_(pa.float64()))})
    ds = bt.from_arrow(table).select(out=col("v").list.mean())
    _assert_matches_engine(ds, table, be)


def test_an_empty_list_still_has_a_length_and_a_distinct_count(be):
    """`len` and `n_unique` count rather than measure, so their answer over nothing is `0`."""
    ds = bt.from_arrow(FLOATS).select(n=col("v").list.len(), u=col("v").list.n_unique())
    _assert_matches_engine(ds, FLOATS, be)


def test_the_positional_reductions_break_ties_on_the_first_occurrence(be):
    """`arg_min([-2, 6, -2])` is `0`, not `2` — which is what an index lookup after an explode
    would give if it reported the *last* match."""
    table = pa.table({"v": pa.array([[-2.0, 6.0, -2.0], [3.0, 3.0]], pa.list_(pa.float64()))})
    ds = bt.from_arrow(table).select(a=col("v").list.arg_min(), b=col("v").list.arg_max())
    _assert_matches_engine(ds, table, be)


def test_a_positional_reduction_over_a_nan_declines(be):
    """The engine orders `NaN` above every number and both libraries treat it as missing, so
    the extreme would be a different element — the same decline the grouped order statistics
    take."""
    table = pa.table({"v": pa.array([[1.0, float("nan"), 2.0]], pa.list_(pa.float64()))})
    ds = bt.from_arrow(table).select(out=col("v").list.arg_max())
    spec = gpu_plan_ops(ds._plan)
    assert spec is not None
    with pytest.raises(Unsupported):
        run_chain(table, spec[1], be)


# --- element access ------------------------------------------------------------------------


@pytest.mark.parametrize("index", [0, 1, 2, -1, -2, 5, -9])
def test_getting_an_element_by_index(be, index):
    """Out of range is null, where both libraries' own accessors raise instead."""
    ds = bt.from_arrow(FLOATS).select(out=col("v").list.get(index))
    _assert_matches_engine(ds, FLOATS, be)


def test_getting_an_element_that_is_itself_null(be):
    """The one case a mask-and-reduce selection gets wrong: it would skip the null and hand
    back a later element."""
    table = pa.table({"v": pa.array([[None, 5.0], [1.0, 2.0]], pa.list_(pa.float64()))})
    ds = bt.from_arrow(table).select(out=col("v").list.get(0))
    _assert_matches_engine(ds, table, be)


@pytest.mark.parametrize("fn", ["first", "last"])
def test_the_end_elements(be, fn):
    ds = bt.from_arrow(FLOATS).select(out=getattr(col("v").list, fn)())
    _assert_matches_engine(ds, FLOATS, be)


def test_element_at_is_one_based(be):
    ds = bt.from_arrow(FLOATS).select(out=col("v").list.element_at(1))
    _assert_matches_engine(ds, FLOATS, be)


@pytest.mark.parametrize("value", [1.0, 5.0, 99.0])
def test_membership_and_position(be, value):
    """`contains` is false over an empty list and null over a missing one; `position` is
    1-based and null when there is no match."""
    ds = bt.from_arrow(FLOATS).select(
        c=col("v").list.contains(value), p=col("v").list.position(value)
    )
    _assert_matches_engine(ds, FLOATS, be)


# --- pairwise vector functions -------------------------------------------------------------


@pytest.mark.parametrize(
    "fn",
    ["dot", "cosine_similarity", "cosine_distance", "l1_distance", "l2_distance",
     "euclidean_distance", "angular_distance", "hamming_distance"],
)  # fmt: skip
def test_a_vector_function_matches_the_engine(be, fn):
    ds = bt.from_arrow(PAIRS).select(out=getattr(col("a").list, fn)(col("b")))
    _assert_matches_engine(ds, PAIRS, be)


def test_a_dot_product_over_two_empty_lists_is_zero_not_null(be):
    """A sum over no terms is zero; a measurement over no values is unknown. The two families
    take opposite answers over the same empty list, which is why they are separated."""
    table = pa.table(
        {
            "a": pa.array([[], [1.0]], pa.list_(pa.float64())),
            "b": pa.array([[], [2.0]], pa.list_(pa.float64())),
        }
    )
    ds = bt.from_arrow(table).select(out=col("a").list.dot(col("b")))
    _assert_matches_engine(ds, table, be)


def test_a_similarity_masks_both_norms_by_both_sides(be):
    """`cosine_similarity([1, null], [2, 3])` is `1.0`, which holds only if the second position
    leaves the right-hand norm as well as the dot product."""
    table = pa.table(
        {
            "a": pa.array([[1.0, None]], pa.list_(pa.float64())),
            "b": pa.array([[2.0, 3.0]], pa.list_(pa.float64())),
        }
    )
    ds = bt.from_arrow(table).select(out=col("a").list.cosine_similarity(col("b")))
    _assert_matches_engine(ds, table, be)


def test_vectors_of_unequal_length_decline_rather_than_truncating(be):
    """The engine raises on a length mismatch, so the translation must not quietly answer it —
    declining sends the chain to the CPU engine, which raises the engine's own error."""
    table = pa.table(
        {
            "a": pa.array([[1.0, 2.0, 3.0]], pa.list_(pa.float64())),
            "b": pa.array([[1.0, 2.0]], pa.list_(pa.float64())),
        }
    )
    ds = bt.from_arrow(table).select(out=col("a").list.dot(col("b")))
    spec = gpu_plan_ops(ds._plan)
    assert spec is not None
    with pytest.raises(Unsupported):
        run_chain(table, spec[1], be)


def test_a_list_to_list_function_is_still_declined(be):
    """`sort` returns a list, and reassembling one costs a Python object per row — a hot-path
    tuple touch rather than a translation. It goes to the CPU engine on purpose."""
    ds = bt.from_arrow(FLOATS).select(out=col("v").list.sort())
    spec = gpu_plan_ops(ds._plan)
    if spec is not None:
        with pytest.raises(Unsupported):
            run_chain(FLOATS, spec[1], be)


# --- the reductions compose with the operators around them ---------------------------------


def test_a_vector_search_shape_runs_end_to_end(be):
    """Filter, score, and rank by similarity — the shape a vector search actually has."""
    ds = (
        bt.from_arrow(PAIRS)
        .with_columns(score=col("a").list.cosine_similarity(col("b")))
        .filter(col("score").is_not_null())
        .select("score")
    )
    _assert_matches_engine(ds, PAIRS, be)


def test_a_grouped_reduction_over_a_vector_norm(be):
    """The norms feed the grouped aggregates, so the whole chain stays on the device."""
    table = pa.table(
        {
            "k": ["a", "a", "b", "b"],
            "v": pa.array([[1.0, 2.0], [3.0], [4.0, 5.0], []], pa.list_(pa.float64())),
        }
    )
    ds = bt.from_arrow(table).group_by("k").agg(m=col("v").list.l2_norm().mean())
    spec = gpu_plan_ops(ds._plan)
    assert spec is not None
    expected = ds.collect()
    got = be.to_arrow(run_chain(table, spec[1], be)).select(expected.column_names)
    assert sorted(_rows(got), key=str) == sorted(_rows(expected), key=str)
    assert got.schema.types == expected.schema.types
