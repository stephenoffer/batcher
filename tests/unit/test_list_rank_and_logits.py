"""`list.gather`, `list.log_softmax`, and `list.entropy` — reranking and logit math in-engine.

`gather` exists because `arg_sort` alone is a dead end: it produces positions the caller had
no way to spend, so every rerank left the engine. The tests pin the pairing, and the edges a
rerank actually hits — a `head(k)` wider than the row, a tie, a null.

The two logit ops are pinned against their closed forms rather than against remembered
numbers, so a change in the kernel has to break the mathematics to pass.
"""

from __future__ import annotations

import math

import pytest

import batcher as bt

pytestmark = pytest.mark.unit


def _one(expr, **columns):
    return bt.from_pydict(columns).select(v=expr).to_pydict()["v"][0]


# --- gather ------------------------------------------------------------------------


def test_gather_takes_the_named_positions_in_order():
    got = _one(bt.col("xs").list.gather(bt.col("i")), xs=[["a", "b", "c"]], i=[[2, 0]])
    assert got == ["c", "a"]


def test_gather_completes_arg_sort_into_a_rerank():
    """The pairing the op exists for: rank by one column, take from another."""
    ranked = bt.col("scores").list.arg_sort().list.reverse()
    got = _one(
        bt.col("docs").list.gather(ranked.list.head(2)),
        docs=[["low", "high", "mid"]],
        scores=[[0.1, 0.9, 0.5]],
    )
    assert got == ["high", "mid"]


def test_gather_handles_a_cutoff_wider_than_the_row():
    """`head(k)` on a short candidate set is ordinary, so it must not raise."""
    ranked = bt.col("s").list.arg_sort().list.reverse().list.head(5)
    got = _one(bt.col("d").list.gather(ranked), d=[["a", "b"]], s=[[0.2, 0.8]])
    assert got == ["b", "a"]


def test_gather_reads_a_negative_index_from_the_end():
    got = _one(bt.col("xs").list.gather(bt.col("i")), xs=[["a", "b", "c"]], i=[[-1, -3]])
    assert got == ["c", "a"]


def test_gather_yields_null_for_an_out_of_range_position():
    got = _one(bt.col("xs").list.gather(bt.col("i")), xs=[["a", "b"]], i=[[0, 9, -9]])
    assert got == ["a", None, None]


def test_gather_may_repeat_a_position():
    got = _one(bt.col("xs").list.gather(bt.col("i")), xs=[["a", "b"]], i=[[1, 1]])
    assert got == ["b", "b"]


def test_gather_is_null_when_either_row_is_null():
    out = (
        bt.from_pydict({"xs": [None, ["a"]], "i": [[0], None]})
        .select(v=bt.col("xs").list.gather(bt.col("i")))
        .to_pydict()["v"]
    )
    assert out == [None, None]


def test_gather_keeps_rows_independent():
    """A shared child buffer must not let row 1's indices read row 0's values."""
    out = (
        bt.from_pydict({"xs": [["a0", "a1"], ["b0", "b1"]], "i": [[0], [0]]})
        .select(v=bt.col("xs").list.gather(bt.col("i")))
        .to_pydict()["v"]
    )
    assert out == [["a0"], ["b0"]]


def test_gather_works_on_a_numeric_list_too():
    got = _one(bt.col("xs").list.gather(bt.col("i")), xs=[[10, 20, 30]], i=[[1, 2]])
    assert got == [20, 30]


# --- log_softmax -------------------------------------------------------------------


def test_log_softmax_of_equal_logits_is_the_log_of_the_uniform_probability():
    got = _one(bt.col("z").list.log_softmax(), z=[[0.0, 0.0, 0.0]])
    assert got == pytest.approx([math.log(1 / 3)] * 3)


def test_log_softmax_exponentiates_back_to_softmax():
    """The defining relationship, checked against the engine's own softmax."""
    logits = [[1.0, 2.0, 3.0]]
    out = (
        bt.from_pydict({"z": logits})
        .select(lg=bt.col("z").list.log_softmax(), p=bt.col("z").list.softmax())
        .to_pydict()
    )
    assert [math.exp(v) for v in out["lg"][0]] == pytest.approx(out["p"][0])


def test_log_softmax_stays_finite_where_softmax_underflows():
    """The reason to work in the log domain at all."""
    got = _one(bt.col("z").list.log_softmax(), z=[[0.0, -900.0]])
    assert all(math.isfinite(v) for v in got)
    assert got[1] < -800


def test_log_softmax_is_shift_invariant():
    a = _one(bt.col("z").list.log_softmax(), z=[[1.0, 2.0, 3.0]])
    b = _one(bt.col("z").list.log_softmax(), z=[[101.0, 102.0, 103.0]])
    assert a == pytest.approx(b)


def test_log_softmax_preserves_a_null_row():
    out = (
        bt.from_pydict({"z": [None, [0.0, 0.0]]})
        .select(v=bt.col("z").list.log_softmax())
        .to_pydict()["v"]
    )
    assert out[0] is None


# --- entropy -----------------------------------------------------------------------


def test_entropy_is_zero_for_a_certain_distribution():
    assert _one(bt.col("p").list.entropy(), p=[[1.0, 0.0, 0.0]]) == 0.0


def test_entropy_is_log_n_for_a_uniform_distribution():
    for n in (2, 3, 8):
        got = _one(bt.col("p").list.entropy(), p=[[1.0 / n] * n])
        assert got == pytest.approx(math.log(n))


def test_entropy_normalizes_unnormalized_weights():
    """A count vector and the distribution it implies must score the same."""
    counts = _one(bt.col("p").list.entropy(), p=[[2.0, 2.0]])
    probs = _one(bt.col("p").list.entropy(), p=[[0.5, 0.5]])
    assert counts == pytest.approx(probs)


def test_entropy_rises_as_mass_spreads_out():
    peaked = _one(bt.col("p").list.entropy(), p=[[0.9, 0.05, 0.05]])
    flat = _one(bt.col("p").list.entropy(), p=[[0.34, 0.33, 0.33]])
    assert peaked < flat


def test_entropy_of_a_row_with_no_mass_is_null():
    out = (
        bt.from_pydict({"p": [[0.0, 0.0], [1.0, 1.0]]})
        .select(h=bt.col("p").list.entropy())
        .to_pydict()["h"]
    )
    assert out[0] is None
    assert out[1] == pytest.approx(math.log(2))


def test_entropy_is_null_for_a_null_row():
    out = (
        bt.from_pydict({"p": [None, [0.5, 0.5]]})
        .select(h=bt.col("p").list.entropy())
        .to_pydict()["h"]
    )
    assert out[0] is None


def test_entropy_composes_into_a_confidence_filter():
    """The use case: route the uncertain rows somewhere more expensive."""
    ds = bt.from_pydict({"id": [1, 2], "p": [[0.99, 0.01], [0.5, 0.5]]})
    uncertain = ds.filter(bt.col("p").list.entropy() > bt.lit(0.5)).to_pydict()
    assert uncertain["id"] == [2]
