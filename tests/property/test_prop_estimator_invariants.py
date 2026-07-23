"""Property: laws the cardinality estimator must obey for *any* input.

These need no oracle — they are invariants of the estimation math, so a counterexample is
unambiguously a bug. They guard the whole estimation layer against a regression that a
hand-written example would miss:

  * **Selectivity is a probability** — `predicate_selectivity ∈ [0, 1]` for every predicate
    over every column-statistics bundle, however the ndv/bounds/nulls/mcv are shaped.
  * **A negation cannot gain rows** — `sel(p) + sel(NOT p) ≤ 1` (SQL drops the null rows
    from both), and both stay in `[0, 1]`.
  * **`combine_ndv` respects the Fréchet bounds** — the combined distinct count lies in
    `[max_i d_i, cap]`, is at least each input, and never exceeds the row cap.
  * **Row counts are non-negative and monotone** — a `Filter` never produces more rows than
    its input; a `Limit(n)` never more than `n`; no operator estimates a negative count.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import batcher as bt
from batcher.config import active_config
from batcher.kyber.stats import StatsEstimator
from batcher.kyber.stats.estimator import combine_ndv
from batcher.kyber.stats.selectivity import predicate_selectivity
from batcher.plan.stats import ColumnStat, Provenance

pytestmark = pytest.mark.property

_CFG = active_config().optimizer.cardinality

# --- strategies -------------------------------------------------------------

_ints = st.integers(min_value=-50, max_value=50)
_freqs = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)


@st.composite
def _column_stats(draw):
    """A `{col: ColumnStat}` bundle with a plausible mix of ndv/bounds/nulls/mcv."""
    ndv = draw(st.one_of(st.none(), st.floats(min_value=1.0, max_value=1000.0, allow_nan=False)))
    lo = draw(st.one_of(st.none(), _ints))
    hi = draw(st.one_of(st.none(), _ints))
    if lo is not None and hi is not None and lo > hi:
        lo, hi = hi, lo
    null_count = draw(st.one_of(st.none(), st.floats(min_value=0.0, max_value=100.0)))
    mcv = draw(st.one_of(st.none(), st.dictionaries(st.sampled_from(["1", "5", "10"]), _freqs)))
    return {
        "x": ColumnStat(
            min=lo, max=hi, ndv=ndv, null_count=null_count, mcv=mcv, provenance=Provenance.DEFAULT
        )
    }


def _predicates():
    """A handful of predicate shapes over column `x`."""
    x = bt.col("x")
    return st.sampled_from(
        [
            x == 5,
            x != 5,
            x < 5,
            x >= 5,
            (x >= 2) & (x <= 8),
            (x == 1) | (x == 2),
            ~(x > 5),
            x.is_null(),
            x.is_in([1, 2, 3]),
            x.str.contains("a"),
        ]
    )


# --- selectivity is a probability -------------------------------------------


@settings(max_examples=200, deadline=None)
@given(cols=_column_stats(), pred=_predicates())
def test_selectivity_is_in_the_unit_interval(cols, pred):
    sel = predicate_selectivity(
        pred,
        {n: c.ndv for n, c in cols.items() if c.ndv},
        _CFG,
        None,
        {n: dict(c.mcv) for n, c in cols.items() if c.mcv},
        {n: (c.min, c.max) for n, c in cols.items() if c.min is not None and c.max is not None},
        {n: c.null_count / 1000.0 for n, c in cols.items() if c.null_count is not None},
    )
    assert 0.0 <= sel <= 1.0


@settings(max_examples=200, deadline=None)
@given(cols=_column_stats(), pred=_predicates())
def test_a_predicate_and_its_negation_do_not_exceed_one(cols, pred):
    ndv = {n: c.ndv for n, c in cols.items() if c.ndv}
    bounds = {n: (c.min, c.max) for n, c in cols.items() if c.min is not None and c.max is not None}
    nulls = {
        n: min(1.0, c.null_count / 1000.0) for n, c in cols.items() if c.null_count is not None
    }
    p = predicate_selectivity(pred, ndv, _CFG, None, None, bounds, nulls)
    notp = predicate_selectivity(~pred, ndv, _CFG, None, None, bounds, nulls)
    assert 0.0 <= p <= 1.0
    assert 0.0 <= notp <= 1.0
    assert p + notp <= 1.0 + 1e-9


# --- combine_ndv respects the Fréchet bounds --------------------------------


@settings(max_examples=200, deadline=None)
@given(
    counts=st.lists(
        st.floats(min_value=1.0, max_value=1e6, allow_nan=False), min_size=1, max_size=5
    ),
    cap=st.floats(min_value=1.0, max_value=1e9, allow_nan=False),
)
def test_combine_ndv_is_between_max_and_cap(counts, cap):
    result = combine_ndv(counts, cap)
    assert result >= 1.0
    assert result <= cap + 1e-6
    # At least the largest single count (never below the functional-dependence floor), unless
    # the cap forces it lower.
    assert result >= min(max(counts), cap) - 1e-6


@settings(max_examples=100, deadline=None)
@given(
    counts=st.lists(
        st.floats(min_value=1.0, max_value=1000.0, allow_nan=False), min_size=1, max_size=4
    ),
    extra=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False),
)
def test_combine_ndv_is_monotone_in_adding_a_column(counts, extra):
    cap = 1e12  # large enough not to bind
    base = combine_ndv(counts, cap)
    more = combine_ndv([*counts, extra], cap)
    assert more >= base - 1e-6  # another key can only add distinct combinations


# --- estimator row counts are non-negative and monotone ---------------------


@settings(max_examples=100, deadline=None)
@given(n=st.integers(min_value=0, max_value=500))
def test_filter_never_exceeds_its_input(n):
    ds = bt.from_pydict({"x": list(range(max(1, n)))})
    est = StatsEstimator(ds._sources, {}, _CFG)
    input_rows = est.estimate(ds._plan).rows
    filtered = est.estimate(ds.filter(bt.col("x") > bt.lit(0))._plan).rows
    assert 0.0 <= filtered <= input_rows + 1e-9


@settings(max_examples=100, deadline=None)
@given(limit=st.integers(min_value=0, max_value=1000))
def test_limit_never_exceeds_n(limit):
    ds = bt.from_pydict({"x": list(range(200))})
    est = StatsEstimator(ds._sources, {}, _CFG)
    rows = est.estimate(ds.limit(limit)._plan).rows
    assert 0.0 <= rows <= limit
