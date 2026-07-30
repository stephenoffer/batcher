"""Plan-shape, soundness-guard, and does-not-fire tests for `sarg_bounded_ordered`.

`kyber/rules/extra/sargable_range` transposes constant arithmetic across `<`/`<=`/`>`/`>=`
— which `sargable.py` deliberately refuses, because the engine's i64 arithmetic wraps. The
whole rule rests on one proof: the column's recorded min/max show the arithmetic stays inside
i64. These tests pin all twelve rewrites, and — more importantly — pin every case where the
proof fails and the rule must decline.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from batcher.config import active_config
from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rules.extra.sargable_range import sarg_bounded_ordered
from batcher.kyber.rules.extra.sargable_range.shared import decompose
from batcher.plan.expr_ir import Binary, Col, Lit
from batcher.plan.stats import ColumnStat, Provenance, RelStats

_INT64_MIN, _INT64_MAX = -(2**63), 2**63 - 1


class _FixedStats:
    """An estimator stub reporting one fixed `RelStats` — the bounds are the whole input."""

    def __init__(self, columns: dict[str, ColumnStat]) -> None:
        self._stats = RelStats(rows=1000.0, provenance=Provenance.EXACT, columns=columns)

    def estimate(self, _node) -> RelStats:
        return self._stats


def _ctx(columns: dict[str, ColumnStat]) -> OptimizerContext:
    return OptimizerContext(
        config=active_config(), sources=[], hub=None, estimator=_FixedStats(columns)
    )


def _bounded(low: int = 0, high: int = 1000) -> OptimizerContext:
    """A context reporting the integer column `x` in `[low, high]` and nothing about `f`."""
    return _ctx({"x": ColumnStat(min=low, max=high)})


def _plan(pred):
    """A Filter over an int column `x` and a float column `f`."""
    ds = bt.from_arrow(pa.table({"x": pa.array([1, 2, 3], pa.int64()), "f": [1.0, 2.0, 3.0]}))
    return ds.filter(pred)._plan


def _fire(pred, ctx: OptimizerContext | None = None):
    """The rewritten predicate's IR, or ``None`` when the rule declined."""
    out = sarg_bounded_ordered(_plan(pred), ctx if ctx is not None else _bounded())
    return None if out is None else out.predicate.to_ir()


# --- registration -----------------------------------------------------------


def test_rule_is_registered_and_declares_every_ordered_comparison():
    registered = {r.name: r for r in DEFAULT_REGISTRY.rules()}
    assert "sarg_bounded_ordered" in registered
    # A predicate written with the literal first arrives mirrored (`500 < x + 1` is a `lt`),
    # and `decompose` normalizes it. The four ordered operators are closed under mirroring, so
    # declaring them covers both spellings; declaring fewer would make the rule silently stop
    # firing on the ones it dropped, with results still correct and only the plan degraded.
    assert registered["sarg_bounded_ordered"].expr_ops == frozenset({"lt", "le", "gt", "ge"})


def test_this_module_is_the_only_bounded_sargable_family():
    # The twin of `test_sargable.py`'s completeness guard, which excludes the `sarg_bounded_`
    # prefix on the understanding that this file covers it. If the family grows a second rule,
    # this fails until it is covered here too.
    assert {r.name for r in DEFAULT_REGISTRY.rules() if r.name.startswith("sarg_bounded_")} == {
        "sarg_bounded_ordered"
    }


# --- the rewrite: all twelve arithmetic-form x comparison combinations -------


@pytest.mark.parametrize(
    ("pred", "want"),
    [
        # col + k: the comparison is unchanged, the constant moves to the literal.
        ((col("x") + 1) < 500, Binary("lt", Col("x"), Lit(499))),
        ((col("x") + 1) <= 500, Binary("le", Col("x"), Lit(499))),
        ((col("x") + 1) > 500, Binary("gt", Col("x"), Lit(499))),
        ((col("x") + 1) >= 500, Binary("ge", Col("x"), Lit(499))),
        # col - k
        ((col("x") - 3) < 2, Binary("lt", Col("x"), Lit(5))),
        ((col("x") - 3) <= 2, Binary("le", Col("x"), Lit(5))),
        ((col("x") - 3) > 2, Binary("gt", Col("x"), Lit(5))),
        ((col("x") - 3) >= 2, Binary("ge", Col("x"), Lit(5))),
        # k - col: the comparison flips, because the column's coefficient is negative.
        ((5 - col("x")) < 2, Binary("gt", Col("x"), Lit(3))),
        ((5 - col("x")) <= 2, Binary("ge", Col("x"), Lit(3))),
        ((5 - col("x")) > 2, Binary("lt", Col("x"), Lit(3))),
        ((5 - col("x")) >= 2, Binary("le", Col("x"), Lit(3))),
        # A negative constant, and the commuted spelling of the addition.
        ((col("x") + -50) > 0, Binary("gt", Col("x"), Lit(50))),
        ((7 + col("x")) <= 9, Binary("le", Col("x"), Lit(2))),
    ],
)
def test_transposes_to_a_bare_column_comparison(pred, want):
    assert _fire(pred) == want.to_ir()


def test_unary_minus_is_the_rsub_form():
    # `-col` lowers to `0 - col`, so negation needs no case of its own.
    assert _fire(-col("x") >= -10) == Binary("le", Col("x"), Lit(10)).to_ir()


def test_literal_on_the_left_is_normalized():
    # `500 < x + 1` is `x + 1 > 500` written the other way round.
    assert _fire(Binary("lt", Lit(500), col("x") + 1)) == Binary("gt", Col("x"), Lit(499)).to_ir()


def test_rewrite_is_idempotent():
    # The output's column side is a bare `Col`, so the rule cannot match its own result —
    # required for the fixpoint driver.
    once = sarg_bounded_ordered(_plan((col("x") + 1) > 500), _bounded())
    assert sarg_bounded_ordered(once, _bounded()) is None


# --- the soundness guards: every way the proof can fail ----------------------


def test_declines_without_recorded_bounds():
    # No min/max means no proof that the addition stays inside i64.
    assert _fire((col("x") + 1) > 500, _ctx({})) is None


def test_declines_when_the_high_bound_lets_the_addition_wrap():
    # `x` may be INT64_MAX, where `x + 1` wraps to INT64_MIN and the two forms disagree.
    assert _fire((col("x") + 1) > 500, _bounded(0, _INT64_MAX)) is None


def test_declines_when_the_low_bound_lets_the_subtraction_wrap():
    assert _fire((col("x") - 3) <= 2, _bounded(_INT64_MIN, 0)) is None


def test_declines_when_the_reverse_subtraction_would_wrap():
    # `k - col` overflows at the *low* end: `0 - INT64_MIN` is not representable.
    assert _fire((0 - col("x")) < 2, _bounded(_INT64_MIN, 0)) is None


def test_declines_when_the_folded_literal_would_not_fit():
    # Bounds are fine, but `lit + k` leaves i64 — the rewrite would introduce the wrap it
    # exists to avoid.
    assert _fire((col("x") - _INT64_MAX) <= _INT64_MAX, _bounded(0, 10)) is None


def test_declines_on_a_float_column_even_with_integral_bounds():
    # Transposing a constant across a float comparison changes the rounding, so the type
    # guard — not the bounds — is what refuses this.
    assert _fire((col("f") + 1) > 5, _ctx({"f": ColumnStat(min=0, max=10)})) is None


def test_declines_for_a_non_integer_constant():
    assert _fire((col("x") + 1.5) > 5) is None


def test_declines_for_equality_which_needs_no_range_proof():
    # `=`/`<>` are `sargable.py`'s job — equality's bijection survives the wrap — and this
    # rule must not duplicate them.
    assert _fire((col("x") + 1) == 500) is None


# --- the guard through the whole optimizer -----------------------------------


def _optimized_predicates(table, pred):
    """The predicates surviving a full optimizer run over an in-memory source.

    Goes through the real pipeline — including `collect_source_stats`, which is what supplies
    the min/max this rule needs — so this asserts the shape a `collect()` would run, not the
    shape the rule produces when called directly.
    """
    from batcher import core, kyber
    from batcher.api.source_stats import collect_source_stats
    from batcher.plan.logical import Filter
    from batcher.plan.visitor import walk

    d = bt.from_arrow(table).filter(pred)
    hub = core.default_hub()
    plan = kyber.optimize_logical(
        d._plan, sources=d._sources, hub=hub, source_stats=collect_source_stats(d._sources, hub)
    )
    return [n.predicate.to_ir() for n in walk(plan) if isinstance(n, Filter)]


@pytest.mark.parametrize(
    "pred",
    [
        (col("x") + 1) < 500,
        (col("x") + 1) >= 500,
        (col("x") - 3) > 2,
        (col("x") - 3) <= 2,
        (5 - col("x")) < 2,
        (5 - col("x")) >= 2,
    ],
)
def test_registered_rule_still_rewrites(pred):
    # A rule the driver's traversal stops reaching is invisible: results stay correct and only
    # the predicate silently keeps its un-sargable shape. So this drives the *registry*.
    table = pa.table({"x": pa.array([0, 1, 2, 500, 1000], pa.int64())})
    got = _optimized_predicates(table, pred)
    assert len(got) == 1, f"expected one surviving Filter, got {got}"
    assert got[0]["left"] == Col("x").to_ir(), "the bare column is no longer exposed"
    assert got[0]["right"]["e"] == "lit", "the constant is no longer folded into a literal"


def test_end_to_end_declines_when_the_column_reaches_the_end_of_i64():
    # The engine's arithmetic wraps here, so `x + 1 > 0` is genuinely not `x > -1`: at
    # INT64_MAX the sum wraps negative. This is the case the differential suite cannot assert
    # against DuckDB (DuckDB promotes instead of wrapping), so it is pinned as plan shape.
    table = pa.table({"x": pa.array([_INT64_MIN, 0, _INT64_MAX], pa.int64())})
    unchanged = Binary("gt", Binary("add", Col("x"), Lit(1)), Lit(0))
    assert _optimized_predicates(table, (col("x") + 1) > 0) == [unchanged.to_ir()]


# --- the decomposition itself ------------------------------------------------


def test_decompose_rejects_equality():
    assert decompose(Binary("eq", col("x") + 1, Lit(5))) is None


def test_decompose_requires_a_bare_column():
    # The raw column is the point of the rewrite, and it is also what makes the range
    # knowable — arithmetic over two columns has neither.
    assert decompose(Binary("gt", (col("x") * 2) + 1, Lit(5))) is None


def test_decompose_rejects_a_boolean_constant():
    # `True` is an `int` subclass; treating it as the constant 1 would be a type error.
    assert decompose(Binary("gt", Binary("add", Col("x"), Lit(True)), Lit(5))) is None
