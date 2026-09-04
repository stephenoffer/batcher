"""`infer_join_predicates` must not fight the zone-map rule forever, either.

The sibling file `test_kyber_constant_key_fixpoint.py` closed this cycle for
`infer_join_predicate_from_constant_key`. The *general* inference rule beside it —
`infer_join_predicates`, which mirrors any `key OP literal` constraint across an inner
join's equi-keys — has the same conflict with `drop_filter_conjunct_implied_by_zonemap`,
and it was the larger half: **29 of the 99 TPC-DS queries** ran the whole
`fixpoint_iterations` budget without converging (q64 and q72 reached 63 and 61 iterations),
logging "phase did not reach a fixpoint" each time.

Answers were never affected — every rule is semantics-preserving — so the symptoms were the
same three as before: the warning, a plan that depends on `OptimizerConfig`, and twenty-odd
whole-plan passes spent on rewrites that cancel out. Closing it takes cold planning of all
99 TPC-DS queries from ~33s to ~27s; the queries that cycled gain 20-38% (q80 2.4s -> 1.5s,
q64 3.5s -> 2.8s), and one that never did pays ~4% for the guard's subtree walk.

The fix that did *not* work is worth recording, because it is the obvious one: testing the
target relation's own statistics. An inferred conjunct does not stay where it is put —
pushdown sinks it toward the scan, and the zone-map rule deletes it at whatever depth the
bounds decide it — so the level it is added at and the level it dies at are different
relations. `zonemap_pruning.implied_by_bounds` therefore follows the column *down*,
re-phrasing it through renames and inner-join outputs, asking the deleting rule's own
oracle at each step.

Both shapes below cycle at the previous behaviour and converge now; they were found by
searching plan shapes for a reproduction, since the TPC-DS queries that exposed it need a
DuckDB extension the unit suite cannot depend on.
"""

from __future__ import annotations

import logging

import pyarrow as pa
import pytest

import batcher as bt
from batcher.plan.expr_ir import col

pytestmark = pytest.mark.unit

#: Substring of the driver's non-convergence diagnostic.
_WARNING = "did not reach a fixpoint"


def _fact() -> bt.Dataset:
    """A "fact" relation whose join key holds one distinct value, so bounds decide it."""
    return bt.from_arrow(
        pa.table(
            {
                "w": pa.array([1] * 64, pa.int64()),
                "d": pa.array(list(range(64)), pa.int64()),
                "q": pa.array([2] * 64, pa.int64()),
            }
        )
    )


def _dim() -> bt.Dataset:
    return bt.from_arrow(
        pa.table(
            {
                "w": pa.array([1] * 8, pa.int64()),
                "name": pa.array([f"n{i}" for i in range(8)], pa.string()),
            }
        )
    )


def _filter_above_a_rename() -> bt.Dataset:
    """A projection renames the key and a filter sits above it.

    The rename is what separates the two levels: the constraint is undecidable against the
    projected relation's statistics and provable one step down, which is exactly the gap
    that made a target-level-only guard insufficient.
    """
    return (
        _fact()
        .select(wh=col("w"), qq=col("q"))
        .filter(col("qq") > 0)
        .join(_dim().select("w", "name"), left_on="wh", right_on="w")
    )


def _three_way_star() -> bt.Dataset:
    """The fact joined to two dimensions on the same key — inference has two ways round."""
    return (
        _fact()
        .join(_dim().select("w", "name"), on="w")
        .join(_dim().select(w2=col("w"), nm=col("name")), left_on="w", right_on="w2")
    )


SHAPES = {"filter_above_a_rename": _filter_above_a_rename, "three_way_star": _three_way_star}


def _fixpoint_warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if _WARNING in r.getMessage()]


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_the_pushdown_phase_reaches_a_fixpoint(caplog, shape):
    with caplog.at_level(logging.WARNING, logger="batcher.kyber"):
        SHAPES[shape]().explain()
    assert _fixpoint_warnings(caplog) == []


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_the_previous_behaviour_did_cycle(caplog, monkeypatch, shape):
    """The reproduction is real: neutralize the guard and the same plan stops converging.

    Without this, a passing test above proves only that *these* plans converge today, which
    a shape that never cycled would also satisfy.
    """
    import batcher.kyber.rules.pushdown as pushdown

    monkeypatch.setattr(pushdown, "implied_by_bounds", lambda *a, **k: False)
    with caplog.at_level(logging.WARNING, logger="batcher.kyber"):
        SHAPES[shape]().explain()
    assert _fixpoint_warnings(caplog), f"{shape} was expected to cycle without the guard"


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_the_plan_does_not_depend_on_the_iteration_cap(shape):
    """The user-visible symptom of a non-confluent phase."""
    plans = set()
    for iterations in (1, 2, 3, 8, 20):
        cfg = bt.Config().replace(optimizer=bt.OptimizerConfig(fixpoint_iterations=iterations))
        with bt.config_context(cfg):
            plans.add(str(SHAPES[shape]().explain()))
    assert len(plans) == 1, f"plan varies with the iteration cap: {len(plans)} distinct plans"


def test_the_rows_are_unchanged():
    """The cycle never changed an answer, so closing it must not either."""
    got = _filter_above_a_rename().collect()
    # 64 fact rows x 8 dimension rows, all on w = 1.
    assert got.num_rows == 64 * 8
    assert set(got.column("qq").to_pylist()) == {2}

    star = _three_way_star().collect()
    assert star.num_rows == 64 * 8 * 8


def test_the_inference_still_fires_where_it_prunes(caplog):
    """The guard must decline only where the predicate prunes nothing.

    A dimension genuinely restricted to a subset of the fact's keys still has its
    constraint mirrored onto the fact side — the whole point of the rule — because the
    fact's bounds do not imply it.
    """
    fact = bt.from_arrow(
        pa.table(
            {
                "w": pa.array(list(range(64)), pa.int64()),
                "q": pa.array([2] * 64, pa.int64()),
            }
        )
    )
    dim = bt.from_arrow(pa.table({"w": pa.array([3], pa.int64()), "name": pa.array(["n"])}))
    plan = fact.join(dim.filter(col("w") == 3), on="w").explain()
    assert "w" in str(plan)
    joined = fact.join(dim.filter(col("w") == 3), on="w").collect()
    assert joined.num_rows == 1
