"""A broadcast is judged filtered only by a predicate that can drop rows of its source.

Kyber pushes the other join side's key range onto a scan. At SF1000 TPC-H q18's `customer`
carried `c_custkey BETWEEN 1 AND 149999999` -- every row -- was judged filtered, and 4.2 GiB
of it was broadcast before the bound declined the plan at run time.
"""

from __future__ import annotations

import types

import pyarrow as pa
import pytest

from batcher import col
from batcher.dist.executors.aligned.route import _filtered
from batcher.plan.logical import Filter, Join, Scan
from batcher.plan.logical.join import JoinOutputCol
from batcher.plan.schema import SchemaRef
from batcher.plan.stats import ColumnStat

pytestmark = pytest.mark.unit

_SCHEMA = SchemaRef.from_arrow(pa.schema([("k", pa.int64())]))


def _body(predicate):
    broadcast = Filter(Scan(1, _SCHEMA), predicate._ir if hasattr(predicate, "_ir") else predicate)
    keep = (JoinOutputCol("left", "k", "k"),)
    return Join(Scan(0, _SCHEMA), broadcast, ("k",), ("k",), "semi", keep)


def _source(lo, hi):
    stats = types.SimpleNamespace(columns={"k": ColumnStat(min=lo, max=hi)})
    return types.SimpleNamespace(statistics=lambda: stats)


@pytest.mark.parametrize(
    ("predicate", "filtered"),
    [
        ((col("k") >= 1) & (col("k") <= 100), False),  # the source's whole range
        ((col("k") > 0) & (col("k") < 101), False),
        ((col("k") >= 1) & (col("k") <= 50), True),  # half of it
        (col("k") > 1, True),
        (col("k") == 7, True),  # not a range: judged conservatively
    ],
)
def test_a_range_the_footers_imply_does_not_count_as_a_filter(predicate, filtered):
    body = _body(predicate)
    assert _filtered(body, 1, frozenset({0}), _source(1, 100)) is filtered


def test_without_bounds_any_range_still_counts():
    body = _body((col("k") >= 1) & (col("k") <= 100))
    assert _filtered(body, 1, frozenset({0})) is True


def test_an_overrunning_broadcast_is_left_to_the_residual_on_a_second_plan(monkeypatch):
    """A broadcast that outgrows its bound declines one plan, not the aligned executor."""
    from batcher.dist.executors.aligned import route

    asked = []

    def choose(plan, sources, *, strict=True, exclude=frozenset(), **_):
        asked.append(exclude)
        return object()

    def run(found, sources, workers, hub, metrics_out, oversized):
        if not asked[-1]:
            oversized.add(0)  # the first plan broadcast source 0 and it overran
            return None
        return "result"

    monkeypatch.setattr(route, "choose_plan", choose)
    monkeypatch.setattr(route, "_run_found", run)
    monkeypatch.setattr(route, "_OVERRAN", {})
    monkeypatch.setattr(route, "_overran_key", lambda plan, sources: ("plan",))
    assert route.try_aligned(None, [], 8) == "result"
    assert asked == [frozenset(), frozenset({0})]
    # The next run of the same plan starts from the plan that ran.
    assert route.try_aligned(None, [], 8) == "result"
    assert asked[2:] == [frozenset({0})]


def test_a_decline_for_any_other_reason_is_not_retried(monkeypatch):
    from batcher.dist.executors.aligned import route

    asked = []
    monkeypatch.setattr(
        route, "choose_plan", lambda *a, exclude=frozenset(), **k: asked.append(exclude) or 1
    )
    monkeypatch.setattr(route, "_run_found", lambda *a: None)
    monkeypatch.setattr(route, "_overran_key", lambda plan, sources: None)
    assert route.try_aligned(None, [], 8) is None
    assert asked == [frozenset()]
