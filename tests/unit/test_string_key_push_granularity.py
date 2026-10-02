"""The string-key pre-aggregation push fires only when the outer grouping is fine enough to pay.

`agg_pushdown.gates._moves_string_keys` lets an aggregate move below a join on a 2x reduction
when the outer group keys are strings from the unique side, because the push then also saves
gathering and row-encoding those strings for every fact row. That saving exists only when the
outer aggregate groups about as finely as the pushed one: TPC-DS q4's `year_total` (customer by
year) gains 204 -> 126 ms, while `lineitem JOIN orders GROUP BY o_orderpriority`, 5 groups, lost
26 -> 95 ms to a 1.5M-group aggregate it did not need. The fine grouping is the positive control
that shows the push is reachable on this fixture at all.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.unit

_ORDERS = 20_000


def _session() -> bt.Session:
    s = bt.Session()
    s.register(
        "orders",
        pa.table(
            {
                "o_k": list(range(_ORDERS)),
                "o_prio": [f"P{i % 5}" for i in range(_ORDERS)],
                "o_name": [f"order-{i:06d}" for i in range(_ORDERS)],
            }
        ),
    )
    n = _ORDERS * 4
    s.register("lineitem", pa.table({"l_k": [i % _ORDERS for i in range(n)], "v": list(range(n))}))
    return s


def _pushed(sql: str) -> bool:
    # One run first, so the group keys' distinct counts are measured: the gate reads the
    # outer aggregate's estimated group count, which a cold in-memory string column cannot
    # supply beyond the default guess.
    session = _session()
    session.sql(sql).collect()
    plan = session.sql(sql).explain()
    return any("aggregate" in ln and "by l_k" in ln for ln in plan.splitlines())


def test_a_fine_string_grouping_still_takes_the_push():
    assert _pushed(
        "SELECT o_name, sum(v) AS s FROM lineitem JOIN orders ON l_k = o_k GROUP BY o_name"
    )


def test_a_coarse_string_grouping_does_not():
    assert not _pushed(
        "SELECT o_prio, sum(v) AS s FROM lineitem JOIN orders ON l_k = o_k GROUP BY o_prio"
    )
