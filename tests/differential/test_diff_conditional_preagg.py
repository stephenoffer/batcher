"""Differential tests vs DuckDB for pre-aggregating a dimension-conditional aggregate.

`pre_aggregation_through_join` may push `SUM(CASE WHEN <dimension> THEN <fact> END)` below an
inner join onto a dimension unique on the key (`kyber.rules.agg_pushdown.conditional`). The
merge re-applies the condition to a per-key partial, so these cases carry what could make that
visible: NULL measures, keys the dimension lacks, NULL keys, a group whose keys all fail the
condition, `MIN`/`MAX`, `ELSE NULL` beside no `ELSE`, and a `COUNT`, which must not be pushed and
must still be right.

The push is asserted to happen on the weekday pivot, which is the positive control that keeps
the comparisons from passing on a plan that never rewrote anything.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential

_DAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]


def _tables() -> dict[str, pa.Table]:
    rng = np.random.default_rng(23)
    n = 40_000
    date_sk = rng.integers(0, 400, n)  # 0..399; the dimension holds 0..349, so some miss
    price = rng.random(n) * 100
    return {
        "sales": pa.table(
            {
                "sold_date_sk": pa.array(date_sk, mask=rng.random(n) < 0.02),
                "price": pa.array(price, mask=rng.random(n) < 0.05),
                "qty": pa.array(rng.integers(1, 9, n)),
            }
        ),
        "dates": pa.table(
            {
                "d_date_sk": list(range(350)),
                "d_week": [k // 7 for k in range(350)],
                "d_day": [_DAYS[k % 7] for k in range(350)],
                # Only weeks below 10 carry a holiday flag at all: a later week's every key fails
                # the condition, the group whose merged sum must stay NULL.
                "d_holiday": [k < 70 and k % 11 == 0 for k in range(350)],
            }
        ),
    }


_PIVOT = (
    "SELECT d_week, "
    + ", ".join(
        f"sum(CASE WHEN d_day = '{d}' THEN price ELSE NULL END) AS s_{d[:3].lower()}" for d in _DAYS
    )
    + " FROM sales, dates WHERE d_date_sk = sold_date_sk GROUP BY d_week"
)

_QUERIES = {
    "weekday_pivot": _PIVOT,
    "no_else": (
        "SELECT d_week, sum(CASE WHEN d_day = 'Friday' THEN price END) AS fri "
        "FROM sales JOIN dates ON d_date_sk = sold_date_sk GROUP BY d_week"
    ),
    "all_false_groups": (
        "SELECT d_week, sum(CASE WHEN d_holiday THEN price END) AS hol "
        "FROM sales JOIN dates ON d_date_sk = sold_date_sk GROUP BY d_week"
    ),
    "min_max": (
        "SELECT d_week, min(CASE WHEN d_day = 'Monday' THEN price END) AS lo, "
        "max(CASE WHEN d_day = 'Monday' THEN price END) AS hi "
        "FROM sales JOIN dates ON d_date_sk = sold_date_sk GROUP BY d_week"
    ),
    "integer_measure": (
        "SELECT d_week, sum(CASE WHEN d_day = 'Sunday' THEN qty END) AS sun_qty "
        "FROM sales JOIN dates ON d_date_sk = sold_date_sk GROUP BY d_week"
    ),
    # Not decomposable this way (an all-false group counts 0, not NULL): must stay correct.
    "count_is_left_alone": (
        "SELECT d_week, count(CASE WHEN d_holiday THEN price END) AS n "
        "FROM sales JOIN dates ON d_date_sk = sold_date_sk GROUP BY d_week"
    ),
    "mixed_with_plain_sum": (
        "SELECT d_week, sum(price) AS total, sum(CASE WHEN d_day = 'Sunday' THEN price END) AS sun "
        "FROM sales JOIN dates ON d_date_sk = sold_date_sk GROUP BY d_week"
    ),
}


def _session(tables: dict[str, pa.Table]) -> tuple[bt.Session, duckdb.DuckDBPyConnection]:
    session, duck = bt.Session(), duckdb.connect()
    for name, table in tables.items():
        session.register(name, table)
        duck.register(name, table)
    return session, duck


@pytest.mark.parametrize("name", sorted(_QUERIES))
def test_a_dimension_conditional_aggregate_matches_duckdb(name):
    session, duck = _session(_tables())
    query = _QUERIES[name]
    for _ in range(2):  # the second run plans with the column statistics the first measured
        assert_same(session.sql(query).collect(distributed=False), duck.sql(query))


def test_the_weekday_pivot_is_pre_aggregated_below_the_join():
    session, _ = _session(_tables())
    ds = session.sql(_PIVOT)
    for _ in range(2):  # the push is gated on the key's distinct count, which a run measures
        ds.collect(distributed=False)
    lines = [ln.strip("│├└─ ") for ln in ds.explain().splitlines()]
    join = next(i for i, ln in enumerate(lines) if ln.startswith("hash_join"))
    # One shared partial for the seven sums over the one measure (`_shared_partials`), keyed
    # by the join key, below the join -- the plan the query runs, as `explain` renders it.
    assert any(ln.startswith("aggregate  [by sold_date_sk · sum]") for ln in lines[join + 1 :]), (
        ds.explain()
    )
