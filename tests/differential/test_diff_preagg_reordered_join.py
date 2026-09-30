"""Differential tests vs DuckDB for pre-aggregating a fact join below a unique dimension.

`pre_aggregation_through_reordered_join` groups `facts ⋈ dates` by the dimension's join key
before the dimension is joined, which is only correct when the dimension is *unique* on that
key: a duplicated key would count each pre-aggregated row once per copy. Uniqueness is
proven by the conductor counting an in-memory key exactly (`with_exact_join_keys`), so these
cases run the whole path, SQL comma-join included, and compare against DuckDB.

The fixtures carry what breaks a naive push: fact rows whose key is NULL or matches no
dimension row, a dimension with a duplicated key (the rewrite must stand down), an
expression aggregate input, `count(*)`, and an empty fact table.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential

_QUERY = """
SELECT c_id, c_name, c_email, d_year,
       sum((f_price - f_cost) / 2) AS margin,
       count(*) AS n
FROM customer, facts, dates
WHERE c_sk = f_customer_sk AND f_date_sk = d_date_sk AND d_year BETWEEN 2001 AND 2003
GROUP BY c_id, c_name, c_email, d_year
"""


def _tables(*, facts: int = 60_000, duplicate_key: bool = False) -> dict[str, pa.Table]:
    rng = np.random.default_rng(7)
    n_cust = 1_500
    sk = np.arange(n_cust)
    if duplicate_key:
        sk[-1] = 0  # key 0 now names two customers
    customer = pa.table(
        {
            "c_sk": sk,
            "c_id": [f"C{i:06d}" for i in range(n_cust)],
            "c_name": [f"name-{i % 97}" for i in range(n_cust)],
            "c_email": [f"user{i}@example.com" for i in range(n_cust)],
        }
    )
    dates = pa.table({"d_date_sk": np.arange(3_650), "d_year": 1998 + np.arange(3_650) // 365})
    cust = rng.integers(0, n_cust + 50, facts)  # some keys match no customer
    customer_sk = pa.array(cust, mask=rng.random(facts) < 0.02)  # and some are NULL
    facts_t = pa.table(
        {
            "f_customer_sk": customer_sk,
            "f_date_sk": rng.integers(0, 3_650, facts),
            "f_price": rng.random(facts) * 100,
            "f_cost": rng.random(facts) * 50,
        }
    )
    return {"customer": customer, "facts": facts_t, "dates": dates}


def _run(duck, tables: dict[str, pa.Table]) -> tuple[pa.Table, str]:
    session = bt.Session()
    for name, table in tables.items():
        session.register(name, table)
        duck.register(name, table)
    session.sql(_QUERY).collect()
    session.sql(_QUERY).collect()  # a second run plans with what the first one measured
    return session.sql(_QUERY).collect(), session.sql(_QUERY).explain()


def _pushed(explained: str) -> bool:
    """Whether an aggregate sits below the join: the rewrite's visible signature."""
    lines = explained.splitlines()
    join = next(i for i, line in enumerate(lines) if "hash_join" in line and "customer_sk" in line)
    return any("aggregate" in line for line in lines[join + 1 :])


def test_unique_dimension_pushes_and_matches_duckdb(duck):
    result, explained = _run(duck, _tables())
    assert _pushed(explained), explained  # positive control: the rewrite really fired
    assert_same(result, duck.sql(_QUERY))


def test_duplicated_dimension_key_stands_down_and_matches_duckdb(duck):
    result, explained = _run(duck, _tables(duplicate_key=True))
    assert not _pushed(explained), explained
    assert_same(result, duck.sql(_QUERY))


def test_empty_fact_table_matches_duckdb(duck):
    result, _ = _run(duck, _tables(facts=0))
    assert result.num_rows == 0
    assert_same(result, duck.sql(_QUERY))
