"""Pre-aggregating a LEFT join's measure side on a Parquet footer's key range, before any run.

`pre_aggregate_join_measures` used to need a *measured* reduction, which a first run over a
Parquet scan never has, and it met TPC-H Q13's join through a column-pruning projection it did
not look through. Now the measure side's key range, read from the footers, bounds the groups
(`gates._provably_reduces`), and the projection folds into the join (`_fold_bare_projection`).

The data carries what the LEFT `COUNT` merge can get wrong: customers with no orders (their
count must be 0, not NULL), orders whose customer does not exist, and NULL order keys.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential

_Q13 = (
    "SELECT c_count, count(*) AS custdist FROM ("
    "SELECT c_custkey, count(o_orderkey) AS c_count FROM customer "
    "LEFT OUTER JOIN orders ON c_custkey = o_custkey AND o_comment NOT LIKE '%special%requests%' "
    "GROUP BY c_custkey) AS c_orders GROUP BY c_count"
)


def _write(tmp_path, key_span: int) -> dict[str, str]:
    """Customers 1..2000 and 40,000 orders over a `key_span`-wide customer-key range."""
    rng = np.random.default_rng(13)
    n = 40_000
    custkey = rng.integers(1, key_span + 1, n)  # keys past 2000 match no customer
    paths = {"customer": str(tmp_path / "customer.parquet"), "orders": str(tmp_path / "orders")}
    pq.write_table(pa.table({"c_custkey": pa.array(range(1, 2001), pa.int64())}), paths["customer"])
    comments = np.where(rng.random(n) < 0.1, "a special deal requests", "plain")
    orders = pa.table(
        {
            "o_orderkey": pa.array(np.arange(n), mask=rng.random(n) < 0.05),
            "o_custkey": pa.array(custkey, pa.int64()),
            "o_comment": pa.array(comments),
        }
    )
    (tmp_path / "orders").mkdir()
    for i in range(4):  # several files, so the range is a merge of footers
        pq.write_table(orders.slice(i * n // 4, n // 4), str(tmp_path / "orders" / f"p{i}.parquet"))
    return paths


def _sessions(paths: dict[str, str]) -> tuple[bt.Session, duckdb.DuckDBPyConnection]:
    session, duck = bt.Session(), duckdb.connect()
    session.register("customer", bt.read.parquet(paths["customer"]))
    session.register("orders", bt.read.parquet(paths["orders"]))
    duck.execute(f"CREATE VIEW customer AS SELECT * FROM read_parquet('{paths['customer']}')")
    duck.execute(f"CREATE VIEW orders AS SELECT * FROM read_parquet('{paths['orders']}/*')")
    return session, duck


def _pushed(ds: bt.Dataset) -> bool:
    """Whether an aggregate grouped by `o_custkey` runs below the join."""
    lines = [ln.strip("│├└─ ⋯") for ln in ds.explain().splitlines()]
    join = next(i for i, ln in enumerate(lines) if ln.startswith("hash_join"))
    return any(ln.startswith("aggregate  [by o_custkey") for ln in lines[join + 1 :])


def test_q13_is_pre_aggregated_on_its_first_plan_and_matches_duckdb(tmp_path):
    session, duck = _sessions(_write(tmp_path, key_span=2_500))
    ds = session.sql(_Q13)
    # The positive control: nothing has run, so the footer range is the only evidence.
    assert _pushed(ds), ds.explain()
    assert_same(ds.collect(distributed=False), duck.sql(_Q13))


def test_a_key_range_too_wide_to_reduce_is_not_pushed(tmp_path):
    session, duck = _sessions(_write(tmp_path, key_span=10_000_000))
    ds = session.sql(_Q13)
    assert not _pushed(ds), ds.explain()
    assert_same(ds.collect(distributed=False), duck.sql(_Q13))
