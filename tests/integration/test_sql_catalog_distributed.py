"""SQL over session and catalog tables returns the same answer single-node and distributed.

`bt.sql` returns a lazy `Dataset`, so every query here is one plan that
`collect(distributed=False)` and `collect(distributed=True, num_workers=4)` both run. The
sources are the two a SQL user actually has on a cluster: a session table registered over a
multi-file Parquet read, and a table in a directory catalog (`bt.Catalog.from_directory`,
Delta underneath) written in four commits so it spans at least four files.

The queries cover what the SQL front-end lowers into something other than a scan: joins and
aggregates, windows, ``QUALIFY``, correlated subqueries (including the per-key ``LIMIT``,
``HAVING`` and empty-group shapes `subquery.shape` fixed), an uncorrelated scalar subquery and
``NOT IN``, a view, and a recursive CTE. A result with an ``ORDER BY`` is compared in order;
the rest as multisets.

**The positive control matters more than any single query.** A distributed collect that
quietly ran on one worker would pass every comparison here. So the file first shows the work
was split: batches of one read are tagged with the process that decoded them, and more than
one process shows up; and an unordered ``LIMIT`` over a ``GROUP BY`` — the one shape the
rules allow to differ between the two paths — keeps different groups.
"""

from __future__ import annotations

import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from _harness import assert_same, assert_same_ordered, has_outermost_order_by
from _ray_cluster import init_test_ray, shutdown_test_ray

pytestmark = pytest.mark.integration

pytest.importorskip("ray", reason="ray not installed")
pytest.importorskip("deltalake", reason="a directory catalog stores Delta tables")

_WORKERS = 4
_N = 8_000
_CUSTOMERS = 97


@pytest.fixture(scope="module", autouse=True)
def _ray_session():
    started = init_test_ray(_WORKERS)
    yield
    shutdown_test_ray(started)


@pytest.fixture(scope="module")
def warehouse(cluster_scratch) -> tuple[str, str]:
    """Four Parquet files of orders, and a four-commit Delta catalog table of customers.

    Both live on `cluster_scratch` because every worker has to read them: a directory
    catalog is paths, and a path on the driver's local disk is invisible to a worker on
    another node.
    """
    orders = pa.table(
        {
            "o_id": pa.array(range(_N), pa.int64()),
            "cust": pa.array([(i * 31) % _CUSTOMERS for i in range(_N)], pa.int64()),
            "region": pa.array([["n", "s", "e", "w"][i % 4] for i in range(_N)]),
            "amount": pa.array([(i * 37) % 1000 for i in range(_N)], pa.int64()),
            "day": pa.array([i % 30 for i in range(_N)], pa.int64()),
        }
    )
    orders_dir = cluster_scratch("sql_catalog_orders")
    for part in range(4):
        pq.write_table(orders.slice(part * _N // 4, _N // 4), orders_dir / f"p{part}.parquet")

    root = cluster_scratch("sql_catalog_wh")
    catalog = bt.Catalog.from_directory(str(root), name="wh")
    catalog.create_namespace("sales")
    session = bt.Session()
    session.catalog.attach(catalog)
    for part in range(4):
        ids = list(range(part, _CUSTOMERS, 4))
        chunk = bt.from_pydict(
            {"cust": ids, "tier": [["gold", "silver", "bronze"][i % 3] for i in ids]}
        )
        chunk.write.table("wh.sales.customers", mode="append", session=session)
    files = list((Path(root) / "sales" / "customers").glob("*.parquet"))
    assert len(files) >= 4, f"the catalog table should span several files, found {files}"
    return str(orders_dir), str(root)


def _session(warehouse) -> bt.Session:
    orders_dir, root = warehouse
    s = bt.Session()
    s.register("orders", bt.read.parquet(orders_dir))
    s.catalog.attach(bt.Catalog.from_directory(root, name="wh"))
    s.sql(
        "CREATE VIEW big_orders AS SELECT o_id, cust, amount FROM orders "
        "WHERE amount > (SELECT avg(amount) FROM orders)"
    )
    return s


_QUERIES = [
    "SELECT c.tier, count(*) AS n, sum(o.amount) AS s FROM orders o "
    "JOIN wh.sales.customers c ON o.cust = c.cust GROUP BY c.tier ORDER BY c.tier",
    "SELECT o_id, sum(amount) OVER (PARTITION BY region ORDER BY o_id "
    "ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS s FROM orders ORDER BY o_id",
    "SELECT region, o_id, amount FROM orders "
    "QUALIFY row_number() OVER (PARTITION BY region ORDER BY amount DESC, o_id) = 1 "
    "ORDER BY region",
    "SELECT cust, o_id FROM orders o "
    "WHERE amount = (SELECT max(amount) FROM orders o2 WHERE o2.cust = o.cust) ORDER BY cust, o_id",
    # Was a strict xfail (sorted runs returned out of order under distributed=True). It passes
    # since the breaker-free distributed scan keeps source order (`dist/executor.py`,
    # `preserve_order=True`); reverting that one argument brings the defect back.
    "SELECT cust FROM wh.sales.customers c "
    "WHERE EXISTS (SELECT 1 FROM orders o WHERE o.cust = c.cust HAVING count(*) > 82) "
    "ORDER BY cust",
    "SELECT cust FROM wh.sales.customers c "
    "WHERE EXISTS (SELECT 1 FROM orders o WHERE o.cust = c.cust HAVING count(*) > 82)",
    "SELECT cust, (SELECT o_id FROM orders o WHERE o.cust = c.cust "
    "ORDER BY amount DESC, o_id LIMIT 1) AS top FROM wh.sales.customers c ORDER BY cust",
    "SELECT cust, (SELECT count(*) + 1 FROM orders o WHERE o.cust = c.cust AND o.amount > 990) "
    "AS n FROM wh.sales.customers c ORDER BY cust",
    "SELECT count(*) AS n FROM orders WHERE amount > (SELECT avg(amount) FROM orders)",
    "SELECT cust FROM wh.sales.customers "
    "WHERE cust NOT IN (SELECT cust FROM orders WHERE region = 'n' AND o_id < 200)",
    "SELECT tier, count(*) AS n FROM big_orders b JOIN wh.sales.customers c USING (cust) "
    "GROUP BY tier",
    "WITH RECURSIVE r(n) AS (SELECT min(day) FROM orders UNION ALL SELECT n + 1 FROM r "
    "WHERE n < 5) SELECT r.n, count(*) AS c FROM r JOIN orders ON orders.day = r.n "
    "GROUP BY r.n ORDER BY r.n",
]


@pytest.mark.parametrize("query", _QUERIES)
def test_sql_result_is_the_same_single_node_and_distributed(warehouse, query):
    ds = _session(warehouse).sql(query)
    single = ds.collect(distributed=False)
    distributed = ds.collect(distributed=True, num_workers=_WORKERS)
    assert single.num_rows > 0, "a query that returns nothing compares nothing"
    assert distributed.schema == single.schema
    expected = _Relation(single)
    if has_outermost_order_by(query):
        assert_same_ordered(distributed, expected)
    else:
        assert_same(distributed, expected)


class _Relation:
    """The single-node table, in the relation shape the comparison helpers read."""

    def __init__(self, table: pa.Table) -> None:
        self._table = table

    def to_arrow_table(self) -> pa.Table:
        return self._table


def test_the_distributed_run_used_more_than_one_worker(warehouse):
    """Positive control: without it, every comparison above is consistent with one worker."""
    orders_dir, _ = warehouse

    # Defined here, not at module level, so Ray pickles it by value: a worker cannot import
    # this test module by name.
    def _tag_process(batch: pa.RecordBatch) -> pa.RecordBatch:
        pid = pa.array([os.getpid()] * batch.num_rows, pa.int64())
        return pa.RecordBatch.from_arrays([*batch.columns, pid], [*batch.schema.names, "pid"])

    s = bt.Session()
    columns = ["o_id", "cust", "region", "amount", "day", "pid"]
    s.register(
        "tagged", bt.read.parquet(orders_dir).map_batches(_tag_process, output_columns=columns)
    )
    pids = s.sql("SELECT DISTINCT pid FROM tagged").collect(distributed=True, num_workers=_WORKERS)
    assert pids.num_rows > 1, f"every batch ran in one process: {pids.to_pydict()}"


def test_an_unordered_limit_is_where_the_paths_may_differ(warehouse):
    """The rules' permitted divergence shows up here, and only as a selection of real rows."""
    s = _session(warehouse)
    whole = s.sql("SELECT cust, count(*) AS n FROM orders GROUP BY cust").collect()
    query = "SELECT cust, count(*) AS n FROM orders GROUP BY cust LIMIT 3"
    single = s.sql(query).collect(distributed=False)
    distributed = s.sql(query).collect(distributed=True, num_workers=_WORKERS)
    rows = set(zip(*whole.to_pydict().values(), strict=True))
    assert single.num_rows == distributed.num_rows == 3
    assert set(zip(*distributed.to_pydict().values(), strict=True)) <= rows
    assert single.to_pydict() != distributed.to_pydict(), (
        "the same three groups on both paths: the distributed run may not have split the work"
    )
