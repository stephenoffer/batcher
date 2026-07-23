# From SQL to DataFrames

You know SQL. This tutorial takes one query and rewrites it as a DataFrame chain, then
proves the two are the *same query*: same plan, same optimizer, same Rust engine. After
that, mixing them is not a compromise. It is choosing a spelling.

Everything here runs as written.

:::{note}
**What you'll build.** One aggregate query, written twice, with `explain()` used as the
proof that both spellings produce the identical optimized plan. Then a `Session` with a
registered catalog, a view, and a Python function callable from SQL. You need `pip install
batcher-engine` and nothing else: no cluster, no files, no GPU.
:::

## 1. The data

```python
import batcher as bt

orders = bt.from_pydict(
    {
        "order_id": [1, 2, 3, 4, 5, 6],
        "customer": ["ann", "bo", "ann", "cy", "bo", "ann"],
        "region": ["us", "eu", "us", "eu", "eu", "us"],
        "amount": [120.0, 40.0, 80.0, 15.0, 60.0, 25.0],
    }
)
print(orders.columns)
# ['order_id', 'customer', 'region', 'amount']
```

## 2. Write it in SQL

`bt.sql(query, **tables)` binds each table named in the `FROM` clause to a Dataset passed as
a keyword argument. The keyword is the table name.

```python
revenue = bt.sql(
    """
    SELECT region, SUM(amount) AS revenue, COUNT(*) AS orders
    FROM o
    WHERE amount >= 25
    GROUP BY region
    HAVING SUM(amount) > 100
    ORDER BY revenue DESC
    """,
    o=orders,
)
print(revenue.to_pydict())
# {'region': ['us'], 'revenue': [225.0], 'orders': [3]}
```

Nothing has executed until `to_pydict()`. `bt.sql` returns a lazy `Dataset`, exactly like
every other operation.

## 3. Write it as a DataFrame

The clauses map one to one, in the order SQL *evaluates* them rather than the order it
writes them:

| SQL | DataFrame |
|---|---|
| `WHERE` | `.filter(...)` |
| `GROUP BY` | `.group_by(...)` |
| `SUM(x) AS y` | `.agg(y=bt.col("x").sum())` |
| `HAVING` | `.filter(...)` after `.agg` |
| `ORDER BY x DESC` | `.sort("x", descending=True)` |
| `SELECT` list | `.select(...)` (or the `agg` output itself) |

```python
same = (
    orders.filter(bt.col("amount") >= 25)
    .group_by("region")
    .agg(revenue=bt.col("amount").sum(), orders=bt.count())
    .filter(bt.col("revenue") > 100)
    .sort("revenue", descending=True)
)
print(same.to_pydict())
# {'region': ['us'], 'revenue': [225.0], 'orders': [3]}
```

`HAVING` is not a special operator. It is a filter that runs after the aggregate, and that
is exactly how you write it.

## 4. Prove they are the same query

Both spellings build one `LogicalPlan`, push it through one optimizer, and run on one Rust
data plane. `explain()` renders the optimized plan without executing it, so comparing the
two renderings settles the question:

```python
print(revenue.explain() == same.explain())
# True
```

There is no separate SQL engine to fall behind the DataFrame one. Pick whichever reads
better for the query in front of you.

## 5. Cross the boundary in either direction

A SQL result is an ordinary Dataset, and a Dataset can be queried with SQL. Neither
direction is a conversion, because there is nothing to convert: both are the same
`LogicalPlan`. This is the thing that is awkward in a SQL-only tool.

::::{tab-set}
:::{tab-item} SQL, then DataFrame
```python
customers = bt.from_pydict(
    {"customer": ["ann", "bo", "cy"], "tier": ["gold", "silver", "silver"]}
)

by_tier = bt.sql(
    "SELECT c.tier, SUM(o.amount) AS revenue "
    "FROM o JOIN c ON o.customer = c.customer "
    "GROUP BY c.tier ORDER BY revenue DESC",
    o=orders,
    c=customers,
)
ranked = by_tier.with_row_index("rank", offset=1)
print(ranked.to_pydict())
# {'rank': [1, 2], 'tier': ['gold', 'silver'], 'revenue': [225.0, 115.0]}
```

The join and the rollup are SQL, because SQL says them well. The row numbering is a
DataFrame method, because SQL would need a window function for it.
:::

:::{tab-item} DataFrame, then SQL
```python
print(
    orders.filter(bt.col("amount") >= 25)
    .sql("SELECT region, COUNT(*) AS n FROM self GROUP BY region ORDER BY region")
    .to_pydict()
)
# {'region': ['eu', 'us'], 'n': [2, 3]}
```

`ds.sql(...)` queries the current dataset, which `self` names. The filter is a DataFrame
call, the rollup is SQL, and the plan does not know the difference.
:::
::::

## 6. A session, for a catalog you keep

`bt.Session` is the `DuckDBPyConnection` / `SparkSession` analogue: a dialect plus a catalog
of tables and Python functions. Register once, query by name.

```python
s = bt.Session()
s.register("orders", orders)

print(s.sql("SELECT COUNT(*) AS n FROM orders").to_pydict())
# {'n': [6]}

s.sql("CREATE VIEW big AS SELECT * FROM orders WHERE amount >= 60")
print(s.sql("SELECT order_id, amount FROM big ORDER BY amount").to_pydict())
# {'order_id': [5, 3, 1], 'amount': [60.0, 80.0, 120.0]}
```

`CREATE VIEW` registers a lazy table. Nothing is materialized until a terminal op.

## 7. Call Python from SQL

A registered function is vectorized: it receives an Arrow array and returns one. It lowers
to the same `map_batches` path the DataFrame API uses, so SQL and Python share one plan
rather than one calling into the other.

```python
import pyarrow.compute as pc

s.register_function("net", lambda a: pc.multiply(a, 0.85))
print(s.sql("SELECT order_id, net(amount) AS net FROM big ORDER BY order_id").to_pydict())
# {'order_id': [1, 3, 5], 'net': [102.0, 68.0, 51.0]}
```

:::{warning}
Per-*row* Python is the thing to avoid, not Python itself. A vectorized function sees the
whole array at once and returns an array. Write one that takes a scalar and you have put a
Python interpreter in the inner loop of a Rust engine, and you will feel it. The same rule
governs `map_batches`: whole batches, never rows.
:::

## 8. Point it at real files

Only the source changes. Every transform and terminal below it is identical, whether the
table came from a dict, a Parquet directory, or a Delta table. This block needs a real
bucket, so it is shown but not run.

```python
# docs: skip
import batcher as bt

lake = bt.read.parquet("s3://bucket/orders/")
bt.sql(
    "SELECT region, SUM(amount) AS revenue FROM o WHERE amount >= 25 GROUP BY region",
    o=lake,
).write.parquet("s3://bucket/revenue_by_region/")
```

## What you learned

- One plan, one optimizer, one engine. SQL and DataFrames are two front ends onto the same
  thing, and `explain()` will show you that.
- A SQL result is a Dataset; a Dataset can be queried with SQL. Chain them freely.
- Expressions are the column language both spellings lower to.

::::{grid} 1 3 3 3
:gutter: 3

:::{grid-item-card} {octicon}`meter;1.1em` Optimizing a slow query
:link: optimizing-a-slow-query
:link-type: doc
Read the plan you just printed, and act on it.
:::

:::{grid-item-card} {octicon}`database;1.1em` Building a lakehouse
:link: building-a-lakehouse
:link-type: doc
Point these queries at a real transactional table.
:::

:::{grid-item-card} {octicon}`broadcast;1.1em` A streaming pipeline
:link: streaming-pipeline
:link-type: doc
The same operators, over a source that never ends.
:::
::::

## See also

- [SQL guide](../user-guide/sql.md): the supported SQL surface in full, including what is
  not supported.
- [Expressions](../user-guide/expressions.md): the column language underneath both spellings.
- [Explain plans](../user-guide/explain-plans.md): how to read the thing you just compared.
- [Plan IR](../deep-dives/plan-ir.md): the single `LogicalPlan` both front ends build.
- [SQL API reference](../api/sql.md): `Session`, `register`, `register_function`.
- [Migration guide](../migration/index.md): if the SQL you know is Spark's or DuckDB's.
