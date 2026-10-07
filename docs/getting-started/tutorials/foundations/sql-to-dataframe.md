# From SQL to DataFrames

You know SQL. This tutorial rewrites one query as a DataFrame chain and proves the two are the *same query*: same plan, same optimizer, same engine. Then it builds a {py:class}`Session <batcher.Session>` with a catalog, a view, and a Python function callable from SQL. All you need is `pip install batcher-engine`.

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

{py:func}`bt.sql(query, **tables) <batcher.sql>` binds each table named in the `FROM` clause to a Dataset passed as
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

{py:obj}`bt.sql <batcher.sql>` returns a lazy {py:class}`Dataset <batcher.Dataset>`, so nothing runs until {py:meth}`to_pydict() <batcher.Dataset.to_pydict>`. CTEs and window functions work as you'd expect:

```python
print(bt.sql(
    "WITH big AS (SELECT * FROM o WHERE amount >= 40) "
    "SELECT region, COUNT(*) AS n FROM big GROUP BY region ORDER BY region",
    o=orders,
).to_pydict())
# {'region': ['eu', 'us'], 'n': [2, 2]}
```

```python
print(bt.sql(
    "SELECT order_id, RANK() OVER (PARTITION BY region ORDER BY amount DESC) AS rk "
    "FROM o WHERE region = 'us' ORDER BY rk",
    o=orders,
).to_pydict())
# {'order_id': [1, 3, 6], 'rk': [1, 2, 3]}
```

## 3. Write it as a DataFrame

The clauses map one to one, in the order SQL *evaluates* them rather than the order it
writes them:

| SQL | DataFrame |
|---|---|
| `WHERE` | `.filter(...)` |
| `GROUP BY` | {py:meth}`.group_by(...) <batcher.Dataset.group_by>` |
| `SUM(x) AS y` | `.agg(y=bt.col("x").sum())` |
| `HAVING` | `.filter(...)` after `.agg` |
| `ORDER BY x DESC` | `.sort("x", descending=True)` |
| `SELECT` list | `.select(...)`, or the `agg` output itself |

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

`HAVING` is just a `filter` after the aggregate. The window from step 2 is an expression with `.over(...)`:

```python
us = orders.filter(bt.col("region") == "us")
print(us.select("order_id", rk=bt.col("amount").rank(descending=True).over("region")).sort("rk").to_pydict())
# {'order_id': [1, 3, 6], 'rk': [1, 2, 3]}
```

## 4. Prove they are the same query

`explain()` renders the optimized plan without executing it, so comparing the two settles the question:

```python
print(revenue.explain() == same.explain())
# True
```

The two spellings differ only until the plan exists. SQL is parsed with sqlglot and translated, and each DataFrame call adds a node. From the `LogicalPlan` on, everything is shared:

![Two inputs, a SQL string passed to bt.sql, ds.sql, or Session.sql and a DataFrame chain such as filter, group_by, and agg, both build one LogicalPlan. The SQL string gets there through a sqlglot parse, and the DataFrame chain adds one node per call. The LogicalPlan goes through the Kyber optimizer, and the optimized plan travels as JSON IR to the Rust engine. explain() renders that optimized plan, which is why it is identical for both spellings. Nothing runs until a terminal such as to_pydict().](/_static/diagrams/sql_dataframe_one_plan.svg)

There is no separate SQL engine. Pick whichever reads better.

## 5. Cross the boundary in either direction

A SQL result is an ordinary Dataset, and a Dataset can be queried with SQL. Neither direction is a conversion.

::::{tab-set}
:::{tab-item} SQL, then DataFrame
```python
customers = bt.from_pydict({"customer": ["ann", "bo", "cy"], "tier": ["gold", "silver", "silver"]})

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

The join and rollup are SQL; the row numbering is a DataFrame method.
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

{py:meth}`ds.sql(...) <batcher.Dataset.sql>` queries the current dataset as `self`.
:::
::::

## 6. A session, for a catalog you keep

{py:class}`bt.Session <batcher.Session>` plays the role of a DuckDB connection or a `SparkSession`: a dialect plus a catalog of tables and Python functions. Register once, then query by name.

```python
s = bt.Session()
s.register("orders", orders)

print(s.sql("SELECT COUNT(*) AS n FROM orders").to_pydict())
# {'n': [6]}

s.sql("CREATE VIEW big AS SELECT * FROM orders WHERE amount >= 60")
print(s.sql("SELECT order_id, amount FROM big ORDER BY amount").to_pydict())
# {'order_id': [5, 3, 1], 'amount': [60.0, 80.0, 120.0]}
```

`CREATE VIEW` registers a lazy table. The session lists everything it holds:

```python
print(sorted(s.list()))
# ['big', 'orders']
```

## 7. Call Python from SQL

A registered function is vectorized: it receives an Arrow array and returns one, and lowers to the same `map_batches` stage the DataFrame API builds.

```python
import pyarrow.compute as pc

s.register_function("net", lambda a: pc.multiply(a, 0.85))
print(s.sql("SELECT order_id, net(amount) AS net FROM big ORDER BY order_id").to_pydict())
# {'order_id': [1, 3, 5], 'net': [102.0, 68.0, 51.0]}
```

:::{tip}
Keep functions vectorized. `vectorized=False` calls your function once per row, which puts Python in the inner loop.
:::

## 8. Point it at real files

Only the source changes, whether the table came from a dict, a Parquet directory, or a Delta table:

```python
# docs: skip
import batcher as bt

lake = bt.read.parquet("s3://bucket/orders/")
bt.sql(
    "SELECT region, SUM(amount) AS revenue FROM o WHERE amount >= 25 GROUP BY region",
    o=lake,
).write.parquet("s3://bucket/revenue_by_region/", mode="overwrite")
```

## Where to go next

Three directions, all of them written either way:

::::{grid} 1 3 3 3
:gutter: 3

:::{grid-item-card} {octicon}`meter;1.1em` Optimizing a slow query
:link: /getting-started/tutorials/foundations/optimizing-a-slow-query
:link-type: doc
Read the plan you just printed, and act on it.
:::

:::{grid-item-card} {octicon}`database;1.1em` Building a lakehouse
:link: /getting-started/tutorials/pipelines/building-a-lakehouse
:link-type: doc
Point these queries at a real transactional table.
:::

:::{grid-item-card} {octicon}`broadcast;1.1em` A streaming pipeline
:link: /getting-started/tutorials/pipelines/streaming-pipeline
:link-type: doc
The same operators, over a source that never ends.
:::
::::

## See also

- {doc}`SQL guide </user-guide/analyze/sql>`: the supported SQL surface in full, including what is
  not supported.
- {doc}`Expressions </user-guide/transform/columns/expressions>`: the column language underneath both spellings.
- {doc}`Explain plans </user-guide/operate/tuning/explain-plans>`: how to read the thing you just compared.
- {doc}`Plan IR </architecture/deep-dives/query/plan-ir>`: the single `LogicalPlan` both front ends build.
- {doc}`SQL API reference </api/relational/sql>`: `Session`, `register`, {py:func}`register_function <batcher.register_function>`.
- {doc}`Migration guide </getting-started/migration/index>`: if the SQL you know is Spark's or DuckDB's.
