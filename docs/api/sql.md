# SQL

{py:obj}`bt.sql <batcher.sql>` runs a SQL query against one or more in-memory datasets and returns a
lazy `Dataset`. The query is parsed, lowered to the same plan the DataFrame API
builds, and optimized by the same passes, so SQL and DataFrame code share one
engine and one set of semantics.

The signature is `bt.sql(query, table_name=ds_or_table, ...) -> Dataset`.
Each keyword binds a table name used in the query to a `Dataset` or a pyarrow
`Table`. The query references those names in its `FROM` and `JOIN` clauses.

## A first query

```python
import batcher as bt

ds = bt.from_pydict({"category": ["a", "b", "a", "b", "a", "c"]})

out = bt.sql(
    "SELECT category, COUNT(*) AS n FROM t GROUP BY category ORDER BY category",
    t=ds,
)
print(out.to_pydict())
# {'category': ['a', 'b', 'c'], 'n': [3, 2, 1]}
```

{py:obj}`bt.sql <batcher.sql>` returns a `Dataset`, so it is lazy and composes with the rest of the API:
add `.filter`, `.with_columns`, or another {py:obj}`bt.sql <batcher.sql>` on top before a terminal
operation runs the whole plan.

## Supported SQL

The SQL surface reads DuckDB syntax by default; pass `dialect=` to parse another
sqlglot dialect (`"postgres"`, `"spark"`, …). It covers the common analytical
shape of a query:

| Clause / feature | Notes |
| --- | --- |
| `SELECT` | Column lists, derived expressions, `AS` aliases, `*`. |
| `WHERE` | Boolean predicates over scalar expressions. |
| `GROUP BY` | With aggregates in the projection; `ROLLUP` / `CUBE` / `GROUPING SETS`. |
| `HAVING` | Filters on aggregated results. |
| `ORDER BY` | `ASC` / `DESC`. |
| `LIMIT` / `OFFSET` | Row caps, with optional row skip. |
| `JOIN` | Inner, left, right, full, and cross joins on equi-keys; an extra non-equi `AND` condition is applied as a filter on the join result. |
| Set operations | `UNION` / `UNION ALL`, `INTERSECT`, `EXCEPT`. |
| `WITH` | Common table expressions (CTEs). |
| Subqueries | Derived tables, `IN` / `NOT IN`, `EXISTS` / `NOT EXISTS`, correlated scalar subqueries. |
| Window functions | `<fn> OVER (PARTITION BY … ORDER BY … [ROWS BETWEEN …])` — ranking, aggregates, and `LAG`/`LEAD`/`FIRST_VALUE`/`LAST_VALUE`, with explicit `ROWS` frames. |
| `QUALIFY` | Filter on a window-function result (referenced by its output alias). |
| `TABLESAMPLE` | `BERNOULLI(p PERCENT)` (fraction) or `RESERVOIR(n ROWS)` (fixed count). |
| `CASE` | `CASE WHEN ... THEN ... ELSE ... END`. |
| `CAST` | `CAST(expr AS type)`. |
| Aggregates | `COUNT`, `SUM`, `MIN`, `MAX`, `AVG`, and the other supported aggregates. |
| Scalar expressions | Arithmetic, comparison, boolean, and function calls (incl. registered Python functions). |
| DDL | `CREATE [OR REPLACE] {TABLE,VIEW} … AS …` and `DROP TABLE` register/unregister a lazy table in the session. |

### WHERE and GROUP BY

```python
events = bt.from_pydict(
    {
        "id": [1, 2, 3, 4, 5],
        "category": ["a", "b", "a", "b", "a"],
        "amount": [10.0, 20.0, 30.0, 40.0, 50.0],
    }
)

out = bt.sql(
    """
    SELECT category, SUM(amount) AS total
    FROM events
    WHERE id > 1
    GROUP BY category
    ORDER BY category
    """,
    events=events,
)
print(out.to_pydict())
# {'category': ['a', 'b'], 'total': [80.0, 60.0]}
```

### CASE and CAST

```python
out = bt.sql(
    """
    SELECT
        CASE WHEN amount >= 30.0 THEN 'high' ELSE 'low' END AS tier,
        CAST(amount AS BIGINT) AS amount_int
    FROM events
    ORDER BY amount
    """,
    events=events,
)
print(out.to_pydict())
# {'tier': ['low', 'low', 'high', 'high', 'high'],
#  'amount_int': [10, 20, 30, 40, 50]}
```

### Joins

Bind one table per keyword. Inner and left joins are supported.

```python
dim = bt.from_pydict({"category": ["a", "b"], "region": ["west", "east"]})

out = bt.sql(
    """
    SELECT e.category, d.region, COUNT(*) AS n
    FROM events e
    JOIN dim d ON e.category = d.category
    GROUP BY e.category, d.region
    ORDER BY e.category
    """,
    events=events,
    dim=dim,
)
print(out.to_pydict())
# {'category': ['a', 'b'], 'region': ['west', 'east'], 'n': [3, 2]}
```

## Sessions, registered tables, and dialects

{py:obj}`bt.Session <batcher.Session>` is the DuckDB-connection / SparkSession
analogue: a context that holds a table catalog, registered Python functions, and a
dialect. The module-level `bt.sql` and `bt.catalog` delegate to a shared default
session, so the global spelling keeps working; build a `Session` to isolate a
workload or pick a non-default dialect.

```python
s = bt.Session()
s.register("events", events)  # like DuckDB con.register / Spark createOrReplaceTempView
out = s.sql("SELECT category, SUM(amount) AS total FROM events GROUP BY category ORDER BY category")
print(out.to_pydict())
# {'category': ['a', 'b'], 'total': [90.0, 60.0]}
```

`dialect=` (on `bt.sql` or `bt.Session(dialect=...)`) selects the sqlglot read
dialect:

```python
out = bt.sql("SELECT STRPOS(category, 'a') AS p FROM events", events=events, dialect="postgres")
print(out.to_pydict())
# {'p': [1, 0, 1, 0, 1]}
```

## Calling Python functions from SQL

Register a Python function with
{py:obj}`register_function <batcher.register_function>` and call it from SQL. The
function runs over Arrow batches (it lowers to `map_batches`), so it composes with
relational operators in one plan. Two call forms:

A **scalar** function — `SELECT f(x)` / `WHERE f(x)` — is vectorized by default
(it receives an Arrow array and returns one):

```python
import pyarrow.compute as pc

s.register_function("bump", lambda a: pc.multiply(a, 10))
out = s.sql("SELECT id, bump(amount) AS scaled FROM events WHERE bump(amount) > 200")
print(out.to_pydict())
# {'id': [3, 4, 5], 'scaled': [300.0, 400.0, 500.0]}
```

A **table** function — `SELECT * FROM f(t)` — transforms a whole relation:

```python
def add_flag(batch):
    big = pc.greater(batch.column("amount"), 25)
    return batch.append_column("big", big)

s.register_function(
    "flagged", add_flag, table=True, output_columns=["id", "category", "amount", "big"]
)
out = s.sql("SELECT id, big FROM flagged(events) ORDER BY id")
print(out.to_pydict())
# {'id': [1, 2, 3, 4, 5], 'big': [False, False, True, True, True]}
```

Scalar functions are not supported in a `GROUP BY` key, an aggregate argument, or
`ORDER BY` directly — compute them in a subquery or a projected alias first.

## Defining tables and views with SQL

`CREATE TABLE/VIEW … AS` and `DROP TABLE` register and unregister a **lazy**
dataset in the session catalog (nothing is materialized; a terminal op runs it):

```python
s.sql("CREATE VIEW big_events AS SELECT id, amount FROM events WHERE amount > 25")
print(s.sql("SELECT * FROM big_events ORDER BY id").to_pydict())
# {'id': [3, 4, 5], 'amount': [30.0, 40.0, 50.0]}
```

## Binding the current dataset

`Dataset.sql` runs a query with the dataset bound to a name (`self` by default,
the Polars spelling), so a query can build on an existing pipeline:

```python
out = events.sql("SELECT id, amount FROM self WHERE amount > 25 ORDER BY id")
print(out.to_pydict())
# {'id': [3, 4, 5], 'amount': [30.0, 40.0, 50.0]}
```

## Next steps

- [Dataset](dataset.md): the DataFrame surface SQL lowers to.
- [Expressions](expressions.md): the scalar functions available in projections.
