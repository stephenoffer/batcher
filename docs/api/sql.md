# SQL

`bt.sql` runs a SQL query against one or more in-memory datasets and returns a
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

`bt.sql` returns a `Dataset`, so it is lazy and composes with the rest of the API:
add `.filter`, `.with_columns`, or another `bt.sql` on top before a terminal
operation runs the whole plan.

## Supported SQL

The SQL surface is a single dialect (it is not multi-dialect). It covers the
common analytical shape of a query:

| Clause / feature | Notes |
| --- | --- |
| `SELECT` | Column lists, derived expressions, `AS` aliases, `*`. |
| `WHERE` | Boolean predicates over scalar expressions. |
| `GROUP BY` | With aggregates in the projection. |
| `HAVING` | Filters on aggregated results. |
| `ORDER BY` | `ASC` / `DESC`. |
| `LIMIT` / `OFFSET` | Row caps, with optional row skip. |
| `JOIN` | Inner and left joins. |
| Window functions | `<fn> OVER (PARTITION BY … ORDER BY …)` — ranking, aggregates, and `LAG`/`LEAD`/`FIRST_VALUE`/`LAST_VALUE`. |
| `QUALIFY` | Filter on a window-function result (referenced by its output alias). |
| `TABLESAMPLE` | `BERNOULLI(p PERCENT)` (fraction) or `RESERVOIR(n ROWS)` (fixed count). |
| `CASE` | `CASE WHEN ... THEN ... ELSE ... END`. |
| `CAST` | `CAST(expr AS type)`. |
| Aggregates | `COUNT`, `SUM`, `MIN`, `MAX`, `AVG`, and the other supported aggregates. |
| Scalar expressions | Arithmetic, comparison, boolean, and function calls. |

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

## What is not here

There is no `SQLContext`, no `from batcher.sql import ...`, and no `ds.sql(...)`
method on a `Dataset`. `bt.sql` is the one entry point. For anything outside the
supported subset, build the query with the DataFrame API, which exposes the full
operator set.

## Next steps

- [Dataset](dataset.md): the DataFrame surface SQL lowers to.
- [Expressions](expressions.md): the scalar functions available in projections.
