# SQL

{py:obj}`bt.sql <batcher.sql>` runs a SQL query against one or more in-memory datasets and returns a
lazy {py:class}`Dataset <batcher.Dataset>`. The query is parsed, lowered to the same plan the DataFrame API
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

{py:obj}`bt.sql <batcher.sql>` returns a `Dataset`, so it's lazy and composes with the rest of the API. Add `.filter`, `.with_columns`, or another {py:obj}`bt.sql <batcher.sql>` on top before a terminal operation runs the whole plan.

## Supported SQL

The SQL surface reads DuckDB syntax by default. Pass `dialect=` to parse another sqlglot dialect, such as `"postgres"` or `"spark"`. The following table lists the clauses and features the surface covers, in the order a query writes them:

| Clause / feature | Notes |
| --- | --- |
| `SELECT` | Column lists, derived expressions, `AS` aliases, `*`, the `* EXCLUDE (…)` / `* REPLACE (… AS c)` / `* RENAME (c AS d)` star modifiers, and `COLUMNS(*)` / `COLUMNS('regex')` dynamic columns (including {py:obj}`func(COLUMNS(...)) <batcher.AggExpr.func>`). |
| `WHERE` | Boolean predicates over scalar expressions. |
| `GROUP BY` | With aggregates in the projection; `ROLLUP` / `CUBE` / `GROUPING SETS`. Positional `GROUP BY <n>` refers to a `SELECT` item. |
| `HAVING` | Filters on aggregated results. |
| `ORDER BY` | `ASC` / `DESC`, `NULLS FIRST` / `NULLS LAST` (default `NULLS LAST`), and positional `ORDER BY <n>`. |
| `LIMIT` / `OFFSET` | Row caps, with optional row skip. |
| `JOIN` | Inner, left, right, full, cross, and `NATURAL` joins on equi-keys (`ON` / `USING` / shared columns); an extra non-equi `AND` condition is applied as a filter on the join result. |
| Set operations | `UNION` / `UNION ALL`, `INTERSECT`, `EXCEPT`. |
| `WITH` | Common table expressions (CTEs). |
| Subqueries | Derived tables, `IN` / `NOT IN`, `EXISTS` / `NOT EXISTS`, `= ANY` / `= SOME` / `<> ALL`, correlated scalar subqueries. See [Subqueries](#subqueries) for the forms that don't translate. |
| Window functions | `<fn> OVER (PARTITION BY ... ORDER BY ... [ROWS BETWEEN ...])`: ranking, aggregates, and `LAG`/`LEAD`/`FIRST_VALUE`/`LAST_VALUE`, with explicit `ROWS` frames. |
| `QUALIFY` | Filter on a window-function result (referenced by its output alias). |
| `TABLESAMPLE` | `BERNOULLI(p PERCENT)` (fraction) or `RESERVOIR(n ROWS)` (fixed count). |
| `CASE` | `CASE WHEN ... THEN ... ELSE ... END`. |
| `CAST` | `CAST(expr AS type)`. |
| Aggregates | `COUNT`, `SUM`, `MIN`, `MAX`, `AVG`, and the other supported aggregates, including the `DISTINCT` forms. See [DISTINCT aggregates](#distinct-aggregates) for what they may be mixed with. |
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

### Star modifiers

`SELECT *` accepts DuckDB's modifiers, so a wide table needs no exhaustive column list to drop, rewrite, or rename a few columns. `EXCLUDE (...)` omits columns, `REPLACE (expr AS c)` swaps a column's expression in place, and `RENAME (c AS d)` renames one. Each keeps every other column in its original position.

```python
out = bt.sql(
    "SELECT * EXCLUDE (id) REPLACE (amount * 2 AS amount) FROM events ORDER BY category, amount",
    events=events,
)
print(out.to_pydict())
# {'category': ['a', 'a', 'a', 'b', 'b'], 'amount': [20.0, 60.0, 100.0, 40.0, 80.0]}
```

`ORDER BY` places nulls last by default, matching DuckDB. Write `NULLS FIRST` or `NULLS LAST` per key to control it explicitly.

### Dynamic columns with COLUMNS(...)

DuckDB's `COLUMNS(*)` and `COLUMNS('regex')` project a set of columns chosen at plan time, and a function applied to `COLUMNS(...)` runs on each matched column. A wide-table transform therefore needs no exhaustive column list.

```python
metrics = bt.from_pydict(
    {"day": ["mon", "tue"], "sales_us": [10, 20], "sales_eu": [30, 40]}
)
out = bt.sql("SELECT day, COLUMNS('sales_.*') * 2 FROM metrics", metrics=metrics)
print(out.to_pydict())
# {'day': ['mon', 'tue'], 'sales_us': [20, 40], 'sales_eu': [60, 80]}
```

### Joins

Bind one table per keyword. The join types listed in the table above all apply here.

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

(distinct-aggregates)=
### DISTINCT aggregates

`COUNT(DISTINCT x)` is a native aggregate and mixes with anything. The others —
`SUM(DISTINCT x)`, `AVG(DISTINCT x)`, `MIN`/`MAX(DISTINCT x)` — are answered by grouping on
the group keys plus `x`, which dedups `x`, and then aggregating that. Any other aggregate in
the same query has to survive that pre-aggregation, so it must combine from per-sub-group
partials: `COUNT`, `SUM`, `MIN`, `MAX`, `BOOL_AND`, `BOOL_OR`, `BIT_AND`, `BIT_OR`,
`BIT_XOR`, `PRODUCT`, and `ANY_VALUE`.

```python
sales = bt.from_pydict(
    {
        "region": ["w", "w", "w", "e", "e"],
        "customer": [1, 1, 2, 3, 3],
        "amount": [10, 10, 20, 30, 40],
    }
)

out = bt.sql(
    """
    SELECT region, SUM(DISTINCT customer) AS customers, SUM(amount) AS total
    FROM sales GROUP BY region ORDER BY region
    """,
    sales=sales,
)
print(out.to_pydict())
# {'region': ['e', 'w'], 'customers': [3, 3], 'total': [70, 40]}
```

`AVG`, `STDDEV`, `VAR`, the quantiles, and a second `COUNT(DISTINCT ...)` over a *different*
column cannot: an average needs a sum and a count, which one column cannot carry. Those raise
rather than approximate. Compute them in a separate subquery and join.

(subqueries)=
### Subqueries

Set-membership subqueries are folded into joins, so they run as one plan rather than once per
row. `= ANY` and `= SOME` are `IN`; `<> ALL` is `NOT IN`; all four spell the same predicate.

```python
vip = bt.from_pydict({"category": ["a", "c"]})

out = bt.sql(
    """
    SELECT category, COUNT(*) AS n
    FROM events
    WHERE category = ANY (SELECT category FROM vip)
    GROUP BY category
    ORDER BY category
    """,
    events=events,
    vip=vip,
)
print(out.to_pydict())
# {'category': ['a'], 'n': [3]}
```

`EXISTS` may also sit under `OR`, where it becomes a boolean marker column instead of a
join, because a join would drop the rows the `OR` keeps.

```python
out = bt.sql(
    """
    SELECT id FROM events
    WHERE EXISTS (SELECT 1 FROM vip WHERE vip.category = events.category)
       OR amount > 40
    ORDER BY id
    """,
    events=events,
    vip=vip,
)
print(out.to_pydict())
# {'id': [1, 3, 5]}
```

Two forms raise rather than translate, both because the honest answer needs SQL's third
truth value and the natural rewrite cannot express it:

- The inequality quantifiers (`> ANY`, `>= ALL`, …). `x > ALL (S)` is UNKNOWN, not TRUE,
  when `S` yields a NULL, and `x > (SELECT max(c) FROM S)` says TRUE because `max` skips
  NULLs.
- `IN` under `OR`. Write it as the `EXISTS` above, qualifying the outer column.

Each error names the rewrite that works.

## Sessions, registered tables, and dialects

{py:obj}`bt.Session <batcher.Session>` is the DuckDB-connection / SparkSession
analogue: a context that holds a table catalog, registered Python functions, and a dialect. The module-level {py:func}`bt.sql <batcher.sql>` and {py:func}`bt.register_function <batcher.register_function>` delegate to a shared default session, so the zero-setup spelling keeps working. Build a {py:class}`Session <batcher.Session>` when you want to register named tables, isolate a workload, or pick a different dialect.

```python
s = bt.Session()
s.register("events", events)  # the DuckDB con.register / Spark createOrReplaceTempView equivalent
out = s.sql("SELECT category, SUM(amount) AS total FROM events GROUP BY category ORDER BY category")
print(out.to_pydict())
# {'category': ['a', 'b'], 'total': [90.0, 60.0]}
```

Pass `dialect=` to either `bt.sql` or `bt.Session(dialect=...)` to select the sqlglot read dialect:

```python
out = bt.sql("SELECT STRPOS(category, 'a') AS p FROM events", events=events, dialect="postgres")
print(out.to_pydict())
# {'p': [1, 0, 1, 0, 1]}
```

## Calling Python functions from SQL

Register a Python function with
{py:obj}`register_function <batcher.register_function>` and call it from SQL. The
function runs over Arrow batches (it lowers to `map_batches`), so it composes with
relational operators in one plan. There are two call forms.

A **scalar** function, called as `SELECT f(x)` or `WHERE f(x)`, is vectorized by default. It receives an Arrow array and returns one:

```python
import pyarrow.compute as pc

s.register_function("bump", lambda a: pc.multiply(a, 10))
out = s.sql("SELECT id, bump(amount) AS scaled FROM events WHERE bump(amount) > 200")
print(out.to_pydict())
# {'id': [3, 4, 5], 'scaled': [300.0, 400.0, 500.0]}
```

A **table** function, called as `SELECT * FROM f(t)`, transforms a whole relation:

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

Scalar functions aren't supported directly in a `GROUP BY` key, an aggregate argument, or `ORDER BY`. Compute them in a subquery or a projected alias first.

## Scoring a model from SQL

`ML_PREDICT(relation, model)` appends a fitted model's predictions to a relation, so a query
can score and then keep filtering, joining and aggregating the result without leaving SQL.
Register the model on the session first, or name a saved one by quoted path:

```python
from batcher.ml import LinearRegression

train = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, 6.0, 8.0]})
s.register_model("doubler", LinearRegression(features=["x"], target="y").fit(train))
s.register("points", bt.from_pydict({"x": [5.0, 10.0]}))

out = s.sql("SELECT x, prediction FROM ML_PREDICT(points, doubler) ORDER BY x")
print(out.to_pydict())
# {'x': [5.0, 10.0], 'prediction': [10.0, 20.0]}
```

The prediction is an ordinary column, so the rest of the query sees it:

```python
out = s.sql("SELECT COUNT(*) AS n FROM ML_PREDICT(points, doubler) WHERE prediction > 15")
print(out.to_pydict())
# {'n': [1]}
```

The model is either a fitted `batcher.ml` estimator, which scores a relation directly, or a
model from another framework (XGBoost, LightGBM, CatBoost, scikit-learn, ONNX) given as an
object or a path to a saved file. A framework model is scored through
{py:meth}`ds.ml.predict <batcher.Dataset.ml>`, so it loads once per worker and each batch
crosses as one dense matrix.

Three settings can be passed as `name => value`:

| Setting | Meaning |
|---|---|
| `features` | The feature columns, in model order. Framework models only; a `batcher.ml` estimator was fitted with its own. |
| `method` | What to compute, such as `predict_proba`. Defaults to `predict`. |
| `output_column` | The name of the appended column. Defaults to `prediction`. |

```python
out = s.sql(
    "SELECT score FROM ML_PREDICT(points, doubler, output_column => 'score') ORDER BY score"
)
print(out.to_pydict())
# {'score': [10.0, 20.0]}
```

BigQuery's own spelling works too, for a query being ported rather than written fresh. It is
BigQuery grammar, so it needs the BigQuery read dialect:

```python
bq = bt.Session(dialect="bigquery")
bq.register("points", bt.from_pydict({"x": [5.0, 10.0]}))
bq.register_model("doubler", LinearRegression(features=["x"], target="y").fit(train))
out = bq.sql("SELECT x, prediction FROM ML.PREDICT(MODEL doubler, TABLE points) ORDER BY x")
print(out.to_pydict())
# {'x': [5.0, 10.0], 'prediction': [10.0, 20.0]}
```

Settings there go in the trailing `STRUCT`, as `ML.PREDICT(MODEL m, TABLE t, STRUCT('score' AS output_column))`.

## Defining tables and views with SQL

`CREATE TABLE/VIEW ... AS` and `DROP TABLE` register and unregister a **lazy** dataset in the session catalog. Nothing is materialized until a terminal operation runs it:

```python
s.sql("CREATE VIEW big_events AS SELECT id, amount FROM events WHERE amount > 25")
print(s.sql("SELECT * FROM big_events ORDER BY id").to_pydict())
# {'id': [3, 4, 5], 'amount': [30.0, 40.0, 50.0]}
```

## Binding the current dataset

{py:meth}`Dataset.sql <batcher.Dataset.sql>` runs a query with the dataset bound to a name (`self` by default,
the Polars spelling), so a query can build on an existing pipeline:

```python
out = events.sql("SELECT id, amount FROM self WHERE amount > 25 ORDER BY id")
print(out.to_pydict())
# {'id': [3, 4, 5], 'amount': [30.0, 40.0, 50.0]}
```

## See also

- {doc}`SQL user guide </user-guide/analyze/sql>`: a guided tour with runnable queries.
- {doc}`Dataset </api/relational/dataset>`: the DataFrame surface SQL lowers to.
- {doc}`Expressions </api/relational/expressions>`: the scalar functions available in projections.
- {doc}`/cookbook/dataset/verbs/sql_interface`: mixing SQL with DataFrame verbs, as a runnable script.
