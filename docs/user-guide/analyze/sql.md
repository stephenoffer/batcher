# SQL

Batcher runs SQL through the same engine as the DataFrame API. {py:obj}`bt.sql(query, ...) <batcher.sql>`
parses a query, binds each named table to a Dataset, and returns a new Dataset.
Because the result is a Dataset, you can keep chaining DataFrame operations onto a
SQL query, or feed a DataFrame pipeline into SQL.

{py:obj}`bt.sql <batcher.sql>` reads DuckDB syntax by default; pass `dialect=` to
read another sqlglot dialect. For a reusable catalog of tables and Python functions,
build a {py:obj}`bt.Session <batcher.Session>`, the DuckDB-connection / SparkSession
analogue. {py:func}`bt.sql <batcher.sql>` and {py:func}`bt.register_function <batcher.register_function>` use a shared default session.

```python
import batcher as bt

ds = bt.from_pydict(
    {
        "category": ["a", "b", "a", "b", "a", "c"],
        "price": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    }
)
```

## Running a query

Pass the query string and bind each table name in the query to a Dataset (or a
pyarrow Table) as a keyword argument.

```python
out = bt.sql(
    "SELECT category, COUNT(*) AS n FROM t GROUP BY category ORDER BY category",
    t=ds,
)
print(out.to_pydict())
# {'category': ['a', 'b', 'c'], 'n': [3, 2, 1]}
```

The keyword name (`t` above) is the table identifier used in the `FROM` clause.

## Supported subset

A query may use:

- `SELECT` with column references, scalar expressions, aggregates
- `WHERE` filters
- `GROUP BY` with `HAVING` (and `ROLLUP` / `CUBE` / `GROUPING SETS`, with `GROUPING()`
  to tell a subtotal row from a real one)
- `ORDER BY` (including `ORDER BY ALL`), `LIMIT` / `OFFSET`, and the ANSI
  `FETCH FIRST n ROWS ONLY`
- `INNER` / `LEFT` / `RIGHT` / `FULL` / `CROSS JOIN` (equi-keys; an extra non-equi
  `AND` condition is applied as a filter), `NATURAL JOIN`, and `ASOF JOIN`
- `UNION` / `INTERSECT` / `EXCEPT`, `WITH` (CTEs), and subqueries
- Window functions over any expression, including a computed `PARTITION BY` / `ORDER BY` key such as `date_trunc('month', ts)`, with explicit `ROWS` / `RANGE` / `GROUPS` frames — including `RANGE BETWEEN INTERVAL '5' MINUTE PRECEDING` for a time window
- `CASE` expressions, `CAST`, and `SIMILAR TO`
- `generate_series(a, b)` / `range(a, b)` in `FROM`, for a generated integer spine
- `UNNEST`, in the `FROM` clause or written directly in the `SELECT` list

You can also register Python functions and call them from SQL, and define tables and
views with `CREATE`/`DROP`. See [Sessions and Python functions](#sessions-tables-and-python-functions).

## Filtering and projection

```python
out = bt.sql("SELECT category, price FROM t WHERE price >= 30 ORDER BY price", t=ds)
print(out.to_pydict())
# {'category': ['a', 'b', 'a', 'c'], 'price': [30.0, 40.0, 50.0, 60.0]}
```

## Aggregation with HAVING

```python
out = bt.sql(
    "SELECT category, SUM(price) AS total FROM t "
    "GROUP BY category HAVING SUM(price) > 60 ORDER BY category",
    t=ds,
)
print(out.to_pydict())
# {'category': ['a'], 'total': [90.0]}
```

## CASE and CAST

```python
out = bt.sql(
    "SELECT category, "
    "CASE WHEN price >= 40 THEN 'high' ELSE 'low' END AS tier, "
    "CAST(price AS BIGINT) AS price_int "
    "FROM t ORDER BY price",
    t=ds,
)
print(out.to_pydict())
# {'category': ['a', 'b', 'a', 'b', 'a', 'c'], 'tier': ['low', 'low', 'low', 'high', 'high', 'high'], 'price_int': [10, 20, 30, 40, 50, 60]}
```

## Joining tables

Bind one Dataset per table named in the query.

```python
dim = bt.from_pydict({"category": ["a", "b"], "region": ["west", "east"]})
out = bt.sql(
    "SELECT t.category, t.price, d.region "
    "FROM t INNER JOIN d ON t.category = d.category "
    "ORDER BY t.price",
    t=ds,
    d=dim,
)
print(out.to_pydict())
# {'category': ['a', 'b', 'a', 'b', 'a'], 'price': [10.0, 20.0, 30.0, 40.0, 50.0], 'region': ['west', 'east', 'west', 'east', 'west']}
```

## Mixing SQL and the DataFrame API

A SQL result is an ordinary Dataset, so you can continue with DataFrame methods.

```python
totals = bt.sql("SELECT category, SUM(price) AS total FROM t GROUP BY category", t=ds)
out = totals.filter(bt.col("total") >= 90).sort("category")
print(out.to_pydict())
# {'category': ['a'], 'total': [90.0]}
```

Both paths build one logical plan, push it through one optimizer, and execute it on
one Rust data plane. There is no separate SQL engine.

(sessions-tables-and-python-functions)=

## Sessions, tables, and Python functions

A {py:obj}`bt.Session <batcher.Session>` holds a dialect plus a catalog: the tables
and the Python functions you registered. Register a dataset as a table, then query it
by name.

```python
s = bt.Session()
s.register("t", ds)
print(s.sql("SELECT COUNT(*) AS n FROM t").to_pydict())
# {'n': [6]}
```

Register a Python function and call it from SQL. A scalar function is vectorized, so it
receives an Arrow array rather than a value. It lowers to the same `map_batches` path as
the DataFrame API, so Python and SQL share one plan:

```python
import pyarrow.compute as pc

s.register_function("discount", lambda a: pc.multiply(a, 0.9))
print(s.sql("SELECT discount(price) AS net FROM t ORDER BY price").to_pydict())
# {'net': [9.0, 18.0, 27.0, 36.0, 45.0, 54.0]}
```

`CREATE TABLE/VIEW AS` and `DROP TABLE` register and unregister a lazy table in the
session. Nothing materializes until a terminal op:

```python
s.sql("CREATE VIEW cheap AS SELECT category, price FROM t WHERE price < 30")
print(s.sql("SELECT * FROM cheap ORDER BY price").to_pydict())
# {'category': ['a', 'b'], 'price': [10.0, 20.0]}
```

{py:meth}`ds.sql("... FROM self") <batcher.Dataset.sql>` binds the current dataset directly:

```python
print(ds.sql("SELECT category FROM self WHERE price >= 50 ORDER BY price").to_pydict())
# {'category': ['a', 'c']}
```

A fitted model registers the same way, with {py:meth}`register_model <batcher.Session.register_model>`, and `ML_PREDICT` then scores a relation inside the query. The prediction is an ordinary column, so the rest of the statement filters, joins and aggregates over it without leaving SQL:

```python
from batcher.ml import LinearRegression

train = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, 6.0, 8.0]})
s.register_model("doubler", LinearRegression(features=["x"], target="y").fit(train))
s.register("points", bt.from_pydict({"x": [5.0, 10.0]}))

print(s.sql("SELECT COUNT(*) AS n FROM ML_PREDICT(points, doubler) WHERE prediction > 15").to_pydict())
# {'n': [1]}
```

Scoring stays inside the plan, so the model runs where the data is rather than pulling rows back to the driver. A saved model can be named by quoted path instead of registering it first; see the {doc}`SQL API </api/relational/sql>`.

## Generative AI functions

`ML_PREDICT` covers the *traditional* model. `AI_GENERATE` is the generative half: a language
model asked to write from a text column, with `ai_query` and `ai_complete` accepted as aliases
so a query ported from Databricks or Snowflake runs as written. `AI_EXTRACT` is the same shape
for pulling typed fields out of the text.

An engine is registered in Python and named in SQL, never written inline. It carries an
endpoint, credentials and sampling settings, so a quoted engine argument is refused rather than
becoming a way to put an API key in query text:

```python
s = bt.Session()
s.register(
    "reviews", bt.from_pydict({"id": [1, 2, 3], "body": ["love it", "broke fast", "it is fine"]})
)


def shouty():
    # Stands in for `bt.ml.http_engine(...)` / `bt.ml.vllm_engine(...)` so this page needs
    # no model: any zero-argument callable returning `list[str] -> list[str]` is an engine.
    return lambda prompts: [p.upper() for p in prompts]


s.register_engine("shouty", shouty)

print(
    s.sql(
        "SELECT id, response FROM AI_GENERATE(reviews, shouty, prompt_column => 'body')"
    ).to_pydict()
)
# {'id': [1, 2, 3], 'response': ['LOVE IT', 'BROKE FAST', 'IT IS FINE']}
```

The relation and the engine are positional; everything that changes the answer is a named
setting. `AI_GENERATE` takes `prompt_column` (required), `template` and `output_column`.
`AI_EXTRACT` takes `prompt_column` and a `schema` written as a column definition list, and
appends one typed column per field:

```python
import json


def grader():
    return lambda prompts: [
        json.dumps({"label": "positive" if "love" in p else "negative"}) for p in prompts
    ]


s.register_engine("grader", grader)

print(
    s.sql(
        "SELECT label, COUNT(*) AS n FROM AI_EXTRACT(reviews, grader,"
        " prompt_column => 'body', schema => ['label string'])"
        " GROUP BY label ORDER BY label"
    ).to_pydict()
)
# {'label': ['negative', 'positive'], 'n': [2, 1]}
```

The generated column is an ordinary column, so the rest of the statement groups, filters and
joins over it without leaving SQL.

### Why these read as tables rather than as functions in the SELECT list

Every warehouse writes its AI call in the `SELECT` list and this does not, for the reason that
also makes `ML_PREDICT` a table function. A Batcher scalar function lowers to an expression
evaluated per row in Rust, and a language-model call is neither expressible there nor wanted
per row: the whole point of the inference path is that an engine loads once per worker and
sees a batch at a time. Writing the call in `FROM` says that rather than hiding it.

### What is not translated

`AI_CLASSIFY` is not. Its grammar is fixed at three arguments, and a relational form needs
four: the relation, the engine, the text column and the labels. Use `AI_EXTRACT` with a
one-field schema, or {py:meth}`ds.ml.classify <batcher.api.dataset.ml.DatasetML.classify>` on
the `Dataset`. `AI_EMBED`, `AI_SIMILARITY`, `AI_AGG` and `AI_FORECAST` are likewise
DataFrame-side; each reports where its capability lives rather than failing as an unknown
table.

The full set is always available on the `Dataset`, where these lower to anyway:
{py:meth}`ds.ml.generate <batcher.api.dataset.ml.DatasetML.generate>`,
{py:meth}`ds.ml.classify <batcher.api.dataset.ml.DatasetML.classify>`,
{py:meth}`ds.ml.extract <batcher.api.dataset.ml.DatasetML.extract>` and
{py:meth}`ds.ml.embed <batcher.api.dataset.ml.DatasetML.embed>`. See
{doc}`the LLM engines page </ml/retrieval/llm/engines>` for the engines they take, and
{doc}`batch inference </ml/inference/index>` for batching, GPU sizing and error handling.

## Matching each row to the nearest one

`ASOF JOIN` matches every left row to the single nearest right row rather than to every
row satisfying the condition. The `ON` splits into exact-match keys, written as
equalities, and one nearest-match key, written as `>=` (look backward) or `<=` (look
forward). It is how you attach the most recent quote to each trade, or the prevailing
price to each event.

```python
trades = bt.from_pydict({"sym": ["A", "A", "B"], "ts": [5, 30, 12], "qty": [1, 2, 3]})
quotes = bt.from_pydict({"sym": ["A", "A", "B"], "ts": [1, 20, 10], "bid": [9.0, 11.0, 7.0]})

out = bt.sql(
    "SELECT t.sym, t.ts, q.bid FROM t ASOF JOIN q "
    "ON t.sym = q.sym AND t.ts >= q.ts ORDER BY t.sym, t.ts",
    t=trades,
    q=quotes,
)
print(out.to_pydict())
# {'sym': ['A', 'A', 'B'], 'ts': [5, 30, 12], 'bid': [9.0, 11.0, 7.0]}
```

`ASOF JOIN` drops a left row that matches nothing; `ASOF LEFT JOIN` keeps it with NULL
right columns.

## Measuring the gap between two timestamps

`date_diff(unit, start, end)` reports how many `unit` boundaries lie between two instants.
The units run from `microsecond` through `year`, and both `DATE` and `TIMESTAMP` inputs
work.

```python
import datetime as dt

events = bt.from_pydict(
    {
        "opened": [dt.datetime(2024, 1, 1, 9, 55), dt.datetime(2024, 1, 1, 9, 0)],
        "closed": [dt.datetime(2024, 1, 1, 10, 5), dt.datetime(2024, 1, 1, 9, 59)],
    }
)
print(
    bt.sql(
        "SELECT date_diff('minute', opened, closed) AS mins,"
        "       date_diff('hour', opened, closed) AS hrs FROM events",
        events=events,
    ).to_pydict()
)
# {'mins': [10, 59], 'hrs': [1, 0]}
```

Both rows above are worth reading twice, because they show the rule. `date_diff` counts
**boundary crossings**, not elapsed time. The first row spans ten minutes and reports one
hour, because a clock hour ticks over between 9:55 and 10:05. The second spans fifty-nine
minutes and reports zero hours, because none does. If you want elapsed time instead,
subtract the two timestamps and read the duration, or take the difference of
`epoch(ts)` values.

`week` is the one unit that does not follow that rule: DuckDB defines it as the number of
whole seven-day spans, truncated toward zero, so a Thursday to the following Monday is `0`
even though a calendar week boundary falls between them. Batcher matches DuckDB here.

## Window functions over an expression

A window function's argument is an ordinary expression, not only a column name, so a
running total of a computed value or a conditional running count is written directly.
`lag` and `lead` take a third argument that fills the rows whose offset falls outside
the partition.

```python
sales = bt.from_pydict(
    {"day": [1, 2, 3, 4], "price": [10.0, 20.0, 30.0, 40.0], "qty": [1, 2, 1, 3]}
)

out = bt.sql(
    "SELECT day, "
    "SUM(price * qty) OVER (ORDER BY day) AS revenue, "
    "SUM(CASE WHEN qty > 1 THEN 1 ELSE 0 END) OVER (ORDER BY day) AS bulk_orders, "
    "LAG(price, 1, 0.0) OVER (ORDER BY day) AS prev_price "
    "FROM s ORDER BY day",
    s=sales,
)
print(out.to_pydict())
# {'day': [1, 2, 3, 4], 'revenue': [10.0, 50.0, 80.0, 200.0], 'bulk_orders': [0, 1, 1, 2], 'prev_price': [0.0, 10.0, 20.0, 30.0]}
```

The default fills only the rows that have no row at that offset. A NULL the column
genuinely holds inside the partition stays NULL.

## Duplicate output names

SQL lets a `SELECT` list emit the same name twice, most often when a join projects a key
from both sides. A Dataset is keyed by column name, so the second one is suffixed —
`id`, then `id_1` — which is the same name DuckDB assigns when it has to make result
names unique.

```python
left = bt.from_pydict({"id": [1, 2], "v": [10, 20]})
right = bt.from_pydict({"id": [1, 2], "w": [7, 8]})

out = bt.sql("SELECT l.id, r.id, l.v FROM l JOIN r ON l.id = r.id ORDER BY l.id", l=left, r=right)
print(out.columns)
# ['id', 'id_1', 'v']
```

## Requirements and limitations

Constructs Batcher rejects rather than approximates. Each raises a clear error, because
answering with a different row set and reporting success is the failure nothing
downstream can detect.

| Construct | Why, and what to write instead |
|---|---|
| `LIMIT n PERCENT`, `FETCH ... WITH TIES` | Both need a cardinality measured before the limit applies. Use a plain row count. |
| `POSITIONAL JOIN` | Row position is not defined for a Batcher relation, which is morsel-parallel and may span nodes. Join on a key, or number both sides with `row_number() OVER (ORDER BY ...)` first. |
| `ASOF JOIN` on a strict `>` or `<` | The nearest-match key is inclusive. Use `>=` or `<=`. |
| A negative list-slice bound, `a[-2:]` | Counts back from the end in DuckDB; the underlying slice clamps to the start. Index from the front, or reverse the list first. |
| A correlated subquery whose correlation is an inequality | An equality correlation decorrelates to a join and is supported; an inequality one is not. |
| `time_bucket` with a width that doesn't divide a day evenly | Buckets start from the Unix epoch, DuckDB starts them from 2000-01-03, so a width such as `INTERVAL 2 DAY` would put every boundary on a different instant. Use a width that divides a day (`1 DAY`, `6 HOUR`, `15 MINUTE`), or `date_trunc` for calendar buckets. |
| Two `UNNEST` calls in one `SELECT` list | SQL zips them into one relation. Unnest one list per query, or use `FROM t, UNNEST(...)` for each. |

One construct succeeds and returns the same values under a **different type**. Adding a
date-granular interval to a `DATE` keeps a `DATE`, where DuckDB and Postgres widen to
`TIMESTAMP`:

```python
out = bt.sql("SELECT DATE '2024-01-01' + INTERVAL 1 DAY AS d")
print(out.schema.field("d").type, out.to_pydict()["d"][0])
# date32[day] 2024-01-02
```

Batcher widens only when the interval carries a time component, so `+ INTERVAL 2 HOUR`
does give a `TIMESTAMP`. This matches Spark and keeps a date column usable as a date;
cast explicitly if you need DuckDB's type. The row values are identical either way.

Descending list sorts agree with DuckDB, NULLs included. `list_reverse_sort` lowers to
`.list.sort_desc()`, which is a kernel of its own rather than `sort().reverse()` — ascending
puts NULLs last, so reversing would lift them to the front, where DuckDB keeps them at the
back. Both spellings return `[2, 1, NULL]` for `[1, NULL, 2]`.

### Known divergences

One construct returns a result that differs from DuckDB's, and it is tracked as a defect
rather than intended behavior:

| Construct | How it differs |
|---|---|
| `epoch_ms` and `to_timestamp(n, scale)` applied to a *column* | The literal forms build a timestamp, as DuckDB does. Given a column the same call reads an epoch count back out instead, because the construct-or-extract choice is made from the argument's syntax rather than its type. Use `to_timestamp(n)` for a second count, which is unambiguous and correct for columns. |

## See also

- {doc}`SQL API </api/relational/sql>`: the {py:class}`Session <batcher.Session>`, function registration, and the supported
  SQL surface.
- {doc}`Expressions </user-guide/transform/columns/expressions>`: the DataFrame column language SQL lowers to.
- {doc}`Joins </user-guide/analyze/joins>` and {doc}`Window functions </user-guide/analyze/window-functions>`: the relational
  operations behind `JOIN` and `OVER`.
- {doc}`/cookbook/dataset/verbs/sql_interface`: mixing SQL with DataFrame verbs, as a runnable script.
