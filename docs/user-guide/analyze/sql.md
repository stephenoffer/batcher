# SQL

This page covers writing SQL against Batcher. SQL is not a separate engine here. {py:obj}`bt.sql(query, ...) <batcher.sql>` parses a query into the same logical plan the DataFrame API builds, so it gets the same optimizer and the same Rust data plane, and it returns an ordinary Dataset. You can keep chaining DataFrame operations onto a SQL result, or feed a DataFrame pipeline into SQL, and mix the two freely.

{py:obj}`bt.sql <batcher.sql>` reads DuckDB syntax by default. Pass `dialect=` to read another sqlglot dialect. For a reusable catalog of tables and Python functions, build a {py:obj}`bt.Session <batcher.Session>`, the analogue of a DuckDB connection or a SparkSession. {py:func}`bt.sql <batcher.sql>` and {py:func}`bt.register_function <batcher.register_function>` use a shared default session.

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

Pass the query string, and bind each table name in the query to a Dataset or a pyarrow Table as a keyword argument.

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

A query may use any of the following:

- `SELECT` with column references, scalar expressions, and aggregates
- `WHERE` filters, and `QUALIFY` to filter on a window function
- `GROUP BY` with `HAVING`, plus `ROLLUP`, `CUBE`, and `GROUPING SETS`, with `GROUPING()` to tell a subtotal row from a real one
- `ORDER BY` (including `ORDER BY ALL`), `LIMIT` / `OFFSET`, and the ANSI `FETCH FIRST n ROWS ONLY`
- `INNER`, `LEFT`, `RIGHT`, `FULL`, and `CROSS JOIN` on equi-keys, where an extra non-equi `AND` condition is applied as a filter, plus `NATURAL JOIN` and `ASOF JOIN`
- `UNION` / `INTERSECT` / `EXCEPT`, `WITH` (CTEs), and subqueries
- Column alias lists on a table, subquery, or CTE, such as `FROM (SELECT ...) AS t(a, b)` and `WITH t(a, b) AS (...)`, which rename the relation's columns positionally
- Window functions over any expression, including a computed `PARTITION BY` / `ORDER BY` key such as `date_trunc('month', ts)`, with explicit `ROWS` / `RANGE` / `GROUPS` frames. `RANGE BETWEEN INTERVAL '5' MINUTE PRECEDING` gives a time window
- `CASE` expressions, `CAST`, and `SIMILAR TO`
- `INTERVAL` literals, including compound (`'1 day 3 hours'`), fractional (`'1.5 hours'`), clock (`'04:05:06'`) and abbreviated (`'1 mon'`) forms
- `generate_series(a, b)` / `range(a, b)` in `FROM`, for a generated integer spine
- `UNNEST`, in the `FROM` clause or written directly in the `SELECT` list
- `PIVOT` and `UNPIVOT`, which translate onto {doc}`the reshaping methods <pivoting>`

You can also register Python functions and call them from SQL, and define tables and views with `CREATE`/`DROP`. See {ref}`sessions-tables-and-python-functions`.

## Filters, aggregates, and expressions

The rest of the supported subset reads the way it reads anywhere else. `WHERE` filters:

```python
out = bt.sql("SELECT category, price FROM t WHERE price >= 30 ORDER BY price", t=ds)
print(out.to_pydict())
# {'category': ['a', 'b', 'a', 'c'], 'price': [30.0, 40.0, 50.0, 60.0]}
```

`GROUP BY` aggregates and `HAVING` filters the groups it produced:

```python
out = bt.sql(
    "SELECT category, SUM(price) AS total FROM t "
    "GROUP BY category HAVING SUM(price) > 60 ORDER BY category",
    t=ds,
)
print(out.to_pydict())
# {'category': ['a'], 'total': [90.0]}
```

`CASE` and `CAST` are ordinary expressions in the select list:

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

Both paths build one logical plan, push it through one optimizer, and execute it on one Rust data plane. There is no separate SQL engine.

Every method on the expression accessors is callable from SQL too, under the name of the namespace and the method: `col("s").str.slugify()` is `str_slugify(s)`. See {doc}`/api/relational/expression-accessors` for the naming rules and the full surface.

The other direction works at the level of one expression. {py:func}`bt.sql_expr <batcher.sql_expr>` parses a SQL expression into an `Expr` you can pass to any DataFrame method, and a trailing `AS name` becomes its alias, so a `select` over `sql_expr` strings is Spark's `selectExpr`. {py:func}`bt.call_function <batcher.call_function>` calls a SQL function by name. A string argument is a column name and a number is a literal, which reaches a function that has no Python constructor of its own.

```python
out = ds.select(
    bt.sql_expr("upper(category) AS cat"),
    bt.sql_expr("price / 10 AS tens"),
    rem=bt.call_function("pmod", "price", 25.0, dialect="spark"),
)
print(out.to_pydict())
# {'cat': ['A', 'B', 'A', 'B', 'A', 'C'], 'tens': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 'rem': [10.0, 20.0, 5.0, 15.0, 0.0, 10.0]}
```

An aggregate call becomes an aggregate expression, so `sql_expr` also works inside `agg`:

```python
top = ds.group_by("category").agg(bt.sql_expr("max(price) AS top")).sort("category")
print(top.to_pydict())
# {'category': ['a', 'b', 'c'], 'top': [50.0, 40.0, 60.0]}
```

`sql_expr` refuses a whole query, a subquery and a window function, all of which need a relation. Use `bt.sql` for those.

(sessions-tables-and-python-functions)=

## Sessions, tables, and Python functions

A {py:obj}`bt.Session <batcher.Session>` holds a dialect plus a catalog: the tables and the Python functions you registered. Register a dataset as a table, then query it by name.

```python
s = bt.Session()
s.register("t", ds)
print(s.sql("SELECT COUNT(*) AS n FROM t").to_pydict())
# {'n': [6]}
```

Register a Python function and call it from SQL. A scalar function is vectorized, so it receives an Arrow array rather than a value. It lowers to the same `map_batches` path as the DataFrame API, so Python and SQL share one plan:

```python
import pyarrow.compute as pc

s.register_function("discount", lambda a: pc.multiply(a, 0.9))
print(s.sql("SELECT discount(price) AS net FROM t ORDER BY price").to_pydict())
# {'net': [9.0, 18.0, 27.0, 36.0, 45.0, 54.0]}
```

`CREATE TABLE/VIEW AS` and `DROP TABLE` register and unregister a lazy table in the session. Nothing materializes until a terminal op:

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

A fitted model registers the same way, and a language-model engine alongside it. Both are then called as table functions inside a query. See {doc}`Model and AI functions in SQL <sql-model-functions>`.

## Matching each row to the nearest one

`ASOF JOIN` matches every left row to the single nearest right row rather than to every row satisfying the condition. The `ON` splits into exact-match keys, written as equalities, and one nearest-match key, written as `>=` (look backward) or `<=` (look forward). It is how you attach the most recent quote to each trade, or the prevailing price to each event.

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

`ASOF JOIN` drops a left row that matches nothing. `ASOF LEFT JOIN` keeps it with NULL right columns.

## Shifting a date or timestamp by an interval

`ts + INTERVAL ...` and `ts - INTERVAL ...` shift an instant. The literal is read into three independent components: calendar months, whole days, and exact microseconds. A calendar shift needs all three. A month is not a fixed number of days, and under a time zone neither is a day.

Four spellings are accepted, and they compose:

| Form | Example |
|---|---|
| A count and a unit | `INTERVAL 3 DAY`, `INTERVAL '2 hours'` |
| Several terms, added together | `INTERVAL '1 day 3 hours'`, `INTERVAL '2 years 3 months'` |
| A fractional count | `INTERVAL '1.5 hours'`, `INTERVAL '2.5 weeks'` |
| A clock, with no unit words | `INTERVAL '04:05:06'` |

Units may be written in full or abbreviated the way PostgreSQL abbreviates them: `y`/`yr`/`year`, `mon`/`month`, `quarter`, `decade`, `century`, `millennium`, `w`/`week`, `d`/`day`, `h`/`hr`/`hour`, `m`/`min`/`minute`, `s`/`sec`/`second`, `ms`/`millisecond`, `us`/`microsecond`. Bare `m` is a minute and `mon` is a month, as in PostgreSQL.

```python
import datetime as dt

runs = bt.from_pydict({"started": [dt.datetime(2024, 1, 31, 22, 30)]})
print(
    bt.sql(
        """
        SELECT started + INTERVAL '1 day 3 hours'   AS shifted,
               started + INTERVAL '1.5 hours'       AS half_shift,
               started - INTERVAL '1 mon'           AS last_month
        FROM runs
        """,
        runs=runs,
    ).to_pydict()
)
# {'shifted': [datetime.datetime(2024, 2, 2, 1, 30)],
#  'half_shift': [datetime.datetime(2024, 2, 1, 0, 0)],
#  'last_month': [datetime.datetime(2023, 12, 31, 22, 30)]}
```

A fractional count spills into the next finer component rather than rounding, which is what keeps `INTERVAL '1.5 months'` meaning "one month and fifteen days" instead of a flat forty-five days that would drift across a month boundary. `INTERVAL '0.25 days'` is six hours for the same reason.

A shift by whole days or whole months keeps a `DATE` a `DATE`. One carrying a time component widens to `TIMESTAMP`, because a `DATE` cannot hold the hours. See {ref}`deliberate-differences` for how that compares to DuckDB.

## Measuring the gap between two timestamps

`date_diff(unit, start, end)` reports how many `unit` boundaries lie between two instants. The units run from `microsecond` through `year`, and both `DATE` and `TIMESTAMP` inputs work.

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

Both rows above are worth reading twice, because they show the rule. `date_diff` counts boundary crossings, not elapsed time. The first row spans ten minutes and reports one hour, because a clock hour ticks over between 9:55 and 10:05. The second spans fifty-nine minutes and reports zero hours, because none does. If you want elapsed time instead, subtract the two timestamps and read the duration, or take the difference of `epoch(ts)` values.

`week` is the one unit that does not follow that rule. DuckDB defines it as the number of whole seven-day spans, truncated toward zero, so a Thursday to the following Monday is `0` even though a calendar week boundary falls between them. Batcher matches DuckDB here.

## Window functions over an expression

A window function's argument is an ordinary expression, not only a column name, so a running total of a computed value or a conditional running count is written directly. `lag` and `lead` take a third argument that fills the rows whose offset falls outside the partition.

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

The default fills only the rows that have no row at that offset. A NULL the column genuinely holds inside the partition stays NULL.

## Duplicate output names

SQL lets a `SELECT` list emit the same name twice, most often when a join projects a key from both sides. A Dataset is keyed by column name, so the second one is suffixed: `id`, then `id_1`. DuckDB assigns the same names when it has to make a result unique.

```python
left = bt.from_pydict({"id": [1, 2], "v": [10, 20]})
right = bt.from_pydict({"id": [1, 2], "w": [7, 8]})

out = bt.sql("SELECT l.id, r.id, l.v FROM l JOIN r ON l.id = r.id ORDER BY l.id", l=left, r=right)
print(out.columns)
# ['id', 'id_1', 'v']
```

## Requirements and limitations

Constructs Batcher rejects rather than approximates. Each raises a clear error, because answering with a different row set and reporting success is the failure nothing downstream can detect.

| Construct | Why, and what to write instead |
|---|---|
| `LIMIT n PERCENT`, `FETCH ... WITH TIES` | Both need a cardinality measured before the limit applies. Use a plain row count. |
| `POSITIONAL JOIN` | Row position is not defined for a Batcher relation, which is morsel-parallel and may span nodes. Join on a key, or number both sides with `row_number() OVER (ORDER BY ...)` first. |
| `ASOF JOIN` on a strict `>` or `<` | The nearest-match key is inclusive. Use `>=` or `<=`. |
| A negative list-slice bound, `a[-2:]` | Counts back from the end in DuckDB, while the underlying slice clamps to the start. Index from the front, or reverse the list first. |
| A correlated subquery whose correlation is an inequality | An equality correlation decorrelates to a join and is supported. An inequality one is not. |
| An inequality quantified subquery, `x > ALL (...)` or `x >= ANY (...)` | Only the equality forms have a faithful rewrite: `= ANY` is `IN` and `<> ALL` is `NOT IN`, by definition. The tempting rewrite of `x > ALL (S)` as `x > (SELECT max(c) FROM S)` is wrong when `S` holds a NULL, because `max` skips it. The rewrite then answers TRUE where SQL says UNKNOWN, which is a silently wrong row rather than an error. Write the `max`/`min` form yourself, with `AND NOT EXISTS (SELECT 1 FROM S WHERE c IS NULL)` to keep the NULL case. |
| `time_bucket` with a width that doesn't divide a day evenly | Buckets start from the Unix epoch and DuckDB starts them from 2000-01-03, so a width such as `INTERVAL 2 DAY` would put every boundary on a different instant. Use a width that divides a day, such as `1 DAY`, `6 HOUR`, or `15 MINUTE`, or `date_trunc` for calendar buckets. |
| Two `UNNEST` calls in one `SELECT` list | SQL zips them into one relation. Unnest one list per query, or use `FROM t, UNNEST(...)` for each. |

One construct succeeds and returns the same values under a different type. Adding a date-granular interval to a `DATE` keeps a `DATE`, where DuckDB and Postgres widen to `TIMESTAMP`:

```python
out = bt.sql("SELECT DATE '2024-01-01' + INTERVAL 1 DAY AS d")
print(out.schema.field("d").type, out.to_pydict()["d"][0])
# date32[day] 2024-01-02
```

Batcher widens only when the interval carries a time component, so `+ INTERVAL 2 HOUR` does give a `TIMESTAMP`. This matches Spark and keeps a date column usable as a date. Cast explicitly if you need DuckDB's type. The row values are identical either way.

Descending list sorts agree with DuckDB, NULLs included. `list_reverse_sort` lowers to `.list.sort(descending=True)`, a kernel of its own rather than `sort().reverse()`. Ascending puts NULLs last, so reversing would lift them to the front, where DuckDB keeps them at the back. Both spellings return `[2, 1, NULL]` for `[1, NULL, 2]`.

(deliberate-differences)=
### Deliberate differences

Three results differ from DuckDB's on purpose. Each is a case where Batcher answers what the value means rather than what DuckDB's implementation happens to produce, so expect the difference and don't report it as a defect.

| Construct | How it differs |
|---|---|
| `corr(y, x)` where either column is flat | Batcher returns NULL, DuckDB returns NaN. With no variance there is no correlation to report, and NULL is the SQL spelling of "no value" that every aggregate here already uses for an undefined result. PostgreSQL answers NULL too. `regr_r2` follows the same rule, so the family stays consistent. |
| `jaro_similarity` and `jaro_winkler_similarity` on non-ASCII text | Batcher measures in *characters*, DuckDB in bytes. `jaro_similarity('ünïcödé', 'abc')` is 0.492 here and 0.475 there, because DuckDB counts the seven-character string as eleven bytes. The character reading is the one the algorithm is defined on. ASCII arguments agree exactly. |
| `DATE + INTERVAL` with a calendar unit | Batcher returns a DATE, DuckDB widens to a TIMESTAMP. The calendar value is the same. Cast explicitly if you need DuckDB's type. |

## See also

- {doc}`Model and AI functions in SQL <sql-model-functions>`: `ML_PREDICT`, `AI_GENERATE`, and `AI_EXTRACT`.
- {doc}`SQL API </api/relational/sql>`: the {py:class}`Session <batcher.Session>`, function registration, and the supported SQL surface.
- {doc}`Expressions </user-guide/transform/columns/expressions>`: the DataFrame column language SQL lowers to.
- {doc}`Joins </user-guide/analyze/joins>` and {doc}`Window functions </user-guide/analyze/window-functions>`: the relational operations behind `JOIN` and `OVER`.
- {doc}`/cookbook/dataset/verbs/sql_interface`: mixing SQL with DataFrame verbs, as a runnable script.
