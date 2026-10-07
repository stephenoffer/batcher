# SQL parameters, scripts, and errors

This page covers the parts of running SQL that sit around the query itself: binding values into it, checking it before it runs, running several statements as a script, and reading the errors Batcher raises when a query can't run. The query language is on {doc}`sql`.

```python
import batcher as bt

orders = bt.from_pydict(
    {
        "id": [1, 2, 3, 4],
        "customer": ["ana", "o'brien", "bo", "ana"],
        "amount": [10.0, 25.0, 40.0, 5.0],
    }
)
```

## Bind values with params

Pass values with `params=` instead of formatting them into the query string. Batcher parses the query first and then puts each value into the parsed query as a typed literal, so a value is never read as SQL. A customer name with a quote in it is just a name:

```python
out = bt.sql(
    "SELECT id, amount FROM orders WHERE customer = ? AND amount > ? ORDER BY id",
    orders=orders,
    params=["o'brien", 20],
)
print(out.to_pydict())
# {'id': [2], 'amount': [25.0]}
```

A `?` placeholder takes the next value from a list, in the order the placeholders are written. `$1`, `$2` take the value at that position, so one value can be used twice. `$name` takes a value from a mapping, and dialects that spell a named parameter `:name`, such as `spark`, take a mapping the same way:

```python
named = bt.sql(
    "SELECT id FROM orders WHERE customer = $who OR amount >= $floor ORDER BY id",
    orders=orders,
    params={"who": "bo", "floor": 25},
)
print(named.to_pydict())
# {'id': [2, 3]}

numbered = bt.sql(
    "SELECT id FROM orders WHERE amount BETWEEN $1 AND $1 * 3 ORDER BY id",
    orders=orders,
    params=[10],
)
print(numbered.to_pydict())
# {'id': [1, 2]}
```

A placeholder works anywhere a literal does, including `LIMIT ?` and the values of an `INSERT`. `params=` is the same on all three entry points: {py:func}`bt.sql <batcher.sql>`, {py:meth}`Session.sql <batcher.Session.sql>`, and {py:meth}`Dataset.sql <batcher.Dataset.sql>`.

Each Python type binds to the SQL type DuckDB's own parameter binding gives it. The table lists them in the order Batcher checks them.

| Python value | SQL value |
| --- | --- |
| `None` | `NULL` |
| `bool` | `BOOLEAN` |
| `int` | an integer literal |
| `decimal.Decimal` | `DECIMAL(p, s)`, with the precision and scale the value has |
| `float` | `DOUBLE`, including `nan` and `inf` |
| `str` | a string literal |
| `datetime.datetime` | `TIMESTAMP`, or `TIMESTAMPTZ` when it carries a time zone |
| `datetime.date` | `DATE` |
| `datetime.time` | `TIME` |
| `bytes` | `BLOB` |

A value of any other type raises {py:exc}`PlanError <batcher.PlanError>`, and so do `bytes` that aren't valid UTF-8, because the engine has no literal for arbitrary binary data. Bind those in a table column instead. A mismatch between the placeholders and the values also raises `PlanError` before anything runs: a missing value, an unused one, or a query that mixes `?` with `$name`.

Tables bind the same way on all three entry points too. Pass them as keywords, or as a `{name: table}` mapping in the second position when a name isn't a valid Python identifier or collides with a keyword such as `dialect` or `params`:

```python
out = bt.sql('SELECT count(*) AS n FROM "order-lines"', {"order-lines": orders})
print(out.to_pydict())
# {'n': [4]}
```

## Check a query without running it

{py:obj}`bt.sql <batcher.sql>` returns a lazy Dataset, and its schema comes from the plan rather than from running it. Reading `.schema` is therefore a way to check a query: it parses and translates the SQL, resolves every column and function, and reports the result columns, without reading a row.

```python
checked = bt.sql("SELECT customer, sum(amount) AS total FROM orders GROUP BY customer", orders=orders)
print(checked.schema.names)
# ['customer', 'total']
```

A few constructs do run while the statement is translated, so for them the check does real work: a `WITH RECURSIVE` CTE runs to its fixpoint, an uncorrelated scalar subquery runs and is inlined as a literal, an uncorrelated `EXISTS` or `NOT IN` runs a `LIMIT 1` probe, and a statement that writes a catalog table writes it. {py:meth}`Session.sql <batcher.Session.sql>` lists them in full.

## Named arguments

A function call can name an argument with `=>`. Batcher honours a named argument or refuses the call, and never drops one. `round` takes `mode =>`, the same tie rule {py:meth}`Expr.round <batcher.Expr.round>` takes, so `'half_to_even'` is DuckDB's `round_even`:

```python
ties = bt.from_pydict({"x": [0.5, 1.5, 2.5]})
out = bt.sql("SELECT round(x, 0, mode => 'half_to_even') AS r FROM ties", ties=ties)
print(out.to_pydict())
# {'r': [0.0, 2.0, 2.0]}
```

The functions SQL reaches by their Python names, such as the `str_*` and `st_*` families, take a named argument for any keyword parameter the Python function has. `str_contains(s, '.', literal => false)` is `bt.col("s").str.contains(".", literal=False)`. A name the function doesn't have raises {py:exc}`SQLUnsupportedError <batcher.SQLUnsupportedError>` that names the argument.

## Run a script

{py:meth}`Session.execute_script <batcher.Session.execute_script>` runs a string of `;`-separated statements in order and returns one Dataset per statement. A table one statement creates is visible to the statements after it:

```python
session = bt.Session()
results = session.execute_script(
    """
    CREATE TABLE big AS SELECT * FROM orders WHERE amount > 20;
    INSERT INTO big VALUES (5, 'cy', 99.0);
    SELECT count(*) AS n FROM big
    """,
    orders=orders,
)
print(results[-1].to_pydict())
# {'n': [3]}
```

A script is **not atomic**. Each statement stays applied as soon as it has run, so when one fails, the statements before it remain in place. The error raised is the failing statement's own, with a note saying which statement failed and how many completed before it. A script that doesn't parse raises before any statement runs.

## SQL errors

Two exception types describe a query that can't run, and both are a {py:exc}`PlanError <batcher.PlanError>`. {py:exc}`SQLSyntaxError <batcher.SQLSyntaxError>` means the text doesn't parse in the session's dialect. {py:exc}`SQLUnsupportedError <batcher.SQLUnsupportedError>` means it parses but uses a construct, function, or named argument Batcher doesn't translate. It's also a `NotImplementedError`, the type those refusals raised before it existed. Both carry `line` and `column` (1-based) and `start` and `end` (0-based character offsets) when Batcher knows where the problem is, and `None` when it doesn't:

```python
try:
    bt.sql("SELECT id,\n       amout_total(amount) FROM orders", orders=orders)
except bt.SQLUnsupportedError as err:
    print(err.line, err.column)
# 2 8
```

## See also

- {doc}`sql`: the SQL Batcher runs, and how it mixes with the DataFrame API.
- {doc}`/api/relational/sql`: the SQL reference, with the full list of supported constructs.
- {doc}`/api/relational/sessions-and-catalogs`: sessions, including read-only sessions and scoping one to a block.
- {doc}`/api/operations/exceptions`: every error type Batcher raises.
