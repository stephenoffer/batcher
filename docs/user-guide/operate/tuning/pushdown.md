# Filter and column pushdown

This page describes which parts of a query Batcher hands to the data source itself, so the
rows and columns you don't need are never read, decoded, or sent over the network. It
covers the predicate shapes that push, what each source backend can express, and why a
predicate that doesn't push still returns the right answer.

## What pushdown is

A filter written against a `Dataset` is an operator in the plan. Before running it, Kyber
looks for a filter sitting directly above a scan and offers that predicate to the source.
A source that understands it applies the filter *where the data lives*: a `WHERE` clause
the database evaluates, a row group a parquet reader skips without decoding, a partition
directory that is never listed.

The saving is not a constant factor. A predicate applied after a result set has crossed
the network has already cost the scan, the transfer, and the driver's memory. Applied in
the database, that work never happens.

Column pushdown works the same way and is why `select` before a wide read matters: a
columnar source reads only the columns the plan still needs.

## Pushdown never changes your results

The engine keeps its own filter operator regardless of what the source did with the
predicate. That single rule is what makes pushdown safe to reason about:

- A source that ignores the predicate entirely returns more rows, and the engine's filter
  removes them.
- A source that applies only part of it returns more rows, and the engine's filter removes
  them.

So a predicate that fails to push is a performance question, never a correctness one. You
never need to check whether a filter "worked".

## What pushes

Batcher translates this subset of a predicate:

| Shape | Example |
|---|---|
| Comparison between a column and a literal | `col("a") > 5` |
| Null tests | `col("a").is_null()`, `col("a").is_not_null()` |
| Set membership | `col("country").is_in(["US", "CA"])` |
| Negation | `~(col("a") == 1)` |
| String prefix, suffix, and substring | `col("s").str.starts_with("US")` |
| A constant | `col("a").is_in([])`, which folds to a constant false |
| `AND` and `OR` of any of the above | `col("a") > 5 & col("b").is_in([1, 2])` |

Anything else stays with the engine. The common cases that don't push are a comparison
between two columns (`col("a") > col("b")`), arithmetic on the filtered column
(`col("a") * 2 > 10`), and a user-defined function.

```python
import batcher as bt

sales = bt.from_pydict(
    {
        "country": ["US", "CA", "US", "MX", "US"],
        "channel": ["web", "web", "store", "web", "store"],
        "amount": [10, 20, 30, 40, 50],
    }
)
sales.write.parquet("sales")

# Both the set membership and the comparison are offered to the parquet reader.
hot = bt.read.parquet("sales").filter(
    bt.col("country").is_in(["US", "CA"]) & (bt.col("amount") > 15)
)
print(hot.select("country", "amount").collect().to_pydict())
```

## Checking what actually pushed

`explain()` annotates a scan with the filter the plan handed it, so you never have to
guess:

```python
plan = bt.read.parquet("sales").filter(
    bt.col("country").is_in(["US", "CA"]) & (bt.col("amount") > 15)
)
print(plan.select("country", "amount").explain())
```

```text
project                         est≈3 (default)
  filter                        est≈3 (default)
    scan                        est≈5 (exact) pushed[country IN (US, CA) AND amount > 15]
```

The `pushed[...]` note is what the plan *offered* the source. Each backend then applies
the part it can express, and the engine keeps its own filter regardless, so a source that
takes only half the predicate still shows the whole offer. A scan with no note received
nothing, which is the case worth looking for:

```python
rolled = (
    bt.read.parquet("sales")
    .group_by("country")
    .agg(total=bt.col("amount").sum())
    .filter(bt.col("total") > 15)
)
print(rolled.explain())
```

```text
filter                          est≈1 (default)
  aggregate                     est≈3 (learned)
    scan                        est≈5 (exact)
```

Here the filter is about summed totals rather than scanned rows, so there is nothing to
push and the scan carries no note. That is correct, not a missed optimization.

The same field is in the JSON form (`explain(format="json")`) as `pushed` on each
operator, for scripting a check into a test.

## Partial translation, and the one case that declines

A conjunction is translated term by term. If one term can't be expressed, the rest are
still pushed, because dropping a term from an `AND` only ever makes the source return
*more* rows, and the engine's filter removes them:

```python
# `amount > 15` pushes; `amount > threshold` (column vs column) does not. The first
# term still reaches the source.
bounded = bt.read.parquet("sales").filter(
    (bt.col("amount") > 15) & (bt.col("amount") > bt.col("amount"))
)
print(bounded.collect().num_rows)
```

A disjunction is all-or-nothing. Dropping one side of an `OR` would *narrow* the filter and
lose rows, so if either side can't be expressed, neither is pushed.

Negation follows from the same rule and is the subtlest case. `NOT` of a widened filter is
a *narrowed* one, so a partially-translated operand under a `NOT` would drop rows that
match. Batcher therefore requires an exact translation beneath a negation and declines
otherwise, which costs some pruning and never a row.

## Row limits

A `limit` sitting directly on a read is offered to the source as a *row cap*, so a
database stops sending after that many rows instead of streaming a whole table you are
about to discard:

```python
preview = bt.read.parquet("sales").limit(3)
print(preview.collect().num_rows)
```

The cap travels the same channel the filter does and carries the same guarantee: it is a
ceiling, never a floor. A source that returns more rows than asked is still correct,
because the engine keeps its own `limit` above the scan. That is why a source which
ignores the cap behaves exactly as it did before.

Two rules decide whether the cap reaches the source at all.

**A limit is a positional prefix, so almost nothing may sit between it and the read.** A
projection passes it through, because the *n*th projected row is the *n*th scanned row.
Everything else blocks it, and `filter` is the case worth understanding: `limit(n)` after
a filter means the first *n* rows *that pass*, while capping the source at *n* would mean
the passing rows of the first *n* — fewer rows, or none at all if the first *n* all fail.
Sorting, aggregation, distinct, and sampling block it for the same kind of reason.

**Only a database whose dialect spells the cap `LIMIT n` receives one.** Batcher emits it
for the PostgreSQL and MySQL families, SQLite, DuckDB, ClickHouse, Snowflake, BigQuery,
Trino, and Redshift. SQL Server and Oracle spell the same thing `TOP` and `FETCH FIRST`,
so they are read uncapped rather than sent a query they would reject, and so is any
connection whose dialect Batcher cannot identify, such as an ODBC DSN. Missing a cap
costs the rows the server would have skipped; sending one a server cannot parse would
turn a working query into an error.

An `offset` is added to the cap rather than subtracted from it, because the engine skips
those rows itself: `limit(10, offset=90)` asks the source for 100 rows.

### Sorted limits

A `sort` between the limit and the read normally blocks the cap, because the first *n*
rows of a sorted relation are not the first *n* of its input. Batcher recovers that case
by pushing the *ordering* along with the cap, so `sort("revenue", descending=True).limit(10)`
reaches a database as `ORDER BY revenue DESC NULLS LAST LIMIT 10` and only ten rows cross
the wire:

```python
top = bt.read.parquet("sales").sort("amount", descending=True).limit(3)
print(top.collect().to_pydict()["amount"])
```

The null placement is always stated explicitly, and that is not a formality. Servers
disagree about where a null sorts by default, so an ordering pushed without it would ask
the server for a different top *n* than Batcher computes. A database whose dialect has no
`NULLS FIRST`/`NULLS LAST` clause, such as MySQL or SQL Server, therefore receives neither
the ordering nor the cap, and the read is unordered and uncapped exactly as before.

Two things still block a sorted cap: a filter between the sort and the read, since the
server's top *n* of the unfiltered relation is not the top *n* of the filtered one, and a
sort on a computed key, which the server cannot name.

## Counting without reading

`count()` is answered without running the query whenever the number is already known: a
file footer, an in-memory table, `limit(n)` over a known count, a global aggregate. When
none of those apply and the source is a database, Batcher asks the server for the count
instead of reading rows to count them:

```python
print(bt.read.parquet("sales").count())
```

This matters more than it sounds. A `COUNT(*)` needs no columns at all, so the projection
that would normally narrow the read is empty, and an empty projection means *every*
column. Counting a warehouse table used to transfer every column of every row to return
one integer.

The same shape rule as row limits applies: a projection cannot change how many rows there
are, so a projected read is still answerable, while a filter, join, aggregate, or limit
above the read is not, and the ordinary path runs.

The count is asked for only when you call `count()`. Planning never spends a query on it,
because the optimizer wants a free estimate and has a better one in its learned
statistics.

## What each backend can express

The predicate is translated into whatever the source actually speaks, so the pushable
subset narrows at the edges. Ordered by how much of the subset each one takes:

| Backend | Sets | Negation | Strings | Notes |
|---|---|---|---|---|
| SQL databases and warehouses | `IN (...)` | `NOT (...)` | `LIKE` | Lists longer than 1,000 members decline, and so does a pattern containing `%`, `_`, `\` or `!` |
| Parquet, ORC, Lance, Delta, Hudi | `is_in` | yes | prefix, suffix, substring | Applied by the Arrow scanner as well as used for row group pruning |
| Native parquet reader | expanded to `OR` of equalities | by De Morgan | no | Lists longer than 64 members decline; temporal literals decline |
| Iceberg | `In` | `Not` | prefix only | Manifest bounds can answer a prefix, but nothing can answer a suffix |
| MongoDB | `$in` | `$nor` | anchored `$regex` | Pattern metacharacters are escaped, so a value matches literally |

Against a database, the column names in a pushed projection or filter are delimited for
that server's dialect: double quotes for PostgreSQL, Snowflake, Oracle and the rest,
backticks for MySQL and BigQuery, square brackets for SQL Server. That is what lets a
column named `order`, `user`, or `end date` be pushed at all. A connection whose dialect
Batcher cannot identify, such as an ODBC DSN, sends names undelimited, so a reserved-word
column there is still best renamed or aliased in your own query.

A prefix filter is worth calling out separately. Before any of this, Kyber rewrites
`starts_with(col, "abc")` into the equivalent range `col >= "abc" AND col < "abd"`, which
every backend already pushes and which a sorted or clustered column can range-prune.
The string translation above is what handles the cases that rewrite declines.

## When a predicate does not push

Even a translatable predicate is only offered when the filter sits directly above the
scan. Two shapes take it out of reach:

- A filter separated from the scan by a pipeline breaker, such as a filter applied after
  an aggregation. That predicate is about the aggregated rows, not the scanned ones.
- A source scanned more than once in one plan, where one of the scans is unfiltered. The
  source is then pre-filtered only by something true of every scan of it, because a
  predicate that suits one branch would silently starve the other.

To get the pruning back, filter as early as you can, directly against the read.

## See also

- {doc}`large-tables`: partition pruning and split planning, which happen before pushdown.
- {doc}`explain-plans`: reading the plan to find which operator costs the time.
- {doc}`/user-guide/moving-data/custom-connectors`: implementing pushdown in your own source.
