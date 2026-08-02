# Migrating to Batcher

This section maps the operations you know from pandas, Polars, PySpark, DuckDB, and Daft
onto their Batcher equivalents, and shows how to prove the port is correct.

Batcher's surface is deliberately close to the libraries you already know, so most of
your vocabulary carries over. Absorb one concept before anything else: a `Dataset` is
*lazy*. Transformations such as `select`, `filter`, `group_by().agg()`, and `join`
build a plan and return a new `Dataset`, and nothing runs until a terminal operation
such as `collect`, `to_arrow`, `to_pandas`, `write`, `count`, or `iter_batches`. This
is the Polars `LazyFrame` model rather than the eager pandas one.

## Coming from

Each card names the single shift that matters most from that system, and links to the
page to read first. The translation tables below are shared across all five, because the
mapping is organized by what you are porting rather than by where it came from.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`table;1.1em` pandas
:link: /getting-started/migration/transforming
:link-type: doc
The one shift is eager to *lazy*. Operations build a plan and run on a terminal
call. `assign`, `groupby`, and `merge` become `with_columns`, `group_by().agg()`, and `join`.
:::

:::{grid-item-card} {octicon}`code;1.1em` Polars
:link: /getting-started/migration/transforming
:link-type: doc
You already know the `LazyFrame` model. Expressions, `group_by().agg()`, `.over(...)`,
and the typed accessors carry over almost verbatim.
:::

:::{grid-item-card} {octicon}`server;1.1em` PySpark
:link: /getting-started/migration/transforming
:link-type: doc
No `SparkSession` and no cluster to start, because it runs in-process. The DataFrame
verbs carry over, and so do the save modes and `MERGE INTO`.
:::

:::{grid-item-card} {octicon}`database;1.1em` DuckDB and SQL
:link: /user-guide/analyze/sql
:link-type: doc
The query itself often ports unchanged. `bt.sql(...)` builds the same plan the
DataFrame verbs build, so you can mix the two.
:::

:::{grid-item-card} {octicon}`file-media;1.1em` Daft
:link: /getting-started/migration/ml-pipelines
:link-type: doc
Both engines are lazy, so that model ports unchanged. The shift is the UDF contract:
`@daft.udf` becomes `@bt.udf`, and declaring `input_columns` wrong is a correctness bug
rather than a slow query, because an undeclared column can be pruned out from under the
function.
:::
::::

## The translation tables

The mapping is split by what you're porting, so each page is one sitting.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`arrow-switch;1.1em` Reading, writing, and interop
:link: /getting-started/migration/reading-and-writing
:link-type: doc
Readers, writers, and the `from_*` / `to_*` bridges to pandas, Polars, Arrow, and torch.
:::

:::{grid-item-card} {octicon}`pencil;1.1em` Transforming and collecting
:link: /getting-started/migration/transforming
:link-type: doc
The verb-by-verb table, terminal operations, and the names that carry over unchanged.
:::

:::{grid-item-card} {octicon}`cpu;1.1em` Batch inference and ML
:link: /getting-started/migration/ml-pipelines
:link-type: doc
Models over batches, GPU pools, the training feed, and resumable writes.
:::

:::{grid-item-card} {octicon}`check-circle;1.1em` Differences and verification
:link: /getting-started/migration/differences
:link-type: doc
What Batcher deliberately does not have, and how to prove the port matches.
:::
::::

## A first port, end to end

The shape of nearly every ported script is the same: read lazily, chain verbs, collect
once at the end.

```python
import batcher as bt
from batcher import col

ds = bt.from_pydict({"city": ["NYC", "LA", "NYC"], "amount": [10, 20, 30]})
out = (
    ds.filter(col("amount") > 10)
    .with_columns(tax=col("amount") * 0.1)
    .group_by("city")
    .agg(total=col("amount").sum(), n=bt.count())
)
print(out.sort("city").to_pydict())
# {'city': ['LA', 'NYC'], 'total': [20, 30], 'n': [1, 1]}
```

Then check it against the original with `equals`, which compares results rather than
plans and ignores row order by default:

```python
original = ds.filter(col("amount") > 10).select("city", "amount")
ported = ds.filter(col("amount") > 10)[["city", "amount"]]
print(ported.equals(original))
# True
```

## Porting with a coding agent

Each source system has an agent skill that turns these tables into a procedure:
`migrate-from-spark`, `migrate-from-polars-or-pandas`, `migrate-from-duckdb-sql`, and
`migrate-from-daft`. Beyond the mappings, each carries the concept shifts that silently
produce wrong or slow results, and a recipe that finishes by proving the ported script
returns the same rows as the original. See {doc}`/agents`.

## Reporting a problem

`bt.show_versions()` prints the Batcher version, the compiled engine version, Python,
the platform, and which optional backends are installed. `bt.versions()` returns the
same information as a dict.

## See also

- {doc}`/agents`: the migration skills, with the failure modes and the
  verification procedure.
- {doc}`/user-guide/index`: the task-oriented guides for the API these pages map onto.
- {doc}`/architecture/overview`: why a `Dataset` is lazy, and what runs where.

```{toctree}
:hidden:

reading-and-writing
transforming
ml-pipelines
differences
```
