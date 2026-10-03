# Migrating to Batcher

You don't have to relearn data engineering to move onto Batcher. This section maps what you know from pandas, Polars, PySpark, DuckDB, Daft, and Ray Data onto Batcher, and shows how to prove the port returns the same rows.

The one concept to learn first: a {py:class}`Dataset <batcher.Dataset>` is *lazy*, like a Polars `LazyFrame`. Verbs build a plan, and a terminal call such as `to_pydict`, `collect`, `count`, or a write runs it.

```python
import batcher as bt
from batcher import col

sales = bt.from_pydict({"region": ["e", "w", "e"], "amount": [10, 20, 30]})
plan = sales.group_by("region").agg(total=col("amount").sum())  # nothing runs yet
print(plan.sort("region").to_pydict())  # runs here
# {'region': ['e', 'w'], 'total': [40, 20]}
```

## Side by side

The same query in the engine you know, and in Batcher:

::::{tab-set}
:::{tab-item} pandas
```python
# docs: skip
out = df[df.amount > 10].groupby("region", as_index=False).agg(total=("amount", "sum"))
```

```python
out = sales.filter(col("amount") > 10).group_by("region").agg(total=col("amount").sum())
print(out.sort("region").to_pydict())
# {'region': ['e', 'w'], 'total': [30, 20]}
```
:::

:::{tab-item} Polars
```python
# docs: skip
out = lf.filter(pl.col("amount") > 10).group_by("region").agg(total=pl.col("amount").sum())
```

```python
out = sales.filter(col("amount") > 10).group_by("region").agg(total=col("amount").sum())
print(out.sort("region").to_pydict())
# {'region': ['e', 'w'], 'total': [30, 20]}
```
:::

:::{tab-item} PySpark
```python
# docs: skip
out = df.filter(F.col("amount") > 10).groupBy("region").agg(F.sum("amount").alias("total"))
```

```python
out = sales.filter(col("amount") > 10).group_by("region").agg(col("amount").sum().alias("total"))
print(out.sort("region").to_pydict())
# {'region': ['e', 'w'], 'total': [30, 20]}
```
:::

:::{tab-item} DuckDB / SQL
```python
# docs: skip
out = duckdb.sql("SELECT region, SUM(amount) AS total FROM sales WHERE amount > 10 GROUP BY region")
```

```python
out = bt.sql("SELECT region, SUM(amount) AS total FROM sales WHERE amount > 10 GROUP BY region", sales=sales)
print(out.sort("region").to_pydict())
# {'region': ['e', 'w'], 'total': [30, 20]}
```
:::
::::

Some vocabulary changes, because Batcher keeps one spelling per operation. Type the name you know and the error names its replacement:

```python
try:
    sales.groupby
except AttributeError as exc:
    print("use `.group_by`" in str(exc))
# True
```

## Coming from

Pick the page to read first by where your code runs today:

![A decision tree from six source systems to the page to read first. pandas, where the shift is eager to lazy, Polars, where the LazyFrame model ports, and PySpark, where there is no SparkSession, all lead to Transforming and collecting, the verb-by-verb table. DuckDB and SQL, where the query often ports, lead to the SQL guide for bt.sql. Daft, where the shift is the UDF contract, leads to the ML pipelines page on batch inference. Ray Data, where there is no object store, leads to the Ray Data port guide. All four first pages then lead to Differences and verification, which covers what Batcher leaves out and how to prove the port returns the same rows. Name-by-name references list every public name for PySpark, Polars, Daft, and Ray Data.](/_static/diagrams/migration_chooser.svg)

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`table;1.1em` pandas
:link: /getting-started/migration/transforming
:link-type: doc
The one shift is eager to *lazy*. Operations build a plan and run on a terminal call. `assign`, `groupby`, and `merge` become `with_columns`, `group_by().agg()`, and `join`.
:::

:::{grid-item-card} {octicon}`code;1.1em` Polars
:link: /getting-started/migration/transforming
:link-type: doc
You already know the `LazyFrame` model. Expressions, {py:meth}`group_by().agg() <batcher.Dataset.group_by>`, {py:meth}`.over(...) <batcher.AggExpr.over>`,
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
The query itself often ports unchanged. {py:func}`bt.sql(...) <batcher.sql>` builds the same plan the
DataFrame verbs build, so you can mix the two.
:::

:::{grid-item-card} {octicon}`file-media;1.1em` Daft
:link: /getting-started/migration/ml-pipelines
:link-type: doc
Both engines are lazy, so the model ports unchanged. The shift is the UDF contract:
`@daft.udf` becomes `@bt.udf`, which declares the `input_columns` it reads.
:::

:::{grid-item-card} {octicon}`stack;1.1em` Ray Data
:link: /getting-started/migration/ray-data
:link-type: doc
The verbs port almost directly. Bulk data never enters the Ray object store, and
distribution is an argument to `collect` rather than a property of the dataset.
:::
::::

## The translation tables

These tables are organized by what you're porting, whichever system it came from.

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

## Name-by-name reference

For PySpark, Polars, Daft, and Ray Data, a generated reference lists every public name with its Batcher spelling and status. It's rendered from the same registry the `AttributeError` guidance reads, so the tables and the error messages agree.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`server;1.1em` PySpark
:link: /getting-started/migration/spark/index
:link-type: doc
`DataFrame`, `Column`, `pyspark.sql.functions`, readers and writers, the session, and the catalog.
:::

:::{grid-item-card} {octicon}`code;1.1em` Polars
:link: /getting-started/migration/polars/index
:link-type: doc
`LazyFrame`, `DataFrame`, expressions and their namespaces, selectors, and readers.
:::

:::{grid-item-card} {octicon}`file-media;1.1em` Daft
:link: /getting-started/migration/daft/index
:link-type: doc
`DataFrame`, `Expression`, `daft.functions`, multimodal and AI functions, and the session.
:::

:::{grid-item-card} {octicon}`stack;1.1em` Ray Data
:link: /getting-started/migration/ray-data/index
:link-type: doc
`Dataset`, expressions and aggregations, readers, preprocessors, and the execution context.
:::
::::

## Verify the port

{py:meth}`equals <batcher.Dataset.equals>` compares results rather than plans, and ignores row order by default:

```python
original = sales.filter(col("amount") > 10).select("region", "amount")
ported = sales.filter("amount > 10")[["region", "amount"]]
print(ported.equals(original))
# True
```

## Rewrite a script with the codemod

`python -m batcher.migrate` rewrites a PySpark, Polars, Daft, or Ray Data script onto Batcher, and back. It needs the `migrate` extra (`pip install "batcher-engine[migrate]"`) but not the compiled engine. Preview a diff, then write it:

```bash
python -m batcher.migrate src/ --from polars --to batcher
python -m batcher.migrate src/ --from polars --to batcher --write --report migrate.json
```

The first command prints a diff and changes nothing. `--write` rewrites in place, `--report` records every call it looked at, and `--check` exits 1 when a file would change, which keeps a converted tree converted in CI.

:::{dropdown} What the codemod leaves for you
It only rewrites what it can prove means the same thing. A call with a different meaning or no equivalent stays as written, with a comment that says why:

```text
# batcher-migrate: Polars `Expr.top_k` differs in Batcher (`Expr.top_k`): Polars top_k(k) returns the k largest values; Batcher's Expr.top_k returns the k most frequent values (to be renamed so top_k means largest)
largest = df.select(pl.col("v").top_k(2))
```

Where the difference is a default, it writes the default out: a Polars `sort` gains `nulls_first=True`, a Ray Data `map_batches` gains `batch_format="numpy"`, and a PySpark write gains `mode="error"`. Search for `batcher-migrate:` to find everything left to port by hand.
:::

## Porting with a coding agent

Each source system has an agent skill that turns these tables into a procedure ending in a verified port: `migrate-from-spark`, `migrate-from-polars-or-pandas`, `migrate-from-duckdb-sql`, `migrate-from-daft`, `migrate-from-ray-data`, and `migrate-from-a-sql-warehouse`. See {doc}`/agents`.

## Reporting a problem

Paste the output of {py:func}`bt.show_versions() <batcher.show_versions>` into the report. {py:func}`bt.versions() <batcher.versions>` returns the same information as a dict.

## See also

- {doc}`/agents`: the migration skills and the verification procedure.
- {doc}`/user-guide/index`: the task-oriented guides for the API these pages map onto.
- {doc}`/getting-started/concepts/lazy`: the lazy, immutable `Dataset` model in one page.
- {doc}`/architecture/overview`: why a `Dataset` is lazy, and what runs where.

```{toctree}
:hidden:
:caption: Port your code

transforming
reading-and-writing
ml-pipelines
ray-data
differences
```

```{toctree}
:hidden:
:caption: Name-by-name reference

spark/index
polars/index
daft/index
ray-data/index
```
