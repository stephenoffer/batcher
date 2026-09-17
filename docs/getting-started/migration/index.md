# Migrating to Batcher

You don't have to relearn data engineering to move onto Batcher. This section maps the operations you know from pandas, Polars, PySpark, DuckDB, Daft, and Ray Data onto their Batcher spellings, and each page ends by showing how to prove the port returns the same rows.

Batcher keeps one spelling for each operation, so some of your vocabulary changes: `groupby`, `merge`, `fillna` and `drop_duplicates` are `group_by`, `join`, `fill_null` and `distinct` here. You don't have to memorize the difference. A familiar name Batcher spells differently raises an error that names its replacement.

One concept matters before anything else. A {py:class}`Dataset <batcher.Dataset>` is *lazy*. Transformations such as `select`, `filter`, `group_by().agg()`, and `join` build a plan and return a new `Dataset`. Nothing runs until a terminal operation such as `collect`, `to_arrow`, `to_pandas`, `write`, `count`, or `iter_batches`. If you know the Polars `LazyFrame`, you already know this model.

## Coming from

Each card names the one shift that matters most from that system and links to the page to read first.

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
Both engines are lazy, so that model ports unchanged. The shift is the UDF contract:
`@daft.udf` becomes `@bt.udf`, and declaring `input_columns` wrong is a correctness bug
rather than a slow query, because an undeclared column can be pruned out from under the
function.
:::

:::{grid-item-card} {octicon}`stack;1.1em` Ray Data
:link: /getting-started/migration/ray-data
:link-type: doc
The verbs port almost directly. The shift is that bulk data never enters the Ray object
store, so there is no object store to size and no spill storm to diagnose, and
distribution is an argument to `collect` rather than a property of the dataset.
:::
::::

## The translation tables

The tables are shared across all six source systems, because they're organized by what you're porting rather than where it came from. Each page is short enough to read in one sitting.

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

For PySpark, Polars, Daft, and Ray Data, a generated reference lists every public name with its Batcher spelling, whether the two engines compute the same thing, and what is missing or different when they don't. Each row reads in both directions, and each section ends with a page that maps Batcher spellings back to the other engine. The pages are rendered from the same migration registry the `AttributeError` guidance reads, so the tables and the error messages agree.

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

## A first port, end to end

Nearly every ported script has the same shape. Read lazily, chain verbs, and collect once at the end.

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

Then check it against the original with {py:meth}`equals <batcher.Dataset.equals>`. It compares results rather than plans, and ignores row order by default:

```python
original = ds.filter(col("amount") > 10).select("city", "amount")
ported = ds.filter(col("amount") > 10)[["city", "amount"]]
print(ported.equals(original))
# True
```

## Rewrite a script with the codemod

`python -m batcher.migrate` rewrites a PySpark, Polars, Daft, or Ray Data script onto Batcher, and a Batcher script back onto any of the four. It reads the same migration registry these pages are built from, needs the `migrate` extra (`pip install "batcher-engine[migrate]"`), and doesn't need the compiled engine, so it runs anywhere Python does.

To translate a Polars project and see the result before anything changes, run the following:

```bash
python -m batcher.migrate src/ --from polars --to batcher
python -m batcher.migrate src/ --from polars --to batcher --write --report migrate.json
```

The first command prints a unified diff and leaves the files alone. The second rewrites them in place and writes every call it looked at to `migrate.json`, with a count of what it rewrote and what it left. `--check` exits with status 1 when any file would change, which is how you keep a converted tree converted in CI. The reverse direction is `--from batcher --to polars`, and exactly one side of every direction is `batcher`.

The codemod only rewrites what it can prove means the same thing. When a name has a different meaning in Batcher, has no Batcher equivalent yet, or sits on an object it can't identify, the call stays as written and the line before it gets a comment that says why:

```text
# batcher-migrate: Polars `Expr.top_k` differs in Batcher (`Expr.top_k`): Polars top_k(k) returns the k largest values; Batcher's Expr.top_k returns the k most frequent values (to be renamed so top_k means largest)
largest = df.select(pl.col("v").top_k(2))
```

Where the difference is a default, it writes the default out. A Polars `sort` gains `nulls_first=True`, a Ray Data `map_batches` gains `batch_format="numpy"`, and a PySpark write gains `mode="error"`. Search the rewritten tree for `batcher-migrate:` to find everything left for you to port by hand. The foreign directions rewrite `.py` files only.

## Porting with a coding agent

Each source system has an agent skill that turns these tables into a procedure:
`migrate-from-spark`, `migrate-from-polars-or-pandas`, `migrate-from-duckdb-sql`,
`migrate-from-daft`, `migrate-from-ray-data`, and `migrate-from-a-sql-warehouse`. Beyond the mappings, each skill carries the concept shifts that silently produce wrong or slow results. Each one finishes by proving the ported script returns the same rows as the original. See {doc}`/agents`.

## Reporting a problem

{py:func}`bt.show_versions() <batcher.show_versions>` prints the Batcher version, the compiled engine version, Python,
the platform, and which optional backends are installed. {py:func}`bt.versions() <batcher.versions>` returns the
same information as a dict.

## See also

- {doc}`/agents`: the migration skills, with the failure modes and the
  verification procedure.
- {doc}`/user-guide/index`: the task-oriented guides for the API these pages map onto.
- {doc}`/getting-started/concepts/lazy`: the lazy, immutable `Dataset` model in one page.
- {doc}`/architecture/overview`: why a `Dataset` is lazy, and what runs where.

```{toctree}
:hidden:

reading-and-writing
transforming
ray-data
ml-pipelines
differences
spark/index
polars/index
daft/index
ray-data/index
```
