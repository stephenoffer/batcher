# Inspect a dataset

This page covers the tools you use to look at a dataset: what each one answers, and what it costs to ask. Some read only the plan, some read a few rows, and some make a full pass over the data, so picking the right one keeps a quick look quick on a large table.

## Setup

The examples on this page run against a small in-memory weather table:

```python
from decimal import Decimal

import pyarrow as pa

import batcher as bt

ds = bt.from_arrow(
    pa.table(
        {
            "city": ["oslo", "lima", "oslo", "cairo"],
            "temp": [-3.5, 21.0, None, 30.5],
            "rain": [1.2, 0.0, 3.4, 0.1],
            "price": pa.array(
                [Decimal("2.50"), Decimal("1.75"), Decimal("2.25"), Decimal("1.00")],
                type=pa.decimal128(6, 2),
            ),
        }
    )
)
```

## What each tool costs

The following table lists the inspection tools from cheapest to most expensive, with the question each one answers:

| Tool | Answers | What runs |
|---|---|---|
| `columns`, `schema`, `dtypes` | What columns are there, and of what type? | Nothing for `columns`. `schema` reads a scan's source schema, or runs a derived plan on zero rows. |
| `show`, `glimpse` | What do the first rows look like? | One bounded head slice. |
| `info` | How many rows, and how many values per column? | A row count and a null count, often answered from metadata. |
| `null_count` | How many values are missing? | One aggregate, lazily. |
| `profile` | How complete and how varied is each column? | One aggregate pass with a distinct-count sketch per column. |
| `describe` | What is the distribution of each numeric column? | One aggregate pass, including exact quantiles. |
| `corr_matrix`, `cov_matrix` | How do the numeric columns move together? | One aggregate pass over every pair of columns. |
| `approx_quantile` and friends | Roughly where is the median or the p99? | A learned sketch when one applies, else a stream over one column. |
| `explain`, `stats` | What will the query do, and where did the time go? | `explain()` runs nothing. `explain(analyze=True)` and `stats()` run the query. |

## Names and types

{py:attr}`ds.columns <batcher.Dataset.columns>` is always free: it's a property of the plan. {py:attr}`ds.schema <batcher.Dataset.schema>` and {py:attr}`ds.dtypes <batcher.Dataset.dtypes>` add the Arrow types. For a plain scan they come straight from the source. For a derived plan, Batcher resolves the types by running the plan on zero rows, so no data is read either way.

```python
assert ds.columns == ["city", "temp", "rain", "price"]
assert ds.schema.field("price").type == pa.decimal128(6, 2)
assert str(ds.dtypes[1]) == "double"
```

## Look at the rows

{py:meth}`ds.show() <batcher.Dataset.show>` prints the first rows as a table. The `limit` is pushed into the plan, so previewing a huge source reads only enough of it to fill the screen. A negative `limit` raises `PlanError`.

```python
ds.show(2)
```

{py:meth}`ds.glimpse() <batcher.Dataset.glimpse>` prints the same head slice transposed, one line per column, which reads better on a wide table:

```python
ds.glimpse()
```

```text
Dataset: 4 columns
$ city  <string          > 'oslo', 'lima', 'oslo', 'cairo'
$ temp  <double          > -3.5, 21.0, None, 30.5
$ rain  <double          > 1.2, 0.0, 3.4, 0.1
$ price <decimal128(6, 2)> Decimal('2.50'), Decimal('1.75'), Decimal('2.25'), Decimal('1.00')
```

{py:meth}`ds.info() <batcher.Dataset.info>` prints the row count, and each column's type and non-null count. It runs a `count` and a `null_count`, never a pass over the values themselves, and closes with an *estimated* size derived from the types and the row count.

```python
ds.info()
```

## Summarize the columns

{py:meth}`ds.null_count() <batcher.Dataset.null_count>` returns a one-row dataset of missing values per column. It's lazy, so nothing runs until you collect it. It counts Arrow nulls only. A floating-point NaN is a value, so unlike pandas `isnull().sum()` it isn't counted. Count NaN with `col(c).is_nan()` when you need it.

```python
assert ds.null_count().to_pydict() == {"city": [0], "temp": [1], "rain": [0], "price": [0]}
```

{py:meth}`ds.profile() <batcher.Dataset.profile>` returns one row per column with the non-null count, the null count and fraction, and an approximate distinct count from a HyperLogLog sketch. It's one aggregate pass however wide the table is, which makes it the quick completeness check before a load. It describes the data. It doesn't time the query, which is what `stats()` is for.

```python
profile = ds.profile().to_pydict()
assert profile["column"] == ["city", "temp", "rain", "price"]
assert profile["null_count"] == [0, 1, 0, 0]
assert profile["approx_distinct"][0] == 3
```

{py:meth}`ds.describe() <batcher.Dataset.describe>` returns summary statistics with a `statistic` label column and one Float64 column per input column. Integer, float, and decimal columns get the count, null count, mean, standard deviation, minimum, the requested percentiles, and maximum. Every other column gets the count and null count only, with the other cells null. It's one aggregate pass, but the percentiles are exact, which is the costly part on a large column.

```python
d = ds.describe(percentiles=(0.5,)).to_pydict()
assert d["statistic"] == ["count", "null_count", "mean", "std", "min", "50%", "max"]
assert d["temp"][0] == 3.0
assert d["price"][d["statistic"].index("max")] == 2.5
assert d["city"][2] is None
```

An input column named `statistic` would overwrite the labels, so `describe()` raises `PlanError` for it. Rename the column first:

```python
labelled = bt.from_pydict({"statistic": [1, 2], "y": [3, 4]})
summary = labelled.rename({"statistic": "statistic_"}).describe().to_pydict()
assert summary["statistic"][0] == "count"
```

## Relate the columns

{py:meth}`ds.corr_matrix() <batcher.Dataset.corr_matrix>` returns the Pearson correlation of every pair of numeric columns, and {py:meth}`ds.cov_matrix() <batcher.Dataset.cov_matrix>` the sample covariance. Both compute every pair in a single pass and return a symmetric matrix with a `column` label column. The correlation diagonal is exactly `1.0`, or `None` for a constant column, which has no correlation. The covariance diagonal is each column's variance. Rows where either column of a pair is null are left out of that pair.

```python
m = ds.corr_matrix(["rain", "price"]).to_pydict()
assert m["column"] == ["rain", "price"]
assert m["rain"][0] == 1.0 and m["price"][1] == 1.0
assert m["rain"][1] == m["price"][0]

c = ds.cov_matrix(["rain", "price"]).to_pydict()
assert c["rain"][1] == c["price"][0]
```

Pass no columns to use every numeric column. Naming a non-numeric column, naming a column twice, or correlating a column that is itself named `column` raises `PlanError`.

## Approximate answers

{py:meth}`ds.approx_quantile() <batcher.Dataset.approx_quantile>`, {py:meth}`ds.approx_median() <batcher.Dataset.approx_median>`, {py:meth}`ds.approx_percentile() <batcher.Dataset.approx_percentile>`, and {py:meth}`ds.approx_count_distinct() <batcher.Dataset.approx_count_distinct>` trade precision for cost. Each one first looks for a sketch a previous run recorded, and answers from it with no scan. When there is none, it streams only the target column through a mergeable sketch, which is far cheaper than the sort an exact quantile needs.

The recorded quantile grid describes the *source* column. It answers only when your dataset passes that column straight through from a single source, through a `select`, a `rename`, or a sort. After a filter or a computed column, the grid would describe the wrong values, so the terminal streams instead.

The quantile terminals return `None` for an empty column, and for any column that isn't integer, float, or decimal, such as a timestamp, a date, or a boolean. That check reads the schema, so it costs nothing.

```python
median = ds.approx_median("rain")
assert 0.0 <= median <= 3.4
assert ds.approx_quantile("price", 0.5) is not None
assert ds.approx_median("city") is None
```

## Answer from metadata first

Many of the questions above are already answered by a Parquet footer, a table manifest, or an in-memory relation's own statistics. `count()`, `min()`, and `null_count()` read those automatically. {py:obj}`ds.meta <batcher.Dataset.meta>` lets you ask the metadata layer directly, and `ds.meta.approx` is the free-or-nothing probe that never executes. See {doc}`metadata-shortcuts` for the rules.

```python
assert ds.meta.shape() == (4, 4)
assert ds.meta.schema.is_numeric("price")
```

## Inspect the query

The tools above inspect the data. To inspect the query that produces it, use {py:meth}`ds.explain() <batcher.Dataset.explain>` for the planned operator tree with row estimates, which runs nothing. `explain(analyze=True)` runs the query and prints each operator's estimate beside its measured rows and time. {py:meth}`ds.stats() <batcher.Dataset.stats>` returns the same measurements as objects you can assert on, and `explain(format="json")` returns them as a JSON string for tooling.

```python
query = ds.group_by("city").agg(rain=bt.col("rain").sum())
aggregate = next(op for op in query.stats().ops if op.kind == "aggregate")
assert aggregate.rows_out == 3
assert "actual=" in query.explain(analyze=True)
```

{doc}`/user-guide/operate/tuning/explain-plans` walks through reading both outputs.

{py:meth}`ds.lineage() <batcher.Dataset.lineage>` maps each output column to the source columns it was computed from, which is how you trace where a sensitive column went. It reads the plan and runs nothing.

```python
derived = ds.with_columns(wet=bt.col("rain") > 1.0)
assert "wet" in derived.lineage()
```

See {doc}`/user-guide/trust/governance` for using lineage with column tags.

## On a cluster

`null_count()` is lazy, so you choose where it runs when you collect it:

```python
# docs: skip
ds.null_count().collect(distributed=True, num_workers=4)
```

`describe()`, `profile()`, `corr_matrix()`, and `cov_matrix()` execute their aggregate pass when you call them and return a small in-memory result. That pass, the scalar terminals such as `count()` and `min()`, and the `approx_*` terminals take no `distributed=` argument. They route with `distributed="auto"`, which distributes when Batcher is connected to a multi-node Ray cluster and the input is large enough to pay for the fan-out.

The `distributed.mode` session option pins that choice for every terminal that doesn't pass `distributed=` itself. It accepts `"auto"`, `"always"`, which forces the Ray path and starts a local Ray when none is running, and `"never"`, which stays single-node. An explicit `distributed=True` or `distributed=False` on a call always wins.

```python
# docs: skip
import batcher.config

with batcher.config.option_context("distributed.mode", "always"):
    summary = ds.describe()
    p99 = ds.approx_percentile("rain", 99)
```

Exact statistics are the same on one node or many, up to the last bits of floating-point sums, whose order depends on the partitioning. An approximate quantile can land on a slightly different value within the sketch's accuracy, because merging quantile sketches depends on how the data was split.

## See also

- {doc}`metadata-shortcuts`: which questions are answered from a footer or manifest, and the `ds.meta` namespace.
- {doc}`aggregations`: the exact and approximate aggregates these tools are built from.
- {doc}`/user-guide/operate/tuning/explain-plans`: reading `explain()`, `explain(analyze=True)`, and `stats()`.
- {doc}`/user-guide/trust/governance`: column-level lineage and tagging.
- {doc}`/user-guide/trust/data-quality`: turn what you find here into enforced checks.
