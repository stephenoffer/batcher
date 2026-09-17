# Transform

This section covers reshaping a dataset in Batcher. It splits along the line the API itself draws: the verbs that decide which rows survive and in what order, and the expression language that decides what a column contains.

Both halves are lazy. Every call returns a new dataset and nothing executes until a terminal operation, so a whole chain reaches the optimizer as one plan. That is what lets Batcher move a filter you wrote last down into the Parquet scan, drop the columns nothing downstream reads, and fuse a run of arithmetic into one compiled pass. An expression is never a Python callback. It lowers to a Rust expression tree that runs over whole Arrow batches, and the Cranelift JIT compiles the arithmetic it supports, so column work on a billion rows never becomes a billion Python calls. When you truly need your own Python, a batch UDF hands it whole Arrow batches, zero-copy, across every core or a cluster.

```python
import batcher as bt

people = bt.from_pydict({"name": ["Ann", "bob", "CARL"], "age": [34, None, 51]})
adults = (
    people.filter(bt.col("age") >= 18)
    .with_columns(name=bt.col("name").str.to_case("title"), decade=(bt.col("age") / 10).floor())
    .sort("age", descending=True)
)
print(adults.to_pydict())
# {'name': ['Carl', 'Ann'], 'age': [51, 34], 'decade': [5.0, 3.0]}
```

## Two halves of one chain

Each verb has one job. `filter`, `distinct`, and `sample` decide which rows survive, `sort` decides their order, and `select` and `with_columns` decide what the columns hold. None of them computes a value on its own. The expression you pass to `filter`, `select`, or `with_columns` does that, and the column language is how you write one.

![A matrix of six verbs against three properties of a table, beside a panel on the column language. filter, distinct and sample change which rows survive and leave row order and columns alone. sort sets the row order and changes neither which rows survive nor the columns. select and with_columns change what the columns hold and leave the rows and their order alone. Every call returns a new lazy Dataset, and nothing runs until a result is asked for. The column language panel shows three expressions: bt.col("age") >= 18, bt.col("name").str.to_case("title"), and (bt.col("age") / 10).floor(). An expression lowers to a Rust expression tree that runs over whole Arrow batches, never one Python call per row, and filter, select and with_columns evaluate it. A batch UDF is the escape hatch, running your Python over whole Arrow batches.](/_static/diagrams/transform_rows_vs_columns.svg)

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`filter;1.1em` Working on rows
:link: /user-guide/transform/rows/index
:link-type: doc
Select and derive, filter, sort, deduplicate, and sample. The verbs that change the shape of the table, with contracts precise enough to test against.
:::

:::{grid-item-card} {octicon}`code;1.1em` The column language
:link: /user-guide/transform/columns/index
:link-type: doc
{py:class}`Expr <batcher.plan.expr_ir.core.Expr>`, the typed accessor namespaces, the type system, and the batch UDF for when an expression genuinely cannot say it.
:::
::::

## Every page in this section

The pages below are listed in reading order, rows first and then columns.

| Page | What it covers |
|---|---|
| {doc}`Transformations <rows/transformations>` | Select and derive columns, match many at once with selectors, and flatten nested data |
| {doc}`Filtering and selection <rows/filtering>` | Predicates, null handling, and how a predicate reaches the scan |
| {doc}`Sorting <rows/sorting>` | Nulls, NaN, ties, top-k, and what makes a sort fast |
| {doc}`Distinct and deduplication <rows/distinct-and-dedup>` | Exact, keyed, and near-duplicate removal |
| {doc}`Sampling and splitting <rows/sampling>` | Reproducible samples and train/test splits |
| {doc}`Expressions <columns/expressions>` | The composable column language: operators, conditionals, nulls, math |
| {doc}`Expression accessors <columns/expression-accessors>` | The `.dt`, `.list`, `.struct`, `.map`, and `.json` namespaces |
| {doc}`The string accessor <columns/string-accessor>` | {py:class}`.str <batcher.plan.expr_ir.namespaces.strings._StrNamespace>`: search, regex, paths, recasing, and compression |
| {doc}`The sequence accessor <columns/sequence-accessor>` | {py:class}`.seq <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace>`: DNA, RNA, protein, and FASTQ-quality columns |
| {doc}`Map columns <columns/map-accessor>` | Building a map and reading it with {py:class}`.map <batcher.plan.expr_ir.namespaces.collections._MapNamespace>` |
| {doc}`Expression recipes <columns/expression-recipes>` | Porting, feature engineering, and text-corpus curation |
| {doc}`The type system <columns/type-system>` | Arrow types, boundary widening, casts, nulls |
| {doc}`User-defined functions <columns/udfs>` | Your Python over whole Arrow batches |
| {doc}`Running a UDF at scale <columns/udfs-at-scale>` | Distributing a UDF stage, tolerating bad rows, and retries |

## See also

- {doc}`/user-guide/analyze/index`: grouping, joining, and windowing the rows you kept.
- {doc}`/cookbook/expressions/index`: the column language as runnable recipes.
- {doc}`/api/relational/expressions`: every `Expr` method, enumerated.

```{toctree}
:hidden:

rows/index
columns/index
```
