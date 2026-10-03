# Transform

This section covers reshaping a dataset in Batcher. The API draws one line through it. Verbs pick the rows and their order, and expressions decide what a column holds.

Both halves are lazy. A whole chain reaches the optimizer as one plan, so the filter you wrote last still moves down into the Parquet scan. Columns nobody reads get dropped. A run of arithmetic fuses into one compiled pass that Rust evaluates a batch at a time.

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

Sometimes you need your own Python. A batch UDF gets it zero-copy Arrow batches:

```python
import pyarrow.compute as pc

shout = people.map_batches(lambda batch: batch.set_column(0, "name", pc.utf8_upper(batch["name"])))
print(shout.to_pydict())
# {'name': ['ANN', 'BOB', 'CARL'], 'age': [34, None, 51]}
```

## Two halves of one chain

`filter`, `distinct` and `sample` drop rows. `sort` orders them. `select` and `with_columns` set what each column holds, though the expression you hand them does the actual computing.

![A matrix of six verbs against three properties of a table, beside a panel on the column language. filter, distinct and sample change which rows survive and leave row order and columns alone. sort sets the row order and changes neither which rows survive nor the columns. select and with_columns change what the columns hold and leave the rows and their order alone. Every call returns a new lazy Dataset, and nothing runs until a result is asked for. The column language panel shows three expressions: bt.col("age") >= 18, bt.col("name").str.to_case("title"), and (bt.col("age") / 10).floor(). An expression lowers to a Rust expression tree that runs over whole Arrow batches, never one Python call per row, and filter, select and with_columns evaluate it. A batch UDF is the escape hatch, running your Python over whole Arrow batches.](/_static/diagrams/transform_rows_vs_columns.svg)

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`filter;1.1em` Working on rows
:link: /user-guide/transform/rows/index
:link-type: doc
The verbs that change the shape of a table, from `select` to `sample`, each with a contract you can test against.
:::

:::{grid-item-card} {octicon}`code;1.1em` The column language
:link: /user-guide/transform/columns/index
:link-type: doc
{py:class}`Expr <batcher.plan.expr_ir.core.Expr>` and its typed accessors, the type system, and a batch UDF for anything an expression can't say.
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
| {doc}`Media columns <columns/media-accessor>` | {py:class}`.image <batcher.plan.expr_ir.image._ImageNamespace>`, {py:class}`.audio <batcher.plan.expr_ir.audio._AudioNamespace>`, and {py:class}`.video <batcher.plan.expr_ir.video._VideoNamespace>`: header facts, decoding, and quality signals |
| {doc}`The sequence accessor <columns/sequence-accessor>` | {py:class}`.seq <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace>`: DNA, RNA, protein, and FASTQ-quality columns |
| {doc}`Map columns <columns/map-accessor>` | Building a map and reading it with {py:class}`.map <batcher.plan.expr_ir.namespaces.collections._MapNamespace>` |
| {doc}`Expression recipes <columns/expression-recipes>` | Porting, feature engineering, and text-corpus curation |
| {doc}`The type system <columns/type-system>` | Arrow types, boundary widening, casts, nulls |
| {doc}`User-defined functions <columns/udfs>` | Your Python over whole Arrow batches |
| {doc}`Running a UDF at scale <columns/udfs-at-scale>` | Distributing a UDF stage, tolerating bad rows, and retries |

## See also

These pages pick up where this section stops.

- {doc}`/user-guide/analyze/index`: group, join or window the rows you kept.
- {doc}`/cookbook/expressions/index`: the column language as runnable recipes.
- {doc}`/api/relational/expressions`: every `Expr` method, enumerated.
- {doc}`/examples/relational`: the same verbs as standalone scripts, each run on every commit.

```{toctree}
:hidden:

rows/index
columns/index
```
