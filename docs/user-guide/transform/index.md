# Transform

Reshape a dataset. This section splits along the line the API itself draws: the verbs that
decide which rows survive, and the expression language that decides what a column contains.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`filter;1.1em` Working on rows
:link: /user-guide/transform/rows/index
:link-type: doc
Select and derive, filter, sort, deduplicate, and sample. The verbs that change the shape of
the table.
:::

:::{grid-item-card} {octicon}`code;1.1em` The column language
:link: /user-guide/transform/columns/index
:link-type: doc
`Expr`, the typed accessor namespaces, the type system, and the batch UDF for when an
expression genuinely cannot say it.
:::
::::

## Every page in this section

| Page | What it covers |
|---|---|
| {doc}`Transformations <rows/transformations>` | Select and derive columns, reshape and explode them |
| {doc}`Filtering and selection <rows/filtering>` | Predicates, null handling, and how a predicate reaches the scan |
| {doc}`Sorting <rows/sorting>` | Nulls, NaN, ties, descending keys, and top-n |
| {doc}`Distinct and deduplication <rows/distinct-and-dedup>` | Exact, keyed, and near-duplicate removal |
| {doc}`Sampling and splitting <rows/sampling>` | Reproducible samples and train/test splits |
| {doc}`Expressions <columns/expressions>` | The composable column language: operators, conditionals, nulls, math |
| {doc}`Expression accessors <columns/expression-accessors>` | `.str`, `.dt`, `.list`, `.struct`, and `.json` |
| {doc}`Expression recipes <columns/expression-recipes>` | Porting, feature engineering, and text-corpus curation |
| {doc}`The type system <columns/type-system>` | Arrow types, boundary widening, casts, nulls |
| {doc}`User-defined functions <columns/udfs>` | Your Python over whole Arrow batches |

## See also

- {doc}`/user-guide/analyze/index`: grouping, joining, and windowing the rows you kept.
- {doc}`/cookbook/expressions/index`: the column language as 34 runnable pages.

```{toctree}
:hidden:

rows/index
columns/index
```
