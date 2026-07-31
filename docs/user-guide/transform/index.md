# Transform

Reshape rows and columns: the expression language, and the operators that select, filter, order, and deduplicate.

::::{grid} 1 3 3 3
:gutter: 3

:::{grid-item-card} {octicon}`pencil;1.1em` Transformations
:link: /user-guide/transform/transformations
:link-type: doc
Select and derive columns; reshape and explode them.
:::

:::{grid-item-card} {octicon}`filter;1.1em` Filtering
:link: /user-guide/transform/filtering
:link-type: doc
Predicates, null handling, sampling.
:::

:::{grid-item-card} {octicon}`code;1.1em` Expressions
:link: /user-guide/transform/expressions
:link-type: doc
The composable column language: operators, conditionals, nulls, math.
:::

:::{grid-item-card} {octicon}`list-unordered;1.1em` Expression accessors
:link: /user-guide/transform/expression-accessors
:link-type: doc
`.str`, `.dt`, `.list`, `.struct`, and `.json`.
:::

:::{grid-item-card} {octicon}`beaker;1.1em` Expression recipes
:link: /user-guide/transform/expression-recipes
:link-type: doc
Porting, feature engineering, and text-corpus curation.
:::

:::{grid-item-card} {octicon}`sort-desc;1.1em` Sorting
:link: /user-guide/transform/sorting
:link-type: doc
Order rows; nulls, NaN, ties, top-n.
:::

:::{grid-item-card} {octicon}`duplicate;1.1em` Distinct and dedup
:link: /user-guide/transform/distinct-and-dedup
:link-type: doc
Exact, keyed, and near-duplicate removal.
:::

:::{grid-item-card} {octicon}`beaker;1.1em` Sampling
:link: /user-guide/transform/sampling
:link-type: doc
Reproducible samples and train/test splits.
:::

:::{grid-item-card} {octicon}`code-square;1.1em` UDFs
:link: /user-guide/transform/udfs
:link-type: doc
Your Python over whole Arrow batches.
:::

:::{grid-item-card} {octicon}`typography;1.1em` Type system
:link: /user-guide/transform/type-system
:link-type: doc
Arrow types, boundary widening, casts, nulls.
:::
::::

```{toctree}
:hidden:

transformations
filtering
expressions
expression-accessors
expression-recipes
sorting
distinct-and-dedup
sampling
udfs
type-system
```
