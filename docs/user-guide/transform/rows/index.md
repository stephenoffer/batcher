# Working on rows

This section covers the verbs that decide which rows survive and in what order: selecting and deriving, filtering, sorting, deduplicating, and sampling. They change the shape of the table rather than the contents of a column, which {doc}`/user-guide/transform/columns/index` covers.

Each of these verbs comes with a contract you can test against, not just a behavior that usually holds. A sort states where nulls and NaN land in both directions. A keyed dedup says which row survives and why. A sample is a function of the data and the seed, so it returns the same rows on one core, sixteen, or a cluster. And every verb builds on the same lazy plan, so a filter written after a join still reaches the scan, and a `top_k` replaces a full sort with a bounded heap.

Order the verbs by what they cut. A sort has to see every row before it can emit one, so every row a predicate removed earlier is a row the sort never orders, and a `distinct` or a `limit` in between cuts more of them still. Filter first and sort last.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`columns;1.1em` Transformations
:link: /user-guide/transform/rows/transformations
:link-type: doc
`select` and `with_columns`, selectors that match many columns at once, and flattening nested data.
:::

:::{grid-item-card} {octicon}`filter;1.1em` Filtering and selection
:link: /user-guide/transform/rows/filtering
:link-type: doc
Predicates, three-valued nulls, and how a filter reaches the scan.
:::

:::{grid-item-card} {octicon}`sort-desc;1.1em` Sorting
:link: /user-guide/transform/rows/sorting
:link-type: doc
Nulls, NaN, ties, top-k, spill, and what makes a sort fast.
:::

:::{grid-item-card} {octicon}`duplicate;1.1em` Distinct and deduplication
:link: /user-guide/transform/rows/distinct-and-dedup
:link-type: doc
The three jobs hiding under the word "dedupe", and picking the right one.
:::

:::{grid-item-card} {octicon}`git-branch;1.1em` Sampling and splitting
:link: /user-guide/transform/rows/sampling
:link-type: doc
Samples and splits that reproduce however the data is laid out.
:::
::::

## See also

- {doc}`/user-guide/analyze/index`: grouping, joining, and windowing, once the rows are the ones you want.
- {doc}`/cookbook/dataset/verbs/index`: the same verbs as runnable recipes.
- {doc}`/api/relational/dataset`: the reference for every verb in this section.

```{toctree}
:hidden:

transformations
filtering
sorting
distinct-and-dedup
sampling
```
