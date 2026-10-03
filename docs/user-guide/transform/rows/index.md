# Working on rows

This section covers the verbs that decide which rows survive and in what order: selecting and deriving, filtering, sorting, deduplicating, and sampling. They change the shape of the table rather than the contents of a column, which {doc}`/user-guide/transform/columns/index` covers.

Each verb comes with a contract you can test against. A sort states where nulls land, a keyed dedup says which row survives, and a seeded sample returns the same rows on one core or a cluster.

```python
import batcher as bt

ev = bt.from_pydict({"user": ["a", "b", "a", "c", "b", "a"], "score": [3.0, None, 7.0, 5.0, 9.0, 7.0]})
print(ev.filter(bt.col("score") > 4).to_pydict())
# {'user': ['a', 'c', 'b', 'a'], 'score': [7.0, 5.0, 9.0, 7.0]}
print(ev.sort("score", descending=True).to_pydict())
# {'user': ['b', 'a', 'a', 'c', 'a', 'b'], 'score': [9.0, 7.0, 7.0, 5.0, 3.0, None]}
```

Dedup, top-k, and seeded sampling are one call each:

```python
print(ev.distinct().sort("user", "score").to_pydict())
# {'user': ['a', 'a', 'b', 'b', 'c'], 'score': [3.0, 7.0, 9.0, None, 5.0]}
print(ev.top_k(2, by="score").to_pydict())
# {'user': ['b', 'a'], 'score': [9.0, 7.0]}
print(ev.sample(n=3, seed=7).sort("score").to_pydict())
# {'user': ['c', 'b', 'b'], 'score': [5.0, 9.0, None]}
```

Filter first and sort last. A sort has to see every row before it emits one, so every row a predicate removes earlier is a row the sort never orders. `top_k` goes further and replaces a full sort with a bounded heap.

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
