# Working on rows

These pages cover the verbs that decide which rows survive and in what order. They change
the shape of the table rather than the contents of a column, which is what
{doc}`/user-guide/transform/columns/index` covers.

The usual order is filter first, sort last. A sort has to see every row before it can emit
one, so every row a predicate cut earlier is a row the sort never has to order, and a
`distinct` or a `limit` in between cuts more of them still.

| Page | What it covers |
|---|---|
| {doc}`Transformations <transformations>` | `select` and `with_columns`, and which of the two you want |
| {doc}`Filtering and selection <filtering>` | Keeping the rows a predicate accepts, and how a predicate reaches the scan |
| {doc}`Sorting <sorting>` | Ordering rows, including nulls, descending keys, and what happens under spill |
| {doc}`Distinct and deduplication <distinct-and-dedup>` | The three different jobs hiding under the word "dedupe", and picking the right one |
| {doc}`Sampling and splitting <sampling>` | Reproducible samples and splits that stay reproducible across partition counts |

## See also

- {doc}`/user-guide/analyze/index`: grouping, joining, and windowing, once the rows are the ones you want.
- {doc}`/cookbook/dataset/verbs/index`: the same verbs as runnable recipes.

```{toctree}
:hidden:

transformations
filtering
sorting
distinct-and-dedup
sampling
```
