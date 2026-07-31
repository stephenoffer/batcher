# Analyze

Reduce many rows to an answer: grouping, joining, ranking, and the SQL front-end over the same engine.

::::{grid} 1 2 2 4
:gutter: 3

:::{grid-item-card} {octicon}`graph;1.1em` Aggregations
:link: /user-guide/analyze/aggregations
:link-type: doc
Group and summarize; pivot; roll up.
:::

:::{grid-item-card} {octicon}`git-merge;1.1em` Joins
:link: /user-guide/analyze/joins
:link-type: doc
Inner, outer, semi, anti, as-of.
:::

:::{grid-item-card} {octicon}`versions;1.1em` Window functions
:link: /user-guide/analyze/window-functions
:link-type: doc
Ranking, running totals, lag and lead.
:::

:::{grid-item-card} {octicon}`table;1.1em` Pivoting
:link: /user-guide/analyze/pivoting
:link-type: doc
Long to wide and back.
:::

:::{grid-item-card} {octicon}`database;1.1em` SQL
:link: /user-guide/analyze/sql
:link-type: doc
Full SQL that lowers to the same engine.
:::

:::{grid-item-card} {octicon}`zap;1.1em` Metadata shortcuts
:link: /user-guide/analyze/metadata-shortcuts
:link-type: doc
Answer from the footer instead of the data, with `ds.meta`.
:::
::::

```{toctree}
:hidden:

aggregations
joins
window-functions
pivoting
sql
metadata-shortcuts
```
