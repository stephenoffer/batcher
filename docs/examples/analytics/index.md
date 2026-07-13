# Analytics recipes

The queries an analyst actually writes, and the trap in each one.

Most of these look trivial until the data gets big or the edge cases show up. A funnel
joined the obvious way explodes. Sessionization done with a naive group-by double-counts.
Top-k per group with a full sort does far more work than it needs to.

Every recipe here runs on a small table you can read, so you can see the answer and check
it by eye. The same code runs on a billion rows.

:::{tip}
Most of these pages carry the main query twice, as a DataFrame chain and as `bt.sql(...)`.
They are two spellings of one logical plan rather than two implementations, so pick
whichever reads better for the question you are asking.
:::

## Users over time

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`stack;1.1em` Cohort analysis
:link: cohort-analysis
:link-type: doc
A row has no cohort; a user does. Label the user, then pivot the triangle.
:::

:::{grid-item-card} {octicon}`meter;1.1em` Retention curves
:link: retention-curves
:link-type: doc
Counting events instead of people is how a retention rate ends up above 100%.
:::

:::{grid-item-card} {octicon}`filter;1.1em` Funnel analysis
:link: funnel-analysis
:link-type: doc
Where the naive join blows up, and the one-pass pivot that replaces it.
:::

:::{grid-item-card} {octicon}`broadcast;1.1em` Sessionization
:link: sessionization
:link-type: doc
Turning a click stream into sessions with a window function, not a calendar day.
:::
::::

## Aggregation and ranking

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`graph;1.1em` Time-series rollups
:link: time-series-rollups
:link-type: doc
`GROUP BY day` cannot emit a day it never saw. Build the calendar spine first.
:::

:::{grid-item-card} {octicon}`search;1.1em` Top-k per group
:link: top-k-per-group
:link-type: doc
Why a windowed rank beats a sort, and which of the three rank functions you want.
:::

:::{grid-item-card} {octicon}`git-merge;1.1em` Basket analysis
:link: basket-analysis
:link-type: doc
The self-join that counts every pair twice, and why lift beats a raw co-occurrence count.
:::

:::{grid-item-card} {octicon}`table;1.1em` Geospatial binning
:link: geospatial-binning
:link-type: doc
Snapping coordinates to a grid, where `round` and `cast` both lie to you.
:::
::::

## Statistics

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`beaker;1.1em` A/B testing
:link: ab-testing
:link-type: doc
Divide by the wrong denominator and the experiment hands you the opposite answer.
:::

:::{grid-item-card} {octicon}`alert;1.1em` Anomaly detection
:link: anomaly-detection
:link-type: doc
The three-sigma rule never fires, because the outlier is inside its own baseline.
:::
::::

## See also

:::{seealso}
- [Aggregations](../../user-guide/aggregations.md) and [window functions](../../user-guide/window-functions.md): the two operators most of these recipes are built from.
- [SQL](../../user-guide/sql.md): the same plans, written as queries.
- [Expressions API](../../api/expressions.md) and [Dataset API](../../api/dataset.md): the reference for everything used here.
- [Data engineering recipes](../data-engineering/index.md): the pipelines that produce the tables these queries read.
:::

```{toctree}
:hidden:

cohort-analysis
funnel-analysis
sessionization
time-series-rollups
top-k-per-group
basket-analysis
ab-testing
anomaly-detection
geospatial-binning
retention-curves
```
