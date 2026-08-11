# Analytics

The queries an analyst actually writes, and the trap in each one.

Most of these look trivial until the data gets big or the edge cases show up. A funnel
joined the obvious way explodes. Sessionization done with a naive group-by double-counts.
Top-k per group with a full sort does far more work than it needs to.

Every recipe here runs on a small table you can read, so you can see the answer and check
it by eye. The same code runs on a billion rows.

:::{tip}
Most of these pages carry the main query twice, as a DataFrame chain and as {py:func}`bt.sql(...) <batcher.sql>`.
They are two spellings of one logical plan rather than two implementations, so pick
whichever reads better for the question you are asking.
:::

## Start with one worked query

{doc}`/cookbook/analytics/aggregates/analytics-query` runs aggregate, join, and window over one small orders table,
spelled both as SQL and as DataFrame code. It is the shape the rest of this section
specialises.

## Users over time

Four questions about what people did, all of them window functions over an event table. See {doc}`behavior/index`.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`stack;1.1em` Cohort analysis
:link: /cookbook/analytics/behavior/cohort-analysis
:link-type: doc
A row has no cohort. A user does. Label the user, then pivot the triangle.
:::

:::{grid-item-card} {octicon}`meter;1.1em` Retention curves
:link: /cookbook/analytics/behavior/retention-curves
:link-type: doc
Counting events instead of people is how a retention rate ends up above 100%.
:::

:::{grid-item-card} {octicon}`filter;1.1em` Funnel analysis
:link: /cookbook/analytics/behavior/funnel-analysis
:link-type: doc
Where the naive join blows up, and the one-pass pivot that replaces it.
:::

:::{grid-item-card} {octicon}`broadcast;1.1em` Sessionization
:link: /cookbook/analytics/behavior/sessionization
:link-type: doc
Turning a click stream into sessions with a window function, not a calendar day.
:::
::::

## Aggregation and ranking

The shapes most analytical queries actually are. See {doc}`aggregates/index`.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`graph;1.1em` Time-series rollups
:link: /cookbook/analytics/aggregates/time-series-rollups
:link-type: doc
`GROUP BY day` cannot emit a day it never saw. Build the calendar spine first.
:::

:::{grid-item-card} {octicon}`search;1.1em` Top-k per group
:link: /cookbook/analytics/aggregates/top-k-per-group
:link-type: doc
Why a windowed rank beats a sort, and which of the three rank functions you want.
:::

:::{grid-item-card} {octicon}`table;1.1em` Geospatial binning
:link: geospatial-binning
:link-type: doc
Snapping coordinates to a grid, where `round` and `cast` both lie to you.
:::
::::

## Drawing conclusions

Past describing the data to deciding something from it. See {doc}`inference/index`.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`git-merge;1.1em` Basket analysis
:link: /cookbook/analytics/inference/basket-analysis
:link-type: doc
The self-join that counts every pair twice, and why lift beats a raw co-occurrence count.
:::

:::{grid-item-card} {octicon}`beaker;1.1em` A/B testing
:link: /cookbook/analytics/inference/ab-testing
:link-type: doc
Divide by the wrong denominator and the experiment hands you the opposite answer.
:::

:::{grid-item-card} {octicon}`alert;1.1em` Anomaly detection
:link: /cookbook/analytics/inference/anomaly-detection
:link-type: doc
The three-sigma rule never fires, because the outlier is inside its own baseline.
:::
::::

## See also

- {doc}`Aggregations </user-guide/analyze/aggregations>` and {doc}`window functions </user-guide/analyze/window-functions>`: the two operators most of these recipes are built from.
- {doc}`SQL </user-guide/analyze/sql>`: the same plans, written as queries.
- {doc}`Expressions API </api/relational/expressions>` and {doc}`Dataset API </api/relational/dataset>`: the reference for everything used here.
- {doc}`Data engineering recipes </cookbook/data-engineering/index>`: the pipelines that produce the tables these queries read.

```{toctree}
:hidden:

behavior/index
aggregates/index
inference/index
geospatial-binning
```
