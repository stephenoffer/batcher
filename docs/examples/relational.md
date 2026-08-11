# Relational operations

This page covers the scripts that exercise the relational core: choosing columns, filtering
rows, joining, aggregating, and computing over windows.

## Projections and filters

`select` decides the entire output shape, so anything it does not name is gone.
`with_columns` keeps every existing column and adds or replaces. Reaching for the first when
you meant the second is how a column quietly disappears three steps later.

```python
import batcher as bt
from batcher import col

orders = bt.from_pydict(
    {
        "o_orderkey": [1, 2, 3],
        "o_custkey": [10, 20, 30],
        "o_totalprice": [173665.47, 46929.18, 193846.25],
    }
)

narrow = orders.select("o_orderkey", "o_totalprice")
assert narrow.columns == ["o_orderkey", "o_totalprice"]

widened = orders.with_columns(price_in_thousands=col("o_totalprice") / 1000.0)
assert widened.columns == [*orders.columns, "price_in_thousands"]
```

Predicates combine with `&`, `|` and `~`, and the parentheses are mandatory because Python
binds those tighter than the comparisons. The subtler point is three-valued logic: a
comparison against null is null rather than false, so a row with a null key survives neither
`x == 1` nor `x != 1`.

## Joins

The join type decides what happens to rows with no partner. Semi and anti are the filtering
joins: both return only left-hand columns, and neither can increase the row count, which is
what makes them the right tool for a membership test.

```python
orders = bt.from_pydict({"id": [1, 2, 3], "cid": [10, 20, 99]})
customers = bt.from_pydict({"cid": [10, 20, 30], "name": ["ada", "bob", "cy"]})

with_customer = orders.join(customers, on="cid", how="semi")
orphans = orders.join(customers, on="cid", how="anti")

assert with_customer.count() + orphans.count() == orders.count()
assert with_customer.columns == orders.columns
```

Fan-out is the trap. An inner join emits one row per matching pair, so a right side with
three rows for a key turns one left row into three. Checking the key's uniqueness first is
one count, and aggregating the many side before the join removes the problem entirely.

## Aggregation

`agg` on a Dataset collapses it to one row; the same expressions after a `group_by` collapse
each group. That symmetry is deliberate, and it means there is no separate grouped spelling
to learn.

```python
lineitem = bt.from_pydict(
    {
        "l_shipmode": ["AIR", "SHIP", "AIR", "MAIL"],
        "l_quantity": [17, 36, 8, 28],
        "l_extendedprice": [21168.23, 45983.16, 13309.6, 28955.64],
    }
)

per_mode = (
    lineitem.group_by("l_shipmode")
    .agg(lines=bt.count(), qty=col("l_quantity").sum())
    .sort("l_shipmode")
)
result = per_mode.to_pydict()

assert sum(result["lines"]) == lineitem.count()
assert result["l_shipmode"] == sorted(result["l_shipmode"])
```

Two distinctions are worth holding onto. `bt.count()` counts rows while `col(x).count()`
counts non-null values of x, and they differ the moment a column has nulls. And an empty sum
is null rather than zero, while an empty count is zero rather than null.

## Windows

A group-by replaces the rows with one row per group; a window adds a column and keeps every
row. Reach for the window when downstream steps still need the detail.

```python
sales = bt.from_pydict(
    {
        "region": ["west", "west", "east", "east"],
        "amount": [10.0, 30.0, 20.0, 20.0],
    }
)

shares = sales.with_columns(
    region_total=col("amount").sum().over(partition_by=["region"])
).with_columns(share=col("amount") / col("region_total"))

values = shares.to_pydict()["share"]
assert all(0.0 < value <= 1.0 for value in values)
assert abs(sum(values) - 2.0) < 1e-9  # the shares within each region sum to one
```

Two behaviours differ from SQL and are worth knowing. `order_by` takes `(column, descending)`
pairs, so the ranking direction is part of the window rather than a separate argument. And
the default frame is the whole partition, so `last_value` returns the partition's last value
rather than the current row.

## Every script on this page

The table below lists the relational scripts in path order.

<!-- library-table: relational,joins,aggregations,windows,dataset -->
| Script | Shows |
| --- | --- |
| `examples/relational/anti_join_reconciliation.py` | Reconciling two datasets: what is in one and not the other, both ways |
| `examples/relational/append_and_concat.py` | Stacking datasets: vstack, append, and concat |
| `examples/relational/casting_and_types.py` | Changing types: cast on an expression, astype on a frame |
| `examples/relational/column_order_and_selection.py` | Controlling the column order of a result |
| `examples/relational/conditional_updates.py` | Updating a column in place, conditionally |
| `examples/relational/counting_without_scanning.py` | Counting rows, and the cheapest way to answer each kind of count question |
| `examples/relational/cross_join_grids.py` | Building a complete grid with a cross join, and filling the gaps |
| `examples/relational/crosstab_and_value_counts.py` | Frequency tables: value_counts for one column, crosstab for two |
| `examples/relational/deduplicate_keeping_latest.py` | Keeping the most recent row per key, which `distinct` cannot do |
| `examples/relational/distinct_and_dedup.py` | Removing duplicates: whole-row distinct versus keyed deduplication |
| `examples/relational/explode_and_unnest.py` | Nested data: exploding a list column and flattening a struct |
| `examples/relational/filter_predicates.py` | Filtering: combining predicates, and what nulls do to them |
| `examples/relational/filtering_by_aggregate.py` | Filtering rows by a property of their group |
| `examples/relational/grouping_sets_cube_rollup.py` | Several grouping levels in one pass: rollup, cube, and grouping sets |
| `examples/relational/incremental_processing.py` | Processing only what is new since the last run |
| `examples/relational/limit_and_slicing.py` | Taking a piece: head, tail, limit, slice, and every-nth |
| `examples/relational/nulls_across_operators.py` | How nulls travel through each operator |
| `examples/relational/pipe_and_compose.py` | Composing pipelines: `pipe` for reuse, and why laziness makes it free |
| `examples/relational/pipeline_composition_patterns.py` | Three ways to structure a long pipeline, and what each costs |
| `examples/relational/pivot_and_unpivot.py` | Long to wide and back: pivot and unpivot |
| `examples/relational/rename_and_drop.py` | Reshaping the column list: rename, drop, and selecting by dtype |
| `examples/relational/renaming_conventions.py` | Keeping column names sane through a multi-join pipeline |
| `examples/relational/sampling_and_row_index.py` | Sampling a large table, and attaching a row number |
| `examples/relational/schema_inspection.py` | Reading a dataset's shape without reading its rows |
| `examples/relational/select_and_project.py` | Choosing columns: `select` replaces the projection, `with_columns` extends it |
| `examples/relational/self_referential_hierarchies.py` | Walking a hierarchy without recursion |
| `examples/relational/set_operations.py` | Set operations: union, intersect, and except, with and without duplicates |
| `examples/relational/sorting.py` | Sorting: direction per key, and where nulls land |
| `examples/relational/top_k_per_group.py` | The top N rows within each group, two ways |
| `examples/relational/wide_to_long_reports.py` | Turning a report into a tidy table, and back |
| `examples/relational/window_free_top_n_per_group.py` | Top-N per group without a window, using a join against the group's threshold |
| `examples/joins/aggregate_before_join.py` | Shrinking a side before joining it |
| `examples/joins/asof_joins.py` | As-of joins: matching the most recent row at or before a timestamp |
| `examples/joins/column_collisions.py` | When both sides have a column of the same name |
| `examples/joins/duplicate_keys_and_fanout.py` | Fan-out: what a non-unique join key does to your row count |
| `examples/joins/inner_and_outer.py` | Inner, left, right and full outer over real tables |
| `examples/joins/join_hints_and_plans.py` | What a join looks like in the plan, and what the shape of the query tells the optimizer |
| `examples/joins/join_null_keys.py` | Null join keys, and why they match nothing |
| `examples/joins/keyless_and_cross.py` | Joining with no key at all, and keeping it safe |
| `examples/joins/multi_key_joins.py` | Joining on more than one column |
| `examples/joins/outer_join_reconciliation.py` | A full outer join, and reading the three populations it produces |
| `examples/joins/range_and_inequality.py` | Joining on a range rather than an equality |
| `examples/joins/self_joins.py` | Joining a table to itself, and keeping the two sides apart |
| `examples/joins/semi_and_anti.py` | Filtering joins: semi keeps matches, anti keeps orphans |
| `examples/joins/star_schema.py` | A star-schema query: one fact table, several small dimensions |
| `examples/aggregations/aggregate_after_join.py` | Aggregating across a join, and the fan-out that silently doubles your totals |
| `examples/aggregations/aggregate_over_windows.py` | Aggregating a windowed column: two-stage summaries |
| `examples/aggregations/approximate_vs_exact.py` | Sketches versus exact aggregates: what you trade and what you keep |
| `examples/aggregations/argmin_argmax.py` | Finding the row that holds an extreme, not just the extreme value |
| `examples/aggregations/basic_reductions.py` | The five reductions every report starts with, whole-table and per group |
| `examples/aggregations/bitwise_and_boolean.py` | Bitwise and boolean folds over a column |
| `examples/aggregations/conditional_aggregates.py` | Counting and summing subsets without a second query |
| `examples/aggregations/correlation.py` | Correlation and covariance between two columns |
| `examples/aggregations/counting_variants.py` | Counting: rows, non-nulls, matches, and distinct values |
| `examples/aggregations/dispersion.py` | Spread: sample versus population, and the scale-free summaries |
| `examples/aggregations/distinct_aggregates.py` | Counting distinct values inside a group, and the cost of doing it exactly |
| `examples/aggregations/distribution_shape.py` | Skewness and kurtosis: is this distribution lopsided, and how heavy are its tails |
| `examples/aggregations/empty_and_edge_cases.py` | What an aggregate returns when there is nothing to aggregate |
| `examples/aggregations/filtering_groups.py` | HAVING: filtering groups after the aggregate, and why order matters |
| `examples/aggregations/first_last_and_mode.py` | Positional and most-common aggregates: first, last, and mode |
| `examples/aggregations/histograms.py` | Binning a column: the histogram aggregate and explicit width buckets |
| `examples/aggregations/list_aggregation.py` | Collecting a group's values into a list column |
| `examples/aggregations/means.py` | Four kinds of average, and when each is the right one |
| `examples/aggregations/multi_key_grouping.py` | Grouping by several columns, and by an expression |
| `examples/aggregations/ordered_aggregates.py` | Aggregates that depend on order, and how to make them deterministic |
| `examples/aggregations/pivot_style_reports.py` | A cross-tab report built from conditional aggregates rather than a pivot |
| `examples/aggregations/products_and_overflow.py` | Multiplicative folds, and the overflow you get for free |
| `examples/aggregations/quantiles.py` | Quantiles: exact, named, and sketch-approximated |
| `examples/aggregations/regression.py` | Least-squares regression as an aggregate, not a model fit |
| `examples/aggregations/rolling_up_hierarchies.py` | Rolling a fine aggregate up to a coarse one without re-scanning |
| `examples/aggregations/streaming_safe_aggregates.py` | Which aggregates can be maintained incrementally, and which cannot |
| `examples/aggregations/weighted_and_ratio_metrics.py` | Ratios of aggregates, not aggregates of ratios |
| `examples/windows/cumulative_distribution.py` | Building a cumulative share, the Pareto "80% of revenue" chart |
| `examples/windows/exclude_current_row.py` | Leave-one-out aggregates: the group's total without this row |
| `examples/windows/first_and_last_value.py` | Reaching the endpoints of a window: first_value, last_value, nth_value |
| `examples/windows/gaps_and_islands.py` | Finding consecutive runs: the gaps-and-islands pattern |
| `examples/windows/lag_and_lead.py` | Looking at neighbouring rows: lag, lead, and period-over-period change |
| `examples/windows/moving_averages.py` | Sliding windows: a moving average over a bounded frame |
| `examples/windows/multiple_windows.py` | Several different windows in one projection |
| `examples/windows/ntile_and_quartiles.py` | Splitting an ordered partition into equal buckets with ntile |
| `examples/windows/percent_rank_and_cume_dist.py` | Relative position: percent_rank and cume_dist |
| `examples/windows/rank_dense_within_time.py` | Ranking within a time partition, and the ties that dates create |
| `examples/windows/ranking_functions.py` | Ranking within a partition: row_number, rank, and dense_rank |
| `examples/windows/rolling_statistics.py` | Rolling statistics beyond the mean: standard deviation, min and max over a frame |
| `examples/windows/running_totals.py` | Cumulative sums: an ordered window with an unbounded preceding frame |
| `examples/windows/share_of_partition.py` | Each row's share of its group, without a join back |
| `examples/windows/window_versus_groupby.py` | Window or group-by: the same aggregate, two different output shapes |
| `examples/dataset/deduplication.py` | Deduplication: exact keys, whole rows, and keeping a chosen survivor |
| `examples/dataset/dq_contracts.py` | Data-quality contracts: validate, fail, drop, or quarantine |
| `examples/dataset/grouping.py` | Grouping: agg, multi-key rollups, and the cube/rollup/grouping-set variants |
| `examples/dataset/iteration.py` | Getting results out: batches, rows, slices, and the single-value cases |
| `examples/dataset/joins.py` | Join types, key spellings, and the as-of join for time series |
| `examples/dataset/meta_columns.py` | Profiling one column: bounds, uniqueness, nulls, and constancy |
| `examples/dataset/meta_comparison.py` | Asking about a join before running it, and reading approximate statistics |
| `examples/dataset/meta_predicates.py` | Cheap yes/no questions about the data, and the column-check shorthands |
| `examples/dataset/meta_schema.py` | Asking about a dataset's shape without executing it |
| `examples/dataset/null_handling.py` | Dataset-level null handling: dropping, filling, and counting missing values |
| `examples/dataset/profiling.py` | Profiling a table you have just been handed |
| `examples/dataset/reading_the_whole_surface.py` | A sweep over the Dataset API: every accessor and metadata method, checked |
| `examples/dataset/reshaping.py` | Reshaping: pivot, unpivot, explode, unnest, and set operations |
| `examples/dataset/sampling_and_splits.py` | Sampling and splitting: reproducible subsets that do not leak |
| `examples/dataset/sql_interface.py` | SQL over the same engine, and mixing SQL with DataFrame verbs |
<!-- /library-table -->
