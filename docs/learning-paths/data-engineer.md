# Data engineer learning path

This path is for building and running data pipelines: read a source, reshape it, join it
against another, aggregate, write the result. The pipeline stays lazy until a terminal
operation, and all per-row work runs in Rust.

## Reading order

1. [Getting started](../getting-started/index.md): install and run a first query.
2. [Your first pipeline](../tutorials/first-pipeline.md): the end-to-end flow.
3. [Reading data](../user-guide/reading-data.md): sources and file formats.
4. [Transformations](../user-guide/transformations.md): `select`, `with_columns`,
   `filter`, `sort`.
5. [Filtering](../user-guide/filtering.md): predicate expressions.
6. [Aggregations](../user-guide/aggregations.md): `group_by` and `.agg`.
7. [Joins](../user-guide/joins.md): join kinds and keys.
8. [Window functions](../user-guide/window-functions.md): ranking and rolling
   aggregates.
9. [Writing data](../user-guide/writing-data.md): output formats and partitioning.
10. [Lakehouse tables](../user-guide/lakehouse.md): Delta read/write/merge and SCD.
11. [Data quality](../user-guide/data-quality.md): validate against a contract and
    quarantine what fails it.
12. [Cloud storage](../user-guide/cloud-storage.md): object-store paths.
13. [Performance and memory](../user-guide/performance.md): caching and spill.
14. [Best practices](../user-guide/best-practices.md) and
    [troubleshooting](../user-guide/troubleshooting.md).
15. [Dataset API reference](../api/dataset.md).

## Example: transform and aggregate

```python
import batcher as bt

orders = bt.from_pydict(
    {
        "region": ["west", "east", "west", "east", "west"],
        "price": [10.0, 20.0, 30.0, 40.0, 50.0],
        "qty": [1, 2, 3, 4, 5],
    }
)

revenue = (
    orders.with_columns(total=bt.col("price") * bt.col("qty"))
    .group_by("region")
    .agg(revenue=bt.col("total").sum(), orders=bt.count())
    .sort("revenue", descending=True)
)
print(revenue.to_pydict())
# {'region': ['west', 'east'], 'revenue': [350.0, 200.0], 'orders': [3, 2]}
```

## Example: join a dimension table

```python
facts = bt.from_pydict({"region": ["west", "east", "west"], "amount": [1, 2, 3]})
dim = bt.from_pydict({"region": ["west", "east"], "label": ["W", "E"]})

joined = facts.join(dim, on="region", how="inner").sort("amount")
print(joined.to_pydict())
# {'region': ['west', 'east', 'west'], 'amount': [1, 2, 3], 'label': ['W', 'E', 'W']}
```

## Runnable examples

These scripts build their own data and run directly with `python examples/<name>.py`:

- `quickstart.py` and `transformations_aggregations_joins.py` cover the pipeline core.
- `data_quality.py` validates rows against a contract and quarantines the failures.
- `lakehouse_scd.py` does a Delta round-trip, then SCD type-2 history.
- `timeseries.py` and `window_functions.py` show time buckets and rolling aggregates.
- `spill.py` runs out-of-core under a bounded budget.


## Recipes for the problems you will actually hit

The [data-engineering cookbook](../examples/data-engineering/index.md) is the applied half
of this path. Each recipe opens on the failure and shows the code that avoids it.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`download;1.1em` Incremental ingest
:link: ../examples/data-engineering/incremental-ingest
:link-type: doc
Read only what is new.
:::

:::{grid-item-card} {octicon}`git-merge;1.1em` CDC pipeline
:link: ../examples/data-engineering/cdc-pipeline
:link-type: doc
Apply a change feed in the order the changes happened.
:::

:::{grid-item-card} {octicon}`filter;1.1em` Deduplication
:link: ../examples/data-engineering/deduplication
:link-type: doc
Exactly-once is a property you build.
:::

:::{grid-item-card} {octicon}`stack;1.1em` Schema evolution
:link: ../examples/data-engineering/schema-evolution
:link-type: doc
The column that changed type under you.
:::
::::

:::{seealso}
- [Integrations](../integrations/index.md) — connecting to Kafka, Snowflake, Delta, and the rest.
- [Building a lakehouse](../tutorials/building-a-lakehouse.md) — the same pieces, end to end.
- [Custom connectors](../user-guide/custom-connectors.md) — when the format you need isn't built in.
:::
