# Data engineer learning path

This path is for building and running data pipelines: read a source, reshape it, join it
against another, aggregate, write the result. The pipeline stays lazy until a terminal
operation, and all per-row work runs in Rust.

## Reading order

1. {doc}`Getting started <../getting-started/index>`: install and run a first query.
1. {doc}`Your first pipeline <../tutorials/first-pipeline>`: the end-to-end flow.
1. {doc}`Reading data </user-guide/moving-data/reading-data>`: sources and file formats.
1. {doc}`Transformations </user-guide/transform/transformations>`: `select`, `with_columns`,
   `filter`, `sort`.
1. {doc}`Filtering </user-guide/transform/filtering>`: predicate expressions.
1. {doc}`Aggregations </user-guide/analyze/aggregations>`: `group_by` and `.agg`.
1. {doc}`Joins </user-guide/analyze/joins>`: join kinds and keys.
1. {doc}`Window functions </user-guide/analyze/window-functions>`: ranking and rolling
   aggregates.
1. {doc}`Writing data </user-guide/moving-data/writing-data>`: output formats and partitioning.
1. {doc}`Lakehouse tables </user-guide/moving-data/lakehouse>`: Delta read/write/merge and SCD.
1. {doc}`Data quality </user-guide/trust/data-quality>`: validate against a contract and
   quarantine what fails it.
1. {doc}`Cloud storage </user-guide/moving-data/cloud-storage>`: object-store paths.
1. {doc}`Performance and memory </user-guide/operate/performance>`: caching and spill.
1. {doc}`Best practices </user-guide/operate/best-practices>` and
   {doc}`troubleshooting </user-guide/operate/troubleshooting>`.
1. {doc}`Dataset API reference </api/relational/dataset>`.

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

The {doc}`data-engineering cookbook </cookbook/data-engineering/index>` is the applied half
of this path. Each recipe opens on the failure and shows the code that avoids it.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`download;1.1em` Incremental ingest
:link: /cookbook/data-engineering/incremental-ingest
:link-type: doc
Read only what is new.
:::

:::{grid-item-card} {octicon}`git-merge;1.1em` CDC pipeline
:link: /cookbook/data-engineering/cdc-pipeline
:link-type: doc
Apply a change feed in the order the changes happened.
:::

:::{grid-item-card} {octicon}`filter;1.1em` Deduplication
:link: /cookbook/data-engineering/deduplication
:link-type: doc
Exactly-once is a property you build.
:::

:::{grid-item-card} {octicon}`stack;1.1em` Schema evolution
:link: /cookbook/data-engineering/schema-evolution
:link-type: doc
The column that changed type under you.
:::
::::

:::{seealso}
- {doc}`Integrations <../integrations/index>`: connecting to Kafka, Snowflake, Delta, and the rest.
- {doc}`Building a lakehouse <../tutorials/building-a-lakehouse>`: the same pieces, end to end.
- {doc}`Custom connectors </user-guide/moving-data/custom-connectors>`: when the format you need isn't built in.
:::


## See also

- {doc}`platform-engineer`: the operational half, once the pipelines exist.
- {doc}`../user-guide/index`: the reference guides this path draws on.
- {doc}`/cookbook/data-engineering/index`: runnable versions of the patterns above.
