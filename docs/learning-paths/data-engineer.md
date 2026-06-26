# Data engineer learning path

For building and operating data pipelines: read sources, transform and aggregate,
join, and write results. The pipeline stays lazy until a terminal operation, and all
per-row work runs in Rust.

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
11. [Data quality](../user-guide/data-quality.md): validate, quarantine, and enforce
    a contract.
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

- `quickstart.py`, `transformations_aggregations_joins.py` — the pipeline core.
- `data_quality.py` — validate and quarantine against a contract.
- `lakehouse_scd.py` — a Delta round-trip and SCD type-2 history.
- `timeseries.py`, `window_functions.py` — time buckets and rolling aggregates.
- `spill.py` — out-of-core execution under a bounded budget.
