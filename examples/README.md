# Examples

Runnable, self-contained scripts for the main ways to use Batcher. Each builds its
own in-memory data and asserts on its output, so you can run any of them directly:

```bash
python examples/quickstart.py
```

Every script here is executed in CI by `tests/docs/test_examples.py`, so an example
that references a removed or renamed API fails the test suite instead of rotting.

## Start here

| Script | Shows |
| --- | --- |
| `quickstart.py` | the headline lazy pipeline: filter → group_by → agg → sort |
| `transformations_aggregations_joins.py` | the DataFrame core: derive, filter, aggregate, join |
| `sql.py` | `bt.sql(...)` over Datasets, composed with the DataFrame API |

## By application type

**Data engineering / ETL**

| Script | Shows |
| --- | --- |
| `data_quality.py` | validate, quarantine, drop, and enforce a data contract with `ds.dq` |
| `lakehouse_scd.py` | a Delta round-trip and an SCD type-2 history build |
| `feature_engineering.py` | derive model-ready columns: scaling, bucketing, encoding, imputation |
| `timeseries.py` | extract date parts, resample to a period, compute period-over-period change |
| `window_functions.py` | per-partition aggregates and ranking with `.over(...)` |

**Machine learning**

| Script | Shows |
| --- | --- |
| `ml_inference.py` | batch inference — score every row with a model-shaped callable via `ds.ml.map_batches` |
| `streaming_pipeline.py` | a Kafka → windowed-aggregate → Delta micro-batch pipeline (skipped; needs a broker) |

**Operating the engine**

| Script | Shows |
| --- | --- |
| `performance_caching.py` | cache a reused result and spill under a tiny memory budget |
| `spill.py` | out-of-core execution under a bounded memory budget |
| `adaptive_optimization.py` | the moat — intra-query re-optimization, result-identical |
| `distributed.py` | the same query single-node vs across Ray workers (needs `[ray]`) |

## By role

- **Data engineer** — `quickstart`, `transformations_aggregations_joins`, `data_quality`,
  `lakehouse_scd`, `timeseries`, `window_functions`, `spill`.
- **Data scientist** — `quickstart`, `sql`, `feature_engineering`, `timeseries`,
  `window_functions`.
- **ML engineer** — `ml_inference`, `feature_engineering`, `streaming_pipeline`.
- **Platform engineer** — `performance_caching`, `spill`, `adaptive_optimization`,
  `distributed`.

## Examples that need setup

`distributed.py` and `streaming_pipeline.py` are marked `# examples: skip` for the test
harness: the first needs the optional `[ray]` extra and spins up a local cluster, the
second needs a Kafka broker and a Delta sink. Both still show the real API shape — run
`distributed.py` directly to see single-node and distributed produce identical results.

For capabilities that need external infrastructure — cloud object stores, warehouses
(Snowflake/BigQuery), and GPU/model inference — see the
[user guide](../docs/user-guide/) and [ML docs](../docs/ml/), where each API is shown
with the setup it requires.
