# Examples

Runnable, self-contained scripts for the main ways to use Batcher. Each builds its
own in-memory data and asserts on its output, so you can run any of them directly:

```bash
python examples/quickstart.py
```

Every script here is executed in CI by `tests/docs/test_examples.py`, so an example
that references a removed or renamed API fails the test suite instead of rotting.

| Script | Shows |
| --- | --- |
| `quickstart.py` | the headline lazy pipeline: filter → group_by → agg → sort |
| `transformations_aggregations_joins.py` | the DataFrame core: derive, filter, aggregate, join |
| `window_functions.py` | per-partition aggregates and ranking with `.over(...)` |
| `sql.py` | `bt.sql(...)` over Datasets, composed with the DataFrame API |
| `spill.py` | out-of-core execution under a bounded memory budget |
| `adaptive_optimization.py` | the moat — intra-query re-optimization, result-identical |
| `distributed.py` | the same query single-node vs across Ray workers (needs `[ray]`) |

`distributed.py` is marked `# examples: skip` for the test harness because it needs
the optional `[ray]` extra and spins up a local cluster; run it directly to see
single-node and distributed produce identical results.

For capabilities that need external infrastructure — cloud object stores, lakehouse
tables (Delta/Iceberg/Hudi), warehouses (Snowflake/BigQuery), Kafka/streaming, and
GPU/ML inference — see the [user guide](../docs/user-guide/) and [ML docs](../docs/ml/),
where the API shape is shown with the setup each one requires.
