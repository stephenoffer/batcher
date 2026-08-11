# Topic overviews

This page covers the scripts at the root of `examples/`. Each is a tour of one topic rather
than a focused demonstration, so they are the place to start on an unfamiliar area before
dropping into the per-API scripts.

`quickstart.py` is the headline pipeline: read, filter, group, aggregate, sort. If you run one
script, run that one.

Two of these need setup and are marked `# examples: skip`, so the test runner collects them
without executing. `distributed.py` needs the optional `[ray]` extra and spins up a local
cluster; `streaming_pipeline.py` needs a Kafka broker and a Delta sink. Both still show the
real API shape, and running `distributed.py` directly is the fastest way to see single-node
and distributed produce identical results.

For a single script that touches every subsystem at once, use
`examples/operations/release_check.py` instead. It checks the S3 read path, the scan, the
plan surface, each relational operator, SQL, expressions, data quality, backend parity,
partition parity, spill parity and the write path, and reports which one failed.

## Every script on this page

The table below lists the root scripts in path order.

<!-- library-table: . -->
| Script | Shows |
| --- | --- |
| `examples/adaptive_optimization.py` | Adaptive re-optimization: the moat |
| `examples/data_quality.py` | Data-quality checks: validate, quarantine, drop, and enforce a contract |
| `examples/distributed.py` | Distributed execution: the same code, single-node or on a cluster (needs external setup) |
| `examples/feature_engineering.py` | Feature engineering: derive model-ready columns from raw tabular data |
| `examples/lakehouse_scd.py` | Lakehouse round-trip plus an SCD type-2 history build |
| `examples/ml_inference.py` | Batch inference: score every row with a model-shaped callable |
| `examples/performance_caching.py` | Performance: caching a reused result and spilling under a tiny memory budget |
| `examples/preprocessors.py` | Feature engineering with fit/transform preprocessor objects |
| `examples/quickstart.py` | Quickstart: build a lazy pipeline and run it |
| `examples/spill.py` | Out-of-core execution: bounded memory via spill-to-disk |
| `examples/sql.py` | SQL over Datasets - and blending SQL with Python |
| `examples/streaming_pipeline.py` | Streaming micro-batch pipeline: Kafka in, windowed aggregate, Delta out (needs external setup) |
| `examples/tabular_ml.py` | An end-to-end tabular ML workflow: split, fit, score, evaluate, monitor |
| `examples/timeseries.py` | Time-series patterns: extract date parts, resample, and compute period change |
| `examples/transformations_aggregations_joins.py` | Transformations, aggregations, and joins - the DataFrame core |
| `examples/window_functions.py` | Window functions: per-partition aggregates and ranking |
<!-- /library-table -->
