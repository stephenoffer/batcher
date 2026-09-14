# Example library

This page indexes the 512 runnable example scripts under `examples/`. Every one of them
executes end to end against the built engine, asserts on its own output, and exits non-zero
if anything is wrong. Running the directory is a release check, not a documentation exercise.

The tables on these pages are generated from the scripts themselves by
`python tools/example_library.py`, so the library cannot drift from the tree. The prose
around them is written by hand.

```bash
python examples/quickstart.py
python examples/operations/release_check.py
python -m pytest tests/docs/test_examples.py -q
```

## What the scripts read

Anything needing more than a handful of literal rows reads the public TPC-H mirror in
`s3://ray-benchmark-data`, plus a corpus of small JPEGs for the multimodal scripts. Nothing
is synthetic while the network is up.

The shared helper in `examples/_common/` restores the canonical TPC-H column names, which
the mirror does not carry, caches a bounded slice of each table locally so five hundred
scripts do not each re-read S3, and falls back to a schema-identical stand-in with a notice on stderr
when there is no network. Point the cache elsewhere with `BATCHER_EXAMPLES_CACHE`, or take
more rows with `BATCHER_EXAMPLES_ROWS`.

Scripts reach the helper with a two-line bootstrap that works both under the test runner and
when you run the file directly:

```python
# docs: skip
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch

orders = tpch("orders")
```

## Hardware is optional

Two families would otherwise need hardware that CI does not have. Both take a flag and
degrade rather than skip, because a check that skips itself checks nothing.

| Family | Default | Opt in |
| --- | --- | --- |
| `examples/gpu/` and the ML device paths | Auto: use an accelerator when the engine sees one, the CPU engine otherwise | `--device gpu`, `--device cpu`, or `BATCHER_EXAMPLES_DEVICE` |
| `examples/dist/` | Single node, still asserting mergeable equivalence across partitions | `--distributed` or `BATCHER_EXAMPLES_DISTRIBUTED=1` |

Asking for `--device gpu` on a machine with no accelerator is an error rather than a silent
downgrade. The one time you type it deliberately is the time you need to know it did not
happen.

## Start at the root

The scripts at the root of `examples/` are tours of one topic rather than focused
demonstrations, so they are where to start on an unfamiliar area before dropping into the
per-API scripts.

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

## The sections

Each page below indexes one part of the library and shows code lifted from the scripts it
covers. Blocks that need the S3 corpus are marked `# docs: skip` and are shown rather than
executed; the rest run as part of the documentation build.

| Page | Scripts | Covers |
| --- | --- | --- |
| {doc}`relational` | 115 | Select, filter, join, aggregate, window, reshape, and the same plans as SQL |
| {doc}`expressions` | 101 | The expression language and every accessor namespace |
| {doc}`tpch` | 30 | All 22 TPC-H queries, plus scan cost and join order |
| {doc}`io` | 47 | Every format, cloud paths, partitioning, schema handling, Delta commits |
| {doc}`machine-learning` | 57 | Preprocessing, estimators, evaluation, retrieval, inference |
| {doc}`multimodal` | 11 | Images, blobs, and text analytics |
| {doc}`accelerators` | 8 | Device selection and parity against the CPU oracle |
| {doc}`distributed` | 18 | Mergeable equivalence, shuffle, streaming |
| {doc}`data-quality` | 23 | Contracts, profiling, drift, governance, security |
| {doc}`operations` | 40 | Plans, profiling, configuration, errors, performance |
| {doc}`analytics` | 46 | Statistics, time series, geospatial, graph |

```{toctree}
:hidden:

relational
expressions
tpch
io
machine-learning
multimodal
accelerators
distributed
data-quality
operations
analytics
```
