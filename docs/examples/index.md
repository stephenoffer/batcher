# Example library

This page indexes the 500 runnable example scripts under `examples/`. Every script executes
end to end against the built engine, asserts on its own output, and exits non-zero if
anything is wrong, so running the directory is a release check rather than a documentation
exercise.

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
the mirror does not carry, caches a bounded slice of each table locally so 500 scripts do
not each re-read S3, and falls back to a schema-identical stand-in with a notice on stderr
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

## The sections

Each page below indexes one part of the library and shows code lifted from the scripts it
covers. Blocks that need the S3 corpus are marked `# docs: skip` and are shown rather than
executed; the rest run as part of the documentation build.

| Page | Scripts | Covers |
| --- | --- | --- |
| {doc}`relational` | 102 | Select, filter, join, aggregate, window, reshape |
| {doc}`expressions` | 105 | The expression language and every accessor namespace |
| {doc}`tpch` | 30 | All 22 TPC-H queries, plus scan cost and join order |
| {doc}`io` | 39 | Every format, cloud paths, partitioning, schema handling |
| {doc}`sql` | 12 | SQL over Datasets, CTEs, windows, dialects, null semantics |
| {doc}`machine-learning` | 57 | Preprocessing, estimators, evaluation, retrieval, inference |
| {doc}`multimodal` | 11 | Images, blobs, and text analytics |
| {doc}`accelerators` | 8 | Device selection and parity against the CPU oracle |
| {doc}`distributed` | 18 | Mergeable equivalence, shuffle, streaming |
| {doc}`lakehouse` | 8 | Delta commits, upserts, time travel, CDC, maintenance |
| {doc}`data-quality` | 22 | Contracts, profiling, drift, governance, security |
| {doc}`operations` | 40 | Plans, profiling, configuration, errors, performance |
| {doc}`analytics` | 42 | Statistics, time series, geospatial, graph |
| {doc}`overviews` | 16 | The topic scripts at the root of `examples/` |

```{toctree}
:hidden:

relational
expressions
tpch
io
sql
machine-learning
multimodal
accelerators
distributed
lakehouse
data-quality
operations
analytics
overviews
```
