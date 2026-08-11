# Examples

500 runnable scripts covering the engine's public surface. Every one executes end to end
against the built engine, asserts on its own output, and exits non-zero if anything is
wrong — so running the whole directory is a release check, not a documentation exercise.

```bash
python examples/quickstart.py
python examples/operations/release_check.py     # the whole engine in one script
```

`tests/docs/test_examples.py` runs all 500 in CI. An example that references a removed or
renamed API fails the suite instead of rotting quietly.

## The data is real

Anything that needs more than a handful of literal rows reads the public TPC-H mirror in
`s3://ray-benchmark-data`, plus a corpus of small JPEGs for the multimodal scripts. Nothing
is synthetic while the network is up.

`examples/_common/` handles the three things the scripts should not each repeat:

- **Canonical column names.** The mirror stores TPC-H positionally (`column0`, `column1`,
  ...) with one trailing all-null column per table, an artifact of the `|`-terminated `.tbl`
  source. The helper restores `l_orderkey`, `o_totalprice` and the rest.
- **A local cache.** The first call downloads a bounded slice and writes it once; every later
  call opens the local Parquet file. Five hundred scripts each re-reading S3 would take
  longer than the check is worth.
- **A loud fallback.** With no network the helper synthesizes a schema-identical stand-in and
  says so on stderr. A quiet fallback would let the corpus rot invisibly.

Scripts reach it with a two-line bootstrap that works both under the runner and when you run
the file directly:

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
```

Environment variables: `BATCHER_EXAMPLES_CACHE` moves the cache, `BATCHER_EXAMPLES_ROWS`
changes the slice size (`full` for whole tables).

## Hardware is optional

Two families would otherwise need hardware CI does not have, so both take a flag and degrade
rather than skip. A check that skips itself checks nothing.

| Family | Default | Opt in |
| --- | --- | --- |
| `gpu/` and the ML device paths | auto: use an accelerator if the engine sees one, else the CPU engine | `--device gpu` (errors if none is visible), `--device cpu`, or `BATCHER_EXAMPLES_DEVICE` |
| `dist/` | single-node, still asserting the mergeable-equivalence contract over several partitions | `--distributed` or `BATCHER_EXAMPLES_DISTRIBUTED=1` |

`--device gpu` on a machine with no accelerator is an error rather than a silent downgrade:
the one time you type it deliberately is the time you need to know it did not happen.

## Where things are

Start with `quickstart.py`, then `tpch/` for real analytics and `relational/` for the core
verbs.

| Directory | Scripts | Covers |
| --- | --- | --- |
| `tpch/` | 30 | all 22 TPC-H queries plus scan costs, join order and pipeline shape |
| `ml/` | 43 | preprocessing, estimators, evaluation, retrieval, embeddings, batch inference |
| `io/` | 39 | every format, cloud paths, globs, partitioning, schema handling, interop |
| `expressions/` | 34 | the per-function expression reference |
| `relational/` | 31 | select, filter, sort, reshape, dedup, set operations, composition |
| `aggregations/` | 27 | every aggregate, grouping, rollup, ratios, ordered and distinct forms |
| `operations/` | 26 | plans, profiling, config, errors, observability, sessions, hardware |
| `expr_text/` | 16 | text quality, cleaning, PII, tokens, similarity, encoding, entities |
| `statistics/` | 16 | dispersion, correlation, quantiles, effect sizes, sampling error |
| `windows/` | 15 | ranking, frames, running totals, gaps-and-islands, leave-one-out |
| `dataset/` | 15 | the Dataset surface itself |
| `joins/` | 14 | every join type, fan-out, keys, collisions, as-of, star schema |
| `perf/` | 14 | pushdown, caching, spill, partitioning, honest measurement |
| `metrics/` | 14 | classification, regression, text and embedding metrics |
| `quality/` | 13 | contracts, profiling, drift, reconciliation, referential integrity |
| `expr_temporal/` | 13 | date parts, truncation, arithmetic, timezones, fiscal calendars |
| `sql_queries/` | 12 | SQL over Datasets, CTEs, windows, dialects, null semantics |
| `geospatial/` | 11 | geometry, predicates, distance, grids, codecs |
| `expr_logic/` | 11 | conditionals, nulls, selectors, horizontal folds, coercion |
| `dist/` | 10 | mergeable equivalence, partitioning, shuffle, transport, fault tolerance |
| `expr_collections/` | 10 | lists, structs, maps, JSON, nesting |
| `expr_numeric/` | 9 | rounding, logs, integer arithmetic, safe division, normalization |
| `gpu/` | 8 | device selection, parity against the CPU oracle, torch inference |
| `lakehouse/` | 8 | Delta commits, upserts, time travel, CDC, compaction, backfill |
| `streams/` | 8 | micro-batches, windows, watermarks, exactly-once, sessions |
| `graph/` | 8 | degree, components, PageRank, shortest paths, projections |
| `text_analytics/` | 7 | word frequencies, n-grams, deduplication, corpus statistics |
| `timeseries_real/` | 7 | resampling, growth, seasonality, cohorts, forecast baselines |
| `security/` | 6 | column masking, row-level security, residency, audit |
| `multimodal/` | 4 | image metadata, decode, blob handling, end-to-end scoring |
| `governance/` | 3 | lineage, masking, PII transforms |
| `expr_vectors/` | 2 | embedding distances, magnitude, pooling |

Sixteen scripts sit at the root as topic overviews (`quickstart`, `sql`, `spill`,
`adaptive_optimization`, and so on).

## Running them

```bash
python examples/tpch/q01_pricing_summary.py           # one script
python -m pytest tests/docs/test_examples.py -q       # all 500, as CI does
```

Two scripts are marked `# examples: skip` and are collected but not executed:
`distributed.py` needs the optional `[ray]` extra and a cluster, and `streaming_pipeline.py`
needs a Kafka broker and a Delta sink. Both still show the real API shape.

A handful of scripts detect a missing optional dependency (`adbc_driver_manager` for the SQL
reader, `xml2arrow` for XML, `openpyxl` for Excel), print which one, and exit cleanly rather
than failing. They exercise fully wherever the extra is installed.

## Writing another one

- Assert on the output. A script that prints and never checks passes while returning
  nonsense.
- Assert order-dependently when the result is ordered. `sorted(x) == x` catches a sort bug;
  a set comparison does not.
- Prefer a real dataset. Reach for literal rows only when three of them make the point better
  than 200,000 would.
- Say something true and non-obvious in the docstring. The engine's behaviour where it
  differs from the obvious guess is the part worth writing down.
