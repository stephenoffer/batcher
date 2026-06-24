# Batcher benchmark suite

A correctness-first comparison of the Batcher data engine against DuckDB and
Polars on a representative mix of analytical queries, plus a TPC-H subset, an
optimizer-scaling micro-benchmark, a shuffle-transport benchmark, and a
distributed equivalence benchmark.

Correctness is checked before any timing is trusted. A query is only timed once
its result matches the reference engine, so a fast wrong answer can never be
reported as a win.

## Layout

The comparison suite is a registry of benchmarks grouped one module per query
family, so it stays clean as it grows from tens to thousands of queries:

```
benchmarks/
  harness.py     correctness check + best-of-N timing (the measurement core)
  registry.py    the Benchmark registry and the suite(...) decorator
  contexts.py    data-context builders (SyntheticContext, TpchContext)
  suites/        one module per query family, each registering its cases
    filtering.py  projection.py  aggregation.py  joins.py  setops.py
    ordering.py   window.py      strings.py      mathfns.py dates.py
    conditional.py  tpch.py
  run.py         the CLI: select, run, report
  distributed.py             single-node == many-partition equivalence + timing
  optimizer_bench.py         Kyber planning latency as the rule set grows
  shuffle_vs_object_store.py Arrow Flight shuffle vs the Ray object store
```

### Adding a benchmark

Add a `@<family>.case("name")` function to the matching module in `suites/`. It
receives the prepared data context and returns one callable per engine (or `None`
for an engine that does not express the query):

```python
# suites/joins.py
from registry import suite

joins = suite("joins", dataset="synthetic")

@joins.case("join-left")
def left_join(ctx):
    return {
        "batcher": lambda: ctx.bf.join(ctx.bd2, on="dim_key", how="left").collect(),
        "duckdb":  lambda: ctx.con.sql("...").to_arrow_table(),
        "polars":  lambda: ctx.pf.lazy().join(ctx.pd_dim2.lazy(), on="dim_key", how="left").collect().to_arrow(),
    }
```

To add a family, create a module in `suites/` and add it to `load_all` in
`suites/__init__.py`. To add a dataset (a new set of input tables), add a context
class to `contexts.py` with a `build` classmethod and teach `run.py` how to size
it. No file grows without bound and there is no per-query file explosion.

## Running

All runners assume the engine is built and the benchmark dependencies are present:

```bash
source .venv/bin/activate          # batcher, duckdb, polars, numpy, pyarrow
just build                         # or: maturin develop --release
```

```bash
python3 benchmarks/run.py                  # operator mix, ~10M synthetic rows
python3 benchmarks/run.py 2000000          # custom synthetic row count
python3 benchmarks/run.py --dataset tpch   # TPC-H subset (default sf 0.1)
python3 benchmarks/run.py --dataset tpch --sf 1.0
python3 benchmarks/run.py --dataset all    # both datasets
python3 benchmarks/run.py --family joins   # one family only
python3 benchmarks/run.py --only window    # cases whose name matches
python3 benchmarks/run.py --list           # list registered benchmarks, do not run
```

The `just` shortcuts are `just bench`, `just bench-tpch`, `just bench-all`,
`just bench-list`, and `just bench-dist`.

The same logical operation is expressed once per engine as a callable returning a
`pyarrow.Table`. The harness (`harness.py`):

1. Verifies correctness first. Every engine's output is compared to a reference
   engine as a sorted row multiset (row order and column order normalized away),
   tolerant of float rounding and of DuckDB's `Decimal` sums vs. Polars/Batcher
   floats. A mismatch marks the row `FAILED` and prints a diff; it does not abort
   the suite.
2. Times best-of-N wall-clock in milliseconds after one warm-up run (`runs=3` at
   5M rows or more, `5` below). Each engine's run is wrapped so that one engine
   failing records an error rather than crashing the run, and an unsupported case
   can be marked `n/a`.
3. Reports an aligned table: `query | batcher_ms | duckdb_ms | polars_ms |
   ratio(batcher/duckdb) | status`.

### Data

`make_data()` (numpy, fixed seed) builds:

- fact (`~scale` rows): `id, k1 (1k distinct), k2 (50 distinct), dim_key (1k
  distinct), x (int), price (float), qty (int), category (string), ts (timestamp
  over three years)`.
- dim (1000 rows): `dim_key, region (string), weight (float)`.

### Queries

| query | shape |
|-------|-------|
| `filter+count` | `COUNT(*) WHERE x > 0` |
| `projection` | compound arithmetic `price*qty - x/2`, summed |
| `groupby-agg` | `GROUP BY k1` to sum/count/avg/min/max |
| `groupby-2key` | `GROUP BY k1, k2` to sum/count |
| `join+groupby` | `fact join dim ON dim_key`, then `GROUP BY region` |
| `sort+limit(top20)` | `ORDER BY price DESC, id ASC LIMIT 20` |
| `distinct` | `DISTINCT (k1, k2)` |
| `window(rn+psum)` | `row_number()` + partition `SUM` over `k2`, keep top-3 per partition |
| `string-filter` | `COUNT(*) WHERE category LIKE '%eta%'` (`str.contains`) |
| `join-left` | left join to a half-size dim, count rows and non-null regions |
| `join-semi` | semi join: fact rows whose key is present in the dim subset |
| `join-anti` | anti join: fact rows whose key is absent from the dim subset |
| `union` | `UNION ALL` and `UNION` over two overlapping id subsets |
| `intersect` | distinct `k1` present in both halves of the table |
| `except` | distinct `k1` that never appear in the `k2 < 25` half |
| `string-upper` | `GROUP BY upper(category)` |
| `math-agg` | integer-exact `SUM(abs(x))` and `SUM(floor(price))` |
| `case-band` | bucket `price` into bands with `CASE`, count per band |
| `date-year` | `GROUP BY EXTRACT(year FROM ts)` |
| `window-frame` | trailing 3-row moving `SUM` with an explicit `ROWS` frame (batcher vs duckdb) |

Several queries aggregate to a single scalar so the per-row work runs at full
scale while the comparison set stays tiny and deterministic. The sort and window
queries use a deterministic tie-break (`id ASC`) so the top-N is identical across
engines despite float ties. `window-frame` is expressed for Batcher and DuckDB
only, since Polars does not express `ROWS` frames cleanly inside `over()`.

## The TPC-H subset

```bash
python3 benchmarks/run.py --dataset tpch          # scale factor 0.1
python3 benchmarks/run.py --dataset tpch --sf 1.0 # scale factor 1.0
```

Data is generated by DuckDB's built-in TPC-H `dbgen` (no network access needed)
and exported to Arrow. To keep the three engines on identical inputs, decimal
columns are cast to `float64` and date columns to `timestamp[us]` once, up front,
and all three engines read the same Arrow tables. The subset is Q1 (grouped
aggregate), Q3 (three-way join with grouped top-10), Q6 (selective scalar
aggregate), and Q10 (four-way join with grouped top-20). The date predicates
follow the TPC-H spec.

## distributed.py: single-node vs many-partition equivalence

```bash
python3 benchmarks/distributed.py            # ~2M rows, 8 partitions
python3 benchmarks/distributed.py 5000000 16
```

Each query runs single-node and again across several partitions via
`collect(distributed=True, num_partitions=...)`. The mergeable algebra
(`partial / combine / finalize` over a hash shuffle) guarantees the two results
are identical, so the benchmark asserts that equivalence first and only then
reports timings. A divergence is a correctness bug and fails the run.

At small local scales the distributed path is slower than single-node because the
partition and shuffle overhead dominates. The value here is the equivalence
guarantee and the scaling shape, not a single-machine speedup. Multi-node
throughput at large scale factors depends on network and cluster size; the engine
keeps per-node memory bounded through the mergeable algebra and spill, and moves
batches over Arrow Flight with credit-based backpressure rather than through the
Ray object store. Published multi-node throughput numbers are not yet measured,
so none are quoted here.

## Reading the numbers

`b/duck` is `batcher_ms / duckdb_ms` (lower means Batcher is faster). Timings vary
run to run; treat them as order-of-magnitude. The status column is the gate: only
`OK` rows have been verified to match the reference engine.

## Daft head-to-head

`daft_compare.py` runs core relational queries on Batcher vs Daft (and DuckDB/Polars)
over the same synthetic data, correctness-gated against DuckDB, reporting best-of-5 ms
and the Batcher/Daft ratio. **Build the release engine first** (`just build-release`).

    python benchmarks/daft_compare.py [n_rows]

