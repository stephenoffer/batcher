# The engine matrix

This page sets every standard suite against every engine that can run it, one number per cell.

## How to read a cell

Every ratio is a geometric mean over the suite of `batcher_ms / engine_ms`, counted only over cases that passed the correctness gate. The following table explains the other cell values:

| Cell | Meaning |
|---|---|
| A ratio | Below 1.00 means Batcher is faster. |
| `--` | The engine can't express this suite, because it has no SQL surface or rejects the SQL. Not a loss. |
| `killed` | The engine took the benchmark process down on this suite. Not a ratio. |
| `n/r` | Not run in this sweep. |

## The five-engine board

Taken 2026-08-28 on a 92-core box, best of five at sf1 and best of three at sf10, one process per suite, every case correctness-gated against DuckDB:

| Suite | DuckDB native | DuckDB same Arrow | Polars | Daft | PyArrow |
|---|---:|---:|---:|---:|---:|
| JSON | **0.25** | **0.17** | **0.01** | **0.04** | `--` |
| ClickBench | **0.63** | **0.16** | **0.31** | **0.11** | `--` |
| Operators | **0.67** | **0.36** | **0.12** | **0.07** | **0.03** |
| TPC-H sf1 | **0.74** | **0.26** | **0.44** | **0.21** | `--` |
| TPC-DS sf1 | **0.92** | `killed` | `--` | `n/r` | `--` |
| TPC-H sf10 | 1.10 | **0.33** | **0.35** | **0.17** | `--` |
| H2O.ai `groupby` | 1.10 | **0.82** | **0.41** | **0.38** | `--` |
| H2O.ai `join` | 1.02 | **0.88** | **0.70** | **0.34** | `--` |

Every like-for-like bar goes to Batcher. Later sweeps measured TPC-H sf10 at **0.963x** against the native store (2026-08-25) and H2O.ai `join` at **0.63x** (2026-09-13). {doc}`/benchmarks/index` carries that newer board.

Spark isn't on this board. On TPC-H sf1, local-mode Spark ran 20x to 50x behind Batcher (2026-08-15, 96-core box). {doc}`/benchmarks/comparisons/vs-spark` has the detail.

## Why some cells are empty

- **PyArrow has no SQL surface**, so it runs only the operator mix.
- **Polars' SQL frontend rejects comma joins**, so it runs none of TPC-DS. Its TPC-H column is measured through the DataFrame API.
- **DuckDB over registered Arrow** and **Daft** don't complete TPC-DS sf1 in the benchmark process. DuckDB on its native store does.
- Cases where an engine's result disagrees with the others are excluded from that engine's geomean.

## Reproduce

Run one suite per invocation, pairwise or with the lineup you want to compare:

```bash
python benchmarks/run.py --benchmark <suite> --engines batcher,duckdb,duckdb_arrow,polars,daft,pyarrow
```

TPC-DS sweeps run pairwise.

## See also

- {doc}`/benchmarks/methodology`: the correctness gate and the hardware behind each board.
- {doc}`/benchmarks/comparisons/index`: one page per engine, with the architectural reason behind each result.
- {doc}`/benchmarks/results/scaling`: how these results move with the data and across a cluster.
- {doc}`/benchmarks/results/tpch`: the TPC-H suite query by query.
