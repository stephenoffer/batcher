# The engine matrix

This page sets every standard suite against every engine that can run it, one number per cell, with each gap labeled by kind. A page about one engine can flatter by what it leaves out, so this one publishes the board whole.

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

Every like-for-like bar goes to Batcher, and so does every column that isn't DuckDB's native store. The three suites above 1.00 have since moved. TPC-H sf10 read 0.963x against the native store on 2026-08-25 on a 96-core node, and the 2026-09-13 board has H2O.ai `join` at 0.63x and `groupby` at 1.05x. {doc}`/benchmarks/index` carries that newer board.

Spark isn't on this board. On TPC-H sf1, measured 2026-08-15 on a 96-core box after its adapter stopped round-tripping results through pandas and got one shuffle partition per core, local-mode Spark ran 20x to 50x behind Batcher. {doc}`/benchmarks/comparisons/vs-spark` has the detail.

## What the gaps are

**PyArrow has no SQL surface.** A case exists for it only where the work can be written against `Table` and `compute`, which means a single operator or a short chain of them. The operator mix is that, and the multi-table SQL suites aren't.

**Polars' SQL frontend rejects comma joins.** `SELECT ... FROM a, b WHERE a.k = b.k` fails with `multiple tables in FROM clause are not currently supported`, so Polars runs none of the TPC-DS queries. Its DataFrame API isn't affected, and the TPC-H column is measured through it.

**DuckDB over registered Arrow can't finish TPC-DS.** It is killed on q64 at 132 GB resident on a scale-factor-1 dataset, reproduced on a busy box and an idle one. On the same query Batcher returns in 3.2 ms and DuckDB on its native store in 58.4 ms. A killed process can't be caught as an exception, so the column is dropped from the suite rather than the suite from the board.

**Daft doesn't complete TPC-DS in the same process.** A run with Daft in the lineup is killed, and the same lineup without Daft completes all 99 queries.

## Disagreements, and whose fault they are

A disagreement isn't always an engine's defect, and the difference decides who should fix it.

**An engine disagreeing with the rest.** On TPC-H, Daft returns wrong results on q6 and q15 and the wrong columns on q18, where Batcher, DuckDB and Polars agree. Those cases are excluded from Daft's geomean rather than counted for it, so it isn't credited with a fast wrong answer.

**The benchmark's fault.** The three `scan-filter_agg-*` cases disagree for Ray Data and Daft, and the suite predicts it. Its columns are `int64` drawn uniformly from `[0, 2^63)`, so a bare sum overflows 64 bits. DuckDB and Batcher widen before summing and return about 4.6e18, while engines that accumulate in `int64` wrap. The suite bounds every `SUM` for that reason and missed its one `AVG`, which sums before it divides. Those three cases are uncomparable, and the fix belongs in `benchmarks/suites/scan/shapes.py`.

## Reproduce

Run one suite per invocation, pairwise or with the lineup you want to compare:

```bash
python benchmarks/run.py --benchmark <suite> --engines batcher,duckdb,duckdb_arrow,polars,daft,pyarrow
```

Put a large lineup on TPC-DS with care. Five engines holding TPC-DS sf1 in one process were killed at 71 GB, which is why the TPC-DS sweeps run pairwise.

## See also

- {doc}`/benchmarks/methodology`: the correctness gate and the hardware behind each board.
- {doc}`/benchmarks/comparisons/index`: one page per engine, with the architectural reason behind each result.
- {doc}`/benchmarks/results/scaling`: how these results move with the data and across a cluster.
- {doc}`/benchmarks/results/tpch`: the TPC-H suite query by query.
