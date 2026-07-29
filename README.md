# Batcher

**One data engine that runs on your laptop and scales to a cluster — and tunes
itself while your query runs.**

Most data tools make you choose: fast on a single machine, or able to scale across
many — rarely both. So teams outgrow their tool and rewrite the pipeline, or run one
system for SQL, another for DataFrames, and a third for ML and pay to keep the seams
from leaking. Batcher is a single engine for all of it: quick on small data, steady
at large scale, for SQL, DataFrame, and ML workloads. The same code runs on one core
or a thousand, so going bigger is a config change, not a rewrite.

```python
import batcher as bt

revenue = (
    bt.read("s3://events/*.parquet")
    .filter(bt.col("status") == "active")
    .group_by("region")
    .agg(total=bt.col("amount").sum())
    .sort("total", descending=True)
)
print(revenue.to_pydict())   # nothing runs until here
```

## What makes it different

Most engines decide how to run your query before they have looked at any data, then
stick with that plan even when the data turns out different — which is the usual
reason a job stalls or runs out of memory. Batcher watches the data as it flows and
adjusts the plan as it goes, so a query that starts on a bad guess corrects itself
instead of failing. No other engine does this *during* a query.

## How it compares

| Tool | Where it stops | What Batcher does instead | Measured |
|------|----------------|---------------------------|----------|
| **DuckDB** | fast, but single-node and plans once | scales out, and re-optimizes mid-query | **won 22/22** TPC-H and **42/43** ClickBench on the same Arrow |
| **Polars** | fast, but single-node | the same code runs from one core to a cluster | **12×–81×** on JSON; **7×–33×** on windows and top-N |
| **Daft** | scales, but plans once | adaptive re-optimization, and a correct q6 | **2.4× faster** cluster-vs-cluster, and Daft's q6 is wrong |
| **Spark** | scales, but heavy on small jobs | runs in-process locally — no cluster to spin up | **5×–33×** on TPC-H |

The speed is measured **correctness-first**: the harness refuses to time a query whose result
doesn't match the oracle, so a fast wrong answer never counts as a win. That gate has caught
real bugs in other engines — on TPC-H q6, Daft and Polars both return `75,207,768.19` where the
official answer is `123,141,078.2283`, and Batcher returns the official answer exactly.

Two places Batcher does **not** win, stated up front: DuckDB reading its own compressed store
still leads on join-heavy TPC-H (~1.4× geomean), and high-concurrency serving is not this
engine's shape at all. Both are detailed below.

## Benchmarks

Numbers, not adjectives — and every one is **correctness-gated**: the harness runs each query
on every engine, checks they return the *identical* result (a sorted row multiset within float
tolerance), and only then trusts the timing. A fast wrong answer is a bug, not a win. Setup:
TPC-H `lineitem` (6M rows at scale 1, 60M at scale 10), read once into Arrow and shared
byte-identically across engines; a 9-node / 128-CPU cluster; 8×T4 GPUs for the ML runs. Full
per-scale tables: [`benchmarks/BENCHMARK_RESULTS.md`](benchmarks/BENCHMARK_RESULTS.md).

**Single-node, head-to-head on the same in-memory Arrow (TPC-H sf1).** Each engine runs the
*identical* zero-copy Arrow input Batcher runs on, so this is a like-for-like **execution**
comparison. Cells are **how many times faster Batcher is** (higher = Batcher faster; a `<1`
value means the competitor is faster, shown as e.g. `0.5× (2× slower)`). Every row is
correctness-gated against DuckDB:

| operator | vs DuckDB | vs Polars | vs PyArrow | vs Spark² |
|-------------------------|:--------:|:--------:|:----------:|:--------:|
| filter → count          | **265×** | **41×**  | **1225×** | **125×** |
| global sum              | **33×**  | **12×**  | **25×**   | **197×** |
| group-by sum (1 key)    | **5.0×** | **2.6×** | **2.0×**  | **28×**  |
| group-by sum (2 keys)   | **4.3×** | **2.7×** | **2.0×**  | **21×**  |
| filter → project        | **3.8×** | 0.8× (1.3× slower) | **14×** | **20×** |
| window running `sum()`  | **2.6×** | **6.3×** | n/a¹      | **17×**  |
| window `lag()`          | **1.9×** | **25×**  | n/a¹      | **13×**  |
| window `rank()`         | **1.4×** | **6.7×** | n/a¹      | **19×**  |
| join → group-by         | **1.4×** | 0.9× (1.1× slower) | **3.6×** | **25×** |
| window whole-partition `sum()` | **1.1×** | 1.0× (1.02× slower) | n/a¹ | —        |
| sort → top-N (`LIMIT`)  | **1.0×** | **33×**  | **180×**  | **24×**  |

**Batcher wins all 11 against DuckDB, all 7 against PyArrow, and 8 of 11 against Polars.**

¹ PyArrow (Acero) has no window functions. ² The DuckDB, Polars and PyArrow columns were
re-measured 2026-07-18 on a release build; the Spark column is from the dated runs in
[`benchmarks/BENCHMARK_RESULTS.md`](benchmarks/BENCHMARK_RESULTS.md).

**Three full suites, not just an operator mix.** Every engine reads the identical zero-copy
Arrow input, so these compare *execution*, not storage formats:

| suite | result vs DuckDB on the same Arrow |
|---|---|
| **TPC-H** — all 22 queries | **won 22 of 22**, 1.1×–6.9× faster |
| **ClickBench** — 43 queries | **won 42 of 43**, 43/43 correct |
| **Semi-structured JSON** — 5 queries | **won 5 of 5**, 3.5×–12.7× faster (**12×–81×** vs Polars) |

Standout queries: TPC-H q15 **6.9×**, q11 **6.8×**; ClickBench q27 **37×**, q40 **16×**.
Correlated subqueries now run, so q21 is comparable — and Batcher wins it 1.9×. Batcher also
beats Spark on every TPC-H query (5×–33×).

**Correctness is a result too.** On TPC-H q6, Batcher returns the official answer
(`123,141,078.2283`); **Daft and Polars both return `75,207,768.19`**, because they fold the
bound `0.06 + 0.01` in IEEE double to `0.06999999999999999` and drop every `l_discount = 0.07`
row. Daft additionally cannot complete q18, q21, or q22.

**Cluster vs cluster.** On an 8-node / 128-CPU cluster with *both* engines distributed and
reading the same S3 parquet, TPC-H sf10 q6: Batcher **224 ms** vs Daft **536 ms** — **2.4×
faster, and correct where Daft is not**.

**How it scales, kept honest.** Against Polars, PyArrow, and Spark, Batcher leads at every
scale we measured. Against DuckDB the story is scale-dependent, and we report it straight:

| scale (single node) | Batcher vs DuckDB (same-input execution) |
|---------------------|-------------------------------------------|
| **sf1** — 6M rows, in memory   | wins **all 22** TPC-H queries (1.1–6.9×) |
| **sf10** — 60M rows, in memory | wins 15 of 21; trails on 6 aggregate/join-heavy queries (1.2–3×) |
| **sf100** — 600M rows, scanned | DuckDB leads 2–11×; Batcher OOMs on the deepest join trees (q3/q4/q5) |

The gap grows with scale for three structural reasons, none of them a tuning knob: DuckDB
**decompresses its native store on the fly** (fewer bytes off memory — Batcher's Arrow-only
contract has no compressed form to read), its **vector-at-a-time engine with selection vectors**
edges Batcher's batch-at-a-time kernels as rows grow, and it **streams** where Batcher's model
materializes each operator's output — which is what OOMs the largest single-node sf100 joins.
Batcher's answer at that scale is **distribution**: the same mergeable operators shard across a
cluster (one partition per node, bounded per-node memory), which is the regime it is built for.
Closing the single-node scale gap to DuckDB is honest, open work — vectorized kernels,
dictionary-aware grouping, and streaming between operators.

**Distributed data plane.** Execution is in-process and native, so a distributed operator costs
no per-operation task-scheduling hop and no pandas bridge. Bulk Arrow batches move worker to
worker over Arrow Flight with credit-based flow control, bypassing the Ray object store, and
streaming `map_batches`, row-exploding `flat_map`, chained multi-stage maps, Parquet reads, and
`iter_torch_batches` training ingest all run on that same path.

**GPU batch inference (8×T4)** — one of the workload families the same engine runs. Stage-overlap
streaming and session-warm pools, which load a model once per session rather than once per job,
are what produce these figures:

- **LLM batch inference** (gpt2 generate) — 814 prompt/s
- Batch inference (ResNet-50) — 2576 img/s at 78% GPU utilization
- Batch embeddings (2048-d vectors) — 2502 img/s at 80% GPU utilization
- Zero-config `map_batches(Model, num_gpus=1)` — 2451 img/s at 82% GPU utilization

Stage-overlap alone lifted a decode → ResNet-50 pipeline from **942 → 2504 img/s** and GPU
utilization from **~30% → 81%** — same result, the device just stops idling through the CPU decode.

**Why the wins happen** — they're structural, not tuning: an in-process native engine over Arrow
(no per-operation scheduler / object-store hop), a high-cardinality aggregation path that hashes
native and composite keys directly and merges its radix partitions in parallel without copying the
key column twice — so `GROUP BY` and `COUNT(DISTINCT)` over string or near-unique keys scale, and
`DISTINCT` shards across every core like the aggregate above it — warm model pools +
stage-overlapped streaming for GPU work, and adaptive re-optimization that re-tunes the plan on
measured cardinalities mid-query.

## Install

Prebuilt wheels ship for Linux, macOS, and Windows on Python 3.10+ — no Rust needed.
Batcher is on PyPI as `batcher-engine` and imported as `batcher` (the bare `batcher`
name belongs to an unrelated project):

```bash
pip install batcher-engine
```

```python
import batcher as bt

ds = bt.from_pydict({"x": [1, 2, 3]})
print(ds.select(doubled=bt.col("x") * 2).to_pydict())
# {'doubled': [2, 4, 6]}
```

Optional features are extras, e.g. `pip install "batcher-engine[ray,cloud]"`.

New here? The [documentation](https://stephenoffer.github.io/batcher/) has a
quickstart, guides, and runnable examples.

## Under the hood

You write Python; the heavy lifting happens in Rust over [Apache
Arrow](https://arrow.apache.org/). Python builds and optimizes your query, and Rust
runs it — so you get Python's ergonomics with native speed. The same engine powers
one core and a whole cluster, which is why a result is identical whether it ran on
your laptop or a hundred machines. The full design (and the math behind the
optimizer) is in the
[documentation](https://stephenoffer.github.io/batcher/) and `architecture.txt`.

> **Status:** young but working, not yet 1.0. Batcher runs SQL and DataFrame
> workloads, single-node and distributed, and is benchmarked for correctness and
> speed against DuckDB and Polars. Expect APIs to change, and some operators and
> large-scale paths are still landing.

## Build from source

Requires a [Rust toolchain](https://rustup.rs):

```bash
uv venv && source .venv/bin/activate
uv pip install -e '.[dev]'
maturin develop           # build the Rust engine into the venv
pytest
```

Or install an unreleased revision directly:
`pip install "git+https://github.com/stephenoffer/batcher.git"`.

On an ephemeral dev box — an Anyscale workspace, a preemptible VM, a rebuilt container —
`rustup`'s default install lives in `$HOME` and does not survive the machine being replaced,
so `cargo` and `just` disappear and the repo stops building for a reason nothing names.
`tools/bootstrap_env.sh` reinstalls the toolchain onto fast local disk from a tarball cached
on durable storage and exports `CARGO_HOME`/`RUSTUP_HOME`/`CARGO_TARGET_DIR`. It is a no-op
once the toolchain is present, so it is safe to source from a shell startup file:

```bash
source tools/bootstrap_env.sh
```

## Layout

- `python/batcher/` — the Python API
- `crates/` — the Rust engine
- `docs/`, `architecture.txt` — design and documentation
- `MAP.md` — the file-level index: what every module and crate file is for, and
  where new code goes. Generated by `just map` from the code's own docstrings, so
  it stays true. Start here when finding your way around.

Apache-2.0 licensed.
