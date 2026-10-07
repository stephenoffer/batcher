# What makes Batcher different

This page describes the design decisions that separate Batcher from DuckDB, Polars, Spark, and Ray Data, and what each one buys.

Plenty of engines are fast at one shape of work. {doc}`../benchmarks/index` has the measured numbers for that. The question here is which properties survive when the work changes: the data outgrows a laptop, the pipeline has to feed a model, or the same query runs every hour for a year. Six decisions account for most of the answer, and every claim below is checked against the code that implements it.

## One algebra from one core to a cluster

Every stateful operator is written once, in `bc-runtime`, as three functions. `partial(batch)` produces a state and `combine(states)` merges states; `finalize(state)` then emits rows. Because `combine` is associative and commutative, partial states merge in any order.

That one implementation runs sequentially on a core, in parallel across many, and across a cluster over Arrow Flight. No second distributed operator exists to drift. The rows, column names and column types come back the same on one node and on a hundred, and a floating-point reduction agrees up to the reassociation a different partition count causes. [`tests/integration/test_distributed.py`](https://github.com/stephenoffer/batcher/blob/main/tests/integration/test_distributed.py) asserts that operator by operator against a local Ray instance, and the runs recorded on real clusters back it with larger data.

Scaling out is a scheduling decision. The script you wrote on a laptop is the script that runs on the cluster:

```python
# docs: run
import batcher as bt

sales = bt.from_pydict({"store": [1, 2, 1, 3], "amount": [30, 10, 50, 20]})
by_store = sales.group_by("store").agg(total=bt.col("amount").sum()).sort("store")
print(by_store.collect(distributed=False).to_pydict())
# {'store': [1, 2, 3], 'total': [80, 10, 20]}
```

```python
# docs: skip
by_store.collect(distributed=True, num_workers=8)  # same rows, eight workers
```

The constraint falls on every new operator. A stateful operator with no mergeable form would be capped at one machine, so every operator is held to the form.

## Speed without a second set of semantics

Batcher has exactly one scalar expression type and one relational plan type, and every execution path consumes the same ones.

![One shared Expr and RelOp feeding three execution tiers. The Tier-0 sequential interpreter is the correctness oracle. The Tier-0 parallel path changes only scheduling and must equal the oracle. The Tier-1 Cranelift JIT must be bit-for-bit identical on its supported subset, and an unsupported expression falls back to the interpreter rather than diverging.](/_static/diagrams/execution_tiers.svg)

The sequential interpreter is the correctness oracle. It is kept simple and deterministic, and everything else is checked against it. The parallel path reuses its operator code and changes only the scheduling. The Cranelift JIT compiles a scalar expression once, caches it process-wide, and reuses it on every morsel.

The dashed edge matters most. An expression the JIT doesn't support isn't an error or a slow compile. It runs on the interpreter. The JIT must match the interpreter bit for bit on its subset and decline everything else, so performance work here doesn't pile up risk. That subset is fixed-width numeric expressions: arithmetic, comparisons including dates and timestamps, boolean logic with SQL's three-valued nulls, and `CASE`. Strings, lists and most functions run on the interpreter with the same result.

## A learned loop that outlives the query

This is the differentiator most often overstated, so here it is at full precision.

![A capability matrix comparing DuckDB, Spark AQE, and Batcher on three properties: re-planning inside one query, running inside the Python process, and carrying what was learned into the next run. DuckDB runs embedded in-process but optimizes once and keeps no cross-run state. Spark AQE, on by default since Spark 3.2, re-plans at stage boundaries, but from a JVM beside Python, and keeps no cross-run state. Batcher re-plans at the same stage-boundary granularity, runs the loop inside the Python process, and carries sketches, calibrated costs, and a bandit into the next run.](/_static/diagrams/adaptive_positioning.svg)

During a query, Batcher re-optimizes at pipeline breakers on cardinalities it has *measured*. When an estimate was wrong by more than `optimizer.reoptimize_error` (2.0 by default), Kyber re-plans the rest of the query on the real numbers. That is stage-boundary re-optimization, the same granularity Spark AQE works at.

Two things set it apart. It runs inside the Python process: AQE has been on by default since Spark 3.2 and also re-plans in local mode (`local[*]`), but it plans in a JVM beside Python, and DuckDB optimizes once with no way to revise. And what it measured survives the query. Core records actual cardinalities, operator times and peak memory into the `MetadataHub`, and the next run of that plan shape reads them: HyperLogLog distinct counts, KLL quantiles, cost coefficients calibrated from measured operator times, and a UCB1 bandit over equivalent join strategies. Neither DuckDB nor open-source Spark feeds measured execution statistics from one run into the next run's plan.

You can watch an estimate change source after one run:

```python
# docs: run
events = bt.from_pydict({"sensor_id": [1, 2, 3, 4, 5, 6], "reading": [4, 8, 15, 16, 23, 42]})
hot = events.filter(bt.col("reading") > 10)
assert "(default)" in hot.explain()
hot.collect()
assert "(learned)" in hot.explain()
```

The within-query loop engages only where it pays. On a single node, `adaptive="auto"` needs a join and a size floor charged per breaker the loop would cut at, because each cut costs a materialization and a re-plan and gives up fusion at that boundary. The floor is 5 million rows, or roughly 320 MB, per breaker. The simplest joined shape has two breakers and qualifies at about 10 million rows; a snowflake with six needs 30 million. Below that, the one-shot plan is faster and is what runs. On a cluster, a plan the one-shot dispatcher can't run correctly, such as an aggregate over a `limit`, takes the staged path at any size.

## The data plane does not touch the object store

On a cluster, Ray schedules tasks and carries control-plane metadata. That's all. Only small `(address, ticket)` strings travel through Ray. Bulk Arrow batches move straight between workers over Arrow Flight under credit-based flow control: one credit is one in-flight batch slot, and a producer blocks at zero credits. An in-flight gauge in `bc-transport` enforces the bound.

Put an object store in the data path and memory pressure turns into spill storms. Ray Data's shuffle goes through its object store and Batcher's does not, and Batcher's distributed shuffles run far ahead of Ray Data's. No benchmark isolates how much of that margin the transport accounts for, and the design isn't unique to Batcher: Daft offers an Arrow Flight shuffle (`shuffle_algorithm="flight_shuffle"`) too.

Within a node, the transport picks the cheapest tier itself. In the same process it reads straight from the local store. Across processes on one node it memory-maps a 64-byte-aligned Arrow IPC file, roughly 23x faster than a loopback Flight hop, and steps aside under memory pressure. Flight carries only what crosses nodes. Published shuffle output sits in RAM with a spill path behind it, so a reducer usually reads from memory.

## Batch, streaming, and models are one engine

Seen from the engine, batch is the bounded special case of streaming over Arrow batches. The same operators process record batches either way, and the same pipeline breakers are where a streaming query checkpoints and where the adaptive layer re-plans.

Model work runs here too. Images, audio and video decode into tensor columns the relational operators already understand, so one pipeline can filter a table, join it, and feed a model with no hand-off. On the GPU path, the CPU decode of the next morsel runs while the current forward pass is in flight. A two-stage ResNet-50 pipeline on 8 T4 GPUs went from 942 to 2,504 images per second that way, with GPU utilization rising from about 30% to 81% ([`benchmarks/BENCHMARK_RESULTS.md`](https://github.com/stephenoffer/batcher/blob/main/benchmarks/BENCHMARK_RESULTS.md)).

A model step is a class handed to `map_batches`, constructed once per worker:

```python
# docs: run
import pyarrow as pa
import pyarrow.compute as pc

class Doubler:
    def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        return batch.append_column("pred", pc.multiply(batch["reading"], 2))

print(hot.map_batches(Doubler).sort("sensor_id").to_pydict())
# {'sensor_id': [3, 4, 5, 6], 'reading': [15, 16, 23, 42], 'pred': [30, 32, 46, 84]}
```

## Correctness is checked against an outside oracle

Speed is only worth as much as the guarantee that the answer is right, and Batcher checks that guarantee against something other than itself.

Relational behavior is differentially tested against DuckDB. Each query runs on both engines, and the results are compared as a row multiset within float tolerance. When they disagree, that is a decision to surface, never a test to weaken. Inside the engine, the sequential interpreter is the reference and the parallel and JIT paths must match it. Property-based tests reach combinations no enumerated case covers: the full optimizer rule set changes the plan and never the answer, converges to a deterministic fixpoint, and every self-tuning knob leaves the result unchanged.

Benchmarks follow the same rule: nothing is timed until its result agrees with the oracle, so a missing number on a benchmark page means a wrong answer, not a slow one. Behavior DuckDB has no opinion on, such as multimodal decode or model scoring, is covered by ordinary tests.

## Practical limits

Each decision has an edge worth knowing before you plan around it:

- Streaming is micro-batch, so a trigger interval sets the latency floor. Flink's record-at-a-time guarantees are out of scope.
- On a single node at TPC-H sf100 (600 million rows), DuckDB is faster. At sf10 Batcher wins.
- Batcher has no `StringView` representation, so string-heavy execution trails DuckDB and Polars.
- Some wall-clock wins spend more CPU than the competitor, between 1.4x and 4.4x on the shapes measured.
- A distributed shuffle has no external shuffle service. A lost worker's buckets are recomputed unless `shuffle_replication` placed a copy elsewhere. See {doc}`fault-tolerance`.
- Iceberg and Delta tables go through `pyiceberg` and `delta-rs`. See {doc}`/integrations/lakehouse/index`.

The code-checked scorecard behind every claim on this page is `docs/architecture/internals/competitive_architecture.md` in the repository.

## See also

The mechanisms above each have a page of their own:

- {doc}`overview`: the two planes and the crate layout these decisions live in.
- {doc}`execution`: the tiers and the scheduler in detail.
- {doc}`optimization`: Kyber's passes and the cost model behind the learned loop.
- {doc}`../benchmarks/index`: the measured numbers, with the methodology behind each.
- {doc}`/architecture/deep-dives/operators/mergeable-algebra`: the `partial`, `combine`, `finalize` contract in full.
- {doc}`/architecture/deep-dives/adaptive/adaptive-reoptimization`: the re-planning loop, breaker by breaker.
