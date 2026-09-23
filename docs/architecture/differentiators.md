# What makes Batcher different

This page describes the design decisions that separate Batcher from DuckDB, Polars, Spark, and Ray Data, and what each one buys.

Most engines are fast at one shape of work. Whether Batcher is fast on yours is a question {doc}`../benchmarks/index` answers with measured numbers. This page answers a different one: which properties survive when the work changes. The data outgrows a laptop. The pipeline has to feed a model. The same query runs every hour for a year.

Six decisions account for most of that, and each section below says what the decision is, what it buys, and where it stops. Every claim is checked against the code that implements it.

## One algebra from one core to a cluster

Every stateful operator is written once, in `bc-runtime`, as three functions. `partial(batch)` produces a state, `combine(states)` merges states, and `finalize(state)` emits rows. `combine` is associative and commutative, so partial states merge in any order.

That one implementation runs sequentially on a core, in parallel across many, and across a cluster over Arrow Flight. There is no second distributed operator with semantics of its own. The rows, the column names and the column types come back the same on one node and on a hundred, and a floating-point reduction agrees up to the reassociation a different partition count causes. [`tests/integration/test_distributed.py`](https://github.com/stephenoffer/batcher/blob/main/tests/integration/test_distributed.py) asserts that operator by operator against a local Ray instance. It skips when Ray isn't installed, which is how CI runs, so the evidence comes from the runs recorded on real clusters.

Scaling out is therefore a scheduling decision, not a rewrite. The script you wrote on a laptop is the script that runs on the cluster. Distribution is also cheap to decline: on the `udf-map` workload at TPC-H sf1, too small for a shuffle to pay, the distributed path took 92 ms against 86 ms on one node.

The price is a constraint on every new operator. A stateful operator with no mergeable form is capped at one machine, and it fails at cluster scale with wrong results rather than an error. Holding every operator to the constraint is what buys the guarantee.

## Speed without a second set of semantics

Batcher has exactly one scalar expression type and one relational plan type, and every execution path consumes the same ones.

![One shared Expr and RelOp feeding three execution tiers. The Tier-0 sequential interpreter is the correctness oracle. The Tier-0 parallel path changes only scheduling and must equal the oracle. The Tier-1 Cranelift JIT must be bit-for-bit identical on its supported subset, and an unsupported expression falls back to the interpreter rather than diverging.](/_static/diagrams/execution_tiers.svg)

The sequential interpreter is the correctness oracle: simple, deterministic, and the thing everything else is checked against. The parallel path reuses the same operator code and changes only the scheduling. The Cranelift JIT compiles a scalar expression once, caches the artifact process-wide, and reuses it across every morsel.

The edge that matters most is the dashed one. An expression the JIT doesn't support is not an error and not a slow compile. It runs on the interpreter. A fast path that disagreed with the oracle would be worse than none, so the JIT must match the interpreter bit for bit on its subset and decline everything else.

That is why performance work here doesn't accumulate risk. The compiled subset covers fixed-width numeric expressions: arithmetic, comparisons including dates and timestamps, boolean logic with SQL's three-valued null semantics, and `CASE`. Strings, lists and most functions run on the interpreter, with the same result.

## A learned loop that outlives the query

This is the differentiator most often overstated, so here it is at full precision.

![A capability matrix comparing DuckDB, Spark AQE, and Batcher on three properties: re-planning inside one query, running on a single node, and carrying what was learned into the next run. DuckDB optimizes once and keeps no cross-run state. Spark AQE re-plans at stage boundaries but needs shuffle stages and keeps no cross-run state. Batcher re-plans at the same stage-boundary granularity, runs the same loop on a single node, and carries sketches, calibrated costs, and a bandit into the next run.](/_static/diagrams/adaptive_positioning.svg)

Batcher re-optimizes during a query at pipeline breakers, using cardinalities it has *measured* rather than estimated. When an estimate was wrong by more than `optimizer.reoptimize_error` (2.0 by default), Kyber re-plans the rest of the query on the real numbers. That is stage-boundary re-optimization, the same granularity Spark AQE works at.

Two things about it are different. First, it runs on a single node. AQE is a cluster mechanism built around shuffle stages, and DuckDB optimizes once, before execution, with no way to revise the plan.

Second, what it measured survives the query. Core records actual cardinalities, operator times, and peak memory into the `MetadataHub`, and the next run of that plan shape reads them. That covers sketch-backed cardinality (HyperLogLog for distinct counts, KLL for quantiles), cost coefficients calibrated from measured operator times rather than fixed constants, and a UCB1 bandit over equivalent join strategies. A query gets a better plan the more often it runs, and neither DuckDB nor Spark keeps anything comparable between runs.

The within-query loop engages only where it can pay for itself. On a single node, `adaptive="auto"` requires a join and a size floor charged per pipeline breaker the loop would cut at, because each cut costs a materialization, a re-plan, and the operator fusion given up at that boundary. The floor is 5 million rows, or roughly 320 MB, per breaker. The simplest joined shape has two breakers and qualifies at about 10 million rows; a snowflake with six needs 30 million. Below that, the one-shot plan is the faster answer and is what runs. On a cluster, a plan the one-shot dispatcher can't run correctly, such as an aggregate over a `limit`, takes the staged path at any size.

## The data plane does not touch the object store

On a cluster, Ray schedules tasks and carries control-plane metadata, and that's all it does. Only small `(address, ticket)` strings travel through Ray. Bulk Arrow batches move directly between workers over Arrow Flight, under credit-based flow control: one credit is one in-flight batch slot, and a producer blocks when its credits reach zero. The bound is enforced rather than intended, by an in-flight gauge in `bc-transport`.

An object store in the data path produces spill storms under memory pressure. Staying out of it is the main reason Batcher's distributed shuffles run far ahead of Ray Data's.

Within a node, the transport picks the cheapest tier on its own. In the same process it reads straight from the local store. Across processes on the same node it memory-maps a 64-byte-aligned Arrow IPC file, which is roughly 23x faster than a loopback Flight hop point to point and steps aside when the node is under memory pressure. Flight carries only what crosses nodes. Published shuffle output sits in RAM with a spill path behind it, so a reducer reads from memory in the common case and from disk under pressure.

## Batch, streaming, and models are one engine

Batch is the bounded special case of streaming over Arrow batches, not a separate code path. The same operators process record batches either way, and the same pipeline breakers are where a streaming query checkpoints and where the adaptive layer re-plans.

Model work runs on the same engine rather than beside it. Images, audio, and video decode into tensor columns the relational operators already understand, so one pipeline can filter a table, join it, and feed a model with no hand-off between systems. On the GPU path, stage-overlapped execution runs the CPU decode of the next morsel while the current morsel's forward pass is still in flight.

The measured effect is large. A two-stage ResNet-50 pipeline on 8 T4 GPUs went from 942 to 2,504 images per second, with GPU utilization rising from about 30% to 81% ([`benchmarks/BENCHMARK_RESULTS.md`](https://github.com/stephenoffer/batcher/blob/main/benchmarks/BENCHMARK_RESULTS.md)).

## Correctness is checked against an outside oracle

Every claim above is only worth as much as the guarantee that the answer is right, so Batcher checks that guarantee against something other than itself.

Relational behavior is differentially tested against DuckDB. The harness runs a query on both engines and compares the results as a row multiset within float tolerance, and a disagreement is a decision to surface, never a test to weaken. Inside the Rust engine, the sequential interpreter is the reference and the parallel and JIT paths must match it. Property-based tests reach the combinations an enumerated case can't: the full optimizer rule set changes the plan and never the answer, it converges to a deterministic fixpoint, and every self-tuning knob leaves the result unchanged.

The benchmark harness holds itself to the same rule. It refuses to time a query whose result doesn't match the oracle, so a missing number on a benchmark page means a wrong answer rather than a slow one. That gate caught Daft returning the wrong revenue on TPC-H q6, at both sf1 and sf10.

The oracle reaches only as far as DuckDB defines. Behavior DuckDB has no opinion on, such as multimodal decode or model scoring, is covered by ordinary tests instead.

## Requirements and limitations

Each decision above has an edge, and these are the ones to know before you plan around them:

- Streaming is micro-batch. A trigger interval sets the latency floor, and Batcher can't express the guarantees a record-at-a-time engine such as Flink offers.
- On a single node at 100 million rows and up, DuckDB is faster on some shapes. At TPC-H sf100 (600 million rows) Batcher loses and runs out of memory on q3, q4 and q5. At sf10 it wins.
- Batcher has no `StringView` representation, so string-heavy execution trails DuckDB and Polars.
- Some wall-clock wins cost more CPU than the competitor spends, between 1.4x and 4.4x on the shapes measured.
- A distributed shuffle has no external shuffle service. A lost worker's buckets are recomputed unless `shuffle_replication` placed a copy elsewhere. See {doc}`fault-tolerance`.
- Iceberg and Delta tables are read and written through `pyiceberg` and `delta-rs`. See {doc}`/integrations/lakehouse/index`.

The code-checked scorecard behind every claim on this page is `docs/architecture/internals/competitive_architecture.md` in the repository.

## See also

- {doc}`overview`: the two planes and the crate layout these decisions live in.
- {doc}`execution`: the tiers and the scheduler in detail.
- {doc}`optimization`: Kyber's passes and the cost model behind the learned loop.
- {doc}`../benchmarks/index`: the measured numbers, with the methodology behind each.
- {doc}`/architecture/deep-dives/operators/mergeable-algebra`: the `partial`, `combine`, `finalize` contract in full.
- {doc}`/architecture/deep-dives/adaptive/adaptive-reoptimization`: the re-planning loop, breaker by breaker.
