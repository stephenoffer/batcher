# What makes Batcher different

This page describes the design decisions that separate Batcher from DuckDB, Polars, Spark, and Ray Data, and states what each one does not yet buy.

Most engines are fast at one shape of work. The interesting question is not whether Batcher is fast, which {doc}`../benchmarks/index` answers with measured numbers, but which properties survive when the work changes: when the data outgrows a laptop, when the pipeline has to feed a model, when the same query runs every hour for a year.

Six decisions account for most of that. Each section below says what the decision is, what it buys, and where it stops today. The limits are drawn from `docs/internals/competitive_architecture.md`, an audit that checks every claim against code and names the ones the code does not support.

## One algebra from one core to a cluster

Every stateful operator is written once, in `bc-runtime`, as three functions: `partial(batch)` produces a state, `combine(states)` merges them, and `finalize(state)` emits rows. `combine` is associative and commutative, so partial states merge in any order.

That single implementation is what runs sequentially on one core, in parallel across many, and across a cluster over Arrow Flight. There is no second distributed operator with its own semantics, which is why a result is identical whether it was produced on one node or a hundred, and why CI can assert exactly that.

The practical consequence is that scaling out is a scheduling decision rather than a rewrite. The same script runs on a laptop and on a cluster, and distribution is cheap enough to decline: at TPC-H scale factor 1, where a network shuffle costs more than it saves, the distributed path stays within about 7% of the single node rather than falling off a cliff.

**Where it stops.** A stateful operator with no mergeable form would be capped at one machine, so adding one is a design constraint rather than a free choice. That constraint is the price of the guarantee.

## Speed without a second set of semantics

There is exactly one scalar expression type and one relational plan type, and all three execution paths consume the same one.

![One shared Expr and RelOp feeding three execution tiers. The Tier-0 sequential interpreter is the correctness oracle. The Tier-0 parallel path changes only scheduling and must equal the oracle. The Tier-1 Cranelift JIT must be bit-for-bit identical on its supported subset, and an unsupported expression falls back to the interpreter rather than diverging.](/_static/diagrams/execution_tiers.svg)

The sequential interpreter is the correctness oracle: simple, deterministic, and checked by everything else. The parallel path reuses the same operator code and changes only the scheduling. The Cranelift JIT compiles the supported subset of scalar expressions once per operator and reuses that across every morsel.

The edge that matters most is the dashed one. An expression the JIT does not support is not an error and not a slow compile. It falls back to the interpreter. A fast path that disagreed with the oracle would be worse than no fast path at all, so the JIT is required to be bit-for-bit identical on its subset and to decline everything else.

This is why performance work here does not accumulate risk. A new tier can be added, and a compiled pipeline can be abandoned mid-query, without a second definition of what a query means.

**Where it stops.** The JIT's supported subset is numeric, null-free arithmetic and comparison. Everything else runs interpreted, so the compiled fast path covers less of a real query than the phrase "JIT-compiled engine" suggests.

## A learned loop that outlives the query

This is the differentiator most often overstated, so it is worth stating precisely.

![A capability matrix comparing DuckDB, Spark AQE, and Batcher on three properties: re-planning inside one query, running on a single node, and carrying what was learned into the next run. DuckDB optimizes once and keeps no cross-run state. Spark AQE re-plans at stage boundaries but needs shuffle stages and keeps no cross-run state. Batcher re-plans at the same stage-boundary granularity, runs the same loop on a single node, and carries sketches, calibrated costs, and a bandit into the next run.](/_static/diagrams/adaptive_positioning.svg)

Batcher re-optimizes during a query at pipeline breakers, using cardinalities it has *measured* rather than estimated. When an estimate was wrong by more than `optimizer.reoptimize_error` (2x by default), Kyber re-plans the remainder of the query on the real numbers.

That much is the same mechanism, at the same granularity, as Spark AQE. It is not finer, and claiming otherwise would be wrong. Two things about it are genuinely different:

- **It runs on a single node.** AQE is a cluster mechanism built around shuffle stages. DuckDB has no equivalent at any scale: it optimizes once, before execution, and cannot revise that plan.
- **What it measured survives the query.** Core records actual cardinalities, operator times, and peak memory into the `MetadataHub`, and the next run of that plan shape reads them. That covers sketch-backed cardinality (HyperLogLog for distinct counts, KLL for quantiles), cost coefficients calibrated from measured operator times rather than fixed constants, and a UCB1 bandit over equivalent join strategies. A query gets a better plan the more often it runs, which neither DuckDB nor Spark offers.

**Where it stops.** The within-query loop engages only on a query that contains a join and whose total scan input clears 20 million rows or roughly 1.3 GB, so most small queries never use it. It also is not free: at TPC-H scale factor 10, the one-shot path beats the adaptive one on 20 of 22 queries (3,889 ms against 4,669 ms), because the gate that turns it on reads a provenance label rather than the measured error history. That is a known open item, recorded in `benchmarks/BENCHMARK_RESULTS.md`, not a property of the design.

## The data plane does not touch the object store

On a cluster, Ray schedules tasks and carries control-plane metadata, and that is all it does. Only small `(address, ticket)` strings travel through Ray. Bulk Arrow batches move directly between workers over Arrow Flight, under credit-based flow control where one credit is one in-flight batch slot and a producer blocks when its credits reach zero.

Routing bulk data through an object store is what produces spill storms under memory pressure, and avoiding it is the main reason Batcher's distributed numbers separate from Ray Data's by a wide margin. The credit bound is not merely intended: it is enforced by an in-flight gauge in `bc-transport`.

Within a node the transport picks the cheapest tier automatically, reading straight from the local store in the same process, memory-mapping a 64-byte-aligned Arrow IPC file across processes on the same node, and using Flight only between nodes. The shared-memory tier is worth roughly 23x a loopback Flight hop point to point, and it steps aside on its own when the node is under memory pressure.

**Where it stops.** The shuffle holds published output in RAM with a spill path, rather than persisting it the way Spark does, so this is a weaker fault-tolerance story than Spark's under repeated node loss.

## Batch, streaming, and models are one engine

Batch is the bounded special case of streaming over Arrow batches, not a separate code path. The same operators process record batches either way, and the same pipeline breakers are where a streaming query checkpoints and where the adaptive layer re-plans.

Model work sits on the same engine rather than beside it. Images, audio, and video decode into tensor columns that the relational operators already understand, so one pipeline can filter a table, join it, and feed a model without a hand-off between systems. On the GPU path, stage-overlapped execution runs the CPU decode of the next morsel while the current morsel's forward pass is still in flight.

The measured effect of that overlap is large and specific: a two-stage ResNet-50 pipeline went from 942 to 2,504 images per second, with GPU utilization rising from about 30% to 81%.

**Where it stops.** Streaming is micro-batch. It cannot express the per-record guarantees Flink provides, and that is a consequence of the execution model rather than a gap in the code.

## Correctness is mechanically proven, not asserted

Every claim above is only worth as much as the guarantee that the engine returns the right answer, so that guarantee is checked against something external rather than against itself.

Relational behavior is differentially tested against DuckDB: the harness runs a query on both engines, compares results as a sorted row multiset within float tolerance, and a disagreement is a decision to surface rather than a test to weaken. Inside the Rust engine, the sequential interpreter is the reference and the parallel and JIT paths must match it. Property-based tests then cover the combinations an enumerated case cannot reach, asserting that the full optimizer rule set changes the plan and never the answer, that it converges to a deterministic fixpoint, and that every self-tuning knob is result-invariant.

The benchmark harness applies the same rule to itself. It refuses to time a query whose result does not match the oracle, so a missing number on a benchmark page means a wrong answer rather than a slow one. That gate is what caught two other engines returning the wrong answer on TPC-H q6.

**Where it stops.** The oracle is DuckDB, so behavior DuckDB does not define, such as multimodal decode or model scoring, is covered by ordinary tests rather than by a second implementation.

## Requirements and limitations

Batcher does not win everywhere, and the places it loses are tracked rather than omitted.

Against DuckDB's own compressed store, rather than shared Arrow, DuckDB remains ahead on join-heavy SQL: 2.08x on the TPC-H suite at scale factor 10, where Batcher wins 4 of 22 queries. Part of that is storage, since DuckDB decompresses its own format as it scans and never pays an Arrow ingest, and part of it is engine work still open.

Single-node performance above roughly 100 million rows trails DuckDB. String execution has no `StringView` representation, and dictionary encoding is decoded at the leaf rather than carried into the kernels, though dictionary-aware comparison against a literal has since closed part of that gap.

Streaming cannot express Flink's guarantees, and lakehouse format support is reached through `pyiceberg` and `delta-rs` rather than a native implementation.

## See also

- {doc}`overview`: the two planes and the crate layout these decisions live in.
- {doc}`execution`: the tiers and the scheduler in detail.
- {doc}`optimization`: Kyber's passes and the cost model behind the learned loop.
- {doc}`../benchmarks/index`: the measured numbers, with the methodology behind each.
- {doc}`/deep-dives/operators/mergeable-algebra`: the `partial`, `combine`, `finalize` contract in full.
- {doc}`/deep-dives/adaptive/adaptive-reoptimization`: the re-planning loop, breaker by breaker.
