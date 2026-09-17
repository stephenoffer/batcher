# Glossary

This page defines the terms you meet across the Batcher documentation, each with a link to the page that covers it in depth.

Accessor namespace
: A group of expression methods for one column type, such as `.str` for strings, `.dt` for dates and times, and `.list` for arrays. Nested data gets `.struct`, `.json`, and `.map`, and media columns get `.image`, `.audio`, and `.video`. See {doc}`/getting-started/concepts/expressions`.

Adaptive re-optimization
: Planning from measured numbers instead of estimates. Across runs, Batcher records what each query did and plans the next run from it. Inside a large joined query, it runs up to a pipeline breaker, measures the real row count, and re-plans the rest. See {doc}`/getting-started/concepts/adaptive`.

Arrow Flight shuffle
: The all-to-all redistribution that puts every row with the same key on the same worker before a distributed group-by or join. Batches move worker to worker over Arrow Flight with credit-based flow control, and Ray carries only addresses and tickets, never the data. See {doc}`/architecture/deep-dives/distribution/shuffle-flight`.

Batch UDF
: Your own Python function, run by {py:meth}`map_batches <batcher.Dataset.map_batches>` on a whole batch at a time, so the call overhead is paid once per batch rather than once per row. The optimizer can't see inside it, so reach for an expression first. See {doc}`/user-guide/transform/columns/udfs`.

Broadcast join
: A join strategy that replicates the build side to every worker, so the probe side never moves. Kyber picks it when the build side fits `optimizer.broadcast_max_bytes`, and otherwise falls back to a hash or sort-merge join. See {doc}`/architecture/deep-dives/operators/join-algorithms`.

Cache
: A stored result you reuse across terminal operations. A dataset doesn't keep its result, so a second terminal call runs the plan again unless you mark it with {py:meth}`cache() <batcher.Dataset.cache>`. See {doc}`/user-guide/operate/tuning/caching`.

Carbonite
: The resource manager in the Python control plane. It checks whether a plan fits, hands out memory reservations and shuffle credits, and decides when to spill, without rewriting a plan or computing a result. See {doc}`/architecture/overview`.

Checkpoint
: A directory, passed as `checkpoint=` to a streaming write, that records source offsets and sink commits for each micro-batch. A restarted query resumes from the last committed offset. See {doc}`/user-guide/moving-data/streaming`.

Control plane
: The Python half of Batcher. It builds a query plan, optimizes it, and decides how much memory the work may use, without touching a row. See {doc}`/getting-started/concepts/index`.

Core
: The executor in the Python control plane. It ships the plan to the Rust engine, runs the adaptive re-optimization loop, and records what actually happened, meaning real row counts, operator times, and peak memory. See {doc}`/architecture/overview`.

Credit
: The unit of credit-based flow control. One credit is one in-flight batch slot, and a producer blocks when its credits reach zero, so a channel's in-flight memory is bounded by its credits times the batch size. See {doc}`/architecture/deep-dives/distribution/credit-flow-control`.

Data plane
: The Rust half of Batcher, where every per-row and per-batch computation runs over Apache Arrow. It meets the control plane at one boundary, a JSON plan plus zero-copy Arrow batches. See {doc}`/architecture/execution`.

Dataset
: A handle to a logical plan plus the inputs bound to it. It holds no data, and every operation returns a new {py:class}`Dataset <batcher.Dataset>` rather than changing the one you have. See {doc}`/getting-started/concepts/lazy`.

Device tier
: The GPU relational backend you request with `collect(backend="gpu")`, which replays a plan on cuDF. A plan reaches the device only when every node translates. Anything else is declined and runs on the CPU engine, so the request is always safe. See {doc}`/architecture/deep-dives/distribution/gpu-execution`.

Exactly-once
: The streaming guarantee that a restart neither loses nor duplicates a row. It comes from a checkpoint, a replayable source that seeks forward, and an idempotent sink that recognizes a replayed batch. See {doc}`/user-guide/moving-data/streaming`.

Expression
: A description of column work, built from {py:obj}`bt.col(...) <batcher.col>`, {py:obj}`bt.lit(...) <batcher.lit>`, operators, and methods. Python ships the expression tree in the plan, and Rust evaluates it over whole Arrow batches. See {doc}`/getting-started/concepts/expressions`.

Governance
: Row filters and column masks that Batcher enforces inside the engine. A {py:class}`SecurityCatalog <batcher.SecurityCatalog>` declares the policy, a {py:class}`Principal <batcher.Principal>` is the identity a query runs as, and the policy applies when a table is read. See {doc}`/user-guide/trust/governance`.

JIT tier
: Tier-1, the just-in-time compiler in `bc-codegen` that turns a scalar expression into native code with Cranelift. On the subset it accepts it must be bit-for-bit identical to the interpreter, and on anything else it falls back to the interpreter. See {doc}`/architecture/deep-dives/query/jit-compilation`.

JSON IR
: The JSON document a plan becomes when Python hands it to Rust. It's a wire contract: Python and Rust agree on a set of operator tags, and Rust rejects a tag it doesn't know. See {doc}`/architecture/deep-dives/query/plan-ir`.

Kyber
: The optimizer in the Python control plane. It rewrites a logical plan through an ordered set of phases, with rewrites such as predicate pushdown and join reordering, then lowers it to a physical plan. It decides and never executes. See {doc}`/architecture/optimization`.

Lakehouse table
: A table in a transactional table format, Delta Lake, Apache Iceberg, or Apache Hudi. Batcher reads and writes all three, and a Delta write is a single atomic commit, so a reader never sees a partial table. See {doc}`/user-guide/moving-data/lakehouse`.

Lazy plan
: The plan a dataset accumulates as you chain operations. Nothing runs until a terminal operation, so the optimizer sees your whole query before it reads any data. See {doc}`/getting-started/concepts/lazy`.

Memory envelope
: The memory budget a query runs within. Batcher senses it from host RAM, honoring a cgroup limit, unless you set `memory.max_memory_bytes`, and Carbonite throttles new allocations at `memory.soft_limit`, 85% of the envelope by default. See {doc}`/architecture/deep-dives/memory/buffer-pool`.

Mergeable operator
: A stateful operator written once as three steps: `partial` builds partition-local state, `combine` merges states in any order, and `finalize` produces rows. The same code runs on one core, every core, or a cluster. See {doc}`/getting-started/concepts/scaling`.

MetadataHub
: Where Batcher keeps what it measured on each run: row counts, operator times, column sketches, fitted cost coefficients, and bandit rewards. Core writes to it after a run, and Kyber reads it before planning the next one. See {doc}`/architecture/deep-dives/adaptive/learned-metadata`.

Micro-batch
: One increment of a streaming query. A trigger fires it, the engine reads and computes the new input, the sink receives the rows, and the checkpoint records the commit. See {doc}`/user-guide/moving-data/streaming`.

Morsel
: The unit of work Batcher schedules across cores, an Arrow `RecordBatch` of 16,384 rows or 1 MiB, whichever it reaches first. `execution.morsel_rows` and `execution.morsel_bytes` set the two bounds. See {doc}`/architecture/deep-dives/operators/morsel-parallelism`.

Output mode
: What each micro-batch of a streaming write emits, set with `output_mode=`. `"append"` emits only rows that won't change again, `"complete"` emits the full result table, and `"update"` emits only the rows whose value changed. See {doc}`/user-guide/moving-data/streaming`.

Partition pruning
: Skipping whole directories of a Hive-partitioned table at plan time. A directory that a predicate on the partition column rules out is never listed, opened, or turned into a task. See {doc}`/user-guide/operate/tuning/large-tables`.

Pipeline
: A maximal chain of operators that streams each batch straight through without materializing it, such as scan, filter, project, and probe. See {doc}`/architecture/execution`.

Pipeline breaker
: An operator that must collect its input before it produces output, such as a hash-join build, an aggregate, a sort, a distinct, or a window. Breakers are where data materializes, spills, shuffles, and gets re-optimized. See {doc}`/architecture/execution`.

Pushdown
: Handing part of a query to the data source, so rows and columns you don't need are never read or decoded. Kyber offers a filter sitting directly above a scan to the source, and a predicate that doesn't push still returns the right answer. See {doc}`/user-guide/operate/tuning/pushdown`.

Quarantine
: A data-quality action that splits a dataset into the rows that pass its checks and the rows that violate them, so you can write the rejected rows to a dead-letter sink. The alternatives are `fail()` and `drop()`. See {doc}`/user-guide/trust/data-quality`.

Save mode
: The `mode=` of a file write, which decides what happens when output already exists. `ds.write` defaults to `"overwrite"`. `"error"` refuses to replace earlier output, `"ignore"` skips the write, and `"overwrite_partitions"` replaces only the partitions the new rows touch. See {doc}`/user-guide/moving-data/writing-data`.

Session
: A reusable catalog of tables and Python functions for SQL, built with {py:obj}`bt.Session <batcher.Session>`. It's the analogue of a DuckDB connection or a SparkSession, and {py:obj}`bt.sql <batcher.sql>` uses a shared default session. See {doc}`/user-guide/analyze/sql`.

Sketch
: A compact, mergeable summary of a column, such as a distinct-value count or quantiles, that Core records after a run. Kyber reads sketches to replace estimates with measurements. See {doc}`/architecture/deep-dives/adaptive/learned-metadata`.

Spill
: Writing part of a stateful operator's state to disk when it no longer fits in the memory envelope, then reading it back in bounded pieces. A query that spills is slower, and its result is identical to the in-memory run. See {doc}`/architecture/deep-dives/memory/spilling`.

Terminal operation
: A call that runs the plan and returns or writes a result, such as {py:meth}`collect() <batcher.Dataset.collect>`, {py:meth}`to_pydict() <batcher.Dataset.to_pydict>`, {py:meth}`iter_batches() <batcher.Dataset.iter_batches>`, or a write. Every other operation only grows the plan. See {doc}`/getting-started/concepts/lazy`.

Tier-0 interpreter
: The sequential reference executor, kept deterministic and obviously correct. The parallel path and the JIT are tested against it, and the parallel path changes only scheduling. See {doc}`/architecture/execution`.

Time travel
: Reading a lakehouse table as it was at an earlier version. It works because a Delta commit retires a file from the log without deleting it from storage, until a vacuum reclaims it. See {doc}`/user-guide/moving-data/lakehouse`.

Trigger
: The cadence of a streaming write, set with {py:class}`Trigger <batcher.Trigger>`. `processing_time` fires a micro-batch on a wall-clock interval, and `available_now` drains every record available at start, then stops. See {doc}`/user-guide/moving-data/streaming`.

Watermark
: A bound on event-time lateness, declared with {py:meth}`with_watermark <batcher.Dataset.with_watermark>`. Once the watermark, the latest event time minus the allowed lateness, passes a window's end, the engine emits and evicts that window and drops rows that arrive later. See {doc}`/user-guide/moving-data/streaming`.
