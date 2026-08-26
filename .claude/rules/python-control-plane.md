# Rule: The Python Control Plane

`python/batcher/` is the control plane and the public face of the engine. It builds
plans, optimizes them, and orchestrates execution — but it **never touches a tuple
in the hot path** (see `.claude/rules/architecture.md`). This rule covers the API
surface and the IR contract; code-quality mechanics (ruff, dedup, dead code, public
API hygiene) live in `.claude/rules/python-quality.md`.

## The user-facing API: lazy, immutable, expression-first

The surface (`api/dataset/`, `plan/expr_ir/`, `python/batcher/__init__.py`) is
deliberately small. Keep it that way.

- **Lazy + immutable.** A `Dataset` is a handle to a `LogicalPlan` plus bound
  inputs. Every operation returns a *new* `Dataset`; nothing mutates. No work
  happens until a terminal op (`collect`, `to_pydict`, `iter_batches`,
  `write.parquet`, ...). At that point `api` orchestrates Kyber → Carbonite → Core.
- **Expressions everywhere, no lambdas in the hot path.** Column work is expressed
  via `Expr` (`col("x") * col("y")`, `.sum()`, `.cast(...)`, `.str.contains(...)`),
  which lowers to `bc_expr::Expr` and runs in Rust. User Python callbacks
  (`map_batches`) operate on whole Arrow batches, never per row.
- **One obvious way to do each thing.** `select` chooses/derives the full output;
  `with_columns` adds/replaces. Don't add a second spelling of an existing
  capability. New surface area must be justified against DuckDB/Polars ergonomics
  and earn its place.
- Mirror SQL / DataFrame conventions users already know (the `_sql/` module and the
  DuckDB differential oracle are your reference for expected semantics).

## The JSON IR is a stable wire contract

Python and Rust meet at one JSON document. This is a hard coupling — treat it like
a serialized protocol.

- Each `LogicalPlan` node implements `to_ir()` returning `{"op": "<tag>", ...}`
  (e.g. `{"op": "hash_join", ...}`, `{"op": "aggregate", ...}`). These tags and the
  `snake_case` operator/function names MUST exactly match `bc_ir::RelOp` /
  `bc_expr::Expr` `serde` tags. Same for expression lowering in `expr_ir.py`.
- Changing the IR is a **two-sided change in one commit**: update Python `to_ir()`
  *and* the Rust `serde` definitions together, and add a differential test that
  exercises the new shape end to end. A drift here is a silent correctness bug.
- `plan/physical.py::PhysicalPlan.to_json()` serializes the lowered IR the engine
  runs. Keep lowering in `plan` (neutral) — not in `kyber`/`carbonite`/`core`.

## The FFI boundary

- Cross it through `core` calling `batcher._native` (`bc_py::execute_plan` and the
  distributed primitives). `api` and the subsystems above don't call `_native`
  directly except where the layering already does.
- Data crosses **zero-copy** as pyarrow `RecordBatch`es (Arrow C Data Interface).
  Don't convert to Python lists/dicts to move data; that is a hot-path tuple touch.
- Narrow numeric types are normalized once at the boundary (Int8/16/32 → Int64,
  Float16/32 → Float64). Rely on that; don't re-implement type coercion upstream.

## Distribution: Ray is scheduling only

- Ray (optional `[ray]` extra) is used for task/actor scheduling and control-plane
  metadata **only**. The data plane bypasses the Ray object store — bulk Arrow
  batches move via `bc-transport` (Arrow Flight), never as Ray objects.
- The distributed executor (`dist/`) composes the *same* mergeable primitives
  (`partial_aggregate` / `partition_batches` / `combine_finalize`) the single-node
  path uses. Distributed is a scheduling concern, not a second semantics.
- **A result MUST be identical whether produced on one node or many — with two stated
  exceptions.** The multiset of rows, every column name, and every column *type* are exact.
  Both exceptions are places where the *query itself* does not determine an answer, so neither
  is a licence to differ anywhere else.

  **Floating-point reductions are identical up to reassociation.** `combine` is associative in
  exact arithmetic, IEEE addition is not, and partition count changes the summation order.
  `bc-runtime`'s Neumaier compensation and Chan's parallel Welford formula bound that error to
  near the last bits; they do not remove it, and nothing can while the partition count is free.
  Write the claim this way. Stating it as bit-identity was worse than useless: it is not what
  the code does, `assert_same` tolerates float rounding so nothing ever checked it, and an
  invariant nobody can fail is one nobody reads.

  **A window function that must break a tie its `ORDER BY` leaves open may break it
  differently.** `row_number()` assigns 1..n within a partition; when several rows share the
  ordering key, which of them gets which number is undefined in SQL, and the two paths resolve
  it by different physical orders — single-node by arrival within the partition, distributed by
  arrival through the shuffle. Verified on a partitioned `row_number` where every row was in a
  tied group: the partitions, their sizes, and the rank set 1..n per partition were identical
  on both sides, and only the assignment of a rank to a *tied* row differed. `rank` and
  `dense_rank` are unaffected — they give peers the same value by definition.

  This one is worth reading as a *bound* rather than a licence, because the engine already
  refuses the same freedom elsewhere: `bc_interp::ops::reshape::sample_n_batches` breaks its
  hash ties by row *content* precisely so a fixed-count `sample` does not depend on how the
  input was split. Making `row_number` deterministic the same way is a decision available to
  anyone who wants it — a content tiebreak, paid per tie — not a limit of the architecture. A
  divergence that is **not** of these two kinds is a defect, however plausible its rows look.

## Streaming and batch

Batch is the bounded special case of streaming over Arrow batches. Operators
process `RecordBatch`es through the pipeline; pipeline breakers (sort, aggregate,
join build) materialize and are exactly where the adaptive layer re-optimizes.
Keep batch and micro-batch paths on the same operator semantics.

## Gate before "done"

The canonical gate matrix is in `CLAUDE.md` — run the rows your change touches. Control-plane
delta: if you changed IR tags you also changed Rust, so the Rust rows apply too.
