# JIT compilation

*Tier-1* is Batcher's just-in-time compiler for scalar expressions. It lives in `bc-codegen` and turns the same `bc_expr::Expr` tree the interpreter evaluates into native machine code with Cranelift. This page describes what it compiles, how it handles nulls, and when it hands work back to the interpreter.

The interpreter materializes a full Arrow array for every node of an expression tree, so `(a - b) * c` over a morsel costs two temporary arrays and three trips through memory. Tier-1 compiles the tree into a single native loop instead: each output element is computed in registers and written straight to the result buffer. The win grows with the depth of the expression.

You can see which tier ran in {py:meth}`Dataset.stats() <batcher.Dataset.stats>`:

```python
import batcher as bt

n = 200_000
ds = bt.from_pydict({"g": [i % 10 for i in range(n)], "a": list(range(n)), "s": [str(i) for i in range(n)]})
numeric = ds.group_by("g").agg(t=bt.sum(bt.col("a") * 2 + 1))
strings = ds.group_by("g").agg(t=bt.sum(bt.col("s").str.len_chars()))
print({op.kind: op.backend for op in numeric.stats().ops})  # {'aggregate': 'jit', 'scan': 'interp'}
print({op.kind: op.backend for op in strings.stats().ops})  # {'aggregate': 'interp', 'scan': 'interp'}
```

The numeric aggregate input compiles. The string function is outside the subset, so the interpreter evaluates it, with the same result either way.

:::{important}
On the subset it accepts, the JIT must be **bit-for-bit identical** to `bc_expr::Expr::eval`. On everything else it must **fall back silently**. The interpreter is the oracle everything is tested against, so there is no "close enough".
:::

## What compiles

[`crates/bc-codegen/src/analyze.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-codegen/src/analyze.rs) is the authority:

| Variant | Compiles | Notes |
|---|---|---|
| `Col` | `Int64`, `Float64` | `Date32` and tz-naive `Timestamp(µs)` as **comparison operands only**: they are an `i32` day count and an `i64` microsecond instant, and Arrow compares them by integer value |
| `Lit` | `Int`, `Float` | plus date/timestamp literals in a comparison. Bool and string literals do not compile |
| `Binary` | `Add`/`Sub`/`Mul`/`Div`/`Mod`, the six comparisons, `And`/`Or` over boolean sub-results | integer `Div`/`Mod` only against a constant divisor (below) |
| `Not` | of a boolean sub-result | |
| `Case` | over the numeric subset | lowered to a `select` chain in the interpreter's reverse-fold order, so the first matching `WHEN` wins |
| `Cast` | `i64 → f64`, or a no-op | rounds exactly as the interpreter's Arrow cast does, so the two tiers stay bit-identical |
| `Math`, `Math2` | value-only math over the numeric subset | libm calls, so the SIMD body excludes them |
| `IsNan`, `IsInf` | over a `Float64` operand | any other operand type declines |
| everything else | no | strings, date functions, lists, structs, `IsNull`, `Coalesce`, media decode → `CodegenError::Unsupported`, and the caller uses `Expr::eval` |

An integer divisor compiles only when it is a **nonzero, non-`-1` constant**, because Cranelift's `sdiv`/`srem` trap on divide-by-zero and on `i64::MIN / -1`. A variable divisor stays on the interpreter.

:::{dropdown} The generated function's ABI
```text
fn(n: i64, cols: *const *const u8, out: *mut u8)
```

`cols` is an array of pointers to each referenced column's raw values buffer, in stable first-seen order, so there's no ceiling on how many columns an expression references. `out` holds `n` `i64`s, `n` `f64`s, or a packed LSB-first Arrow bitmask for a boolean result, so the `BooleanArray` wraps it with no repack. The Kleene body uses a wider signature that also carries per-column validity in and a validity buffer out. Promotion mirrors Arrow: any `f64` operand makes the subtree `f64`.
:::

## Nulls: three paths

Arrow columns carry a validity bitmap, and the JIT loop reads raw values that are garbage at null slots. `CompiledExpr::eval` decides per batch:

1. **No nulls in any referenced column.** Run the loop. This is the fast path.
1. **Nulls, and the expression is null-propagating.** Compute over the raw buffers, then AND the inputs' validity bitmaps into the result. `kleene::is_null_propagating` admits `Col`, `Lit`, `Add`/`Sub`/`Mul`, comparisons, value-only math, numeric casts, `Not`, and the constant-divisor `Div`/`Mod`.
1. **Nulls, and the expression is a compound predicate.** `false AND null` is `false`, so a combined mask would be wrong. `needs_kleene` selects a second body compiled in a value-plus-validity ABI: real three-valued logic, still on the JIT.

Anything else with nulls, such as `Case` or `Coalesce`, falls back to the interpreter for that batch only.

```text
  once, per operator                 per batch, in CompiledExpr::eval
  ──────────────────                 ────────────────────────────────

  try_compile(expr, first morsel)
        │
        ├─ outside the subset ──────► Tier-0 for every batch, forever
        │
        └─ Ok(Arc<CompiledExpr>) ───► shared across rayon workers
                                            │
                                            ├─ no nulls in any referenced column
                                            │     └─► scalar or SIMD loop         ← fast path
                                            │
                                            ├─ nulls, null-propagating expression
                                            │     └─► loop over raw buffers,
                                            │         then AND the validity bitmaps
                                            │
                                            ├─ nulls, compound predicate (And/Or)
                                            │     └─► Kleene body: value+validity ABI
                                            │
                                            └─ nulls, anything else (Case, Coalesce)
                                                  └─► Err ──► Tier-0, this batch only
```

![Where the JIT gives up, and how much it gives up each time. The compiled path runs down the left: bc-codegen's analyze() decides which types and operators can compile, a successful compile yields one Arc<CompiledExpr> that is Send plus Sync and shared by every worker, compiled exactly once, and bc-interp drives eval(batch) over 16,384 rows at a time, reusing that artifact for every morsel and never recompiling. The compile cache on the right, keyed on the expression and the schema and capped at 1024 entries, is asked before anything compiles, and a refusal is remembered too. The two refusal edges differ in blast radius. Failing analyze means the expression is outside the subset, so that operator never compiles at all and runs on Expr::eval, the interpreter and the oracle, for the life of the query. Failing at eval, on nulls the compiled body cannot carry, costs this batch only and the next one tries again. The subset is narrow on purpose: numeric, date and timestamp columns, arithmetic and comparison, and no strings.](/_static/diagrams/jit_fallback.svg)

Nulls take a path through the JIT rather than out of it, so a nullable numeric column still compiles:

```python
import batcher as bt

ds = bt.from_pydict({"a": [1.0, None, 3.0], "b": [10.0, 20.0, None]})
print(ds.select(y=(bt.col("a") - bt.col("b")) * 2).to_pydict())  # {'y': [-18.0, None, None]}
```

## SIMD

[`crates/bc-codegen/src/simd.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-codegen/src/simd.rs) emits a vector body when every node is vectorizable: numeric leaves, integer `+`/`-`/`*`, float `+`/`-`/`*`/`/`, the comparisons, `Not`, and numeric casts. Width comes from `bc_arrow::HardwareProfile`: 2 f64 lanes on SSE2 and NEON, 4 on AVX2, and 8 on AVX-512. Automatic detection caps at 4, because 512-bit code can down-clock the core, so the 8-lane width is opt-in. A scalar remainder loop handles the tail.

Integer `Div`/`Mod`, float `Mod`, `Math`, and `Case` stay scalar so every lane is bit-identical to the interpreter. Boolean `And`/`Or` vectorize as a mask combine on a null-free batch, with the Kleene body as the fallback.

## Compile once

`compile_expr` is a pure function of `(expr, the types of the columns it references, the SIMD override)`, and the sample batch is consulted only for types. So the artifact is reused across every morsel, operator instance, and `execute_plan` call that shares the triple. The process-wide memo in [`crates/bc-codegen/src/cache.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-codegen/src/cache.rs) makes the compile an admission price paid once, which matters most on the streaming and per-UDF paths that call `execute_plan` in a loop.

:::{important}
The cache keys on the *full structural rendering* of that triple and compares for equality, never merely by hash, so a collision can't hand back code compiled for a different expression. It caps at 1024 entries and also remembers which expressions are unsupported.
:::

## Who calls it

Only the parallel paths, `bc-interp::par` and the streaming executor in `bc-interp::stream`. The sequential oracle passes `&None`:

```rust
// crates/bc-interp/src/ops/mod.rs
pub(crate) fn filter_batch(batch: &RecordBatch, predicate: &Expr) -> Result<RecordBatch, InterpError> {
    filter_batch_jit(batch, predicate, &None, None)   // the oracle never JITs
}

fn eval_jit(jit: &Jit, expr: &Expr, batch: &RecordBatch) -> Result<ArrayRef, InterpError> {
    if let Some(compiled) = jit {
        if let Ok(arr) = compiled.eval(batch) {
            return Ok(arr);           // Tier-1
        }
    }
    Ok(expr.eval(batch)?)             // Tier-0: per-batch fallback
}
```

The parallel path compiles once per operator from the first morsel and shares the `Arc<CompiledExpr>` across rayon workers.

There is no user-facing switch, because the result is identical either way. The `backend` tag reads `interp`, `jit`, or `interp+jit` when some of an operator's expressions compiled and others fell back, and the process-wide `backends` counter in {doc}`/user-guide/operate/running/metrics` tallies the same tags across queries. The default streaming executor keeps `Filter` and `Project` on the interpreter, where Arrow's comparison kernels are already SIMD, and compiles aggregate group keys and inputs. The materializing parallel executor (`par.rs`) compiles filters and projections too.

Growing the subset follows a fixed rule: teach the interpreter first, then either teach the JIT *and* prove parity, or leave the JIT to fall back.

:::{dropdown} Where the code lives
- [`crates/bc-codegen/src/lib.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-codegen/src/lib.rs): `CompiledExpr`, the ABI, dispatch between scalar and SIMD
- [`crates/bc-codegen/src/analyze.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-codegen/src/analyze.rs): subset validation and type inference
- [`crates/bc-codegen/src/emit.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-codegen/src/emit.rs): the scalar Cranelift emitter
- [`crates/bc-codegen/src/simd.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-codegen/src/simd.rs): the vector emitter and its lane rules
- [`crates/bc-codegen/src/kleene.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-codegen/src/kleene.rs): `needs_kleene` / `is_null_propagating`
- [`crates/bc-codegen/src/cache.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-codegen/src/cache.rs): the process-wide compile memo
:::

## See also

- {doc}`Architecture </architecture/index>`: where a second execution tier is allowed to live.
- {doc}`Execution engine </architecture/internals/execution>`: the tiering contract at the architecture level.
- {doc}`Performance </user-guide/operate/tuning/performance>`: writing predicates that land on this tier.
- {doc}`Analytics benchmarks </benchmarks/results/analytics>`: the operator benchmarks on numeric filter and projection shapes.
- {doc}`Expression evaluation </architecture/deep-dives/query/expression-evaluation>`: the Tier-0 oracle it must match.
- {doc}`Morsel parallelism </architecture/deep-dives/operators/morsel-parallelism>`: the loop the compiled artifact runs inside.
- {doc}`Cost model </architecture/deep-dives/adaptive/cost-model>`: how `jit_speedup` prices a compilable expression.
