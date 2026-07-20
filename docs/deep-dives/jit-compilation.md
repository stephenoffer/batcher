# JIT compilation

*Tier-1* is Batcher's just-in-time compiler for scalar expressions. It lives in `bc-codegen` and turns an `Expr` tree into native machine code with Cranelift. This page describes what it compiles, how it handles nulls, and why it falls back more often than you might expect.

The problem it solves is allocation. The interpreter materializes a full Arrow array for every node of an expression tree. For `(a - b) * c` over a morsel that is two temporary 16,384-element arrays, three kernel passes, and three trips through memory. The values never stay in registers.

Tier-1 fixes that one thing. It compiles the expression tree into a single native loop: one pass over the row index, each output element computed in registers and written straight to the result buffer. No intermediates. The win grows with the depth of the expression.

It's a narrow tool, deliberately.

:::{important}
On the subset it accepts, the JIT must be **bit-for-bit identical** to `bc_expr::Expr::eval`.
On everything else it must **fall back silently**. A JIT that disagrees with the interpreter
is worse than no JIT, because the interpreter is the oracle everything else is tested
against. There is no third option where it is "close enough".
:::

## What compiles

From `crates/bc-codegen/src/lib.rs` and `analyze.rs`:

| Variant | Compiles | Notes |
|---|---|---|
| `Col` | `Int64`, `Float64` | `Date32` and tz-naive `Timestamp(µs)` as **comparison operands only**: they are an `i32` day count and an `i64` microsecond instant, and Arrow compares them by integer value |
| `Lit` | `Int`, `Float` | plus date/timestamp literals in a comparison. Bool and string literals do not compile |
| `Binary` | `Add`/`Sub`/`Mul`/`Div`/`Mod`, the six comparisons, `And`/`Or` over boolean sub-results | integer `Div`/`Mod` only against a constant divisor (below) |
| `Not` | of a boolean sub-result | |
| `Case` | over the numeric subset | lowered to a `select` chain in the interpreter's reverse-fold order, so the first matching `WHEN` wins |
| `Cast` | exact numeric (`i64 → f64`, or a no-op) | |
| everything else | no | strings, dates beyond comparison, lists, structs, `IsNull`, media decode → `CodegenError::Unsupported`, and the caller uses `Expr::eval` |

One subtlety that is easy to get wrong and is worth stating: an integer divisor compiles only
when it is a **nonzero, non-`-1` constant**. Cranelift's `sdiv`/`srem` trap on divide-by-zero
and on `i64::MIN / -1`; a constant divisor that is neither cannot trap, and truncation-toward-
zero plus dividend-signed remainder are exactly Arrow's semantics. A variable divisor stays on
the interpreter.

## The generated code

```text
fn(n: i64, cols: *const *const u8, out: *mut u8)
```

`cols` is an array of pointers, one per referenced column, in stable first-seen order. Each entry points at that column's raw values buffer. Passing them as an array rather than as separate arguments means there's no fixed ceiling on how many distinct columns an expression may reference. `out` is a fresh output buffer holding `n` `i64`s, `n` `f64`s, or, for a boolean result, a packed Arrow bitmask of `ceil(n/8)` zeroed bytes that the loop ORs one bit per row into, LSB-first, so the resulting `BooleanArray` wraps the buffer with no repack pass.

The Kleene body described below uses a wider signature that also carries per-column validity in and a validity buffer out.

Type promotion mirrors Arrow: if any operand in a subtree is `f64`, the whole subtree is
computed in `f64` with `i64 → f64` conversions inserted; otherwise it computes in `i64`.

## Nulls: three paths, not one

Arrow columns carry a validity bitmap, and the JIT loop reads a raw values buffer that has
garbage at null slots. Three cases, decided per batch in `CompiledExpr::eval`:

1. **No nulls in any referenced column.** Run the loop. This is the fast path.
2. **Nulls, and the expression is null-propagating.** Compute over the raw buffers, then AND
   the inputs' validity bitmaps together and apply the combined mask to the result. This is
   correct exactly when the SQL result is null *iff* an input is null, and when no operation
   can trap on the garbage at a masked-out slot: `Col`, `Lit`, `Add`/`Sub`/`Mul`, the
   comparisons, value-only math, exact casts, `Not`, and the constant-divisor `Div`/`Mod`
   above. The predicate is `kleene::is_null_propagating`.
3. **Nulls, and the expression is a compound predicate** (`And`/`Or` somewhere). The
   combined-mask trick is *wrong* here: `false AND null` is `false`, not null. So a second
   body is compiled in a value+validity ABI that carries per-column validity into the loop
   and reads a validity buffer back out: real three-valued logic, still on the JIT.
   `needs_kleene` selects it.

Anything else with nulls (`Case`, `Coalesce`) falls back to the interpreter for that batch.

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

The two fallbacks are independent, and both land on the same oracle. `try_compile` returning
`None` means the *expression* is outside the subset. `eval` returning `Err` means *this
batch* is, and only this batch reverts.

## SIMD

The scalar loop is the baseline. `crates/bc-codegen/src/simd.rs` emits a vector body when
every node is in the vectorizable subset: numeric leaves, integer `+`/`-`/`*` and float
`+`/`-`/`*`/`/`, the comparisons (the big filter win), `Not`, and exact numeric casts.

Width comes from the host at compile time via `bc_arrow::HardwareProfile`: 2 f64 lanes on SSE2 and NEON, 4 on AVX2, and 8 on AVX-512. Detection caps the automatic choice at 4 even on an AVX-512 host, because 512-bit code can down-clock the core, so the 8-lane width is opt-in. Cranelift legalizes a wider IR vector into native instructions where the ISA has them and splits it into 128-bit ops otherwise. A width that doesn't lower natively is at worst a no-op, never a wrong answer. A scalar remainder loop handles the rows past the last full step.

The excluded ops are excluded because their per-lane result would not be bit-identical:
integer `Div`/`Mod` scalarize and can trap, float `Mod` is an `fmod` libcall, `Math`/`Math2`
are libm libcalls, and `Case` is a branch. Boolean `And`/`Or` vectorize as a bitwise mask
combine, correct on a null-free batch, with the Kleene body kept as the per-batch fallback
when a referenced column turns out to carry nulls.

## Compile once, or lose

Cranelift compilation is the JIT's entire fixed cost, and `compile_expr` is a pure function
of `(expr, the types of the columns it references, the SIMD override)`. The sample batch is
consulted only for column types, never for values. So the artifact is reusable across every
morsel, every operator instance, and every `execute_plan` call that shares the triple.

Without a memo, the engine recompiles every filter and projection on *each* `execute_plan`. That cost is fixed, so it doesn't shrink with the input: on a small query it's pure loss, and it's worst on the per-batch streaming path and the per-operator UDF path, both of which call `execute_plan` in a loop. The memo in `crates/bc-codegen/src/cache.rs` is what makes the compile an admission price paid once rather than once per call.

:::{warning}
`crates/bc-codegen/src/cache.rs` keys its process-wide `HashMap` on the *full structural
rendering* of that triple, compared for equality and never merely hashed. A hash collision
handing back code compiled for a different expression is a silent wrong answer, and a silent
wrong answer is the one failure mode this whole tier is built to make impossible.
:::

The cache caps at 1024 entries (each owns a `JITModule`, a page or two of executable memory),
and a known-unsupported expression is remembered as such so the analysis is not repeated
either.

## Who calls it

Only the parallel executor. The sequential oracle passes `&None`:

```rust
// crates/bc-interp/src/ops/mod.rs
pub(crate) fn filter_batch(batch: &RecordBatch, predicate: &Expr) -> Result<RecordBatch, InterpError> {
    filter_batch_jit(batch, predicate, &None)   // the oracle never JITs
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

`bc-interp::par` compiles once per operator, using the first morsel as the type sample
(`ops::try_compile`), and shares the `Arc<CompiledExpr>` across rayon workers (`CompiledExpr`
is `Send + Sync`). Compiling per morsel would lose to the interpreter outright.

## Using it

There is no user-facing switch and no way to observe which tier ran, by design: the result
is identical either way. What you can observe is the shape that compiles:

```python
import batcher as bt

ds = bt.from_pydict({"a": [1, 2, 3, 4], "b": [10.0, 20.0, 30.0, 40.0], "s": ["x", "y", "x", "z"]})

# Numeric, null-free, arithmetic + comparison: this is the Tier-1 subset.
fast = ds.filter((bt.col("a") * 2 + 1) > bt.col("b") / 4).select("a", "b")

# A string function is outside the subset: the interpreter evaluates it.
slow = ds.filter(bt.col("s").str.starts_with("x")).select("a", "s")

print(fast.to_pydict())
print(slow.to_pydict())
```

```text
{'a': [1], 'b': [10.0]}
{'a': [1, 3], 's': ['x', 'x']}
```

## The honest limits

The subset is small. Most real analytical predicates touch a string, a date function, or a
null, and land on the interpreter. The JIT's leverage is concentrated in numeric-heavy
projection and filter chains, which is why the engine's benchmark wins cluster there
(`filter → count` at 0.20x DuckDB) and not in the general case (`filter → project` at 1.08x).

Growing the subset is the obvious work, and the rule for doing it is fixed: teach the
interpreter first, then either teach the JIT *and* prove parity against it, or leave the JIT
to fall back. Never ship a JIT path that disagrees with the oracle.

## Where the code lives

- `crates/bc-codegen/src/lib.rs`: `CompiledExpr`, the ABI, dispatch between scalar and SIMD
- `crates/bc-codegen/src/analyze.rs`: subset validation and type inference
- `crates/bc-codegen/src/emit.rs`: the scalar Cranelift emitter
- `crates/bc-codegen/src/simd.rs`: the vector emitter and its lane rules
- `crates/bc-codegen/src/kleene.rs`: `needs_kleene` / `is_null_propagating`
- `crates/bc-codegen/src/cache.rs`: the process-wide compile memo

## See also

:::{seealso}
- [Architecture](../architecture/index.md): where a second execution tier is allowed to live
- [Execution engine](../internals/execution.md): the tiering contract at the architecture level
- `docs/internals/mathematical_foundations.md` (in the repo, not a site page): the parity argument, stated formally
- [Performance](../user-guide/performance.md): writing predicates that land on this tier
- [Analytics benchmarks](../benchmarks/analytics.md): the `filter → count` and `filter → project` ratios quoted above
- [Expression evaluation](expression-evaluation.md): the Tier-0 oracle it must match
- [Morsel parallelism](morsel-parallelism.md): the loop the compiled artifact runs inside
- [Cost model](cost-model.md): how `jit_speedup` prices a compilable expression
:::
