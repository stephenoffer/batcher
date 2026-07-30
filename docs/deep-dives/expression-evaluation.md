# Expression evaluation

An *expression* is one `bc_expr::Expr` tree evaluated over one Arrow `RecordBatch`. Every scalar computation in the engine is one of these: a filter predicate, a projected column, a sort key, a group key, a window's `PARTITION BY`. There is exactly one such type, and `Expr::eval` is the correctness oracle for the whole system. The JIT, the parallel executor, and the distributed path are all measured against what it produces.

:::{important}
`Expr::eval` is the oracle. Every other tier is checked against it, which means it is not
allowed to be clever: it is allowed to be obviously right. When a new variant lands, it lands
here first. Only then does anything else get to compute it faster.
:::

## The shape of evaluation

`eval` is vectorized and recursive. Each node evaluates its children to full-length
`ArrayRef`s and applies an Arrow compute kernel:

```rust
// crates/bc-expr/src/eval/dispatch.rs
impl Expr {
    pub fn eval(&self, batch: &RecordBatch) -> Result<ArrayRef, ExprError> {
        match self {
            Expr::Col { name } => /* look up the column, decode a dictionary at the leaf */,
            Expr::Lit { value } => /* materialize a full-length constant array */,
            Expr::Binary { op, left, right } => {
                eval_binary(*op, left.eval(batch)?, right.eval(batch)?)
            }
            // ... one arm per variant
        }
    }
}
```

So `(a - b) * c` over a 16,384-row morsel makes three kernel passes and allocates two
intermediate arrays that exist only to be consumed by the next node:

```text
   Expr tree                    Tier-0: eval, bottom-up over one 16,384-row morsel
   ─────────────                ──────────────────────────────────────────────────

   binary(Mul)                  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
     ├── binary(Sub)            │  col "a"    │  │  col "b"    │  │  col "c"    │
     │     ├── col "a"          │  16,384×i64 │  │  16,384×i64 │  │  16,384×i64 │
     │     └── col "b"          └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
     └── col "c"                       └────────┬───────┘                │
                                          Sub kernel                     │
                                                ▼                        │
                                       ┌─────────────────┐               │
                                       │ tmp  16,384×i64 │  ← allocated  │
                                       └────────┬────────┘               │
                                                └────────┬───────────────┘
                                                    Mul kernel
                                                         ▼
                                                ┌─────────────────┐
                                                │ out  16,384×i64 │  ← allocated
                                                └─────────────────┘

   3 kernel passes. 2 intermediate arrays. 3 trips through memory.
   Tier-1 runs the same tree as one loop: nothing but `out` is allocated.
```

That is the cost of the interpreter, and it is the cost the {doc}`JIT <jit-compilation>` exists
to remove.

Two exceptions are worth knowing, because they change the constant factor without changing
the semantics:

**Scalar literal broadcast.** `try_scalar_binary` (`crates/bc-expr/src/eval/binary.rs`)
recognizes `<numeric column> <arith|cmp> <numeric literal>` in either operand order and
broadcasts the literal as a length-1 Arrow `Scalar`, a `Datum`, instead of materializing N copies of it. Same kernels, same promotion rules, bit-identical result. What it avoids is allocating a 16,384-element array of the number `1`.

**Dictionary decode at the leaf.** A `DictionaryArray` (common from Parquet) is decoded to
its value type in the `Col` arm, so every downstream kernel sees a plain array and no kernel
has to special-case dictionary encoding. The dictionary-native operations read the column
directly instead.

## Type promotion and null semantics

Promotion follows Arrow: if either operand of an arithmetic or comparison node is a float,
the node computes in `Float64`, otherwise in `Int64`. Narrow numerics never reach here. The FFI boundary widens `Int8/16/32 → Int64` and `Float16/32 → Float64` once in `crates/bc-py/src/normalize.rs`, so the kernels below see a small set of types.

Nulls propagate the way SQL says they do, which is not the way a naive `map` would:

| Node family | Validity of the result |
|---|---|
| arithmetic, comparison | propagates: any null input, null output |
| `And` / `Or` | Kleene three-valued. `false AND null` is `false`, not null; `true OR null` is `true` |
| `Case`, `Coalesce` | selects a branch, so validity is not a function of the inputs' validity at all |

The Kleene row is why the JIT has a separate ABI for compound predicates
(`crates/bc-codegen/src/kleene.rs`) and cannot use a combined validity mask for them.

:::{warning}
`x != x` does not detect NaN in this engine. The `!=` operator uses a *total* ordering, in
which `NaN == NaN`, so the familiar idiom silently returns all-false. Use `is_nan`, which is
its own `Expr` variant for exactly this reason. `is_inf` is likewise a variant rather than a
comparison, because an infinite literal does not survive a JSON round-trip and so cannot be
written as one. Both fall back to the interpreter in the JIT.
:::

## The function surface

The variants beyond the arithmetic core are grouped by family, one module each under
`crates/bc-expr/src/eval/`:

| Module | What it holds |
|---|---|
| `binary.rs` | arithmetic, comparison, boolean, bitwise, the scalar fast path |
| `cast.rs` | `CAST`, which is strict and errors on a bad value, and `TRY_CAST`, which yields null |
| `str/` | string functions: `contains`, `replace`, `substr`, regex, JSON, and the rest |
| `date.rs` | date/time extraction, `date_trunc`, `strftime`/`strptime`, date offsets |
| `math.rs` | unary/binary math, `coalesce`, `greatest`/`least`, `is_nan`/`is_inf` |
| `list.rs`, `list_ops/` | list construction, indexing, slicing, `filter`/`transform` |
| `map.rs`, `hash.rs`, `in_list.rs` | map lookup, hashing, `IN (...)` |
| `media/` | image/audio/video decode: library-backed, per-row, heavy |
| `security/`, `timezone.rs` | masking/encryption, timezone conversion |

The cast dtype vocabulary is not per-module. `bc_arrow::dtype_from_name` is the single name-to-type table, and the Python `CAST_DTYPES` set in `plan/types.py` is pinned to the live engine vocabulary by `tests/unit/test_dtype_registry_parity.py`, so the two cannot drift.

## Media decode is different

`.image`, `.audio`, and `.video` decodes are the one family that breaks the "per-row work is
cheap" assumption. Decoding a JPEG is thousands of times more expensive than adding two
integers, and the *input* is tiny (a 5 KB encoded blob), so a whole corpus of images can look
like a single morsel to the scheduler and get one core.

`Expr::contains_media_decode` (`crates/bc-expr/src/analyze.rs`) exists for exactly this: a
cheap static walk of the tree, consulted before execution, that tells the parallel executor
to lift its morsel-count-based worker cap and use every core. The match is exhaustive by
construction (a new `Expr` variant is a compile error there until it is classified), so a
future decode kernel cannot silently miss the signal.

## Using it

```python
import batcher as bt

ds = bt.from_pydict({"a": [1, 2, 3, None], "b": [10.0, 20.0, 30.0, 40.0]})

out = ds.select(
    "a",
    ratio=bt.col("b") / 10,
    big=(bt.col("a") > 1) & (bt.col("b") > 15),
    label=bt.when(bt.col("a").is_null()).then(bt.lit("missing")).otherwise(bt.lit("ok")),
).to_pydict()
print(out)
```

```text
{'a': [1, 2, 3, None],
 'ratio': [1.0, 2.0, 3.0, 4.0],
 'big': [False, True, True, None],
 'label': ['ok', 'ok', 'ok', 'missing']}
```

Row 4 shows both rules at once. `big` is null because `null > 1` is null and
`null AND true` is null: Kleene, not `false`. `label` is `'missing'` because `Case` selects
a branch on the *result* of `is_null`, which is never itself null.

## What it costs, and where it loses

The intermediate-array cost is real and it's why the JIT exists, but consider what the interpreter buys with it. Every sub-expression is a materialized Arrow array, so a batch can be handed to any Arrow kernel and any operator at any point, and an operator's state lives in Arrow rather than in registers. That is what lets a compiled pipeline be abandoned at a pipeline breaker without losing progress.

Kernel dispatch is per batch, not per row, so the overhead amortizes over 16,384 rows. On the operator benchmarks a filter-then-project over TPC-H `lineitem` at scale factor 1 runs in 13.9 ms against DuckDB's 12.9 ms and Polars' 9.2 ms. This is a shape where the engine is competitive but not ahead, and the gap is expression-evaluation overhead, not scheduling.

## The two tiers, side by side

Both tiers consume the same `Expr`. That is the whole guarantee.

::::{tab-set}
:::{tab-item} Tier-0 (bc-expr)
```text
every variant, every type, every null shape
one Arrow kernel pass per node, one intermediate array per node
sequential path: always this.  parallel path: whenever the JIT declines.
this is the answer everything else is compared against
```
:::

:::{tab-item} Tier-1 (bc-codegen)
```text
numeric Col/Lit/Binary/Not/Case/Cast only: a small subset on purpose
one Cranelift-compiled loop, values in registers, only the output allocated
bit-for-bit identical to Tier-0 on that subset, or it falls back to it
compiled once per (expr, column types, simd) and reused across every morsel
```
:::
::::

## Where the code lives

- `crates/bc-expr/src/lib.rs`: the `Expr` enum, the wire contract, serde tag `e`
- `crates/bc-expr/src/eval/dispatch.rs`: `Expr::eval`, the oracle
- `crates/bc-expr/src/eval/`: one module per function family
- `crates/bc-expr/src/analyze.rs`: static predicates over a tree, touching no data
- `crates/bc-py/src/normalize.rs`: the boundary type normalization the kernels rely on

## See also

:::{seealso}
- {doc}`Architecture <../architecture/index>`: why every scalar computation lives on this side of the boundary
- {doc}`Execution engine <../internals/execution>`: where `eval` is called from
- {doc}`Expressions <../user-guide/expressions>`: the Python surface that builds these trees
- {doc}`Expression reference <../api/expressions>`: every `Expr` method and accessor namespace
- {doc}`Analytics benchmarks <../benchmarks/analytics>`: the filter-then-project numbers above, in context
- {doc}`JIT compilation <jit-compilation>`: the Tier-1 path and what it can and cannot compile
- {doc}`Plan IR <plan-ir>`: how an `Expr` gets here from Python
- {doc}`Tensor columns <tensor-columns>`: what the media decode kernels produce
:::
