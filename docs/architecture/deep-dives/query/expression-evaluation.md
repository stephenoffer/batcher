# Expression evaluation

An *expression* is one `bc_expr::Expr` tree evaluated over one Arrow `RecordBatch`. Every scalar computation in the engine is one of these: a filter predicate, a projected column, a sort key, a group key, a window's `PARTITION BY`. There is exactly one such type, and `Expr::eval` is the correctness oracle the JIT, the parallel executor, and the distributed path are all measured against.

```python
import batcher as bt

ds = bt.from_pydict({"a": [1, 2, 3, None], "b": [10.0, 20.0, 30.0, 40.0]})
out = ds.select(
    "a",
    ratio=bt.col("b") / 10,
    big=(bt.col("a") > 1) & (bt.col("b") > 15),
    label=bt.when(bt.col("a").is_null()).then(bt.lit("missing")).otherwise(bt.lit("ok")),
)
print(out.to_pydict())
# {'a': [1, 2, 3, None], 'ratio': [1.0, 2.0, 3.0, 4.0], 'big': [False, True, True, None], 'label': ['ok', 'ok', 'ok', 'missing']}
```

Row 4 shows two rules at once. `big` is null because `null > 1` is null and `null AND true` is null. `label` is `'missing'` because `Case` selects on the result of {py:meth}`is_null <batcher.plan.expr_ir.core.Expr.is_null>`, which is never itself null.

:::{important}
`Expr::eval` is the oracle, so it is allowed to be obviously right rather than clever. A new variant lands here first. Only then does anything else get to compute it faster.
:::

## The shape of evaluation

`eval` is vectorized and recursive. Each node evaluates its children to full-length `ArrayRef`s and applies an Arrow compute kernel:

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

So `(a - b) * c` over a 16,384-row morsel makes three kernel passes and allocates two intermediate arrays, which is the cost the {doc}`JIT </architecture/deep-dives/query/jit-compilation>` removes by running the same tree as one loop.

:::{dropdown} The kernel passes for `(a - b) * c`, drawn out
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
:::

Two shortcuts change the constant factor without changing the semantics:

- **Scalar literal broadcast.** `try_scalar_binary` ([`crates/bc-expr/src/eval/binary.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-expr/src/eval/binary.rs)) recognizes `<numeric column> <arith|cmp> <numeric literal>` in either order and passes the literal as a length-1 Arrow `Scalar` instead of materializing N copies. Same kernels, bit-identical result.
- **Dictionary decode at the leaf.** A `DictionaryArray`, common from Parquet, is decoded in the `Col` arm, so no downstream kernel special-cases dictionary encoding.

Kernel dispatch is per batch, not per row, so its overhead amortizes over 16,384 rows. Every sub-expression is also a materialized Arrow array, which is what lets the JIT decline an expression, or a single batch, with nothing lost.

## Type promotion and null semantics

Promotion follows Arrow: if either operand of an arithmetic or comparison node is a float, the node computes in `Float64`, otherwise in `Int64`. The FFI boundary widens `Int8/16/32 → Int64` and `Float16/32 → Float64` once, in [`crates/bc-py/src/normalize.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-py/src/normalize.rs), so the kernels see a small set of types.

Nulls propagate the way SQL says:

| Node family | Validity of the result |
|---|---|
| arithmetic, comparison | propagates: any null input, null output |
| `And` / `Or` | Kleene three-valued. `false AND null` is `false`, not null; `true OR null` is `true` |
| `Case`, `Coalesce` | selects a branch, so validity is not a function of the inputs' validity at all |

```python
import batcher as bt

ds = bt.from_pydict({"x": [1.0, None]})
out = ds.select(
    or_true=(bt.col("x") > 0) | bt.lit(True),
    and_false=(bt.col("x") > 0) & bt.lit(False),
    plus=bt.col("x") + 1,
)
print(out.to_pydict())  # {'or_true': [True, True], 'and_false': [False, False], 'plus': [2.0, None]}
```

The Kleene row is why the JIT has a separate ABI for compound predicates ([`crates/bc-codegen/src/kleene.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-codegen/src/kleene.rs)).

![The validity bitmap travelling through one expression over four rows. An Arrow column is two buffers: column a holds 3, 17, a null and 24 with validity 1, 1, 0, 1, and column b holds 9, 2, 9 and 8 with validity all 1. A null slot still holds a payload, and the bitmap beside it is the only thing that says to ignore that payload. Comparison carries the bitmap forward: a greater than 10 is false, true, unknown, true, with validity 1, 1, 0, 1, because Arrow's compare kernels return null and never false where an input is null, so row 3 is unknown rather than excluded. b less than 5 is false, true, false, false, all valid. Their and_kleene is false, true, false, false with validity 1, 1, 1, 1: false AND null is false, so row 3 is valid again, where Arrow's plain and would have propagated the null instead. What that boolean column means then depends on how it is used. Kept as a column, three values survive: true, false and unknown. Used as a filter, truthy() ANDs the values with the validity and leaves no bitmap on the result, so unknown folds to false and the row goes.](/_static/diagrams/expr_eval_nulls.svg)

:::{warning}
`x != x` does not detect NaN here. `!=` uses a *total* ordering in which `NaN == NaN`, so the familiar idiom returns all-false. Use {py:meth}`is_nan <batcher.plan.expr_ir.core.Expr.is_nan>`, which is its own {py:class}`Expr <batcher.plan.expr_ir.core.Expr>` variant for exactly this reason:

```python
import batcher as bt

ds = bt.from_pydict({"x": [1.0, float("nan"), None]})
print(ds.select(ne=bt.col("x") != bt.col("x"), nan=bt.col("x").is_nan()).to_pydict())
# {'ne': [False, False, None], 'nan': [False, True, None]}
```
:::

## The function surface

The variants beyond the arithmetic core are grouped by family, one module each under [`crates/bc-expr/src/eval/`](https://github.com/stephenoffer/batcher/tree/main/crates/bc-expr/src/eval):

| Module | What it holds |
|---|---|
| `binary.rs` | arithmetic, comparison, boolean, bitwise, the scalar fast path |
| `cast.rs` | `CAST`, which is strict and errors on a bad value, and `TRY_CAST`, which yields null |
| `str/` | string functions: `contains`, `replace`, `substr`, regex, JSON, and the rest |
| `temporal/` | date/time extraction, `date_trunc`, `strftime`/`strptime`, date offsets, timezone conversion |
| `math.rs` | unary/binary math, `greatest`/`least`, `is_nan`/`is_inf` |
| `branch/` | `CASE` and `coalesce` |
| `list.rs`, `list_ops/` | list construction, indexing, slicing, `filter`/`transform` |
| `map.rs`, `hash.rs`, `in_list.rs` | map lookup, hashing, `IN (...)` |
| `media/` | image/audio/video decode: library-backed, per-row, heavy |
| `security/` | masking and encryption |

The cast row in practice:

```python
import batcher as bt

ds = bt.from_pydict({"s": ["1", "2", "x"]})
print(ds.select(n=bt.col("s").try_cast("int64")).to_pydict())  # {'n': [1, 2, None]}
```

`bc_arrow::dtype_from_name` is the single name-to-type table for casts, and the Python `CAST_DTYPES` set is pinned to it by [`tests/unit/test_dtype_registry_parity.py`](https://github.com/stephenoffer/batcher/blob/main/tests/unit/test_dtype_registry_parity.py).

## Media decode

{py:class}`.image <batcher.plan.expr_ir.image._ImageNamespace>`, {py:class}`.audio <batcher.plan.expr_ir.audio._AudioNamespace>`, and {py:class}`.video <batcher.plan.expr_ir.video._VideoNamespace>` decodes are thousands of times more expensive per row than integer arithmetic, while their input is a few kilobytes, so a whole corpus can look like one morsel to the scheduler. `Expr::contains_media_decode` ([`crates/bc-expr/src/analyze.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-expr/src/analyze.rs)) is a static walk that tells the parallel executor to lift its morsel-count worker cap and use every core. Its match is exhaustive, so a new variant is a compile error until it is classified.

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
numeric and temporal Col/Lit/Binary/Not/Case/Cast/Math: a small subset on purpose
nullable inputs via a combined validity mask or a Kleene ABI
one Cranelift-compiled loop, values in registers, only the output allocated
bit-for-bit identical to Tier-0 on that subset, or it falls back to it
compiled once per (expr, column types, simd) and reused across every morsel
```
:::
::::

:::{dropdown} Where the code lives
- [`crates/bc-expr/src/lib.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-expr/src/lib.rs): the `Expr` enum, the wire contract, serde tag `e`
- [`crates/bc-expr/src/eval/dispatch.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-expr/src/eval/dispatch.rs): `Expr::eval`, the oracle
- [`crates/bc-expr/src/eval/`](https://github.com/stephenoffer/batcher/tree/main/crates/bc-expr/src/eval): one module per function family
- [`crates/bc-expr/src/analyze.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-expr/src/analyze.rs): static predicates over a tree, touching no data
- [`crates/bc-py/src/normalize.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-py/src/normalize.rs): the boundary type normalization the kernels rely on
:::

## See also

- {doc}`Architecture </architecture/index>`: why every scalar computation lives on this side of the boundary.
- {doc}`Execution engine </architecture/internals/execution>`: where `eval` is called from.
- {doc}`Expressions </user-guide/transform/columns/expressions>`: the Python surface that builds these trees.
- {doc}`Expression reference </api/relational/expressions>`: every `Expr` method and accessor namespace.
- {doc}`Analytics benchmarks </benchmarks/results/analytics>`: the operator benchmarks against DuckDB and Polars.
- {doc}`JIT compilation </architecture/deep-dives/query/jit-compilation>`: the Tier-1 path and what it compiles.
- {doc}`Plan IR </architecture/deep-dives/query/plan-ir>`: how an `Expr` gets here from Python.
- {doc}`Tensor columns </architecture/deep-dives/memory/tensor-columns>`: what the media decode kernels produce.
