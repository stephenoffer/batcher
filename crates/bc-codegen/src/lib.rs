//! `bc-codegen` — a Cranelift JIT backend for `bc-expr` scalar expressions.
//!
//! # What this does
//!
//! [`compile_and_eval`] takes a [`bc_expr::Expr`] and an Arrow
//! [`RecordBatch`](arrow::array::RecordBatch), compiles the expression to native
//! machine code with Cranelift, and runs it once over the whole batch, producing
//! a single result [`ArrayRef`](arrow::array::ArrayRef). The output is bit-for-bit
//! identical to `bc_expr::Expr::eval` on the supported subset — that is the
//! contract that lets the interpreter act as a differential oracle for the JIT
//! (see the tests in this crate).
//!
//! # Supported subset
//!
//! Only a numeric, null-free subset compiles. Anything outside it returns
//! [`CodegenError::Unsupported`] so the caller can fall back to the interpreter:
//!
//! * `Col` — an `Int64` or `Float64` column with `null_count() == 0`.
//! * `Lit` — `Int` or `Float` literals (Bool/Str are unsupported).
//! * `Binary` with `Add`/`Sub`/`Mul`/`Div`/`Mod` (arithmetic),
//!   `Eq`/`Ne`/`Lt`/`Le`/`Gt`/`Ge` (comparison), or `And`/`Or` over two boolean
//!   sub-results (e.g. compound filter predicates).
//! * `Not` of a boolean sub-result.
//! * `Case` (`CASE WHEN`) over the numeric subset: every `WHEN` is a boolean
//!   predicate and every `THEN`/`ELSE` is numeric; lowered to a `select` chain
//!   in the interpreter's reverse-fold order (first matching `WHEN` wins).
//!
//! Type promotion matches Arrow / `bc-expr`: if *any* operand in a subtree is
//! `f64` the whole subtree is computed in `f64` (with `i64 -> f64` conversions
//! inserted as needed); otherwise it is computed in `i64`. Comparisons produce a
//! `BooleanArray`; arithmetic produces an `Int64Array` or `Float64Array`
//! depending on whether the operands promoted to float.
//!
//! `IsNull`/`IsNotNull`, string/date/list/struct functions, string
//! columns, and any column containing nulls are all `Unsupported`.
//!
//! # How `bc-interp` uses this
//!
//! The interpreter tier tries the JIT first and silently falls back on any
//! `Unsupported` (or other [`CodegenError`]) result:
//!
//! ```ignore
//! match bc_codegen::compile_and_eval(expr, batch) {
//!     Ok(array) => array,
//!     Err(_) => expr.eval(batch)?, // interpreter oracle
//! }
//! ```
//!
//! Because the interpreter materializes a full intermediate `ArrayRef` for every
//! sub-expression, a compound expression like `(a - b) * c` allocates two
//! temporary arrays and makes three kernel passes. The JIT instead fuses the
//! whole tree into one tight loop over the row index, computing each output
//! element in registers and writing it straight to the result buffer — no
//! intermediate arrays, one pass. The win grows with expression depth.
//!
//! # ABI of the generated function
//!
//! The compiled function has the signature (all pointers are non-null, aligned,
//! and valid for `n` elements):
//!
//! ```text
//! fn(n: i64, col0: *const u8, col1: *const u8, ..., out: *mut u8)
//! ```
//!
//! Each `colK` points at the raw values buffer of the K-th referenced column
//! (`PrimitiveArray::values()`, i.e. `&[i64]` or `&[f64]`), in stable
//! first-seen order. `out` points at a freshly allocated buffer of `n` elements:
//! `i64` for integer arithmetic, `f64` for float arithmetic, and `u8` (0/1) for
//! comparisons, which is then copied into the result Arrow array. The loop body
//! reads `colK[i]`, evaluates the expression, and stores to `out[i]`.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{ArrayRef, BooleanArray, Float64Array, Int64Array, PrimitiveArray, RecordBatch};
use arrow::buffer::ScalarBuffer;
use arrow::datatypes::{ArrowPrimitiveType, DataType, Float64Type, Int64Type};

use cranelift_codegen::ir::{types, Type};
use cranelift_jit::JITModule;

use crate::analyze::analyze;
use crate::compile::{compile, compile_simd};

mod analyze;
mod compile;
mod emit;

/// Errors surfaced by the JIT backend. `Unsupported` is the signal for the
/// caller to fall back to the interpreter; the rest are genuine failures.
#[derive(Debug, thiserror::Error)]
pub enum CodegenError {
    /// The expression (or a column type) is outside the compilable subset.
    #[error("unsupported by JIT backend: {0}")]
    Unsupported(String),

    /// A column referenced by the expression is absent from the batch.
    #[error("unknown column: {0}")]
    UnknownColumn(String),

    /// Cranelift failed to build or finalize the function.
    #[error("cranelift error: {0}")]
    Cranelift(String),
}

/// The scalar type a sub-expression evaluates to in the generated code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScalarTy {
    I64,
    F64,
    Bool,
}

impl ScalarTy {
    pub(crate) fn clif(self) -> Type {
        match self {
            ScalarTy::I64 => types::I64,
            ScalarTy::F64 => types::F64,
            ScalarTy::Bool => types::I8,
        }
    }
}

/// Compile `expr` to native code and evaluate it over `batch`.
///
/// Returns [`CodegenError::Unsupported`] for any expression outside the
/// supported subset (see the crate docs) so the caller can fall back to
/// `bc_expr::Expr::eval`.
pub fn compile_and_eval(
    expr: &bc_expr::Expr,
    batch: &RecordBatch,
) -> Result<ArrayRef, CodegenError> {
    compile_expr(expr, batch)?.eval(batch)
}

/// A JIT-compiled expression that can be evaluated over many batches.
///
/// Compile once (per query/operator), then call [`CompiledExpr::eval`] on each
/// morsel — this is the form the engine uses, so the one-time compile cost is
/// amortized across the whole input (a per-batch compile would lose to the
/// interpreter).
pub struct CompiledExpr {
    columns: Vec<String>,
    result_ty: ScalarTy,
    compiled: Compiled,
    /// Whether the expression is the *null-propagating subset* (`Col`/`Lit`,
    /// `Add`/`Sub`/`Mul`/`Div`/`Mod`/comparison, value-only math/cast, and `Not`).
    /// For these the result is null exactly where any referenced column is null — a
    /// single combined validity mask — and the JIT can compute values over the raw
    /// buffers (garbage at null slots is masked away, and these ops cannot trap;
    /// `analyze` only admits non-trapping constant integer divisors). Anything else
    /// (Kleene `And`/`Or`, `Case`, `Coalesce`) falls back to the interpreter when a
    /// referenced column contains nulls.
    null_safe: bool,
    /// Whether this was compiled in the Kleene value+validity ABI (it contains a
    /// boolean `And`/`Or` and no `Case`/`Coalesce`). When set, `eval` supplies
    /// per-column validity arrays and reads back a validity buffer, so nullable
    /// compound predicates run on the JIT with correct three-valued logic instead of
    /// falling back to the interpreter.
    kleene: bool,
}

/// True if `expr` should use the Kleene value+validity ABI: it contains a boolean
/// `And`/`Or` (whose null semantics the simple combined-mask gets wrong) and every
/// node is one `emit_validity` supports — i.e. no `Case`/`Coalesce`, whose result
/// value itself depends on validity. Such an `And`/`Or` over a `Case` keeps falling
/// back to the interpreter on nullable input (the value-only path).
fn needs_kleene(expr: &bc_expr::Expr) -> bool {
    contains_and_or(expr) && kleene_supported(expr)
}

fn contains_and_or(expr: &bc_expr::Expr) -> bool {
    use bc_expr::{BinaryOp, Expr};
    match expr {
        Expr::Binary {
            op: BinaryOp::And | BinaryOp::Or,
            ..
        } => true,
        Expr::Binary { left, right, .. } | Expr::Math2 { left, right, .. } => {
            contains_and_or(left) || contains_and_or(right)
        }
        Expr::Not { input } | Expr::Cast { input, .. } | Expr::Math { input, .. } => {
            contains_and_or(input)
        }
        _ => false,
    }
}

fn kleene_supported(expr: &bc_expr::Expr) -> bool {
    use bc_expr::Expr;
    match expr {
        Expr::Col { .. } | Expr::Lit { .. } => true,
        Expr::Binary { left, right, .. } | Expr::Math2 { left, right, .. } => {
            kleene_supported(left) && kleene_supported(right)
        }
        Expr::Not { input } | Expr::Cast { input, .. } | Expr::Math { input, .. } => {
            kleene_supported(input)
        }
        _ => false, // Case/Coalesce/etc. — value depends on validity; not supported.
    }
}

/// True if `expr` is the pure-`F64` arithmetic subset the vector (`compile_simd`)
/// path handles: every leaf is an `F64` column or a float literal, every interior
/// node is `+`/`-`/`*`/`/`. These are exactly the ops whose `F64X2` lanes are IEEE-
/// identical to the scalar path (so parity holds), and whose result is `F64` — no
/// comparisons (boolean output), no `Mod` (libcall), no int/cast lanes, no libm.
fn simd_supported(expr: &bc_expr::Expr, cols: &ColumnSet) -> bool {
    use bc_expr::{BinaryOp, Expr, Literal};
    match expr {
        Expr::Col { name } => cols.ty.get(name) == Some(&ScalarTy::F64),
        Expr::Lit {
            value: Literal::Float(_),
        } => true,
        Expr::Binary {
            op: BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div,
            left,
            right,
        } => simd_supported(left, cols) && simd_supported(right, cols),
        _ => false,
    }
}

/// True if every node is in the null-propagating subset, so the JIT may run on
/// nullable input and recover correctness by masking the output with the inputs'
/// combined validity. This holds for ops whose SQL result is null **iff** an input
/// is null and which never trap on a garbage value at a masked-out slot: column
/// refs, literals, `Add`/`Sub`/`Mul`/comparisons, value-only unary/binary math,
/// and exact numeric casts.
///
/// `Div`/`Mod` are included: this flag is only consulted *after* `analyze` already
/// compiled the expression, and `analyze` admits an integer divisor only when it is
/// a nonzero, non-`-1` constant (float div is IEEE and never traps). So a Div/Mod
/// that reached here cannot trap on the garbage value at a masked-out null slot, and
/// its SQL result is null iff a value input is null — exactly simple propagation.
/// `Not` is included similarly: `NOT null = null` (and a garbage bool can't trap).
///
/// Excludes boolean `And`/`Or`, `Case`, `Coalesce` — their null semantics (Kleene /
/// branch selection / first-non-null) are *not* simple propagation (e.g.
/// `false AND null = false`, not null), so the combined-mask recovery would give a
/// wrong validity; those need per-node validity tracking and stay on the interpreter
/// for nullable input. (A node the JIT cannot compile makes the whole compile fall
/// back before this flag is consulted, so listing a not-yet-compiled op is harmless.)
fn is_null_propagating(expr: &bc_expr::Expr) -> bool {
    use bc_expr::{BinaryOp, Expr};
    match expr {
        Expr::Col { .. } | Expr::Lit { .. } => true,
        Expr::Binary { op, left, right } => {
            matches!(
                op,
                BinaryOp::Add
                    | BinaryOp::Sub
                    | BinaryOp::Mul
                    | BinaryOp::Div
                    | BinaryOp::Mod
                    | BinaryOp::Eq
                    | BinaryOp::Ne
                    | BinaryOp::Lt
                    | BinaryOp::Le
                    | BinaryOp::Gt
                    | BinaryOp::Ge
            ) && is_null_propagating(left)
                && is_null_propagating(right)
        }
        // `NOT null = null`; the garbage bool at a null slot is masked out and can't
        // trap, so logical NOT over a propagating operand still propagates.
        Expr::Not { input } => is_null_propagating(input),
        // Value-only math and exact numeric casts propagate nulls and never trap.
        Expr::Math { input, .. } | Expr::Cast { input, .. } => is_null_propagating(input),
        Expr::Math2 { left, right, .. } => is_null_propagating(left) && is_null_propagating(right),
        _ => false,
    }
}

// SAFETY: after finalization the JIT code is immutable, and the generated
// function is a pure, reentrant computation over caller-provided pointers (reads
// inputs, writes a distinct output buffer per call, no shared mutable state).
// The owned module is never used for further compilation. So a shared
// `&CompiledExpr` is safe to call from multiple threads concurrently.
unsafe impl Send for CompiledExpr {}
unsafe impl Sync for CompiledExpr {}

/// JIT-compile `expr`, using `batch` only as a representative for column types.
/// Returns [`CodegenError::Unsupported`] for anything outside the supported
/// subset (the caller then uses the interpreter).
pub fn compile_expr(
    expr: &bc_expr::Expr,
    batch: &RecordBatch,
) -> Result<CompiledExpr, CodegenError> {
    let mut cols = ColumnSet::default();
    let result_ty = analyze(expr, batch, &mut cols)?;
    let kleene = needs_kleene(expr);
    // The vector path is a drop-in for the scalar F64 value path (same ABI, same
    // null-propagating masking in `eval`), so it needs no flag — only a different
    // compiled body. Used for the pure-F64 arithmetic subset.
    let simd = !kleene && result_ty == ScalarTy::F64 && simd_supported(expr, &cols);
    let compiled = if simd {
        compile_simd(expr, &cols)?
    } else {
        compile(expr, &cols, result_ty, kleene)?
    };
    Ok(CompiledExpr {
        columns: cols.order,
        result_ty,
        compiled,
        null_safe: is_null_propagating(expr),
        kleene,
    })
}

impl CompiledExpr {
    /// Evaluate over `batch`. Returns [`CodegenError::Unsupported`] if the batch
    /// violates a JIT precondition (a referenced column is missing, the wrong
    /// type, or contains nulls) so the caller can fall back to the interpreter
    /// for that particular batch.
    pub fn eval(&self, batch: &RecordBatch) -> Result<ArrayRef, CodegenError> {
        if self.kleene {
            return self.eval_kleene(batch);
        }
        let n = batch.num_rows();
        let mut col_ptrs: Vec<*const u8> = Vec::with_capacity(self.columns.len());
        // Combined validity for the null-propagating path: starts all-valid and
        // ANDs in each referenced nullable column. `None` means no nulls anywhere
        // (the common fast path — no masking needed).
        let mut validity: Option<Vec<bool>> = None;
        for name in &self.columns {
            let arr = batch
                .column_by_name(name)
                .ok_or_else(|| CodegenError::UnknownColumn(name.clone()))?;
            if arr.null_count() != 0 {
                if !self.null_safe {
                    // Outside the null-propagating subset (Div/Mod/And/Or/Case/…):
                    // let the interpreter handle this batch's null semantics.
                    return Err(CodegenError::Unsupported(format!(
                        "column `{name}` contains nulls"
                    )));
                }
                let mask = validity.get_or_insert_with(|| vec![true; n]);
                for (i, slot) in mask.iter_mut().enumerate() {
                    *slot &= arr.is_valid(i);
                }
            }
            let ptr = match arr.data_type() {
                DataType::Int64 => arr
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .as_ptr() as *const u8,
                DataType::Float64 => arr
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap()
                    .values()
                    .as_ptr() as *const u8,
                other => {
                    return Err(CodegenError::Unsupported(format!(
                        "column `{name}` has type {other:?}"
                    )))
                }
            };
            col_ptrs.push(ptr);
        }

        let p = self.compiled.ptr;
        let nargs = self.compiled.nargs;
        // Values are computed over the raw buffers (garbage at null slots); the
        // validity mask then nulls those slots out — bit-identical to the
        // interpreter, whose arrow kernels propagate nulls the same way.
        match self.result_ty {
            ScalarTy::I64 => {
                let mut out = vec![0i64; n];
                run(p, nargs, n, &col_ptrs, out.as_mut_ptr() as *mut u8);
                Ok(Arc::new(finish_primitive::<Int64Type>(out, validity)))
            }
            ScalarTy::F64 => {
                let mut out = vec![0f64; n];
                run(p, nargs, n, &col_ptrs, out.as_mut_ptr() as *mut u8);
                Ok(Arc::new(finish_primitive::<Float64Type>(out, validity)))
            }
            ScalarTy::Bool => {
                let mut out = vec![0u8; n];
                run(p, nargs, n, &col_ptrs, out.as_mut_ptr());
                let bools = out.into_iter().map(|b| b != 0);
                Ok(Arc::new(match validity {
                    None => BooleanArray::from(bools.collect::<Vec<bool>>()),
                    Some(valid) => {
                        BooleanArray::from_iter(bools.zip(valid).map(|(b, ok)| ok.then_some(b)))
                    }
                }))
            }
        }
    }

    /// Kleene path: supply each referenced column's value buffer **and** a parallel
    /// per-row validity array (1 = valid), run the value+validity ABI, and build the
    /// boolean result from the value and validity outputs. A Kleene expression is
    /// always boolean, so the result is a `BooleanArray` whose nulls come from the
    /// computed validity (correct three-valued logic for `And`/`Or`).
    fn eval_kleene(&self, batch: &RecordBatch) -> Result<ArrayRef, CodegenError> {
        let n = batch.num_rows();
        let mut col_ptrs: Vec<*const u8> = Vec::with_capacity(self.columns.len());
        let mut valid_arrays: Vec<Vec<u8>> = Vec::with_capacity(self.columns.len());
        for name in &self.columns {
            let arr = batch
                .column_by_name(name)
                .ok_or_else(|| CodegenError::UnknownColumn(name.clone()))?;
            let ptr = match arr.data_type() {
                DataType::Int64 => arr
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .as_ptr() as *const u8,
                DataType::Float64 => arr
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap()
                    .values()
                    .as_ptr() as *const u8,
                other => {
                    return Err(CodegenError::Unsupported(format!(
                        "column `{name}` has type {other:?}"
                    )))
                }
            };
            col_ptrs.push(ptr);
            // Per-column validity bytes (all-valid when the column has no nulls).
            let mut v = vec![1u8; n];
            if arr.null_count() != 0 {
                for (i, slot) in v.iter_mut().enumerate() {
                    *slot = arr.is_valid(i) as u8;
                }
            }
            valid_arrays.push(v);
        }
        let null_ptrs: Vec<*const u8> = valid_arrays.iter().map(|v| v.as_ptr()).collect();
        let mut out = vec![0u8; n];
        let mut valid_out = vec![1u8; n];
        run_kleene(
            self.compiled.ptr,
            self.compiled.nargs,
            n,
            &col_ptrs,
            &null_ptrs,
            out.as_mut_ptr(),
            valid_out.as_mut_ptr(),
        );
        Ok(Arc::new(BooleanArray::from_iter(
            out.iter()
                .zip(valid_out)
                .map(|(b, ok)| (ok != 0).then_some(*b != 0)),
        )))
    }
}

/// Build a primitive array from the JIT's raw output values, nulling out positions
/// the validity mask marks invalid. The common `None`-validity (no-null) path is
/// **zero-copy**: the output `Vec`'s allocation becomes the Arrow value buffer
/// directly, avoiding the element-by-element re-collect a `FromIterator` build does.
fn finish_primitive<P: ArrowPrimitiveType>(
    values: Vec<P::Native>,
    validity: Option<Vec<bool>>,
) -> PrimitiveArray<P> {
    match validity {
        None => PrimitiveArray::<P>::new(ScalarBuffer::from(values), None),
        Some(valid) => values
            .into_iter()
            .zip(valid)
            .map(|(v, ok)| ok.then_some(v))
            .collect(),
    }
}

/// A compiled function pointer plus the module that owns its code. The module
/// must outlive every call (dropping it frees the executable memory).
pub(crate) struct Compiled {
    pub(crate) ptr: *const u8,
    pub(crate) nargs: usize,
    pub(crate) _module: JITModule,
}

/// Invoke the compiled function.
///
/// The generated ABI is `(n: i64, cols: *const *const u8, out: *mut u8)` — the
/// column base pointers are passed as one array the callee indexes, so there is no
/// per-arity trampoline and no ceiling on the column count. Borrows the function
/// pointer so a [`CompiledExpr`] can be called many times.
fn run(p: *const u8, nargs: usize, n: usize, cols: &[*const u8], out: *mut u8) {
    let n = n as i64;
    debug_assert_eq!(nargs, cols.len());
    // SAFETY: `p` is a finalized Cranelift function with the `(i64, *const *const
    // u8, *mut u8)` signature built in `compile`; `cols` has exactly `nargs`
    // pointers (the columns referenced, in order), each valid for `n` elements
    // (allocated above / validated null-free in `analyze`), and `out` for `n`.
    unsafe {
        let f: extern "C" fn(i64, *const *const u8, *mut u8) = std::mem::transmute(p);
        f(n, cols.as_ptr(), out);
    }
}

/// Invoke the Kleene-ABI compiled function:
/// `(n, cols: *const *const u8, nulls: *const *const u8, out: *mut u8, valid: *mut u8)`.
/// `cols`/`nulls` each carry `nargs` pointers; `out` and `valid` cover `n` bytes each.
#[allow(clippy::too_many_arguments)]
fn run_kleene(
    p: *const u8,
    nargs: usize,
    n: usize,
    cols: &[*const u8],
    nulls: &[*const u8],
    out: *mut u8,
    valid: *mut u8,
) {
    let n = n as i64;
    debug_assert_eq!(nargs, cols.len());
    debug_assert_eq!(nargs, nulls.len());
    // SAFETY: `p` is the finalized Kleene-ABI function built in `compile` with
    // `kleene = true`; `cols`/`nulls` each have exactly `nargs` pointers valid for
    // `n` elements, and `out`/`valid` are distinct buffers of `n` bytes.
    unsafe {
        let f: extern "C" fn(i64, *const *const u8, *const *const u8, *mut u8, *mut u8) =
            std::mem::transmute(p);
        f(n, cols.as_ptr(), nulls.as_ptr(), out, valid);
    }
}

/// Referenced columns in stable first-seen order, plus their scalar types.
#[derive(Default)]
pub(crate) struct ColumnSet {
    pub(crate) order: Vec<String>,
    pub(crate) ty: HashMap<String, ScalarTy>,
}

impl ColumnSet {
    /// Index of `name` in the argument list (its position in first-seen order).
    pub(crate) fn index(&self, name: &str) -> usize {
        self.order.iter().position(|c| c == name).unwrap()
    }
}

/// The libm symbol a single-arg [`bc_expr::MathFunc`] lowers to, or `None` if it
/// is not lowered via a libcall (handled elsewhere or left to the interpreter).
/// Keeping this map in one place keeps `analyze` and `emit_typed` in sync.
pub(crate) fn libm_unary_symbol(func: bc_expr::MathFunc) -> Option<&'static str> {
    use bc_expr::MathFunc::*;
    Some(match func {
        Ln => "log",
        Log10 => "log10",
        Log2 => "log2",
        Exp => "exp",
        Sin => "sin",
        Cos => "cos",
        Tan => "tan",
        Sinh => "sinh",
        Cosh => "cosh",
        Tanh => "tanh",
        Asin => "asin",
        Acos => "acos",
        Atan => "atan",
        Cbrt => "cbrt",
        _ => return None,
    })
}

/// The libm symbol a two-arg [`bc_expr::Math2Func`] lowers to, or `None` if it is
/// not a single libm call (`Round`, which takes a digit count, stays on the
/// interpreter).
pub(crate) fn libm_binary_symbol(func: bc_expr::Math2Func) -> Option<&'static str> {
    use bc_expr::Math2Func::*;
    Some(match func {
        Pow => "pow",
        Atan2 => "atan2",
        // Round (digit count), and the integer-semantics gcd/lcm/hypot, are not a
        // single libm call — they stay on the interpreter (the JIT falls back).
        Round | Gcd | Lcm | Hypot => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use bc_expr::{BinaryOp, Expr, Literal};
    use std::sync::Arc;

    use arrow::array::{Array, StringArray};
    use arrow::datatypes::{Field, Schema};

    /// Tiny deterministic xorshift PRNG (avoids an external `rand` dep).
    struct Rng(u64);
    impl Rng {
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn i64_small(&mut self) -> i64 {
            // Keep magnitudes modest so i64 arithmetic can't overflow in tests.
            (self.next_u64() % 2001) as i64 - 1000
        }
        fn f64_small(&mut self) -> f64 {
            (self.i64_small() as f64) / 7.0
        }
    }

    fn col(name: &str) -> Expr {
        Expr::Col { name: name.into() }
    }
    fn lit_i(v: i64) -> Expr {
        Expr::Lit {
            value: Literal::Int(v),
        }
    }
    fn lit_f(v: f64) -> Expr {
        Expr::Lit {
            value: Literal::Float(v),
        }
    }
    fn bin(op: BinaryOp, l: Expr, r: Expr) -> Expr {
        Expr::Binary {
            op,
            left: Box::new(l),
            right: Box::new(r),
        }
    }

    fn make_batch(n: usize, seed: u64) -> RecordBatch {
        let mut rng = Rng(seed);
        let a: Vec<i64> = (0..n).map(|_| rng.i64_small()).collect();
        // avoid zero divisors for `b`
        let b: Vec<i64> = (0..n)
            .map(|_| {
                let v = rng.i64_small();
                if v == 0 {
                    1
                } else {
                    v
                }
            })
            .collect();
        let c: Vec<f64> = (0..n)
            .map(|_| {
                let v = rng.f64_small();
                if v == 0.0 {
                    1.5
                } else {
                    v
                }
            })
            .collect();
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Int64, false),
            Field::new("c", DataType::Float64, false),
        ]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(Int64Array::from(a)),
                Arc::new(Int64Array::from(b)),
                Arc::new(Float64Array::from(c)),
            ],
        )
        .unwrap()
    }

    /// Assert the JIT and interpreter produce identical arrays.
    fn assert_parity(expr: &Expr, batch: &RecordBatch) {
        let jit = compile_and_eval(expr, batch).expect("should compile");
        let oracle = expr.eval(batch).expect("interpreter eval");
        assert_eq!(
            jit.data_type(),
            oracle.data_type(),
            "result dtype mismatch for {expr:?}"
        );
        // Exact array equality (bit-for-bit for f64, since identical ops).
        assert_eq!(&jit, &oracle, "value mismatch for {expr:?}");
    }

    #[test]
    fn nan_float_comparisons_match_interpreter() {
        // Total ordering on NaN (NaN == NaN, NaN > every non-NaN) is the
        // interpreter's contract; the JIT must agree, not fall back to IEEE. The
        // fuzzer deliberately never produces NaN, so this pins the NaN cases.
        use arrow::datatypes::{Field, Schema};
        let nan = f64::NAN;
        let a = vec![1.0, nan, 3.0, nan];
        let b = vec![2.0, 2.0, nan, nan];
        let schema = Schema::new(vec![
            Field::new("a", DataType::Float64, false),
            Field::new("b", DataType::Float64, false),
        ]);
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(Float64Array::from(a)),
                Arc::new(Float64Array::from(b)),
            ],
        )
        .unwrap();
        for op in [
            BinaryOp::Eq,
            BinaryOp::Ne,
            BinaryOp::Lt,
            BinaryOp::Le,
            BinaryOp::Gt,
            BinaryOp::Ge,
        ] {
            let expr = bin(op, col("a"), col("b"));
            // assert_parity already checks the JIT equals the interpreter oracle.
            assert_parity(&expr, &batch);
        }
    }

    #[test]
    fn jit_handles_more_than_four_columns() {
        // The old fixed-arity ABI capped at 4 distinct columns; the pointer-array
        // ABI lifts that. A 6-column expression must compile and match the oracle.
        let n = 64;
        let mut rng = Rng(0xC0FFEE);
        let names = ["a", "b", "c", "d", "e", "f"];
        let fields: Vec<Field> = names
            .iter()
            .map(|nm| Field::new(*nm, DataType::Int64, false))
            .collect();
        let columns: Vec<ArrayRef> = names
            .iter()
            .map(|_| {
                Arc::new(Int64Array::from(
                    (0..n).map(|_| rng.i64_small()).collect::<Vec<_>>(),
                )) as ArrayRef
            })
            .collect();
        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap();
        // a + b + c + d + e + f  (references all six columns)
        let sum = names
            .iter()
            .map(|nm| col(nm))
            .reduce(|acc, c| bin(BinaryOp::Add, acc, c))
            .unwrap();
        let compiled = compile_expr(&sum, &batch).expect("compiles with 6 columns");
        assert_eq!(compiled.columns.len(), 6);
        assert_parity(&sum, &batch);
    }

    #[test]
    fn jit_value_math_on_nullable_columns_matches_oracle() {
        // Value-only math / cast now propagate nulls in the JIT (run on garbage,
        // mask the output). The result must equal the interpreter, nulls included.
        let schema = Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("i", DataType::Int64, true),
        ]);
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(Float64Array::from(vec![Some(4.0), None, Some(9.0), None])),
                Arc::new(Int64Array::from(vec![Some(3), Some(-5), None, Some(7)])),
            ],
        )
        .unwrap();
        let math = |func, e| Expr::Math {
            func,
            input: Box::new(e),
        };
        // sqrt(x) + cast(i as float), both operands nullable → null-propagating.
        let expr = bin(
            BinaryOp::Add,
            math(bc_expr::MathFunc::Sqrt, col("x")),
            Expr::Cast {
                input: Box::new(col("i")),
                dtype: "float64".into(),
                try_cast: false,
            },
        );
        assert!(
            is_null_propagating(&expr),
            "expr should be null-propagating"
        );
        assert_parity(&expr, &batch);
    }

    #[test]
    fn differential_integer_overflow_wraps_identically() {
        // i64 overflow must WRAP identically in both tiers. The JIT's `iadd/isub/imul`
        // wrap; the interpreter uses `*_wrapping` to match (a checked kernel would
        // error here and diverge). Pre-fix, the interpreter eval would error.
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Int64, false),
        ]);
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(Int64Array::from(vec![i64::MAX, i64::MIN, 100])),
                Arc::new(Int64Array::from(vec![1, 1, 200])),
            ],
        )
        .unwrap();
        assert_parity(&bin(BinaryOp::Add, col("a"), col("b")), &batch);
        assert_parity(&bin(BinaryOp::Sub, col("a"), col("b")), &batch);
        assert_parity(&bin(BinaryOp::Mul, col("a"), col("b")), &batch);
    }

    #[test]
    fn differential_arithmetic_and_compare() {
        let batch = make_batch(257, 0x1234_5678);
        let exprs = vec![
            // a + b
            bin(BinaryOp::Add, col("a"), col("b")),
            // a * 2 + b
            bin(
                BinaryOp::Add,
                bin(BinaryOp::Mul, col("a"), lit_i(2)),
                col("b"),
            ),
            // (a - b) * c   (mixed int/float -> float)
            bin(
                BinaryOp::Mul,
                bin(BinaryOp::Sub, col("a"), col("b")),
                col("c"),
            ),
            // a > b
            bin(BinaryOp::Gt, col("a"), col("b")),
            // a * 1.5 <= c
            bin(
                BinaryOp::Le,
                bin(BinaryOp::Mul, col("a"), lit_f(1.5)),
                col("c"),
            ),
            // c / a  (float / int -> float)
            bin(BinaryOp::Div, col("c"), col("a")),
            // c % 2.0 (float remainder)
            bin(BinaryOp::Mod, col("c"), lit_f(2.0)),
            // a == b
            bin(BinaryOp::Eq, col("a"), col("b")),
            // a != b
            bin(BinaryOp::Ne, col("a"), col("b")),
            // a >= b
            bin(BinaryOp::Ge, col("a"), col("b")),
            // c < a
            bin(BinaryOp::Lt, col("c"), col("a")),
            // plain column
            col("a"),
            col("c"),
        ];
        for e in &exprs {
            assert_parity(e, &batch);
        }
    }

    #[test]
    fn differential_boolean_ops() {
        let batch = make_batch(257, 0x0B00_1234);
        let gt_ab = || bin(BinaryOp::Gt, col("a"), col("b"));
        let lt_ca = || bin(BinaryOp::Lt, col("c"), col("a"));
        let exprs = vec![
            // (a > b) AND (c < a)
            bin(BinaryOp::And, gt_ab(), lt_ca()),
            // (a > b) OR (a == b)
            bin(BinaryOp::Or, gt_ab(), bin(BinaryOp::Eq, col("a"), col("b"))),
            // NOT (a > b)
            Expr::Not {
                input: Box::new(gt_ab()),
            },
            // NOT ((a > b) AND (c < a))  — De Morgan-ish nesting
            Expr::Not {
                input: Box::new(bin(BinaryOp::And, gt_ab(), lt_ca())),
            },
            // ((a > 1) AND (b > 1)) OR (c < 0.0)  — mixed compound predicate
            bin(
                BinaryOp::Or,
                bin(
                    BinaryOp::And,
                    bin(BinaryOp::Gt, col("a"), lit_i(1)),
                    bin(BinaryOp::Gt, col("b"), lit_i(1)),
                ),
                bin(BinaryOp::Lt, col("c"), lit_f(0.0)),
            ),
        ];
        for e in &exprs {
            assert_parity(e, &batch);
        }
    }

    #[test]
    fn differential_math() {
        use bc_expr::MathFunc;
        let batch = make_batch(257, 0x0F00_D00D);
        let math = |func: MathFunc, input: Expr| Expr::Math {
            func,
            input: Box::new(input),
        };
        let exprs = vec![
            // sqrt over int and float columns (int promotes to f64 -> Float64).
            math(MathFunc::Sqrt, col("a")),
            math(MathFunc::Sqrt, col("c")),
            // floor/ceil/trunc over float (and int, which promotes to f64).
            math(MathFunc::Floor, col("c")),
            math(MathFunc::Ceil, col("c")),
            math(MathFunc::Trunc, col("c")),
            math(MathFunc::Floor, col("a")),
            // abs: float -> Float64, int -> Int64 (type preserved).
            math(MathFunc::Abs, col("c")),
            math(MathFunc::Abs, col("a")),
            // nested: sqrt(abs(c)) and floor((a - b) * c).
            math(MathFunc::Sqrt, math(MathFunc::Abs, col("c"))),
            math(
                MathFunc::Floor,
                bin(
                    BinaryOp::Mul,
                    bin(BinaryOp::Sub, col("a"), col("b")),
                    col("c"),
                ),
            ),
        ];
        for e in &exprs {
            assert_parity(e, &batch);
        }

        // Math funcs that stay on the interpreter (different rounding mode /
        // select / constant-multiply / reciprocal — out of scope for the JIT)
        // must NOT compile so the interpreter handles them and parity is
        // preserved. (Ln/Sin/Cbrt etc. ARE now supported — see
        // `differential_transcendental`.)
        for func in [
            MathFunc::Round,
            MathFunc::Sign,
            MathFunc::Degrees,
            MathFunc::Radians,
            MathFunc::Cot,
        ] {
            assert!(matches!(
                compile_and_eval(&math(func, col("c")), &batch),
                Err(CodegenError::Unsupported(_))
            ));
        }
    }

    /// Build a batch with domain-safe columns for the transcendentals:
    /// `p` strictly positive (ln/log/sqrt domain), `u` in [-1, 1] (asin/acos),
    /// and `q` a small positive base (pow). Deterministic.
    fn make_math_batch(n: usize, seed: u64) -> RecordBatch {
        let mut rng = Rng(seed);
        // p: positive in (0, ~286] (|i64_small| up to 1000, /3.5 + small offset).
        let p: Vec<f64> = (0..n)
            .map(|_| (rng.i64_small().unsigned_abs() as f64) / 3.5 + 0.1)
            .collect();
        // u: in [-1, 1] (i64_small / 1000 lands in [-1, 1]).
        let u: Vec<f64> = (0..n).map(|_| rng.i64_small() as f64 / 1000.0).collect();
        // q: small positive base in (0, ~6] for pow.
        let q: Vec<f64> = (0..n)
            .map(|_| (rng.i64_small().unsigned_abs() % 600) as f64 / 100.0 + 0.05)
            .collect();
        let schema = Schema::new(vec![
            Field::new("p", DataType::Float64, false),
            Field::new("u", DataType::Float64, false),
            Field::new("q", DataType::Float64, false),
        ]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(Float64Array::from(p)),
                Arc::new(Float64Array::from(u)),
                Arc::new(Float64Array::from(q)),
            ],
        )
        .unwrap()
    }

    /// Differential parity for every transcendental / two-arg math function the
    /// JIT now lowers via a libm libcall. Domains are chosen so neither tier
    /// produces a NaN (which would fail `assert_eq!` regardless of parity); the
    /// equality is exact (bit-for-bit), so a single-ULP divergence FAILS here.
    #[test]
    fn differential_transcendental() {
        use bc_expr::{Math2Func, MathFunc};
        let batch = make_math_batch(257, 0x7A45_C0DE);
        let math = |func: MathFunc, input: Expr| Expr::Math {
            func,
            input: Box::new(input),
        };
        let math2 = |func: Math2Func, l: Expr, r: Expr| Expr::Math2 {
            func,
            left: Box::new(l),
            right: Box::new(r),
        };

        // Domain-unrestricted single-arg funcs apply to any finite input; run
        // them over `p` (positive) and `u` ([-1,1]) to cover both sides.
        let any_domain = [
            MathFunc::Exp,
            MathFunc::Sin,
            MathFunc::Cos,
            MathFunc::Atan,
            MathFunc::Sinh,
            MathFunc::Cosh,
            MathFunc::Tanh,
            MathFunc::Cbrt,
        ];
        for f in any_domain {
            assert_parity(&math(f, col("p")), &batch);
            assert_parity(&math(f, col("u")), &batch);
        }

        // tan is finite for these inputs (no value near pi/2 + k*pi).
        assert_parity(&math(MathFunc::Tan, col("u")), &batch);

        // log-family require a positive argument: use `p`.
        for f in [MathFunc::Ln, MathFunc::Log10, MathFunc::Log2] {
            assert_parity(&math(f, col("p")), &batch);
        }

        // asin/acos require the argument in [-1, 1]: use `u`.
        for f in [MathFunc::Asin, MathFunc::Acos] {
            assert_parity(&math(f, col("u")), &batch);
        }

        // Int input is promoted to f64 before the call (interpreter casts too).
        // `q` is float, but verify the promotion path with an int literal arg.
        assert_parity(&math(MathFunc::Exp, lit_i(2)), &batch);

        // Two-arg: pow over a modest positive base, atan2 over arbitrary finite
        // operands.
        assert_parity(&math2(Math2Func::Pow, col("q"), col("u")), &batch);
        assert_parity(&math2(Math2Func::Pow, col("q"), col("p")), &batch);
        assert_parity(&math2(Math2Func::Atan2, col("u"), col("p")), &batch);
        assert_parity(&math2(Math2Func::Atan2, col("p"), col("q")), &batch);

        // Nested: ln(exp(p)) and pow(q, sin(u)) — exercise composition through
        // the libcalls.
        assert_parity(&math(MathFunc::Ln, math(MathFunc::Exp, col("p"))), &batch);
        assert_parity(
            &math2(Math2Func::Pow, col("q"), math(MathFunc::Sin, col("u"))),
            &batch,
        );

        // Math2::Round (round(x, digits)) is NOT a single libm call -> must stay
        // unsupported so the interpreter handles it.
        assert!(matches!(
            compile_and_eval(&math2(Math2Func::Round, col("p"), lit_i(2)), &batch),
            Err(CodegenError::Unsupported(_))
        ));
    }

    #[test]
    fn differential_cast() {
        let batch = make_batch(257, 0xCA57_0001);
        let cast = |input: Expr, dtype: &str| Expr::Cast {
            input: Box::new(input),
            dtype: dtype.into(),
            try_cast: false,
        };
        let exprs = vec![
            // int64 col -> float64 (exact convert).
            cast(col("a"), "float64"),
            // float64 col -> float64 (no-op).
            cast(col("c"), "float64"),
            // int64 col -> int64 (no-op).
            cast(col("a"), "int64"),
            // compound: cast(a -> float64) + c  (mixed convert + float arith).
            bin(BinaryOp::Add, cast(col("a"), "float64"), col("c")),
            // `double` is the other name for float64 in parse_dtype.
            cast(col("a"), "double"),
            // `long` is the other name for int64 in parse_dtype.
            cast(col("a"), "long"),
        ];
        for e in &exprs {
            assert_parity(e, &batch);
        }

        // float64 -> int64 has subtle Arrow rounding/saturation semantics that
        // could mismatch `fcvt`, so it must NOT compile (interpreter handles it).
        assert!(matches!(
            compile_and_eval(&cast(col("c"), "int64"), &batch),
            Err(CodegenError::Unsupported(_))
        ));
        // A non-numeric / unsupported target dtype also stays unsupported.
        assert!(matches!(
            compile_and_eval(&cast(col("a"), "int32"), &batch),
            Err(CodegenError::Unsupported(_))
        ));
    }

    #[test]
    fn differential_case() {
        use bc_expr::CaseBranch;
        let batch = make_batch(257, 0xCA5E_0001);

        let case = |branches: Vec<CaseBranch>, otherwise: Expr| Expr::Case {
            branches,
            otherwise: Box::new(otherwise),
        };
        let branch = |when: Expr, then: Expr| CaseBranch { when, then };
        let cast_f = |e: Expr| Expr::Cast {
            input: Box::new(e),
            dtype: "float64".into(),
            try_cast: false,
        };

        let exprs = vec![
            // CASE WHEN a > 0 THEN a ELSE 0 - a END   (== abs(a), all i64)
            case(
                vec![branch(bin(BinaryOp::Gt, col("a"), lit_i(0)), col("a"))],
                bin(BinaryOp::Sub, lit_i(0), col("a")),
            ),
            // CASE WHEN c > 0.5 THEN c ELSE c * 2.0 END   (all f64)
            case(
                vec![branch(bin(BinaryOp::Gt, col("c"), lit_f(0.5)), col("c"))],
                bin(BinaryOp::Mul, col("c"), lit_f(2.0)),
            ),
            // 2-branch, first-WHEN-wins ordering matters:
            // CASE WHEN a > b THEN a WHEN a > 0 THEN b ELSE 0 END
            case(
                vec![
                    branch(bin(BinaryOp::Gt, col("a"), col("b")), col("a")),
                    branch(bin(BinaryOp::Gt, col("a"), lit_i(0)), col("b")),
                ],
                lit_i(0),
            ),
            // Mixed type: THEN is int-valued, ELSE float -> Case promotes to f64.
            // The interpreter's `zip` requires matched types, so the int side is
            // an explicit cast to float64 (exactly what the SQL frontend emits);
            // the JIT's `promote_to` then sees an already-f64 then-branch and an
            // f64 else, and the whole Case is f64.
            // CASE WHEN a > b THEN cast(a AS float64) ELSE c END
            case(
                vec![branch(
                    bin(BinaryOp::Gt, col("a"), col("b")),
                    cast_f(col("a")),
                )],
                col("c"),
            ),
            // Mixed THEN types across branches, all aligned to f64 for the
            // interpreter, exercising the f64 result-type fold.
            // CASE WHEN a > 0 THEN cast(a) WHEN c > 0.0 THEN c ELSE c - 1.0 END
            case(
                vec![
                    branch(bin(BinaryOp::Gt, col("a"), lit_i(0)), cast_f(col("a"))),
                    branch(bin(BinaryOp::Gt, col("c"), lit_f(0.0)), col("c")),
                ],
                bin(BinaryOp::Sub, col("c"), lit_f(1.0)),
            ),
            // Nested compound then/else (still numeric, total).
            // CASE WHEN a >= b THEN a + b ELSE a - b END
            case(
                vec![branch(
                    bin(BinaryOp::Ge, col("a"), col("b")),
                    bin(BinaryOp::Add, col("a"), col("b")),
                )],
                bin(BinaryOp::Sub, col("a"), col("b")),
            ),
        ];
        for e in &exprs {
            assert_parity(e, &batch);
        }

        // A boolean THEN/ELSE is outside the numeric subset -> must not compile.
        let bool_case = case(
            vec![CaseBranch {
                when: bin(BinaryOp::Gt, col("a"), lit_i(0)),
                then: bin(BinaryOp::Gt, col("a"), col("b")),
            }],
            bin(BinaryOp::Lt, col("a"), col("b")),
        );
        assert!(matches!(
            compile_and_eval(&bool_case, &batch),
            Err(CodegenError::Unsupported(_))
        ));
    }

    #[test]
    fn differential_many_seeds() {
        for seed in 1..20u64 {
            let batch = make_batch(64, seed.wrapping_mul(0x9E37_79B9));
            // (a - b) * c + a
            let e = bin(
                BinaryOp::Add,
                bin(
                    BinaryOp::Mul,
                    bin(BinaryOp::Sub, col("a"), col("b")),
                    col("c"),
                ),
                col("a"),
            );
            assert_parity(&e, &batch);
        }
    }

    #[test]
    fn empty_batch() {
        let batch = make_batch(0, 7);
        assert_parity(&bin(BinaryOp::Add, col("a"), col("b")), &batch);
        assert_parity(&bin(BinaryOp::Gt, col("a"), col("c")), &batch);
    }

    #[test]
    fn integer_division_by_nonconstant_is_unsupported() {
        // Integer div/rem by a *column* (possibly zero) would trap on a zero
        // divisor (cranelift `sdiv`/`srem`), so they must NOT compile — the
        // interpreter (which guards zero) handles them. Float div/rem stays
        // compilable. (Div/rem by a safe constant DOES compile — see below.)
        let batch = make_batch(8, 3);
        assert!(matches!(
            compile_and_eval(&bin(BinaryOp::Div, col("a"), col("b")), &batch),
            Err(CodegenError::Unsupported(_))
        ));
        assert!(matches!(
            compile_and_eval(&bin(BinaryOp::Mod, col("a"), col("b")), &batch),
            Err(CodegenError::Unsupported(_))
        ));
        // c is f64, so c / a promotes to float and DOES compile.
        assert_parity(&bin(BinaryOp::Div, col("c"), col("a")), &batch);
        // The unsafe constant divisors (0, -1) must also stay on the interpreter:
        // 0 would div-by-zero, -1 would overflow `i64::MIN / -1`.
        assert!(matches!(
            compile_and_eval(&bin(BinaryOp::Div, col("a"), lit_i(0)), &batch),
            Err(CodegenError::Unsupported(_))
        ));
        assert!(matches!(
            compile_and_eval(&bin(BinaryOp::Mod, col("a"), lit_i(-1)), &batch),
            Err(CodegenError::Unsupported(_))
        ));
    }

    #[test]
    fn differential_integer_division_by_constant() {
        // `x / k` and `x % k` for a constant k ∉ {0, -1} compile and match the
        // interpreter bit-for-bit (truncate toward zero; remainder takes the
        // dividend's sign). Covers positive and negative dividends (make_batch
        // produces a, b in [-1000, 1000]) and a negative constant divisor.
        let batch = make_batch(257, 0xD150_0001);
        let exprs = vec![
            bin(BinaryOp::Div, col("a"), lit_i(2)),   // a / 2
            bin(BinaryOp::Mod, col("a"), lit_i(10)),  // a % 10  (bucketing)
            bin(BinaryOp::Div, col("a"), lit_i(100)), // a / 100
            bin(BinaryOp::Mod, col("b"), lit_i(7)),   // b % 7
            bin(BinaryOp::Div, col("a"), lit_i(-3)),  // negative divisor
            bin(BinaryOp::Mod, col("a"), lit_i(-4)),  // negative divisor, rem
            // nested: (a + b) % 8  — compound dividend, constant divisor.
            bin(
                BinaryOp::Mod,
                bin(BinaryOp::Add, col("a"), col("b")),
                lit_i(8),
            ),
        ];
        for e in &exprs {
            assert_parity(e, &batch);
        }
    }

    #[test]
    fn unsupported_string_column() {
        let schema = Schema::new(vec![Field::new("s", DataType::Utf8, false)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(StringArray::from(vec!["x", "y"]))],
        )
        .unwrap();
        let e = bin(BinaryOp::Eq, col("s"), col("s"));
        assert!(matches!(
            compile_and_eval(&e, &batch),
            Err(CodegenError::Unsupported(_))
        ));
    }

    fn nullable_batch() -> RecordBatch {
        // a, b nullable Int64; c nullable Float64 — nulls scattered across rows.
        let a = Int64Array::from(vec![Some(5), None, Some(3), Some(-2), None, Some(8)]);
        let b = Int64Array::from(vec![Some(1), Some(2), None, Some(4), Some(5), None]);
        let c = Float64Array::from(vec![Some(1.5), Some(2.0), None, None, Some(0.5), Some(9.0)]);
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("b", DataType::Int64, true),
            Field::new("c", DataType::Float64, true),
        ]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(a), Arc::new(b), Arc::new(c)],
        )
        .unwrap()
    }

    #[test]
    fn nullable_null_propagating_parity() {
        let batch = nullable_batch();
        // Each expr is in the null-propagating subset → JIT computes values and
        // masks; the result must match the interpreter bit-for-bit (nulls included).
        let cases = [
            bin(BinaryOp::Add, col("a"), col("b")), // int + int, both nullable
            bin(BinaryOp::Mul, col("a"), lit_i(2)), // int * literal
            bin(BinaryOp::Sub, col("a"), col("b")), // int - int
            bin(BinaryOp::Gt, col("a"), lit_i(0)),  // nullable comparison
            bin(BinaryOp::Add, col("c"), lit_f(1.0)), // float + literal
            bin(
                BinaryOp::Lt,
                bin(BinaryOp::Sub, col("a"), col("b")),
                col("c"),
            ), // (a-b) < c — three nullable cols combine
        ];
        for e in &cases {
            assert_parity(e, &batch);
        }
    }

    #[test]
    fn nullable_div_mod_not_propagating_parity() {
        // Div/Mod by a safe constant and logical NOT propagate nulls exactly (null in
        // → null out, no trap at masked slots), so they now run on the JIT over
        // nullable input instead of falling back. Pin both the classification and
        // bit-for-bit parity with the interpreter.
        let batch = nullable_batch();
        let not_gt = Expr::Not {
            input: Box::new(bin(BinaryOp::Gt, col("a"), lit_i(0))),
        };
        let cases = [
            bin(BinaryOp::Div, col("a"), lit_i(2)), // int / constant — bucketing
            bin(BinaryOp::Mod, col("a"), lit_i(3)), // int % constant
            bin(BinaryOp::Div, col("c"), lit_f(2.0)), // float / literal (IEEE)
            not_gt,                                 // NOT (a > 0) over a nullable col
        ];
        for e in &cases {
            assert!(
                is_null_propagating(e),
                "expected null-propagating classification for {e:?}"
            );
            // assert_parity drives the JIT eval (it errors out on fallback), so a
            // pass proves the JIT — not the interpreter — produced the masked result.
            assert_parity(e, &batch);
        }
    }

    #[test]
    fn nullable_all_valid_has_no_nulls() {
        // A null-propagating expr over a *null-free* batch must still match exactly
        // and carry no validity bitmap (the fast path).
        let batch = make_batch(64, 0xF00D);
        assert_parity(&bin(BinaryOp::Add, col("a"), col("b")), &batch);
    }

    #[test]
    fn nullable_and_or_kleene_parity() {
        // Boolean AND/OR over nullable columns now compile in the Kleene value+
        // validity ABI and run on the JIT with correct three-valued logic
        // (false AND null = false, true OR null = true), matching the interpreter
        // bit-for-bit instead of falling back. `assert_parity` drives the JIT eval,
        // so a pass proves the Kleene path — not the interpreter — produced it.
        let batch = nullable_batch();
        let and = |l, r| bin(BinaryOp::And, l, r);
        let or = |l, r| bin(BinaryOp::Or, l, r);
        let p1 = || {
            and(
                bin(BinaryOp::Gt, col("a"), lit_i(0)),
                bin(BinaryOp::Lt, col("b"), lit_i(5)),
            )
        };
        let cases = [
            p1(),
            or(
                bin(BinaryOp::Gt, col("a"), lit_i(0)),
                bin(BinaryOp::Lt, col("b"), lit_i(5)),
            ),
            // nested mix across three nullable columns
            or(p1(), bin(BinaryOp::Gt, col("c"), lit_f(1.0))),
            // NOT over a Kleene AND (NOT null = null)
            Expr::Not {
                input: Box::new(p1()),
            },
        ];
        for e in &cases {
            assert!(needs_kleene(e), "expected Kleene compile for {e:?}");
            assert_parity(e, &batch);
        }
    }

    #[test]
    fn nullable_case_still_falls_back() {
        // CASE's result value depends on which branch is selected, so it is not
        // Kleene-supported and not null-propagating; over a nullable column it still
        // defers to the interpreter for correct branch-null semantics.
        use bc_expr::CaseBranch;
        let schema = Schema::new(vec![Field::new("a", DataType::Int64, true)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(Int64Array::from(vec![Some(1), None, Some(3)]))],
        )
        .unwrap();
        let case = Expr::Case {
            branches: vec![CaseBranch {
                when: bin(BinaryOp::Gt, col("a"), lit_i(0)),
                then: col("a"),
            }],
            otherwise: Box::new(lit_i(0)),
        };
        let compiled = compile_expr(&case, &batch).expect("compiles on the sample shape");
        assert!(matches!(
            compiled.eval(&batch),
            Err(CodegenError::Unsupported(_))
        ));
    }

    #[test]
    fn unsupported_bool_literal_and_ops() {
        let batch = make_batch(8, 3);
        let blit = Expr::Lit {
            value: Literal::Bool(true),
        };
        assert!(matches!(
            compile_and_eval(&blit, &batch),
            Err(CodegenError::Unsupported(_))
        ));
        // AND/OR over *non-boolean* operands stays unsupported (the JIT has no
        // truthiness for numbers). AND/OR over booleans IS supported — see
        // `differential_boolean_ops`.
        let e = bin(BinaryOp::And, col("a"), col("b"));
        assert!(matches!(
            compile_and_eval(&e, &batch),
            Err(CodegenError::Unsupported(_))
        ));
    }

    /// Benchmark the engine's core thesis — a fused, compile-once native JIT
    /// kernel beats the array-at-a-time interpreter — across a spread of
    /// representative compound expressions. Ignored by default (timing, not
    /// correctness). Run with:
    /// `cargo test -p bc-codegen --release -- --ignored bench_jit_vs_interpreter --nocapture`.
    ///
    /// For each expression we compile **once** with `compile_expr` (so the
    /// reported JIT time is steady-state per-batch eval, not compilation) and
    /// time `iters` evals of the compiled kernel against `iters` evals of the
    /// interpreter. We also assert bit-for-bit parity inside the bench so it can
    /// never silently measure wrong code.
    #[test]
    #[ignore]
    fn bench_jit_vs_interpreter() {
        use bc_expr::{CaseBranch, MathFunc};
        use std::time::Instant;

        fn math(func: MathFunc, input: Expr) -> Expr {
            Expr::Math {
                func,
                input: Box::new(input),
            }
        }

        let n = 1_000_000;
        let iters = 50;
        let batch = make_batch(n, 0xDEAD_BEEF);

        // A spread of representative expressions of increasing complexity.
        // Each is total over `make_batch` data (no div/mod, sqrt/ln only over
        // `abs(..)` so no NaN) — keeping JIT and interpreter bit-for-bit equal.
        let cases: Vec<(&str, Expr)> = vec![
            // simple: (a - b) * c
            (
                "(a - b) * c",
                bin(
                    BinaryOp::Mul,
                    bin(BinaryOp::Sub, col("a"), col("b")),
                    col("c"),
                ),
            ),
            // compound arithmetic: (a - b) * c + a * 2.0
            (
                "(a - b) * c + a * 2.0",
                bin(
                    BinaryOp::Add,
                    bin(
                        BinaryOp::Mul,
                        bin(BinaryOp::Sub, col("a"), col("b")),
                        col("c"),
                    ),
                    bin(BinaryOp::Mul, col("a"), lit_f(2.0)),
                ),
            ),
            // boolean predicate: (a > b) AND (c < a)
            (
                "(a > b) AND (c < a)",
                bin(
                    BinaryOp::And,
                    bin(BinaryOp::Gt, col("a"), col("b")),
                    bin(BinaryOp::Lt, col("c"), col("a")),
                ),
            ),
            // transcendental math: sqrt(abs(c)) + ln(abs(a) + 1.0)
            (
                "sqrt(abs(c)) + ln(abs(a) + 1.0)",
                bin(
                    BinaryOp::Add,
                    math(MathFunc::Sqrt, math(MathFunc::Abs, col("c"))),
                    math(
                        MathFunc::Ln,
                        bin(BinaryOp::Add, math(MathFunc::Abs, col("a")), lit_f(1.0)),
                    ),
                ),
            ),
            // CASE: CASE WHEN a > 0 THEN a*c ELSE 0-a END  (then/else both F64)
            (
                "CASE WHEN a > 0 THEN a*c ELSE 0-a END",
                Expr::Case {
                    branches: vec![CaseBranch {
                        when: bin(BinaryOp::Gt, col("a"), lit_i(0)),
                        then: bin(BinaryOp::Mul, col("a"), col("c")),
                    }],
                    otherwise: Box::new(bin(BinaryOp::Sub, lit_f(0.0), col("a"))),
                },
            ),
        ];

        println!(
            "\nJIT vs interpreter (n={n} rows, {iters} iters/expr, compile-once JIT)\n\
             {:<40} {:>14} {:>14} {:>9}",
            "expression", "interp ns/iter", "jit ns/iter", "speedup"
        );
        println!("{}", "-".repeat(80));

        for (label, expr) in &cases {
            // Compile once; this is the steady-state kernel we re-run per batch.
            let compiled = compile_expr(expr, &batch).expect("should compile");

            // Correctness: the compiled kernel must match the interpreter oracle
            // bit-for-bit, else the timing below is meaningless.
            let jit_out = compiled.eval(&batch).expect("jit eval");
            let interp_out = expr.eval(&batch).expect("interp eval");
            assert_eq!(
                jit_out.data_type(),
                interp_out.data_type(),
                "result dtype mismatch for `{label}`"
            );
            assert_eq!(&jit_out, &interp_out, "value mismatch for `{label}`");

            // Warmup (already done above for both paths via the parity check).
            let t0 = Instant::now();
            for _ in 0..iters {
                std::hint::black_box(compiled.eval(&batch).unwrap());
            }
            let jit = t0.elapsed() / iters;

            let t1 = Instant::now();
            for _ in 0..iters {
                std::hint::black_box(expr.eval(&batch).unwrap());
            }
            let interp = t1.elapsed() / iters;

            let speedup = interp.as_secs_f64() / jit.as_secs_f64();
            println!(
                "{:<40} {:>14} {:>14} {:>8.2}x",
                label,
                interp.as_nanos(),
                jit.as_nanos(),
                speedup
            );
        }
        println!();
    }

    /// Randomized differential fuzz: generate thousands of random expression
    /// trees from the supported subset and assert the JIT is bit-for-bit equal
    /// to the interpreter on every one. Deterministic (fixed seed) for CI.
    ///
    /// The generator tracks each node's value *kind* (numeric vs boolean) so it
    /// only ever combines compatible nodes, keeping every tree inside the
    /// compilable subset. To preserve exact array equality (`assert_parity` uses
    /// `assert_eq!`, and `NaN != NaN`) it: never emits integer or float
    /// Div/Mod (so no `inf`/`NaN` from division), and only applies `sqrt` to
    /// `abs(..)` of its operand (so no `NaN` from a negative radicand). All
    /// other ops are total over the modest-magnitude `make_batch` data.
    #[test]
    fn differential_fuzz() {
        use bc_expr::MathFunc;

        /// Value kind of a generated node, so we only combine compatible nodes.
        #[derive(Clone, Copy, PartialEq)]
        enum Kind {
            Num,
            Bool,
        }

        fn math(func: MathFunc, input: Expr) -> Expr {
            Expr::Math {
                func,
                input: Box::new(input),
            }
        }

        /// Generate a numeric-valued expression of depth at most `depth`.
        fn gen_num(rng: &mut Rng, depth: u32) -> Expr {
            // Leaf: a column or a literal.
            if depth == 0 || rng.next_u64() % 3 == 0 {
                return match rng.next_u64() % 5 {
                    0 => col("a"),
                    1 => col("b"),
                    2 => col("c"),
                    3 => lit_i(rng.i64_small()),
                    _ => lit_f(rng.f64_small()),
                };
            }
            match rng.next_u64() % 5 {
                // Arithmetic over two numeric children (Add/Sub/Mul only — no
                // Div/Mod, to avoid div-by-zero traps and inf/NaN results).
                0..=2 => {
                    let op = match rng.next_u64() % 3 {
                        0 => BinaryOp::Add,
                        1 => BinaryOp::Sub,
                        _ => BinaryOp::Mul,
                    };
                    bin(op, gen_num(rng, depth - 1), gen_num(rng, depth - 1))
                }
                // Unary math wrapper. `sqrt`/`ln`/`log10`/`log2` are only ever
                // applied to `abs(..) + 1` so the argument is strictly positive
                // and the result is never NaN. The remaining transcendentals are
                // finite for all finite inputs. asin/acos are kept OUT of the
                // fuzzer (their domain is [-1, 1], which arbitrary subtrees
                // violate, producing NaN that would fail `assert_eq!`).
                3 => {
                    let which = rng.next_u64() % 11;
                    let inner = gen_num(rng, depth - 1);
                    // `abs(inner) + 1` -> strictly positive, for domain-restricted
                    // funcs (sqrt/ln/logN); the +1 keeps the argument > 0.
                    let pos = bin(
                        BinaryOp::Add,
                        math(MathFunc::Abs, inner.clone()),
                        lit_f(1.0),
                    );
                    match which {
                        0 => math(MathFunc::Floor, inner),
                        1 => math(MathFunc::Ceil, inner),
                        2 => math(MathFunc::Trunc, inner),
                        3 => math(MathFunc::Abs, inner),
                        4 => math(MathFunc::Sqrt, pos),
                        5 => math(MathFunc::Ln, pos),
                        6 => math(MathFunc::Log10, pos),
                        7 => math(MathFunc::Log2, pos),
                        8 => math(MathFunc::Atan, inner),
                        9 => math(MathFunc::Tanh, inner),
                        // exp can overflow to +inf for large inputs, which is
                        // still bit-identical (inf == inf), but feed it a bounded
                        // argument via atan (range (-pi/2, pi/2)) to stay finite.
                        _ => math(MathFunc::Exp, math(MathFunc::Atan, inner)),
                    }
                }
                _ => {
                    // abs of a numeric child (type-preserving).
                    math(MathFunc::Abs, gen_num(rng, depth - 1))
                }
            }
        }

        /// Generate a boolean-valued expression of depth at most `depth`.
        fn gen_bool(rng: &mut Rng, depth: u32) -> Expr {
            // Leaf (or forced at depth 0): a comparison of two numeric children.
            if depth == 0 || rng.next_u64() % 3 == 0 {
                let op = match rng.next_u64() % 6 {
                    0 => BinaryOp::Eq,
                    1 => BinaryOp::Ne,
                    2 => BinaryOp::Lt,
                    3 => BinaryOp::Le,
                    4 => BinaryOp::Gt,
                    _ => BinaryOp::Ge,
                };
                let d = depth.saturating_sub(1);
                return bin(op, gen_num(rng, d), gen_num(rng, d));
            }
            match rng.next_u64() % 4 {
                0 => Expr::Not {
                    input: Box::new(gen_bool(rng, depth - 1)),
                },
                1 => bin(
                    BinaryOp::And,
                    gen_bool(rng, depth - 1),
                    gen_bool(rng, depth - 1),
                ),
                2 => bin(
                    BinaryOp::Or,
                    gen_bool(rng, depth - 1),
                    gen_bool(rng, depth - 1),
                ),
                _ => {
                    let op = match rng.next_u64() % 6 {
                        0 => BinaryOp::Eq,
                        1 => BinaryOp::Ne,
                        2 => BinaryOp::Lt,
                        3 => BinaryOp::Le,
                        4 => BinaryOp::Gt,
                        _ => BinaryOp::Ge,
                    };
                    bin(op, gen_num(rng, depth - 1), gen_num(rng, depth - 1))
                }
            }
        }

        // Fixed master seed -> reproducible in CI. Each iteration derives a
        // fresh sub-seed for the tree (and reuses a batch, re-rolled a few
        // times to vary the data the trees run over).
        let mut master = Rng(0xC0FF_EE12_3456_789A);
        let batches = [
            make_batch(129, 0x1111_2222),
            make_batch(129, 0x3333_4444),
            make_batch(129, 0x5555_6666),
        ];

        const ITERS: usize = 2000;
        for it in 0..ITERS {
            let seed = master.next_u64();
            let mut rng = Rng(seed | 1); // never seed xorshift with 0
            let kind = if rng.next_u64() % 2 == 0 {
                Kind::Num
            } else {
                Kind::Bool
            };
            let expr = match kind {
                Kind::Num => gen_num(&mut rng, 4),
                Kind::Bool => gen_bool(&mut rng, 4),
            };
            let batch = &batches[it % batches.len()];
            assert_parity(&expr, batch);
        }
    }

    /// Differential fuzzer for the Kleene path: random boolean `And`/`Or`/`Not` trees
    /// of comparisons over **nullable** columns, asserting the JIT's value+validity
    /// output matches the interpreter's three-valued logic bit-for-bit. The null-free
    /// `differential_fuzz` validates the value bits; this validates the validity bits
    /// (`false AND null = false`, `true OR null = true`, `NOT null = null`, …).
    #[test]
    fn differential_fuzz_kleene_nullable() {
        fn make_nullable(n: usize, seed: u64) -> RecordBatch {
            let mut rng = Rng(seed);
            let mut a = Vec::with_capacity(n);
            let mut b = Vec::with_capacity(n);
            let mut c = Vec::with_capacity(n);
            for _ in 0..n {
                // ~1 in 4 values null, independently per column.
                a.push((rng.next_u64() % 4 != 0).then(|| rng.i64_small()));
                b.push((rng.next_u64() % 4 != 0).then(|| rng.i64_small()));
                c.push((rng.next_u64() % 4 != 0).then(|| rng.f64_small()));
            }
            let schema = Schema::new(vec![
                Field::new("a", DataType::Int64, true),
                Field::new("b", DataType::Int64, true),
                Field::new("c", DataType::Float64, true),
            ]);
            RecordBatch::try_new(
                Arc::new(schema),
                vec![
                    Arc::new(Int64Array::from(a)),
                    Arc::new(Int64Array::from(b)),
                    Arc::new(Float64Array::from(c)),
                ],
            )
            .unwrap()
        }
        fn gen_num(rng: &mut Rng, depth: u32) -> Expr {
            if depth == 0 || rng.next_u64() % 3 == 0 {
                return match rng.next_u64() % 5 {
                    0 => col("a"),
                    1 => col("b"),
                    2 => col("c"),
                    3 => lit_i(rng.i64_small()),
                    _ => lit_f(rng.f64_small()),
                };
            }
            let op = match rng.next_u64() % 3 {
                0 => BinaryOp::Add,
                1 => BinaryOp::Sub,
                _ => BinaryOp::Mul,
            };
            bin(op, gen_num(rng, depth - 1), gen_num(rng, depth - 1))
        }
        fn gen_bool(rng: &mut Rng, depth: u32) -> Expr {
            if depth == 0 || rng.next_u64() % 3 == 0 {
                let op = match rng.next_u64() % 6 {
                    0 => BinaryOp::Eq,
                    1 => BinaryOp::Ne,
                    2 => BinaryOp::Lt,
                    3 => BinaryOp::Le,
                    4 => BinaryOp::Gt,
                    _ => BinaryOp::Ge,
                };
                let d = depth.saturating_sub(1);
                return bin(op, gen_num(rng, d), gen_num(rng, d));
            }
            match rng.next_u64() % 3 {
                0 => Expr::Not {
                    input: Box::new(gen_bool(rng, depth - 1)),
                },
                1 => bin(
                    BinaryOp::And,
                    gen_bool(rng, depth - 1),
                    gen_bool(rng, depth - 1),
                ),
                _ => bin(
                    BinaryOp::Or,
                    gen_bool(rng, depth - 1),
                    gen_bool(rng, depth - 1),
                ),
            }
        }
        let mut master = Rng(0x5EED_1234_ABCD_0001);
        let batches = [
            make_nullable(131, 0xAAAA_0001),
            make_nullable(131, 0xBBBB_0002),
            make_nullable(131, 0xCCCC_0003),
        ];
        for it in 0..2000 {
            let mut rng = Rng(master.next_u64() | 1);
            let expr = gen_bool(&mut rng, 4);
            assert_parity(&expr, &batches[it % batches.len()]);
        }
    }

    #[test]
    fn simd_f64_arithmetic_parity_across_sizes() {
        use bc_expr::BinaryOp::{Add, Div, Mul, Sub};
        let exprs = [
            bin(Add, col("c"), lit_f(1.5)),
            bin(Mul, col("c"), col("c")),
            bin(Sub, bin(Mul, col("c"), lit_f(2.0)), col("c")),
            bin(Div, bin(Add, col("c"), lit_f(10.0)), lit_f(3.0)), // constant denominator
        ];
        // Confirm these actually take the vector path (not silently the scalar one).
        let sample = make_batch(8, 1);
        for e in &exprs {
            let mut cols = ColumnSet::default();
            analyze(e, &sample, &mut cols).unwrap();
            assert!(simd_supported(e, &cols), "expected SIMD path for {e:?}");
        }
        // Parity at even AND odd sizes — odd exercises the scalar remainder loop.
        for &n in &[1usize, 2, 3, 7, 8, 64, 129] {
            let batch = make_batch(n, 0x51AB ^ n as u64);
            for e in &exprs {
                assert_parity(e, &batch);
            }
        }
    }

    /// Pure-F64 arithmetic fuzzer over NULLABLE columns: exercises the SIMD value
    /// lanes, the scalar remainder (odd sizes), and the null-propagating mask
    /// (applied in `eval` exactly as on the scalar path). `+,-,*` only — no `/`, so
    /// no `0/0` NaN can make `assert_eq!` spuriously differ (inf still compares equal).
    #[test]
    fn simd_f64_fuzz_nullable() {
        fn make_c(n: usize, seed: u64) -> RecordBatch {
            let mut rng = Rng(seed);
            let c: Vec<Option<f64>> = (0..n)
                .map(|_| (rng.next_u64() % 4 != 0).then(|| rng.f64_small()))
                .collect();
            let schema = Schema::new(vec![Field::new("c", DataType::Float64, true)]);
            RecordBatch::try_new(Arc::new(schema), vec![Arc::new(Float64Array::from(c))]).unwrap()
        }
        fn gen(rng: &mut Rng, depth: u32) -> Expr {
            if depth == 0 || rng.next_u64() % 3 == 0 {
                return if rng.next_u64() % 2 == 0 {
                    col("c")
                } else {
                    lit_f(rng.f64_small())
                };
            }
            let op = match rng.next_u64() % 3 {
                0 => BinaryOp::Add,
                1 => BinaryOp::Sub,
                _ => BinaryOp::Mul,
            };
            bin(op, gen(rng, depth - 1), gen(rng, depth - 1))
        }
        let mut master = Rng(0xABCD_0123_4567_89AB);
        let batches = [make_c(129, 0x11), make_c(130, 0x22), make_c(63, 0x33)];
        for it in 0..1000 {
            let mut rng = Rng(master.next_u64() | 1);
            let e = gen(&mut rng, 4);
            assert_parity(&e, &batches[it % batches.len()]);
        }
    }

    /// Differential fuzzer for the newer JIT nodes: `Cast` and `Case`.
    ///
    /// These paths were added after the original `differential_fuzz` and were
    /// not exercised by it. This generator builds numeric trees that include
    /// `Cast` (always to `float64`, plus int-only `int64` no-ops) and `Case`
    /// nodes, then asserts the JIT result is bit-for-bit identical to the
    /// interpreter oracle.
    ///
    /// Type discipline (so the interpreter's Arrow `zip` accepts the Case, and
    /// so casts stay exact):
    ///   * A Case is generated in ONE fixed result type — either all-integer
    ///     (every then/otherwise is an integer-only subtree) or all-float
    ///     (every then/otherwise is wrapped in `Cast(_, "float64")`, so it is
    ///     uniformly f64). Mixing int/float then-branches is never produced.
    ///   * `Cast` only ever targets `float64` (int->float or float->float, both
    ///     exact), or `int64` applied to an already-integer subtree (a no-op).
    ///     Float->int is NEVER generated (unsupported / not bit-exact).
    ///
    /// All numeric children obey the same NaN/inf/trap-free rules as the base
    /// fuzzer (no integer div/mod; sqrt/ln over abs(..)+1; no asin/acos).
    #[test]
    fn differential_fuzz_extended() {
        use bc_expr::{CaseBranch, MathFunc};

        fn math(func: MathFunc, input: Expr) -> Expr {
            Expr::Math {
                func,
                input: Box::new(input),
            }
        }
        fn cast(input: Expr, dtype: &str) -> Expr {
            Expr::Cast {
                input: Box::new(input),
                dtype: dtype.into(),
                try_cast: false,
            }
        }
        fn cast_f(e: Expr) -> Expr {
            cast(e, "float64")
        }

        /// Integer-ONLY numeric subtree (result is always i64). Uses only int
        /// columns (`a`, `b`), int literals, Add/Sub/Mul, and Abs — none of
        /// which introduce a float, so the whole tree stays Int64-typed. This
        /// lets all-integer Case branches share one matched type.
        fn gen_int(rng: &mut Rng, depth: u32) -> Expr {
            if depth == 0 || rng.next_u64() % 3 == 0 {
                return match rng.next_u64() % 3 {
                    0 => col("a"),
                    1 => col("b"),
                    _ => lit_i(rng.i64_small()),
                };
            }
            match rng.next_u64() % 4 {
                0..=2 => {
                    let op = match rng.next_u64() % 3 {
                        0 => BinaryOp::Add,
                        1 => BinaryOp::Sub,
                        _ => BinaryOp::Mul,
                    };
                    bin(op, gen_int(rng, depth - 1), gen_int(rng, depth - 1))
                }
                _ => math(MathFunc::Abs, gen_int(rng, depth - 1)),
            }
        }

        /// General numeric subtree (int- or float-valued), trap/NaN/inf-free.
        /// Mirrors the base fuzzer's `gen_num` safeguards and additionally may
        /// emit `Cast`/`Case` nodes so those paths get nested exercise too.
        fn gen_num(rng: &mut Rng, depth: u32) -> Expr {
            if depth == 0 || rng.next_u64() % 3 == 0 {
                return match rng.next_u64() % 5 {
                    0 => col("a"),
                    1 => col("b"),
                    2 => col("c"),
                    3 => lit_i(rng.i64_small()),
                    _ => lit_f(rng.f64_small()),
                };
            }
            match rng.next_u64() % 7 {
                0 | 1 => {
                    let op = match rng.next_u64() % 3 {
                        0 => BinaryOp::Add,
                        1 => BinaryOp::Sub,
                        _ => BinaryOp::Mul,
                    };
                    bin(op, gen_num(rng, depth - 1), gen_num(rng, depth - 1))
                }
                2 => {
                    let which = rng.next_u64() % 11;
                    let inner = gen_num(rng, depth - 1);
                    let pos = bin(
                        BinaryOp::Add,
                        math(MathFunc::Abs, inner.clone()),
                        lit_f(1.0),
                    );
                    match which {
                        0 => math(MathFunc::Floor, inner),
                        1 => math(MathFunc::Ceil, inner),
                        2 => math(MathFunc::Trunc, inner),
                        3 => math(MathFunc::Abs, inner),
                        4 => math(MathFunc::Sqrt, pos),
                        5 => math(MathFunc::Ln, pos),
                        6 => math(MathFunc::Log10, pos),
                        7 => math(MathFunc::Log2, pos),
                        8 => math(MathFunc::Atan, inner),
                        9 => math(MathFunc::Tanh, inner),
                        _ => math(MathFunc::Exp, math(MathFunc::Atan, inner)),
                    }
                }
                3 => math(MathFunc::Abs, gen_num(rng, depth - 1)),
                // Cast to float64: exact whether the child is int or float.
                4 => cast_f(gen_num(rng, depth - 1)),
                // Cast to int64 over an integer-typed child: a no-op (exact).
                5 => cast(gen_int(rng, depth - 1), "int64"),
                // Nested Case producing a numeric value.
                _ => gen_case(rng, depth - 1),
            }
        }

        /// Generate a NUMERIC-valued `Case` of a single fixed result type.
        /// 1..=3 branches; each `when` is a boolean subtree (reusing the base
        /// `gen_bool`); then/otherwise are all the same numeric type.
        fn gen_case(rng: &mut Rng, depth: u32) -> Expr {
            let n_branches = 1 + (rng.next_u64() % 3) as usize; // 1..=3
            let float_case = rng.next_u64() % 2 == 0;
            // Per-branch then/otherwise generators, uniformly typed.
            let then_of = |rng: &mut Rng| {
                if float_case {
                    cast_f(gen_num(rng, depth))
                } else {
                    gen_int(rng, depth)
                }
            };
            let branches: Vec<CaseBranch> = (0..n_branches)
                .map(|_| CaseBranch {
                    when: gen_bool(rng, depth),
                    then: then_of(rng),
                })
                .collect();
            let otherwise = then_of(rng);
            Expr::Case {
                branches,
                otherwise: Box::new(otherwise),
            }
        }

        /// Boolean subtree generator (comparisons of numeric children + the
        /// And/Or/Not combinators). Identical in spirit to the base fuzzer's
        /// `gen_bool`; defined locally so the extended fuzzer is self-contained.
        fn gen_bool(rng: &mut Rng, depth: u32) -> Expr {
            if depth == 0 || rng.next_u64() % 3 == 0 {
                let op = match rng.next_u64() % 6 {
                    0 => BinaryOp::Eq,
                    1 => BinaryOp::Ne,
                    2 => BinaryOp::Lt,
                    3 => BinaryOp::Le,
                    4 => BinaryOp::Gt,
                    _ => BinaryOp::Ge,
                };
                let d = depth.saturating_sub(1);
                return bin(op, gen_num(rng, d), gen_num(rng, d));
            }
            match rng.next_u64() % 4 {
                0 => Expr::Not {
                    input: Box::new(gen_bool(rng, depth - 1)),
                },
                1 => bin(
                    BinaryOp::And,
                    gen_bool(rng, depth - 1),
                    gen_bool(rng, depth - 1),
                ),
                2 => bin(
                    BinaryOp::Or,
                    gen_bool(rng, depth - 1),
                    gen_bool(rng, depth - 1),
                ),
                _ => {
                    let op = match rng.next_u64() % 6 {
                        0 => BinaryOp::Eq,
                        1 => BinaryOp::Ne,
                        2 => BinaryOp::Lt,
                        3 => BinaryOp::Le,
                        4 => BinaryOp::Gt,
                        _ => BinaryOp::Ge,
                    };
                    bin(op, gen_num(rng, depth - 1), gen_num(rng, depth - 1))
                }
            }
        }

        let mut master = Rng(0xCA5E_CA57_F0F0_1234);
        let batches = [
            make_batch(129, 0xA1A1_B2B2),
            make_batch(129, 0xC3C3_D4D4),
            make_batch(129, 0xE5E5_F6F6),
        ];

        const ITERS: usize = 2000;
        for it in 0..ITERS {
            let seed = master.next_u64();
            let mut rng = Rng(seed | 1); // never seed xorshift with 0
                                         // Always root in a Cast or Case so every iteration exercises
                                         // at least one of the newer nodes; nested gen_num may add more.
            let expr = if rng.next_u64() % 2 == 0 {
                gen_case(&mut rng, 3)
            } else {
                // Root cast: float64 of any subtree, or an int64 no-op cast.
                if rng.next_u64() % 2 == 0 {
                    cast_f(gen_num(&mut rng, 3))
                } else {
                    cast(gen_int(&mut rng, 3), "int64")
                }
            };
            let batch = &batches[it % batches.len()];
            assert_parity(&expr, batch);
        }
    }

    #[test]
    fn unknown_column() {
        let batch = make_batch(4, 9);
        let e = col("nope");
        assert!(matches!(
            compile_and_eval(&e, &batch),
            Err(CodegenError::UnknownColumn(_))
        ));
    }
}
