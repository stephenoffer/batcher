//! Vector (SIMD) emitter for the JIT's vectorizable `Expr` subset.
//!
//! Produces a `lanes`-wide `F64xL` / `I64xL` value holding rows `[i, i+lanes)`, so a
//! loop stepping `i` by `lanes` (optionally `unroll` chains per iteration) computes
//! many rows at once. The width is chosen at compile time from the host
//! [`HardwareProfile`](bc_arrow::HardwareProfile) (2 on SSE2/NEON, 4 on AVX2, opt-in
//! 8 on AVX-512); Cranelift legalizes a wider IR vector to native instructions where
//! the ISA has them and splits it into 128-bit ops otherwise — either way the result
//! is identical, so a width that doesn't lower natively is at worst a no-op.
//!
//! The subset (see [`simd_ty`](crate::simd_ty)) is exactly the ops whose per-lane
//! result is bit-for-bit identical to the scalar [`Codegen`](crate::emit::Codegen):
//!
//! * `Col` (`I64`/`F64`) and `Lit` (`Int`/`Float`) leaves.
//! * Integer `Add`/`Sub`/`Mul` (two's-complement wrap is per-lane identical) and
//!   float `Add`/`Sub`/`Mul`/`Div` (IEEE per-lane identical).
//! * Comparisons (`Eq`/`Ne`/`Lt`/`Le`/`Gt`/`Ge`) over numeric operands — the big
//!   filter win — producing a boolean lane mask, with the same total-order NaN
//!   semantics the scalar path uses.
//! * `Not` of a boolean sub-result, and exact numeric `Cast` (`i64 -> f64`, or a
//!   no-op).
//!
//! Excluded (they stay on the scalar [`Codegen`] / interpreter): `And`/`Or` (the
//! Kleene validity ABI owns nullable compound predicates), integer `Div`/`Mod`
//! (scalarized `sdiv`/`srem`, can trap), float `Mod` (an `fmod` libcall), `Math`/
//! `Math2` (libm libcalls), `Case`, and temporal operands. A scalar remainder loop
//! handles the rows past the last full `lanes*unroll` step.
//!
//! # Boolean lanes
//!
//! A boolean sub-result is an `I64xL` **canonical mask** — all-ones for true,
//! all-zeros for false — the form `icmp`/`fcmp` produce. `Not` is `bnot` (flips a
//! canonical mask to the other canonical value); the only boolean sources are
//! comparisons and `Not`, so every boolean lane stays canonical. The mask is
//! converted to consecutive `0`/`1` bits in the Arrow bitmask only at the store site
//! (in `compile_simd`).

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{types, InstBuilder, MemFlags, Type, Value};
use cranelift_frontend::FunctionBuilder;

use crate::{ColumnSet, ScalarTy};

/// The `lanes`-wide Cranelift vector type for a scalar lane type. `lanes` is one of
/// 2/4/8 (validated by the profile); a `Bool` lane lives in the matching `I64xL`
/// mask. Panics on an unsupported width — `compile_simd` only ever passes 2/4/8.
pub(crate) fn vec_ty(scalar: ScalarTy, lanes: usize) -> Type {
    match (scalar, lanes) {
        (ScalarTy::F64, 2) => types::F64X2,
        (ScalarTy::F64, 4) => types::F64X4,
        (ScalarTy::F64, 8) => types::F64X8,
        // Bool is carried as an I64xL canonical mask.
        (ScalarTy::I64 | ScalarTy::Bool, 2) => types::I64X2,
        (ScalarTy::I64 | ScalarTy::Bool, 4) => types::I64X4,
        (ScalarTy::I64 | ScalarTy::Bool, 8) => types::I64X8,
        _ => unreachable!("compile_simd passes lanes in {{2,4,8}} and numeric/bool lane types"),
    }
}

/// Vector emitter over a chain's base row index `i` (the first of `lanes` rows).
pub(crate) struct SimdCodegen<'a, 'b> {
    pub(crate) b: &'a mut FunctionBuilder<'b>,
    pub(crate) cols: &'a ColumnSet,
    pub(crate) col_ptrs: &'a [Value],
    /// The chain's base row index (first lane); the chain covers `[i, i+lanes)`.
    pub(crate) i: Value,
    /// Lanes per vector (2/4/8).
    pub(crate) lanes: usize,
}

impl SimdCodegen<'_, '_> {
    /// Emit the vector value of `expr` for rows `[i, i+lanes)`.
    pub(crate) fn emit(&mut self, expr: &bc_expr::Expr) -> Value {
        self.emit_typed(expr).0
    }

    /// Emit the vector value of `expr` with its scalar (lane) type. A `Bool` lane
    /// type means an `I64xL` canonical mask (all-ones / all-zeros). The expression
    /// is pre-validated by [`simd_ty`](crate::simd_ty), so the `unreachable!` arms
    /// are genuinely unreachable.
    fn emit_typed(&mut self, expr: &bc_expr::Expr) -> (Value, ScalarTy) {
        use bc_expr::{BinaryOp, Expr, Literal};
        match expr {
            Expr::Col { name } => {
                // Contiguous `lanes*8`-byte load. Unaligned (notrap-only) flags: the
                // input/output buffers are only 8-byte aligned, so the engine never
                // asserts wider vector alignment for these accesses.
                let ty = self.cols.ty[name];
                let base = self.col_ptrs[self.cols.index(name)];
                let off = self.b.ins().imul_imm(self.i, 8);
                let addr = self.b.ins().iadd(base, off);
                let flags = MemFlags::new().with_notrap();
                match ty {
                    ScalarTy::I64 => (
                        self.b
                            .ins()
                            .load(vec_ty(ScalarTy::I64, self.lanes), flags, addr, 0),
                        ScalarTy::I64,
                    ),
                    ScalarTy::F64 => (
                        self.b
                            .ins()
                            .load(vec_ty(ScalarTy::F64, self.lanes), flags, addr, 0),
                        ScalarTy::F64,
                    ),
                    // `simd_ty` admits only I64/F64 columns (temporal is excluded).
                    _ => unreachable!("simd_ty excludes non-numeric columns"),
                }
            }
            Expr::Lit { value } => match value {
                Literal::Int(x) => {
                    let s = self.b.ins().iconst(types::I64, *x);
                    (
                        self.b.ins().splat(vec_ty(ScalarTy::I64, self.lanes), s),
                        ScalarTy::I64,
                    )
                }
                Literal::Float(x) => {
                    let s = self.b.ins().f64const(*x);
                    (
                        self.b.ins().splat(vec_ty(ScalarTy::F64, self.lanes), s),
                        ScalarTy::F64,
                    )
                }
                _ => unreachable!("simd_ty admits only Int/Float literals"),
            },
            Expr::Not { input } => {
                // Boolean NOT on a canonical mask: bitwise-not flips all-ones <->
                // all-zeros, keeping the result canonical.
                let (v, _) = self.emit_typed(input);
                (self.b.ins().bnot(v), ScalarTy::Bool)
            }
            Expr::Cast { input, dtype, .. } => {
                let (v, vt) = self.emit_typed(input);
                let target = match bc_arrow::dtype_from_name(dtype) {
                    Some(arrow::datatypes::DataType::Int64) => ScalarTy::I64,
                    Some(arrow::datatypes::DataType::Float64) => ScalarTy::F64,
                    _ => unreachable!("validated in simd_ty"),
                };
                match (vt, target) {
                    // int64 -> float64: lane-wise exact convert (matches Arrow).
                    (ScalarTy::I64, ScalarTy::F64) => (
                        self.b
                            .ins()
                            .fcvt_from_sint(vec_ty(ScalarTy::F64, self.lanes), v),
                        ScalarTy::F64,
                    ),
                    // No-op casts pass the value through unchanged.
                    (ScalarTy::I64, ScalarTy::I64) => (v, ScalarTy::I64),
                    (ScalarTy::F64, ScalarTy::F64) => (v, ScalarTy::F64),
                    _ => unreachable!("validated in simd_ty"),
                }
            }
            Expr::Binary { op, left, right } => {
                let (mut lv, lt) = self.emit_typed(left);
                let (mut rv, rt) = self.emit_typed(right);
                let is_cmp = matches!(
                    op,
                    BinaryOp::Eq
                        | BinaryOp::Ne
                        | BinaryOp::Lt
                        | BinaryOp::Le
                        | BinaryOp::Gt
                        | BinaryOp::Ge
                );
                // Promote to f64 lanes if either side is f64 (matches Arrow).
                let promote_f64 = lt == ScalarTy::F64 || rt == ScalarTy::F64;
                if promote_f64 {
                    let fty = vec_ty(ScalarTy::F64, self.lanes);
                    if lt == ScalarTy::I64 {
                        lv = self.b.ins().fcvt_from_sint(fty, lv);
                    }
                    if rt == ScalarTy::I64 {
                        rv = self.b.ins().fcvt_from_sint(fty, rv);
                    }
                }
                if is_cmp {
                    (self.emit_cmp(*op, lv, rv, promote_f64), ScalarTy::Bool)
                } else if promote_f64 {
                    (self.emit_farith(*op, lv, rv), ScalarTy::F64)
                } else {
                    (self.emit_iarith(*op, lv, rv), ScalarTy::I64)
                }
            }
            _ => unreachable!("simd_ty validated the vectorizable subset"),
        }
    }

    /// Integer vector arithmetic. Only `Add`/`Sub`/`Mul` reach here (`simd_ty`
    /// excludes integer `Div`/`Mod`); two's-complement wrap is per-lane identical
    /// to the scalar `iadd`/`isub`/`imul`, so parity holds.
    fn emit_iarith(&mut self, op: bc_expr::BinaryOp, l: Value, r: Value) -> Value {
        use bc_expr::BinaryOp::*;
        match op {
            Add => self.b.ins().iadd(l, r),
            Sub => self.b.ins().isub(l, r),
            Mul => self.b.ins().imul(l, r),
            _ => unreachable!("simd_ty admits only integer +,-,* "),
        }
    }

    /// Float vector arithmetic. `Add`/`Sub`/`Mul`/`Div` are IEEE per-lane identical
    /// to the scalar path; `Mod` (an `fmod` libcall) is excluded by `simd_ty`.
    fn emit_farith(&mut self, op: bc_expr::BinaryOp, l: Value, r: Value) -> Value {
        use bc_expr::BinaryOp::*;
        match op {
            Add => self.b.ins().fadd(l, r),
            Sub => self.b.ins().fsub(l, r),
            Mul => self.b.ins().fmul(l, r),
            Div => self.b.ins().fdiv(l, r),
            _ => unreachable!("simd_ty admits only float +,-,*,/"),
        }
    }

    /// Vector comparison producing an `I64xL` canonical mask (all-ones for true).
    /// Mirrors the scalar [`Codegen::emit_cmp`](crate::emit::Codegen) lane-wise: for
    /// floats it builds the same total-order NaN result (NaN == NaN, NaN sorts above
    /// every non-NaN) from IEEE compares plus `Unordered` NaN tests, so the vector
    /// path agrees with the interpreter on NaN, not bare IEEE. For NaN-free lanes
    /// every NaN test is zero and it collapses to the plain IEEE compare.
    fn emit_cmp(&mut self, op: bc_expr::BinaryOp, l: Value, r: Value, is_float: bool) -> Value {
        use bc_expr::BinaryOp::*;
        if is_float {
            let a_nan = self.b.ins().fcmp(FloatCC::Unordered, l, l);
            let b_nan = self.b.ins().fcmp(FloatCC::Unordered, r, r);
            let a_ord = self.b.ins().bnot(a_nan); // l is not NaN
            let b_ord = self.b.ins().bnot(b_nan); // r is not NaN
            let both_nan = self.b.ins().band(a_nan, b_nan);
            match op {
                Eq => {
                    let feq = self.b.ins().fcmp(FloatCC::Equal, l, r);
                    self.b.ins().bor(feq, both_nan)
                }
                Ne => {
                    let feq = self.b.ins().fcmp(FloatCC::Equal, l, r);
                    let eq = self.b.ins().bor(feq, both_nan);
                    self.b.ins().bnot(eq)
                }
                Lt => {
                    let lt = self.b.ins().fcmp(FloatCC::LessThan, l, r);
                    let rhs = self.b.ins().bor(b_nan, lt);
                    self.b.ins().band(a_ord, rhs)
                }
                Le => {
                    let le = self.b.ins().fcmp(FloatCC::LessThanOrEqual, l, r);
                    let rhs = self.b.ins().bor(b_nan, le);
                    let main = self.b.ins().band(a_ord, rhs);
                    self.b.ins().bor(main, both_nan)
                }
                Gt => {
                    let gt = self.b.ins().fcmp(FloatCC::GreaterThan, l, r);
                    let rhs = self.b.ins().bor(a_nan, gt);
                    self.b.ins().band(b_ord, rhs)
                }
                Ge => {
                    let ge = self.b.ins().fcmp(FloatCC::GreaterThanOrEqual, l, r);
                    let rhs = self.b.ins().bor(a_nan, ge);
                    let main = self.b.ins().band(b_ord, rhs);
                    self.b.ins().bor(main, both_nan)
                }
                _ => unreachable!("emit_cmp only handles comparisons"),
            }
        } else {
            let cc = match op {
                Eq => IntCC::Equal,
                Ne => IntCC::NotEqual,
                Lt => IntCC::SignedLessThan,
                Le => IntCC::SignedLessThanOrEqual,
                Gt => IntCC::SignedGreaterThan,
                Ge => IntCC::SignedGreaterThanOrEqual,
                _ => unreachable!("emit_cmp only handles comparisons"),
            };
            self.b.ins().icmp(cc, l, r)
        }
    }
}
