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
                let flags = MemFlags::new().with_notrap();
                match ty {
                    ScalarTy::I64 => {
                        let off = self.b.ins().imul_imm(self.i, 8);
                        let addr = self.b.ins().iadd(base, off);
                        (
                            self.b
                                .ins()
                                .load(vec_ty(ScalarTy::I64, self.lanes), flags, addr, 0),
                            ScalarTy::I64,
                        )
                    }
                    ScalarTy::F64 => {
                        let off = self.b.ins().imul_imm(self.i, 8);
                        let addr = self.b.ins().iadd(base, off);
                        (
                            self.b
                                .ins()
                                .load(vec_ty(ScalarTy::F64, self.lanes), flags, addr, 0),
                            ScalarTy::F64,
                        )
                    }
                    // tz-naive Timestamp-µs is an i64 instant buffer — a plain I64xL
                    // load, carried as an integer lane for the comparison.
                    ScalarTy::TsUs => {
                        let off = self.b.ins().imul_imm(self.i, 8);
                        let addr = self.b.ins().iadd(base, off);
                        (
                            self.b
                                .ins()
                                .load(vec_ty(ScalarTy::I64, self.lanes), flags, addr, 0),
                            ScalarTy::TsUs,
                        )
                    }
                    // Date32 is an i32 day-count buffer (4-byte stride). `sload32x2`
                    // loads 2 lanes and sign-extends each to i64 — bit-identical to the
                    // scalar path's `load i32 + sextend`. The dispatch pins a Date32
                    // predicate to `lanes == 2`, so this I64X2 matches the other lanes.
                    ScalarTy::Date32 => {
                        debug_assert_eq!(self.lanes, 2, "Date32 SIMD is pinned to 2 lanes");
                        let off = self.b.ins().imul_imm(self.i, 4);
                        let addr = self.b.ins().iadd(base, off);
                        (self.b.ins().sload32x2(flags, addr, 0), ScalarTy::Date32)
                    }
                    // `simd_ty` admits only I64/F64/Date32/TsUs columns.
                    ScalarTy::Bool => unreachable!("a bare boolean column is not a simd_ty leaf"),
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
                // A date literal is its i32 day count, a timestamp literal its i64 µs
                // instant — each splat to an i64 lane to compare against the matching
                // temporal column (loaded as sign-extended / native i64).
                Literal::Date(d) => {
                    let s = self.b.ins().iconst(types::I64, *d as i64);
                    (
                        self.b.ins().splat(vec_ty(ScalarTy::I64, self.lanes), s),
                        ScalarTy::Date32,
                    )
                }
                Literal::Timestamp(t) => {
                    let s = self.b.ins().iconst(types::I64, *t);
                    (
                        self.b.ins().splat(vec_ty(ScalarTy::I64, self.lanes), s),
                        ScalarTy::TsUs,
                    )
                }
                _ => unreachable!("simd_ty admits only Int/Float/Date/Timestamp literals"),
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
            // `And`/`Or` of two boolean canonical masks: a bitwise `band`/`bor` keeps
            // the mask canonical (all-ones / all-zeros). Correct for a null-free batch;
            // the dispatch guarantees a batch with nulls falls back to the interpreter.
            Expr::Binary {
                op: op @ (BinaryOp::And | BinaryOp::Or),
                left,
                right,
            } => {
                let (lv, _) = self.emit_typed(left);
                let (rv, _) = self.emit_typed(right);
                let v = match op {
                    BinaryOp::And => self.b.ins().band(lv, rv),
                    BinaryOp::Or => self.b.ins().bor(lv, rv),
                    _ => unreachable!("matched And/Or"),
                };
                (v, ScalarTy::Bool)
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
                // A temporal comparison (date / timestamp) runs on the i64 lanes the
                // operands were loaded into — an integer compare, never a float promote.
                let is_temporal = |t: ScalarTy| matches!(t, ScalarTy::Date32 | ScalarTy::TsUs);
                if is_cmp && (is_temporal(lt) || is_temporal(rt)) {
                    return (self.emit_cmp(*op, lv, rv, false), ScalarTy::Bool);
                }
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
    /// Mirrors the scalar [`emit::canon_total_order_key`](crate::emit::canon_total_order_key)
    /// lane-wise: the interpreter canonicalizes float operands (`bc_arrow::canon_f64` — the
    /// two zeros are one value, all NaNs are one value and greatest) and then compares with
    /// Arrow's `cmp` kernel, i.e. `f64::total_cmp`'s monotonic i64 key. We reproduce both
    /// steps per lane, so the vector path is bit-for-bit identical to the interpreter and to
    /// the scalar JIT — not bare IEEE, and not the raw-bit order that split `-0.0` from `0.0`.
    fn emit_cmp(&mut self, op: bc_expr::BinaryOp, l: Value, r: Value, is_float: bool) -> Value {
        use bc_expr::BinaryOp::*;
        let cc = match op {
            Eq => IntCC::Equal,
            Ne => IntCC::NotEqual,
            Lt => IntCC::SignedLessThan,
            Le => IntCC::SignedLessThanOrEqual,
            Gt => IntCC::SignedGreaterThan,
            Ge => IntCC::SignedGreaterThanOrEqual,
            _ => unreachable!("emit_cmp only handles comparisons"),
        };
        if is_float {
            let ity = vec_ty(ScalarTy::I64, self.lanes);
            let lk = self.canon_total_order_key(l, ity);
            let rk = self.canon_total_order_key(r, ity);
            self.b.ins().icmp(cc, lk, rk)
        } else {
            self.b.ins().icmp(cc, l, r)
        }
    }

    /// Per-lane [`emit::canon_total_order_key`](crate::emit::canon_total_order_key):
    /// canonicalize the float (`-0.0` -> `0.0`, every NaN -> one quiet NaN), then map to
    /// `f64::total_cmp`'s monotonic i64 key. `ity` is the `I64xL` vector type.
    ///
    /// A vector `fcmp` yields an all-ones/all-zeros **lane mask** rather than the scalar
    /// path's 0/1 boolean, so the two selects are `bitselect`s over that mask.
    fn canon_total_order_key(&mut self, v: Value, ity: Type) -> Value {
        let fty = vec_ty(ScalarTy::F64, self.lanes);
        let bits = self.b.ins().bitcast(ity, MemFlags::new(), v);

        let zero_f = self.b.ins().f64const(0.0);
        let zero_v = self.b.ins().splat(fty, zero_f);
        // `Equal` is an *ordered* compare: true for `+0.0` and `-0.0`, false for NaN.
        let is_zero = self.b.ins().fcmp(FloatCC::Equal, v, zero_v);
        let is_zero = self.b.ins().bitcast(ity, MemFlags::new(), is_zero);
        // A lane is unordered with itself exactly when it is NaN — any sign, any payload.
        let is_nan = self.b.ins().fcmp(FloatCC::Unordered, v, v);
        let is_nan = self.b.ins().bitcast(ity, MemFlags::new(), is_nan);

        let zero_i = self.b.ins().iconst(types::I64, 0);
        let zero_bits = self.b.ins().splat(ity, zero_i);
        let nan_i = self
            .b
            .ins()
            .iconst(types::I64, bc_arrow::CANONICAL_NAN_BITS_F64 as i64);
        let nan_bits = self.b.ins().splat(ity, nan_i);

        let bits = self.b.ins().bitselect(is_zero, zero_bits, bits);
        let bits = self.b.ins().bitselect(is_nan, nan_bits, bits);

        // `bits ^ (((bits >> 63) as u64) >> 1)`, lane-wise.
        let sign = self.b.ins().sshr_imm(bits, 63);
        let mask = self.b.ins().ushr_imm(sign, 1);
        self.b.ins().bxor(bits, mask)
    }
}
