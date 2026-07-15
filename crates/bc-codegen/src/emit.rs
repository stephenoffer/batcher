//! Per-element IR emitter: recurses over a validated `Expr` building Cranelift
//! values at the current loop index, producing one output element per row.

use std::collections::HashMap;

use cranelift_codegen::ir::{types, InstBuilder, MemFlags, Value};
use cranelift_frontend::FunctionBuilder;

use crate::{libm_binary_symbol, libm_unary_symbol, ColumnSet, ScalarTy};

/// Per-element IR emitter. Recurses over the expression building Cranelift
/// values; loads happen at the current loop index `i`.
pub(crate) struct Codegen<'a, 'b> {
    pub(crate) b: &'a mut FunctionBuilder<'b>,
    pub(crate) cols: &'a ColumnSet,
    pub(crate) col_ptrs: &'a [Value],
    /// Per-column validity (i8, 1 = valid) base pointers, parallel to `col_ptrs`.
    /// `Some` only in the Kleene value+validity compile mode (`emit_validity`).
    pub(crate) null_ptrs: Option<&'a [Value]>,
    pub(crate) i: Value,
    /// Imported libc `fmod`, used for float remainder.
    pub(crate) fmod: cranelift_codegen::ir::FuncRef,
    /// Imported libm symbols (`log`, `sin`, `pow`, ...) keyed by name, used to
    /// lower the transcendental and two-arg math functions via libcalls.
    pub(crate) libm: &'a HashMap<&'static str, cranelift_codegen::ir::FuncRef>,
}

impl Codegen<'_, '_> {
    /// Emit the value of `expr` for the current row, returning it with its
    /// scalar type. The expression is pre-validated, so `unreachable!` guards
    /// genuinely unreachable arms.
    pub(crate) fn emit(&mut self, expr: &bc_expr::Expr) -> Value {
        let (v, _ty) = self.emit_typed(expr);
        v
    }

    /// Emit the i8 validity (1 = valid, 0 = null) of `expr` for the current row,
    /// used only in the Kleene (value + validity) compile mode. `And`/`Or` follow
    /// the Kleene truth tables — a result is non-null even when one operand is null
    /// if the other alone determines it (`false AND null = false`, `true OR null =
    /// true`). Every other supported node propagates nulls (validity = AND of its
    /// operands'). `compile`'s `needs_kleene` guarantees only these node kinds reach
    /// here (no `Case`/`Coalesce`, whose value itself depends on validity).
    pub(crate) fn emit_validity(&mut self, expr: &bc_expr::Expr) -> Value {
        use bc_expr::{BinaryOp, Expr};
        match expr {
            Expr::Col { name } => {
                // Load this row's validity byte from the column's parallel i8 array.
                let idx = self.cols.index(name);
                let base = self.null_ptrs.expect("kleene mode sets null_ptrs")[idx];
                let addr = self.b.ins().iadd(base, self.i); // 1 byte per element
                self.b.ins().load(types::I8, MemFlags::trusted(), addr, 0)
            }
            Expr::Lit { .. } => self.b.ins().iconst(types::I8, 1),
            Expr::Binary {
                op: BinaryOp::And,
                left,
                right,
            } => {
                let lvalid = self.emit_validity(left);
                let rvalid = self.emit_validity(right);
                let (lv, _) = self.emit_typed(left);
                let (rv, _) = self.emit_typed(right);
                // Non-null iff both operands valid, OR either is a valid `false`
                // (which alone settles the AND to false).
                let both = self.b.ins().band(lvalid, rvalid);
                let l_false = {
                    let lnot = self.b.ins().bxor_imm(lv, 1);
                    self.b.ins().band(lvalid, lnot)
                };
                let r_false = {
                    let rnot = self.b.ins().bxor_imm(rv, 1);
                    self.b.ins().band(rvalid, rnot)
                };
                let t = self.b.ins().bor(both, l_false);
                self.b.ins().bor(t, r_false)
            }
            Expr::Binary {
                op: BinaryOp::Or,
                left,
                right,
            } => {
                let lvalid = self.emit_validity(left);
                let rvalid = self.emit_validity(right);
                let (lv, _) = self.emit_typed(left);
                let (rv, _) = self.emit_typed(right);
                // Non-null iff both operands valid, OR either is a valid `true`.
                let both = self.b.ins().band(lvalid, rvalid);
                let l_true = self.b.ins().band(lvalid, lv);
                let r_true = self.b.ins().band(rvalid, rv);
                let t = self.b.ins().bor(both, l_true);
                self.b.ins().bor(t, r_true)
            }
            // Arithmetic / comparison / two-arg math: result null iff an operand is.
            Expr::Binary { left, right, .. } | Expr::Math2 { left, right, .. } => {
                let l = self.emit_validity(left);
                let r = self.emit_validity(right);
                self.b.ins().band(l, r)
            }
            // Unary value ops propagate their operand's validity unchanged.
            Expr::Not { input } | Expr::Cast { input, .. } | Expr::Math { input, .. } => {
                self.emit_validity(input)
            }
            _ => unreachable!("needs_kleene guarantees a validity-supported node"),
        }
    }

    fn emit_typed(&mut self, expr: &bc_expr::Expr) -> (Value, ScalarTy) {
        use bc_expr::{BinaryOp, Expr, Literal};
        match expr {
            Expr::Col { name } => {
                let ty = self.cols.ty[name];
                let idx = self.cols.index(name);
                let base = self.col_ptrs[idx];
                if ty == ScalarTy::Date32 {
                    // Date32 is an i32 buffer (4-byte stride): load the day count and
                    // sign-extend to i64 so it shares the i64 comparison path. Signed
                    // extension preserves the i32 ordering Arrow's date comparison
                    // uses, so the result is bit-for-bit identical to the interpreter.
                    let off = self.b.ins().imul_imm(self.i, 4);
                    let addr = self.b.ins().iadd(base, off);
                    let v32 = self.b.ins().load(types::I32, MemFlags::trusted(), addr, 0);
                    return (self.b.ins().sextend(types::I64, v32), ScalarTy::Date32);
                }
                let off = self.b.ins().imul_imm(self.i, 8);
                let addr = self.b.ins().iadd(base, off);
                let v = self.b.ins().load(ty.clif(), MemFlags::trusted(), addr, 0);
                (v, ty)
            }
            Expr::Lit { value } => match value {
                Literal::Int(x) => (self.b.ins().iconst(types::I64, *x), ScalarTy::I64),
                Literal::Float(x) => (self.b.ins().f64const(*x), ScalarTy::F64),
                // A date literal is its i32 day count, widened to the i64 date operand.
                Literal::Date(d) => (self.b.ins().iconst(types::I64, *d as i64), ScalarTy::Date32),
                // A timestamp literal is its i64 microsecond instant.
                Literal::Timestamp(t) => (self.b.ins().iconst(types::I64, *t), ScalarTy::TsUs),
                _ => unreachable!("validated in analyze"),
            },
            Expr::Not { input } => {
                // Boolean NOT: flip the i8 0/1 value (no nulls on this path).
                let (v, _) = self.emit_typed(input);
                let one = self.b.ins().iconst(types::I8, 1);
                (self.b.ins().bxor(v, one), ScalarTy::Bool)
            }
            Expr::Binary { op, left, right } if matches!(op, BinaryOp::And | BinaryOp::Or) => {
                // Both operands are booleans (i8 0/1); bitwise band/bor is the
                // logical op and matches the interpreter on the null-free path.
                let (lv, _) = self.emit_typed(left);
                let (rv, _) = self.emit_typed(right);
                let v = match op {
                    BinaryOp::And => self.b.ins().band(lv, rv),
                    BinaryOp::Or => self.b.ins().bor(lv, rv),
                    _ => unreachable!(),
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
                // Promote to f64 if either side is f64 (matches Arrow).
                let promote_f64 = lt == ScalarTy::F64 || rt == ScalarTy::F64;
                if promote_f64 {
                    if lt == ScalarTy::I64 {
                        lv = self.b.ins().fcvt_from_sint(types::F64, lv);
                    }
                    if rt == ScalarTy::I64 {
                        rv = self.b.ins().fcvt_from_sint(types::F64, rv);
                    }
                }
                if is_cmp {
                    let v = self.emit_cmp(*op, lv, rv, promote_f64);
                    (v, ScalarTy::Bool)
                } else if promote_f64 {
                    (self.emit_farith(*op, lv, rv), ScalarTy::F64)
                } else {
                    (self.emit_iarith(*op, lv, rv), ScalarTy::I64)
                }
            }
            Expr::Cast { input, dtype, .. } => {
                // Validated in `analyze` to one of: I64->F64 (convert),
                // I64->I64 / F64->F64 (no-op). Classify the target the same way.
                let (v, vt) = self.emit_typed(input);
                let target = match dtype.as_str() {
                    "int64" | "long" => ScalarTy::I64,
                    "float64" | "double" => ScalarTy::F64,
                    _ => unreachable!("validated in analyze"),
                };
                match (vt, target) {
                    // int64 -> float64: exact, matches Arrow's int->float cast.
                    (ScalarTy::I64, ScalarTy::F64) => {
                        (self.b.ins().fcvt_from_sint(types::F64, v), ScalarTy::F64)
                    }
                    // No-op casts: pass the value through unchanged.
                    (ScalarTy::I64, ScalarTy::I64) => (v, ScalarTy::I64),
                    (ScalarTy::F64, ScalarTy::F64) => (v, ScalarTy::F64),
                    _ => unreachable!("validated in analyze"),
                }
            }
            Expr::Math { func, input } => {
                use bc_expr::MathFunc::*;
                let (v, vt) = self.emit_typed(input);
                match func {
                    // `abs`: float -> fabs; int -> select(x < 0, 0 - x, x), with i64::MIN
                    // saturated to i64::MAX (its negation overflows) so the result is never a
                    // negative "absolute value" and matches the interpreter's `saturating_abs`.
                    Abs => match vt {
                        ScalarTy::F64 => (self.b.ins().fabs(v), ScalarTy::F64),
                        ScalarTy::I64 => {
                            let zero = self.b.ins().iconst(types::I64, 0);
                            let neg = self.b.ins().isub(zero, v);
                            let is_neg = self.b.ins().icmp(
                                cranelift_codegen::ir::condcodes::IntCC::SignedLessThan,
                                v,
                                zero,
                            );
                            let abs = self.b.ins().select(is_neg, neg, v);
                            // saturate: abs(i64::MIN) wraps back to i64::MIN (still < 0) — map
                            // that one value to i64::MAX to match `i64::saturating_abs`.
                            let i64_min = self.b.ins().iconst(types::I64, i64::MIN);
                            let i64_max = self.b.ins().iconst(types::I64, i64::MAX);
                            let is_min = self.b.ins().icmp(
                                cranelift_codegen::ir::condcodes::IntCC::Equal,
                                v,
                                i64_min,
                            );
                            (self.b.ins().select(is_min, i64_max, abs), ScalarTy::I64)
                        }
                        ScalarTy::Bool | ScalarTy::Date32 | ScalarTy::TsUs => {
                            unreachable!("validated in analyze")
                        }
                    },
                    // floor/ceil/sqrt/trunc operate on f64; promote an int input to
                    // f64 first, exactly as the interpreter's `cast` does.
                    Floor | Ceil | Sqrt | Trunc => {
                        let f = if vt == ScalarTy::I64 {
                            self.b.ins().fcvt_from_sint(types::F64, v)
                        } else {
                            v
                        };
                        let out = match func {
                            Floor => self.b.ins().floor(f),
                            Ceil => self.b.ins().ceil(f),
                            Sqrt => self.b.ins().sqrt(f),
                            Trunc => self.b.ins().trunc(f),
                            _ => unreachable!(),
                        };
                        (out, ScalarTy::F64)
                    }
                    // Transcendentals: promote an int input to f64 (matching the
                    // interpreter's cast) then call the corresponding libm symbol.
                    _ => {
                        let sym = libm_unary_symbol(*func).expect("validated in analyze");
                        let f = if vt == ScalarTy::I64 {
                            self.b.ins().fcvt_from_sint(types::F64, v)
                        } else {
                            v
                        };
                        let call = self.b.ins().call(self.libm[sym], &[f]);
                        (self.b.inst_results(call)[0], ScalarTy::F64)
                    }
                }
            }
            Expr::Math2 { func, left, right } => {
                // `pow`/`atan2` -> libm libcall; both operands are promoted to f64
                // first (matching the interpreter's Float64 cast), so the result
                // is bit-for-bit identical to `eval_math2`.
                let sym = libm_binary_symbol(*func).expect("validated in analyze");
                let (lv, lt) = self.emit_typed(left);
                let (rv, rt) = self.emit_typed(right);
                let lf = if lt == ScalarTy::I64 {
                    self.b.ins().fcvt_from_sint(types::F64, lv)
                } else {
                    lv
                };
                let rf = if rt == ScalarTy::I64 {
                    self.b.ins().fcvt_from_sint(types::F64, rv)
                } else {
                    rv
                };
                let call = self.b.ins().call(self.libm[sym], &[lf, rf]);
                (self.b.inst_results(call)[0], ScalarTy::F64)
            }
            Expr::Case {
                branches,
                otherwise,
            } => {
                // Compute the common result type the same way `analyze` does:
                // F64 if any of `otherwise`/`then` is F64, else I64.
                let result_ty = {
                    let mut t = self.case_ty(otherwise);
                    for branch in branches {
                        if self.case_ty(&branch.then) == ScalarTy::F64 {
                            t = ScalarTy::F64;
                        }
                    }
                    t
                };
                // Mirror the interpreter's fold (bc-expr): start from the default
                // and fold the branches in REVERSE, overriding `acc` where the
                // WHEN holds. Because branch[0] is applied last, the first
                // matching WHEN wins — identical to the interpreter's
                // `acc = zip(when, then, acc)` over `branches.iter().rev()`.
                let (ov, ot) = self.emit_typed(otherwise);
                let mut acc = self.promote_to(ov, ot, result_ty);
                for branch in branches.iter().rev() {
                    let (cond, _) = self.emit_typed(&branch.when);
                    let (tv, tt) = self.emit_typed(&branch.then);
                    let then_val = self.promote_to(tv, tt, result_ty);
                    // `select(cond, if_true, if_false)`: the i8 0/1 boolean is
                    // nonzero exactly when the WHEN matches.
                    acc = self.b.ins().select(cond, then_val, acc);
                }
                (acc, result_ty)
            }
            _ => unreachable!("validated in analyze"),
        }
    }

    /// Result scalar type of a Case sub-expression (column type or literal type),
    /// used to compute the Case's common type without re-emitting. Only `Col`,
    /// `Lit`, arithmetic, and nested `Case` can change int/float-ness; for
    /// everything else we conservatively fall back to recomputing via the same
    /// promotion rule. This mirrors `analyze` (already validated), so it never
    /// sees Bool for a then/otherwise position.
    fn case_ty(&self, expr: &bc_expr::Expr) -> ScalarTy {
        use bc_expr::{BinaryOp, Expr, Literal, MathFunc};
        match expr {
            Expr::Col { name } => self.cols.ty[name],
            Expr::Lit { value } => match value {
                Literal::Int(_) => ScalarTy::I64,
                Literal::Float(_) => ScalarTy::F64,
                _ => unreachable!("validated in analyze"),
            },
            Expr::Binary { op, left, right } => {
                // Arithmetic promotes to F64 if either side is F64 (comparisons
                // are Bool, which can't appear in a then/otherwise position).
                if matches!(
                    op,
                    BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Mod
                ) && (self.case_ty(left) == ScalarTy::F64
                    || self.case_ty(right) == ScalarTy::F64)
                {
                    ScalarTy::F64
                } else {
                    ScalarTy::I64
                }
            }
            Expr::Cast { dtype, .. } => match dtype.as_str() {
                "float64" | "double" => ScalarTy::F64,
                _ => ScalarTy::I64,
            },
            Expr::Math { func, input } => match func {
                // `abs` preserves the input type; the rest produce f64.
                MathFunc::Abs => self.case_ty(input),
                _ => ScalarTy::F64,
            },
            // Two-arg math (pow/atan2) always produces f64.
            Expr::Math2 { .. } => ScalarTy::F64,
            Expr::Case {
                branches,
                otherwise,
            } => {
                let mut t = self.case_ty(otherwise);
                for branch in branches {
                    if self.case_ty(&branch.then) == ScalarTy::F64 {
                        t = ScalarTy::F64;
                    }
                }
                t
            }
            _ => ScalarTy::I64,
        }
    }

    /// Promote `value` (of `from`) to `target`. Only int->f64 needs an
    /// instruction (`fcvt_from_sint`, exact, matching Arrow); same-type is a
    /// passthrough. f64->i64 never occurs: then/otherwise are constrained to the
    /// common type, so a float anywhere forces the whole Case to f64.
    fn promote_to(&mut self, value: Value, from: ScalarTy, target: ScalarTy) -> Value {
        match (from, target) {
            (ScalarTy::I64, ScalarTy::F64) => self.b.ins().fcvt_from_sint(types::F64, value),
            _ => value,
        }
    }

    fn emit_iarith(&mut self, op: bc_expr::BinaryOp, l: Value, r: Value) -> Value {
        use bc_expr::BinaryOp::*;
        match op {
            Add => self.b.ins().iadd(l, r),
            Sub => self.b.ins().isub(l, r),
            Mul => self.b.ins().imul(l, r),
            Div => self.b.ins().sdiv(l, r),
            Mod => self.b.ins().srem(l, r),
            _ => unreachable!(),
        }
    }

    fn emit_farith(&mut self, op: bc_expr::BinaryOp, l: Value, r: Value) -> Value {
        use bc_expr::BinaryOp::*;
        match op {
            Add => self.b.ins().fadd(l, r),
            Sub => self.b.ins().fsub(l, r),
            Mul => self.b.ins().fmul(l, r),
            Div => self.b.ins().fdiv(l, r),
            // No native f64 remainder instruction; Arrow's `f64 % f64` lowers to
            // libc `fmod`, so call it to get bit-identical results.
            Mod => {
                let call = self.b.ins().call(self.fmod, &[l, r]);
                self.b.inst_results(call)[0]
            }
            _ => unreachable!(),
        }
    }

    fn emit_cmp(&mut self, op: bc_expr::BinaryOp, l: Value, r: Value, is_float: bool) -> Value {
        use bc_expr::BinaryOp::*;
        use cranelift_codegen::ir::condcodes::IntCC;
        let cc = match op {
            Eq => IntCC::Equal,
            Ne => IntCC::NotEqual,
            Lt => IntCC::SignedLessThan,
            Le => IntCC::SignedLessThanOrEqual,
            Gt => IntCC::SignedGreaterThan,
            Ge => IntCC::SignedGreaterThanOrEqual,
            _ => unreachable!(),
        };
        let raw = if is_float {
            // The interpreter compares floats with Arrow's `cmp` kernel, which uses
            // `ArrowNativeTypeOp` — i.e. `f64::total_cmp` for ordering and raw-bit
            // equality for `==`. That is a TOTAL order, not IEEE: `-0.0 < 0.0`,
            // a negative NaN sorts below `-inf`, a positive NaN above `+inf`, and two
            // NaNs are equal only when their bit patterns are identical. Plain IEEE
            // `fcmp` (or the old "any NaN sorts greatest, all NaNs equal" shim) gets
            // all of those wrong. `total_cmp` maps the bits to a monotonic i64 key
            // (`bits ^ (((bits >> 63) as u64) >> 1)`) and does a signed integer
            // compare; reproduce exactly that so the JIT is bit-for-bit identical to
            // the interpreter on -0.0, every NaN sign/payload, and the infinities.
            let lk = total_order_key(self.b, l);
            let rk = total_order_key(self.b, r);
            self.b.ins().icmp(cc, lk, rk)
        } else {
            self.b.ins().icmp(cc, l, r)
        };
        // `icmp` yields an i8 boolean already in 0/1 form for the result store;
        // mask to be safe and match the `out: *mut u8` ABI.
        self.b.ins().band_imm(raw, 1)
    }
}

/// Reinterpret an `f64` value as the monotonic `i64` total-order key that
/// [`f64::total_cmp`] uses: `bits ^ (((bits >> 63) as u64) >> 1)`. A signed integer
/// compare of two keys is bit-for-bit identical to `a.total_cmp(&b)` (and key
/// equality is bit equality, since the map is a bijection), so it matches the
/// interpreter's Arrow float comparison on `-0.0`, every NaN sign/payload, and the
/// infinities. Used by both the scalar and (lane-wise) the SIMD comparison paths.
pub(crate) fn total_order_key(b: &mut FunctionBuilder, v: Value) -> Value {
    let bits = b.ins().bitcast(types::I64, MemFlags::new(), v);
    // Arithmetic shift: all-ones when the sign bit is set, all-zeros otherwise.
    let sign = b.ins().sshr_imm(bits, 63);
    // Logical shift by 1: 0x7fff_ffff_ffff_ffff for negatives, 0 for non-negatives.
    let mask = b.ins().ushr_imm(sign, 1);
    b.ins().bxor(bits, mask)
}
