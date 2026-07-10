//! Kleene / null-propagation support analysis for the JIT.
//!
//! Pure predicates over `bc_expr::Expr` that gate the two nullable-input compile paths in
//! `lib.rs`: `needs_kleene` selects the three-valued-logic ABI for a compound predicate
//! (`AND`/`OR`, where `false AND null = false`), and `is_null_propagating` admits the
//! cheaper combined-validity-mask recovery for an expression whose result is null *iff* an
//! input is null. Split out of `lib.rs` on a responsibility seam (analysis, not codegen);
//! they touch no codegen state, only the shared `Expr`.

use bc_expr::{BinaryOp, Expr};

/// True when `expr` is a compound predicate (`AND`/`OR` somewhere) *and* every node is in
/// the Kleene-supported subset, so the JIT must use the three-valued-logic ABI rather than
/// the plain bitwise path (which is wrong for `false AND null` / `true OR null`).
pub(crate) fn needs_kleene(expr: &Expr) -> bool {
    contains_and_or(expr) && kleene_supported(expr)
}

fn contains_and_or(expr: &Expr) -> bool {
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

fn kleene_supported(expr: &Expr) -> bool {
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
pub(crate) fn is_null_propagating(expr: &Expr) -> bool {
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
