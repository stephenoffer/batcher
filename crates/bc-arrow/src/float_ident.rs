//! The engine's one definition of **float identity**.
//!
//! Two questions decide every float-valued relational operation: *are these the same
//! value?* and *which is greater?* Both have an answer SQL fixes and IEEE does not:
//!
//! * `-0.0` and `0.0` are **the same value** (IEEE agrees for `==`, but their raw bits
//!   differ, so bit-equality and Arrow's total order split them);
//! * every NaN is **one value**, and it is **greater than every number** (IEEE says every
//!   comparison with NaN is false, so NaN is neither equal, less, nor greater — which is
//!   not an order at all, and lets a `max()` written as `v > cur` silently skip NaN).
//!
//! This is DuckDB's semantics, and it is what a `GROUP BY`, `DISTINCT`, a join key, an
//! `ORDER BY`, and a scalar `=` must *all* agree on — otherwise the same column means two
//! different things depending on which operator reads it. [`canon_f64`] is that agreement,
//! expressed once: fold `-0.0` into `0.0` and every NaN bit-pattern into one quiet NaN,
//! and every remaining value keeps its exact bits, so distinct finite values stay distinct.
//!
//! It lives in `bc-arrow` — the lowest crate the expression evaluator (`bc-expr`), the
//! stateful primitives (`bc-runtime`), the sketches (`bc-sketches`), and the JIT
//! (`bc-codegen`) all see — precisely so there is no second copy to drift from this one.
//! A float path that hashes, compares, or groups raw bits without coming through here is a
//! bug: it is how `-0.0` once counted as its own group, its own distinct value, and its own
//! join key.

use std::sync::Arc;

use arrow::array::{ArrayRef, AsArray};
use arrow::datatypes::{DataType, Float32Type, Float64Type};

/// Bits of the one canonical quiet NaN every NaN folds to.
///
/// The sign bit is clear, so [`f64::total_cmp`] ranks it above every number — the order
/// `ORDER BY` already sorts in. (A *negative* NaN would rank below `-inf`, which is why
/// the sign must be normalized away rather than preserved.)
pub const CANONICAL_NAN_BITS_F64: u64 = 0x7ff8_0000_0000_0000;

/// Bits of the one canonical quiet NaN for a 32-bit float. Mirrors [`CANONICAL_NAN_BITS_F64`].
pub const CANONICAL_NAN_BITS_F32: u32 = 0x7fc0_0000;

/// The canonical `f64` denoting `v`'s identity: `-0.0` → `0.0`, every NaN → one quiet NaN.
///
/// Comparing two canonical values on Arrow's total order (`f64::total_cmp`) yields exactly
/// SQL's answer: the zeros compare equal, all NaNs compare equal and greatest, and every
/// other pair is unchanged. See the module docs.
#[inline]
pub fn canon_f64(v: f64) -> f64 {
    f64::from_bits(canon_f64_bits(v))
}

/// Canonical `u64` **bits** for an `f64` — [`canon_f64`] for the raw-bit hashing paths.
///
/// Hashing these bits makes hash equality agree with `=`/`GROUP BY`: the two zeros hash
/// alike and all NaNs hash alike, while distinct finite values keep their exact bits and so
/// stay distinct.
#[inline]
pub fn canon_f64_bits(v: f64) -> u64 {
    if v.is_nan() {
        CANONICAL_NAN_BITS_F64
    } else if v == 0.0 {
        0 // `+0.0` bits — the `==` folds `-0.0` in with it
    } else {
        v.to_bits()
    }
}

/// The canonical `f32` denoting `v`'s identity. Mirrors [`canon_f64`] for a 32-bit float.
///
/// A top-level `Float32` is widened to `Float64` at the FFI boundary, but one *nested*
/// inside a list/struct is not, so a float leaf can still arrive as `f32` and must fold the
/// same way.
#[inline]
pub fn canon_f32(v: f32) -> f32 {
    f32::from_bits(canon_f32_bits(v))
}

/// Canonical `u32` **bits** for an `f32`. Mirrors [`canon_f64_bits`].
#[inline]
pub fn canon_f32_bits(v: f32) -> u32 {
    if v.is_nan() {
        CANONICAL_NAN_BITS_F32
    } else if v == 0.0 {
        0
    } else {
        v.to_bits()
    }
}

/// Total order over `f64` matching the order `ORDER BY` sorts in and `=` compares in.
///
/// Equivalent to `canon_f64(a).total_cmp(&canon_f64(b))`, written directly: all NaNs are
/// equal and greater than every number, and `-0.0` compares `Equal` to `0.0` (so which of
/// the two an extreme returns is first-seen — the rule every other engine applies).
#[inline]
pub fn float_total_cmp(a: f64, b: f64) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        // Neither is NaN, so `partial_cmp` is total here — and it folds the two zeros.
        (false, false) => a.partial_cmp(&b).unwrap_or(Ordering::Equal),
    }
}

/// [`canon_f64`] applied to a whole array — the array-level entry point for every path that
/// compares, orders, or ranks a float column through an Arrow kernel.
///
/// Arrow's `cmp`, `sort_to_indices`, `lexsort`, and `RowConverter` all rank floats on
/// `f64::total_cmp` over the **raw** bits, which is a different relation from the engine's
/// float identity on exactly two shapes: it splits `-0.0` from `0.0`, and it ranks a
/// *negative* NaN below `-inf` while a positive one ranks above `+inf`. Canonicalizing the
/// input first makes those same kernels compute the engine's relation, so a path can keep
/// using them (and their vectorized speed) and still agree with `GROUP BY` / `=` / `MIN`.
///
/// Use it on the **key** of an ordering, not on the output: sort the canonical key to get the
/// permutation, then gather the *original* rows, so the ordering is corrected without
/// rewriting the user's data (a `-NaN` in, a `-NaN` out).
///
/// A non-float array is returned untouched, and so is a float array holding neither `-0.0`
/// nor a NaN — the overwhelmingly common case, which costs one scan and no allocation.
pub fn canon_float_array(a: &ArrayRef) -> ArrayRef {
    match a.data_type() {
        DataType::Float64 => {
            let f = a.as_primitive::<Float64Type>();
            // Null slots' payloads never matter (they compare/order as null either way), so
            // scanning `values()` wholesale can only over-trigger, never miss.
            if !f.values().iter().any(|v| needs_canon_f64(*v)) {
                return Arc::clone(a);
            }
            Arc::new(f.unary::<_, Float64Type>(canon_f64))
        }
        DataType::Float32 => {
            let f = a.as_primitive::<Float32Type>();
            if !f.values().iter().any(|v| needs_canon_f32(*v)) {
                return Arc::clone(a);
            }
            Arc::new(f.unary::<_, Float32Type>(canon_f32))
        }
        _ => Arc::clone(a),
    }
}

/// Whether `v` is one of the two shapes [`canon_f64`] rewrites: a NaN, or negative zero.
#[inline]
pub fn needs_canon_f64(v: f64) -> bool {
    v.is_nan() || v.to_bits() == 0x8000_0000_0000_0000
}

/// [`needs_canon_f64`] for a 32-bit float.
#[inline]
pub fn needs_canon_f32(v: f32) -> bool {
    v.is_nan() || v.to_bits() == 0x8000_0000
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Ordering;

    /// The bit patterns that make raw-bit reasoning disagree with SQL.
    fn awkward() -> Vec<f64> {
        vec![
            0.0,
            -0.0,
            f64::NAN,
            -f64::NAN,
            f64::from_bits(0x7ff8_0000_0000_0001), // a different NaN payload
            f64::from_bits(0xfff8_0000_0000_0001), // a negative NaN, different payload
            f64::INFINITY,
            f64::NEG_INFINITY,
            1.5,
            -1.5,
        ]
    }

    #[test]
    fn the_two_zeros_are_one_value() {
        assert_eq!(canon_f64_bits(0.0), canon_f64_bits(-0.0));
        assert_eq!(canon_f32_bits(0.0), canon_f32_bits(-0.0));
        assert_eq!(float_total_cmp(-0.0, 0.0), Ordering::Equal);
    }

    #[test]
    fn every_nan_is_one_value_regardless_of_sign_or_payload() {
        let nans = [
            f64::NAN,
            -f64::NAN,
            f64::from_bits(0x7ff8_0000_0000_0001),
            f64::from_bits(0xfff8_0000_0000_0001),
        ];
        for n in nans {
            assert_eq!(canon_f64_bits(n), CANONICAL_NAN_BITS_F64, "{n:?}");
            assert_eq!(float_total_cmp(n, f64::NAN), Ordering::Equal, "{n:?}");
            // ...and greater than every number, including the infinities.
            assert_eq!(
                float_total_cmp(n, f64::INFINITY),
                Ordering::Greater,
                "{n:?}"
            );
            assert_eq!(
                float_total_cmp(f64::NEG_INFINITY, n),
                Ordering::Less,
                "{n:?}"
            );
        }
    }

    #[test]
    fn distinct_finite_values_stay_distinct() {
        assert_ne!(canon_f64_bits(1.5), canon_f64_bits(-1.5));
        assert_ne!(canon_f64_bits(1.5), canon_f64_bits(1.5000000000000002));
        assert_eq!(canon_f64(1.5), 1.5);
        assert_eq!(canon_f64(f64::INFINITY), f64::INFINITY);
    }

    /// The property the whole module exists for: comparing canonical values on Arrow's
    /// total order is the same relation as `float_total_cmp`. The JIT reproduces the
    /// former and the interpreter the latter, so they must not be able to disagree.
    #[test]
    fn canon_then_total_cmp_equals_float_total_cmp() {
        for a in awkward() {
            for b in awkward() {
                assert_eq!(
                    canon_f64(a).total_cmp(&canon_f64(b)),
                    float_total_cmp(a, b),
                    "{a:?} vs {b:?}"
                );
            }
        }
    }

    /// A total order must be transitive and antisymmetric, or a sort over it is undefined.
    #[test]
    fn the_order_is_total() {
        for a in awkward() {
            for b in awkward() {
                assert_eq!(
                    float_total_cmp(a, b),
                    float_total_cmp(b, a).reverse(),
                    "{a:?} vs {b:?}"
                );
                for c in awkward() {
                    if float_total_cmp(a, b) == Ordering::Less
                        && float_total_cmp(b, c) == Ordering::Less
                    {
                        assert_eq!(float_total_cmp(a, c), Ordering::Less, "{a:?} {b:?} {c:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn canon_is_idempotent() {
        for v in awkward() {
            assert_eq!(canon_f64_bits(canon_f64(v)), canon_f64_bits(v), "{v:?}");
        }
    }
}
