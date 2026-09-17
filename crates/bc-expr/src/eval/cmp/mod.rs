//! Comparison kernels for a column against a one-value literal.
//!
//! [`crate::eval::binary`] broadcasts a literal operand as an Arrow `Scalar` and hands the pair
//! to `arrow_ord::cmp`, which is right for every type and generic over all of them. The kernels
//! here answer the same question for one type each, faster, and decline anything they cannot
//! answer bit-for-bit — so the generic path stays the definition and these stay interchangeable
//! with it.

mod string;

pub(crate) use string::{string_scalar_cmp, try_string_range};
