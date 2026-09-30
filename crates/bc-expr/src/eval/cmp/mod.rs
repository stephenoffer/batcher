//! Comparison kernels beside the generic one: typed fast paths, and the nested path.
//!
//! [`crate::eval::binary`] broadcasts a literal operand as an Arrow `Scalar` and hands the pair
//! to `arrow_ord::cmp`, which is right for every type and generic over all of them. The kernels
//! here answer the same question for one type each, faster, and decline anything they cannot
//! answer bit-for-bit — so the generic path stays the definition and these stay interchangeable
//! with it.
//!
//! [`nested`] is the one exception to "one type each, a faster twin of the generic path": it
//! *is* the path for lists, structs and maps, which the generic kernels refuse.

mod nested;
mod string;

pub(crate) use nested::{eval_nested_cmp, is_nested};
pub(crate) use string::{string_scalar_cmp, try_string_range};
