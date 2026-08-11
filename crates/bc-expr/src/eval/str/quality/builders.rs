//! Shared column builders for the text-quality measures.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Array, Int64Array, StringArray};

/// Build a Float64 column from a per-row measure, preserving the input's nulls.
pub(super) fn float_column<F>(s: &StringArray, f: F) -> ArrayRef
where
    F: Fn(&str) -> Option<f64>,
{
    let values: Vec<Option<f64>> = (0..s.len())
        .map(|i| if s.is_null(i) { None } else { f(s.value(i)) })
        .collect();
    Arc::new(Float64Array::from(values))
}

/// Build an Int64 column from a per-row count, preserving the input's nulls.
pub(super) fn int_column<F>(s: &StringArray, f: F) -> ArrayRef
where
    F: Fn(&str) -> i64,
{
    let values: Vec<Option<i64>> = (0..s.len())
        .map(|i| (!s.is_null(i)).then(|| f(s.value(i))))
        .collect();
    Arc::new(Int64Array::from(values))
}
