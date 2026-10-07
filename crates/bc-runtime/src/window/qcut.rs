//! `qcut`: bin every row by the quantiles of its own partition (pandas `qcut`).
//!
//! The edges are data-dependent, which is what separates this from `cut`: `cut` takes its
//! breaks at plan time and lowers to a `CASE` chain, but "the quartiles of this column" are
//! not known until the column is. So each partition is read whole, its non-null values are
//! sorted once, the requested quantiles become the edges, and every row is assigned the
//! 0-based index of the right-closed interval it falls in.
//!
//! The quantile rule is numpy's `linear` method — the one pandas' `qcut` reaches through
//! `Series.quantile` — down to its virtual-index arithmetic and its two-sided `lerp`, so an
//! edge is the same double pandas computes and a value sitting exactly on an edge lands in
//! the same bin. Order within a partition is irrelevant: the function never reads it.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, Int64Array};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Float64Type};

use crate::error::RuntimeError;

/// Bin each row of `values` by the `probs`-quantiles of its partition.
///
/// `probs` are strictly increasing probabilities in `[0, 1]`, validated by the caller.
/// Edge `k` is the `probs[k]` quantile of the partition's non-null, non-NaN values; a row
/// gets `k` when `edge[k] < x <= edge[k + 1]`, with the lowest edge itself in bin 0 (pandas'
/// `include_lowest`). A null or NaN row, and a row outside `[edge[0], edge[last]]` (possible
/// only when the probabilities do not span `[0, 1]`), is null.
///
/// Tied input can make two edges equal. Then the bin between them is empty, and the call
/// errors unless `drop_duplicates`, which merges equal edges and numbers the surviving bins
/// consecutively — pandas' `duplicates="raise"` / `"drop"`.
pub(crate) fn qcut_window(
    partitions: &[Vec<usize>],
    values: &ArrayRef,
    probs: &[f64],
    drop_duplicates: bool,
    num_rows: usize,
) -> Result<ArrayRef, RuntimeError> {
    if !values.data_type().is_numeric() && values.data_type() != &DataType::Null {
        return Err(RuntimeError::UnsupportedWindow {
            func: "qcut".to_string(),
            dtype: values.data_type().to_string(),
        });
    }
    let floats = cast(values, &DataType::Float64)?;
    let arr = floats.as_primitive::<Float64Type>();
    let value_at =
        |row: usize| (arr.is_valid(row) && !arr.value(row).is_nan()).then(|| arr.value(row));
    let mut out = vec![None::<i64>; num_rows];
    let mut sorted = Vec::new();
    for part in partitions {
        sorted.clear();
        sorted.extend(part.iter().filter_map(|&row| value_at(row)));
        if sorted.is_empty() {
            continue;
        }
        sorted.sort_unstable_by(f64::total_cmp);
        let edges = partition_edges(&sorted, probs, drop_duplicates)?;
        for &row in part {
            out[row] = value_at(row).and_then(|x| bin_of(&edges, x));
        }
    }
    Ok(Arc::new(Int64Array::from(out)))
}

/// The partition's edges, deduplicated or refused when tied input collapsed two of them.
fn partition_edges(
    sorted: &[f64],
    probs: &[f64],
    drop_duplicates: bool,
) -> Result<Vec<f64>, RuntimeError> {
    let mut edges: Vec<f64> = probs.iter().map(|&p| linear_quantile(sorted, p)).collect();
    // The edges are non-decreasing (the probabilities increase), so a duplicate is always
    // adjacent. pandas exempts a two-edge list, where the single bin is all there is.
    if edges.len() > 2 && edges.windows(2).any(|w| w[0] == w[1]) {
        if !drop_duplicates {
            return Err(RuntimeError::QcutDuplicateEdges {
                edges: format!("{edges:?}"),
            });
        }
        edges.dedup();
    }
    Ok(edges)
}

/// `x`'s 0-based bin among right-closed `edges`, or `None` outside them.
///
/// pandas' `searchsorted(edges, x, side="left") - 1`, with `x == edges[0]` moved into the
/// first bin. Fewer than two edges bound no interval at all -- a constant partition whose
/// quantiles all deduplicated to one value -- so every row is null, as in pandas.
#[inline]
fn bin_of(edges: &[f64], x: f64) -> Option<i64> {
    if edges.len() < 2 {
        return None;
    }
    if x == edges[0] {
        return Some(0);
    }
    let below = edges.partition_point(|&e| e < x);
    (below >= 1 && below < edges.len()).then(|| below as i64 - 1)
}

/// numpy's `quantile(sorted, p, method="linear")` on an already sorted, NaN-free slice.
///
/// Spelled exactly the way numpy computes it -- the virtual index `(n - 1)·p`, then its
/// two-sided `lerp` -- because a value lying exactly on an edge is decided by the edge's last
/// bit: pandas' `qcut([3, -1, 2.5, 2.5, 10, -4, 0], 3)` has an edge at `-4.4e-16`, not 0,
/// which is what puts the 0 in the middle bin.
fn linear_quantile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    let last = n - 1;
    // numpy's `linear` method: `get_virtual_index = (n - 1) * quantiles`.
    let virtual_index = last as f64 * p;
    let floor = virtual_index.floor();
    let (lo, hi) = if virtual_index >= last as f64 {
        (last, last)
    } else if virtual_index < 0.0 {
        (0, 0)
    } else {
        let lo = floor as usize;
        (lo, lo + 1)
    };
    let (a, b) = (sorted[lo], sorted[hi]);
    let t = virtual_index - floor;
    // numpy `_lerp`: interpolate from the nearer end, so `t = 1` returns `b` exactly.
    let diff = b - a;
    if t >= 0.5 {
        b - diff * (1.0 - t)
    } else {
        a + diff * t
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Float64Array;

    fn ints(arr: &ArrayRef) -> Vec<Option<i64>> {
        arr.as_primitive::<arrow::datatypes::Int64Type>()
            .iter()
            .collect()
    }

    fn quartiles() -> Vec<f64> {
        vec![0.0, 0.25, 0.5, 0.75, 1.0]
    }

    /// pandas `qcut([1..8], 4, labels=False)` is `[0,0,1,1,2,2,3,3]`; a null stays null.
    #[test]
    fn quartiles_match_pandas() {
        let values: ArrayRef = Arc::new(Float64Array::from(vec![
            Some(5.0),
            Some(1.0),
            None,
            Some(8.0),
            Some(2.0),
            Some(7.0),
            Some(3.0),
            Some(6.0),
            Some(4.0),
        ]));
        let parts = vec![(0..9).collect::<Vec<usize>>()];
        let got = qcut_window(&parts, &values, &quartiles(), false, 9).unwrap();
        assert_eq!(
            ints(&got),
            vec![
                Some(2),
                Some(0),
                None,
                Some(3),
                Some(0),
                Some(3),
                Some(1),
                Some(2),
                Some(1)
            ]
        );
    }

    /// A value exactly on an interior edge is in the lower (right-closed) bin, and the
    /// minimum is in bin 0: pandas `qcut([1, 2, 3], 2)` puts 2 with 1.
    #[test]
    fn edges_are_right_closed_with_the_minimum_included() {
        let values: ArrayRef = Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0]));
        let parts = vec![vec![0usize, 1, 2]];
        let got = qcut_window(&parts, &values, &[0.0, 0.5, 1.0], false, 3).unwrap();
        assert_eq!(ints(&got), vec![Some(0), Some(0), Some(1)]);
    }

    /// Tied input that collapses two edges raises by default and merges them under
    /// `drop_duplicates`, numbering the surviving bins consecutively.
    #[test]
    fn duplicate_edges_raise_or_collapse() {
        let values: ArrayRef = Arc::new(Float64Array::from(vec![1.0, 1.0, 1.0, 1.0, 5.0, 9.0]));
        let parts = vec![(0..6).collect::<Vec<usize>>()];
        let err = qcut_window(&parts, &values, &quartiles(), false, 6).unwrap_err();
        assert!(err.to_string().contains("duplicates"), "{err}");
        // Edges [1, 1, 1, 4, 9] dedupe to [1, 4, 9].
        let got = qcut_window(&parts, &values, &quartiles(), true, 6).unwrap();
        assert_eq!(
            ints(&got),
            vec![Some(0), Some(0), Some(0), Some(0), Some(1), Some(1)]
        );
        // A constant column dedupes to a single edge, which bounds no bin: null, as pandas
        // `qcut([3, 3], 4, duplicates="drop")` answers NaN.
        let constant: ArrayRef = Arc::new(Float64Array::from(vec![3.0, 3.0]));
        let got = qcut_window(&[vec![0usize, 1]], &constant, &quartiles(), true, 2).unwrap();
        assert_eq!(ints(&got), vec![None, None]);
    }

    /// Partitions are binned by their own quantiles, and an all-null partition is null.
    #[test]
    fn partitions_are_independent() {
        let values: ArrayRef = Arc::new(Float64Array::from(vec![
            Some(1.0),
            Some(100.0),
            Some(2.0),
            Some(200.0),
            None,
        ]));
        let parts = vec![vec![0usize, 2], vec![1usize, 3], vec![4usize]];
        let got = qcut_window(&parts, &values, &[0.0, 0.5, 1.0], false, 5).unwrap();
        assert_eq!(ints(&got), vec![Some(0), Some(0), Some(1), Some(1), None]);
    }

    /// The numpy-linear quantile, checked against `numpy.quantile` values.
    #[test]
    fn linear_quantile_matches_numpy() {
        let v = [1.0, 2.0, 3.0, 4.0, 10.0];
        assert_eq!(linear_quantile(&v, 0.0), 1.0);
        assert_eq!(linear_quantile(&v, 1.0), 10.0);
        assert_eq!(linear_quantile(&v, 0.5), 3.0);
        assert_eq!(linear_quantile(&v, 0.9), 7.6000000000000005);
        assert_eq!(linear_quantile(&[4.0], 0.3), 4.0);
        // pandas' third-quantile probability after its `p * 100 / 100` round trip lands a
        // hair below the integer index, so the edge sits just below 0.
        let v = [-4.0, -1.0, 0.0, 2.5, 2.5, 3.0, 10.0];
        let p = (1.0f64 / 3.0) * 100.0 / 100.0;
        assert_eq!(linear_quantile(&v, p), -4.440892098500626e-16);
    }
}
