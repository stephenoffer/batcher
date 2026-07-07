//! Sketch / statistics FFI: HyperLogLog distinct counts, KLL/TDigest quantiles,
//! Misra-Gries heavy hitters, and reservoir sampling over Arrow batches.
//!
//! These wrap `bc_sketches` for the control plane's metadata-learning path
//! (`core.column_statistics` / `core.heavy_hitters`): mergeable summaries the
//! optimizer consumes for cardinality, selectivity, and skew. Extracted from `lib`
//! along the statistics seam to keep the FFI root within the size budget.

use arrow::array::{Array, RecordBatch};
use arrow::compute::cast;
use arrow::datatypes::DataType;
use arrow_pyarrow::PyArrowType;
use bc_sketches::Mergeable;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Estimate the number of distinct (non-null) values in a column across batches,
/// using HyperLogLog++. Mergeable, so it can be computed per partition.
#[pyfunction]
pub(crate) fn estimate_distinct(
    column: &str,
    batches: Vec<PyArrowType<RecordBatch>>,
) -> PyResult<f64> {
    let mut sketch: Option<bc_sketches::ColumnStats> = None;
    for batch in batches {
        let b = batch.0;
        let col = b
            .column_by_name(column)
            .ok_or_else(|| PyRuntimeError::new_err(format!("no column {column:?}")))?;
        let stats = bc_sketches::ColumnStats::from_array(col);
        match &mut sketch {
            Some(s) => s.merge(&stats),
            None => sketch = Some(stats),
        }
    }
    Ok(sketch.map_or(0.0, |s| s.distinct_estimate()))
}

/// Per-column statistics for the optimizer (the W2 metadata FFI seam): for each
/// requested column, merge `ColumnStats` (HLL distinct + KLL quantiles) across all
/// batches and return a dict of scalar summaries. Keys per column:
/// `ndv` (distinct estimate), `count`, `null_count`, `null_fraction`, `avg_bytes`
/// (measured per-row byte width), and `min`/`max` (`None` for non-numeric columns).
/// Mergeable, so it composes across partitions — Core can collect this during
/// execution and persist it to the MetadataHub for Kyber's `__column_ndv__` /
/// `__column_avg_bytes__` / range-selectivity to consume.
#[pyfunction]
pub(crate) fn column_stats(
    columns: Vec<String>,
    batches: Vec<PyArrowType<RecordBatch>>,
) -> PyResult<std::collections::HashMap<String, std::collections::HashMap<String, Option<f64>>>> {
    let merged = merge_column_stats(&columns, &batches);
    let mut out = std::collections::HashMap::new();
    for (name, s) in merged {
        let mut d = std::collections::HashMap::new();
        d.insert("ndv".to_string(), Some(s.distinct_estimate()));
        d.insert("count".to_string(), Some(s.count as f64));
        d.insert("null_count".to_string(), Some(s.null_count as f64));
        d.insert("null_fraction".to_string(), Some(s.null_fraction()));
        d.insert("avg_bytes".to_string(), Some(s.avg_byte_width()));
        d.insert("min".to_string(), s.min());
        d.insert("max".to_string(), s.max());
        out.insert(name, d);
    }
    Ok(out)
}

/// Merge `ColumnStats` for each requested column across all batches in one pass.
/// Shared by `column_stats`, `column_quantiles`, and `column_stats_full` so each
/// column's HLL+KLL sketch is built once per call site.
pub(crate) fn merge_column_stats(
    columns: &[String],
    batches: &[PyArrowType<RecordBatch>],
) -> std::collections::HashMap<String, bc_sketches::ColumnStats> {
    let mut merged: std::collections::HashMap<String, bc_sketches::ColumnStats> =
        std::collections::HashMap::new();
    for batch in batches {
        let b = &batch.0;
        for name in columns {
            if let Some(col) = b.column_by_name(name) {
                let stats = bc_sketches::ColumnStats::from_array(col);
                merged
                    .entry(name.clone())
                    .and_modify(|s| s.merge(&stats))
                    .or_insert(stats);
            }
        }
    }
    merged
}

/// Quantile boundaries at `probs` for a numeric column's sketch; an empty list
/// unless every probability resolves (i.e. the column is numeric / has a KLL).
pub(crate) fn quantile_values(s: &bc_sketches::ColumnStats, probs: &[f64]) -> Vec<f64> {
    let vals: Vec<f64> = probs.iter().filter_map(|&q| s.quantile(q)).collect();
    if vals.len() == probs.len() {
        vals
    } else {
        Vec::new()
    }
}

/// Per-column quantile boundaries (the KLL sketch) for histogram-based range
/// selectivity in the optimizer. For each numeric column, return the value at each
/// requested probability in `probs` (so Kyber can interpolate `fraction <= literal`);
/// non-numeric columns return an empty list. Mergeable across batches, so Core can
/// collect it online and persist it to the MetadataHub alongside `column_stats`.
#[pyfunction]
pub(crate) fn column_quantiles(
    columns: Vec<String>,
    batches: Vec<PyArrowType<RecordBatch>>,
    probs: Vec<f64>,
) -> PyResult<std::collections::HashMap<String, Vec<f64>>> {
    let merged = merge_column_stats(&columns, &batches);
    Ok(merged
        .into_iter()
        .map(|(name, s)| (name, quantile_values(&s, &probs)))
        .collect())
}

/// Combined per-column summary **and** quantiles in a single sketch pass — the seam
/// `core.column_statistics` uses, replacing two separate FFI calls (`column_stats`
/// then `column_quantiles`) that each rebuilt the same HLL+KLL sketch over the data.
/// Returns `(stats, quantiles)`: `stats` is `column_stats`' per-column scalar dict,
/// `quantiles` is `column_quantiles`' per-column boundary list.
#[pyfunction]
#[allow(clippy::type_complexity)]
pub(crate) fn column_stats_full(
    columns: Vec<String>,
    batches: Vec<PyArrowType<RecordBatch>>,
    probs: Vec<f64>,
) -> PyResult<(
    std::collections::HashMap<String, std::collections::HashMap<String, Option<f64>>>,
    std::collections::HashMap<String, Vec<f64>>,
)> {
    let merged = merge_column_stats(&columns, &batches);
    let mut stats = std::collections::HashMap::new();
    let mut quants = std::collections::HashMap::new();
    for (name, s) in merged {
        let mut d = std::collections::HashMap::new();
        d.insert("ndv".to_string(), Some(s.distinct_estimate()));
        d.insert("count".to_string(), Some(s.count as f64));
        d.insert("null_count".to_string(), Some(s.null_count as f64));
        d.insert("null_fraction".to_string(), Some(s.null_fraction()));
        d.insert("avg_bytes".to_string(), Some(s.avg_byte_width()));
        d.insert("min".to_string(), s.min());
        d.insert("max".to_string(), s.max());
        quants.insert(name.clone(), quantile_values(&s, &probs));
        stats.insert(name, d);
    }
    Ok((stats, quants))
}

/// Tail-accurate quantiles (the TDigest sketch) for numeric columns. Where the
/// coarse KLL grid in `column_quantiles` is built for range selectivity, TDigest
/// is accurate in the tails (p99/p999) — what an `approx_quantile` answer wants.
/// For each numeric column, returns the value at each requested probability;
/// non-numeric or empty columns return an empty list. Mergeable across batches.
#[pyfunction]
pub(crate) fn tail_quantiles(
    columns: Vec<String>,
    batches: Vec<PyArrowType<RecordBatch>>,
    probs: Vec<f64>,
) -> PyResult<std::collections::HashMap<String, Vec<f64>>> {
    let mut digests: std::collections::HashMap<String, bc_sketches::TDigest> =
        std::collections::HashMap::new();
    for batch in &batches {
        let b = &batch.0;
        for name in &columns {
            if let Some(col) = b.column_by_name(name) {
                let Ok(f) = cast(col, &DataType::Float64) else {
                    continue;
                };
                let Some(arr) = f.as_any().downcast_ref::<arrow::array::Float64Array>() else {
                    continue;
                };
                let d = digests.entry(name.clone()).or_default();
                for i in 0..arr.len() {
                    if arr.is_valid(i) {
                        d.add(arr.value(i));
                    }
                }
            }
        }
    }
    let mut out = std::collections::HashMap::new();
    for (name, mut d) in digests {
        let vals: Vec<f64> = probs.iter().filter_map(|&q| d.quantile(q)).collect();
        out.insert(
            name,
            if vals.len() == probs.len() {
                vals
            } else {
                Vec::new()
            },
        );
    }
    Ok(out)
}

/// Build a TDigest over `column`'s numeric values across `batches` and return its
/// serialized bytes — the *partial* step of a mergeable approximate quantile. Returns
/// `None` when the column is missing, non-numeric, or has no valid values. Paired with
/// `tdigest_quantile`: each partition (or streamed chunk) builds one sketch, the driver
/// merges them, so an approximate quantile never collects the column to one place.
#[pyfunction]
pub(crate) fn tdigest_partial(
    column: String,
    batches: Vec<PyArrowType<RecordBatch>>,
) -> PyResult<Option<Vec<u8>>> {
    let mut digest = bc_sketches::TDigest::default();
    let mut any = false;
    for batch in &batches {
        let b = &batch.0;
        if let Some(col) = b.column_by_name(&column) {
            let Ok(f) = cast(col, &DataType::Float64) else {
                continue;
            };
            let Some(arr) = f.as_any().downcast_ref::<arrow::array::Float64Array>() else {
                continue;
            };
            for i in 0..arr.len() {
                if arr.is_valid(i) {
                    digest.add(arr.value(i));
                    any = true;
                }
            }
        }
    }
    Ok(any.then(|| digest.to_bytes()))
}

/// Merge serialized TDigest `sketches` (from `tdigest_partial`) and return the value at
/// quantile `q` — the *combine + finalize* step. `None` when no sketch carried data.
#[pyfunction]
pub(crate) fn tdigest_quantile(sketches: Vec<Vec<u8>>, q: f64) -> PyResult<Option<f64>> {
    let mut merged: Option<bc_sketches::TDigest> = None;
    for bytes in &sketches {
        if let Some(d) = bc_sketches::TDigest::from_bytes(bytes) {
            match merged.as_mut() {
                Some(m) => m.merge(&d),
                None => merged = Some(d),
            }
        }
    }
    Ok(merged.and_then(|mut m| m.quantile(q)))
}

/// Heavy hitters (the Misra-Gries `FrequentItems` sketch) per column: the values
/// whose frequency exceeds `fraction` of the rows, with their estimated counts.
/// Kyber consumes this for skew detection (a hot join key → salting). Values are
/// rendered to strings (cast to Utf8) so any column type can be labelled; columns
/// that cannot cast are skipped. Mergeable in spirit — built across all batches.
#[pyfunction]
pub(crate) fn heavy_hitters(
    columns: Vec<String>,
    batches: Vec<PyArrowType<RecordBatch>>,
    fraction: f64,
) -> PyResult<std::collections::HashMap<String, Vec<(String, u64)>>> {
    // Misra-Gries capacity: 1/fraction guarantees all keys above `fraction` survive.
    let capacity = ((1.0 / fraction).ceil() as usize).max(1);
    let mut items: std::collections::HashMap<String, bc_sketches::FrequentItems<String>> =
        std::collections::HashMap::new();
    for batch in &batches {
        let b = &batch.0;
        for name in &columns {
            if let Some(col) = b.column_by_name(name) {
                let Ok(s) = cast(col, &DataType::Utf8) else {
                    continue;
                };
                let Some(arr) = s.as_any().downcast_ref::<arrow::array::StringArray>() else {
                    continue;
                };
                let fi = items
                    .entry(name.clone())
                    .or_insert_with(|| bc_sketches::FrequentItems::new(capacity));
                for i in 0..arr.len() {
                    if arr.is_valid(i) {
                        fi.add(arr.value(i).to_string());
                    }
                }
            }
        }
    }
    let mut out = std::collections::HashMap::new();
    for (name, fi) in items {
        out.insert(name, fi.heavy_hitters(fraction));
    }
    Ok(out)
}

/// A uniform random row sample (the reservoir sketch, Algorithm R) of size `k`
/// across all batches, returned as one `RecordBatch`. Used for sampling-based
/// estimation / `TABLESAMPLE` without materializing the whole input. When the
/// input has at most `k` rows, returns them all.
#[pyfunction]
pub(crate) fn reservoir_sample(
    batches: Vec<PyArrowType<RecordBatch>>,
    k: usize,
) -> PyResult<PyArrowType<RecordBatch>> {
    use arrow::array::UInt64Array;

    if batches.is_empty() {
        return Err(PyRuntimeError::new_err(
            "reservoir_sample: no input batches",
        ));
    }
    let schema = batches[0].0.schema();
    let refs: Vec<&RecordBatch> = batches.iter().map(|b| &b.0).collect();
    let combined = arrow::compute::concat_batches(&schema, refs)
        .map_err(|e| PyRuntimeError::new_err(format!("concat failed: {e}")))?;
    let total = combined.num_rows();
    if total <= k {
        return Ok(PyArrowType(combined));
    }
    // Reservoir of global row indices; deterministic seed keeps it reproducible.
    let mut reservoir = bc_sketches::ReservoirSample::new(k);
    for idx in 0..total {
        reservoir.add(idx as u64);
    }
    let indices = UInt64Array::from(reservoir.sample().to_vec());
    let mut cols = Vec::with_capacity(combined.num_columns());
    for col in combined.columns() {
        let taken = arrow::compute::take(col, &indices, None)
            .map_err(|e| PyRuntimeError::new_err(format!("take failed: {e}")))?;
        cols.push(taken);
    }
    let sampled = RecordBatch::try_new(schema, cols)
        .map_err(|e| PyRuntimeError::new_err(format!("rebatch failed: {e}")))?;
    Ok(PyArrowType(sampled))
}
