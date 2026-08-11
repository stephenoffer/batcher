//! Sketch / statistics FFI: HyperLogLog distinct counts, KLL/TDigest quantiles,
//! Misra-Gries heavy hitters, and reservoir sampling over Arrow batches.
//!
//! These wrap `bc_sketches` for the control plane's metadata-learning path
//! (`core.column_statistics` / `core.heavy_hitters`): mergeable summaries the
//! optimizer consumes for cardinality, selectivity, and skew. Extracted from `lib`
//! along the statistics seam to keep the FFI root within the size budget.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, RecordBatch};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Field, Schema};
use arrow_pyarrow::PyArrowType;
use bc_sketches::{FrequentItems, Mergeable};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Temporal types whose order the integer backing (days / ticks) preserves — the ones a
/// range/quantile pass can treat numerically. Excludes `Interval` (not a totally-ordered
/// scalar). Mirrors `bc_runtime::shuffle::is_temporal_key` so the sort *sample* here and the
/// range *partition* there agree on which keys are numeric-backed.
fn is_temporal_key(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Date32
            | DataType::Date64
            | DataType::Time32(_)
            | DataType::Time64(_)
            | DataType::Timestamp(_, _)
            | DataType::Duration(_)
    )
}

/// Cast a temporal column to `Int64` via its order-preserving backing (`Date32`/`Time32`
/// are `i32`-backed, so route through `Int32`). Same representation the range partition
/// uses, so quantile boundaries sampled here route rows there identically.
fn temporal_to_i64(col: &ArrayRef) -> Option<ArrayRef> {
    match cast(col, &DataType::Int64) {
        Ok(a) => Some(a),
        Err(_) => cast(col, &DataType::Int32)
            .and_then(|a| cast(&a, &DataType::Int64))
            .ok(),
    }
}

/// Replace each requested *temporal* column with its `Int64` backing, so the KLL quantile
/// sketch (numeric-only) can summarize a `Date`/`Timestamp` sort key. Non-temporal columns
/// and untouched batches pass through by clone. Scoped to `column_quantiles` (the sort
/// sample) — the optimizer's `column_stats` still sees dates as dates.
fn temporal_cols_as_i64(
    columns: &[String],
    batches: &[PyArrowType<RecordBatch>],
) -> Vec<PyArrowType<RecordBatch>> {
    let targets: std::collections::HashSet<&str> = columns.iter().map(|s| s.as_str()).collect();
    batches
        .iter()
        .map(|pb| {
            let b = &pb.0;
            let mut changed = false;
            let mut fields: Vec<Arc<Field>> = Vec::with_capacity(b.num_columns());
            let mut cols: Vec<ArrayRef> = Vec::with_capacity(b.num_columns());
            for (i, f) in b.schema().fields().iter().enumerate() {
                let c = b.column(i);
                if targets.contains(f.name().as_str()) && is_temporal_key(c.data_type()) {
                    if let Some(i64c) = temporal_to_i64(c) {
                        fields.push(Arc::new(Field::new(
                            f.name(),
                            DataType::Int64,
                            f.is_nullable(),
                        )));
                        cols.push(i64c);
                        changed = true;
                        continue;
                    }
                }
                fields.push(f.clone());
                cols.push(c.clone());
            }
            if !changed {
                return PyArrowType(b.clone());
            }
            match RecordBatch::try_new(Arc::new(Schema::new(fields)), cols) {
                Ok(rb) => PyArrowType(rb),
                Err(_) => PyArrowType(b.clone()),
            }
        })
        .collect()
}

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

/// Per-column distinct-count estimates (HLL only), computed in parallel across batches.
///
/// The distinct-count-only counterpart to `column_stats`. Kyber needs a column's `ndv`
/// to order joins (`|L||R| / max(ndv_L, ndv_R)`) and to size a `GROUP BY` (the product
/// of its key `ndv`s); without one the estimator falls back to `max(|L|, |R|)` and
/// `0.1 · rows`, which is what steers a plan into multi-gigabyte intermediates. Cold —
/// before any run has been measured — nothing else supplies `ndv`: a Parquet footer
/// carries row counts, null counts, and min/max, but no distinct count.
///
/// `column_stats` cannot serve that need cheaply because it also builds a KLL quantile
/// sketch per column (~50 ns/row, ~7x the HLL's cost). Dropping the KLL and merging the
/// per-batch HLLs with rayon makes seeding a source's `ndv` cheap enough to do on the
/// query path. The sketch is `Mergeable` — `merge` is associative and commutative — so
/// the parallel fold returns exactly what a sequential build would.
///
/// The GIL is released for the duration: once the Arrow arrays are in hand no Python
/// object is touched.
#[pyfunction]
pub(crate) fn column_ndv(
    py: Python<'_>,
    columns: Vec<String>,
    batches: Vec<PyArrowType<RecordBatch>>,
) -> PyResult<std::collections::HashMap<String, f64>> {
    use rayon::prelude::*;

    Ok(py.allow_threads(move || {
        let sketches: Vec<Option<bc_sketches::HyperLogLog>> = columns
            .par_iter()
            .map(|name| {
                batches
                    .par_iter()
                    .filter_map(|b| b.0.column_by_name(name))
                    .map(|col| {
                        let mut hll = bc_sketches::HyperLogLog::default_precision();
                        hll.add_array(col);
                        hll
                    })
                    .reduce_with(|mut a, b| {
                        a.merge(&b);
                        a
                    })
            })
            .collect();
        columns
            .into_iter()
            .zip(sketches)
            .filter_map(|(name, hll)| hll.map(|h| (name, h.estimate())))
            .collect()
    }))
}

/// Per-column statistics for the optimizer (the W2 metadata FFI seam): for each
/// requested column, merge `ColumnStats` (HLL distinct + KLL quantiles) across all
/// batches and return a dict of scalar summaries. Keys per column:
/// `ndv` (distinct estimate), `count`, `null_count`, `null_fraction`, `avg_bytes`
/// (measured per-row byte width), and `min`/`max` (`None` for non-numeric columns).
/// Mergeable, so it composes across partitions — Core can collect this during
/// execution and persist it to the MetadataHub for Kyber's `__column_ndv__` /
/// `__column_avg_bytes__` / range-selectivity to consume.
///
/// The GIL is released across the sketch build, as `column_ndv` does. Without that the
/// rayon fold inside `merge_column_stats` cannot actually run in parallel — measured on
/// TPC-DS `store_sales` (2.9 M rows, 23 columns), holding the GIL pinned it at **1.00x
/// parallelism and 2,345 ms** while `column_ndv` on the same arrays reached **12.9x and
/// 74 ms**. That difference is the whole reason the optimizer could not afford to ask for
/// these statistics.
#[pyfunction]
pub(crate) fn column_stats(
    py: Python<'_>,
    columns: Vec<String>,
    batches: Vec<PyArrowType<RecordBatch>>,
) -> PyResult<std::collections::HashMap<String, std::collections::HashMap<String, Option<f64>>>> {
    let merged = py.allow_threads(|| merge_column_stats(&columns, &batches));
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

/// Accumulate one `ColumnStats` per requested column across all batches, in one pass.
/// Shared by `column_stats`, `column_quantiles`, and `column_stats_full` so each
/// column's HLL+KLL sketch is built once per call site.
///
/// Each array is folded into the column's existing sketch rather than summarized into a
/// fresh one and merged. The difference is one HLL per *column* instead of one per
/// (column, morsel): a `HyperLogLog::default_precision()` is a 16 KB register array, so the
/// build-then-merge shape allocated and zeroed 16 KB per morsel per column and then walked
/// all of it again to take the register-wise maximum. On a 49-morsel, 16-column relation
/// that is roughly 12 MB allocated and 12 MB merged, on the query path, to summarize data
/// already in hand.
///
/// The distinct estimate is unchanged bit-for-bit (an HLL register is a maximum either way).
/// See `ColumnStats::update` for why the quantile sketch is not, and why that is fine.
///
/// The walk is parallel over **(column x chunk-of-batches)**, which is what makes these
/// statistics affordable enough for the optimizer to ask for them on the query path. Serial,
/// this was the single most expensive thing in planning a join: 2.2 s to summarize TPC-DS
/// `store_sales` (2.9 M rows, 23 columns) on a 16-core box — longer than the query it was
/// meant to speed up, so nothing could call it and every join estimate fell back to a
/// default.
///
/// The chunking is what keeps the allocation win described above. Folding batch-by-batch
/// inside a chunk means one 16 KB HLL per (column, *chunk*) rather than per (column, morsel),
/// so the register arrays stay bounded no matter how many morsels arrive. `CHUNK_TARGET` is a
/// fixed count rather than the core count on purpose: the merge shape then does not vary with
/// the machine, so a KLL quantile — the one part of `ColumnStats` a different merge order can
/// perturb — is reproducible run to run and box to box.
pub(crate) fn merge_column_stats(
    columns: &[String],
    batches: &[PyArrowType<RecordBatch>],
) -> std::collections::HashMap<String, bc_sketches::ColumnStats> {
    use rayon::prelude::*;

    /// Batch groups to split each column into. Fixed, so the merge tree is machine-independent.
    const CHUNK_TARGET: usize = 16;

    if batches.is_empty() || columns.is_empty() {
        return std::collections::HashMap::new();
    }
    let chunk = batches.len().div_ceil(CHUNK_TARGET).max(1);
    columns
        .par_iter()
        .filter_map(|name| {
            let per_chunk: Vec<bc_sketches::ColumnStats> = batches
                .par_chunks(chunk)
                .filter_map(|group| {
                    let mut acc: Option<bc_sketches::ColumnStats> = None;
                    for batch in group {
                        if let Some(col) = batch.0.column_by_name(name) {
                            acc.get_or_insert_with(bc_sketches::ColumnStats::empty)
                                .update(col);
                        }
                    }
                    acc
                })
                .collect();
            let mut it = per_chunk.into_iter();
            let mut merged = it.next()?;
            for rest in it {
                merged.merge(&rest);
            }
            Some((name.clone(), merged))
        })
        .collect()
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
    py: Python<'_>,
    columns: Vec<String>,
    batches: Vec<PyArrowType<RecordBatch>>,
    probs: Vec<f64>,
) -> PyResult<std::collections::HashMap<String, Vec<f64>>> {
    // A temporal sort key (Date/Timestamp) has no numeric KLL, so summarize it on its
    // order-preserving Int64 backing — the same representation the range partition compares
    // on — so a distributed `ORDER BY <date>` gets balanced boundaries instead of an empty
    // grid (one overloaded reducer). Numeric columns are untouched.
    let batches = temporal_cols_as_i64(&columns, &batches);
    // GIL released for the sketch build; see `column_stats` for why it dominates the cost.
    let merged = py.allow_threads(|| merge_column_stats(&columns, &batches));
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
    py: Python<'_>,
    columns: Vec<String>,
    batches: Vec<PyArrowType<RecordBatch>>,
    probs: Vec<f64>,
) -> PyResult<(
    std::collections::HashMap<String, std::collections::HashMap<String, Option<f64>>>,
    std::collections::HashMap<String, Vec<f64>>,
)> {
    // GIL released for the sketch build; see `column_stats` for why it dominates the cost.
    let merged = py.allow_threads(|| merge_column_stats(&columns, &batches));
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

/// The non-null values of an integer column, widened to `i64` — `None` for any other type.
///
/// Restricted to the plain integer types **on purpose**. Their `Utf8` rendering is the
/// decimal of the integer, so a survivor widened to `i64` and cast back renders exactly as
/// the original column would have. `Date32`/`Timestamp` are *also* integers underneath but
/// render as dates and times, and `Boolean` renders as `true`/`false`; widening those would
/// silently change the key the optimizer matches on, so they stay on the string path.
fn int_values(col: &ArrayRef) -> Option<Vec<i64>> {
    use arrow::array::*;

    macro_rules! widen {
        ($ty:ty) => {{
            let a = col.as_any().downcast_ref::<$ty>().expect("dtype matched");
            Some(
                a.iter()
                    .flatten()
                    .map(|v| i64::try_from(v).unwrap_or(i64::MIN))
                    .collect(),
            )
        }};
    }
    match col.data_type() {
        DataType::Int8 => widen!(Int8Array),
        DataType::Int16 => widen!(Int16Array),
        DataType::Int32 => widen!(Int32Array),
        DataType::Int64 => widen!(Int64Array),
        DataType::UInt8 => widen!(UInt8Array),
        DataType::UInt16 => widen!(UInt16Array),
        DataType::UInt32 => widen!(UInt32Array),
        DataType::UInt64 => widen!(UInt64Array),
        _ => None,
    }
}

/// Render integer heavy hitters to the strings the `Utf8` cast of `dtype` would have given.
///
/// The survivors go back through `cast` — the *same* conversion the string path applies to
/// the whole column — via a tiny array of the column's original type. That is what makes the
/// native-integer summary safe: the labels the optimizer and the shuffle's salting path key
/// on cannot drift from what casting the column produced, because they are produced by
/// casting.
fn render_int_hits(hits: &[(i64, u64)], dtype: &DataType) -> Vec<(String, u64)> {
    use arrow::array::{Array, Int64Array, StringArray};

    let keys: ArrayRef = std::sync::Arc::new(Int64Array::from(
        hits.iter().map(|(v, _)| *v).collect::<Vec<i64>>(),
    ));
    // Back to the column's own type, then to text — exactly the string path's conversion.
    let Ok(native) = cast(&keys, dtype) else {
        return Vec::new();
    };
    let Ok(text) = cast(&native, &DataType::Utf8) else {
        return Vec::new();
    };
    let Some(arr) = text.as_any().downcast_ref::<StringArray>() else {
        return Vec::new();
    };
    hits.iter()
        .enumerate()
        .filter(|(i, _)| arr.is_valid(*i))
        .map(|(i, (_, n))| (arr.value(i).to_string(), *n))
        .collect()
}

/// Heavy hitters (the Misra-Gries `FrequentItems` sketch) per column: the values
/// whose frequency exceeds `fraction` of the rows, with their estimated counts.
/// Kyber consumes this for skew detection (a hot join key → salting). Values are
/// rendered to strings (cast to Utf8) so any column type can be labelled; columns
/// that cannot cast are skipped. Mergeable in spirit — built across all batches.
///
/// **An integer column is summarized over its native values, not its rendering.** The
/// obvious implementation casts the whole column to `Utf8` and feeds one `String` per row
/// into the summary — for a 1M-row join key that is a million integers formatted into a
/// million heap strings, each hashed and then thrown away, since the summary only ever
/// monitors ~`capacity` (≈20) of them. Measured at **90 ns/value**, it made skew detection
/// cost more than the join it was protecting. Integers are hashed as `i64` instead, and only
/// the surviving handful are rendered — through the *same* `cast`, applied to a tiny array
/// of the survivors' original type, so the strings the optimizer and the shuffle's salting
/// path match on are byte-identical to what this always produced.
#[pyfunction]
pub(crate) fn heavy_hitters(
    columns: Vec<String>,
    batches: Vec<PyArrowType<RecordBatch>>,
    fraction: f64,
) -> PyResult<std::collections::HashMap<String, Vec<(String, u64)>>> {
    // Misra-Gries capacity: 1/fraction guarantees all keys above `fraction` survive.
    let capacity = ((1.0 / fraction).ceil() as usize).max(1);
    let mut ints: std::collections::HashMap<String, (DataType, bc_sketches::FrequentItems<i64>)> =
        std::collections::HashMap::new();
    let mut strs: std::collections::HashMap<String, bc_sketches::FrequentItems<String>> =
        std::collections::HashMap::new();
    for batch in &batches {
        let b = &batch.0;
        for name in &columns {
            let Some(col) = b.column_by_name(name) else {
                continue;
            };
            if let Some(vals) = int_values(col) {
                let (_, fi) = ints
                    .entry(name.clone())
                    .or_insert_with(|| (col.data_type().clone(), FrequentItems::new(capacity)));
                for v in vals {
                    fi.add(v);
                }
                continue;
            }
            let Ok(s) = cast(col, &DataType::Utf8) else {
                continue;
            };
            let Some(arr) = s.as_any().downcast_ref::<arrow::array::StringArray>() else {
                continue;
            };
            let fi = strs
                .entry(name.clone())
                .or_insert_with(|| FrequentItems::new(capacity));
            // `add_ref`, not `add`: only the branch that takes a free slot needs to own the
            // key, so the row's `String` is built solely when it enters the summary.
            for i in 0..arr.len() {
                if arr.is_valid(i) {
                    fi.add_ref(arr.value(i));
                }
            }
        }
    }
    let mut out = std::collections::HashMap::new();
    for (name, fi) in strs {
        out.insert(name, fi.heavy_hitters(fraction));
    }
    for (name, (dtype, fi)) in ints {
        let hits = fi.heavy_hitters(fraction);
        out.insert(name, render_int_hits(&hits, &dtype));
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
