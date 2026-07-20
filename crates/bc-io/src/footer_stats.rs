//! Aggregate Parquet footer statistics across many files, natively.
//!
//! This is the metadata half of the scan: before a single data page is read, the planner
//! needs the dataset's row count, byte size, and per-column min/max/null-count so Kyber
//! can prune files, answer `count()`/`min()`/`max()` from metadata alone, and cost a join.
//!
//! Doing that in Python costs one pybind11 object construction per *column chunk*
//! (`meta.row_group(rg).column(ci).statistics.min`), which is O(files x row_groups x
//! columns) FFI calls on the driver, single-threaded and under the GIL. On a 200-file
//! dataset with 20 row-groups and 30 columns that is 120,000 chunks -> ~750 ms of pure
//! interpreter overhead sitting on top of ~95 ms of actual footer I/O. The footers were
//! never the cost; walking them from Python was.
//!
//! Here the same walk is a native loop over `ParquetMetaData` that the reader has usually
//! already parsed and cached ([`crate::load_metadata_cached`]), so a warm call performs no
//! I/O at all. Typed min/max come from `StatisticsConverter`, which maps Parquet physical
//! statistics onto the file's *Arrow* type (decimals, timestamps, dictionaries, and
//! schema-evolved columns included) rather than re-deriving that mapping by hand.
//!
//! The result crosses to Python as Arrow: a 2-row `RecordBatch` (row 0 = min, row 1 = max)
//! whose schema is the columns themselves, so every bound keeps its exact type with no
//! per-value conversion. Semantics match the Python accumulator it replaces exactly,
//! including the NaN-poisoning rule below — the Python path remains as the fallback for
//! any file this cannot read, so the two must agree.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float32Array, Float64Array, RecordBatch};
use arrow::compute::{concat, take};
use arrow::datatypes::{DataType, Field, Schema};
use parquet::arrow::arrow_reader::statistics::StatisticsConverter;
use parquet::file::metadata::RowGroupMetaData;

use crate::IoError;

/// Per-column footer facts that are *not* the typed bounds (those travel as Arrow).
///
/// Field-for-field the Python `_ColAcc` this replaces, so `_finalize_columns` can consume
/// it unchanged. `count` (summed `num_values`) is deliberately absent: the Python
/// accumulator maintains it and no consumer reads it.
#[derive(Debug, Clone)]
pub struct ColumnFooterStats {
    /// Column name, as it appears in the Arrow schema.
    pub name: String,
    /// Whether *any* chunk carried a statistics block. A column with none is dropped by
    /// the caller rather than reported with unknown bounds.
    pub has_stats: bool,
    /// Summed null count over the chunks that reported one. Meaningful only when
    /// `null_known` — otherwise it is a partial sum and must not be published.
    pub null_count: u64,
    /// True only if *every* chunk reported a null count. One chunk without one makes the
    /// total a lower bound, which would be a wrong answer for `count(col)`, not a loose one.
    pub null_known: bool,
    /// A NaN appeared in a bound. Parquet omits NaN from min/max while SQL ranks it
    /// greatest, so such a bound is unordered: the caller discards min *and* max rather
    /// than pruning on a value that does not mean what it appears to mean.
    pub nan_seen: bool,
    /// Parquet's (estimated) distinct count, last one seen. The caller publishes it only
    /// for a single-row-group source, since per-chunk counts are not additive.
    pub distinct: Option<i64>,
}

/// Everything the planner reads from a Parquet dataset's footers.
#[derive(Debug)]
pub struct FooterStats {
    /// Exact total row count across every file whose footer was read.
    pub total_rows: i64,
    /// Summed uncompressed row-group byte size.
    pub total_bytes: i64,
    /// Total row groups, which decides whether `distinct` is publishable.
    pub row_group_count: i64,
    /// How many files' footers were read. Fewer than requested means some were skipped;
    /// the caller must not report an exact row count when it disagrees with the file count.
    pub files_read: usize,
    /// Per-column non-bound facts, in schema order.
    pub columns: Vec<ColumnFooterStats>,
    /// Whether every row group of every file declares the *same* leading sorting column,
    /// ascending and nulls-last — the necessary precondition for claiming the dataset is
    /// globally sorted, and nothing more.
    ///
    /// Deliberately not the claim itself. `SourceStatistics.sorted_by` lets Kyber *delete*
    /// a `Sort`, so getting it wrong reorders a user's rows silently rather than merely
    /// costing time, and proving it also requires that row groups are ordered within each
    /// file and files ordered across the dataset. That proof stays in the one already-tested
    /// implementation (`io.stats.sortedness`); this flag exists so the caller can tell,
    /// cheaply, whether it needs to run it at all. False — the overwhelmingly common case,
    /// since most writers record no `sorting_columns` — means no sort claim is possible and
    /// the caller can skip the proof entirely.
    pub sort_declared: bool,
    /// Two rows — row 0 = min, row 1 = max — with one field per entry in `columns`, typed
    /// as that column is typed. A null bound means "unknown", never "no value".
    pub bounds: RecordBatch,
}

/// A column's bounds accumulated across files, as raw per-row-group arrays awaiting one
/// reduction.
///
/// Each file's arrays are appended whole rather than collapsed on arrival. Collapsing per
/// file looks cheaper — it keeps the accumulator at one value per file — but it costs a
/// sort-and-take per (file, column), and on a many-small-files dataset those are tiny
/// arrays where the Arrow kernel's fixed overhead dwarfs the handful of elements it
/// compares: 1,000 files x 30 columns meant 60,000 kernel calls to reduce 5 values each.
/// Batching them into one reduction per column cut this pass roughly in half.
///
/// Memory stays bounded by [`BOUND_COLLAPSE_AT`]: once a column has accumulated that many
/// bounds they are folded to one, so the accumulator never grows with the dataset even
/// though the common case pays a single reduction. Keeping bounds as arrays rather than
/// native scalars is what lets one code path handle every Arrow type without a match over
/// all of them.
struct BoundAcc {
    field: Field,
    mins: Vec<ArrayRef>,
    maxes: Vec<ArrayRef>,
    /// Accumulated element count, to decide when an intermediate collapse is due.
    pending: usize,
}

/// How many accumulated bounds a column may hold before they are folded to one.
///
/// High enough that an ordinary dataset reduces exactly once at the end, low enough that a
/// pathological one (hundreds of thousands of files) cannot grow the accumulator without
/// limit. The reduction is associative, so folding early changes cost, never the answer.
const BOUND_COLLAPSE_AT: usize = 8192;

/// Reduce `array` to a length-1 array holding its smallest (or largest) non-null value.
///
/// Generic over the Arrow type by construction: it sorts for the single extreme rather
/// than matching on `DataType`, so decimals, timestamps, strings, and binary all reduce
/// through the same code. `nulls_first: false` on an ascending sort puts unknowns last,
/// so index 0 is the smallest *known* bound; an all-null (or unsortable) input yields
/// `None`, which the caller reads as "unknown" and therefore never prunes on.
fn reduce_extreme(array: &ArrayRef, descending: bool) -> Option<ArrayRef> {
    if array.is_empty() || array.null_count() == array.len() {
        return None;
    }
    let opts = arrow::compute::SortOptions {
        descending,
        nulls_first: false,
    };
    let idx = arrow::compute::sort_to_indices(array, Some(opts), Some(1)).ok()?;
    if idx.is_empty() {
        return None;
    }
    let picked = take(array.as_ref(), &idx, None).ok()?;
    if picked.is_null(0) {
        return None;
    }
    Some(picked)
}

/// Whether any bound in `array` is NaN.
///
/// Only floats can be, and the array is one entry per row group, so this is a cheap native
/// scan. It exists because a NaN bound is not merely imprecise — Parquet leaves NaN out of
/// min/max, so a "max" computed alongside one is the largest non-NaN value while SQL says
/// NaN is greatest. Pruning on it would drop matching rows.
fn has_nan(array: &ArrayRef) -> bool {
    match array.data_type() {
        DataType::Float64 => array
            .as_any()
            .downcast_ref::<Float64Array>()
            .is_some_and(|a| a.iter().flatten().any(f64::is_nan)),
        DataType::Float32 => array
            .as_any()
            .downcast_ref::<Float32Array>()
            .is_some_and(|a| a.iter().flatten().any(f32::is_nan)),
        _ => false,
    }
}

/// Fold one file's row-group statistics into the running accumulators.
///
/// `arrow_schema`/`parquet_schema` are *this file's*, not a dataset-wide schema: a
/// `StatisticsConverter` validates the Arrow field against the Parquet column it matched,
/// so building it per file is what lets a schema-evolved dataset (a column added, widened,
/// or reordered between files) contribute the files where the column is present instead of
/// failing the whole pass.
fn fold_file(
    arrow_schema: &Schema,
    parquet_schema: &parquet::schema::types::SchemaDescriptor,
    row_groups: &[RowGroupMetaData],
    order: &mut Vec<String>,
    cols: &mut HashMap<String, ColumnFooterStats>,
    bounds: &mut HashMap<String, BoundAcc>,
) {
    for (field_idx, field) in arrow_schema.fields().iter().enumerate() {
        let name = field.name();
        let entry = cols.entry(name.clone()).or_insert_with(|| {
            order.push(name.clone());
            ColumnFooterStats {
                name: name.clone(),
                has_stats: false,
                null_count: 0,
                null_known: true,
                nan_seen: false,
                distinct: None,
            }
        });

        // The non-bound facts come from the chunk statistics directly. `StatisticsConverter`
        // reports a missing statistics block and a present-but-null bound identically, and
        // the two differ here: the first makes the null count unknowable, the second does not.
        let parquet_idx = parquet_column_index(parquet_schema, arrow_schema, name, field_idx);
        if let Some(pidx) = parquet_idx {
            for rg in row_groups {
                match rg.column(pidx).statistics() {
                    Some(stats) => {
                        entry.has_stats = true;
                        match stats.null_count_opt() {
                            Some(n) => entry.null_count += n,
                            None => entry.null_known = false,
                        }
                        if let Some(d) = stats.distinct_count_opt() {
                            entry.distinct = Some(d as i64);
                        }
                    }
                    None => entry.null_known = false,
                }
            }
        }

        // Typed bounds via the converter, appended whole for one batched reduction later.
        let Ok(conv) = StatisticsConverter::try_new(name, arrow_schema, parquet_schema) else {
            continue;
        };
        let (Ok(mins), Ok(maxes)) = (
            conv.row_group_mins(row_groups.iter()),
            conv.row_group_maxes(row_groups.iter()),
        ) else {
            continue;
        };
        if has_nan(&mins) || has_nan(&maxes) {
            entry.nan_seen = true;
        }
        let acc = bounds.entry(name.clone()).or_insert_with(|| BoundAcc {
            field: Field::new(name, mins.data_type().clone(), true),
            mins: Vec::new(),
            maxes: Vec::new(),
            pending: 0,
        });
        // A dataset that widens a column mid-way (Int32 -> Int64) would otherwise fail the
        // final concat. Skip the divergent file's bounds rather than lose the column: a
        // narrower bound set still prunes correctly, it just prunes less.
        if acc.field.data_type() != mins.data_type() {
            continue;
        }
        acc.pending += mins.len();
        acc.mins.push(mins);
        acc.maxes.push(maxes);
        if acc.pending >= BOUND_COLLAPSE_AT {
            let dt = acc.field.data_type().clone();
            acc.mins = vec![fold_bound(&acc.mins, false, &dt)];
            acc.maxes = vec![fold_bound(&acc.maxes, true, &dt)];
            acc.pending = 1;
        }
    }
}

/// Collapse the per-file bounds into the 2-row (min, max) batch that crosses to Python.
fn build_bounds(
    order: &[String],
    cols: &HashMap<String, ColumnFooterStats>,
    bounds: &mut HashMap<String, BoundAcc>,
) -> Result<RecordBatch, IoError> {
    let mut fields: Vec<Field> = Vec::new();
    let mut arrays: Vec<ArrayRef> = Vec::new();
    for name in order {
        // Columns the caller will drop anyway (no statistics at all) carry no bounds.
        if !cols.get(name).is_some_and(|c| c.has_stats) {
            continue;
        }
        let Some(acc) = bounds.get(name) else {
            continue;
        };
        let dt = acc.field.data_type().clone();
        let lo = fold_bound(&acc.mins, false, &dt);
        let hi = fold_bound(&acc.maxes, true, &dt);
        // Row 0 = min, row 1 = max, in one array of this column's own type.
        let joined =
            concat(&[lo.as_ref(), hi.as_ref()]).map_err(|e| IoError::Arrow(e.to_string()))?;
        fields.push(Field::new(name, dt, true));
        arrays.push(joined);
    }
    let schema = Arc::new(Schema::new(fields));
    if arrays.is_empty() {
        // No bounds at all is a valid answer (row counts still are); an explicit 2-row
        // empty-schema batch keeps the "row 0 = min, row 1 = max" contract unconditional.
        return RecordBatch::try_new_with_options(
            schema,
            vec![],
            &arrow::record_batch::RecordBatchOptions::new().with_row_count(Some(2)),
        )
        .map_err(|e| IoError::Arrow(e.to_string()));
    }
    RecordBatch::try_new(schema, arrays).map_err(|e| IoError::Arrow(e.to_string()))
}

/// Reduce the per-file extremes to one, as a length-1 array (null when nothing is known).
fn fold_bound(parts: &[ArrayRef], descending: bool, dt: &DataType) -> ArrayRef {
    let null = || arrow::array::new_null_array(dt, 1);
    if parts.is_empty() {
        return null();
    }
    let refs: Vec<&dyn Array> = parts.iter().map(|a| a.as_ref()).collect();
    let Ok(all) = concat(&refs) else {
        return null();
    };
    reduce_extreme(&all, descending).unwrap_or_else(null)
}

/// Per-**file** bounds for `columns`, in the add-action layout, built natively.
///
/// The sibling of [`parquet_footer_stats`]: that one collapses a dataset to one row of
/// bounds for pruning a scan, this one keeps a row per file so a copy-on-write `MERGE` can
/// skip whole files on an ordinary directory with no transaction log
/// (`io.stats.key_pruning`). Layout matches what a lakehouse log already publishes —
/// `path | num_records | min.<col> | max.<col> | null_count.<col>` — so one consumer
/// prunes both.
///
/// Footers come from the same validated cache the statistics pass fills, so a query that
/// does both reads each footer once rather than twice; on 1,000 local files the footer
/// fetch was ~95 % of the Python implementation's cost, and it was paying it a second time.
///
/// A file whose footer is unreadable, or which records no statistic for a column, yields
/// NULL bounds — which every consumer must read as *unknown* (keep the file), never as
/// *no match*.
pub fn parquet_file_manifest(uris: &[String], columns: &[String]) -> Result<RecordBatch, IoError> {
    let metas = crate::load_metadata_many(uris)?;
    let mut paths: Vec<Option<&str>> = Vec::with_capacity(uris.len());
    let mut rows: Vec<Option<i64>> = Vec::with_capacity(uris.len());
    // One entry per file per column, concatenated once at the end so each column is built
    // with a single kernel call rather than one per file. `None` is an unknown bound, left
    // unmaterialized until build time: the file that would have named the column's type may
    // be the very one that was unreadable, so the null array cannot be made yet.
    let mut mins: HashMap<&str, Vec<Option<ArrayRef>>> = HashMap::new();
    let mut maxes: HashMap<&str, Vec<Option<ArrayRef>>> = HashMap::new();
    let mut nulls: HashMap<&str, Vec<Option<i64>>> = HashMap::new();
    let mut types: HashMap<&str, DataType> = HashMap::new();
    let mut saw_any = false;

    for (uri, meta) in uris.iter().zip(metas.iter()) {
        paths.push(Some(uri.as_str()));
        let Some(amd) = meta else {
            // Unreadable: a row of NULLs, which keeps the file rather than dropping it.
            // Every column vector must still advance, or the batch comes out ragged.
            rows.push(None);
            for c in columns {
                mins.entry(c).or_default().push(None);
                maxes.entry(c).or_default().push(None);
                nulls.entry(c).or_default().push(None);
            }
            continue;
        };
        saw_any = true;
        let md = amd.metadata();
        rows.push(Some(md.file_metadata().num_rows()));
        let row_groups = md.row_groups();
        for c in columns {
            let (mut lo, mut hi, mut nc) = (None, None, None);
            if let Ok(conv) = StatisticsConverter::try_new(c, amd.schema(), amd.parquet_schema()) {
                if let (Ok(lo_a), Ok(hi_a)) = (
                    conv.row_group_mins(row_groups.iter()),
                    conv.row_group_maxes(row_groups.iter()),
                ) {
                    types.entry(c).or_insert_with(|| lo_a.data_type().clone());
                    // EVERY row group must have contributed a bound. Reducing over only the
                    // row groups that did would produce a bound covering *part* of the file
                    // — which prunes away rows that are actually there. A single missing
                    // statistic makes the whole file's interval unknown, and unknown keeps
                    // the file. A NaN bound is unordered and poisons it the same way.
                    let complete = lo_a.null_count() == 0 && hi_a.null_count() == 0;
                    if complete && !has_nan(&lo_a) && !has_nan(&hi_a) {
                        lo = reduce_extreme(&lo_a, false);
                        hi = reduce_extreme(&hi_a, true);
                        // Tied to the bounds on purpose: the Python path this mirrors reports
                        // one `(min, max, null_count)` triple that is known or not as a unit,
                        // and a fast path that answers differently is a fast path that prunes
                        // differently. Decoupling them would be an improvement — an all-null
                        // column could then be skipped outright — but it is a semantic change
                        // to make deliberately in both paths, not a side effect of this one.
                        // A chunk with no null count contributes 0, matching that path.
                        nc = conv
                            .row_group_null_counts(row_groups.iter())
                            .ok()
                            .map(|counts| counts.iter().flatten().sum::<u64>() as i64);
                    }
                }
            }
            mins.entry(c).or_default().push(lo);
            maxes.entry(c).or_default().push(hi);
            nulls.entry(c).or_default().push(nc);
        }
    }
    if !saw_any {
        return Err(IoError::Arrow("no readable footer".into()));
    }
    build_manifest(&paths, &rows, columns, &mins, &maxes, &nulls, &types)
}

/// Assemble the manifest columns into the add-action-layout batch.
#[allow(clippy::too_many_arguments)]
fn build_manifest(
    paths: &[Option<&str>],
    rows: &[Option<i64>],
    columns: &[String],
    mins: &HashMap<&str, Vec<Option<ArrayRef>>>,
    maxes: &HashMap<&str, Vec<Option<ArrayRef>>>,
    nulls: &HashMap<&str, Vec<Option<i64>>>,
    types: &HashMap<&str, DataType>,
) -> Result<RecordBatch, IoError> {
    use arrow::array::{Int64Array, StringArray};

    let mut fields = vec![
        Field::new("path", DataType::Utf8, false),
        Field::new("num_records", DataType::Int64, true),
    ];
    let mut arrays: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(paths.to_vec())),
        Arc::new(Int64Array::from(rows.to_vec())),
    ];
    // Unknown bounds become null arrays only here, where the column's type is finally known
    // from whichever file did describe it (`Null` if none did).
    let join = |parts: &Vec<Option<ArrayRef>>, dt: &DataType| -> Result<ArrayRef, IoError> {
        let filled: Vec<ArrayRef> = parts
            .iter()
            .map(|p| {
                p.clone()
                    .unwrap_or_else(|| arrow::array::new_null_array(dt, 1))
            })
            .collect();
        let refs: Vec<&dyn Array> = filled.iter().map(|a| a.as_ref()).collect();
        concat(&refs).map_err(|e| IoError::Arrow(e.to_string()))
    };
    for c in columns {
        // A column no file described has no type to give its bounds; publish the null
        // counts alone rather than inventing one, so the consumer still sees "unknown".
        let dt = types.get(c.as_str()).cloned().unwrap_or(DataType::Null);
        if let (Some(lo), Some(hi)) = (mins.get(c.as_str()), maxes.get(c.as_str())) {
            fields.push(Field::new(format!("min.{c}"), dt.clone(), true));
            arrays.push(join(lo, &dt)?);
            fields.push(Field::new(format!("max.{c}"), dt.clone(), true));
            arrays.push(join(hi, &dt)?);
        }
        if let Some(nc) = nulls.get(c.as_str()) {
            fields.push(Field::new(format!("null_count.{c}"), DataType::Int64, true));
            arrays.push(Arc::new(Int64Array::from(nc.clone())));
        }
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)
        .map_err(|e| IoError::Arrow(e.to_string()))
}

/// Aggregate footer statistics over `uris`, reading each file's footer at most once.
///
/// Footers load concurrently through the shared runtime and the validated metadata cache,
/// so a file the reader has already touched costs no I/O. A file whose footer cannot be
/// read is skipped, not fatal — matching the Python path's best-effort contract — and
/// `files_read` reports the shortfall so the caller can decline to claim an exact count.
pub fn parquet_footer_stats(uris: &[String]) -> Result<FooterStats, IoError> {
    let metas = crate::load_metadata_many(uris)?;

    let mut order: Vec<String> = Vec::new();
    let mut cols: HashMap<String, ColumnFooterStats> = HashMap::new();
    let mut bounds: HashMap<String, BoundAcc> = HashMap::new();
    let (mut total_rows, mut total_bytes, mut row_group_count, mut files_read) =
        (0i64, 0i64, 0i64, 0);
    // A gap in the footers breaks any sort proof — the missing file could hold anything —
    // so an unreadable file forces the flag off, exactly as `proved_sorted_by` bails on one.
    let mut sort_key: Option<i32> = None;
    let mut sort_possible = metas.iter().all(Option::is_some) && !metas.is_empty();

    for amd in metas.iter().flatten() {
        files_read += 1;
        let md = amd.metadata();
        total_rows += md.file_metadata().num_rows();
        let row_groups = md.row_groups();
        row_group_count += row_groups.len() as i64;
        for rg in row_groups {
            total_bytes += rg.total_byte_size();
            if sort_possible {
                sort_possible = agrees_on_sort_key(rg, &mut sort_key);
            }
        }
        fold_file(
            amd.schema(),
            amd.parquet_schema(),
            row_groups,
            &mut order,
            &mut cols,
            &mut bounds,
        );
    }

    let batch = build_bounds(&order, &cols, &mut bounds)?;
    let columns = order
        .iter()
        .filter_map(|n| cols.get(n).cloned())
        .filter(|c| c.has_stats)
        .collect();
    Ok(FooterStats {
        total_rows,
        total_bytes,
        row_group_count,
        files_read,
        columns,
        bounds: batch,
        sort_declared: sort_possible && sort_key.is_some(),
    })
}

/// Whether `rg` declares the same ascending, nulls-last leading sorting column as its
/// predecessors, recording it on the first row group seen.
///
/// Descending and nulls-first are *different* orderings than the canonical one
/// `sorted_by` denotes, so they are refused rather than reinterpreted — reinterpreting
/// them is precisely the wrong-order bug this whole path is careful about.
fn agrees_on_sort_key(rg: &RowGroupMetaData, sort_key: &mut Option<i32>) -> bool {
    let Some(cols) = rg.sorting_columns() else {
        return false;
    };
    let Some(first) = cols.first() else {
        return false;
    };
    if first.descending || first.nulls_first {
        return false;
    }
    match sort_key {
        Some(existing) => *existing == first.column_idx,
        None => {
            *sort_key = Some(first.column_idx);
            true
        }
    }
}

/// Index of `name`'s leaf column in the Parquet schema, or `None` if it has none.
///
/// Falls back to positional matching only when the schemas are the same width and the leaf
/// at that position carries the same name — a flat file's common case. A nested or evolved
/// schema that does not line up yields `None`, which costs the column its null count but
/// never mis-attributes another column's statistics to it.
fn parquet_column_index(
    parquet_schema: &parquet::schema::types::SchemaDescriptor,
    _arrow_schema: &Schema,
    name: &str,
    field_idx: usize,
) -> Option<usize> {
    if field_idx < parquet_schema.num_columns() {
        let col = parquet_schema.column(field_idx);
        if col.path().parts().len() == 1 && col.path().parts()[0] == name {
            return Some(field_idx);
        }
    }
    (0..parquet_schema.num_columns()).find(|&i| {
        let col = parquet_schema.column(i);
        let p = col.path();
        p.parts().len() == 1 && p.parts()[0] == name
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, Int64Array, StringArray};
    use arrow::datatypes::DataType;
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;

    /// Write `batches` to a temp Parquet file with a small row-group size and return its path.
    fn write_file(dir: &std::path::Path, name: &str, batch: &RecordBatch, rg: usize) -> String {
        let path = dir.join(name);
        let file = std::fs::File::create(&path).unwrap();
        let props = WriterProperties::builder()
            .set_max_row_group_size(rg)
            .build();
        let mut w = ArrowWriter::try_new(file, batch.schema(), Some(props)).unwrap();
        w.write(batch).unwrap();
        w.close().unwrap();
        path.to_string_lossy().into_owned()
    }

    fn tmpdir() -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("bcio_fs_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&d);
        d
    }

    fn int_batch(vals: Vec<i64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, true)]));
        RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vals))]).unwrap()
    }

    /// The bounds are the global extremes across every row group of every file, and the
    /// row count is exact — this is the contract Kyber prunes and answers `count()` on.
    #[test]
    fn aggregates_bounds_and_rows_across_files_and_row_groups() {
        let d = tmpdir();
        let f1 = write_file(&d, "a1.parquet", &int_batch((0..100).collect()), 10);
        let f2 = write_file(&d, "a2.parquet", &int_batch((500..600).collect()), 10);
        let st = parquet_footer_stats(&[f1, f2]).unwrap();

        assert_eq!(st.total_rows, 200);
        assert_eq!(st.row_group_count, 20);
        assert_eq!(st.files_read, 2);
        assert_eq!(st.columns.len(), 1);
        assert!(st.columns[0].has_stats);
        assert_eq!(st.columns[0].null_count, 0);
        assert!(st.columns[0].null_known);

        assert_eq!(st.bounds.num_rows(), 2);
        let a = st
            .bounds
            .column_by_name("a")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(a.value(0), 0, "global min");
        assert_eq!(a.value(1), 599, "global max");
    }

    /// Nulls are counted exactly and the count stays known, since Parquet records a null
    /// count per chunk for every type.
    #[test]
    fn sums_null_counts_exactly() {
        let d = tmpdir();
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, true)]));
        let vals: Vec<Option<i64>> = (0..100)
            .map(|i| if i % 4 == 0 { None } else { Some(i) })
            .collect();
        let b = RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vals))]).unwrap();
        let f = write_file(&d, "nulls.parquet", &b, 10);
        let st = parquet_footer_stats(&[f]).unwrap();
        assert_eq!(st.columns[0].null_count, 25);
        assert!(st.columns[0].null_known);
    }

    /// A NaN *bound* is poison: SQL ranks NaN greatest while Parquet leaves it out of
    /// min/max, so a bound computed alongside one does not mean what it appears to.
    ///
    /// Tested on the detector directly, because a spec-conforming writer (arrow-rs and
    /// pyarrow both) already excludes NaN, so no file written here can produce one. The
    /// guard exists for foreign writers that do emit it — and testing it through a local
    /// round trip would assert the writer's behavior while leaving this code uncovered.
    #[test]
    fn detects_nan_in_a_bounds_array() {
        let poisoned: ArrayRef = Arc::new(Float64Array::from(vec![Some(1.0), Some(f64::NAN)]));
        let clean: ArrayRef = Arc::new(Float64Array::from(vec![Some(1.0), None, Some(3.0)]));
        let ints: ArrayRef = Arc::new(Int64Array::from(vec![1, 2]));
        assert!(has_nan(&poisoned));
        assert!(!has_nan(&clean), "a null bound is unknown, not poisoned");
        assert!(!has_nan(&ints), "a non-float column can never be poisoned");
    }

    /// The floats a spec-conforming writer *does* record still aggregate correctly: NaN is
    /// absent from the footer, so the bounds are the extremes of the real values.
    #[test]
    fn aggregates_float_bounds_ignoring_nan_rows() {
        let d = tmpdir();
        let schema = Arc::new(Schema::new(vec![Field::new("f", DataType::Float64, true)]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![3.5, f64::NAN, 1.5]))],
        )
        .unwrap();
        let f = write_file(&d, "nan.parquet", &b, 10);
        let st = parquet_footer_stats(&[f]).unwrap();
        let vals = st
            .bounds
            .column_by_name("f")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(vals.value(0), 1.5);
        assert_eq!(vals.value(1), 3.5);
    }

    /// Bounds keep the column's own Arrow type across the boundary — a string column comes
    /// back as strings, not as a stringified scalar.
    #[test]
    fn preserves_arrow_types_for_non_numeric_columns() {
        let d = tmpdir();
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from(vec!["pear", "apple", "quince"]))],
        )
        .unwrap();
        let f = write_file(&d, "s.parquet", &b, 2);
        let st = parquet_footer_stats(&[f]).unwrap();
        let s = st
            .bounds
            .column_by_name("s")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(s.value(0), "apple");
        assert_eq!(s.value(1), "quince");
    }

    /// An unreadable file is skipped, not fatal, and `files_read` reports the shortfall so
    /// the caller can decline to claim an exact row count.
    #[test]
    fn skips_unreadable_files_and_reports_the_shortfall() {
        let d = tmpdir();
        let good = write_file(&d, "good.parquet", &int_batch((0..10).collect()), 5);
        let st = parquet_footer_stats(&[good, "/nonexistent/nope.parquet".to_string()]).unwrap();
        assert_eq!(st.files_read, 1);
        assert_eq!(st.total_rows, 10);
    }

    /// An ordinary file declares no `sorting_columns`, so no sort claim is even possible —
    /// the caller skips the (expensive, dangerous) proof entirely. This is the common case.
    #[test]
    fn reports_no_sort_declaration_for_an_ordinary_file() {
        let d = tmpdir();
        let f = write_file(&d, "plain.parquet", &int_batch((0..50).collect()), 10);
        assert!(!parquet_footer_stats(&[f]).unwrap().sort_declared);
    }

    /// A writer that *does* record an ascending, nulls-last sorting column sets the flag,
    /// which routes the caller to the real proof rather than to a claim.
    #[test]
    fn reports_a_sort_declaration_when_the_writer_records_one() {
        use parquet::format::SortingColumn;
        let d = tmpdir();
        let path = d.join("sorted.parquet");
        let batch = int_batch((0..50).collect());
        let props = WriterProperties::builder()
            .set_max_row_group_size(10)
            .set_sorting_columns(Some(vec![SortingColumn::new(0, false, false)]))
            .build();
        let file = std::fs::File::create(&path).unwrap();
        let mut w = ArrowWriter::try_new(file, batch.schema(), Some(props)).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
        let st = parquet_footer_stats(&[path.to_string_lossy().into_owned()]).unwrap();
        assert!(st.sort_declared);
    }

    /// A descending declaration is a *different* ordering than `sorted_by` denotes, so it
    /// must not set the flag — reinterpreting it is the wrong-order bug.
    #[test]
    fn refuses_a_descending_sort_declaration() {
        use parquet::format::SortingColumn;
        let d = tmpdir();
        let path = d.join("desc.parquet");
        let batch = int_batch((0..50).rev().collect());
        let props = WriterProperties::builder()
            .set_max_row_group_size(10)
            .set_sorting_columns(Some(vec![SortingColumn::new(0, true, false)]))
            .build();
        let file = std::fs::File::create(&path).unwrap();
        let mut w = ArrowWriter::try_new(file, batch.schema(), Some(props)).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
        let st = parquet_footer_stats(&[path.to_string_lossy().into_owned()]).unwrap();
        assert!(!st.sort_declared);
    }

    /// The intermediate collapse at `BOUND_COLLAPSE_AT` is only sound because folding is
    /// associative — folding some bounds early and the rest later must give what folding
    /// them all at once gives. That is asserted here on `fold_bound` directly.
    ///
    /// Deliberately *not* asserted by writing a file with `BOUND_COLLAPSE_AT` row groups:
    /// `ArrowWriter` overflows its stack closing a file with that many (~8.7k) row groups
    /// in a debug build, so such a test would fail on the writer while proving nothing
    /// about this code. Reading such a file is fine — verified separately against a
    /// pyarrow-written one.
    #[test]
    fn folding_bounds_is_associative() {
        let dt = DataType::Int64;
        let parts: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from(vec![Some(5), Some(9), None])),
            Arc::new(Int64Array::from(vec![Some(2), Some(7)])),
            Arc::new(Int64Array::from(vec![Some(11), Some(3)])),
        ];
        let all_at_once_min = fold_bound(&parts, false, &dt);
        let all_at_once_max = fold_bound(&parts, true, &dt);

        // Collapse the first two, then fold the survivor with the third — what the
        // threshold does mid-pass.
        let early_min = fold_bound(&parts[..2].to_vec(), false, &dt);
        let early_max = fold_bound(&parts[..2].to_vec(), true, &dt);
        let staged_min = fold_bound(&vec![early_min, parts[2].clone()], false, &dt);
        let staged_max = fold_bound(&vec![early_max, parts[2].clone()], true, &dt);

        assert_eq!(&staged_min, &all_at_once_min);
        assert_eq!(&staged_max, &all_at_once_max);
        let as_i64 = |a: &ArrayRef| a.as_any().downcast_ref::<Int64Array>().unwrap().value(0);
        assert_eq!(as_i64(&all_at_once_min), 2);
        assert_eq!(as_i64(&all_at_once_max), 11);
    }

    /// The manifest keeps one row per file, in URI order, with that file's own bounds —
    /// which is what lets a `MERGE` skip files a key cannot be in.
    #[test]
    fn manifest_reports_per_file_bounds_in_order() {
        let d = tmpdir();
        let f1 = write_file(&d, "m1.parquet", &int_batch((0..100).collect()), 10);
        let f2 = write_file(&d, "m2.parquet", &int_batch((500..600).collect()), 10);
        let uris = vec![f1.clone(), f2.clone()];
        let m = parquet_file_manifest(&uris, &["a".to_string()]).unwrap();

        assert_eq!(m.num_rows(), 2);
        let paths = m
            .column_by_name("path")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .unwrap();
        assert_eq!(paths.value(0), f1);
        assert_eq!(paths.value(1), f2);

        let rows = m
            .column_by_name("num_records")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .unwrap();
        assert_eq!((rows.value(0), rows.value(1)), (100, 100));

        let lo = m
            .column_by_name("min.a")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let hi = m
            .column_by_name("max.a")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        // Per file, NOT collapsed across the dataset — that distinction is the whole point.
        assert_eq!((lo.value(0), hi.value(0)), (0, 99));
        assert_eq!((lo.value(1), hi.value(1)), (500, 599));

        let nc = m
            .column_by_name("null_count.a")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .unwrap();
        assert_eq!((nc.value(0), nc.value(1)), (0, 0));
    }

    /// An unreadable file yields NULL bounds and keeps its row. A dropped row would shift
    /// every later file's bounds onto the wrong path; a NULL bound means "unknown", which
    /// every consumer must treat as keep-the-file.
    #[test]
    fn manifest_keeps_a_row_with_null_bounds_for_an_unreadable_file() {
        let d = tmpdir();
        let good = write_file(&d, "mg.parquet", &int_batch((7..20).collect()), 5);
        let uris = vec!["/nonexistent/gone.parquet".to_string(), good];
        let m = parquet_file_manifest(&uris, &["a".to_string()]).unwrap();

        assert_eq!(m.num_rows(), 2, "the unreadable file keeps its row");
        let lo = m.column_by_name("min.a").unwrap();
        assert!(lo.is_null(0), "unknown, not a bound");
        assert!(!lo.is_null(1));
        assert!(m.column_by_name("num_records").unwrap().is_null(0));
    }

    /// A file whose column has NO usable bounds must report unknown, not a partial bound.
    ///
    /// An all-null column records no min/max, so reducing over "the row groups that had
    /// one" would describe part of the file and prune away rows that are really there.
    #[test]
    fn manifest_reports_unknown_when_a_column_has_no_bounds() {
        use arrow::array::Int64Array;
        let d = tmpdir();
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, true)]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(
                vec![None, None, None, None] as Vec<Option<i64>>
            ))],
        )
        .unwrap();
        let f = write_file(&d, "allnull.parquet", &b, 2);
        let m = parquet_file_manifest(&[f], &["a".to_string()]).unwrap();
        assert!(m.column_by_name("min.a").unwrap().is_null(0));
        assert!(m.column_by_name("max.a").unwrap().is_null(0));
        // Tied to the bounds, matching the Python path it must not diverge from.
        assert!(m.column_by_name("null_count.a").unwrap().is_null(0));
    }

    /// A column no file carries yields NULL bounds rather than an error or a wrong type.
    #[test]
    fn manifest_handles_an_absent_column() {
        let d = tmpdir();
        let f = write_file(&d, "mabs.parquet", &int_batch((0..10).collect()), 5);
        let m = parquet_file_manifest(&[f], &["not_here".to_string()]).unwrap();
        assert_eq!(m.num_rows(), 1);
        assert!(m.column_by_name("null_count.not_here").unwrap().is_null(0));
    }

    /// Empty input is a valid, empty answer rather than an error.
    #[test]
    fn handles_empty_input() {
        let st = parquet_footer_stats(&[]).unwrap();
        assert_eq!(st.total_rows, 0);
        assert_eq!(st.files_read, 0);
        assert!(st.columns.is_empty());
        assert_eq!(st.bounds.num_rows(), 2);
    }
}
