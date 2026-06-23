//! Bloom-filter FFI for the distributed runtime join reduction.
//!
//! `build_key_bloom` builds a bloom over a side's join keys, `merge_blooms` folds
//! per-partition blooms into one, and `bloom_filter_batches` drops rows whose key
//! can't be on the other side. Keys are Arrow row-encoded so equal *values* map
//! identically regardless of which side built the bloom — the membership test thus
//! agrees with the join's equality, and (no false negatives) only ever removes
//! provably-non-matching rows.

use arrow::array::{
    Array, ArrayRef, BooleanArray, Int64Array, LargeStringArray, RecordBatch, StringArray,
};
use arrow::compute::filter_record_batch;
use arrow::datatypes::DataType;
use arrow::row::{RowConverter, SortField};
use arrow_pyarrow::PyArrowType;
use bc_sketches::{BloomFilter, Mergeable};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::{to_pyerr, unwrap_batches};

/// FNV-1a 64-bit — a tiny portable hash. Computed identically by the pure-Python
/// `BloomIndex` reader (`plan/bloom_index.py`), so a column bloom built here can be
/// probed in the optimizer without calling the engine.
fn fnv1a_64(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

/// Build a data-skipping bloom over one column's values, for the optimizer to prune
/// equality/`IN` predicates a value can't satisfy. Hashes each non-null value's
/// canonical bytes (Int64 → signed 8-byte LE, Utf8 → bytes) with FNV-1a so the
/// pure-Python `BloomIndex` agrees. Returns `None` for an unindexable column type
/// (only Int64 and Utf8/LargeUtf8 are indexed; `unwrap_batches` widens narrow ints
/// to Int64 first), so non-indexed columns simply have no skip index.
#[pyfunction]
pub fn build_column_bloom(
    batches: Vec<PyArrowType<RecordBatch>>,
    col_index: usize,
    expected_items: u64,
) -> PyResult<Option<Vec<u8>>> {
    let batches = unwrap_batches(batches);
    let mut bloom = BloomFilter::with_params(expected_items, 0.01);
    let mut indexed = false;
    for batch in &batches {
        if col_index >= batch.num_columns() {
            continue;
        }
        let col = batch.column(col_index);
        match col.data_type() {
            DataType::Int64 => {
                let a = col.as_any().downcast_ref::<Int64Array>().unwrap();
                for i in 0..a.len() {
                    if a.is_valid(i) {
                        bloom.add_hash(fnv1a_64(&a.value(i).to_le_bytes()));
                    }
                }
                indexed = true;
            }
            DataType::Utf8 => {
                let a = col.as_any().downcast_ref::<StringArray>().unwrap();
                for i in 0..a.len() {
                    if a.is_valid(i) {
                        bloom.add_hash(fnv1a_64(a.value(i).as_bytes()));
                    }
                }
                indexed = true;
            }
            DataType::LargeUtf8 => {
                let a = col.as_any().downcast_ref::<LargeStringArray>().unwrap();
                for i in 0..a.len() {
                    if a.is_valid(i) {
                        bloom.add_hash(fnv1a_64(a.value(i).as_bytes()));
                    }
                }
                indexed = true;
            }
            _ => return Ok(None), // unindexable column type → no skip index
        }
    }
    Ok(if indexed {
        Some(bloom.to_bytes())
    } else {
        None
    })
}

/// Arrow row-encode the `key_indices` columns of `batch` (one comparable byte row
/// per record). Equal key *values* encode to identical bytes regardless of converter
/// instance, so a bloom built on one side and probed on the other agrees on equality.
fn key_rows(batch: &RecordBatch, key_indices: &[usize]) -> PyResult<arrow::row::Rows> {
    let fields: Vec<SortField> = key_indices
        .iter()
        .map(|&i| SortField::new(batch.column(i).data_type().clone()))
        .collect();
    let converter = RowConverter::new(fields).map_err(to_pyerr)?;
    let cols: Vec<ArrayRef> = key_indices
        .iter()
        .map(|&i| batch.column(i).clone())
        .collect();
    converter.convert_columns(&cols).map_err(to_pyerr)
}

/// Build a serialized bloom filter over the `key_indices` columns of `batches`.
///
/// The distributed join builds this over the *small* side's join keys and ships it
/// to the large side's mappers ([`bloom_filter_batches`]) to drop provably-
/// non-matching rows before they are shuffled. `expected_items` sizes the filter (a
/// ~1% false-positive rate); an overshoot only leaks a few extra rows, never drops a
/// match. Returns the bytes of `BloomFilter::to_bytes`.
#[pyfunction]
pub fn build_key_bloom(
    batches: Vec<PyArrowType<RecordBatch>>,
    key_indices: Vec<usize>,
    expected_items: u64,
) -> PyResult<Vec<u8>> {
    let batches = unwrap_batches(batches);
    let mut bloom = BloomFilter::with_params(expected_items, 0.01);
    for batch in &batches {
        if batch.num_rows() == 0 {
            continue;
        }
        let rows = key_rows(batch, &key_indices)?;
        for i in 0..batch.num_rows() {
            bloom.add(&rows.row(i));
        }
    }
    Ok(bloom.to_bytes())
}

/// Merge serialized bloom filters (built per-partition) into one — the bloom of the
/// union of their key sets. All must share dimensions (same `expected_items`). Empty
/// input or all-unparseable → `None`.
#[pyfunction]
pub fn merge_blooms(blooms: Vec<Vec<u8>>) -> PyResult<Option<Vec<u8>>> {
    let mut acc: Option<BloomFilter> = None;
    for bytes in &blooms {
        let bloom = BloomFilter::from_bytes(bytes)
            .ok_or_else(|| PyRuntimeError::new_err("malformed bloom-filter bytes"))?;
        match acc {
            None => acc = Some(bloom),
            Some(ref mut a) => a.merge(&bloom),
        }
    }
    Ok(acc.map(|b| b.to_bytes()))
}

/// Keep only the rows of `batches` whose `key_indices` key may be in `bloom_bytes`.
///
/// A pure superset filter: the bloom has no false negatives, so every row whose key
/// is genuinely in the build side survives — the join result is unchanged, only the
/// rows shuffled shrink. Empty/unparseable blooms are an error (the caller decides
/// whether to filter at all). Keys must be encoded the same way they were built
/// (same column types), which an equi-join guarantees.
#[pyfunction]
pub fn bloom_filter_batches(
    batches: Vec<PyArrowType<RecordBatch>>,
    key_indices: Vec<usize>,
    bloom_bytes: Vec<u8>,
) -> PyResult<Vec<PyArrowType<RecordBatch>>> {
    let bloom = BloomFilter::from_bytes(&bloom_bytes)
        .ok_or_else(|| PyRuntimeError::new_err("malformed bloom-filter bytes"))?;
    let batches = unwrap_batches(batches);
    let mut out = Vec::with_capacity(batches.len());
    for batch in &batches {
        if batch.num_rows() == 0 {
            out.push(PyArrowType(batch.clone()));
            continue;
        }
        let rows = key_rows(batch, &key_indices)?;
        let mask: BooleanArray = (0..batch.num_rows())
            .map(|i| Some(bloom.contains(&rows.row(i))))
            .collect();
        let filtered = filter_record_batch(batch, &mask).map_err(to_pyerr)?;
        out.push(PyArrowType(filtered));
    }
    Ok(out)
}
