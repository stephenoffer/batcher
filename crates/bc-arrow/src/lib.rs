//! `bc-arrow` — Arrow building blocks shared across the engine.
//!
//! Arrow is the single in-memory columnar contract for Batcher: every operator,
//! the interpreter, the (future) JIT, and the Python boundary all speak Arrow
//! `RecordBatch`es. This crate re-exports the arrow surface the rest of the
//! engine uses and will grow NUMA-aware buffer allocation and C-Data-Interface
//! helpers as the data plane lands.
//!
//! Keeping these re-exports in one place means the rest of the workspace pins a
//! single arrow version through `bc_arrow::arrow` rather than depending on the
//! crate directly, so an arrow bump is a one-line change here.
//!
//! It is also the single home for the data plane's performance-threshold defaults
//! ([`RuntimeTuning`]): the lowest crate `bc-runtime` and `bc-interp` both see, so a
//! shared tuning contract lives here, is mirrored into `bc_ir::EngineConfig`, and is
//! shipped from the Python control plane.

pub use arrow;
pub use arrow::array::{Array, ArrayRef, RecordBatch};
pub use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
pub use arrow::error::ArrowError;

/// A unit of data flow between operators: an Arrow `RecordBatch`.
///
/// Named `Morsel` to match the scheduler vocabulary — the engine processes data
/// in fixed-size morsels (default 16384 rows) for cache efficiency and
/// fine-grained work-stealing. For now it is a transparent alias; it will gain
/// arena/refcount metadata once the shared-memory data plane is built.
pub type Morsel = RecordBatch;

/// Default morsel size in rows (§1.4 of the architecture plan).
///
/// Chosen to fit comfortably in L2/L3 cache while amortizing per-morsel
/// scheduling overhead and staying small enough for fine-grained load balancing.
pub const DEFAULT_MORSEL_ROWS: usize = 16_384;

/// Default morsel *byte* budget (≈ `DEFAULT_MORSEL_ROWS` × 64 B/row).
///
/// The engine sizes morsels by row count by default, which is byte-blind: 16 384
/// rows of `Int64` is ~128 KiB, but 16 384 rows of multi-MB blobs is terabytes.
/// This budget caps a morsel's working-set bytes so wide/variable-width data is
/// split finely enough to stay cache- and memory-friendly. The value matches the
/// historical row default for narrow data, so nothing changes there.
pub const DEFAULT_MORSEL_BYTES: usize = 1 << 20; // 1 MiB

/// The fixed per-row byte width of a data type, when it is constant regardless of
/// the data — i.e. the type's values are not variable-length.
///
/// Returns `None` for variable-width types (`Utf8`/`Binary`/`List`/`Struct`/…),
/// whose per-row width is data-dependent and must be *measured* (see the
/// `ColumnStats` average-width path). This is the cheap, allocation-free lower
/// bound the cost model and morselizer reach for first.
pub fn fixed_width(dt: &DataType) -> Option<usize> {
    use DataType::*;
    Some(match dt {
        Null => 0,
        Boolean | Int8 | UInt8 => 1,
        Int16 | UInt16 | Float16 => 2,
        Int32 | UInt32 | Float32 | Date32 | Time32(_) => 4,
        Int64 | UInt64 | Float64 | Date64 | Time64(_) | Timestamp(_, _) | Duration(_) => 8,
        Interval(_) => 16,
        Decimal128(_, _) => 16,
        Decimal256(_, _) => 32,
        FixedSizeBinary(n) => *n as usize,
        FixedSizeList(field, n) => fixed_width(field.data_type())? * (*n as usize),
        _ => return None, // Utf8/LargeUtf8/Binary/LargeBinary/List/Struct/Map/...
    })
}

/// A morsel-size target with two independent bounds: a morsel is "full" when it
/// reaches **either** `rows` or `bytes`.
///
/// The row bound preserves the historical, cache-tuned behavior; the byte bound
/// keeps wide/variable-width morsels (large strings, embeddings, blob handles)
/// from ballooning. Row-only callers use [`MorselTarget::rows`], which sets
/// `bytes = usize::MAX` so the byte check short-circuits out of the hot path and
/// behavior is byte-for-byte identical to the pre-byte-aware engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MorselTarget {
    pub rows: usize,
    pub bytes: usize,
}

impl MorselTarget {
    /// A row-only target (byte bound disabled). The historical default.
    pub fn rows(rows: usize) -> Self {
        Self {
            rows,
            bytes: usize::MAX,
        }
    }

    /// A target bounded by both a row count and a byte budget.
    pub fn new(rows: usize, bytes: usize) -> Self {
        Self { rows, bytes }
    }

    /// Whether the byte bound is active (i.e. not the row-only sentinel).
    pub fn byte_bounded(&self) -> bool {
        self.bytes != usize::MAX
    }
}

impl Default for MorselTarget {
    fn default() -> Self {
        Self::new(DEFAULT_MORSEL_ROWS, DEFAULT_MORSEL_BYTES)
    }
}

/// Performance-threshold defaults for the data plane's hot paths — the single home
/// for the tunables Kyber/Core may want to override per query.
///
/// These are **performance-only** knobs (parallel-vs-serial thresholds, the probe
/// bloom, merge fan-in, skew detection): they change *how* an operator runs, never
/// the relation it produces. Every field's [`Default`] equals the historical `const`
/// it replaced, so absent any override behavior is bit-identical. This struct is
/// mirrored field-for-field into `bc_ir::EngineConfig` and shipped from Python's
/// `ExecutionConfig`; the parallel executor threads the live values into the
/// `bc-runtime` `_with` overloads, while the sequential oracle and all existing
/// callers stay on this default tuning.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RuntimeTuning {
    /// False-positive rate for the hash-join probe-side bloom pre-filter.
    pub bloom_fp_rate: f64,
    /// Build-row floor above which the probe bloom pays for itself.
    pub bloom_min_build_rows: usize,
    /// Window row count above which per-partition sorts run across cores.
    pub window_parallel_row_threshold: usize,
    /// Concatenated-input row count above which `combine` regroups via parallel
    /// hash-radix partitioning.
    pub radix_parallel_threshold: usize,
    /// Maximum runs merged per pass in the external (spilling) sort's k-way merge.
    pub sort_merge_fanin: usize,
    /// A join bucket is "hot" when it exceeds this multiple of the average bucket.
    pub skew_bucket_factor: usize,
    /// Absolute row floor below which a bucket is never treated as skewed.
    pub skew_min_bucket_rows: usize,
    /// Absolute byte floor below which a bucket is never treated as skewed.
    pub skew_min_bucket_bytes: usize,
}

impl Default for RuntimeTuning {
    fn default() -> Self {
        Self {
            bloom_fp_rate: 0.01,
            bloom_min_build_rows: 1 << 16,
            window_parallel_row_threshold: 1 << 15,
            radix_parallel_threshold: 200_000,
            sort_merge_fanin: 16,
            skew_bucket_factor: 4,
            skew_min_bucket_rows: 4 * DEFAULT_MORSEL_ROWS,
            skew_min_bucket_bytes: 4 * DEFAULT_MORSEL_BYTES,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{Field, TimeUnit};
    use std::sync::Arc;

    #[test]
    fn fixed_width_primitives() {
        assert_eq!(fixed_width(&DataType::Int64), Some(8));
        assert_eq!(fixed_width(&DataType::Float64), Some(8));
        assert_eq!(fixed_width(&DataType::Int32), Some(4));
        assert_eq!(fixed_width(&DataType::Boolean), Some(1));
        assert_eq!(fixed_width(&DataType::Null), Some(0));
        assert_eq!(
            fixed_width(&DataType::Timestamp(TimeUnit::Microsecond, None)),
            Some(8)
        );
        assert_eq!(fixed_width(&DataType::Decimal128(10, 2)), Some(16));
    }

    #[test]
    fn fixed_width_fixed_size_list_multiplies() {
        let inner = Arc::new(Field::new("e", DataType::Float32, false));
        assert_eq!(
            fixed_width(&DataType::FixedSizeList(inner, 128)),
            Some(4 * 128)
        );
        assert_eq!(fixed_width(&DataType::FixedSizeBinary(20)), Some(20));
    }

    #[test]
    fn fixed_width_variable_types_are_none() {
        assert_eq!(fixed_width(&DataType::Utf8), None);
        assert_eq!(fixed_width(&DataType::LargeUtf8), None);
        assert_eq!(fixed_width(&DataType::Binary), None);
        assert_eq!(fixed_width(&DataType::LargeBinary), None);
        let inner = Arc::new(Field::new("e", DataType::Int64, true));
        assert_eq!(fixed_width(&DataType::List(inner)), None);
    }

    #[test]
    fn runtime_tuning_defaults_equal_the_literals() {
        let t = RuntimeTuning::default();
        assert_eq!(t.bloom_fp_rate, 0.01);
        assert_eq!(t.bloom_min_build_rows, 1 << 16);
        assert_eq!(t.window_parallel_row_threshold, 1 << 15);
        assert_eq!(t.radix_parallel_threshold, 200_000);
        assert_eq!(t.sort_merge_fanin, 16);
        assert_eq!(t.skew_bucket_factor, 4);
        assert_eq!(t.skew_min_bucket_rows, 4 * DEFAULT_MORSEL_ROWS);
        assert_eq!(t.skew_min_bucket_bytes, 4 * DEFAULT_MORSEL_BYTES);
    }

    #[test]
    fn morsel_target_row_only_is_unbounded_in_bytes() {
        let t = MorselTarget::rows(16_384);
        assert_eq!(t.bytes, usize::MAX);
        assert!(!t.byte_bounded());
        let d = MorselTarget::default();
        assert!(d.byte_bounded());
        assert_eq!(d.rows, DEFAULT_MORSEL_ROWS);
        assert_eq!(d.bytes, DEFAULT_MORSEL_BYTES);
    }
}
