//! The driving relation as a sequence of units read on demand by the workers that scan it.

use arrow::array::RecordBatch;

use crate::InterpError;

/// A driving relation read one unit at a time, on demand, by the worker that scans it.
///
/// `bc-py` implements it over Parquet row groups decoded by `bc-io`, which the interpreter
/// cannot depend on (the crate DAG points the other way). A unit is the smallest piece that is
/// read independently; units are numbered in the relation's row order, and reading them in
/// order and concatenating reproduces the relation.
pub trait UnitSource: Sync {
    /// How many units the relation has.
    fn units(&self) -> usize;
    /// The relation's total rows, if known without reading it. A Parquet footer records it;
    /// a source that returns `None` only forgoes the runtime-filter sizing that needs it.
    fn rows(&self) -> Option<usize> {
        None
    }
    /// The rows of unit `unit`, in order.
    ///
    /// # Errors
    /// Whatever reading the unit reports.
    fn read(&self, unit: usize) -> Result<Vec<RecordBatch>, InterpError>;

    /// This relation reading only `columns`, in that order, with a [`LOCATOR`] column appended
    /// that [`UnitSource::fetch`] resolves back to the row — or `None` when it cannot locate
    /// its rows, or judges that reading `columns` first would save nothing.
    ///
    /// The narrowed view has the same units, and each unit the same rows in the same order.
    fn narrowed(&self, columns: &[String]) -> Option<Box<dyn UnitSource + '_>> {
        let _ = columns;
        None
    }

    /// The rows a narrowed view reported under `locators`, every column, in `locators`' order.
    ///
    /// # Errors
    /// Whatever reading them reports, and [`InterpError::NotChunkable`] from a source that
    /// never narrows.
    fn fetch(&self, locators: &[u64]) -> Result<Vec<RecordBatch>, InterpError> {
        let _ = locators;
        Err(InterpError::NotChunkable)
    }
}

/// The column a [narrowed](UnitSource::narrowed) source appends: a `UInt64` naming each row.
pub const LOCATOR: &str = "__bc_locator";

/// The driving scan of one worker's pipeline, read lazily: `units` of `src` stand in for
/// `sources[source_id]`.
pub(crate) struct LazyScan<'a> {
    pub(crate) source_id: usize,
    pub(crate) src: &'a dyn UnitSource,
    pub(crate) units: std::ops::Range<usize>,
}
