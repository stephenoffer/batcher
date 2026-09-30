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
}

/// The driving scan of one worker's pipeline, read lazily: `units` of `src` stand in for
/// `sources[source_id]`.
pub(crate) struct LazyScan<'a> {
    pub(crate) source_id: usize,
    pub(crate) src: &'a dyn UnitSource,
    pub(crate) units: std::ops::Range<usize>,
}
