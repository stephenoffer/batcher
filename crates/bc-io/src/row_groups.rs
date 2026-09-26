//! Reading a Parquet relation one row group at a time, for a caller that schedules the row
//! groups itself.
//!
//! `bc-py`'s engine-side scan hands every row group of a source to the engine's workers, each
//! of which reads its own (`bc_interp::UnitSource`). These are the two things that needs: the
//! list of row groups, and a read of one of them decoded on the calling thread.

use arrow::record_batch::RecordBatch;

use crate::IoError;

/// The row groups of each file, in the order the files are given, as `(file, row group, rows)`.
///
/// The unit list a caller reading row group by row group works from (`bc-py`'s worker-read scan,
/// over `bc_interp::UnitSource`): in file order and then row-group order, which is the order a
/// whole-file read returns rows in. Footers come from the process cache, so asking costs nothing
/// once any read has touched the files.
///
/// # Errors
/// [`IoError`] when any file's footer cannot be read: a unit list with a file missing would
/// silently drop its rows, so it is not returned at all.
pub fn parquet_row_groups(uris: &[String]) -> Result<Vec<(usize, usize, usize)>, IoError> {
    let metas = crate::load_metadata_many(uris)?;
    let mut units = Vec::new();
    for (file, meta) in metas.iter().enumerate() {
        let Some(meta) = meta else {
            return Err(IoError::Store(format!("footer unreadable: {}", uris[file])));
        };
        for (rg, group) in meta.metadata().row_groups().iter().enumerate() {
            units.push((file, rg, usize::try_from(group.num_rows()).unwrap_or(0)));
        }
    }
    Ok(units)
}

/// One row group of one Parquet object, with the projection and an optional pushed predicate.
///
/// [`read_parquet_filtered`] for a single row group, decoded on the calling thread — callable
/// from any number of threads at once, which is how a pool of engine workers each reads its own
/// row groups (see `read_parquet_inner`'s `inline` for why the decode is not spawned). The
/// predicate prunes and filters exactly as it does there, so the rows are a superset of the
/// matching ones and the engine keeps its `Filter`.
///
/// # Errors
/// As [`read_parquet_filtered`].
pub fn read_parquet_row_group(
    uri: &str,
    row_group: usize,
    columns: Option<&[String]>,
    batch_size: usize,
    predicate: Option<&str>,
) -> Result<Vec<RecordBatch>, IoError> {
    // Polled on the calling thread, inside the runtime's context: the decode runs here, and the
    // store's blocking file reads still find the runtime they are handed to.
    let _context = crate::runtime().enter();
    futures::executor::block_on(crate::read_parquet_inner(
        uri,
        &[row_group],
        columns,
        batch_size,
        predicate,
        true,
    ))
}
