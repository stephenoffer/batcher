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
    parquet_row_groups_surviving(uris, None)
}

/// [`parquet_row_groups`], less the row groups whose footer statistics prove `predicate` (the
/// native reader's JSON predicate) matches none of their rows.
///
/// A row group pruned here is one every read of it would have pruned anyway, so dropping it
/// from the unit list changes no row. What it changes is the *schedule*. A caller that splits
/// its units into contiguous ranges, one per worker, hands each worker a share of the list —
/// and on a table clustered by the predicate's column the survivors sit together, so they
/// landed in one or two ranges while the other workers' ranges pruned to nothing: ClickBench's
/// `CounterID = 62` keeps 6 of 90 row groups, all six in the first two of sixteen ranges, and
/// the scan ran at 28 % of the cores. Ranged after pruning, the six go to six workers.
///
/// An unparseable predicate prunes nothing, exactly as a read given it would.
///
/// # Errors
/// As [`parquet_row_groups`].
pub fn parquet_row_groups_surviving(
    uris: &[String],
    predicate: Option<&str>,
) -> Result<Vec<(usize, usize, usize)>, IoError> {
    let metas = crate::load_metadata_many(uris)?;
    let pred = predicate.and_then(crate::predicate::parse);
    let mut units = Vec::new();
    for (file, meta) in metas.iter().enumerate() {
        let Some(meta) = meta else {
            return Err(IoError::Store(format!("footer unreadable: {}", uris[file])));
        };
        let groups = meta.metadata().row_groups();
        let all: Vec<usize> = (0..groups.len()).collect();
        let kept = match pred.as_ref() {
            Some(p) => crate::predicate::surviving_row_groups(meta.metadata(), p, &all),
            None => all,
        };
        for rg in kept {
            let rows = groups.get(rg).map_or(0, |g| g.num_rows());
            units.push((file, rg, usize::try_from(rows).unwrap_or(0)));
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
    read_parquet_row_group_late(uri, row_group, columns, batch_size, predicate, None)
}

/// [`read_parquet_row_group`], with the caller's own filter decoded first (`late`).
///
/// `late` is shared by every read of one scan: it decides, from what its first row groups kept,
/// whether decoding the predicate columns first pays (see [`crate::LateFilter`]). The rows
/// returned are those `late` keeps, or every row while it is off; the caller keeps its filter
/// either way, which is what makes a mask that only *approximates* its filter still correct.
///
/// # Errors
/// As [`read_parquet_row_group`].
pub fn read_parquet_row_group_late(
    uri: &str,
    row_group: usize,
    columns: Option<&[String]>,
    batch_size: usize,
    predicate: Option<&str>,
    late: Option<&std::sync::Arc<crate::LateFilter>>,
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
        late,
    ))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{ArrayRef, Int64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;

    use super::*;

    /// A file whose column `a` runs 0..rows in order, in row groups of `per_group` rows.
    fn write_sorted(path: &std::path::Path, rows: i64, per_group: usize) {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
        let a: ArrayRef = Arc::new(Int64Array::from((0..rows).collect::<Vec<_>>()));
        let batch = RecordBatch::try_new(schema.clone(), vec![a]).unwrap();
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(per_group))
            .build();
        let file = std::fs::File::create(path).unwrap();
        let mut w = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
    }

    /// The surviving units are exactly the row groups a read with the predicate would not
    /// prune, across files and in file order; no predicate, or one that does not parse, keeps
    /// them all.
    #[test]
    fn surviving_units_are_the_row_groups_the_footers_cannot_rule_out() {
        let dir = std::env::temp_dir().join(format!("bcio_surv_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (p0, p1) = (dir.join("a.parquet"), dir.join("b.parquet"));
        write_sorted(&p0, 1_000, 250); // a: [0,250) [250,500) [500,750) [750,1000)
        write_sorted(&p1, 500, 250); // a: [0,250) [250,500)
        let uris = vec![
            p0.to_str().unwrap().to_string(),
            p1.to_str().unwrap().to_string(),
        ];
        let all = parquet_row_groups(&uris).unwrap();
        assert_eq!(all.len(), 6);
        assert_eq!(parquet_row_groups_surviving(&uris, None).unwrap(), all);
        assert_eq!(
            parquet_row_groups_surviving(&uris, Some("not a predicate")).unwrap(),
            all
        );
        let ge = r#"{"node":"cmp","col":"a","op":"ge","lit":600}"#;
        assert_eq!(
            parquet_row_groups_surviving(&uris, Some(ge)).unwrap(),
            vec![(0, 2, 250), (0, 3, 250)]
        );
        let lt = r#"{"node":"cmp","col":"a","op":"lt","lit":100}"#;
        let kept = parquet_row_groups_surviving(&uris, Some(lt)).unwrap();
        assert_eq!(kept, vec![(0, 0, 250), (1, 0, 250)]);
        // A unit survives exactly when a filtered read of its row group returns rows.
        for unit @ (file, rg, _) in all {
            let rows: usize = read_parquet_row_group(&uris[file], rg, None, 1024, Some(lt))
                .unwrap()
                .iter()
                .map(RecordBatch::num_rows)
                .sum();
            assert_eq!(rows > 0, kept.contains(&unit), "file {file} rg {rg}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
