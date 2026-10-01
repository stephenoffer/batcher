//! Reading a Parquet relation one row group at a time, for a caller that schedules the row
//! groups itself.
//!
//! `bc-py`'s engine-side scan hands every row group of a source to the engine's workers, each
//! of which reads its own (`bc_interp::UnitSource`). These are the two things that needs: the
//! list of row groups, and a read of one of them decoded on the calling thread.

use std::sync::Arc;

use arrow::datatypes::{DataType, Field};
use arrow::record_batch::RecordBatch;
use futures::{StreamExt, TryStreamExt};
use parquet::arrow::arrow_reader::{
    ArrowReaderMetadata, ArrowReaderOptions, RowSelection, RowSelector,
};
use parquet::arrow::{ParquetRecordBatchStreamBuilder, RowNumber};

use crate::IoError;

/// The column a locating read appends: each row's position in its file, from zero, as `Int64`.
///
/// The name is the reader's own and cannot collide with a Parquet column in practice; it is
/// what [`read_parquet_rows`] takes back to fetch the same rows again.
pub const ROW_NUMBER: &str = "__bc_row_number";

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
    read_parquet_row_group_late(uri, row_group, columns, batch_size, predicate, None, false)
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
    late: Option<&Arc<crate::LateFilter>>,
    locate: bool,
) -> Result<Vec<RecordBatch>, IoError> {
    // Polled on the calling thread, inside the runtime's context: the decode runs here, and the
    // store's blocking file reads still find the runtime they are handed to.
    let _context = crate::runtime().enter();
    let unit = crate::Unit {
        inline: true,
        late,
        locate,
    };
    futures::executor::block_on(crate::read_parquet_inner(
        uri,
        &[row_group],
        columns,
        batch_size,
        predicate,
        &unit,
    ))
}

/// Each top-level column's uncompressed bytes across every row group of `uris`, by name.
///
/// What a caller weighs before reading some columns ahead of the rest: the footer records
/// what decoding each would handle. Footers come from the process cache.
///
/// # Errors
/// As [`parquet_row_groups`].
pub fn parquet_column_bytes(uris: &[String]) -> Result<Vec<(String, u64)>, IoError> {
    let metas = crate::load_metadata_many(uris)?;
    let mut out: Vec<(String, u64)> = Vec::new();
    for (file, meta) in metas.iter().enumerate() {
        let Some(meta) = meta else {
            return Err(IoError::Store(format!("footer unreadable: {}", uris[file])));
        };
        for group in meta.metadata().row_groups() {
            for chunk in group.columns() {
                let Some(root) = chunk.column_path().parts().first() else {
                    continue;
                };
                let size = u64::try_from(chunk.uncompressed_size()).unwrap_or(0);
                match out.iter_mut().find(|(name, _)| name == root) {
                    Some((_, total)) => *total += size,
                    None => out.push((root.clone(), size)),
                }
            }
        }
    }
    Ok(out)
}

/// `amd` with [`ROW_NUMBER`] added as a virtual column, so a read through it locates its rows.
///
/// The position is the parquet crate's own row-number column, which counts through the row
/// groups and pages a read skips, so it names the row in the file whatever pruning or row
/// filter removed the rows around it.
pub(crate) fn with_row_numbers(amd: &ArrowReaderMetadata) -> Result<ArrowReaderMetadata, IoError> {
    let field = Field::new(ROW_NUMBER, DataType::Int64, false).with_extension_type(RowNumber);
    let options = ArrowReaderOptions::new().with_virtual_columns(vec![Arc::new(field)])?;
    Ok(ArrowReaderMetadata::try_new(
        amd.metadata().clone(),
        options,
    )?)
}

/// The rows of one Parquet object at `rows` — positions in the file, as a locating read reports
/// them in [`ROW_NUMBER`] — with the projection, in ascending position order.
///
/// This is the second half of a late-materialized top-N: the narrow columns a sort needs are
/// read and sorted first, and only the few rows that win are fetched whole. Each row group
/// holding one is decoded with a row selection of just those rows, so a page none of them
/// falls in is never decompressed, and a row group holding none is never read.
///
/// # Errors
/// [`IoError`] when the file cannot be read, or when `rows` is not ascending and unique or
/// names a position past the file's end — a fetch must return exactly the rows it was asked
/// for, so one it cannot is refused rather than answered short.
pub fn read_parquet_rows(
    uri: &str,
    rows: &[u64],
    columns: Option<&[String]>,
    batch_size: usize,
) -> Result<Vec<RecordBatch>, IoError> {
    crate::runtime().block_on(read_rows_async(uri, rows, columns, batch_size.max(1)))
}

async fn read_rows_async(
    uri: &str,
    rows: &[u64],
    columns: Option<&[String]>,
    batch_size: usize,
) -> Result<Vec<RecordBatch>, IoError> {
    if rows.windows(2).any(|w| w[0] >= w[1]) {
        return Err(IoError::Store("row positions must ascend".into()));
    }
    let resolved = crate::store::resolve(uri)?;
    let (size, amd) = crate::load_metadata_cached(uri, &resolved).await?;
    // Each row group holding a wanted row, with a selection of exactly those rows in it.
    let mut plans: Vec<(usize, RowSelection)> = Vec::new();
    let (mut first, mut next) = (0u64, 0usize);
    for (rg, group) in amd.metadata().row_groups().iter().enumerate() {
        let end = first + u64::try_from(group.num_rows()).unwrap_or(0);
        let mut selectors = Vec::new();
        let mut at = first;
        while next < rows.len() && rows[next] < end {
            let row = rows[next];
            selectors.push(RowSelector::skip((row - at) as usize));
            selectors.push(RowSelector::select(1));
            at = row + 1;
            next += 1;
        }
        if !selectors.is_empty() {
            selectors.push(RowSelector::skip((end - at) as usize));
            plans.push((rg, RowSelection::from(selectors)));
        }
        first = end;
    }
    if next < rows.len() {
        return Err(IoError::Store(format!(
            "row {} is past the end of {uri} ({first} rows)",
            rows[next]
        )));
    }
    let projection = columns.map(|cols| {
        crate::projection::exact_columns(amd.parquet_schema(), cols.iter().map(String::as_str))
    });
    let reads = plans.into_iter().map(|(rg, selection)| {
        let reader = crate::split_read::object_reader(&resolved.store, &resolved.path, size);
        let mut b = ParquetRecordBatchStreamBuilder::new_with_metadata(reader, amd.clone())
            .with_batch_size(batch_size)
            .with_row_groups(vec![rg])
            .with_row_selection(selection);
        if let Some(p) = projection.clone() {
            b = b.with_projection(p);
        }
        // Spawned, so the row groups decode on the runtime's pool in parallel.
        tokio::spawn(async move { b.build()?.try_collect::<Vec<RecordBatch>>().await })
    });
    let per_rg: Vec<Vec<RecordBatch>> = futures::stream::iter(reads)
        .buffered(crate::rg_concurrency())
        .map(|joined| {
            joined
                .map_err(|e| IoError::Store(e.to_string()))?
                .map_err(IoError::from)
        })
        .try_collect()
        .await?;
    let mut batches: Vec<RecordBatch> = per_rg.into_iter().flatten().collect();
    if let Some(cols) = columns {
        crate::reorder_to_projection(&mut batches, cols);
    }
    Ok(batches)
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

    /// A file of `rows` rows: `a` = position (every 7th null), `s` text, in row groups of
    /// `per_group` rows and pages of 100 rows.
    fn write_wide(path: &std::path::Path, rows: i64, per_group: usize) {
        use arrow::array::StringArray;
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("s", DataType::Utf8, true),
            Field::new("p", DataType::Int64, false),
        ]));
        let a = Int64Array::from(
            (0..rows)
                .map(|i| (i % 7 != 0).then_some(i))
                .collect::<Vec<_>>(),
        );
        let s = StringArray::from((0..rows).map(|i| Some(format!("s{i}"))).collect::<Vec<_>>());
        let p = Int64Array::from((0..rows).collect::<Vec<_>>());
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(a) as ArrayRef, Arc::new(s), Arc::new(p)],
        )
        .unwrap();
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(per_group))
            .set_data_page_row_count_limit(100)
            .set_write_batch_size(100)
            .build();
        let file = std::fs::File::create(path).unwrap();
        let mut w = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
    }

    fn int_column(batches: &[RecordBatch], name: &str) -> Vec<Option<i64>> {
        batches
            .iter()
            .flat_map(|b| {
                let col = b.column_by_name(name).unwrap();
                let col = col.as_any().downcast_ref::<Int64Array>().unwrap();
                col.iter().collect::<Vec<_>>()
            })
            .collect()
    }

    /// A locating read reports each row's position in the file — through pruned pages and a
    /// row filter alike — and fetching those positions returns exactly those rows, every
    /// column, in position order, from as many row groups as they span.
    #[test]
    fn located_rows_fetch_back_exactly() {
        let dir = std::env::temp_dir().join(format!("bcio_locate_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.parquet");
        write_wide(&path, 5_000, 1_000);
        let uri = path.to_str().unwrap();
        let cols = vec!["a".to_string()];
        let keep = |b: &RecordBatch| -> arrow::array::BooleanArray {
            let a = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            a.iter()
                .map(|v| Some(v.is_some_and(|v| v % 97 == 5)))
                .collect()
        };
        let late = Arc::new(
            crate::LateFilter::new(vec![crate::RowPredicate {
                columns: cols.clone(),
                mask: Arc::new(keep),
                dictionary: false,
            }])
            .unwrap(),
        );
        let mut located = Vec::new();
        for rg in 0..5 {
            for filtered in [Some(&late), None] {
                let out =
                    read_parquet_row_group_late(uri, rg, Some(&cols), 256, None, filtered, true)
                        .unwrap();
                let a = int_column(&out, "a");
                let at = int_column(&out, ROW_NUMBER);
                assert_eq!(a.len(), at.len());
                // `a` is the position itself wherever it is not null.
                for (a, at) in a.iter().zip(&at) {
                    let at = at.unwrap();
                    assert!((rg as i64 * 1_000..(rg as i64 + 1) * 1_000).contains(&at));
                    if let Some(a) = a {
                        assert_eq!(*a, at);
                    }
                }
                if filtered.is_none() {
                    assert_eq!(at.len(), 1_000, "an unfiltered read locates every row");
                    let want: Vec<Option<i64>> = (rg as i64 * 1_000..(rg as i64 + 1) * 1_000)
                        .map(Some)
                        .collect();
                    assert_eq!(at, want);
                }
            }
            let out =
                read_parquet_row_group_late(uri, rg, Some(&cols), 256, None, Some(&late), true)
                    .unwrap();
            located.extend(int_column(&out, ROW_NUMBER).into_iter().flatten());
        }
        // The late filter alternates while undecided; collect the kept rows from a fetch of
        // a fixed set instead, which is the property the top-N relies on.
        let wanted: Vec<u64> = vec![0, 5, 99, 100, 101, 999, 1_000, 2_345, 4_999];
        let got = read_parquet_rows(uri, &wanted, None, 3).unwrap();
        let p: Vec<i64> = int_column(&got, "p").into_iter().flatten().collect();
        assert_eq!(p, wanted.iter().map(|&w| w as i64).collect::<Vec<_>>());
        assert_eq!(got[0].num_columns(), 3, "every column, and no row number");
        let s: Vec<String> = got
            .iter()
            .flat_map(|b| {
                let s = b
                    .column(1)
                    .as_any()
                    .downcast_ref::<arrow::array::StringArray>();
                s.unwrap()
                    .iter()
                    .map(|v| v.unwrap().to_string())
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(
            s,
            wanted.iter().map(|w| format!("s{w}")).collect::<Vec<_>>()
        );
        // A projection is honoured in the order asked for.
        let proj = vec!["p".to_string(), "a".to_string()];
        let got = read_parquet_rows(uri, &[7, 8], Some(&proj), 1024).unwrap();
        assert_eq!(got[0].schema().field(0).name(), "p");
        assert_eq!(int_column(&got, "a"), vec![None, Some(8)]);
        // Nothing asked for is nothing read; a bad request is refused, never answered short.
        assert!(read_parquet_rows(uri, &[], None, 16).unwrap().is_empty());
        assert!(read_parquet_rows(uri, &[5, 5], None, 16).is_err());
        assert!(read_parquet_rows(uri, &[9, 3], None, 16).is_err());
        assert!(read_parquet_rows(uri, &[5_000], None, 16).is_err());
        assert!(!located.is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }
}
