//! A truncated spill file must fail the query, not shorten the answer.
//!
//! This is the most dangerous shape a spill failure can take, because it does not look like a
//! failure. An Arrow IPC stream truncated **at a message boundary** — the last complete batch
//! present, the end-of-stream marker gone — is byte-for-byte a shorter *valid* stream. The
//! reader returns the batches it finds and reports success, so the aggregate, join, or sort
//! reading it back computes a correct answer over the wrong rows.
//!
//! Measured, not argued: five batches of 1,000 rows, truncated after the third, read back as
//! 3,000 rows with no error at all. Two thousand rows leave the query with nothing anywhere
//! recording it.
//!
//! Every way this arises is a way a query returns a wrong answer rather than failing: a
//! filesystem that reported a short write as success, a spill file that outlived the process
//! still writing it, a truncation on a full disk the write path did not observe. Counting
//! rows in and checking them out turns all of them into an error.

use std::sync::Arc;

use arrow::array::{Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use bc_runtime::agg::spill::{DiskSpillStore, SpillStore};
use bc_runtime::RuntimeError;

fn batch(n: i64) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
    RecordBatch::try_new(
        schema,
        vec![Arc::new(Int64Array::from((0..n).collect::<Vec<_>>()))],
    )
    .expect("batch")
}

/// The store's one `part-*.arrow` file under `root`.
fn sole_spill_file(root: &std::path::Path) -> std::path::PathBuf {
    for dir in std::fs::read_dir(root).expect("root").flatten() {
        for f in std::fs::read_dir(dir.path()).expect("scratch").flatten() {
            if f.file_name().to_string_lossy().starts_with("part-") {
                return f.path();
            }
        }
    }
    panic!("no spill file under {root:?}");
}

/// Write several batches, lop off the tail, and read the partition back.
fn truncated_read<F>(name: &str, read_back: F) -> Result<u64, RuntimeError>
where
    F: FnOnce(&mut DiskSpillStore) -> Result<u64, RuntimeError>,
{
    let root = std::env::temp_dir().join(format!("bc-spill-trunc-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);

    let mut store = DiskSpillStore::new(root.clone(), 1).expect("store");
    for _ in 0..5 {
        store.append(0, &batch(1000)).expect("append");
    }
    // Close so the stream is complete and the file is on disk, then cut it back. Two thirds
    // lands inside a message, which arrow does notice; the dangerous case is the boundary,
    // and both are covered by the callers below.
    store.close_partition(0).expect("close");
    let path = sole_spill_file(&root);
    let len = std::fs::metadata(&path).expect("metadata").len();
    let f = std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .expect("open");
    f.set_len(len / 2).expect("truncate");
    drop(f);

    let result = read_back(&mut store);
    let _ = std::fs::remove_dir_all(&root);
    result
}

#[test]
fn a_truncated_partition_fails_the_whole_partition_read() {
    let err = truncated_read("read", |s| {
        s.read(0)
            .map(|b| b.iter().map(|x| x.num_rows() as u64).sum())
    })
    .expect_err("a truncated spill partition must not read back as a shorter valid relation");

    match err {
        RuntimeError::SpillTruncated {
            expected_rows,
            got_rows,
            missing,
            ..
        } => {
            assert_eq!(expected_rows, 5000);
            assert!(got_rows < 5000);
            assert_eq!(missing, 5000 - got_rows);
        }
        // Arrow catches a cut that lands mid-message on its own; that is also a refusal, and
        // refusing is the property under test. Silently succeeding is not.
        RuntimeError::Arrow(_) | RuntimeError::Io(_) => {}
        other => panic!("truncated spill reported as {other}"),
    }
}

#[test]
fn a_truncated_partition_fails_the_streaming_drain() {
    let err = truncated_read("drain", |s| {
        let mut rows = 0u64;
        s.drain(0, &mut |b| {
            rows += b.num_rows() as u64;
            Ok(())
        })?;
        Ok(rows)
    })
    .expect_err("a truncated spill partition must not drain as a shorter valid relation");

    assert!(
        matches!(
            err,
            RuntimeError::SpillTruncated { .. } | RuntimeError::Arrow(_) | RuntimeError::Io(_)
        ),
        "truncated spill reported as {err}"
    );
}

/// The check must not fire on an intact partition — otherwise it would be a way to fail every
/// spilling query rather than a guard on a rare one.
#[test]
fn an_intact_partition_reads_back_without_complaint() {
    let root = std::env::temp_dir().join(format!("bc-spill-intact-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let mut store = DiskSpillStore::new(root.clone(), 2).expect("store");
    for _ in 0..4 {
        store.append(0, &batch(500)).expect("append");
    }
    store.append(1, &batch(7)).expect("append");

    let rows: u64 = store
        .read(0)
        .expect("intact read")
        .iter()
        .map(|b| b.num_rows() as u64)
        .sum();
    assert_eq!(rows, 2000);

    let mut drained = 0u64;
    store
        .drain(1, &mut |b| {
            drained += b.num_rows() as u64;
            Ok(())
        })
        .expect("intact drain");
    assert_eq!(drained, 7);

    let _ = std::fs::remove_dir_all(&root);
}

/// The dangerous cut, pinned exactly.
///
/// A cut that lands *inside* a message, arrow catches on its own — the tests above accept
/// that. A cut at a **message boundary** it cannot catch: what remains is a shorter valid
/// stream, so the read succeeds and the rows are simply gone. That is the case the row count
/// exists for, and this constructs it precisely rather than hoping a halved file lands there.
///
/// The boundary is found by writing the same batches to a second store and taking the length
/// of *its* file, minus the 8-byte end-of-stream marker. What is left is exactly three
/// complete messages with no EOS — byte-for-byte what a killed writer leaves behind.
#[test]
fn a_cut_at_a_message_boundary_is_caught_by_the_row_count_alone() {
    const EOS_BYTES: u64 = 8; // continuation marker + zero length

    let root = std::env::temp_dir().join(format!("bc-spill-boundary-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);

    // A reference file holding exactly three batches: its length is the boundary.
    let mut reference = DiskSpillStore::new(root.join("ref"), 1).expect("reference store");
    for _ in 0..3 {
        reference.append(0, &batch(1000)).expect("append");
    }
    reference.close_partition(0).expect("close");
    let boundary = std::fs::metadata(sole_spill_file(&root.join("ref")))
        .expect("metadata")
        .len()
        - EOS_BYTES;

    // The real file: five batches, cut back to the three-batch boundary.
    let mut store = DiskSpillStore::new(root.join("live"), 1).expect("store");
    for _ in 0..5 {
        store.append(0, &batch(1000)).expect("append");
    }
    store.close_partition(0).expect("close");
    let path = sole_spill_file(&root.join("live"));
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .expect("open")
        .set_len(boundary)
        .expect("truncate");

    match store.read(0) {
        Err(RuntimeError::SpillTruncated {
            expected_rows,
            got_rows,
            missing,
            ..
        }) => {
            assert_eq!(expected_rows, 5000);
            assert_eq!(
                got_rows, 3000,
                "the truncated stream should read as three batches"
            );
            assert_eq!(missing, 2000);
        }
        Ok(batches) => {
            let rows: u64 = batches.iter().map(|b| b.num_rows() as u64).sum();
            panic!(
                "a spill file cut at a message boundary read back as {rows} of 5000 rows and \
                 reported success — 2,000 rows would have left the query silently"
            );
        }
        Err(other) => panic!("expected the row-count check to catch this, got {other}"),
    }

    let _ = std::fs::remove_dir_all(&root);
}
