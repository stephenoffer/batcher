//! A spill store must not hold one open file per partition it has finished writing.
//!
//! `DiskSpillStore` keeps a partition's `StreamWriter` open from the first append until the
//! partition is read back, which is fine when partitions are few and fixed. The external
//! sort's pass 0 uses the store differently: a *run* is a partition, it writes each run once
//! and never returns to it, and a sort large enough to spill has thousands to millions of
//! runs. Held open, that is one descriptor per run — so the sort dies on `EMFILE` on exactly
//! the inputs spilling exists to serve, with the disk nowhere near full and nothing in the
//! error naming spill as the cause.
//!
//! `close_partition` is the fix, and this is the assertion that it works: descriptors after
//! writing many partitions must not grow with the partition count.

#![cfg(target_os = "linux")]

use std::sync::Arc;

use arrow::array::{Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use bc_runtime::agg::spill::{DiskSpillStore, SpillStore};

/// Open descriptors of this process, counted from `/proc/self/fd`.
fn open_fds() -> usize {
    std::fs::read_dir("/proc/self/fd").expect("proc fd").count()
}

#[test]
fn closing_a_written_partition_releases_its_descriptor() {
    const PARTITIONS: usize = 512;

    let root = std::env::temp_dir().join(format!("bc-spill-fds-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);

    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![1_i64, 2, 3]))])
        .expect("batch");

    let mut store = DiskSpillStore::new(root.clone(), PARTITIONS).expect("spill store");
    let before = open_fds();
    for partition in 0..PARTITIONS {
        store.append(partition, &batch).expect("append");
        store.close_partition(partition).expect("close");
    }
    let after = open_fds();

    // A handful of slack for anything else the process opens; the point is that it is not
    // proportional to PARTITIONS. Without `close_partition` this is `before + 512`.
    assert!(
        after <= before + 8,
        "spill store leaked descriptors: {before} open before writing {PARTITIONS} \
         partitions, {after} after — this is what turns a large external sort into EMFILE"
    );

    // Closing must not lose data: every partition still reads back.
    for partition in 0..PARTITIONS {
        let batches = store.read(partition).expect("read");
        let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(rows, 3, "partition {partition} lost rows when closed early");
    }

    let _ = std::fs::remove_dir_all(&root);
}

/// `close_partition` is called on the read path too, so calling it twice — or on a partition
/// that was never written — must be a no-op rather than an error.
#[test]
fn closing_is_idempotent_and_safe_on_an_unwritten_partition() {
    let root = std::env::temp_dir().join(format!("bc-spill-fds-idem-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);

    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
    let batch =
        RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![7_i64]))]).expect("batch");

    let mut store = DiskSpillStore::new(root.clone(), 2).expect("spill store");
    store.append(0, &batch).expect("append");
    store.close_partition(0).expect("first close");
    store.close_partition(0).expect("second close");
    // Never written.
    store.close_partition(1).expect("close unwritten");

    assert_eq!(store.read(0).expect("read written").len(), 1);
    assert!(store.read(1).expect("read unwritten").is_empty());

    let _ = std::fs::remove_dir_all(&root);
}
