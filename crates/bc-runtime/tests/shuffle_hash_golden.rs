//! The shuffle's key-to-reducer mapping, pinned per branch.
//!
//! # What this is defending against
//!
//! Hash partitioning decides which reducer owns a key. If two workers in one cluster
//! disagree about that, a `GROUP BY` group splits across two reducers and both halves are
//! finalized separately, while a join's matching rows land on different machines and never
//! meet. The result is a **wrong answer with no error**, at cluster scale only, and it
//! cannot be reproduced on one host — which is exactly why it needs a test that does not
//! depend on reproducing it.
//!
//! The engine used `ahash::RandomState` here. Fixed seeds make `ahash` deterministic within
//! one binary, but it selects an AES-NI backend from the **compile-time**
//! `target_feature`, so a worker built with `-C target-cpu=native` and one built without
//! disagree. `bc_arrow::PortableBuildHasher` replaces it on exactly the paths whose value
//! crosses a process boundary.
//!
//! # How to use these vectors
//!
//! Every number below was produced by this engine and is now a contract. A change to any
//! of them means keys will route differently than they did — which is *legal* only as a
//! deliberate, announced break, because a rolling upgrade with mixed versions in one
//! cluster hits precisely the split-group bug above. Do not re-baseline a failure here
//! without saying why in the commit.
//!
//! `just check-hash-portability` runs this file twice under different ISA flags. That is
//! the part that actually proves portability; these vectors are what make the two runs
//! comparable.

use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, Int64Array, StringArray};
use bc_runtime::shuffle::bucket_of_rows;

/// Both extraction modes of `bucket_of` are exercised: a power-of-two count masks the
/// hash's **low** bits, a non-power-of-two multiplies by its **high** bits. A hash that
/// avalanches only one end passes one and skews the other.
const PARTITION_COUNTS: [usize; 2] = [8, 7];

fn buckets(keys: &[ArrayRef], rows: usize, parts: usize) -> Vec<u32> {
    bucket_of_rows(keys, rows, parts).expect("bucket_of_rows")
}

/// Single Int64 key — `partition_int_key`'s fast path, the most common shuffle in the
/// engine (`GROUP BY id`, a join on a surrogate key).
#[test]
fn golden_single_int_key() {
    let keys: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(vec![
        0,
        1,
        2,
        3,
        100,
        -1,
        i64::MIN,
        i64::MAX,
    ]))];
    assert_eq!(buckets(&keys, 8, 8), vec![7, 7, 3, 4, 4, 3, 5, 2]);
    assert_eq!(buckets(&keys, 8, 7), vec![1, 3, 6, 1, 5, 2, 2, 6]);
}

/// Composite Int64 key — the `(part, supplier)` shape, which folds each column through the
/// hasher rather than row-encoding.
#[test]
fn golden_composite_int_key() {
    let keys: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(vec![1, 1, 2, 2])),
        Arc::new(Int64Array::from(vec![10, 20, 10, 20])),
    ];
    assert_eq!(buckets(&keys, 4, 8), vec![0, 0, 3, 2]);
    assert_eq!(buckets(&keys, 4, 7), vec![6, 2, 6, 2]);
}

/// Mixed Int64 + Utf8, null-free — `partition_mixed_key`, which hashes raw column values
/// and skips the `RowConverter`.
#[test]
fn golden_mixed_int_and_string_key() {
    let keys: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(vec![1, 2, 3])),
        Arc::new(StringArray::from(vec!["alpha", "beta", "gamma"])),
    ];
    assert_eq!(buckets(&keys, 3, 8), vec![4, 6, 6]);
    assert_eq!(buckets(&keys, 3, 7), vec![2, 6, 4]);
}

/// A nullable key falls out of the fast paths onto the `RowConverter`, so this pins a
/// different hasher call site than the tests above.
#[test]
fn golden_nullable_key_uses_the_row_converter() {
    let keys: Vec<ArrayRef> = vec![Arc::new(StringArray::from(vec![
        Some("a"),
        None,
        Some("b"),
        None,
    ]))];
    let got = buckets(&keys, 4, 8);
    assert_eq!(got, vec![6, 6, 1, 6]);
    // Whatever bucket nulls take, every null must take the *same* one, or two rows that
    // compare equal to the group assigner would be finalized on different reducers.
    assert_eq!(got[1], got[3], "all nulls must co-locate");
}

/// Float keys route through `canonicalize_float_keys` first. `-0.0` and `0.0` compare
/// equal and every NaN compares equal, so each pair must land in one bucket — the
/// divergence that has already produced a split group in this engine.
#[test]
fn golden_float_key_canonicalizes_before_hashing() {
    let keys: Vec<ArrayRef> = vec![Arc::new(Float64Array::from(vec![
        0.0,
        -0.0,
        f64::NAN,
        -f64::NAN,
        1.5,
    ]))];
    for parts in PARTITION_COUNTS {
        let got = buckets(&keys, 5, parts);
        assert_eq!(
            got[0], got[1],
            "0.0 and -0.0 must co-locate ({parts} parts)"
        );
        assert_eq!(got[2], got[3], "every NaN must co-locate ({parts} parts)");
    }
    assert_eq!(buckets(&keys, 5, 8), vec![7, 7, 2, 2, 0]);
}

/// Equal keys must map to one bucket regardless of where they sit in the batch, which is
/// the property co-partitioning is built on.
#[test]
fn equal_keys_always_co_locate() {
    let keys: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(vec![42, 7, 42, 7, 42]))];
    for parts in PARTITION_COUNTS {
        let got = buckets(&keys, 5, parts);
        assert_eq!(got[0], got[2]);
        assert_eq!(got[0], got[4]);
        assert_eq!(got[1], got[3]);
    }
}

/// No bucket may be left empty on a well-spread key set, and none may take a wildly
/// disproportionate share. Skew here is not a wrong answer, but it is a straggler on every
/// distributed query, so it is worth failing on.
#[test]
fn keys_spread_across_every_bucket() {
    let values: Vec<i64> = (0..4096).collect();
    let keys: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(values))];
    for parts in [8usize, 7, 16, 64] {
        let got = buckets(&keys, 4096, parts);
        let mut counts = vec![0usize; parts];
        for bucket in got {
            counts[bucket as usize] += 1;
        }
        let expected = 4096 / parts;
        assert!(
            counts.iter().all(|&c| c > 0),
            "{parts} partitions: a bucket got nothing — {counts:?}"
        );
        assert!(
            counts.iter().all(|&c| c < expected * 2),
            "{parts} partitions: skewed — {counts:?}"
        );
    }
}
