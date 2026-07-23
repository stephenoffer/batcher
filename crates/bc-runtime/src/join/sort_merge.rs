//! Sort-merge equi-join: the no-hash-table join for two large (or already-sorted)
//! inputs. Produces the same [`JoinIndices`](super::JoinIndices) relation as the
//! hash join for every join type (output order differs — these are unordered
//! relations), so the executor can pick it on the build side without changing
//! semantics. Split out of the join module along the algorithm seam.

use std::cmp::Ordering;

use arrow::array::{Array, ArrayRef, UInt32Array};
use arrow::row::{RowConverter, Rows, SortField};

use super::{null_mask, JoinIndices, JoinType};
use crate::error::RuntimeError;

/// Sort `idx` into ascending encoded-key order, skipping the sort when the indices
/// already arrive that way (one O(n) pass). Pre-sorted input — time-series, an
/// upstream `Sort`, sorted lakehouse files — then merges without the O(n log n) sort,
/// which is what makes sort-merge the right pick for already-ordered inputs.
/// Result-identical: the merge consumes ascending keys either way, and equal-key
/// group order does not affect the unordered join relation.
fn sort_indices_if_unsorted(idx: &mut [u32], enc: &Rows) {
    let already = idx
        .windows(2)
        .all(|w| enc.row(w[0] as usize) <= enc.row(w[1] as usize));
    if !already {
        idx.sort_by(|&a, &b| enc.row(a as usize).cmp(&enc.row(b as usize)));
    }
}

/// Sort-merge join: sort both sides by key, then merge. Produces the **same
/// [`JoinIndices`](super::JoinIndices) relation** as
/// [`hash_join_indices`](super::hash_join_indices) for every join type (output order
/// differs — these are unordered relations). The win is no hash table: both sides
/// stream in key order, so it suits two large (or already-sorted) inputs the way
/// Spark's default join does. NULL keys never match (`NULL ≠ NULL`).
pub fn sort_merge_join_indices(
    left_keys: &[ArrayRef],
    right_keys: &[ArrayRef],
    join_type: JoinType,
) -> Result<JoinIndices, RuntimeError> {
    let n_left = left_keys.first().map_or(0, |a| a.len());
    let n_right = right_keys.first().map_or(0, |a| a.len());

    // Canonicalize signed zero on float keys so `-0.0` and `0.0` join as equal, exactly as
    // the hash-join path does (see `join/mod.rs`). Arrow's row encoding gives them distinct
    // bytes, so without this the sort-merge strategy silently returns fewer matches than the
    // hash strategy for the same query — a Kyber strategy choice must not change results.
    let l_canon = crate::keys::canonicalize_float_keys(left_keys);
    let r_canon = crate::keys::canonicalize_float_keys(right_keys);
    let left_keys: &[ArrayRef] = l_canon.as_deref().unwrap_or(left_keys);
    let right_keys: &[ArrayRef] = r_canon.as_deref().unwrap_or(right_keys);

    // One shared converter so left/right encoded keys are mutually comparable.
    let fields: Vec<SortField> = right_keys
        .iter()
        .map(|a| SortField::new(a.data_type().clone()))
        .collect();
    let converter = RowConverter::new(fields)?;
    let left_enc = converter.convert_columns(left_keys)?;
    let right_enc = converter.convert_columns(right_keys)?;
    let left_null = null_mask(left_keys, n_left);
    let right_null = null_mask(right_keys, n_right);

    // Sort the non-null-key rows of each side by encoded key (null keys never match
    // and are handled with the unmatched rows below).
    let mut l: Vec<u32> = (0..n_left as u32)
        .filter(|&i| !left_null[i as usize])
        .collect();
    let mut r: Vec<u32> = (0..n_right as u32)
        .filter(|&i| !right_null[i as usize])
        .collect();
    // Skip the O(n log n) sort on a side that already arrives in ascending key order
    // (pre-sorted lakehouse / time-series input, or an upstream `Sort`): a one-pass
    // check is O(n). The merge only needs ascending keys — equal-key group order is
    // irrelevant to the unordered result — so the as-is order is bit-equivalent.
    sort_indices_if_unsorted(&mut l, &left_enc);
    sort_indices_if_unsorted(&mut r, &right_enc);

    // Left/Full/Anti preserve unmatched left rows; Right/Full preserve unmatched
    // right rows (Semi emits only *matched* left rows, once each).
    let emit_left_unmatched = matches!(join_type, JoinType::Left | JoinType::Full | JoinType::Anti);
    let emit_right_unmatched = matches!(join_type, JoinType::Right | JoinType::Full);

    // The output is at least as large as the bigger side (each matched/unmatched row emits
    // once); pre-size to that lower bound so the common near-1:1 join skips early reallocs.
    let out_hint = l.len().max(r.len());
    let mut left_out: Vec<Option<u32>> = Vec::with_capacity(out_hint);
    let mut right_out: Vec<Option<u32>> = Vec::with_capacity(out_hint);
    let mut push = |lo: Option<u32>, ro: Option<u32>| {
        left_out.push(lo);
        right_out.push(ro);
    };

    let (mut i, mut j) = (0usize, 0usize);
    while i < l.len() && j < r.len() {
        match left_enc
            .row(l[i] as usize)
            .cmp(&right_enc.row(r[j] as usize))
        {
            Ordering::Less => {
                if emit_left_unmatched {
                    push(Some(l[i]), None);
                }
                i += 1;
            }
            Ordering::Greater => {
                if emit_right_unmatched {
                    push(None, Some(r[j]));
                }
                j += 1;
            }
            Ordering::Equal => {
                // Extents of the equal-key group on each side.
                let key = left_enc.row(l[i] as usize);
                let mut i2 = i + 1;
                while i2 < l.len() && left_enc.row(l[i2] as usize) == key {
                    i2 += 1;
                }
                let mut j2 = j + 1;
                while j2 < r.len() && right_enc.row(r[j2] as usize) == key {
                    j2 += 1;
                }
                match join_type {
                    // Semi: each matched left row once (no right column).
                    JoinType::Semi => {
                        for &li in &l[i..i2] {
                            push(Some(li), None);
                        }
                    }
                    // Anti: matched rows are dropped (only unmatched left survives).
                    JoinType::Anti => {}
                    // Inner/Left/Right/Full: the group cross product.
                    _ => {
                        for &li in &l[i..i2] {
                            for &rj in &r[j..j2] {
                                push(Some(li), Some(rj));
                            }
                        }
                    }
                }
                i = i2;
                j = j2;
            }
        }
    }
    // Tails: rows past the other side's end are all unmatched.
    while i < l.len() {
        if emit_left_unmatched {
            push(Some(l[i]), None);
        }
        i += 1;
    }
    while j < r.len() {
        if emit_right_unmatched {
            push(None, Some(r[j]));
        }
        j += 1;
    }
    // Null-key rows match nothing but are still part of their relation for outer joins.
    if emit_left_unmatched {
        for (li, &is_null) in left_null.iter().enumerate() {
            if is_null {
                push(Some(li as u32), None);
            }
        }
    }
    if emit_right_unmatched {
        for (rj, &is_null) in right_null.iter().enumerate() {
            if is_null {
                push(None, Some(rj as u32));
            }
        }
    }

    Ok(JoinIndices {
        left: UInt32Array::from(left_out),
        right: UInt32Array::from(right_out),
    })
}

#[cfg(test)]
mod fuzz_tests {
    use std::sync::Arc;

    use arrow::array::{ArrayRef, Float64Array, Int64Array};

    use super::sort_merge_join_indices;
    use crate::join::{hash_join_indices, JoinType};

    /// Tiny deterministic xorshift RNG (no dev-dep needed).
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    const JTS: [JoinType; 6] = [
        JoinType::Inner,
        JoinType::Left,
        JoinType::Right,
        JoinType::Full,
        JoinType::Semi,
        JoinType::Anti,
    ];

    /// Reconstruct the join output as a sorted multiset of *value* pairs (not row
    /// indices), so two strategies that pick different rows within a duplicate-key
    /// group still compare equal iff they emit the same logical relation. `None` = a
    /// null-supplying side. Semi/Anti carry only the left value (right is always null).
    fn value_pairs_i64(
        idx: &crate::join::JoinIndices,
        left: &[Option<i64>],
        right: &[Option<i64>],
    ) -> Vec<(Option<i64>, Option<i64>)> {
        use arrow::array::Array;
        let mut out: Vec<(Option<i64>, Option<i64>)> = (0..idx.left.len())
            .map(|k| {
                let l = idx
                    .left
                    .is_valid(k)
                    .then(|| left[idx.left.value(k) as usize])
                    .flatten();
                let r = idx
                    .right
                    .is_valid(k)
                    .then(|| right[idx.right.value(k) as usize])
                    .flatten();
                (l, r)
            })
            .collect();
        out.sort();
        out
    }

    fn i64_col(v: &[Option<i64>]) -> ArrayRef {
        Arc::new(Int64Array::from(v.to_vec()))
    }

    /// Single-column i64: sort-merge must equal the hash oracle over random inputs with
    /// heavy duplication, nulls, empty and single-row sides, for every join type.
    #[test]
    fn fuzz_single_i64_matches_hash() {
        let mut rng = Rng(0xDEADBEEF);
        for _ in 0..400 {
            let nl = rng.below(9) as usize;
            let nr = rng.below(9) as usize;
            // small key domain -> lots of collisions and duplicate groups
            let dom = 1 + rng.below(4);
            let gen = |rng: &mut Rng, n: usize| -> Vec<Option<i64>> {
                (0..n)
                    .map(|_| {
                        if rng.below(5) == 0 {
                            None
                        } else {
                            Some(rng.below(dom) as i64)
                        }
                    })
                    .collect()
            };
            let lv = gen(&mut rng, nl);
            let rv = gen(&mut rng, nr);
            let left = vec![i64_col(&lv)];
            let right = vec![i64_col(&rv)];
            for jt in JTS {
                let h = hash_join_indices(&left, &right, jt).unwrap();
                let s = sort_merge_join_indices(&left, &right, jt).unwrap();
                assert_eq!(
                    value_pairs_i64(&h, &lv, &rv),
                    value_pairs_i64(&s, &lv, &rv),
                    "sort-merge != hash for {jt:?}\n left={lv:?}\n right={rv:?}"
                );
            }
        }
    }

    /// Reconstruct value pairs for a two-column i64 key.
    fn value_pairs_i64x2(
        idx: &crate::join::JoinIndices,
        la: &[Option<i64>],
        lb: &[Option<i64>],
        ra: &[Option<i64>],
        rb: &[Option<i64>],
    ) -> Vec<((Option<i64>, Option<i64>), (Option<i64>, Option<i64>))> {
        use arrow::array::Array;
        let mut out: Vec<_> = (0..idx.left.len())
            .map(|k| {
                let l = if idx.left.is_valid(k) {
                    let i = idx.left.value(k) as usize;
                    (la[i], lb[i])
                } else {
                    (None, None)
                };
                let r = if idx.right.is_valid(k) {
                    let i = idx.right.value(k) as usize;
                    (ra[i], rb[i])
                } else {
                    (None, None)
                };
                (l, r)
            })
            .collect();
        out.sort();
        out
    }

    /// Two-column i64 key (composite): sort-merge must equal the hash oracle. Exercises
    /// multi-column key encoding and partial-null rows (one column null → whole row null).
    #[test]
    fn fuzz_two_i64_matches_hash() {
        let mut rng = Rng(0x1234_5678);
        for _ in 0..300 {
            let nl = rng.below(8) as usize;
            let nr = rng.below(8) as usize;
            let gen = |rng: &mut Rng, n: usize| -> (Vec<Option<i64>>, Vec<Option<i64>>) {
                let mut a = Vec::with_capacity(n);
                let mut b = Vec::with_capacity(n);
                for _ in 0..n {
                    a.push((rng.below(6) != 0).then(|| rng.below(3) as i64));
                    b.push((rng.below(6) != 0).then(|| rng.below(3) as i64));
                }
                (a, b)
            };
            let (la, lb) = gen(&mut rng, nl);
            let (ra, rb) = gen(&mut rng, nr);
            let left = vec![i64_col(&la), i64_col(&lb)];
            let right = vec![i64_col(&ra), i64_col(&rb)];
            for jt in JTS {
                let h = hash_join_indices(&left, &right, jt).unwrap();
                let s = sort_merge_join_indices(&left, &right, jt).unwrap();
                assert_eq!(
                    value_pairs_i64x2(&h, &la, &lb, &ra, &rb),
                    value_pairs_i64x2(&s, &la, &lb, &ra, &rb),
                    "sort-merge != hash (2col) for {jt:?}\n la={la:?} lb={lb:?}\n ra={ra:?} rb={rb:?}"
                );
            }
        }
    }

    /// Drive the **cache-radix** i64 path (build > `RADIX_MIN_BUILD_ROWS` triggers it for
    /// left-driven joins) and cross-check it against the sort-merge path, which shares no
    /// code with radix (it row-encodes and merges). A radix bucket that disagreed with key
    /// equality would silently drop matches; comparing against an independent strategy
    /// catches that. Includes a skewed (one hot bucket) and a many-distinct-keys shape.
    #[test]
    fn radix_large_i64_matches_sort_merge() {
        let build_rows = (1usize << 16) + 5000; // just over RADIX_MIN_BUILD_ROWS
                                                // Many distinct keys spread across the hash space, plus a hot key (skew) and dups.
        let build: Vec<Option<i64>> = (0..build_rows as i64)
            .map(|k| {
                if k % 50 == 0 {
                    Some(7) // hot key -> one hot bucket, long chain
                } else {
                    Some(k)
                }
            })
            .collect();
        // probe: hits present keys, the hot key, misses, and nulls.
        let probe: Vec<Option<i64>> = (0..30_000i64)
            .map(|i| match i % 7 {
                0 => None,
                1 => Some(7),        // hot
                2 => Some(-(i + 1)), // guaranteed miss (build is >= 0)
                _ => Some((i * 3) % build_rows as i64),
            })
            .collect();
        let left = vec![i64_col(&probe)];
        let right = vec![i64_col(&build)];
        for jt in [
            JoinType::Inner,
            JoinType::Left,
            JoinType::Semi,
            JoinType::Anti,
        ] {
            let h = hash_join_indices(&left, &right, jt).unwrap(); // radix path
            let s = sort_merge_join_indices(&left, &right, jt).unwrap(); // independent
            assert_eq!(
                value_pairs_i64(&h, &probe, &build),
                value_pairs_i64(&s, &probe, &build),
                "radix (large i64) != sort-merge for {jt:?}"
            );
        }
    }

    fn f64_col(v: &[Option<f64>]) -> ArrayRef {
        Arc::new(Float64Array::from(v.to_vec()))
    }

    /// Float keys with the sharp values (`-0.0`, `0.0`, `NaN`, nulls): sort-merge must
    /// agree with the hash oracle. Both canonicalize, so `-0.0` matches `0.0` and NaN
    /// matches NaN, identically across the two strategies.
    #[test]
    fn fuzz_float_matches_hash() {
        let choices = [
            Some(0.0f64),
            Some(-0.0f64),
            Some(1.5),
            Some(f64::NAN),
            None,
            Some(2.0),
        ];
        let mut rng = Rng(0xABCD_1234);
        // compare via bit patterns so NaN/-0.0 are exact; but canonicalization means
        // the *pairing* is what matters — compare counts of matched vs unmatched instead.
        for _ in 0..300 {
            let nl = rng.below(7) as usize;
            let nr = rng.below(7) as usize;
            let gen = |rng: &mut Rng, n: usize| -> Vec<Option<f64>> {
                (0..n)
                    .map(|_| choices[rng.below(choices.len() as u64) as usize])
                    .collect()
            };
            let lv = gen(&mut rng, nl);
            let rv = gen(&mut rng, nr);
            let left = vec![f64_col(&lv)];
            let right = vec![f64_col(&rv)];
            for jt in JTS {
                let h = hash_join_indices(&left, &right, jt).unwrap();
                let s = sort_merge_join_indices(&left, &right, jt).unwrap();
                // Compare on the output row multiset: (left present?, right present?)
                // plus, for present sides, the *canonical* key class. Signed zero and NaN
                // both canonicalize, so compare via a normalized f64 key.
                let norm = |x: Option<f64>| -> Option<i64> {
                    x.map(|v| {
                        if v.is_nan() {
                            i64::MAX
                        } else if v == 0.0 {
                            0
                        } else {
                            v.to_bits() as i64
                        }
                    })
                };
                let lv_n: Vec<Option<i64>> = lv.iter().map(|&x| norm(x)).collect();
                let rv_n: Vec<Option<i64>> = rv.iter().map(|&x| norm(x)).collect();
                assert_eq!(
                    value_pairs_i64(&h, &lv_n, &rv_n),
                    value_pairs_i64(&s, &lv_n, &rv_n),
                    "sort-merge != hash (float) for {jt:?}\n left={lv:?}\n right={rv:?}"
                );
            }
        }
    }
}

#[cfg(test)]
mod signed_zero_tests {
    use std::sync::Arc;

    use arrow::array::{ArrayRef, Float64Array};

    use super::sort_merge_join_indices;
    use crate::join::{hash_join_indices, JoinType};

    /// B10: sort-merge and hash equi-joins must agree on signed-zero float keys — a Kyber
    /// strategy choice must never change the result. Before canonicalization, sort-merge
    /// treated `-0.0` and `0.0` as distinct (Arrow's row bytes differ) while hash matched
    /// them, so the two strategies produced different row counts for the same query.
    #[test]
    fn sort_merge_matches_hash_on_signed_zero() {
        let left: Vec<ArrayRef> = vec![Arc::new(Float64Array::from(vec![-0.0f64, 1.5, 2.0]))];
        let right: Vec<ArrayRef> = vec![Arc::new(Float64Array::from(vec![0.0f64, 1.5]))];

        let sm = sort_merge_join_indices(&left, &right, JoinType::Inner).unwrap();
        let hj = hash_join_indices(&left, &right, JoinType::Inner).unwrap();
        assert_eq!(
            sm.left.len(),
            hj.left.len(),
            "sort-merge and hash must emit the same number of matches for signed-zero keys"
        );
        // -0.0 matches 0.0, and 1.5 matches 1.5 -> exactly two pairs.
        assert_eq!(sm.left.len(), 2);
    }
}
