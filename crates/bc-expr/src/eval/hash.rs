//! `Expr::Hash` — a deterministic, typed 64-bit row hash.
//!
//! The building block for a reproducible split, a surrogate key, and hash bucketing.
//! Its two contracts are worth stating plainly, because both are load-bearing:
//!
//! * **Typed, not textual.** Each value is hashed from its own bytes — an integer from
//!   its two's-complement bits, a float from its (canonicalized) IEEE bits, a string
//!   from its UTF-8. The `concat_ws(cast(col, 'string'), …)` idiom this replaces has to
//!   render every value to text first, which costs an allocation per value per row and
//!   silently ties the result to how floats print.
//! * **Stable forever.** A train/test split assigns rows by this digest, so changing it
//!   reshuffles every split that was ever taken. It is `SplitMix64`-based, written out,
//!   and pinned by golden tests — no `ahash`, no `DefaultHasher`, nothing whose value
//!   may vary across builds or releases.
//!
//! `-0.0` hashes as `0.0` and every NaN hashes alike, so two values that compare equal
//! never land in different buckets. Null is a distinct value rather than an absorbing
//! one, so `(1, NULL)` and `(NULL, 1)` differ and neither collides with `(1, 1)`.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, BooleanArray, Int64Array, StringArray};
use arrow::compute::cast;
use arrow::datatypes::DataType;

use crate::ExprError;

/// A distinct, arbitrary constant mixed in for a null so it is a *value*, not an
/// absence: without it `hash(1, NULL)` and `hash(NULL, 1)` would coincide.
const NULL_TAG: u64 = 0x9E37_79B9_7F4A_7C15;

/// SplitMix64's finalizer — an avalanching 64-bit integer hash.
fn mix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Fold `value` into the running hash. Order-sensitive: mixing after the combine means
/// swapping two columns changes the digest.
fn combine(acc: u64, value: u64) -> u64 {
    mix64(acc ^ value.wrapping_add(0x9E37_79B9_7F4A_7C15).rotate_left(31))
}

/// Hash a byte string (FNV-1a — short, dependency-free, fine after the `mix64` fold).
fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

/// Canonicalize a float so values that compare equal hash equally: `-0.0` becomes
/// `0.0`, and every NaN bit pattern becomes one.
fn hash_f64(v: f64) -> u64 {
    if v.is_nan() {
        return 0x7ff8_0000_0000_0000;
    }
    if v == 0.0 {
        return 0; // collapses -0.0 and 0.0
    }
    v.to_bits()
}

/// Evaluate `hash(inputs…, seed)` over already-evaluated argument arrays.
///
/// Every column is folded into a per-row accumulator seeded from `seed`. Lists (an
/// embedding, a packed training sequence) hash from their elements; anything else
/// outside the primitive set is coerced to `Utf8` first, so a hash is always defined.
pub(crate) fn eval_hash(args: &[ArrayRef], seed: i64, rows: usize) -> Result<ArrayRef, ExprError> {
    let mut acc = vec![mix64(seed as u64); rows];
    for arr in args {
        fold_column(&mut acc, arr)?;
    }
    // `u64 -> i64` reinterprets the bits (Arrow has no UInt64 in our type lattice);
    // `Expr.hash` documents the value as a signed 64-bit digest.
    let out: Int64Array = acc.iter().map(|h| Some(*h as i64)).collect();
    Ok(Arc::new(out))
}

fn fold_column(acc: &mut [u64], arr: &ArrayRef) -> Result<(), ExprError> {
    match arr.data_type() {
        DataType::Boolean => {
            let a = arr
                .as_any()
                .downcast_ref::<BooleanArray>()
                .expect("boolean");
            for (i, slot) in acc.iter_mut().enumerate() {
                let v = if a.is_null(i) {
                    NULL_TAG
                } else {
                    u64::from(a.value(i))
                };
                *slot = combine(*slot, v);
            }
        }
        DataType::Float16 | DataType::Float32 | DataType::Float64 => {
            let f = cast(arr, &DataType::Float64)?;
            let a = f
                .as_any()
                .downcast_ref::<arrow::array::Float64Array>()
                .expect("float64");
            for (i, slot) in acc.iter_mut().enumerate() {
                let v = if a.is_null(i) {
                    NULL_TAG
                } else {
                    hash_f64(a.value(i))
                };
                *slot = combine(*slot, v);
            }
        }
        DataType::Utf8 | DataType::LargeUtf8 => {
            let s = cast(arr, &DataType::Utf8)?;
            let a = s.as_any().downcast_ref::<StringArray>().expect("utf8");
            for (i, slot) in acc.iter_mut().enumerate() {
                let v = if a.is_null(i) {
                    NULL_TAG
                } else {
                    hash_bytes(a.value(i).as_bytes())
                };
                *slot = combine(*slot, v);
            }
        }
        dt if dt.is_integer() || matches!(dt, DataType::Date32 | DataType::Date64) => {
            let i64s = cast(arr, &DataType::Int64)?;
            let a = i64s.as_any().downcast_ref::<Int64Array>().expect("int64");
            for (i, slot) in acc.iter_mut().enumerate() {
                let v = if a.is_null(i) {
                    NULL_TAG
                } else {
                    a.value(i) as u64
                };
                *slot = combine(*slot, v);
            }
        }
        DataType::Timestamp(_, _) => {
            let i64s = cast(arr, &DataType::Int64)?;
            let a = i64s.as_any().downcast_ref::<Int64Array>().expect("int64");
            for (i, slot) in acc.iter_mut().enumerate() {
                let v = if a.is_null(i) {
                    NULL_TAG
                } else {
                    a.value(i) as u64
                };
                *slot = combine(*slot, v);
            }
        }
        DataType::List(_) | DataType::LargeList(_) | DataType::FixedSizeList(_, _) => {
            fold_list(acc, arr)?;
        }
        _ => {
            // Binary / decimal / struct: fall back to the textual form rather than
            // refusing to hash. Correct and deterministic, just not free.
            let s = cast(arr, &DataType::Utf8)?;
            let a = s.as_any().downcast_ref::<StringArray>().expect("utf8");
            for (i, slot) in acc.iter_mut().enumerate() {
                let v = if a.is_null(i) {
                    NULL_TAG
                } else {
                    hash_bytes(a.value(i).as_bytes())
                };
                *slot = combine(*slot, v);
            }
        }
    }
    Ok(())
}

/// Fold a `List`-family column: hash each child element once, then fold each row's own
/// slice of them (plus its length, so `[1]` and `[1, 1]` differ).
///
/// A list is hashed from its *elements*, not from its text: a tensor column — a packed
/// training sequence, an embedding — has no `Utf8` cast at all, so the textual fallback
/// would fail outright on exactly the columns an ML pipeline carries.
fn fold_list(acc: &mut [u64], arr: &ArrayRef) -> Result<(), ExprError> {
    use arrow::array::{FixedSizeListArray, LargeListArray, ListArray};

    let (child, bounds): (ArrayRef, Vec<(usize, usize)>) = match arr.data_type() {
        DataType::List(_) => {
            let a = arr.as_any().downcast_ref::<ListArray>().expect("list");
            let off = a.value_offsets();
            let b = (0..a.len())
                .map(|i| (off[i] as usize, off[i + 1] as usize))
                .collect();
            (Arc::clone(a.values()), b)
        }
        DataType::LargeList(_) => {
            let a = arr
                .as_any()
                .downcast_ref::<LargeListArray>()
                .expect("large");
            let off = a.value_offsets();
            let b = (0..a.len())
                .map(|i| (off[i] as usize, off[i + 1] as usize))
                .collect();
            (Arc::clone(a.values()), b)
        }
        _ => {
            let a = arr
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .expect("fixed list");
            let w = a.value_length() as usize;
            let b = (0..a.len()).map(|i| (i * w, i * w + w)).collect();
            (Arc::clone(a.values()), b)
        }
    };

    // One pass over the child gives a per-element digest; the row folds its own slice.
    let mut elements = vec![0u64; child.len()];
    fold_column(&mut elements, &child)?;

    for (i, slot) in acc.iter_mut().enumerate() {
        if arr.is_null(i) {
            *slot = combine(*slot, NULL_TAG);
            continue;
        }
        let (start, end) = bounds[i];
        let mut h = combine(0, (end - start) as u64); // length, so [1] != [1, 1]
        for e in &elements[start..end] {
            h = combine(h, *e);
        }
        *slot = combine(*slot, h);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Float64Array;

    fn hash_of(args: Vec<ArrayRef>, seed: i64) -> Vec<i64> {
        let rows = args[0].len();
        let out = eval_hash(&args, seed, rows).unwrap();
        let a = out.as_any().downcast_ref::<Int64Array>().unwrap();
        (0..a.len()).map(|i| a.value(i)).collect()
    }

    #[test]
    fn equal_rows_hash_equally_and_different_rows_do_not() {
        let a: ArrayRef = Arc::new(Int64Array::from(vec![1, 1, 2]));
        let got = hash_of(vec![a], 0);
        assert_eq!(got[0], got[1]);
        assert_ne!(got[0], got[2]);
    }

    #[test]
    fn the_seed_changes_every_digest() {
        let a: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3]));
        assert_ne!(hash_of(vec![Arc::clone(&a)], 0), hash_of(vec![a], 1));
    }

    #[test]
    fn column_order_matters() {
        let a: ArrayRef = Arc::new(Int64Array::from(vec![1]));
        let b: ArrayRef = Arc::new(Int64Array::from(vec![2]));
        assert_ne!(
            hash_of(vec![Arc::clone(&a), Arc::clone(&b)], 0),
            hash_of(vec![b, a], 0)
        );
    }

    /// Values that compare equal must bucket together: `-0.0 == 0.0`, and all NaNs
    /// are one value to the engine's total ordering.
    #[test]
    fn negative_zero_and_nan_are_canonicalized() {
        let a: ArrayRef = Arc::new(Float64Array::from(vec![0.0, -0.0]));
        let got = hash_of(vec![a], 0);
        assert_eq!(got[0], got[1]);

        let n: ArrayRef = Arc::new(Float64Array::from(vec![f64::NAN, -f64::NAN]));
        let got = hash_of(vec![n], 0);
        assert_eq!(got[0], got[1]);
    }

    /// A null is a value, not an absence — otherwise `(1, NULL)` and `(NULL, 1)` collide.
    #[test]
    fn nulls_are_positional_values() {
        let a: ArrayRef = Arc::new(Int64Array::from(vec![Some(1), None]));
        let b: ArrayRef = Arc::new(Int64Array::from(vec![None, Some(1)]));
        let got = hash_of(vec![a, b], 0);
        assert_ne!(got[0], got[1]);

        let ones: ArrayRef = Arc::new(Int64Array::from(vec![1, 1]));
        let both = hash_of(vec![Arc::clone(&ones), ones], 0);
        assert_ne!(got[0], both[0]);
    }

    /// The digest is a wire-visible constant: a change reshuffles every split ever
    /// taken with it, so it is pinned rather than merely tested for properties.
    #[test]
    fn golden_digests_are_stable() {
        let a: ArrayRef = Arc::new(Int64Array::from(vec![0, 1, -1]));
        assert_eq!(
            hash_of(vec![a], 0),
            vec![
                7_776_768_183_763_457_969,
                460_422_991_341_443_459,
                -1_695_834_481_542_859_942
            ]
        );
    }

    #[test]
    fn list_columns_hash_by_their_elements() {
        use arrow::array::ListArray;
        use arrow::datatypes::Int64Type;
        let a: ArrayRef = Arc::new(ListArray::from_iter_primitive::<Int64Type, _, _>(vec![
            Some(vec![Some(1), Some(2)]),
            Some(vec![Some(1), Some(2)]),
            Some(vec![Some(2), Some(1)]),
            Some(vec![Some(1)]),
            None,
        ]));
        let got = hash_of(vec![a], 0);
        assert_eq!(got[0], got[1], "equal lists must hash equally");
        assert_ne!(got[0], got[2], "element order matters");
        assert_ne!(got[0], got[3], "length matters");
        assert_ne!(got[3], got[4], "a null list is not an empty one");
    }

    /// A packed training sequence / embedding is a `FixedSizeList`, which has no `Utf8`
    /// cast at all — the textual fallback would fail outright on it.
    #[test]
    fn fixed_size_list_columns_hash_by_their_elements() {
        use arrow::array::FixedSizeListArray;
        use arrow::datatypes::Int64Type;
        let a: ArrayRef = Arc::new(FixedSizeListArray::from_iter_primitive::<Int64Type, _, _>(
            vec![
                Some(vec![Some(1), Some(2)]),
                Some(vec![Some(1), Some(2)]),
                Some(vec![Some(3), Some(4)]),
            ],
            2,
        ));
        let got = hash_of(vec![a], 0);
        assert_eq!(got[0], got[1]);
        assert_ne!(got[0], got[2]);
    }

    #[test]
    fn hashes_spread_across_the_range() {
        let a: ArrayRef = Arc::new(Int64Array::from((0..4096).collect::<Vec<i64>>()));
        let got = hash_of(vec![a], 7);
        let mut buckets = [0usize; 8];
        for h in &got {
            buckets[((*h as u64) >> 61) as usize] += 1;
        }
        for b in buckets {
            assert!(b > 380 && b < 640, "uneven buckets: {buckets:?}");
        }
        let distinct: std::collections::HashSet<_> = got.iter().collect();
        assert_eq!(distinct.len(), got.len(), "collision in 4096 distinct keys");
    }
}
