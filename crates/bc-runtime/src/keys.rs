//! The one canonical form for grouping/partitioning keys.
//!
//! Every path that turns a key column into a hash — the group assigner
//! (`agg::group::assign`), the radix combine (`agg::group::combine`), and the shuffle
//! (`shuffle`) — MUST agree on what makes two keys "the same". They are separate code
//! paths for performance reasons, but they answer one semantic question, so the answer
//! lives here once.
//!
//! Getting this wrong does not merely reorder rows: if the shuffle disagrees with the
//! assigner about key identity, two rows that are one group land on different reducers
//! and the query returns **two groups where the oracle returns one** — a silent
//! wrong-answer bug that only appears once the input is big enough to shuffle. Both
//! sides of that divergence have happened here (a float key split across `-0.0`/`0.0`;
//! null int keys scattered across every bucket), which is why the policy is centralized
//! rather than restated per call site.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Array};
use arrow::compute::SortOptions;
use arrow::datatypes::DataType;
use rayon::prelude::*;
// The engine's single definition of float identity: `-0.0` folds into `0.0` and every NaN
// bit-pattern into one. It lives in `bc-arrow`, the lowest crate this and `bc-expr` share,
// exactly so the grouping keys here and the scalar comparisons there cannot drift apart.
pub(crate) use bc_arrow::{canon_f32, canon_f64_bits as canon_f64, float_total_cmp};
use bc_arrow::{needs_canon_f32, needs_canon_f64};

/// A fixed hash for null keys so every null row lands in one partition — and therefore one
/// group. Grouping inside the partition still compares keys, so a non-null value that
/// collides with this hash is never conflated with null; only co-location depends on it.
pub(crate) const NULL_HASH: u64 = 0xa5a5_5a5a_dead_beef;

/// The hasher every *cross-process* key hash goes through.
///
/// It lives here, beside the rest of the key-identity policy, for the reason stated at the
/// top of this module: the paths that must agree about what makes two keys the same should
/// read their answer from one place instead of restating it.
///
/// It is deliberately **not** `ahash`, which the shuffle used with fixed seeds and a
/// comment claiming that made it deterministic. Fixed seeds make `ahash` deterministic
/// within one binary; it picks an AES-NI backend from the compile-time `target_feature`,
/// so two workers built with different `-C target-cpu` disagree. That was measured, not
/// inferred — compiling this crate with `+aes` moved a single-`Int64`-key batch from
/// buckets `[7,1,1,2,6,7,7,7]` to `[2,0,3,6,5,6,1,7]` with no source change. On a cluster
/// with mixed builds, that splits one `GROUP BY` group across two reducers and drops join
/// matches, silently.
///
/// `bc_arrow::PortableBuildHasher` is specified, endian-pinned, and identical everywhere.
/// See `crates/bc-runtime/tests/shuffle_hash_golden.rs` for the routing vectors it pins.
///
/// **Only for values that leave the process.** Hash-table bucket selection inside one
/// `execute_plan` call — the group assigner, the join build map, `distinct`, spill
/// partitioning — stays on `ahash`, which is faster and cannot be observed from outside.
pub(crate) const SHUFFLE_HASHER: bc_arrow::PortableBuildHasher =
    bc_arrow::PortableBuildHasher::with_seed(0x5348_5546_464C_4530);

/// Whether a data type has a floating-point leaf that needs canonicalizing — a top-level
/// float, a float nested inside a list/struct key, or a float behind a dictionary encoding.
///
/// **The `Dictionary` arm is not symmetry with the others; it closes a silent wrong answer.**
/// A dictionary's identity is its *codes*, and codes are not float identity: `-0.0` and `0.0`
/// are two entries, every NaN bit pattern is its own entry, and nothing forbids a dictionary
/// that simply spells one value twice. A float dictionary reaching this as a key without the
/// arm therefore splits one `GROUP BY` group in two and drops join matches — exactly what this
/// module exists to prevent. It used to be out of reach only because the FFI boundary decoded
/// dictionaries on the way in and `Expr::eval` decoded again at a `Col` leaf; both of those are
/// performance decisions that `rfc-streaming-executor.md` Proposal 3 would reverse, and neither
/// is a place a correctness invariant should be resting.
/// [`canon_array`] answers the arm by decoding, since a canonical *value* under a duplicated
/// code is still two groups. `a_float_dictionary_follows_the_engines_float_identity` in
/// `bc-interp/tests/dictionary_operators.rs` pins it.
fn contains_float(dt: &DataType) -> bool {
    match dt {
        DataType::Float32 | DataType::Float64 => true,
        DataType::List(f) | DataType::LargeList(f) | DataType::FixedSizeList(f, _) => {
            contains_float(f.data_type())
        }
        DataType::Struct(fields) => fields.iter().any(|f| contains_float(f.data_type())),
        DataType::Dictionary(_, value) => contains_float(value),
        _ => false,
    }
}

/// Canonicalize the float leaves of one array, preserving its type, offsets, and nulls.
///
/// Returns `None` when the array holds no float leaf (nothing to rewrite), so a
/// non-float column is passed through with no allocation. Recurses through
/// `List`/`LargeList`/`FixedSizeList`/`Struct` so a float buried in a nested key is folded
/// exactly as a top-level one — the `RowConverter` these keys are later encoded with splits
/// `-0.0`/`0.0` (and every NaN) at *every* depth, not just the top.
fn canon_array(arr: &ArrayRef) -> Option<ArrayRef> {
    use arrow::array::{FixedSizeListArray, Float32Array, LargeListArray, ListArray, StructArray};
    match arr.data_type() {
        DataType::Float64 => {
            let f = arr.as_any().downcast_ref::<Float64Array>()?;
            // Nothing to fold unless a NaN or a `-0.0` is actually present, and on real data
            // neither usually is. Deciding that with a scan of the raw value buffer — a
            // branch-free, auto-vectorized linear read — is far cheaper than the rebuild it
            // guards, which allocates and writes a whole second copy of the column (~48 MB at
            // 6M rows, serially) before anything else in the operator starts. Slots under a
            // null are read here too; a null's payload is arbitrary, so at worst it triggers a
            // rewrite that was not needed, which is still correct.
            if !f.values().iter().any(|x| needs_canon_f64(*x)) {
                return None;
            }
            // `canon_f64` returns hash bits; map back to the f64 they denote so the column
            // keeps its Float64 type. `f64::from_bits` round-trips every value it emits.
            let canon: Float64Array = f
                .iter()
                .map(|v| v.map(|x| f64::from_bits(canon_f64(x))))
                .collect();
            Some(Arc::new(canon) as ArrayRef)
        }
        DataType::Float32 => {
            let f = arr.as_any().downcast_ref::<Float32Array>()?;
            if !f.values().iter().any(|x| needs_canon_f32(*x)) {
                return None;
            }
            let canon: Float32Array = f.iter().map(|v| v.map(canon_f32)).collect();
            Some(Arc::new(canon) as ArrayRef)
        }
        DataType::List(field) => {
            let l = arr.as_any().downcast_ref::<ListArray>()?;
            let child = canon_array(l.values())?;
            Some(Arc::new(ListArray::new(
                field.clone(),
                l.offsets().clone(),
                child,
                l.nulls().cloned(),
            )) as ArrayRef)
        }
        DataType::LargeList(field) => {
            let l = arr.as_any().downcast_ref::<LargeListArray>()?;
            let child = canon_array(l.values())?;
            Some(Arc::new(LargeListArray::new(
                field.clone(),
                l.offsets().clone(),
                child,
                l.nulls().cloned(),
            )) as ArrayRef)
        }
        DataType::FixedSizeList(field, size) => {
            let l = arr.as_any().downcast_ref::<FixedSizeListArray>()?;
            let child = canon_array(l.values())?;
            Some(Arc::new(FixedSizeListArray::new(
                field.clone(),
                *size,
                child,
                l.nulls().cloned(),
            )) as ArrayRef)
        }
        // A float behind a dictionary is canonicalized by **decoding** it, not by rewriting
        // its values in place. Folding the values alone would leave `-0.0` and `0.0` as two
        // codes that now both spell `0.0`, and every path that reaches here hashes or encodes
        // the codes — so the group would still split. Decoding is what `decode_dict_keys` does
        // for the same class of bug on strings, and for the same reason: two dictionaries built
        // independently assign different codes to one value, so codes only compare against a
        // shared dictionary. Unconditional on a float dictionary (not gated on a `-0.0`/NaN
        // being present), because a dictionary may spell one ordinary value under two codes.
        DataType::Dictionary(_, value) if contains_float(value) => {
            // A cast to the dictionary's own value type cannot fail on well-formed input;
            // declining on an error leaves the caller with the array it already had rather
            // than turning a decode into a query failure.
            let decoded = arrow::compute::cast(arr, value).ok()?;
            Some(canon_array(&decoded).unwrap_or(decoded))
        }
        DataType::Struct(fields) => {
            let s = arr.as_any().downcast_ref::<StructArray>()?;
            let mut changed = false;
            let cols: Vec<ArrayRef> = s
                .columns()
                .iter()
                .map(|c| match canon_array(c) {
                    Some(n) => {
                        changed = true;
                        n
                    }
                    None => Arc::clone(c),
                })
                .collect();
            // A struct with no float leaf changed nothing — pass it through untouched.
            changed.then(|| {
                Arc::new(StructArray::new(fields.clone(), cols, s.nulls().cloned())) as ArrayRef
            })
        }
        _ => None,
    }
}

/// Rewrite float key columns into their canonical form, leaving every other column as-is.
///
/// The raw-hash paths call [`canon_f64`] per value, but the general shuffle/join path
/// encodes keys with arrow's `RowConverter`, whose float encoding is *not* canonical: it
/// maps `-0.0` and `0.0` to different bytes (and each NaN bit-pattern to its own), so they
/// would hash to different buckets and never meet at a reducer. Canonicalizing the array up
/// front means every downstream encoder — `RowConverter` included — sees the same bits the
/// assigner groups by, so no key-identity policy has to be restated inside each encoder.
///
/// A float leaf **nested** inside a list or struct key is canonicalized too: the encoder
/// splits `-0.0`/`0.0` at every depth, so a `GROUP BY`/`JOIN`/`DISTINCT` on a `List<Float64>`
/// or `Struct{Float64}` column silently split one group/dropped one match without this.
///
/// Returns `None` when there is nothing to rewrite (the common case: no float key), so the
/// caller keeps using the original arrays with no allocation.
pub(crate) fn canonicalize_float_keys(keys: &[ArrayRef]) -> Option<Vec<ArrayRef>> {
    if !keys.iter().any(|k| contains_float(k.data_type())) {
        return None;
    }
    Some(
        keys.iter()
            .map(|k| canon_array(k).unwrap_or_else(|| Arc::clone(k)))
            .collect(),
    )
}

/// Decode any dictionary-encoded key column to its value type, so two sides of a join agree
/// on *encoding* and not merely on value.
///
/// This belongs beside [`canonicalize_float_keys`] because it is the same class of bug and the
/// same argument: key identity has to be one thing, and an encoding difference is a way for two
/// paths to disagree about it while both being individually correct. The float case is two
/// spellings of one number; this is two spellings of one string.
///
/// It is load-bearing the moment a dictionary can reach an operator, which is what
/// `rfc-streaming-executor.md` Proposal 3 would allow. The concrete failure is not a wrong
/// answer but a hard error, because `arrow::row::RowConverter` is built from one side's type and
/// fed both: "RowConverter column schema mismatch, expected Utf8 got Dictionary(Int32, Utf8)".
/// That arises whenever the two sides are reached by different operator chains — a bare `Scan`
/// on one side and anything that decodes on the other — which is the ordinary shape of a
/// fact-to-dimension join, not a corner case.
///
/// Decoding rather than teaching the join to compare codes is deliberate here: two dictionaries
/// built independently assign different codes to the same value, so codes are only comparable
/// against a shared dictionary. A dictionary-native join key means unifying the dictionaries
/// first, which is a performance project (`competitor_technique_review.md` item 6) rather than
/// the correctness floor this provides.
///
/// Returns `None` when no key is dictionary-encoded — the common case and, until the boundary
/// stops decoding, every case — so the caller allocates nothing.
pub(crate) fn decode_dict_keys(keys: &[ArrayRef]) -> Option<Vec<ArrayRef>> {
    if !keys
        .iter()
        .any(|k| matches!(k.data_type(), DataType::Dictionary(_, _)))
    {
        return None;
    }
    Some(
        keys.iter()
            .map(|k| match k.data_type() {
                DataType::Dictionary(_, value) => {
                    // A cast to the dictionary's own value type cannot fail on well-formed
                    // input; keeping the original on an error leaves the caller exactly as
                    // it was rather than turning a decode into a query failure.
                    arrow::compute::cast(k, value).unwrap_or_else(|_| Arc::clone(k))
                }
                _ => Arc::clone(k),
            })
            .collect(),
    )
}

/// Canonicalize the array component of each ORDER BY key `(array, options)`, folding
/// `-0.0`/`0.0` and every NaN bit pattern exactly as [`canonicalize_float_keys`] does for
/// grouping keys, while preserving each key's `SortOptions`. The order path ranks raw bits
/// (`RowConverter` → `rows_equal` → `peer_boundary`), so without this `-0.0` and `0.0` are not
/// peers and a negative NaN sorts below `-inf` — disagreeing with the `GROUP BY`/`=`/`MIN` the
/// same column feeds. Returns `None` when no key contains a float leaf (the common case), so the
/// caller keeps the originals allocation-free.
pub(crate) fn canonicalize_float_order_keys(
    keys: &[(ArrayRef, SortOptions)],
) -> Option<Vec<(ArrayRef, SortOptions)>> {
    if !keys.iter().any(|(k, _)| contains_float(k.data_type())) {
        return None;
    }
    Some(
        keys.iter()
            .map(|(k, opts)| (canon_array(k).unwrap_or_else(|| Arc::clone(k)), *opts))
            .collect(),
    )
}

/// Element floor above which an elementwise pass over the universe is worth handing to rayon.
///
/// The same trade as [`PARALLEL_SORT_MIN_ROWS`] and deliberately the same size, but it guards a
/// *linear* pass rather than an `n log n` one, so it earns less per element and matters more
/// that it does not fire early. This operator answers a 10,000-row join in under four
/// milliseconds; a few hundred microseconds of pool hand-off would be visible there, while at
/// five million rows a side these passes were 200 ms of a 900 ms join, on one core of 96.
pub(crate) const PARALLEL_MAP_MIN: usize = 32_768;

/// Map a slice elementwise into a fresh `Vec`, on rayon once it is large enough to pay.
///
/// `collect` from an indexed parallel iterator writes each element at the index it was read
/// from, so the result is the sequential `map`'s output element for element. Every caller here
/// passes a pure function of one value, which is what makes that equivalence hold.
pub(crate) fn map_u64<T: Copy + Send + Sync>(
    src: &[T],
    f: impl Fn(T) -> u64 + Send + Sync,
) -> Vec<u64> {
    if src.len() >= PARALLEL_MAP_MIN {
        src.par_iter().map(|&v| f(v)).collect()
    } else {
        src.iter().map(|&v| f(v)).collect()
    }
}

/// An order-preserving `u64` for each value of a primitive key column, or `None` when the
/// type has no such encoding.
///
/// The sorts are ~70% of a two-condition range join's time (`report_range_join_phases`), and
/// almost all of that was comparator overhead rather than the sort itself: comparing two
/// `arrow::row::Row`s is an indirect load plus a `memcmp`, against a register compare here.
/// Null-keyed rows never reach this — they are excluded from the universe up front — so no
/// null byte is needed and the whole key fits in the 64 bits.
///
/// Floats use the standard total-order transform, which lands exactly on this engine's float
/// contract because [`canonicalize_float_keys`](crate::keys::canonicalize_float_keys) has
/// already folded `-0.0` into `0.0` and every NaN into one positive quiet NaN: the transform
/// then ranks that NaN above `+inf`, which is where the rest of the engine puts it.
pub(crate) fn u64_order_keys(arr: &ArrayRef, descending: bool) -> Option<Vec<u64>> {
    use arrow::array::PrimitiveArray;
    use arrow::datatypes::*;

    #[inline]
    fn signed(v: i64) -> u64 {
        (v as u64) ^ (1u64 << 63)
    }
    #[inline]
    fn float(bits: u64) -> u64 {
        if bits >> 63 == 1 {
            !bits
        } else {
            bits ^ (1u64 << 63)
        }
    }

    macro_rules! prim {
        ($ty:ty, $f:expr) => {{
            let a = arr.as_any().downcast_ref::<PrimitiveArray<$ty>>()?;
            #[allow(clippy::redundant_closure_call)]
            Some(map_u64(a.values(), |v| $f(v)))
        }};
    }

    let keys: Option<Vec<u64>> = match arr.data_type() {
        DataType::Int8 => prim!(Int8Type, |v: i8| signed(i64::from(v))),
        DataType::Int16 => prim!(Int16Type, |v: i16| signed(i64::from(v))),
        DataType::Int32 => prim!(Int32Type, |v: i32| signed(i64::from(v))),
        DataType::Int64 => prim!(Int64Type, signed),
        DataType::UInt8 => prim!(UInt8Type, |v: u8| u64::from(v)),
        DataType::UInt16 => prim!(UInt16Type, |v: u16| u64::from(v)),
        DataType::UInt32 => prim!(UInt32Type, |v: u32| u64::from(v)),
        DataType::UInt64 => prim!(UInt64Type, |v: u64| v),
        DataType::Float32 => prim!(Float32Type, |v: f32| float(f64::from(v).to_bits())),
        DataType::Float64 => prim!(Float64Type, |v: f64| float(v.to_bits())),
        DataType::Date32 => prim!(Date32Type, |v: i32| signed(i64::from(v))),
        DataType::Date64 => prim!(Date64Type, signed),
        DataType::Time32(TimeUnit::Second) => {
            prim!(Time32SecondType, |v: i32| signed(i64::from(v)))
        }
        DataType::Time32(TimeUnit::Millisecond) => {
            prim!(Time32MillisecondType, |v: i32| signed(i64::from(v)))
        }
        DataType::Time64(TimeUnit::Microsecond) => prim!(Time64MicrosecondType, signed),
        DataType::Time64(TimeUnit::Nanosecond) => prim!(Time64NanosecondType, signed),
        DataType::Timestamp(TimeUnit::Second, _) => prim!(TimestampSecondType, signed),
        DataType::Timestamp(TimeUnit::Millisecond, _) => prim!(TimestampMillisecondType, signed),
        DataType::Timestamp(TimeUnit::Microsecond, _) => prim!(TimestampMicrosecondType, signed),
        DataType::Timestamp(TimeUnit::Nanosecond, _) => prim!(TimestampNanosecondType, signed),
        DataType::Duration(TimeUnit::Second) => prim!(DurationSecondType, signed),
        DataType::Duration(TimeUnit::Millisecond) => prim!(DurationMillisecondType, signed),
        DataType::Duration(TimeUnit::Microsecond) => prim!(DurationMicrosecondType, signed),
        DataType::Duration(TimeUnit::Nanosecond) => prim!(DurationNanosecondType, signed),
        _ => None,
    };
    keys.map(|mut k| {
        if descending {
            // Complementing an order-preserving key reverses the order, and it is elementwise,
            // so it fans out on the same terms as the map above.
            if k.len() >= PARALLEL_MAP_MIN {
                k.par_iter_mut().for_each(|v| *v = !*v);
            } else {
                for v in &mut k {
                    *v = !*v;
                }
            }
        }
        k
    })
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canon_folds_both_zeros_and_all_nans() {
        assert_eq!(canon_f64(0.0), canon_f64(-0.0));
        assert_eq!(canon_f64(f64::NAN), canon_f64(-f64::NAN));
        assert_eq!(
            canon_f64(f64::from_bits(0x7ff8_0000_0000_0001)),
            canon_f64(f64::NAN)
        );
        // distinct finite values stay distinct
        assert_ne!(canon_f64(1.0), canon_f64(1.5));
        assert_ne!(canon_f64(1.0), canon_f64(-1.0));
    }

    #[test]
    fn canonicalize_rewrites_negative_zero_and_preserves_nulls() {
        let keys: Vec<ArrayRef> = vec![Arc::new(Float64Array::from(vec![
            Some(-0.0),
            Some(0.0),
            None,
            Some(2.5),
        ]))];
        let out = canonicalize_float_keys(&keys).expect("float key present");
        let f = out[0].as_any().downcast_ref::<Float64Array>().unwrap();
        // both zeros now carry the same bits, so any encoder buckets them together
        assert_eq!(f.value(0).to_bits(), f.value(1).to_bits());
        assert!(f.is_null(2));
        assert_eq!(f.value(3), 2.5);
    }

    /// A float **dictionary** key must group by the engine's float identity, not by the
    /// dictionary's codes.
    ///
    /// The adversarial shape is the one a real encoder produces: `-0.0` and `0.0` get
    /// separate entries, each NaN bit pattern gets its own, and an ordinary value can be
    /// spelled twice. Grouping on codes reports one group per *entry*; the oracle — the same
    /// column decoded — reports one per value. Before `contains_float` recursed into
    /// `Dictionary`, this column reached the assigner uncanonicalized and split every one of
    /// those groups. See the note on `contains_float`.
    #[test]
    fn canonicalize_folds_float_dictionary_to_value_identity() {
        use arrow::array::{DictionaryArray, Int32Array};
        use arrow::datatypes::Int32Type;

        // Entries 0 and 1 are two spellings of zero; 2 and 3 two NaN patterns; 4 and 5 the
        // same ordinary value under two codes. Six entries, three values.
        let values: ArrayRef = Arc::new(Float64Array::from(vec![
            0.0,
            -0.0,
            f64::NAN,
            f64::from_bits(0x7ff8_0000_0000_0001),
            1.5,
            1.5,
        ]));
        let codes = Int32Array::from(vec![0, 1, 2, 3, 4, 5]);
        let dict: ArrayRef = Arc::new(
            DictionaryArray::<Int32Type>::try_new(codes, Arc::clone(&values)).expect("dictionary"),
        );

        let canon = canonicalize_float_keys(std::slice::from_ref(&dict))
            .expect("a float dictionary has a float leaf");
        let (_, groups, _) =
            crate::agg::assign_groups(&canon, 6).expect("assign over a canonicalized key");

        // The decoded oracle: {0.0, NaN, 1.5}.
        let (_, oracle, _) = crate::agg::assign_groups(
            std::slice::from_ref(
                &canonicalize_float_keys(std::slice::from_ref(&values)).expect("float leaf")[0],
            ),
            6,
        )
        .expect("assign over the decoded key");
        assert_eq!(
            groups, oracle,
            "dictionary must group as its decoded values do"
        );
        assert_eq!(
            groups, 3,
            "two zeros, two NaNs and a duplicated 1.5 are three groups"
        );
    }

    #[test]
    fn canonicalize_skips_non_float_keys() {
        use arrow::array::Int64Array;
        let keys: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(vec![1, 2, 3]))];
        assert!(canonicalize_float_keys(&keys).is_none());
    }

    /// A `RowConverter` encodes two `Rows` equal iff their canonical bytes match. This
    /// helper drives the actual encoder used by the join/shuffle general path so a test
    /// proves the *observable* key identity, not just the array bits.
    fn rows_equal_after_canon(a: ArrayRef, b: ArrayRef) -> bool {
        use arrow::row::{RowConverter, SortField};
        let dt = a.data_type().clone();
        let ca = canon_array(&a).unwrap_or(a);
        let cb = canon_array(&b).unwrap_or(b);
        let conv = RowConverter::new(vec![SortField::new(dt)]).unwrap();
        let ra = conv.convert_columns(&[ca]).unwrap();
        let rb = conv.convert_columns(&[cb]).unwrap();
        ra.row(0) == rb.row(0)
    }

    /// A float nested in a `List` key must fold `-0.0`/`0.0` (and NaN) — otherwise a join
    /// on a list-of-floats column silently drops the match. This is the bug the recursion
    /// fixes; without it `RowConverter` gives `List[-0.0]` and `List[0.0]` different bytes.
    #[test]
    fn canonicalize_folds_float_inside_list_key() {
        use arrow::array::{Float64Array, ListArray};
        use arrow::buffer::OffsetBuffer;
        use arrow::datatypes::Field;
        let field = Arc::new(Field::new("item", DataType::Float64, true));
        let mk = |v: f64| -> ArrayRef {
            Arc::new(ListArray::new(
                field.clone(),
                OffsetBuffer::new(vec![0, 1].into()),
                Arc::new(Float64Array::from(vec![v])),
                None,
            ))
        };
        assert!(
            rows_equal_after_canon(mk(-0.0), mk(0.0)),
            "List[-0.0] must equal List[0.0]"
        );
        assert!(
            rows_equal_after_canon(mk(f64::NAN), mk(f64::from_bits(0x7ff8_0000_0000_0001))),
            "List[NaN] patterns must unify"
        );
        // Distinct finite values inside a list must stay distinct.
        assert!(!rows_equal_after_canon(mk(1.0), mk(2.0)));
    }

    /// A float nested in a `Struct` key field must fold too (a `GROUP BY struct_col`).
    #[test]
    fn canonicalize_folds_float_inside_struct_key() {
        use arrow::array::{Float64Array, Int64Array, StructArray};
        use arrow::datatypes::{Field, Fields};
        let fields: Fields = vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Float64, false),
        ]
        .into();
        let mk = |f: f64| -> ArrayRef {
            Arc::new(StructArray::new(
                fields.clone(),
                vec![
                    Arc::new(Int64Array::from(vec![7])) as ArrayRef,
                    Arc::new(Float64Array::from(vec![f])) as ArrayRef,
                ],
                None,
            ))
        };
        assert!(
            rows_equal_after_canon(mk(-0.0), mk(0.0)),
            "struct with -0.0 field must equal struct with 0.0 field"
        );
        assert!(!rows_equal_after_canon(mk(1.0), mk(2.0)));
    }

    /// End-to-end: a hash join keyed on a `List<Float64>` must match `-0.0` with `0.0`.
    /// Before the recursion this returned zero rows (a silent dropped match).
    #[test]
    fn join_on_list_float_key_matches_signed_zero() {
        use crate::join::{hash_join_indices, JoinType};
        use arrow::array::{Float64Array, ListArray};
        use arrow::buffer::OffsetBuffer;
        use arrow::datatypes::Field;
        let field = Arc::new(Field::new("item", DataType::Float64, true));
        let mk = |v: f64| -> Vec<ArrayRef> {
            vec![Arc::new(ListArray::new(
                field.clone(),
                OffsetBuffer::new(vec![0, 1].into()),
                Arc::new(Float64Array::from(vec![v])),
                None,
            ))]
        };
        let idx = hash_join_indices(&mk(-0.0), &mk(0.0), JoinType::Inner).unwrap();
        assert_eq!(idx.left.len(), 1, "list<-0.0> must join list<0.0>");
    }

    /// A list of a non-float type is left untouched (no needless allocation / rewrite).
    #[test]
    fn canonicalize_skips_non_float_nested_keys() {
        use arrow::array::{Int64Array, ListArray};
        use arrow::buffer::OffsetBuffer;
        use arrow::datatypes::Field;
        let field = Arc::new(Field::new("item", DataType::Int64, true));
        let list: ArrayRef = Arc::new(ListArray::new(
            field,
            OffsetBuffer::new(vec![0, 1].into()),
            Arc::new(Int64Array::from(vec![1])),
            None,
        ));
        assert!(canonicalize_float_keys(&[list]).is_none());
    }
}
