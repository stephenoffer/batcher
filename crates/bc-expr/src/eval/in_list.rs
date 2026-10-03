//! `x IN (lit, lit, …)` — hash-set membership.
//!
//! Replaces the O(N·k) `(x = l0) OR (x = l1) OR …` chain the SQL front end would
//! otherwise build with a single hash-set lookup per row (O(N) total). Null input →
//! null, matching the OR-of-equals Kleene semantics it folds from (a null never
//! equals any literal, and `NULL OR NULL = NULL`). This is also the kernel a runtime
//! join filter uses to prune a probe side by the build side's key set.

use std::hash::Hash;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, BooleanArray};
use arrow::buffer::BooleanBuffer;
use arrow::datatypes::{DataType, Date32Type, Float64Type, Int64Type};
use arrow::error::ArrowError;
use bc_arrow::canon_f64_bits;

use crate::{BinaryOp, ExprError, Literal};

/// The set type for the hashed path — see [`crate::eval::FastSet`] for why it is not
/// `std::collections::HashSet`, and for the measurement that moved it.
use crate::eval::FastSet;

/// At/below this many members, a linear scan of the set beats hashing every input row.
///
/// The per-row cost of a hash set is a full hash (`std`'s SipHash for strings — tens of
/// cycles) plus a probe; a linear scan is `len` equality compares, each a length check
/// then a short `memcmp`/integer compare that rejects in one or two operations. For a
/// tiny set (the overwhelmingly common `IN ('MAIL', 'SHIP')` / `IN (1, 2, 3)` shape) the
/// scan drops the per-row hash entirely — the win where the input column is cache-resident
/// and the kernel is compute-bound. (When the column is large and streamed from RAM the
/// filter is memory-bandwidth-bound on materializing the values, so this trims
/// instructions without moving wall-clock — e.g. TPC-H Q12's 60M-row `l_shipmode` scan.)
/// Above the threshold the hash set's O(1) probe dominates, so the set is built and
/// probed as before.
const LINEAR_SCAN_MAX: usize = 8;

/// A membership test over a set of `T`: a linear scan when the set is tiny
/// ([`LINEAR_SCAN_MAX`]), a hash set otherwise. `contains` is identical either way (set
/// membership is method-independent), so the produced mask is bit-for-bit unchanged.
enum Members<T> {
    Linear(Vec<T>),
    Hashed(FastSet<T>),
}

impl<T: Hash + Eq> Members<T> {
    fn new(items: Vec<T>) -> Self {
        if items.len() <= LINEAR_SCAN_MAX {
            Members::Linear(items)
        } else {
            Members::Hashed(items.into_iter().collect())
        }
    }

    #[inline]
    fn contains(&self, value: &T) -> bool {
        match self {
            Members::Linear(items) => items.iter().any(|m| m == value),
            Members::Hashed(set) => set.contains(value),
        }
    }
}

/// A membership test over an *ordered* domain, guarded by the set's `[min, max]`.
///
/// A value outside the members' range cannot be a member, so two predictable compares reject
/// it without hashing. That costs an in-range row two compares it did not pay before, and
/// saves an out-of-range row the whole hash — which is the shape that matters here, because
/// the sets this kernel is handed by a pushed-down join filter are a narrow key range probed
/// by a whole fact table. Bit-identical either way: the bounds only ever short-circuit a
/// `false` the set lookup would have returned anyway.
///
/// Only for types with a total order matching equality — integers and dates. Floats are
/// keyed by their raw bit pattern (see the `Float64` arm), which is not ordered like the
/// values, and strings would pay a `memcmp` against each bound to save one hash.
struct Ranged<T> {
    members: Members<T>,
    /// `None` for an empty set: nothing is a member.
    bounds: Option<(T, T)>,
    /// One bit per value in `[lo, hi]`, when that span is at most [`BITMAP_MAX_SPAN`] wide.
    bitmap: Option<Vec<u64>>,
}

/// Widest `[min, max]` span a set is held as a bitmap for: 64K bits, 8 KB, L1-resident.
///
/// A dashboard's `supplier IN (1, 7, 42, …)` or a join filter's narrow key range is a set of
/// small integers, and a bit test answers it with one load and no hash. Measured on TPC-H
/// `l_suppkey IN (11 keys)` the hashed probe was the largest single cost of the filter.
const BITMAP_MAX_SPAN: i64 = 1 << 16;

impl<T: Hash + Eq + Ord + Copy + Into<i64>> Ranged<T> {
    fn new(items: Vec<T>) -> Self {
        let bounds = items.iter().copied().min().zip(items.iter().copied().max());
        let bitmap = bounds.and_then(|(lo, hi)| {
            let (lo, hi): (i64, i64) = (lo.into(), hi.into());
            // `checked_sub`: the span of `i64::MIN..=i64::MAX` does not fit, and is not small.
            let span = hi.checked_sub(lo).filter(|&s| s < BITMAP_MAX_SPAN)? as usize + 1;
            let mut words = vec![0u64; span.div_ceil(64)];
            for &v in &items {
                let off = (v.into() - lo) as usize;
                words[off / 64] |= 1 << (off % 64);
            }
            Some(words)
        });
        Self {
            members: Members::new(items),
            bounds,
            bitmap,
        }
    }

    #[inline]
    fn contains(&self, value: T) -> bool {
        match self.bounds {
            Some((lo, hi)) => value >= lo && value <= hi && self.members.contains(&value),
            None => false,
        }
    }

    /// One membership bit per value (nulls included; the caller masks them).
    ///
    /// With a bitmap the test is branch-free -- an offset, a clamped word load and a bit --
    /// and packed by `cmp::fill`, so it vectorizes where a probe per row cannot. The offset is
    /// taken with wrapping arithmetic: a value below `lo` wraps past the span and reads false,
    /// and no value of `i64` can wrap back *into* a span of at most [`BITMAP_MAX_SPAN`], since
    /// that would need a value at least 2^64 below `lo`.
    fn mask(&self, values: &[T]) -> BooleanBuffer {
        match (&self.bitmap, self.bounds) {
            (Some(words), Some((lo, _))) => {
                let lo: i64 = lo.into();
                let last = (words.len() * 64 - 1) as u64;
                crate::eval::cmp::fill(values, |v| {
                    let off = v.into().wrapping_sub(lo) as u64;
                    let word = words[(off.min(last) / 64) as usize];
                    (off <= last) & ((word >> (off % 64)) & 1 != 0)
                })
            }
            _ => BooleanBuffer::collect_bool(values.len(), |i| self.contains(values[i])),
        }
    }
}

/// `utf8_col IN (...)` over a small set of members of at most seven bytes, as integer compares.
///
/// Each row is first reduced to one `u64` key: its bytes little-endian, zero above its length,
/// with the length in the top byte -- which a string of at most seven bytes leaves free, so the
/// key is injective on such strings -- and `u64::MAX` for a row of eight bytes or more, which
/// no member's key can equal (a member's top byte is its length, at most seven). Membership is
/// then a fixed number of `u64` equalities per row, which `cmp::fill` packs without a branch.
/// The slice path it replaces paid a `memcmp` call per member of the row's length, behind an
/// unpredictable branch on each: 5.2 ns a row on a three-member set, against ~1.5 here.
///
/// `None` for a set past [`LINEAR_SCAN_MAX`] or with a member of eight bytes or more, which
/// keep the slice path.
fn short_str_membership(a: &arrow::array::StringArray, items: &[&str]) -> Option<BooleanArray> {
    if items.len() > LINEAR_SCAN_MAX || items.iter().any(|m| m.len() > 7) {
        return None;
    }
    // Padded to a fixed width with a key no row produces (top byte 0xFF, and not `u64::MAX`),
    // so the compare loop has a constant trip count.
    let mut members = [u64::MAX - 1; LINEAR_SCAN_MAX];
    for (slot, m) in members.iter_mut().zip(items) {
        *slot = short_key(m.as_bytes(), 0, m.len());
    }
    let values = a.values().as_slice();
    let keys: Vec<u64> = a
        .value_offsets()
        .windows(2)
        .map(|w| short_key(values, w[0] as usize, (w[1] - w[0]) as usize))
        .collect();
    let bits = crate::eval::cmp::fill(&keys, |k| {
        members.iter().fold(false, |hit, &m| hit | (m == k))
    });
    Some(masked(bits, a.nulls()))
}

/// The `u64` key [`short_str_membership`] compares: the bytes, then the length in the top byte;
/// `u64::MAX` past seven bytes.
#[inline(always)]
fn short_key(bytes: &[u8], start: usize, len: usize) -> u64 {
    if len > 7 {
        return u64::MAX;
    }
    le_word(bytes, start, len) | ((len as u64) << 56)
}

/// The first `min(len, 8)` bytes of `bytes[start..]`, little-endian, zero above them.
#[inline(always)]
fn le_word(bytes: &[u8], start: usize, len: usize) -> u64 {
    let keep = len.min(8);
    let word = match bytes.get(start..start + 8) {
        Some(eight) => u64::from_le_bytes(eight.try_into().expect("eight bytes")),
        None => {
            let mut buf = [0u8; 8];
            buf[..keep].copy_from_slice(&bytes[start..start + keep]);
            return u64::from_le_bytes(buf);
        }
    };
    // A shift by 64 is undefined, so eight bytes keep the whole word.
    word & u64::MAX.checked_shr(64 - 8 * keep as u32).unwrap_or(0)
}

/// Evaluate `array IN set` to a `BooleanArray` (null where `array` is null).
/// Every literal converted under one arm's extractor, or `None` if *any* of them does not.
///
/// The typed arms below are accelerations of the `=` chain, not a second semantics, so an
/// arm may only be taken when it can represent the whole set. Dropping the members it
/// cannot represent is what made `date_col IN ('2000-06-30', '2000-09-27')` return **no
/// rows**: `literal_date` accepts only `Literal::Date`, both string literals were filtered
/// away, and the set was empty. The single-literal spelling was correct the whole time,
/// because one equality is never folded into an `InList` — so `d = '…'` matched and
/// `d IN ('…', '…')` did not. `Expr.is_in` on the public API had it too, including
/// `is_in([1.0, 2.0])` against an `Int64` column.
///
/// Returning `None` sends the call to `membership_generic`, which is the OR-of-equality the
/// fold collapsed from and therefore carries `eval_binary`'s coercions — the property the
/// fallback arm below already relies on, applied to the typed arms as well.
fn all_converted<'a, T>(
    set: &'a [Literal],
    f: impl Fn(&'a Literal) -> Option<T>,
) -> Option<Vec<T>> {
    set.iter().map(f).collect()
}

pub(crate) fn eval_in_list(array: &ArrayRef, set: &[Literal]) -> Result<ArrayRef, ExprError> {
    let out: BooleanArray = match array.data_type() {
        DataType::Int64 => {
            let a = array.as_primitive::<Int64Type>();
            let Some(items) = all_converted(set, literal_i64) else {
                return membership_generic(array, set);
            };
            let members = Ranged::new(items);
            masked(members.mask(a.values()), a.nulls())
        }
        DataType::Date32 => {
            let a = array.as_primitive::<Date32Type>();
            let Some(items) = all_converted(set, literal_date) else {
                return membership_generic(array, set);
            };
            let members = Ranged::new(items);
            masked(members.mask(a.values()), a.nulls())
        }
        // A float column reaches `InList` both from a user's `is_in` and from the fold rule,
        // which collapses a chain of `float_col = <int literal>` disjuncts into one. Either
        // way `x IN (a, b)` must mean `x = a OR x = b`, and `=` compares by float identity
        // (`bc_arrow::float_ident`): `-0.0` equals `0.0`, and NaN equals NaN. Membership is
        // therefore keyed by the *canonical* bits on both sides. Raw bits made `-0.0 IN (0)`
        // false while `-0.0 = 0` was true -- so a single-element list, which the optimizer
        // rewrites to `=`, kept a row that the same list with a second member dropped.
        DataType::Float64 => {
            let a = array.as_primitive::<Float64Type>();
            let Some(items) = all_converted(set, |l| literal_f64(l).map(canon_f64_bits)) else {
                return membership_generic(array, set);
            };
            let members = Members::new(items);
            membership(a.nulls(), a.len(), |i| {
                members.contains(&canon_f64_bits(a.value(i)))
            })
        }
        DataType::Utf8 => {
            let a = array.as_string::<i32>();
            let Some(items) = all_converted(set, literal_str) else {
                return membership_generic(array, set);
            };
            if let Some(out) = short_str_membership(a, &items) {
                return Ok(Arc::new(out));
            }
            let members = Members::new(items);
            membership(a.nulls(), a.len(), |i| members.contains(&a.value(i)))
        }
        // A dictionary-encoded column: evaluate membership on the *dictionary values* (a
        // handful of distinct entries) once, then gather one bit per row through the keys.
        // This is the classic dict-accelerated `IN` — O(distinct + rows) rather than
        // O(rows) full-value probes — and is bit-identical to the decoded path: `take`
        // maps a null key to a null output (a null row) and a null dictionary value to its
        // null membership bit, exactly matching `NULL IN set = NULL`.
        DataType::Dictionary(_, _) => {
            let dict = array.as_any_dictionary();
            let member_over_values = eval_in_list(dict.values(), set)?;
            let out = arrow::compute::take(&member_over_values, dict.keys(), None)?;
            return Ok(out);
        }
        // Any other column type — a timestamp against date literals, a decimal price against
        // integer literals, a boolean. The fold rule that emits `InList` is a
        // predicate-*shape* rewrite: it sees literals, not the column's dtype (which it
        // cannot know without the schema), so it can hand this kernel a type the typed arms
        // above do not accelerate. Delegating to the very form it folded from is what keeps
        // the rewrite unconditionally safe — the answer here is `eval_binary`'s, including
        // its coercions, so `IN` can neither refuse a pair `=` accepts nor invent one it
        // rejects. Before that arm existed this returned "in_list unsupported for {dtype}",
        // which failed queries the unfolded chain ran happily.
        _ => return membership_generic(array, set),
    };
    Ok(Arc::new(out))
}

/// Membership over a column type the typed arms do not cover, by the OR-of-equality the
/// fold collapsed from.
///
/// Compares the column against each literal with the *same* `eval_binary` the `col = lit`
/// path uses — so coercion, float canonicalization, and type promotion are whatever that
/// path does — then ORs the result bits and re-applies the input's null mask. That mask is
/// the whole of the null story: every member is a non-null literal, so a row is null
/// exactly when its input is, which is `NULL IN set = NULL`. A literal of a kind the
/// column cannot be compared to surfaces `eval_binary`'s own error rather than a bespoke
/// one, and a member the typed arms would have skipped (a `Literal` of the wrong kind)
/// simply never matches, matching their `filter_map`.
fn membership_generic(array: &ArrayRef, set: &[Literal]) -> Result<ArrayRef, ExprError> {
    let n = array.len();
    let mut hit = BooleanBuffer::new_unset(n);
    for member in set {
        let eq = crate::eval::binary::eval_binary(BinaryOp::Eq, array, &member.to_array(n))?;
        let eq = eq.as_any().downcast_ref::<BooleanArray>().ok_or_else(|| {
            ArrowError::ComputeError(format!(
                "in_list: `=` on {:?} is not boolean",
                array.data_type()
            ))
        })?;
        // `values()` ignores `eq`'s nulls, i.e. treats "null = literal" as "no match".
        // Correct here because the null rows are re-masked below.
        hit = &hit | eq.values();
    }
    Ok(Arc::new(BooleanArray::new(hit, array.nulls().cloned())))
}

/// One bool per row: `null` where the input is null, else whether the value is a member.
///
/// `contains` is called for **every** slot, including null ones, and the result is masked
/// afterwards. That is deliberate: testing validity per row makes the loop unpredictably
/// branchy, while `collect_bool` fills the mask 64 bits at a time with no per-row branch, and
/// the arrow accessors are in-bounds at a null slot (a primitive reads its buffer; a string's
/// offsets are valid for every slot), so reading one is defined — its answer is simply thrown
/// away. ANDing the values with the validity bitmap keeps the *value* bits zero under a null,
/// so the output is bit-for-bit what the per-row `valid(i).then(…)` build produced.
fn membership(
    nulls: Option<&arrow::buffer::NullBuffer>,
    n: usize,
    contains: impl Fn(usize) -> bool,
) -> BooleanArray {
    let values = BooleanBuffer::collect_bool(n, contains);
    match nulls {
        None => BooleanArray::new(values, None),
        Some(nb) => BooleanArray::new(&values & nb.inner(), Some(nb.clone())),
    }
}

/// A membership mask with the input's validity applied, as [`membership`] produces it.
fn masked(values: BooleanBuffer, nulls: Option<&arrow::buffer::NullBuffer>) -> BooleanArray {
    match nulls {
        None => BooleanArray::new(values, None),
        Some(nb) => BooleanArray::new(&values & nb.inner(), Some(nb.clone())),
    }
}

/// Whether `x IN set` is answered without probing a hash table: an integer or date set small
/// enough to scan or narrow enough to hold as a bitmap. `Expr::eval_cost` prices these like a
/// comparison, which is what they cost; a hashed or string set keeps its higher price.
pub(crate) fn is_direct(set: &[Literal]) -> bool {
    let bounds = |vals: Vec<i64>| {
        vals.iter()
            .min()
            .zip(vals.iter().max())
            .is_some_and(|(lo, hi)| hi.checked_sub(*lo).is_some_and(|s| s < BITMAP_MAX_SPAN))
    };
    if set.len() <= LINEAR_SCAN_MAX {
        return set
            .iter()
            .all(|l| matches!(l, Literal::Int(_)) || matches!(l, Literal::Date(_)))
            && !set.is_empty();
    }
    if let Some(ints) = all_converted(set, literal_i64) {
        return bounds(ints);
    }
    all_converted(set, literal_date)
        .is_some_and(|days| bounds(days.into_iter().map(i64::from).collect()))
}

fn literal_i64(lit: &Literal) -> Option<i64> {
    match lit {
        Literal::Int(v) => Some(*v),
        _ => None,
    }
}

fn literal_f64(lit: &Literal) -> Option<f64> {
    match lit {
        // An integer literal promotes to `f64` exactly as the folded `col = lit` compare
        // does (both lose precision identically above 2^53).
        Literal::Int(v) => Some(*v as f64),
        Literal::Float(v) => Some(*v),
        _ => None,
    }
}

fn literal_date(lit: &Literal) -> Option<i32> {
    match lit {
        Literal::Date(v) => Some(*v),
        _ => None,
    }
}

fn literal_str(lit: &Literal) -> Option<&str> {
    match lit {
        Literal::Str(v) => Some(v.as_str()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::{Int64Array, StringArray};

    use super::*;

    fn run(arr: ArrayRef, set: &[Literal]) -> Vec<Option<bool>> {
        let out = eval_in_list(&arr, set).unwrap();
        let b = out.as_boolean();
        (0..b.len())
            .map(|i| (!b.is_null(i)).then(|| b.value(i)))
            .collect()
    }

    #[test]
    fn int_membership_with_nulls() {
        let arr: ArrayRef = Arc::new(Int64Array::from(vec![Some(1), Some(2), None, Some(5)]));
        let set = [Literal::Int(1), Literal::Int(5)];
        // 1 ∈ set, 2 ∉, null → null, 5 ∈
        assert_eq!(
            run(arr, &set),
            vec![Some(true), Some(false), None, Some(true)]
        );
    }

    #[test]
    fn str_membership() {
        let arr: ArrayRef = Arc::new(StringArray::from(vec![Some("13"), Some("99"), None]));
        let set = [Literal::Str("13".into()), Literal::Str("31".into())];
        assert_eq!(run(arr, &set), vec![Some(true), Some(false), None]);
    }

    /// A decimal column against integer literals — the shape the fold rule emits without
    /// knowing the dtype (it sees foldable `int` literals; the column they are compared to
    /// is whatever the file said). It has no typed arm, so it takes `membership_generic`,
    /// which must answer what the `col = lit` chain it folded from would have — including
    /// the null, and including `eval_binary`'s scale alignment, which is the whole reason
    /// this cannot be a bespoke comparison.
    #[test]
    fn decimal_column_against_int_literals_matches_the_or_chain() {
        use arrow::array::Decimal128Array;
        // Scale 2: 1.00, 2.00, null, 5.00 — against the integer literals 1 and 5.
        let arr: ArrayRef = Arc::new(
            Decimal128Array::from(vec![Some(100), Some(200), None, Some(500)])
                .with_precision_and_scale(10, 2)
                .unwrap(),
        );
        let set = [Literal::Int(1), Literal::Int(5)];
        assert_eq!(
            run(arr, &set),
            vec![Some(true), Some(false), None, Some(true)]
        );
    }

    /// A timestamp column against date literals — the shape the fold rule emits without
    /// knowing the dtype (a `date` literal is foldable; the column it is compared to may well
    /// be a Timestamp). It has no typed arm, so it takes `membership_generic`, which must
    /// answer what the `col = lit` chain it folded from would have.
    ///
    /// This is the case that pushed the DATE-to-TIMESTAMP widening into `eval_binary`: arrow
    /// rejects `Timestamp == Date32` outright, so before that both the chain and the folded
    /// form raised, while DuckDB answers the query. `tests/differential/test_diff_in_list.py`
    /// pins the end-to-end result against DuckDB; this pins the kernel.
    #[test]
    fn timestamp_column_against_date_literals_matches_the_or_chain() {
        use arrow::array::TimestampMicrosecondArray;
        // 1970-01-02 and 1970-01-03 as microseconds; day 1 is in the set, day 2 is not.
        let day = 86_400_000_000i64;
        let arr: ArrayRef = Arc::new(TimestampMicrosecondArray::from(vec![
            Some(day),
            Some(2 * day),
            None,
        ]));
        let set = [Literal::Date(1), Literal::Date(5)];
        assert_eq!(run(arr, &set), vec![Some(true), Some(false), None]);
    }

    /// The widening is to midnight, not a truncation of the timestamp to its date: a stamp
    /// *within* the matching day is not a member. That is DuckDB's answer and SQL's, and it
    /// is the direction that cannot lose information.
    #[test]
    fn a_timestamp_inside_the_day_is_not_a_member_of_that_date() {
        use arrow::array::TimestampMicrosecondArray;
        let day = 86_400_000_000i64;
        let noon = day + day / 2;
        let arr: ArrayRef = Arc::new(TimestampMicrosecondArray::from(vec![Some(noon), Some(day)]));
        let set = [Literal::Date(1), Literal::Date(5)];
        assert_eq!(run(arr, &set), vec![Some(false), Some(true)]);
    }

    /// The generic arm must be bit-identical to the typed one where both apply, so the
    /// fallback can never become a second, subtly different membership semantics.
    #[test]
    fn generic_arm_agrees_with_the_typed_arm() {
        let arr: ArrayRef = Arc::new(Int64Array::from(vec![Some(1), Some(2), None, Some(5)]));
        let set = [Literal::Int(1), Literal::Int(5)];
        let typed = eval_in_list(&arr, &set).unwrap();
        let generic = membership_generic(&arr, &set).unwrap();
        assert_eq!(&typed, &generic);
    }

    #[test]
    fn empty_set_is_all_false_or_null() {
        let arr: ArrayRef = Arc::new(Int64Array::from(vec![Some(1), None]));
        assert_eq!(run(arr, &[]), vec![Some(false), None]);
    }

    #[test]
    fn large_set_uses_hashed_path_identically() {
        // A set past LINEAR_SCAN_MAX takes the HashSet branch; membership must match the
        // linear branch exactly. 12 members (> 8), probing values in and out of the set.
        let set: Vec<Literal> = (0..12).map(Literal::Int).collect();
        let arr: ArrayRef = Arc::new(Int64Array::from(vec![Some(0), Some(11), Some(12), None]));
        assert_eq!(
            run(arr, &set),
            vec![Some(true), Some(true), Some(false), None]
        );
    }

    #[test]
    fn dictionary_in_list_equals_decoded() {
        use arrow::array::DictionaryArray;
        use arrow::datatypes::Int32Type;
        // A low-cardinality Utf8 dictionary column with a null row.
        let values = ["MAIL", "SHIP", "AIR", "RAIL"];
        let dict: DictionaryArray<Int32Type> =
            [Some("MAIL"), Some("AIR"), None, Some("SHIP"), Some("RAIL")]
                .into_iter()
                .collect();
        let _ = values;
        let dict_arr: ArrayRef = Arc::new(dict.clone());
        let decoded: ArrayRef = arrow::compute::cast(&dict_arr, &arrow::datatypes::DataType::Utf8)
            .expect("decode dict");
        let set = [Literal::Str("MAIL".into()), Literal::Str("SHIP".into())];
        // The dict-accelerated path must equal the decoded full-value path, bit for bit.
        assert_eq!(run(dict_arr, &set), run(decoded, &set));
        // And the expected values: MAIL∈, AIR∉, null→null, SHIP∈, RAIL∉.
        assert_eq!(
            run(Arc::new(dict), &set),
            vec![Some(true), Some(false), None, Some(true), Some(false)]
        );
    }

    /// A float column can reach `InList` (the fold collapses `float_col = <int>` chains, and
    /// `is_in` produces one directly). It must not error, and must agree with `=`, which
    /// compares by float identity: `-0.0` matches `0`, as `-0.0 = 0` does. This test once
    /// pinned the opposite on the premise that `col = lit` compared raw bits; it does not --
    /// the unoptimized `f = 0` keeps `-0.0`, and the optimizer-invariance property test found
    /// `f IN (0)` (rewritten to `=`) and the bare `InList` returning different rows.
    #[test]
    fn float_membership_matches_float_identity_equality() {
        use arrow::array::Float64Array;
        let arr: ArrayRef = Arc::new(Float64Array::from(vec![
            Some(1.0),
            Some(2.0),
            Some(3.0),
            Some(-0.0),
            Some(0.0),
            Some(f64::NAN),
            None,
        ]));
        // The fold only produces integer-valued literals for a float column.
        let set = [Literal::Int(0), Literal::Int(1), Literal::Int(2)];
        assert_eq!(
            run(arr, &set),
            vec![
                Some(true),  // 1.0 ∈
                Some(true),  // 2.0 ∈
                Some(false), // 3.0 ∉
                Some(true),  // -0.0 matches literal 0, as `-0.0 = 0` does
                Some(true),  // 0.0 matches literal 0
                Some(false), // NaN is not in a set without NaN
                None,        // null → null
            ]
        );
    }

    /// The set side is canonicalized too: a `-0.0` literal matches both zeros, and a NaN
    /// literal matches a NaN of any bit pattern, as `=` does.
    #[test]
    fn float_set_literals_are_canonicalized() {
        use arrow::array::Float64Array;
        let other_nan = f64::from_bits(0x7ff0_0000_0000_0001);
        let arr: ArrayRef = Arc::new(Float64Array::from(vec![
            Some(0.0),
            Some(-0.0),
            Some(f64::NAN),
            Some(other_nan),
            Some(1.0),
        ]));
        let set = [Literal::Float(-0.0), Literal::Float(f64::NAN)];
        assert_eq!(
            run(arr, &set),
            vec![Some(true), Some(true), Some(true), Some(true), Some(false)]
        );
    }

    #[test]
    fn float_membership_uses_hashed_path_past_threshold() {
        use arrow::array::Float64Array;
        // > LINEAR_SCAN_MAX members exercises the HashSet<u64> branch identically.
        let set: Vec<Literal> = (0..12).map(Literal::Int).collect();
        let arr: ArrayRef = Arc::new(Float64Array::from(vec![
            Some(0.0),
            Some(11.0),
            Some(12.0),
            None,
        ]));
        assert_eq!(
            run(arr, &set),
            vec![Some(true), Some(true), Some(false), None]
        );
    }

    #[test]
    fn small_and_large_string_sets_agree() {
        let arr: ArrayRef = Arc::new(StringArray::from(vec![Some("MAIL"), Some("AIR"), None]));
        let small = [Literal::Str("MAIL".into()), Literal::Str("SHIP".into())];
        // Pad to > LINEAR_SCAN_MAX so the same values take the hashed path.
        let mut large = small.to_vec();
        large.extend((0..10).map(|i| Literal::Str(format!("X{i}"))));
        assert_eq!(
            run(arr.clone(), &small),
            vec![Some(true), Some(false), None]
        );
        assert_eq!(run(arr, &large), vec![Some(true), Some(false), None]);
    }

    /// A set the typed arm cannot represent must fall back, not silently shrink.
    ///
    /// `literal_date` accepts only `Literal::Date`, so string members were filtered away
    /// and `date_col IN ('2000-06-30', '2000-09-27')` matched nothing at all. The fallback
    /// is the OR-of-equality this kernel folds from, which coerces the string exactly as
    /// `col = '2000-06-30'` does — and that spelling was always correct, which is what made
    /// the bug so quiet.
    #[test]
    fn a_string_literal_against_a_date_column_still_matches() {
        use arrow::array::Date32Array;
        // 11138 = 2000-06-30, 11227 = 2000-09-27 (days since epoch).
        let arr: ArrayRef = Arc::new(Date32Array::from(vec![Some(11138), Some(11227), None]));
        let set = [
            Literal::Str("2000-06-30".into()),
            Literal::Str("2000-09-27".into()),
        ];
        assert_eq!(run(arr, &set), vec![Some(true), Some(true), None]);
    }

    /// The same defect on the numeric arms: a float literal against an `Int64` column.
    #[test]
    fn a_float_literal_against_an_int_column_still_matches() {
        let arr: ArrayRef = Arc::new(Int64Array::from(vec![Some(1), Some(2), Some(3)]));
        let set = [Literal::Float(1.0), Literal::Float(2.0)];
        assert_eq!(run(arr, &set), vec![Some(true), Some(true), Some(false)]);
    }

    /// A mixed set must not lose the members the typed arm *can* hold either.
    #[test]
    fn a_mixed_set_keeps_every_member() {
        let arr: ArrayRef = Arc::new(Int64Array::from(vec![Some(1), Some(2), Some(7)]));
        let set = [Literal::Int(1), Literal::Float(2.0)];
        assert_eq!(run(arr, &set), vec![Some(true), Some(true), Some(false)]);
    }

    /// The typed fast path is still taken for a homogeneous set — the fallback is only for
    /// sets it cannot represent, so this must stay bit-identical to the pre-existing arms.
    #[test]
    fn a_homogeneous_set_is_unchanged() {
        let arr: ArrayRef = Arc::new(Int64Array::from(vec![Some(1), Some(5), Some(9), None]));
        let set = [Literal::Int(1), Literal::Int(5)];
        assert_eq!(
            run(arr, &set),
            vec![Some(true), Some(true), Some(false), None]
        );
    }
}

#[cfg(test)]
mod bitmap_and_short_key_tests {
    use arrow::array::{Date32Array, Int64Array, StringArray};

    use super::*;

    fn agrees(arr: ArrayRef, set: &[Literal]) {
        let typed = eval_in_list(&arr, set).unwrap();
        let generic = membership_generic(&arr, set).unwrap();
        assert_eq!(&typed, &generic, "set={set:?}");
    }

    /// Spans just under, at and over the bitmap's limit, negative and extreme values, every
    /// row a member, a near-member or an outsider — held to the OR-of-equality oracle.
    #[test]
    fn the_bitmap_agrees_with_the_or_chain_across_its_size_limit() {
        for span in [
            0,
            1,
            63,
            64,
            65,
            BITMAP_MAX_SPAN - 1,
            BITMAP_MAX_SPAN,
            BITMAP_MAX_SPAN + 1,
        ] {
            for base in [-70_000_i64, -1, 0, 5, i64::MAX - span, i64::MIN] {
                let members = [base, base + span / 3, base + span];
                let set: Vec<Literal> = members.iter().map(|&v| Literal::Int(v)).collect();
                let mut rows: Vec<Option<i64>> = members
                    .iter()
                    .flat_map(|&m| [m.checked_sub(1), Some(m), m.checked_add(1)])
                    .collect();
                rows.extend([None, Some(i64::MIN), Some(i64::MAX), Some(0)]);
                agrees(Arc::new(Int64Array::from(rows)), &set);
            }
        }
        // The extreme span cannot be a bitmap at all.
        let set = [Literal::Int(i64::MIN), Literal::Int(i64::MAX)];
        let rows = vec![Some(i64::MIN), Some(0), Some(i64::MAX), None];
        agrees(Arc::new(Int64Array::from(rows)), &set);
    }

    #[test]
    fn the_bitmap_serves_dates_before_and_after_the_epoch() {
        let set: Vec<Literal> = [-719_528, -1, 0, 11_016, 11_017]
            .iter()
            .map(|&d| Literal::Date(d))
            .collect();
        let rows: Vec<Option<i32>> = (-10..12_000)
            .step_by(7)
            .map(Some)
            .chain([
                Some(-719_528),
                Some(11_016),
                None,
                Some(i32::MIN),
                Some(i32::MAX),
            ])
            .collect();
        agrees(Arc::new(Date32Array::from(rows)), &set);
    }

    /// Members and rows straddling every edge of the eight-byte key: empty, one to nine bytes,
    /// embedded and trailing NULs (which zero padding must not confuse with an end), shared
    /// prefixes, multi-byte UTF-8, a value sliced away from offset zero, and nulls.
    #[test]
    fn short_string_keys_agree_with_the_or_chain() {
        let rows: Vec<Option<&str>> = vec![
            Some(""),
            Some("A"),
            Some("AIR"),
            Some("AIR\0"),
            Some("AI"),
            Some("MAIL"),
            Some("SHIP"),
            Some("REG AIR"),
            Some("REG AIRX"),
            Some("REG AIRXY"),
            Some("é"),
            Some("日本"),
            None,
            Some("SHIPS"),
            // Eight bytes whose last would be the length byte of "AIR" if the key kept it.
            Some("AIR\0\0\0\0\x03"),
            Some("REG AI\0"),
        ];
        let sets: Vec<Vec<&str>> = vec![
            vec!["MAIL", "SHIP", "AIR"],
            vec!["", "AIR\0", "REG AIR"],
            // An eight-byte member keeps the slice path.
            vec!["", "AIR\0", "REG AIRX"],
            vec!["é", "日本", "A"],
            // Past the linear-scan size, which keeps the hashed slice path.
            vec!["A", "B", "C", "D", "E", "F", "G", "H", "AIR", "SHIP"],
            // A member too long for a key keeps the slice path.
            vec!["REG AIRXY", "AIR"],
        ];
        let arr = StringArray::from(rows);
        for set in sets {
            let lits: Vec<Literal> = set.iter().map(|s| Literal::Str((*s).into())).collect();
            agrees(Arc::new(arr.clone()), &lits);
            agrees(Arc::new(arr.slice(2, 9)), &lits);
        }
        let long = StringArray::from(vec!["REG AIRX"]);
        assert!(short_str_membership(&long, &["REG AIRX"]).is_none());
        assert!(short_str_membership(&long, &["REG AIR"]).is_some());
    }

    /// The sets priced like a comparison are exactly the ones the kernel answers without a
    /// hash probe: a few integers or dates, or a narrow range of them.
    #[test]
    fn direct_sets_are_small_or_narrow_integer_sets() {
        let ints = |v: &[i64]| v.iter().map(|&x| Literal::Int(x)).collect::<Vec<_>>();
        assert!(is_direct(&ints(&[1, 7, 42])));
        assert!(is_direct(&ints(&[
            1, 7, 42, 99, 123, 256, 512, 1024, 2048, 4096, 8192
        ])));
        let wide: Vec<i64> = (0..20).map(|i| i * 100_000).collect();
        assert!(!is_direct(&ints(&wide)));
        assert!(!is_direct(&[Literal::Str("MAIL".into())]));
        assert!(!is_direct(&[]));
        let days: Vec<Literal> = (0..20).map(|d| Literal::Date(d * 30)).collect();
        assert!(is_direct(&days));
    }
}
