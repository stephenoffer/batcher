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
}

impl<T: Hash + Eq + Ord + Copy> Ranged<T> {
    fn new(items: Vec<T>) -> Self {
        let bounds = items.iter().copied().min().zip(items.iter().copied().max());
        Self {
            members: Members::new(items),
            bounds,
        }
    }

    #[inline]
    fn contains(&self, value: T) -> bool {
        match self.bounds {
            Some((lo, hi)) => value >= lo && value <= hi && self.members.contains(&value),
            None => false,
        }
    }
}

/// Evaluate `array IN set` to a `BooleanArray` (null where `array` is null).
pub(crate) fn eval_in_list(array: &ArrayRef, set: &[Literal]) -> Result<ArrayRef, ExprError> {
    let out: BooleanArray = match array.data_type() {
        DataType::Int64 => {
            let a = array.as_primitive::<Int64Type>();
            let members = Ranged::new(set.iter().filter_map(literal_i64).collect());
            membership(a.nulls(), a.len(), |i| members.contains(a.value(i)))
        }
        DataType::Date32 => {
            let a = array.as_primitive::<Date32Type>();
            let members = Ranged::new(set.iter().filter_map(literal_date).collect());
            membership(a.nulls(), a.len(), |i| members.contains(a.value(i)))
        }
        // A float column can reach `InList`: the fold rule collapses a chain of
        // `float_col = <int literal>` disjuncts (integers are foldable literals) into an
        // `InList` without inspecting the column's type. Membership is keyed by the raw
        // 64-bit pattern so it is bit-for-bit identical to the `col = lit` path it folds
        // from: that path compares by total order (`-0.0 != 0.0`, `NaN` matches nothing),
        // and total-order equality *is* bit equality. The fold only ever produces
        // integer-valued literals (0.0, 1.0, …, all canonical positive bits), so a column
        // `-0.0`/`NaN` correctly never lands in the set.
        DataType::Float64 => {
            let a = array.as_primitive::<Float64Type>();
            let members = Members::new(
                set.iter()
                    .filter_map(literal_f64)
                    .map(f64::to_bits)
                    .collect(),
            );
            membership(a.nulls(), a.len(), |i| {
                members.contains(&a.value(i).to_bits())
            })
        }
        DataType::Utf8 => {
            let a = array.as_string::<i32>();
            let members = Members::new(set.iter().filter_map(literal_str).collect());
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

    /// A float column can reach `InList` (the fold collapses `float_col = <int>` chains).
    /// It must not error, and must match the `col = lit` total-order semantics it folds
    /// from: `-0.0` does not match integer `0`, and `NaN` matches nothing.
    #[test]
    fn float_membership_matches_total_order_equality() {
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
                Some(false), // -0.0 does NOT match literal 0 (total order, like `col = 0`)
                Some(true),  // 0.0 matches literal 0
                Some(false), // NaN matches nothing
                None,        // null → null
            ]
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
}
