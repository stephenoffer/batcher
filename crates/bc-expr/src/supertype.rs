//! The common-supertype lattice over Arrow types — one answer for every tier.
//!
//! Three places in the engine have to answer "what single type holds both of these?":
//! the binary/`coalesce`/`greatest` operand coercion in `eval::binary`, the set-operation
//! branch coercion in `bc-interp`, and the Python control plane's `Dataset.schema`
//! inference (`plan.types.lattice.promote`). They used to answer it three times, and
//! they disagreed: the control plane advertised `int64` for `NULL UNION int64` while the
//! engine rejected the query, and a `decimal(10,2)` column compared against a
//! `decimal(12,4)` one raised `Invalid comparison operation` on data a Parquet ingest
//! produces routinely.
//!
//! This module is the engine's single answer. The Python lattice mirrors it deliberately
//! (see that module's docstring) — a control plane that predicts a different type than
//! the engine produces makes `Dataset.schema` a lie, which is worse than an error.
//!
//! The rule is **never narrow either side**. Every promotion here is lossless, with one
//! deliberate exception inherited from SQL: an int/float mix meets at `Float64`, which is
//! what DuckDB does (`SELECT 1 UNION SELECT 1.5` is `DOUBLE`). When there is no such
//! type, the answer is `None` and the caller raises a typed error rather than guessing.

use arrow::datatypes::{DataType, TimeUnit};

/// The widest of two `TimeUnit`s (`Second` < `Millisecond` < `Microsecond` < `Nanosecond`).
///
/// "Widest" means finest resolution: converting a coarser unit to a finer one is exact,
/// so both sides survive. The reverse would truncate.
fn finer(a: &TimeUnit, b: &TimeUnit) -> TimeUnit {
    let rank = |u: &TimeUnit| match u {
        TimeUnit::Second => 0,
        TimeUnit::Millisecond => 1,
        TimeUnit::Microsecond => 2,
        TimeUnit::Nanosecond => 3,
    };
    if rank(a) >= rank(b) {
        *a
    } else {
        *b
    }
}

/// The maximum precision a `Decimal128` can carry.
const DECIMAL128_MAX_PRECISION: u8 = 38;

/// Digits an `Int64` needs when it is widened into a decimal (`i64::MAX` has 19).
const INT64_DECIMAL_DIGITS: u8 = 19;

/// Unify two decimals the way SQL does: keep the finer scale, and keep enough integer
/// digits for whichever side had more.
///
/// `decimal(10,2)` and `decimal(12,4)` have 8 and 8 integer digits; the result keeps 8
/// integer digits and scale 4, so `decimal(12,4)` — exactly what DuckDB returns. Returns
/// `None` when the union would need more than 38 digits, because the alternatives are
/// silently truncating the scale or overflowing on the cast, and a typed error naming the
/// two types is more useful than either.
fn unify_decimal(p1: u8, s1: i8, p2: u8, s2: i8) -> Option<DataType> {
    let scale = s1.max(s2);
    let int_digits = (p1 as i16 - s1 as i16).max(p2 as i16 - s2 as i16);
    let precision = int_digits + scale as i16;
    if precision <= 0 || precision > DECIMAL128_MAX_PRECISION as i16 {
        return None;
    }
    Some(DataType::Decimal128(precision as u8, scale))
}

/// True for the integer types the engine can see (the FFI boundary widens narrow and
/// unsigned integers to `Int64`, but a batch read straight from a file has not been
/// through it, so accept the whole family).
fn is_int(t: &DataType) -> bool {
    use DataType::*;
    matches!(
        t,
        Int8 | Int16 | Int32 | Int64 | UInt8 | UInt16 | UInt32 | UInt64
    )
}

fn is_float(t: &DataType) -> bool {
    matches!(t, DataType::Float16 | DataType::Float32 | DataType::Float64)
}

/// The single type both `a` and `b` widen into without either being narrowed, or `None`
/// when no such type exists.
///
/// `None` is the honest answer, not a failure to try: an `Int64`/`Utf8` pair has no
/// lossless common type, and callers turn it into a typed error naming both sides. The
/// one place this deliberately departs from "lossless" is the int/float mix, which meets
/// at `Float64` as every SQL dialect does.
pub fn common_supertype(a: &DataType, b: &DataType) -> Option<DataType> {
    use DataType::*;
    if a == b {
        return Some(a.clone());
    }
    // `Null` is the type of a column that held nothing — an all-null Parquet column, a
    // `lit(None)`, a branch of a UNION that selected no typed value. It carries no values
    // to lose, so it adopts whatever it is paired with. Without this arm an all-null
    // column could not be compared, coalesced, or unioned with anything at all.
    if matches!(a, Null) {
        return Some(b.clone());
    }
    if matches!(b, Null) {
        return Some(a.clone());
    }
    // A dictionary is an *encoding* of its value type, not a distinct logical type; both a
    // dict-encoded and a plain column read as the value type (which is what the FFI
    // boundary already produces). Unwrap and unify the values.
    if let Dictionary(_, v) = a {
        return common_supertype(v, b);
    }
    if let Dictionary(_, v) = b {
        return common_supertype(a, v);
    }
    match (a, b) {
        // Numerics. A float on either side wins (DuckDB promotes int ∪ float to DOUBLE);
        // otherwise two integers meet at Int64, which holds every narrow and unsigned
        // width the engine can see.
        _ if is_float(a) || is_float(b) => {
            let numeric_or_decimal = |t: &DataType| {
                is_float(t) || is_int(t) || matches!(t, Boolean | Decimal128(..) | Decimal256(..))
            };
            if numeric_or_decimal(a) && numeric_or_decimal(b) {
                // A decimal meeting a float goes to Float64 too: DOUBLE dominates DECIMAL
                // in DuckDB, and casting the float down to the decimal's scale would
                // truncate its sub-scale precision.
                Some(Float64)
            } else {
                None
            }
        }
        _ if is_int(a) && is_int(b) => Some(Int64),
        // Boolean widens into a number, as DuckDB does (`SELECT true UNION SELECT 1` is
        // INTEGER, with `true` reading as 1). It never widens into anything else.
        _ if matches!(a, Boolean) && is_int(b) => Some(Int64),
        _ if is_int(a) && matches!(b, Boolean) => Some(Int64),
        (Decimal128(p1, s1), Decimal128(p2, s2)) => unify_decimal(*p1, *s1, *p2, *s2),
        // An integer widened into a decimal keeps the decimal exact, which is why this is
        // not routed through Float64: a money column summed against a count must not lose
        // its last cents to a float round-trip. `Boolean` is deliberately NOT included:
        // arrow has no bool→decimal cast, so naming a decimal here would advertise a type
        // the engine cannot actually produce, which is the one failure worse than
        // declining. (A property test over the whole lattice caught exactly that.)
        (Decimal128(p, s), _) if is_int(b) => unify_decimal(*p, *s, INT64_DECIMAL_DIGITS, 0),
        (_, Decimal128(p, s)) if is_int(a) => unify_decimal(INT64_DECIMAL_DIGITS, 0, *p, *s),
        // Temporal. Two instants in the same zone meet at the finer resolution, so a
        // `timestamp[ms]` file and a `timestamp[us]` file reconcile instead of raising.
        // A *differing* zone is a genuine semantic disagreement about which instant a
        // value denotes, so it is declined here; `eval::binary` handles the specific
        // aware-vs-naive comparison shape with its own documented zone-stripping rule.
        (Timestamp(u1, tz1), Timestamp(u2, tz2)) if tz1 == tz2 => {
            Some(Timestamp(finer(u1, u2), tz1.clone()))
        }
        // A date is a timestamp at midnight; widening the date side is exact, and DuckDB
        // likewise returns TIMESTAMP for `DATE UNION TIMESTAMP`.
        (Date32 | Date64, Timestamp(u, tz)) => Some(Timestamp(*u, tz.clone())),
        (Timestamp(u, tz), Date32 | Date64) => Some(Timestamp(*u, tz.clone())),
        // Date64 is milliseconds where Date32 is days: the wider of the two.
        (Date32, Date64) | (Date64, Date32) => Some(Date64),
        // Time-of-day. Time32 carries second/millisecond, Time64 microsecond/nanosecond,
        // so any pairing across the two meets in Time64 and a pairing within one stays.
        (Time32(u1), Time32(u2)) => Some(Time32(finer(u1, u2))),
        (Time64(u1), Time64(u2)) => Some(Time64(finer(u1, u2))),
        (Time32(_), Time64(u)) | (Time64(u), Time32(_)) => Some(Time64(*u)),
        (Duration(u1), Duration(u2)) => Some(Duration(finer(u1, u2))),
        // Same logical type, wider offsets — always lossless, exactly as int32/int64 widen.
        (Utf8, LargeUtf8) | (LargeUtf8, Utf8) => Some(LargeUtf8),
        (Binary, LargeBinary) | (LargeBinary, Binary) => Some(LargeBinary),
        // Everything else — an Int64/Utf8 pair, two timestamps in different zones, a
        // struct against a list — has no lossless common type. Say so.
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::TimeUnit::{Microsecond, Millisecond, Nanosecond, Second};

    /// An all-null column adopts whatever it meets, in either operand position. This is
    /// the arm that makes `coalesce(all_null_col, int_col)` and `NULL UNION int` work at
    /// all — both raised `Invalid argument error` before it existed.
    #[test]
    fn null_adopts_the_other_side() {
        assert_eq!(
            common_supertype(&DataType::Null, &DataType::Int64),
            Some(DataType::Int64)
        );
        assert_eq!(
            common_supertype(&DataType::Utf8, &DataType::Null),
            Some(DataType::Utf8)
        );
        assert_eq!(
            common_supertype(&DataType::Null, &DataType::Null),
            Some(DataType::Null)
        );
    }

    /// Numeric promotion matches DuckDB: a float on either side wins, two integers of any
    /// width meet at Int64.
    #[test]
    fn numerics_promote_like_duckdb() {
        assert_eq!(
            common_supertype(&DataType::Int64, &DataType::Float64),
            Some(DataType::Float64)
        );
        assert_eq!(
            common_supertype(&DataType::Int32, &DataType::UInt8),
            Some(DataType::Int64)
        );
        assert_eq!(
            common_supertype(&DataType::Float32, &DataType::Float64),
            Some(DataType::Float64)
        );
        assert_eq!(
            common_supertype(&DataType::Boolean, &DataType::Int64),
            Some(DataType::Int64)
        );
    }

    /// `decimal(10,2)` and `decimal(12,4)` unify to `decimal(12,4)` — 8 integer digits
    /// (the max of both sides') plus the finer scale — which is what DuckDB returns for
    /// the same union. Neither side is narrowed.
    #[test]
    fn decimals_keep_the_finer_scale_and_the_wider_integer_part() {
        assert_eq!(
            common_supertype(&DataType::Decimal128(10, 2), &DataType::Decimal128(12, 4)),
            Some(DataType::Decimal128(12, 4))
        );
        // Wider integer part, coarser scale: 10 integer digits + scale 4 = precision 14.
        assert_eq!(
            common_supertype(&DataType::Decimal128(14, 4), &DataType::Decimal128(10, 0)),
            Some(DataType::Decimal128(14, 4))
        );
    }

    /// An integer widened into a decimal stays exact rather than going through Float64,
    /// so a money column keeps its cents. Int64 needs 19 integer digits.
    #[test]
    fn integer_widens_into_the_decimal_not_into_a_float() {
        assert_eq!(
            common_supertype(&DataType::Decimal128(10, 2), &DataType::Int64),
            Some(DataType::Decimal128(21, 2))
        );
        // A float, by contrast, dominates the decimal (DuckDB: DOUBLE beats DECIMAL).
        assert_eq!(
            common_supertype(&DataType::Decimal128(10, 2), &DataType::Float64),
            Some(DataType::Float64)
        );
    }

    /// Past 38 digits there is no Decimal128 that holds both, and the honest answer is
    /// `None` so the caller raises rather than truncating the scale.
    #[test]
    fn a_decimal_union_that_cannot_fit_declines() {
        assert_eq!(
            common_supertype(&DataType::Decimal128(38, 0), &DataType::Decimal128(38, 10)),
            None
        );
    }

    /// Timestamps in the same zone meet at the finer unit; a date widens into a timestamp.
    #[test]
    fn temporal_types_widen_to_the_finer_resolution() {
        assert_eq!(
            common_supertype(
                &DataType::Timestamp(Millisecond, None),
                &DataType::Timestamp(Microsecond, None)
            ),
            Some(DataType::Timestamp(Microsecond, None))
        );
        assert_eq!(
            common_supertype(
                &DataType::Date32,
                &DataType::Timestamp(Nanosecond, Some("UTC".into()))
            ),
            Some(DataType::Timestamp(Nanosecond, Some("UTC".into())))
        );
        assert_eq!(
            common_supertype(&DataType::Time32(Second), &DataType::Time64(Microsecond)),
            Some(DataType::Time64(Microsecond))
        );
        assert_eq!(
            common_supertype(&DataType::Duration(Second), &DataType::Duration(Nanosecond)),
            Some(DataType::Duration(Nanosecond))
        );
    }

    /// Two timestamps in *different* zones denote different instants for the same stored
    /// value, so there is no common type and the pair is declined.
    #[test]
    fn timestamps_in_different_zones_decline() {
        assert_eq!(
            common_supertype(
                &DataType::Timestamp(Microsecond, Some("UTC".into())),
                &DataType::Timestamp(Microsecond, None)
            ),
            None
        );
    }

    /// A dictionary is an encoding, not a type: it unifies through its value type, which
    /// is what both a dict-encoded and a plain column read as.
    #[test]
    fn dictionary_unifies_through_its_value_type() {
        let dict = DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
        assert_eq!(
            common_supertype(&dict, &DataType::Utf8),
            Some(DataType::Utf8)
        );
        assert_eq!(
            common_supertype(&DataType::LargeUtf8, &dict),
            Some(DataType::LargeUtf8)
        );
    }

    /// Wider offsets are lossless, so a `Utf8`/`LargeUtf8` pair meets at the large variant.
    #[test]
    fn offset_widths_widen() {
        assert_eq!(
            common_supertype(&DataType::Utf8, &DataType::LargeUtf8),
            Some(DataType::LargeUtf8)
        );
        assert_eq!(
            common_supertype(&DataType::LargeBinary, &DataType::Binary),
            Some(DataType::LargeBinary)
        );
    }

    /// Genuinely incompatible pairs return `None` so callers raise a typed error instead
    /// of a lenient cast that would silently null the non-conforming side.
    #[test]
    fn incompatible_pairs_decline() {
        assert_eq!(common_supertype(&DataType::Int64, &DataType::Utf8), None);
        assert_eq!(common_supertype(&DataType::Boolean, &DataType::Utf8), None);
        assert_eq!(common_supertype(&DataType::Date32, &DataType::Int64), None);
    }

    /// Every type the lattice has a rule about, so the algebraic properties below
    /// exercise the real arms rather than the identity fast path.
    fn lattice_types() -> Vec<DataType> {
        vec![
            DataType::Null,
            DataType::Boolean,
            DataType::Int32,
            DataType::Int64,
            DataType::UInt8,
            DataType::Float32,
            DataType::Float64,
            DataType::Decimal128(10, 2),
            DataType::Decimal128(12, 4),
            DataType::Utf8,
            DataType::LargeUtf8,
            DataType::Binary,
            DataType::LargeBinary,
            DataType::Date32,
            DataType::Date64,
            DataType::Timestamp(Millisecond, None),
            DataType::Timestamp(Microsecond, None),
            DataType::Timestamp(Microsecond, Some("UTC".into())),
            DataType::Time32(Second),
            DataType::Time64(Nanosecond),
            DataType::Duration(Second),
            DataType::Duration(Nanosecond),
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
        ]
    }

    /// The lattice is commutative — the answer cannot depend on operand order, or two
    /// callers that happen to pass the pair the other way round would disagree.
    #[test]
    fn the_lattice_is_commutative() {
        for a in lattice_types() {
            for b in lattice_types() {
                assert_eq!(
                    common_supertype(&a, &b),
                    common_supertype(&b, &a),
                    "not commutative for {a:?} / {b:?}"
                );
            }
        }
    }

    /// Every type the lattice names must be one arrow can actually cast **both** inputs
    /// into. This is the property the lattice exists to provide, and it is the one a table
    /// of hand-written expectations cannot check: a rule naming an unreachable type looks
    /// plausible in isolation and then fails at the cast, after the read has been paid
    /// for. It has already earned its place — the first draft widened `Boolean` into a
    /// `Decimal128`, for which arrow has no cast at all.
    #[test]
    fn every_supertype_is_reachable_by_cast_from_both_sides() {
        for a in lattice_types() {
            for b in lattice_types() {
                let Some(common) = common_supertype(&a, &b) else {
                    continue;
                };
                for side in [&a, &b] {
                    assert!(
                        arrow::compute::can_cast_types(side, &common),
                        "{side:?} cannot cast to {common:?} (from {a:?} / {b:?})"
                    );
                }
            }
        }
    }

    /// Idempotent: a type paired with itself is itself, for every type. Anything else
    /// would make a no-op union rewrite the schema.
    #[test]
    fn the_lattice_is_idempotent() {
        for t in [
            DataType::Null,
            DataType::Int64,
            DataType::Utf8,
            DataType::Decimal128(10, 2),
            DataType::Timestamp(Microsecond, Some("UTC".into())),
        ] {
            assert_eq!(common_supertype(&t, &t), Some(t.clone()), "for {t:?}");
        }
    }
}
