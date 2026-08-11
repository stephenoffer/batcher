//! The cast dtype-*name* grammar — the one place a wire name becomes an Arrow type.
//!
//! `bc_expr::Expr::Cast` carries its target as a raw string, so this grammar is part of
//! the JSON IR contract with the Python control plane (`batcher.plan.types.resolve_dtype`
//! parses the identical grammar, pinned by a parity test).
//!
//! It has two halves. The **fixed** names are a lookup table: `int64`, `string`,
//! `date32`, and their SQL aliases. The **parametrized** names carry their parameters in
//! parentheses — `decimal(12,4)`, `timestamp(us, UTC)`, `time64(ns)`, `duration(s)` —
//! and cannot be enumerated, because there are 38 × 39 legal decimals alone.
//!
//! The parametrized half is why this module exists. A fixed table could not name a
//! decimal at all, so a Parquet money column could be read, summed and compared but never
//! *cast* — `cast("decimal(12,4)")` raised `unknown cast dtype`, and there was no
//! spelling that worked. The same held for every timestamp that was not microseconds and
//! for every time-of-day and duration type. Those are not exotic types; they are what
//! financial and event data arrive as.

use arrow::datatypes::{DataType, TimeUnit};

/// Map a cast dtype *name* to its Arrow `DataType`, or `None` when nothing parses it.
///
/// Accepts a fixed name (`int64`, `string`, …) or a parametrized one
/// (`decimal(p,s)`, `timestamp(unit[, tz])`, `time32(unit)`, `time64(unit)`,
/// `duration(unit)`). Whitespace inside the parentheses is ignored, so both
/// `decimal(12,4)` and `decimal(12, 4)` resolve. Matching is case-sensitive on the type
/// name (the control plane lowercases before it reaches the wire) but a timezone keeps
/// its case, since `UTC` and `America/New_York` are case-sensitive identifiers.
pub fn dtype_from_name(name: &str) -> Option<DataType> {
    if let Some(fixed) = fixed_dtype(name) {
        return Some(fixed);
    }
    let (head, args) = split_parametrized(name)?;
    match head {
        "decimal" | "decimal128" | "numeric" => decimal(&args, false),
        "decimal256" => decimal(&args, true),
        "timestamp" | "datetime" => timestamp(&args),
        "time32" => time_of_day(&args, Some(false)),
        "time64" => time_of_day(&args, Some(true)),
        "time" => time_of_day(&args, None),
        "duration" | "interval" => Some(DataType::Duration(time_unit(args.first()?)?)),
        _ => None,
    }
}

/// The names that take no parameters. Aliases map to the same type as their canonical
/// spelling, and the SQL spellings (`varchar`, `bigint`, `real`) are accepted because
/// that is what a user porting a query reaches for first.
fn fixed_dtype(name: &str) -> Option<DataType> {
    Some(match name {
        "int64" | "long" | "bigint" => DataType::Int64,
        "int32" | "int" | "integer" => DataType::Int32,
        "int16" | "smallint" => DataType::Int16,
        "int8" | "tinyint" => DataType::Int8,
        "uint64" | "ubigint" => DataType::UInt64,
        "uint32" | "uinteger" => DataType::UInt32,
        "uint16" | "usmallint" => DataType::UInt16,
        "uint8" | "utinyint" => DataType::UInt8,
        "float64" | "double" => DataType::Float64,
        "float32" | "float" | "real" => DataType::Float32,
        "float16" | "half" => DataType::Float16,
        "bool" | "boolean" => DataType::Boolean,
        "string" | "utf8" | "varchar" | "text" => DataType::Utf8,
        "large_string" | "large_utf8" => DataType::LargeUtf8,
        "binary" | "blob" | "bytea" => DataType::Binary,
        "large_binary" => DataType::LargeBinary,
        "date" | "date32" => DataType::Date32,
        "date64" => DataType::Date64,
        // The bare, unparametrized spellings keep the resolutions they have always had, so
        // an existing plan's `cast("timestamp")` lowers to exactly the type it used to.
        "timestamp" | "datetime" => DataType::Timestamp(TimeUnit::Microsecond, None),
        "time" | "time64" => DataType::Time64(TimeUnit::Microsecond),
        "time32" => DataType::Time32(TimeUnit::Millisecond),
        "duration" | "interval" => DataType::Duration(TimeUnit::Microsecond),
        // A cast *to* null is how a plan spells "this column is empty of values", which
        // the lattice then lets any other type absorb.
        "null" => DataType::Null,
        _ => return None,
    })
}

/// Split `name(a, b)` into its head and its comma-separated arguments, or `None` when it
/// is not of that shape. The arguments are trimmed but keep their case, because a
/// timezone identifier is case-sensitive where a type name is not.
fn split_parametrized(name: &str) -> Option<(&str, Vec<&str>)> {
    let open = name.find('(')?;
    let rest = name.strip_suffix(')')?;
    let head = name[..open].trim();
    let inner = &rest[open + 1..];
    if head.is_empty() {
        return None;
    }
    let args: Vec<&str> = if inner.trim().is_empty() {
        Vec::new()
    } else {
        inner.split(',').map(str::trim).collect()
    };
    Some((head, args))
}

/// `decimal(p)` (scale 0) or `decimal(p, s)`, bounded by what the width can carry.
///
/// An out-of-range precision returns `None` rather than being clamped: silently building a
/// `decimal(38, 4)` where the caller asked for `decimal(50, 4)` would overflow on the very
/// values the extra digits were requested for.
fn decimal(args: &[&str], wide: bool) -> Option<DataType> {
    let max_precision: u8 = if wide { 76 } else { 38 };
    let precision: u8 = args.first()?.parse().ok()?;
    let scale: i8 = match args.get(1) {
        Some(s) => s.parse().ok()?,
        None => 0,
    };
    if precision == 0 || precision > max_precision {
        return None;
    }
    // Arrow allows a negative scale (a multiplier of ten), but never one exceeding the
    // precision — that would be a number with no integer digits and phantom fractional ones.
    if scale > precision as i8 {
        return None;
    }
    Some(if wide {
        DataType::Decimal256(precision, scale)
    } else {
        DataType::Decimal128(precision, scale)
    })
}

/// `timestamp(unit)` or `timestamp(unit, tz)`.
fn timestamp(args: &[&str]) -> Option<DataType> {
    let unit = time_unit(args.first()?)?;
    let tz = match args.get(1) {
        Some(z) if !z.is_empty() => Some((*z).into()),
        _ => None,
    };
    Some(DataType::Timestamp(unit, tz))
}

/// `time(unit)` / `time32(unit)` / `time64(unit)`.
///
/// Arrow splits time-of-day across two widths by resolution: `Time32` carries seconds and
/// milliseconds, `Time64` microseconds and nanoseconds, and neither can carry the other's
/// units. `wide: None` (the unqualified `time(unit)`) therefore picks the width the unit
/// requires, which is the only spelling a user should have to know. An explicit
/// `time32(us)` names an impossible type and returns `None` rather than silently promoting
/// to `Time64`, since the caller asked for a specific physical width.
fn time_of_day(args: &[&str], wide: Option<bool>) -> Option<DataType> {
    let unit = time_unit(args.first()?)?;
    let needs_64 = matches!(unit, TimeUnit::Microsecond | TimeUnit::Nanosecond);
    match wide {
        Some(true) if needs_64 => Some(DataType::Time64(unit)),
        Some(false) if !needs_64 => Some(DataType::Time32(unit)),
        None if needs_64 => Some(DataType::Time64(unit)),
        None => Some(DataType::Time32(unit)),
        Some(_) => None,
    }
}

/// A time-unit name, in the Arrow spellings plus the words SQL users write.
fn time_unit(name: &str) -> Option<TimeUnit> {
    Some(match name {
        "s" | "sec" | "second" | "seconds" => TimeUnit::Second,
        "ms" | "milli" | "millisecond" | "milliseconds" => TimeUnit::Millisecond,
        "us" | "micro" | "microsecond" | "microseconds" => TimeUnit::Microsecond,
        "ns" | "nano" | "nanosecond" | "nanoseconds" => TimeUnit::Nanosecond,
        _ => return None,
    })
}

/// Every *fixed* dtype name `dtype_from_name` accepts, including aliases.
///
/// The single source of truth the FFI introspection helper (`bc-py`) exposes so the Python
/// `CAST_DTYPES` set can be parity-tested against the live engine vocabulary rather than a
/// snapshot that can rot. The parametrized names are a grammar rather than a set, so they
/// are pinned by round-tripping a list of spellings through both sides instead.
pub const CAST_DTYPE_NAMES: &[&str] = &[
    "int64",
    "long",
    "bigint",
    "int32",
    "int",
    "integer",
    "int16",
    "smallint",
    "int8",
    "tinyint",
    "uint64",
    "ubigint",
    "uint32",
    "uinteger",
    "uint16",
    "usmallint",
    "uint8",
    "utinyint",
    "float64",
    "double",
    "float32",
    "float",
    "real",
    "float16",
    "half",
    "bool",
    "boolean",
    "string",
    "utf8",
    "varchar",
    "text",
    "large_string",
    "large_utf8",
    "binary",
    "blob",
    "bytea",
    "large_binary",
    "date",
    "date32",
    "date64",
    "timestamp",
    "datetime",
    "time",
    "time32",
    "time64",
    "duration",
    "interval",
    "null",
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Every name in the published list must resolve, and the list must not contain a name
    /// the resolver rejects — the two stay in lockstep or the Python parity test lies.
    #[test]
    fn every_listed_fixed_name_resolves() {
        for name in CAST_DTYPE_NAMES {
            assert!(
                dtype_from_name(name).is_some(),
                "CAST_DTYPE_NAMES lists `{name}` but dtype_from_name rejects it"
            );
        }
    }

    /// Aliases collapse to one type, so a user porting SQL and a user writing Arrow names
    /// get the same plan.
    #[test]
    fn aliases_collapse_to_one_type() {
        assert_eq!(dtype_from_name("long"), dtype_from_name("int64"));
        assert_eq!(dtype_from_name("bigint"), dtype_from_name("int64"));
        assert_eq!(dtype_from_name("double"), dtype_from_name("float64"));
        assert_eq!(dtype_from_name("varchar"), Some(DataType::Utf8));
        assert_eq!(dtype_from_name("blob"), Some(DataType::Binary));
    }

    /// The unparametrized spellings keep exactly the types they had before the grammar
    /// existed. A plan already on disk must lower to the same thing it always did.
    #[test]
    fn bare_names_keep_their_historical_types() {
        assert_eq!(dtype_from_name("int64"), Some(DataType::Int64));
        assert_eq!(dtype_from_name("int32"), Some(DataType::Int32));
        assert_eq!(dtype_from_name("float64"), Some(DataType::Float64));
        assert_eq!(dtype_from_name("float32"), Some(DataType::Float32));
        assert_eq!(dtype_from_name("bool"), Some(DataType::Boolean));
        assert_eq!(dtype_from_name("string"), Some(DataType::Utf8));
        assert_eq!(dtype_from_name("date"), Some(DataType::Date32));
        assert_eq!(
            dtype_from_name("timestamp"),
            Some(DataType::Timestamp(TimeUnit::Microsecond, None))
        );
        assert_eq!(
            dtype_from_name("datetime"),
            Some(DataType::Timestamp(TimeUnit::Microsecond, None))
        );
    }

    /// A decimal names its precision and scale, with scale defaulting to 0, and spaces
    /// inside the parentheses are ignored.
    #[test]
    fn decimals_parse_their_precision_and_scale() {
        assert_eq!(
            dtype_from_name("decimal(12,4)"),
            Some(DataType::Decimal128(12, 4))
        );
        assert_eq!(
            dtype_from_name("decimal(12, 4)"),
            Some(DataType::Decimal128(12, 4))
        );
        assert_eq!(
            dtype_from_name("numeric(9)"),
            Some(DataType::Decimal128(9, 0))
        );
        assert_eq!(
            dtype_from_name("decimal256(50, 10)"),
            Some(DataType::Decimal256(50, 10))
        );
    }

    /// A precision the width cannot carry is rejected rather than clamped: clamping would
    /// overflow on exactly the values the extra digits were asked for.
    #[test]
    fn an_out_of_range_decimal_is_rejected_not_clamped() {
        assert_eq!(dtype_from_name("decimal(39,2)"), None);
        assert_eq!(dtype_from_name("decimal(0,0)"), None);
        assert_eq!(dtype_from_name("decimal(4,6)"), None);
        assert_eq!(dtype_from_name("decimal256(77,2)"), None);
        assert_eq!(dtype_from_name("decimal(x,2)"), None);
    }

    /// A timestamp names its resolution and, optionally, its zone.
    #[test]
    fn timestamps_parse_their_unit_and_zone() {
        assert_eq!(
            dtype_from_name("timestamp(ns)"),
            Some(DataType::Timestamp(TimeUnit::Nanosecond, None))
        );
        assert_eq!(
            dtype_from_name("timestamp(us, UTC)"),
            Some(DataType::Timestamp(
                TimeUnit::Microsecond,
                Some("UTC".into())
            ))
        );
        assert_eq!(
            dtype_from_name("timestamp(ms, America/New_York)"),
            Some(DataType::Timestamp(
                TimeUnit::Millisecond,
                Some("America/New_York".into())
            ))
        );
        // The zone keeps its case where the type name does not.
        assert_ne!(
            dtype_from_name("timestamp(us, utc)"),
            dtype_from_name("timestamp(us, UTC)")
        );
    }

    /// Time-of-day picks the width its resolution requires when unqualified, and an
    /// explicit width that cannot carry the unit is rejected rather than silently promoted.
    #[test]
    fn time_of_day_picks_the_width_its_unit_requires() {
        assert_eq!(
            dtype_from_name("time(us)"),
            Some(DataType::Time64(TimeUnit::Microsecond))
        );
        assert_eq!(
            dtype_from_name("time(ms)"),
            Some(DataType::Time32(TimeUnit::Millisecond))
        );
        assert_eq!(
            dtype_from_name("time64(ns)"),
            Some(DataType::Time64(TimeUnit::Nanosecond))
        );
        assert_eq!(
            dtype_from_name("time32(s)"),
            Some(DataType::Time32(TimeUnit::Second))
        );
        // Time32 cannot carry microseconds, and saying so beats quietly widening.
        assert_eq!(dtype_from_name("time32(us)"), None);
        assert_eq!(dtype_from_name("time64(s)"), None);
    }

    /// Durations parse their unit, in the Arrow spelling or the SQL word.
    #[test]
    fn durations_parse_their_unit() {
        assert_eq!(
            dtype_from_name("duration(s)"),
            Some(DataType::Duration(TimeUnit::Second))
        );
        assert_eq!(
            dtype_from_name("duration(nanosecond)"),
            Some(DataType::Duration(TimeUnit::Nanosecond))
        );
        assert_eq!(dtype_from_name("duration(fortnight)"), None);
    }

    /// Malformed spellings resolve to nothing, so the caller raises a named error rather
    /// than casting to something the user did not ask for.
    #[test]
    fn malformed_names_resolve_to_nothing() {
        for bad in [
            "",
            "decimal",  // bare `decimal` is not a type — precision is required
            "decimal(", // unclosed
            "decimal)",
            "(12,4)", // no head
            "timestamp()",
            "int64(4)",
            "not_a_type",
            "DECIMAL(12,4)", // the control plane lowercases before the wire
        ] {
            assert_eq!(dtype_from_name(bad), None, "`{bad}` should not resolve");
        }
    }
}
