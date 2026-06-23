//! `bc-expr` — scalar expression IR and its evaluation.
//!
//! There is exactly ONE expression representation in Batcher (`Expr`), and it is
//! the single source consumed by *both* the Tier-0 interpreter and (later) the
//! JIT codegen backends. That shared source is what guarantees semantic parity
//! across execution tiers — the interpreter is the correctness oracle the
//! compiled tiers are differential-tested against, and they can only agree if
//! they evaluate the same IR.
//!
//! Evaluation here is vectorized: an `Expr` is evaluated over a whole Arrow
//! `RecordBatch` (a morsel) at once using arrow compute kernels. Literals are
//! currently materialized to full-length arrays for simplicity; a later pass
//! will switch to Arrow `Datum` scalars + selection vectors for true late
//! materialization.

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanArray, Date32Array, Float64Array, Int64Array, RecordBatch, StringArray,
    TimestampMicrosecondArray,
};
use arrow::compute::kernels::boolean;
use arrow::compute::kernels::zip::zip;
use arrow::compute::{cast_with_options, is_not_null, is_null, CastOptions};
use serde::Deserialize;

mod error;
pub use error::ExprError;

mod eval;
use eval::binary::{as_bool, coerce_numeric, eval_binary, try_scalar_binary};
use eval::date::{
    eval_date, eval_date_offset, eval_date_trunc, eval_strftime, eval_strptime, parse_dtype,
};
use eval::image::eval_image;
use eval::list::{
    eval_array, eval_list, eval_list_binary, eval_list_contains, eval_list_get, eval_list_join,
    eval_struct_field, rebuild_list, require_list,
};
use eval::math::{eval_coalesce, eval_extreme, eval_is_nan, eval_math, eval_math2};
use eval::str::eval_str;

/// A scalar expression over the columns of a record batch.
///
/// Deserialized from the language-agnostic JSON IR emitted by the Python control
/// plane, so the variant tags (`e`, `op`) are the stable wire contract.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "e", rename_all = "snake_case")]
pub enum Expr {
    /// Reference to an input column by name.
    Col { name: String },
    /// A constant literal.
    Lit { value: Literal },
    /// A binary operation over two sub-expressions.
    Binary {
        op: BinaryOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    /// Logical negation of a boolean sub-expression.
    Not { input: Box<Expr> },

    /// Cast a sub-expression to a target Arrow type (by name). `try_cast` selects
    /// DuckDB `TRY_CAST` semantics: a value that cannot be converted yields NULL
    /// instead of erroring the query (arrow `safe` cast). The default (`false`)
    /// is a strict `CAST` that errors on an invalid value.
    Cast {
        input: Box<Expr>,
        dtype: String,
        #[serde(default)]
        try_cast: bool,
    },

    /// Null predicate (true where the argument is null).
    IsNull { input: Box<Expr> },

    /// Non-null predicate.
    IsNotNull { input: Box<Expr> },

    /// IEEE NaN predicate (true where a float value is NaN; null → null). A
    /// first-class op because the `!=` operator uses total ordering (NaN == NaN),
    /// so the `x != x` trick cannot detect NaN. The JIT falls back to interpret it.
    IsNan { input: Box<Expr> },

    /// SQL CASE: the first branch whose `when` is true yields its `then`,
    /// otherwise `otherwise`.
    Case {
        branches: Vec<CaseBranch>,
        otherwise: Box<Expr>,
    },

    /// A string function over a Utf8 sub-expression.
    Str {
        #[serde(rename = "fn")]
        func: StrFunc,
        input: Box<Expr>,
        #[serde(default)]
        pattern: Option<String>,
        #[serde(default)]
        replacement: Option<String>,
        #[serde(default)]
        start: Option<i64>,
        #[serde(default)]
        length: Option<i64>,
    },

    /// A date/time field extraction over a Date/Timestamp sub-expression.
    Date {
        #[serde(rename = "fn")]
        func: DateFunc,
        input: Box<Expr>,
    },

    /// An image decode op over a binary (image-bytes) sub-expression. Decoding is
    /// library-backed (heavy), so the JIT falls back to this interpreter path.
    Image {
        #[serde(rename = "fn")]
        func: ImageFunc,
        input: Box<Expr>,
        #[serde(default)]
        width: Option<i64>,
        #[serde(default)]
        height: Option<i64>,
    },

    /// First non-null among the sub-expressions, per row (SQL COALESCE).
    Coalesce { inputs: Vec<Expr> },

    /// An array literal `[e0, e1, …]` — each row becomes a `List` of the
    /// per-row element values (all elements coerced to a common type).
    Array { elements: Vec<Expr> },

    /// A unary math function over a numeric sub-expression.
    Math {
        #[serde(rename = "fn")]
        func: MathFunc,
        input: Box<Expr>,
    },

    /// A scalar reduction over each row's `List` value (e.g. list length, sum).
    List {
        #[serde(rename = "fn")]
        func: ListFunc,
        input: Box<Expr>,
    },

    /// `NULLIF(left, right)`: null where `left == right`, else `left`.
    #[serde(rename = "nullif")]
    NullIf { left: Box<Expr>, right: Box<Expr> },

    /// `GREATEST(a, b, …)`: the largest argument per row, ignoring nulls.
    Greatest { inputs: Vec<Expr> },

    /// `LEAST(a, b, …)`: the smallest argument per row, ignoring nulls.
    Least { inputs: Vec<Expr> },

    /// A two-argument math function over numeric sub-expressions (→ Float64).
    Math2 {
        #[serde(rename = "fn")]
        func: Math2Func,
        left: Box<Expr>,
        right: Box<Expr>,
    },

    /// `list[index]` — the element at 0-based `index` of each row's `List`
    /// (null where the row is null or the index is out of range). Type-preserving.
    ListGet { input: Box<Expr>, index: i64 },

    /// `struct.field` — extract a named field from a `Struct` column
    /// (type-preserving; null where the struct row is null).
    StructField { input: Box<Expr>, field: String },

    /// `list.contains(value)` — true where any element equals the literal. → Bool.
    ListContains { input: Box<Expr>, value: Literal },

    /// `list.slice(offset, length)` — the 0-based sub-range of each row's `List`.
    ListSlice {
        input: Box<Expr>,
        offset: i64,
        #[serde(default)]
        length: Option<i64>,
    },

    /// `date_trunc(unit, ts)` — truncate a timestamp to the start of `unit`
    /// (year/month/day/hour/minute/second). → Timestamp(us).
    DateTrunc { input: Box<Expr>, unit: String },

    /// `strftime(ts, format)` — format a Date/Timestamp with a chrono/strftime
    /// `format` string (e.g. `%Y-%m-%d`). Null instants format to null. → Utf8.
    Strftime { input: Box<Expr>, format: String },

    /// `strptime(s, format)` — parse a Utf8 column into a Timestamp(microsecond)
    /// using a chrono/strftime `format`. Unparseable values → NULL (DuckDB
    /// `try_strptime`). The inverse of `Strftime`.
    Strptime { input: Box<Expr>, format: String },

    /// `offset_by` — shift a Date32/Timestamp by a calendar+fixed offset. `months`
    /// (incl. years×12) shift calendar months with end-of-month clamping; `days`
    /// (incl. weeks×7) and `micros` are exact. Months/days preserve a Date32;
    /// `micros != 0` on a Date32 errors (sub-day offset has no Date representation).
    /// Type-preserving (Date32→Date32, Timestamp→Timestamp). Null → null.
    DateOffset {
        input: Box<Expr>,
        #[serde(default)]
        months: i64,
        #[serde(default)]
        days: i64,
        #[serde(default)]
        micros: i64,
    },

    /// `list_join(list, sep)` — concatenate each row's `List` elements (cast to
    /// Utf8, nulls skipped) with `separator` → Utf8. Backs SQL `string_agg`.
    ListJoin { input: Box<Expr>, separator: String },

    /// A pairwise reduction over two `List` columns of equal length per row
    /// (`dot`/`cosine_similarity`/`l2_distance`) → Float64. The vector-search
    /// primitives; the query vector is typically a broadcast `array(...)` literal.
    ListBinary {
        #[serde(rename = "fn")]
        func: ListBinaryFunc,
        left: Box<Expr>,
        right: Box<Expr>,
    },
}

/// Pairwise list reductions over two equal-length numeric `List` columns (→ Float64).
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ListBinaryFunc {
    /// Dot product `Σ aᵢ·bᵢ` over the paired elements.
    Dot,
    /// Cosine similarity `dot(a,b) / (‖a‖·‖b‖)`; null if either vector has zero norm.
    CosineSimilarity,
    /// Euclidean distance `sqrt(Σ (aᵢ−bᵢ)²)` between the two vectors.
    L2Distance,
}

/// Two-argument math functions (→ Float64).
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Math2Func {
    /// `pow(a, b)` = a raised to b.
    Pow,
    /// `atan2(y, x)`.
    Atan2,
    /// `round(x, digits)` — round to `digits` decimal places.
    Round,
}

/// Image decode operations for the `.image` namespace. `Decode` reads each
/// image's dimensions into a `{width, height}` struct; `ToTensor` decodes,
/// resizes to `(width, height)`, and flattens to a fixed-size RGB8 pixel list.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageFunc {
    Decode,
    ToTensor,
}

/// Per-row scalar reductions over a `List` column. `len`/`n_unique` → Int64; the
/// numeric reductions (`sum`/`min`/`max`/`mean`) cast elements to Float64. Null
/// list rows stay null; empty lists reduce to null (no elements) except `len`
/// (0) and `n_unique` (0).
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ListFunc {
    Len,
    Sum,
    Min,
    Max,
    Mean,
    NUnique,
    /// Sort each row's list ascending → `List` (same element type).
    Sort,
    /// Reverse each row's list → `List` (same element type).
    Reverse,
    /// Product of (non-null) elements → Float64; empty/null row → null.
    Product,
    /// Sample standard deviation `sqrt(Σ(x-mean)²/(n-1))` → Float64; null when
    /// fewer than 2 non-null elements.
    Std,
    /// Sample variance `Σ(x-mean)²/(n-1)` → Float64; null when n<2.
    Var,
    /// Distinct elements preserving first-occurrence order → `List` (same element
    /// type); null elements are dropped.
    Unique,
    /// Median of the (non-null) elements → Float64; for an even count the average
    /// of the two middle values; empty/null row → null.
    Median,
    /// 0-based index of the minimum non-null element (first on ties) → Int64;
    /// empty/null row → null.
    ArgMin,
    /// 0-based index of the maximum non-null element (first on ties) → Int64;
    /// empty/null row → null.
    ArgMax,
    /// Euclidean (L2) norm `sqrt(Σ xᵢ²)` of the non-null elements → Float64;
    /// empty/null row → null. The vector magnitude used in similarity search.
    L2Norm,
    /// Concatenate a `List<List<T>>` into a `List<T>` per row, in order (DuckDB
    /// `flatten`; Polars `list.explode`-free flatten). Null inner lists are skipped;
    /// a null outer row stays null. Element type `T` is preserved.
    Flatten,
}

/// Unary math functions. `abs` preserves the input numeric type; the rest yield
/// Float64.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MathFunc {
    Abs,
    Round,
    Floor,
    Ceil,
    Sqrt,
    Ln,
    Log10,
    Log2,
    Exp,
    Sin,
    Cos,
    Tan,
    /// −1 / 0 / +1 by sign (0 maps to 0, matching DuckDB `sign`).
    Sign,
    /// Truncate toward zero.
    Trunc,
    /// Cube root.
    Cbrt,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
    /// Radians → degrees.
    Degrees,
    /// Degrees → radians.
    Radians,
    /// Cotangent (1/tan).
    Cot,
}

/// String functions. `upper`/`lower` → Utf8; `len` → Int64; `contains`/
/// `starts_with`/`ends_with` → Boolean; `substr` (1-based, char-oriented) → Utf8.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StrFunc {
    Upper,
    Lower,
    Len,
    Contains,
    StartsWith,
    EndsWith,
    Substr,
    Replace,
    Trim,
    LTrim,
    RTrim,
    Reverse,
    /// Repeat the string `start` times (`start` reused as the count; ≤0 → empty).
    Repeat,
    /// Left-pad to `start` characters with `pattern` (cycled); truncates if longer.
    Lpad,
    /// Right-pad to `start` characters with `pattern` (cycled); truncates if longer.
    Rpad,
    /// 1-based position of `pattern` in the string (0 if absent). → Int64.
    Position,
    /// The last `start` characters (`start` reused as the count).
    Right,
    /// Unicode codepoint of the first character (0 for empty). → Int64.
    Ascii,
    /// Split on `pattern` → a `List<Utf8>` (null input → null list).
    Split,
    /// True where `pattern` (a regex) matches anywhere in the string. → Boolean.
    RegexpMatches,
    /// Replace the first match of regex `pattern` with `replacement`. → Utf8.
    RegexpReplace,
    /// Replace *every* match of regex `pattern` with `replacement` (DuckDB
    /// `regexp_replace(..., 'g')`; Polars `replace_all`). → Utf8.
    RegexpReplaceAll,
    /// `split_part(string, delim, n)`: the `n`-th (1-based) field of the string
    /// split on `pattern` (the delimiter); `''` if `n` is out of range (DuckDB
    /// `split_part`; `start` carries `n`). → Utf8.
    SplitPart,
    /// Extract capture group `start` of regex `pattern` ('' if no match). → Utf8.
    RegexpExtract,
    /// Extract the string value at JSON `pattern` path (e.g. `$.a.b`); null if the
    /// input isn't valid JSON or the path is missing. → Utf8.
    JsonExtractString,
    /// Extract the integer value at JSON `pattern` path; null if the input isn't
    /// valid JSON, the path is missing, or the value isn't integral. → Int64.
    JsonExtractInt,
    /// Extract the numeric value at JSON `pattern` path as a float; null if absent
    /// or non-numeric. → Float64.
    JsonExtractFloat,
    /// Extract the boolean value at JSON `pattern` path; null if absent or
    /// non-boolean. → Boolean.
    JsonExtractBool,
    /// Deterministic FNV-1a 64-bit hash of the UTF-8 bytes (→ Int64; the u64 digest
    /// reinterpreted as i64). Stable across partitions, runs, and machines — the
    /// building block for surrogate keys and slowly-changing-dimension change
    /// detection. Null → null.
    Hash64,
    /// Capitalize the first letter of each word, lowercasing the rest. A word is a
    /// maximal run of alphanumerics (DuckDB `initcap`). → Utf8.
    Initcap,
    /// Number of UTF-8 bytes in the string (`v.len()`; DuckDB `octet_length`). → Int64.
    OctetLength,
    /// Number of bits in the string (bytes × 8; DuckDB `bit_length`). → Int64.
    BitLength,
    /// Uppercase hex of the UTF-8 bytes, e.g. "abc" → "616263" (DuckDB `hex`). → Utf8.
    Hex,
    /// `translate(string, from, to)`: each char that appears at index i of `from`
    /// (`pattern`) is replaced by the char at index i of `to` (`replacement`); if
    /// `to` is shorter, chars in `from` beyond its length are deleted; chars not in
    /// `from` pass through (DuckDB `translate`). → Utf8.
    Translate,
    /// Standard base64 encoding of the UTF-8 bytes (DuckDB `to_base64`). → Utf8.
    Base64,
    /// Decode standard base64 to bytes, then interpret as UTF-8 (DuckDB
    /// `from_base64`). Invalid base64 or non-UTF-8 bytes → null. → Utf8 (nullable).
    FromBase64,
    /// Parse pairs of hex digits to bytes, then interpret as UTF-8 (DuckDB
    /// `unhex`). Odd length, non-hex, or non-UTF-8 bytes → null. → Utf8 (nullable).
    Unhex,
    /// SQL `LIKE`: anchored match where `pattern`'s `%` matches any run of chars,
    /// `_` matches exactly one char, every other char is literal. → Boolean.
    Like,
    /// SQL `ILIKE`: case-insensitive `LIKE`. → Boolean.
    Ilike,
}

/// Date/time field extractions (→ Int64). Wire tags are snake_case (the contract
/// with the Python `.dt` namespace).
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DateFunc {
    Year,
    Month,
    Day,
    Hour,
    Minute,
    Second,
    Quarter,
    /// ISO week of the year (1–53).
    Week,
    /// Day of week with Sunday = 0 (matches DuckDB `dayofweek`).
    DayOfWeek,
    /// Day of the year (1–366).
    DayOfYear,
    /// Seconds since the Unix epoch (DuckDB `epoch`). → Int64.
    Epoch,
    /// Full weekday name e.g. "Monday" (DuckDB `dayname`, chrono `%A`). → Utf8.
    Dayname,
    /// Full month name e.g. "January" (DuckDB `monthname`, chrono `%B`). → Utf8.
    Monthname,
    /// ISO day of week: Monday = 1 … Sunday = 7 (DuckDB `isodow`). → Int64.
    Isodow,
    /// The century, e.g. 2021 → 21, 1999 → 20 (DuckDB `century`). → Int64.
    Century,
    /// The decade, e.g. 2021 → 202 (DuckDB `decade`, `year/10`). → Int64.
    Decade,
    /// The millennium, e.g. 2021 → 3, 2000 → 2 (DuckDB `millennium`,
    /// `(Y-1)/1000 + 1`). → Int64.
    Millennium,
    /// The last day of the month of the instant, at 00:00:00 (DuckDB `last_day`).
    /// → Timestamp(Microsecond) (compare against `last_day(ts)::TIMESTAMP`).
    LastDay,
    /// Whether the instant's year is a leap year (DuckDB `isfinite`-style predicate
    /// `extract('isoyear')`-independent). → Boolean.
    IsLeapYear,
    /// Number of days in the instant's month, 28–31 (DuckDB
    /// `days_in_month`-equivalent). → Int64.
    DaysInMonth,
    /// ISO 8601 week-numbering year (DuckDB `isoyear`), which can differ from the
    /// calendar year near January 1st. → Int64.
    IsoYear,
}

/// One `WHEN condition THEN value` branch of a `Case`.
#[derive(Debug, Clone, Deserialize)]
pub struct CaseBranch {
    pub when: Expr,
    pub then: Expr,
}

/// A constant value. Kept deliberately small for the bootstrap engine; widened
/// as the type system grows.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Literal {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    /// Microseconds since the Unix epoch (tz-naive Timestamp(Microsecond)).
    Timestamp(i64),
    /// Days since the Unix epoch (Date32).
    Date(i32),
}

/// Binary operators. Comparisons yield booleans; arithmetic yields the numeric
/// promotion arrow's kernels choose; boolean ops require boolean inputs.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BinaryOp {
    // comparison
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    // arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    // boolean
    And,
    Or,
    // string
    Concat,
    // bitwise (Int64)
    BitAnd,
    BitOr,
    BitXor,
    ShiftLeft,
    ShiftRight,
    /// Add `right` calendar months to a Date32/Timestamp `left` (negative to
    /// subtract); used for `date + INTERVAL n MONTH/YEAR`.
    AddMonths,
}

impl Expr {
    /// Evaluate the expression against `batch`, returning a full-length column.
    pub fn eval(&self, batch: &RecordBatch) -> Result<ArrayRef, ExprError> {
        match self {
            Expr::Col { name } => batch
                .column_by_name(name)
                .cloned()
                .ok_or_else(|| ExprError::UnknownColumn(name.clone())),
            Expr::Lit { value } => Ok(value.to_array(batch.num_rows())),
            Expr::Not { input } => {
                let arr = input.eval(batch)?;
                let b = as_bool(&arr, "not")?;
                Ok(Arc::new(boolean::not(b)?))
            }
            Expr::Binary { op, left, right } => {
                // Fast path: a numeric literal operand broadcasts as a scalar instead
                // of materializing a full N-length array (bit-identical result).
                if let Some(out) = try_scalar_binary(*op, left, right, batch)? {
                    return Ok(out);
                }
                let l = left.eval(batch)?;
                let r = right.eval(batch)?;
                eval_binary(*op, &l, &r)
            }
            Expr::Cast {
                input,
                dtype,
                try_cast,
            } => {
                let arr = input.eval(batch)?;
                cast_expr(&arr, &parse_dtype(dtype)?, *try_cast)
            }
            Expr::IsNull { input } => Ok(Arc::new(is_null(&input.eval(batch)?)?)),
            Expr::IsNotNull { input } => Ok(Arc::new(is_not_null(&input.eval(batch)?)?)),
            Expr::IsNan { input } => eval_is_nan(&input.eval(batch)?),
            Expr::Case {
                branches,
                otherwise,
            } => {
                // Fold from the default upward: later branches are overridden by
                // earlier ones (first matching WHEN wins).
                let mut acc = otherwise.eval(batch)?;
                for branch in branches.iter().rev() {
                    let mask_arr = branch.when.eval(batch)?;
                    let mask = as_bool(&mask_arr, "case")?;
                    // SQL CASE semantics: a WHEN that evaluates to NULL is *not*
                    // taken (it falls through to ELSE), matching DuckDB. `zip` would
                    // otherwise let a null mask pick the THEN branch, so collapse a
                    // null mask element to false (true only where value AND valid).
                    let mask = match mask.nulls() {
                        Some(n) => BooleanArray::new(mask.values() & n.inner(), None),
                        None => mask.clone(),
                    };
                    let then = branch.then.eval(batch)?;
                    // `zip` requires matching branch types; coerce Int64/Float64
                    // (and decimal) to a common numeric type the way COALESCE and
                    // the binary ops do, so a `when(...).then(0).otherwise(x)` over a
                    // float column (or `clip`/`fill_nan`) doesn't error on a mixed
                    // int/float literal.
                    let (then, acc_c) = coerce_numeric(&then, &acc)?;
                    acc = zip(&mask, &then.as_ref(), &acc_c.as_ref())?;
                }
                Ok(acc)
            }
            Expr::Str {
                func,
                input,
                pattern,
                replacement,
                start,
                length,
            } => {
                let arr = input.eval(batch)?;
                eval_str(
                    *func,
                    &arr,
                    pattern.as_deref(),
                    replacement.as_deref(),
                    *start,
                    *length,
                )
            }
            Expr::Date { func, input } => {
                let arr = input.eval(batch)?;
                eval_date(*func, &arr)
            }
            Expr::Image {
                func,
                input,
                width,
                height,
            } => {
                let arr = input.eval(batch)?;
                eval_image(*func, &arr, *width, *height)
            }
            Expr::Coalesce { inputs } => eval_coalesce(inputs, batch),
            Expr::Array { elements } => eval_array(elements, batch),
            Expr::ListJoin { input, separator } => eval_list_join(&input.eval(batch)?, separator),
            Expr::Math { func, input } => {
                let arr = input.eval(batch)?;
                eval_math(*func, &arr)
            }
            Expr::List { func, input } => {
                let arr = input.eval(batch)?;
                eval_list(*func, &arr)
            }
            Expr::NullIf { left, right } => {
                let l = left.eval(batch)?;
                let r = right.eval(batch)?;
                let eq = eval_binary(BinaryOp::Eq, &l, &r)?;
                let mask = as_bool(&eq, "nullif")?;
                Ok(arrow::compute::nullif(&l, mask)?)
            }
            Expr::Greatest { inputs } => eval_extreme(inputs, batch, true),
            Expr::Least { inputs } => eval_extreme(inputs, batch, false),
            Expr::Math2 { func, left, right } => {
                let l = left.eval(batch)?;
                let r = right.eval(batch)?;
                eval_math2(*func, &l, &r)
            }
            Expr::ListGet { input, index } => {
                let arr = input.eval(batch)?;
                eval_list_get(&arr, *index)
            }
            Expr::StructField { input, field } => {
                let arr = input.eval(batch)?;
                eval_struct_field(&arr, field)
            }
            Expr::ListContains { input, value } => {
                let arr = input.eval(batch)?;
                eval_list_contains(&arr, value)
            }
            Expr::ListBinary { func, left, right } => {
                let l = left.eval(batch)?;
                let r = right.eval(batch)?;
                eval_list_binary(*func, &l, &r)
            }
            Expr::DateTrunc { input, unit } => {
                let arr = input.eval(batch)?;
                eval_date_trunc(&arr, unit)
            }
            Expr::Strftime { input, format } => {
                let arr = input.eval(batch)?;
                eval_strftime(&arr, format)
            }
            Expr::Strptime { input, format } => {
                let arr = input.eval(batch)?;
                eval_strptime(&arr, format)
            }
            Expr::DateOffset {
                input,
                months,
                days,
                micros,
            } => {
                let arr = input.eval(batch)?;
                eval_date_offset(&arr, *months, *days, *micros)
            }
            Expr::ListSlice {
                input,
                offset,
                length,
            } => {
                let arr = input.eval(batch)?;
                let list = require_list(&arr, "list.slice")?;
                rebuild_list(list, |s, e| {
                    let begin = (s as i64 + (*offset).max(0)).min(e as i64) as usize;
                    let end = match length {
                        Some(l) => (begin as i64 + (*l).max(0)).min(e as i64) as usize,
                        None => e,
                    };
                    (begin..end).map(|k| k as u32).collect()
                })
            }
        }
    }
}

/// Cast `arr` to `target` with DuckDB float→int semantics. Arrow's float→int cast
/// truncates toward zero; DuckDB rounds half-to-even (`cast(2.5)` = 2, `cast(3.5)`
/// = 4), so float inputs are rounded to an integral value before the cast. All
/// other casts defer to the arrow kernel unchanged. (The JIT never compiles
/// float→int, so this interpreter-only behavior keeps tier parity intact.)
///
/// `try_cast` selects arrow's *safe* cast (a value that cannot be converted
/// becomes NULL — DuckDB `TRY_CAST`); the strict default (`false`) errors on an
/// invalid value (DuckDB `CAST`).
fn cast_expr(
    arr: &ArrayRef,
    target: &arrow::datatypes::DataType,
    try_cast: bool,
) -> Result<ArrayRef, ExprError> {
    use arrow::datatypes::DataType::{
        Float16, Float32, Float64, Int16, Int32, Int64, Int8, UInt16, UInt32, UInt64, UInt8,
    };
    let opts = CastOptions {
        safe: try_cast,
        ..Default::default()
    };
    let int_target = matches!(
        target,
        Int8 | Int16 | Int32 | Int64 | UInt8 | UInt16 | UInt32 | UInt64
    );
    let float_src = matches!(arr.data_type(), Float16 | Float32 | Float64);
    if int_target && float_src {
        // Round half-to-even first (DuckDB), then cast the now-integral floats.
        let f = cast_with_options(arr, &Float64, &opts)?;
        let f = f
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("cast to Float64 yields Float64Array");
        let rounded: Float64Array = f.iter().map(|o| o.map(f64::round_ties_even)).collect();
        let rounded: ArrayRef = Arc::new(rounded);
        return Ok(cast_with_options(&rounded, target, &opts)?);
    }
    Ok(cast_with_options(arr, target, &opts)?)
}

impl Literal {
    /// Materialize the literal as an array of length `n`.
    ///
    /// O(n) for now; replaced by `Datum` scalars once the kernels are threaded
    /// through selection vectors.
    pub(crate) fn to_array(&self, n: usize) -> ArrayRef {
        match self {
            Literal::Int(v) => Arc::new(Int64Array::from(vec![*v; n])),
            Literal::Float(v) => Arc::new(Float64Array::from(vec![*v; n])),
            Literal::Bool(v) => Arc::new(BooleanArray::from(vec![*v; n])),
            Literal::Str(v) => Arc::new(StringArray::from(vec![v.as_str(); n])),
            Literal::Timestamp(v) => Arc::new(TimestampMicrosecondArray::from(vec![*v; n])),
            Literal::Date(v) => Arc::new(Date32Array::from(vec![*v; n])),
        }
    }
}

#[cfg(test)]
mod str_date_tests {
    use super::*;
    use arrow::array::{Array, Date32Array, RecordBatch, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};

    fn batch_str() -> RecordBatch {
        let s = StringArray::from(vec![Some("Hello"), Some("wOrld"), None, Some("abcdef")]);
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)])),
            vec![Arc::new(s)],
        )
        .unwrap()
    }

    fn s(name: &str) -> Box<Expr> {
        Box::new(Expr::Col {
            name: name.to_string(),
        })
    }

    fn strf(func: StrFunc, pattern: Option<&str>, start: Option<i64>, length: Option<i64>) -> Expr {
        Expr::Str {
            func,
            input: s("s"),
            pattern: pattern.map(|p| p.to_string()),
            replacement: None,
            start,
            length,
        }
    }

    #[test]
    fn upper_lower_preserve_nulls() {
        let b = batch_str();
        let up = strf(StrFunc::Upper, None, None, None).eval(&b).unwrap();
        let up = up.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(up.value(0), "HELLO");
        assert_eq!(up.value(1), "WORLD");
        assert!(up.is_null(2));
    }

    #[test]
    fn len_counts_chars() {
        let b = batch_str();
        let l = strf(StrFunc::Len, None, None, None).eval(&b).unwrap();
        let l = l.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(l.value(0), 5);
        assert!(l.is_null(2));
    }

    #[test]
    fn contains_starts_ends() {
        let b = batch_str();
        let c = strf(StrFunc::Contains, Some("ell"), None, None)
            .eval(&b)
            .unwrap();
        let c = c.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert!(c.value(0) && !c.value(1) && c.is_null(2));
        let sw = strf(StrFunc::StartsWith, Some("abc"), None, None)
            .eval(&b)
            .unwrap();
        assert!(sw.as_any().downcast_ref::<BooleanArray>().unwrap().value(3));
    }

    #[test]
    fn substr_one_based() {
        let b = batch_str();
        let r = strf(StrFunc::Substr, None, Some(2), Some(3))
            .eval(&b)
            .unwrap();
        let r = r.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(r.value(0), "ell"); // "Hello"[2..5)
        assert_eq!(r.value(3), "bcd");
        // length to end
        let r2 = strf(StrFunc::Substr, None, Some(3), None).eval(&b).unwrap();
        assert_eq!(
            r2.as_any().downcast_ref::<StringArray>().unwrap().value(3),
            "cdef"
        );
    }

    #[test]
    fn like_ilike_semantics() {
        let s = StringArray::from(vec![
            Some("abc"),
            Some("a.b"),
            Some("axb"),
            None,
            Some("HELLO"),
        ]);
        let b = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)])),
            vec![Arc::new(s)],
        )
        .unwrap();
        // `a%` is anchored: matches anything starting with "a".
        let r = strf(StrFunc::Like, Some("a%"), None, None)
            .eval(&b)
            .unwrap();
        let r = r.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert!(r.value(0) && r.value(1) && r.value(2));
        assert!(r.is_null(3) && !r.value(4));
        // `a.b` literal-matches "a.b" only (the `.` is NOT a wildcard).
        let r = strf(StrFunc::Like, Some("a.b"), None, None)
            .eval(&b)
            .unwrap();
        let r = r.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert!(!r.value(0) && r.value(1) && !r.value(2));
        // `_` matches exactly one char: "a_b" matches "a.b" and "axb" but not "abc".
        let r = strf(StrFunc::Like, Some("a_b"), None, None)
            .eval(&b)
            .unwrap();
        let r = r.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert!(!r.value(0) && r.value(1) && r.value(2));
        // ILIKE is case-insensitive.
        let r = strf(StrFunc::Ilike, Some("hello"), None, None)
            .eval(&b)
            .unwrap();
        let r = r.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert!(r.value(4));
    }

    #[test]
    fn date_year_month_day() {
        // 2021-03-15 = day 18701 since epoch.
        let d = Date32Array::from(vec![Some(18701), None]);
        let b = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("d", DataType::Date32, true)])),
            vec![Arc::new(d)],
        )
        .unwrap();
        let year = Expr::Date {
            func: DateFunc::Year,
            input: Box::new(Expr::Col { name: "d".into() }),
        };
        let y = year.eval(&b).unwrap();
        let y = y.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(y.value(0), 2021);
        assert!(y.is_null(1));
    }

    #[test]
    fn case_coerces_int_then_against_float_otherwise() {
        // `when(true).then(0).otherwise(x)` over a Float64 column must coerce the
        // Int64 `then` to Float64 rather than erroring on mismatched zip types —
        // this is what makes `clip` / `fill_nan` / mixed when-then-otherwise work.
        use arrow::array::Float64Array;
        let x = Float64Array::from(vec![Some(1.0), Some(5.0)]);
        let b = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, true)])),
            vec![Arc::new(x)],
        )
        .unwrap();
        let case = Expr::Case {
            branches: vec![CaseBranch {
                when: Expr::Lit {
                    value: Literal::Bool(true),
                },
                then: Expr::Lit {
                    value: Literal::Int(0),
                },
            }],
            otherwise: Box::new(Expr::Col { name: "x".into() }),
        };
        let out = case.eval(&b).unwrap();
        let out = out.as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(out.value(0), 0.0);
        assert_eq!(out.value(1), 0.0);
    }

    #[test]
    fn case_null_when_falls_through_to_else() {
        // SQL semantics: `CASE WHEN (x < 2) THEN 99 ELSE x` over x = [1, 5, null].
        // The null row's WHEN is null → not taken → ELSE (x stays null), it must NOT
        // pick the THEN branch.
        use arrow::array::Float64Array;
        let x = Float64Array::from(vec![Some(1.0), Some(5.0), None]);
        let b = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, true)])),
            vec![Arc::new(x)],
        )
        .unwrap();
        let lt = Expr::Binary {
            op: BinaryOp::Lt,
            left: Box::new(Expr::Col { name: "x".into() }),
            right: Box::new(Expr::Lit {
                value: Literal::Float(2.0),
            }),
        };
        let case = Expr::Case {
            branches: vec![CaseBranch {
                when: lt,
                then: Expr::Lit {
                    value: Literal::Float(99.0),
                },
            }],
            otherwise: Box::new(Expr::Col { name: "x".into() }),
        };
        let out = case.eval(&b).unwrap();
        let out = out.as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(out.value(0), 99.0); // 1 < 2 → then
        assert_eq!(out.value(1), 5.0); // 5 < 2 false → else
        assert!(out.is_null(2)); // null when → else (null), not 99
    }
}
