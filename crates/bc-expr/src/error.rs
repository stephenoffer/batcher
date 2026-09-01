//! The crate's error type: every way scalar expression evaluation can fail.

use arrow::error::ArrowError;
use thiserror::Error;

/// Errors raised while evaluating a scalar expression.
#[derive(Debug, Error)]
pub enum ExprError {
    #[error("unknown column: {0}")]
    UnknownColumn(String),

    /// A struct field the struct does not have. Distinct from `UnknownColumn` for the
    /// same reason `ExpectedType` is distinct from `ExpectedString`: the noun was wrong.
    /// `col("s").struct.field("zz")` reported "unknown column: zz" when `zz` is a *field*
    /// of `s`, sending the reader to hunt for a top-level column that was never the
    /// problem. Naming the fields the struct does have is what makes it actionable, the
    /// same reason the control plane's `ColumnNotFoundError` carries `available`.
    #[error("struct has no field `{field}`; its fields are: {available}")]
    UnknownField { field: String, available: String },

    #[error("operator `{op}` expected a boolean argument, got {got}")]
    ExpectedBoolean { op: String, got: String },

    #[error("unknown cast target type: {0}")]
    UnknownType(String),

    #[error("string function {func} expected a Utf8 argument, got {got}")]
    ExpectedString { func: String, got: String },

    /// A type mismatch in a function that is *not* a string function. It exists because
    /// `ExpectedString` was being reused for the list/map/struct/numeric/temporal argument
    /// checks, which produced messages that were wrong in both halves: `map.keys` on a
    /// Utf8 column reported "string function MapKeys expected a Utf8 argument, got Utf8" —
    /// naming the wrong family, and claiming it expected exactly what it had just rejected.
    #[error("{func} expected {want}, got {got}")]
    ExpectedType {
        func: String,
        want: &'static str,
        got: String,
    },

    #[error("string function {func} requires a {arg} argument")]
    MissingArgument { func: String, arg: &'static str },

    /// A scalar argument that deserialized fine but cannot produce a defined result
    /// (a zero-width chunk, an overlap wider than the chunk). The control plane
    /// validates these at the API edge; this guards a hand-written IR document.
    ///
    /// The message names `func` and nothing else. It used to open with "string function",
    /// which was true of the first callers and of none of the ones added since: the image,
    /// audio, geometry and spatial kernels all raise it, so a bad `n_fft` was reported as
    /// "string function audio.Spectrogram: ..." — a family name that sends the reader
    /// looking through the wrong half of the engine.
    #[error("{func}: {reason}")]
    InvalidArgument { func: String, reason: String },

    /// The key material itself is deliberately absent from this message: an error
    /// string is the one value in the engine that reliably reaches a log file.
    #[error("{func}: key must be 32 bytes, given as 64 hex characters or as base64")]
    InvalidKey { func: &'static str },

    /// A key *reference* (`env:NAME` / `file:PATH`) could not be resolved on this node.
    /// The reference is named (it is not secret and is what an operator needs to fix the
    /// misconfiguration); the resolved key never appears here.
    #[error("{func}: could not resolve key reference {reference}")]
    KeyRefUnresolved {
        func: &'static str,
        reference: String,
    },

    #[error("integer division or modulo by zero")]
    DivideByZero,

    /// A zero step in `sequence`/`range`. Reported separately from `DivideByZero`: a user
    /// who writes `sequence(1, 5, 0)` performed no division, so "integer division or modulo
    /// by zero" sends them looking for an arithmetic expression they never wrote.
    #[error("sequence: step must be non-zero")]
    ZeroSequenceStep,

    #[error("invalid regular expression: {pattern}")]
    InvalidRegex { pattern: String },

    #[error("media function {func} expected a Binary argument, got {got}")]
    ExpectedBinary { func: String, got: String },

    #[error("media function {func} requires a {arg} argument")]
    MissingImageArg { func: String, arg: &'static str },

    /// A target dimension (width/height) that is not a positive value representable as a
    /// `u32`. Casting an out-of-range `i64` with `as u32` would silently wrap — a negative
    /// value to a ~4-billion dimension (an unbounded allocation / OOM), or a value past
    /// `u32::MAX` to a small one (a silently wrong output size) — so it is rejected here.
    #[error(
        "image function {func}: {arg} must be a positive integer no larger than {max}, got {value}"
    )]
    InvalidImageDim {
        func: String,
        arg: &'static str,
        value: i64,
        max: u32,
    },

    #[error("audio.resample requires a positive target sample rate")]
    MissingAudioRate,

    #[error("image decode failed: {0}")]
    ImageDecode(String),

    #[error("{func} requires building the engine with the `{feature}` cargo feature")]
    FeatureDisabled { func: String, feature: &'static str },

    #[error(transparent)]
    Arrow(#[from] ArrowError),
}

/// A nested Arrow type rendered so a person can read it, for the `got`/`want` fields above.
///
/// `DataType`'s `Display` is its `Debug`, which for a leaf is what you want (`Int64`, `Utf8`)
/// and for anything nested is a struct literal. So every type-mismatch message on a list,
/// struct, map or dictionary column arrived like this:
///
/// ```text
/// string function RegexpCount expected a Utf8 argument, got List(Field { name: "item",
/// data_type: Int64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} })
/// ```
///
/// Six of those seven fields are noise, the reader has to find `Int64` inside them, and the
/// one thing they came for -- "it is a list of integers" -- is the hardest part to see. It is
/// the commonest error on the whole expression surface: applying a string, sequence or
/// temporal function to a column that is not one.
///
/// Leaves keep Arrow's own spelling rather than pyarrow's, because the sentence around them
/// already uses it ("expected a Utf8 argument"): rendering the argument as `list<item: int64>`
/// beside a `Utf8` in the same line would leave the reader with two vocabularies and no
/// statement of which is which. Only the nesting changes.
pub fn type_name(dtype: &arrow::datatypes::DataType) -> String {
    use arrow::datatypes::DataType;

    match dtype {
        DataType::List(f) | DataType::LargeList(f) | DataType::ListView(f) => {
            format!("List<{}>", type_name(f.data_type()))
        }
        DataType::LargeListView(f) => format!("List<{}>", type_name(f.data_type())),
        DataType::FixedSizeList(f, width) => {
            format!("FixedSizeList<{}, {width}>", type_name(f.data_type()))
        }
        DataType::Struct(fields) => format!("Struct<{}>", named_fields(fields)),
        // A map's single child is a struct of (key, value); naming that struct would report
        // the layout rather than the type the caller wrote.
        DataType::Map(entries, _) => match entries.data_type() {
            DataType::Struct(kv) if kv.len() == 2 => format!(
                "Map<{}, {}>",
                type_name(kv[0].data_type()),
                type_name(kv[1].data_type())
            ),
            other => format!("Map<{}>", type_name(other)),
        },
        DataType::Dictionary(key, value) => {
            format!("Dictionary<{}, {}>", type_name(key), type_name(value))
        }
        // Every leaf, and the handful of nested types with no shorter honest rendering.
        other => other.to_string(),
    }
}

/// ``name: Type`` for each field, comma-separated — the body of a `Struct<...>`.
fn named_fields(fields: &arrow::datatypes::Fields) -> String {
    fields
        .iter()
        .map(|f| format!("{}: {}", f.name(), type_name(f.data_type())))
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::type_name;
    use arrow::datatypes::{DataType, Field, Fields};
    use std::sync::Arc;

    #[test]
    fn a_leaf_keeps_arrows_own_spelling() {
        assert_eq!(type_name(&DataType::Utf8), "Utf8");
        assert_eq!(type_name(&DataType::Int64), "Int64");
    }

    #[test]
    fn a_list_names_its_item_type_and_nothing_else() {
        let list = DataType::List(Arc::new(Field::new("item", DataType::Int64, true)));
        assert_eq!(type_name(&list), "List<Int64>");
    }

    #[test]
    fn nesting_recurses() {
        let inner = DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)));
        let outer = DataType::List(Arc::new(Field::new("item", inner, true)));
        assert_eq!(type_name(&outer), "List<List<Utf8>>");
    }

    #[test]
    fn a_struct_names_its_fields() {
        let fields = Fields::from(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("b", DataType::Utf8, true),
        ]);
        assert_eq!(
            type_name(&DataType::Struct(fields)),
            "Struct<a: Int64, b: Utf8>"
        );
    }

    #[test]
    fn a_map_names_its_key_and_value_not_its_entry_struct() {
        let kv = Fields::from(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new("value", DataType::Int64, true),
        ]);
        let map = DataType::Map(
            Arc::new(Field::new("entries", DataType::Struct(kv), false)),
            false,
        );
        assert_eq!(type_name(&map), "Map<Utf8, Int64>");
    }

    #[test]
    fn a_fixed_size_list_keeps_its_width() {
        let f = DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float64, true)), 384);
        assert_eq!(type_name(&f), "FixedSizeList<Float64, 384>");
    }

    #[test]
    fn a_dictionary_names_both_halves() {
        let d = DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
        assert_eq!(type_name(&d), "Dictionary<Int32, Utf8>");
    }

    #[test]
    fn the_rendering_carries_no_arrow_field_debris() {
        // The property the whole function exists for, stated once against the shape that
        // produced the original message.
        let list = DataType::List(Arc::new(Field::new("item", DataType::Int64, true)));
        let rendered = type_name(&list);
        assert!(!rendered.contains("Field {"), "{rendered}");
        assert!(!rendered.contains("metadata"), "{rendered}");
    }
}
