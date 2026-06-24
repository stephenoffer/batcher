use arrow::error::ArrowError;
use thiserror::Error;

/// Errors raised while evaluating a scalar expression.
#[derive(Debug, Error)]
pub enum ExprError {
    #[error("unknown column: {0}")]
    UnknownColumn(String),

    #[error("operator `{op}` expected a boolean argument, got {got}")]
    ExpectedBoolean { op: String, got: String },

    #[error("unknown cast target type: {0}")]
    UnknownType(String),

    #[error("string function {func} expected a Utf8 argument, got {got}")]
    ExpectedString { func: String, got: String },

    #[error("string function {func} requires a {arg} argument")]
    MissingArgument { func: String, arg: &'static str },

    #[error("integer division or modulo by zero")]
    DivideByZero,

    #[error("invalid regular expression: {pattern}")]
    InvalidRegex { pattern: String },

    #[error("image function {func} expected a Binary argument, got {got}")]
    ExpectedBinary { func: String, got: String },

    #[error("image function {func} requires a {arg} argument")]
    MissingImageArg { func: String, arg: &'static str },

    #[error("image decode failed: {0}")]
    ImageDecode(String),

    #[error("{func} requires building the engine with the `{feature}` cargo feature")]
    FeatureDisabled { func: String, feature: &'static str },

    #[error(transparent)]
    Arrow(#[from] ArrowError),
}
