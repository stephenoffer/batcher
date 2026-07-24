//! The crate's error type: every way scalar expression evaluation can fail.

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
    #[error("string function {func}: {reason}")]
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

    #[error("image function {func} expected a Binary argument, got {got}")]
    ExpectedBinary { func: String, got: String },

    #[error("image function {func} requires a {arg} argument")]
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
