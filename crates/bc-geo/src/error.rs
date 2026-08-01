//! The one error type every `bc-geo` entry point returns.
//!
//! Geometry work fails for exactly three reasons and callers act differently on
//! each, so they are separate variants rather than one string: the bytes are not a
//! geometry (`Parse`), the geometry is a geometry but not the *kind* this operation
//! is defined on (`Unsupported`), or the operation's numeric preconditions are not
//! met (`Invalid`). `bc-expr` maps the first two to a null result and the third to a
//! query error, which is only expressible if the distinction survives the boundary.

use thiserror::Error;

/// Failure modes of parsing, writing, or computing over a geometry.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum GeoError {
    /// The input is not a well-formed geometry in the claimed encoding.
    #[error("invalid {encoding}: {detail}")]
    Parse {
        /// The encoding that was being read (`"WKB"`, `"WKT"`, `"GeoJSON"`).
        encoding: &'static str,
        /// What specifically was wrong, in terms a user can act on.
        detail: String,
    },

    /// The geometry is well-formed but this operation is not defined on its type.
    #[error("{op} is not defined on {geom_type}")]
    Unsupported {
        /// The operation that was attempted (`"st_exterior_ring"`).
        op: &'static str,
        /// The geometry type it was attempted on (`"POINT"`).
        geom_type: &'static str,
    },

    /// A numeric or structural precondition of the operation was violated.
    #[error("{0}")]
    Invalid(String),
}

impl GeoError {
    /// A `Parse` failure while reading `encoding`.
    pub fn parse(encoding: &'static str, detail: impl Into<String>) -> Self {
        GeoError::Parse {
            encoding,
            detail: detail.into(),
        }
    }

    /// An `Invalid` failure carrying an operator-facing explanation.
    pub fn invalid(detail: impl Into<String>) -> Self {
        GeoError::Invalid(detail.into())
    }

    /// True when the failure means "this input is not a geometry / not this shape",
    /// which the expression layer surfaces as a null rather than a query error.
    ///
    /// `Invalid` is deliberately excluded: it reports a caller mistake (a negative
    /// buffer quadrant count, a grid precision out of range) that is a property of the
    /// *plan*, not of one row, so nulling it would hide the bug on every row.
    pub fn is_row_local(&self) -> bool {
        matches!(self, GeoError::Parse { .. } | GeoError::Unsupported { .. })
    }
}

/// The crate's result alias.
pub type GeoResult<T> = Result<T, GeoError>;
