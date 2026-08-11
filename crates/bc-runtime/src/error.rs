//! The crate's error type: how the stateful runtime structures report failure.

use arrow::error::ArrowError;
use thiserror::Error;

/// Errors raised by runtime-library structures.
#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("aggregate {func} is not supported for column type {dtype}")]
    UnsupportedAggregate { func: String, dtype: String },

    #[error("aggregate {func} requires an input column")]
    MissingAggregateInput { func: String },

    #[error("integer SUM overflowed i64; cast the column to a wider type first")]
    SumOverflow,

    #[error(
        "a {dtype} column reached {bytes} bytes, past Arrow's 2 GiB limit for 32-bit string \
         offsets; cast the column to large_string (or large_binary) so its offsets are 64-bit"
    )]
    ByteOffsetOverflow { dtype: String, bytes: usize },

    #[error("window function {func} is not supported for column type {dtype}")]
    UnsupportedWindow { func: String, dtype: String },

    #[error("window function {func} requires an input column")]
    MissingWindowInput { func: String },

    #[error("window function {func} requires order keys")]
    WindowRequiresOrder { func: String },

    #[error(
        "ASOF tolerance and direction='nearest' need a numeric or temporal `on` key \
         to measure a distance, but the key is {dtype}"
    )]
    AsofKeyNotMeasurable { dtype: String },

    #[error(
        "a RANGE window frame with an offset needs exactly one ORDER BY key to measure \
         the offset against, got {got}"
    )]
    RangeFrameNeedsOneOrderKey { got: usize },

    #[error(
        "a RANGE window frame with an offset needs a numeric or temporal ORDER BY key \
         to measure the offset against, but the key is {dtype}"
    )]
    RangeFrameKeyNotMeasurable { dtype: String },

    #[error("malformed spilled partial: expected {expected} columns, got {got}")]
    MalformedPartial { expected: usize, got: usize },

    #[error("range-partition key must be a numeric column, got {dtype}")]
    NonNumericRangeKey { dtype: String },

    #[error("range join cannot run on this condition: {reason}")]
    UnsupportedRangeJoin { reason: String },

    /// The spill filesystem ran out of room (or the writer's quota did).
    ///
    /// Separated from the generic [`RuntimeError::Io`] because it is the one spill failure a
    /// user can act on, and because bare "No space left on device" says nothing about
    /// *which* device: spill scratch defaults to the system temp directory, which on a
    /// container is often a small overlay or a tmpfs sized well below the query's spill
    /// volume, while the large volume the user assumed was being used sits elsewhere. The
    /// path and the volume written so far are what turn that into a decision.
    #[error(
        "spill ran out of disk space after writing {written_bytes} bytes to {dir} \
         (the query needs more spill scratch than that filesystem has). Point \
         memory.spill_dir at a larger filesystem, set memory.spill_remote_uri to overflow \
         to object storage, or lower memory.max_memory_bytes so less of the query \
         materializes at once. Underlying error: {source}"
    )]
    SpillOutOfSpace {
        dir: String,
        written_bytes: u64,
        source: std::io::Error,
    },

    /// A spill partition read back fewer rows than were written to it.
    ///
    /// Its own variant because of *how* this failure presents. An Arrow IPC stream truncated
    /// at a message boundary — the last complete batch written, the end-of-stream marker
    /// missing — is indistinguishable from a shorter valid stream: the reader returns the
    /// batches it finds and reports success. Measured, not assumed: five batches of 1,000
    /// rows truncated after the third read back as 3,000 rows with no error at all.
    ///
    /// Every way that happens is a way a query returns a *wrong answer* rather than failing:
    /// a filesystem that reported a short write as success, a spill file that outlived the
    /// process that was still writing it, a truncation on a full disk that the write path
    /// did not see. Counting rows on the way in and checking them on the way out is what
    /// turns all of them into an error.
    #[error(
        "spill partition {partition} in {dir} read back {got_rows} rows but {expected_rows} \
         were written — the spill file is truncated or was modified underneath the query. \
         This would otherwise have silently dropped {missing} rows from the result."
    )]
    SpillTruncated {
        dir: String,
        partition: usize,
        expected_rows: u64,
        got_rows: u64,
        missing: u64,
    },

    #[error("spill i/o error: {0}")]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Arrow(#[from] ArrowError),
}
