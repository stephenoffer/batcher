//! Date/time evaluation: field extraction, timezone conversion, and construction.
//!
//! The three modules here are the two directions of the same contract plus the zone
//! shift between them. `date` pulls fields *out* of a Date/Timestamp (and owns the
//! truncation, formatting, and offset arithmetic that go with it); `make` builds one
//! *from* integer parts or an epoch count; `timezone` moves an instant between zones.
//!
//! They are grouped because they must agree on the conventions no single one of them
//! can enforce alone — microsecond resolution, tz-naive UTC storage, flooring rather
//! than truncating toward zero for anything pre-1970, and answering null rather than
//! raising when a value names no real instant.

pub(crate) mod date;
pub(crate) mod make;
pub(crate) mod timezone;
