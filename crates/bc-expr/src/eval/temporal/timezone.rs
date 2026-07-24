//! Timezone conversion for `Expr::ConvertTimezone` (`convert_timezone`).
//!
//! Batcher timestamps are tz-naive microseconds since the Unix epoch. This shifts
//! each instant's *wall-clock* from `from_tz` to `to_tz`, DST-aware via the IANA
//! database (chrono-tz). The JIT does not compile this; the interpreter is the only
//! path. An unknown tz name errors the batch; an ambiguous/nonexistent local time
//! (DST gaps/overlaps) yields null for that row.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, TimestampMicrosecondArray};
use chrono::{DateTime, TimeZone};
use chrono_tz::Tz;

use crate::ExprError;

/// `convert_timezone(from_tz, to_tz, ts)` — reinterpret each naive timestamp as a
/// wall-clock time in `from_tz` and return the corresponding wall-clock time in
/// `to_tz`. Nulls (and DST-ambiguous instants) propagate to null.
pub(crate) fn eval_convert_timezone(
    arr: &ArrayRef,
    from_tz: &str,
    to_tz: &str,
) -> Result<ArrayRef, ExprError> {
    let from: Tz = from_tz
        .parse()
        .map_err(|_| ExprError::UnknownType(format!("timezone `{from_tz}`")))?;
    let to: Tz = to_tz
        .parse()
        .map_err(|_| ExprError::UnknownType(format!("timezone `{to_tz}`")))?;

    let ts = arr
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()
        .ok_or_else(|| ExprError::ExpectedString {
            func: "convert_timezone".into(),
            got: arr.data_type().to_string(),
        })?;

    let out: TimestampMicrosecondArray = ts
        .iter()
        .map(|o| o.and_then(|micros| convert(micros, from, to)))
        .collect();
    Ok(Arc::new(out))
}

/// Shift one naive-UTC-stored instant's wall-clock from `from` to `to`; `None` for a
/// DST gap/overlap where the local time is nonexistent/ambiguous.
fn convert(micros: i64, from: Tz, to: Tz) -> Option<i64> {
    // The stored micros are a naive wall-clock; read them back as such.
    let naive = DateTime::from_timestamp_micros(micros)?.naive_utc();
    // Interpret that wall-clock as a `from`-zone local time → a concrete instant.
    let instant = from.from_local_datetime(&naive).single()?;
    // Re-express the instant as a `to`-zone wall-clock, stored again as naive micros.
    instant
        .with_timezone(&to)
        .naive_local()
        .and_utc()
        .timestamp_micros()
        .into()
}
