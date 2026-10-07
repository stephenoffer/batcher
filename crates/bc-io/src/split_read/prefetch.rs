//! Read-ahead for a caller that reads a remote Parquet file one row group at a time.
//!
//! **A unit-at-a-time reader is latency-bound, not bandwidth-bound.** The engine's unit
//! executors (`bc_interp::stream::chunked`) have each worker thread read one row group and then
//! process it, so a node has at most one GET in flight per core. A TPC-H sf1000 `lineitem` row
//! group's projected columns are one ~1.8 MiB range, and on three 16-core nodes q1 and q6 ran at
//! ~250 MB/s of receive per node with the CPUs 12-24% busy -- a fifth of the link, waiting on
//! first bytes rather than moving them.
//!
//! [`prefetch_row_group`] starts the GET for a row group the caller is about to want, in the
//! background, and parks the bytes here. The remote reader ([`super::MaybeSplitReader`]) asks
//! [`take`] for every range it reads: a range inside a parked span is served from it (and the
//! span is dropped once used), anything else is read as before. So the bytes a read returns are
//! the object's bytes either way -- this moves *when* they are fetched, never which ones.
//!
//! Bounded on both axes, because nothing guarantees a prefetched span is ever read (a query
//! that stops early, a range that turns out to be pruned): [`MAX_BYTES`] of spans in flight or
//! parked per process, [`MAX_ENTRIES`] spans, the oldest evicted first.

use std::collections::VecDeque;
use std::ops::Range;
use std::sync::{Arc, Mutex, OnceLock};

use bytes::Bytes;
use futures::future::{BoxFuture, FutureExt, Shared};

/// Bytes of parked or in-flight spans the process may hold before a new prefetch is skipped.
const MAX_BYTES: u64 = 1 << 30;

/// Spans the process may hold; past it the oldest is dropped to admit a new one.
const MAX_ENTRIES: usize = 512;

/// A span's bytes, fetched once and shared with whichever reader asks first.
type SpanFuture = Shared<BoxFuture<'static, Result<Bytes, Arc<str>>>>;

struct Span {
    object: Arc<str>,
    range: Range<u64>,
    bytes: SpanFuture,
    /// Bytes of the span readers have taken so far. A row group can be read in two passes (a
    /// late filter's predicate columns, then the rest), so a span is dropped once most of it
    /// has been served rather than on its first use.
    served: u64,
}

#[derive(Default)]
struct Store {
    spans: VecDeque<Span>,
    held: u64,
}

fn store() -> &'static Mutex<Store> {
    static S: OnceLock<Mutex<Store>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(Store::default()))
}

/// Park `bytes` (the object's bytes over `range`, still being fetched) for the next reader.
///
/// Returns `false`, parking nothing, when the process already holds [`MAX_BYTES`].
pub(crate) fn park(object: Arc<str>, range: Range<u64>, bytes: SpanFuture) -> bool {
    let len = range.end - range.start;
    let mut s = store().lock().unwrap_or_else(|p| p.into_inner());
    if s.held + len > MAX_BYTES {
        return false;
    }
    while s.spans.len() >= MAX_ENTRIES {
        if let Some(old) = s.spans.pop_front() {
            s.held -= old.range.end - old.range.start;
        }
    }
    s.held += len;
    s.spans.push_back(Span {
        object,
        range,
        bytes,
        served: 0,
    });
    true
}

/// `ranges` of `object` served from one parked span covering all of them, or `None` to read them
/// as usual. The span is dropped once readers have taken most of its bytes: a row group is read
/// once, in one or two passes, and holding it longer only holds memory.
pub(crate) fn take(
    object: &str,
    ranges: &[Range<u64>],
) -> Option<BoxFuture<'static, Result<Vec<Bytes>, Arc<str>>>> {
    let (lo, hi) = (
        ranges.iter().map(|r| r.start).min()?,
        ranges.iter().map(|r| r.end).max()?,
    );
    let asked: u64 = ranges.iter().map(|r| r.end - r.start).sum();
    let (bytes, base) = {
        let mut s = store().lock().unwrap_or_else(|p| p.into_inner());
        let at = s
            .spans
            .iter()
            .position(|sp| &*sp.object == object && sp.range.start <= lo && hi <= sp.range.end)?;
        let span = &mut s.spans[at];
        span.served += asked;
        let (len, done) = (span.range.end - span.range.start, span.served);
        let out = (span.bytes.clone(), span.range.start);
        if done * 10 >= len * 9 {
            s.spans.remove(at);
            s.held -= len;
        }
        out
    };
    let wanted: Vec<Range<u64>> = ranges.to_vec();
    Some(
        async move {
            let all = bytes.await?;
            Ok(wanted
                .iter()
                .map(|r| all.slice((r.start - base) as usize..(r.end - base) as usize))
                .collect())
        }
        .boxed(),
    )
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)] // a one-range read is the case under test
mod tests {
    use super::*;

    fn ready(bytes: &'static [u8]) -> SpanFuture {
        async move { Ok(Bytes::from_static(bytes)) }
            .boxed()
            .shared()
    }

    /// A parked span serves exactly the slices asked of it, across passes, until used up.
    #[test]
    fn a_parked_span_serves_its_slices_once() {
        let id: Arc<str> = Arc::from("s3://t/once#10#v");
        assert!(park(id.clone(), 100..110, ready(b"0123456789")));
        let got = futures::executor::block_on(take(&id, &[102..105, 108..110]).unwrap()).unwrap();
        assert_eq!(
            got,
            vec![Bytes::from_static(b"234"), Bytes::from_static(b"89")]
        );
        // Half the span served: still parked for a second pass, then dropped once used up.
        let rest = futures::executor::block_on(take(&id, &[100..102, 105..108]).unwrap()).unwrap();
        assert_eq!(
            rest,
            vec![Bytes::from_static(b"01"), Bytes::from_static(b"567")]
        );
        assert!(
            take(&id, &[100..101]).is_none(),
            "a used-up span is dropped"
        );
    }

    /// A range outside every parked span, or of another object, is not served.
    #[test]
    fn an_uncovered_range_is_not_served() {
        let id: Arc<str> = Arc::from("s3://t/miss#10#v");
        assert!(park(id.clone(), 0..10, ready(b"abcdefghij")));
        assert!(take(&id, &[5..12]).is_none());
        assert!(take("s3://t/other#10#v", &[1..2]).is_none());
        assert!(take(&id, &[1..2]).is_some());
    }
}
