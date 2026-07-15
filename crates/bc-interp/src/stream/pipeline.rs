//! The lazy pipeline adapters: scan, the per-morsel transforms, and the early-exiting limit.
//!
//! Each is an `Iterator` over morsels that pulls exactly one morsel from its child, transforms
//! it, and yields — so a chain of them holds one morsel per stage and nothing else.

use arrow::array::RecordBatch;

use super::Morsels;
use crate::InterpError;

/// Rows per morsel handed out by a scan.
///
/// A source relation arrives as whatever batching the FFI boundary gave it, and that can be one
/// enormous `RecordBatch`. Re-slicing it here is what makes "one morsel per stage" a real bound
/// rather than a promise: the slices are zero-copy views, so this costs nothing and caps what a
/// downstream filter or probe has to hold.
const SCAN_MORSEL_ROWS: usize = bc_arrow::DEFAULT_MORSEL_ROWS;

/// Stream a source relation as zero-copy morsel-sized slices.
pub(super) fn scan_stream(batches: &[RecordBatch]) -> Morsels<'_> {
    Box::new(batches.iter().flat_map(|b| {
        let rows = b.num_rows();
        if rows == 0 {
            // A zero-row batch still carries the schema, and a downstream breaker needs it over
            // an empty relation. Slicing `0..0` would yield nothing and lose it.
            return Either::One(std::iter::once(Ok(b.clone())));
        }
        Either::Many((0..rows).step_by(SCAN_MORSEL_ROWS).map(move |off| {
            let len = SCAN_MORSEL_ROWS.min(rows - off);
            Ok(b.slice(off, len))
        }))
    }))
}

/// A two-shape iterator, so `scan_stream`'s `flat_map` can return either the schema-carrying
/// singleton or the sliced morsels without boxing per batch.
enum Either<A, B> {
    One(A),
    Many(B),
}

impl<A, B, T> Iterator for Either<A, B>
where
    A: Iterator<Item = T>,
    B: Iterator<Item = T>,
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        match self {
            Either::One(i) => i.next(),
            Either::Many(i) => i.next(),
        }
    }
}

/// `LIMIT n OFFSET k`, streaming — and **stopping**.
///
/// This is the operator whose streaming form changes complexity rather than just memory. The
/// materializing path runs the entire subtree, builds the whole relation, and then throws all
/// but `n` rows away; here the iterator simply stops pulling once it has `n`, so the scan below
/// it never reads the rest. `LIMIT 10` over a billion rows now costs ten rows of work.
///
/// The row-by-row bookkeeping mirrors `ops::limit` exactly (skip `offset`, take `n`, slice the
/// straddling morsel), so the rows and their order are the oracle's.
pub(super) fn limit_stream(child: Morsels<'_>, n: usize, offset: usize) -> Morsels<'_> {
    Box::new(Limit {
        child,
        remaining_skip: offset,
        remaining_take: n,
        schema: None,
        emitted_any: false,
        done: false,
    })
}

/// `LIMIT`'s state. A struct rather than a chain of closures because the schema and the
/// "emitted anything?" flag are read *after* the child is exhausted, and two closures cannot
/// both hold them.
struct Limit<'a> {
    child: Morsels<'a>,
    remaining_skip: usize,
    remaining_take: usize,
    schema: Option<arrow::datatypes::SchemaRef>,
    emitted_any: bool,
    /// Set once the child is spent (or the limit is satisfied), so the schema-only batch is
    /// emitted exactly once and the child is never pulled again.
    done: bool,
}

impl Iterator for Limit<'_> {
    type Item = Result<RecordBatch, InterpError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.done {
                return None;
            }
            // Satisfied: stop pulling. This is the early exit — the scan below never reads on.
            // But `LIMIT 0` still owes a downstream breaker the *schema*, and the only place to
            // learn it is the child. So pull exactly one batch for its schema before stopping —
            // never more. Without this a `LIMIT 0` build side yields nothing, and an anti/left
            // join over it wrongly returns empty instead of all its probe rows. This mirrors
            // `ops::limit`, which reads the schema off `batches.first()` before its own loop.
            if self.remaining_take == 0 {
                self.done = true;
                if self.schema.is_none() {
                    if let Some(Ok(b)) = self.child.next() {
                        self.schema = Some(b.schema());
                    }
                }
                return self.schema_only();
            }
            let batch = match self.child.next() {
                Some(Ok(b)) => b,
                Some(Err(e)) => return Some(Err(e)),
                None => {
                    self.done = true;
                    return self.schema_only();
                }
            };
            if self.schema.is_none() {
                self.schema = Some(batch.schema());
            }
            let rows = batch.num_rows();
            if self.remaining_skip >= rows {
                self.remaining_skip -= rows;
                continue; // wholly inside the offset — pull the next morsel
            }
            let start = self.remaining_skip;
            self.remaining_skip = 0;
            let take_n = (rows - start).min(self.remaining_take);
            self.remaining_take -= take_n;
            self.emitted_any = true;
            return Some(Ok(batch.slice(start, take_n)));
        }
    }
}

impl Limit<'_> {
    /// `Limit(_, 0)` is the canonical empty marker, and a wholly-skipped limit emits nothing
    /// either. Both still owe a downstream breaker a schema over zero rows — exactly what
    /// `ops::limit` returns in the same situation.
    fn schema_only(&mut self) -> Option<Result<RecordBatch, InterpError>> {
        if self.emitted_any {
            return None;
        }
        self.emitted_any = true; // emit it once
        let schema = self.schema.clone()?;
        Some(Ok(RecordBatch::new_empty(schema)))
    }
}
