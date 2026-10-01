//! The lazy pipeline adapters: scan, the per-morsel transforms, and the early-exiting limit.
//!
//! Each is an `Iterator` over morsels that pulls exactly one morsel from its child, transforms
//! it, and yields — so a chain of them holds one morsel per stage and nothing else.

use arrow::array::{BooleanArray, RecordBatch};
use arrow::compute::filter_record_batch;
use bc_expr::{ConjunctOrder, Expr};
use bc_ir::RelOp;

use super::builds::node_key;
use super::{build_with, Ctx, Morsels};
use crate::{ops, InterpError};

/// Rows per morsel handed out by a scan.
///
/// A source relation arrives as whatever batching the FFI boundary gave it, and that can be one
/// enormous `RecordBatch`. Re-slicing it here is what makes "one morsel per stage" a real bound
/// rather than a promise: the slices are zero-copy views, so this costs nothing and caps what a
/// downstream filter or probe has to hold.
const SCAN_MORSEL_ROWS: usize = bc_arrow::DEFAULT_MORSEL_ROWS;

/// A Filter, or a run of Filters stacked directly on one another, as one per-morsel stage.
///
/// Filter and Project stay on the interpreter here, and that is a measured choice rather than
/// an oversight: wiring the Tier-1 JIT into this path (compile once per operator on the first
/// morsel, reuse across the rest) was tried and measured 1.01x over TPC-H in an interleaved
/// A/B, with five queries slower. `par.rs` still compiles, where the fused shapes make it pay.
///
/// **A stacked run gathers once.** Each filter's own conjunct order is kept, and each one's
/// mask is carried to the next, which ANDs its own onto it over the ungathered morsel
/// (`ops::filter_mask_within`, which declines where that is not sound); the rows are gathered
/// after the last. A filter over a filter is a shape the planner emits routinely -- the implied
/// `x <= max` bound it derives from `x IN (...)`, or a pushed-down predicate it kept separate --
/// and on a string-heavy morsel the intermediate gather was the larger cost (`memmove` plus
/// `FilterBytes` ~49% of `l_shipmode IN (...) AND l_suppkey IN (...)`). Each filter's rows in
/// and out are metered exactly as before. A filter a runtime join filter is attached to, or
/// one already materialized, ends the run, so it keeps its own stream (`build_with`).
pub(super) fn filter_stream<'a>(
    input: &'a RelOp,
    predicate: &'a Expr,
    id: Option<u32>,
    ctx: Ctx<'a>,
) -> Result<Morsels<'a>, InterpError> {
    // Innermost last while collecting, so `run` reads outermost first; reversed below.
    let mut run: Vec<(Option<u32>, &Expr)> = vec![(id, predicate)];
    let mut base = input;
    while let RelOp::Filter { input, predicate } = base {
        if !plain(base, ctx) {
            break;
        }
        run.push((ctx.id(base), predicate));
        base = input;
    }
    run.reverse();
    let orders: Vec<Option<ConjunctOrder>> =
        run.iter().map(|(_, p)| ConjunctOrder::new(p)).collect();
    let child = build_with(base, ctx)?;
    Ok(Box::new(child.map(move |b| {
        let mut cur = b?;
        // The keep mask of the filters run so far over `cur`, not yet applied to it.
        let mut pending: Option<BooleanArray> = None;
        for (k, (&(id, predicate), order)) in run.iter().zip(&orders).enumerate() {
            let last = k + 1 == run.len();
            let rows_in = pending
                .as_ref()
                .map_or(cur.num_rows(), BooleanArray::true_count) as u64;
            let t = std::time::Instant::now();
            let mask = match pending.take() {
                Some(live) => {
                    let within =
                        ops::filter_mask_within(&cur, predicate, &None, order.as_ref(), &live)?;
                    if within.is_none() {
                        cur = filter_record_batch(&cur, &live)?;
                    }
                    within
                }
                None if !last => Some(ops::truthy(&ops::filter_mask_jit(
                    &cur,
                    predicate,
                    &None,
                    order.as_ref(),
                )?)),
                None => None,
            };
            match mask {
                Some(mask) if !last => {
                    let kept = mask.true_count() as u64;
                    if let (Some(m), Some(id)) = (ctx.meter, id) {
                        m.morsel_ungathered(id, rows_in, kept, &cur, t.elapsed().as_nanos() as u64);
                    }
                    pending = Some(mask);
                }
                Some(mask) => {
                    cur = filter_record_batch(&cur, &mask)?;
                    ctx.morsel(id, rows_in, &cur, t);
                }
                None => {
                    cur = ops::filter_batch_jit(&cur, predicate, &None, order.as_ref())?;
                    ctx.morsel(id, rows_in, &cur, t);
                }
            }
        }
        Ok(cur)
    })))
}

/// Whether `node` is built as nothing more than its own operator: no runtime join filter is
/// attached to it and it is not an already-materialized leaf -- see `build_with` and
/// `build_node`, which would otherwise wrap or replace its stream.
fn plain(node: &RelOp, ctx: Ctx<'_>) -> bool {
    let key = node_key(node);
    ctx.cache.filters_for(key).is_none()
        && ctx.mats.and_then(|m| m.get(&key)).is_none()
        && ctx.cache.probe_leaf(key).is_none()
}

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

/// Stream one unit of a lazily-read driving relation as morsel-sized slices, reading it only
/// when the pipeline first pulls from it.
///
/// The owned counterpart of [`scan_stream`]: the unit's batches are decoded on the pulling
/// worker and dropped once its morsels have passed through, so a worker holds one unit rather
/// than its whole share of the relation. Zero-row batches are dropped — the caller keeps the
/// schema carrier for a relation that turns out empty.
pub(super) fn unit_stream<'a>(
    src: &'a dyn super::chunked::units::UnitSource,
    units: std::ops::Range<usize>,
) -> Morsels<'a> {
    Box::new(units.flat_map(move |unit| {
        let decoded: Vec<Result<RecordBatch, InterpError>> = match src.read(unit) {
            Ok(batches) => batches
                .into_iter()
                .flat_map(|b| {
                    let rows = b.num_rows();
                    (0..rows)
                        .step_by(SCAN_MORSEL_ROWS)
                        .map(move |off| Ok(b.slice(off, SCAN_MORSEL_ROWS.min(rows - off))))
                        .collect::<Vec<_>>()
                })
                .collect(),
            Err(e) => vec![Err(e)],
        };
        decoded
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
pub(crate) fn limit_stream(child: Morsels<'_>, n: usize, offset: usize) -> Morsels<'_> {
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
