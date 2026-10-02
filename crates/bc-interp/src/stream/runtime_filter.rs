//! Sink each hash join's build-side key set down its probe pipeline, to the scan.
//!
//! [`bc_runtime::join::KeyFilter`] is the digest and owns the soundness argument; this module is
//! the *placement* decision — which join's keys may filter which probe-side node, and when to
//! stop bothering. The two are split exactly along the crate seam the engine uses everywhere
//! else: `bc-runtime` owns the state, `bc-interp` orchestrates.
//!
//! Placement is where the value is. The streaming executor already prepares every build side
//! before a probe row is read ([`super::prebuild_joins`]), so by the time the probe pipeline is
//! composed the key set is a known constant. Applying it *at the join* would save only a hash
//! lookup — [`bc_runtime::join`]'s own probe bloom already does that. Applying it at the
//! **scan** drops the row before every predicate, projection and copy on the way up. TPC-H q21's
//! `lineitem` probe is 6M rows carrying a date comparison; 411 surviving suppliers reduce it
//! ~24x before that comparison is evaluated once.
//!
//! ## What may be filtered
//!
//! Only a join whose **probe (left) side is reducible**: `Inner` and `Semi`. Those are the join
//! types where a probe row with no match contributes nothing, so dropping it is invisible in the
//! result. `Left`/`Full` must emit their unmatched probe rows null-extended, and `Anti` emits
//! *exactly* the unmatched ones — for those the filter would delete answers, so they get none.
//! This is the same law the control plane's `FILTERABLE_SIDES` encodes for its plan-time
//! sideways-information-passing rules; the two must agree, and they do.
//!
//! Only a single `Int64` equi-key, and only while the key can be traced down the probe pipeline
//! to the node the filter is applied at — through `Filter` (which renames nothing), through a
//! `Project` that passes the column straight through, and through an **inner `HashJoin`**, into
//! whichever side its `output` mapping says the column comes from. Anything else stops the
//! descent, and the filter is placed at the deepest node reached. A key that cannot be traced at
//! all is simply not filtered. See [`sink_target`] for why crossing an inner join is sound, and
//! why it is what makes this optimization reach the plans that need it.
//!
//! ## Keeping the downside bounded
//!
//! A filter that removes nothing is pure cost, and nothing at plan time knows the probe side's
//! key distribution. Three guards bound that, all of them leaning on one property: applying the
//! filter is **optional per morsel**, because it only ever removes provably-non-matching rows,
//! so declining to apply it leaves the same relation and is always legal.
//!
//!   - Per join, [`worth_filtering`] places a filter only over a probe side of at least
//!     [`MIN_PROBE_ROWS`] that is several times its build side — a cost comparison between the
//!     digest and the pass it can save, rather than a flat size gate on the whole query.
//!   - Per morsel, [`apply`] computes the mask but skips the copy unless the mask actually
//!     removes something worth copying for.
//!   - Per filter, a [`Gauge`] watches the keep-rate across morsels and switches a persistently
//!     useless filter off for the rest of the query.
//!
//! These bound the loss. The win is a row count, and it is certain where it applies: at TPC-H
//! sf1, `explain(analyze=True)` shows q21's `lineitem` probe dropping from 3,793,296 rows to
//! 156,739 — a 24x reduction — and its `l_receiptdate > l_commitdate` predicate falling from
//! 122.7 ms of CPU to 5.4 ms; q3's `orders` probe drops 728,486 → 147,126.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use arrow::array::{BooleanArray, RecordBatch};
use arrow::buffer::BooleanBuffer;
use bc_ir::{JoinType, RelOp};
use bc_runtime::join::KeyFilter;

use super::meter::Meter;
use super::{node_key, BuildCache};
use crate::error::InterpError;
use crate::ops;

/// Rows a filter must see before its keep-rate is judged.
///
/// Small enough that a useless filter is switched off early in a big scan, large enough that a
/// leading run of matching rows — a clustered fact table often starts with one — cannot condemn
/// a filter that is highly selective overall. Roughly four morsels.
const GAUGE_WARMUP_ROWS: u64 = 65_536;

/// Keep-rate at or above which a filter is judged not worth its per-row cost, in 1/256ths.
///
/// The cost being weighed is not the membership test — that is one L2 probe per row. It is the
/// `filter_record_batch` the mask forces: a **copy of every column** of every surviving row.
/// At the scan, where this filter is placed, the batch is at its widest (projection pushdown
/// narrows it above, not below), so a filter that keeps most rows replaces a zero-copy morsel
/// slice with a full materialization of the relation. Half is the point where that copy is
/// already being paid on most of the rows it was supposed to remove; the cutoff is set from
/// that argument rather than from a measured sweep.
const GAUGE_KEEP_CUTOFF: u64 = 128; // 128/256 = 0.5

/// Where a runtime filter applies, and the running judgement of whether it should.
pub(crate) struct PendingFilter {
    /// The probe-side column the filter tests, named as it is *at the node it applies to*.
    column: String,
    filter: Arc<KeyFilter>,
    gauge: Gauge,
    /// Apply on every morsel, bypassing both the [`Gauge`] and the per-morsel selectivity skip.
    ///
    /// Set only by [`Switch::Force`]. Both of those guards decline to filter when filtering would
    /// not pay, and on a test-sized batch that is nearly always — so leaving them in place under
    /// `force` would mean a differential suite that "covers" this code by never running it.
    /// Resolved once here rather than read per morsel, because the switch is an environment read.
    force: bool,
}

/// The filters placed on a lazily-read scan, as the reader is handed them
/// ([`crate::UnitSource::read_keyed`]).
///
/// Every filter, whatever its [`Gauge`] says: the gauge judges masking a decoded morsel, and a
/// reader that tests the key during decode is judging a different trade -- `bc-io`'s late
/// materialization times both ways and keeps the faster on its own.
pub(crate) fn scan_keys(filters: &[PendingFilter]) -> Vec<crate::ScanKeyFilter> {
    filters
        .iter()
        .map(|p| (p.column.clone(), Arc::clone(&p.filter)))
        .collect()
}

/// The self-disabling counter described in the module note.
#[derive(Default)]
struct Gauge {
    seen: AtomicU64,
    kept: AtomicU64,
    off: AtomicBool,
}

impl Gauge {
    /// Whether the filter should still be applied. Relaxed throughout: this is a performance
    /// heuristic read and written by many workers, and every possible interleaving produces a
    /// correct — merely differently fast — result.
    #[inline]
    fn enabled(&self) -> bool {
        !self.off.load(Ordering::Relaxed)
    }

    /// Record one morsel's outcome and switch the filter off if it is not earning its keep.
    fn record(&self, seen: u64, kept: u64) {
        let total = self.seen.fetch_add(seen, Ordering::Relaxed) + seen;
        let total_kept = self.kept.fetch_add(kept, Ordering::Relaxed) + kept;
        if total >= GAUGE_WARMUP_ROWS && total_kept * 256 >= total * GAUGE_KEEP_CUTOFF {
            self.off.store(true, Ordering::Relaxed);
        }
    }

    /// The keep-rate seen so far, in 1/256ths — `128` (one half) before anything is seen, so an
    /// unmeasured filter neither jumps ahead of a proven selective one nor falls behind a proven
    /// useless one. Only an ordering hint: see [`apply`].
    fn keep_rate(&self) -> u64 {
        let seen = self.seen.load(Ordering::Relaxed);
        if seen == 0 {
            return GAUGE_KEEP_CUTOFF;
        }
        self.kept.load(Ordering::Relaxed) * 256 / seen
    }
}

/// Filters to apply to a node's output, keyed by [`node_key`].
///
/// A node can carry more than one: a probe pipeline feeding two joins on different keys — the
/// `lineitem` scan under TPC-H q21's supplier and orders joins — is reduced by both.
pub(crate) type RuntimeFilters = HashMap<usize, Vec<PendingFilter>>;

/// Rows a probe side must scan before a filter is placed over it at all.
///
/// Below this the probe side is a few morsels, nothing shards (`parallel::MIN_ROWS_TO_SHARD` is
/// the same four morsels), and the whole relation is a fraction of a millisecond of work — there
/// is nothing for a filter to save that the query is waiting on. This is the small-query guard:
/// a join under it takes the identical code path it took before runtime filters existed.
const MIN_PROBE_ROWS: usize = 4 * bc_arrow::DEFAULT_MORSEL_ROWS;

/// Probe rows a filter needs per build row before the build side is worth digesting.
///
/// This replaces a flat 16M-row gate on the query's *largest input*, which kept the filter off
/// every query whose fact table was smaller than that, however lopsided its joins. TPC-DS sf1 q37
/// and q82 are the shape: an 11.7M-row `inventory` scan carrying a range predicate, joined to a
/// handful of `item` rows and 61 `date_dim` rows. The `item` key set keeps a tiny fraction of
/// `inventory`, and without it most of the query's operator time was that range predicate,
/// evaluated over every row.
///
/// The argument is a comparison between two linear passes that are each cheap per row:
///
///   - **The cost** is [`KeyFilter::build_for_probe`] — one serial pass over the build side's
///     key column, abandoned early for a sparse high-cardinality key — plus a mask of one
///     cache-resident lookup per probe row, which the [`Gauge`] cuts off after
///     [`GAUGE_WARMUP_ROWS`] if it keeps more than half the rows.
///   - **The saving** is every per-row operation above the filtered node, for each row the
///     filter removes: predicates, projections, the join's own hash probe, the gather.
///
/// A build side as large as its probe side is the case that does not pay: the digest touches as
/// many rows as the mask could ever save, and its key set is rarely selective — TPC-H q4's
/// `orders SEMI lineitem`, the cautionary tale in `KeyFilter`'s own notes, builds 3.8M rows
/// against a 1.5M-row probe. Requiring the probe to be several times the build keeps the digest
/// a minor term next to the pass it might save, and leaves "is it selective?" — which nothing at
/// plan time can answer — to the per-morsel [`Gauge`], whose worst case is a mask over its
/// warmup.
///
/// Before this was a per-join ratio the gate was conservative for a stated reason: an A/B on a
/// shared box could not resolve a crossover. The ratio does not need one. Below it the filter is
/// not placed at all, and above it the digest is at most a quarter of a pass over the probe side.
const MIN_PROBE_ROWS_PER_BUILD_ROW: usize = 4;

/// How `BATCHER_RUNTIME_JOIN_FILTER` overrides the default behaviour.
#[derive(PartialEq, Eq)]
enum Switch {
    /// `0` — never filter. The A/B and kill-switch setting.
    Off,
    /// `force` — filter regardless of [`worth_filtering`]. **The test hook**, and it is not
    /// optional: the row gate makes this optimization inert on any input small enough to be a
    /// test fixture, so without a way to force it on, every differential and oracle test would
    /// exercise the path that does nothing and the code would ship unverified.
    Force,
    /// Unset or anything else — the shipped behaviour, gated per join by [`worth_filtering`].
    Default,
}

/// Read the switch. Per call rather than cached in a `OnceLock`, so a harness can alternate
/// settings *query by query* inside one process. That is not a nicety: this engine's benchmarks
/// run on a shared box where machine load drifts over minutes — long enough that running one arm
/// to completion and then the other attributes the load difference to the change. One `getenv`
/// per query (this runs once per `prebuild_joins`, never per row) buys a measurement that is
/// actually about the code.
///
/// The setting only ever changes how fast a query runs, never what it returns, which is what
/// makes it a legitimate switch rather than a semantic flag.
fn switch() -> Switch {
    match std::env::var("BATCHER_RUNTIME_JOIN_FILTER").as_deref() {
        Ok("0") => Switch::Off,
        Ok("force") => Switch::Force,
        _ => Switch::Default,
    }
}

/// Digest every reducible join's build side in `plan` and place the filters over the probe side.
///
/// Runs once per query, after [`super::prebuild_joins`] has filled `cache` — every key set it
/// reads is therefore already computed, and this adds one pass over each build side's key
/// column, no execution.
///
/// `driving` is `(source id, rows)` for a relation the executor streams rather than holds: its
/// entry in `sources` is then a zero-row schema carrier, and without its real size each join's
/// probe-size estimate — which both [`worth_filtering`] and the digest's limits read — would
/// read it as empty. That is
/// what kept every filter off for TPC-H at sf10 and sf100, whose `lineitem` is streamed.
pub(crate) fn plan_filters(
    plan: &RelOp,
    sources: &[Vec<RecordBatch>],
    cache: &BuildCache,
    driving: Option<(usize, usize)>,
    meter: Option<&Meter>,
) -> RuntimeFilters {
    let mut out = RuntimeFilters::new();
    let switch = switch();
    if switch == Switch::Off {
        return out;
    }
    let rows = SourceRows {
        sources,
        driving,
        meter,
        force: switch == Switch::Force,
    };
    collect(plan, cache, &rows, &mut out);
    out
}

/// Row counts of a query's relations, with a streamed relation's real size in place of its
/// zero-row carrier (see [`plan_filters`]).
struct SourceRows<'a> {
    sources: &'a [Vec<RecordBatch>],
    driving: Option<(usize, usize)>,
    /// The query's meter, told which operators each placed filter reduces the input of.
    meter: Option<&'a Meter>,
    /// [`Switch::Force`]: place every filter the digest admits, bypassing [`worth_filtering`].
    force: bool,
}

impl SourceRows<'_> {
    fn of(&self, source_id: usize) -> usize {
        match self.driving {
            Some((id, rows)) if id == source_id => rows,
            _ => self.sources.get(source_id).map_or(0, |relation| {
                relation.iter().map(RecordBatch::num_rows).sum::<usize>()
            }),
        }
    }

    /// Rows scanned anywhere under `plan`: an upper bound on the rows a probe side can feed its
    /// join, which is all [`KeyFilter::build_for_probe`] needs to judge the ratio.
    fn scanned(&self, plan: &RelOp) -> usize {
        match plan {
            RelOp::Scan { source_id } => self.of(*source_id),
            _ => plan.children().iter().map(|c| self.scanned(c)).sum(),
        }
    }
}

/// Whether a join's filter is worth placing over a probe side scanning `probe_rows` rows, against
/// a build side of `build_rows` — see [`MIN_PROBE_ROWS`] and [`MIN_PROBE_ROWS_PER_BUILD_ROW`].
///
/// Decided per join rather than per query: one query can hold a lopsided star join that pays and
/// a join of two equal halves that does not, and a query-wide gate gets one of them wrong.
fn worth_filtering(probe_rows: usize, build_rows: usize) -> bool {
    probe_rows >= MIN_PROBE_ROWS
        && probe_rows >= build_rows.saturating_mul(MIN_PROBE_ROWS_PER_BUILD_ROW)
}

fn collect(plan: &RelOp, cache: &BuildCache, rows: &SourceRows<'_>, out: &mut RuntimeFilters) {
    if let RelOp::HashJoin {
        left,
        left_keys,
        right_keys,
        join_type,
        ..
    } = plan
    {
        let probe_rows = rows.scanned(left);
        place_for_join(
            plan, left, left_keys, right_keys, *join_type, cache, probe_rows, rows, out,
        );
    }
    for child in plan.children() {
        collect(child, cache, rows, out);
    }
}

/// Place one join's filter, if it has one to give.
#[allow(clippy::too_many_arguments)]
fn place_for_join(
    join: &RelOp,
    probe: &RelOp,
    left_keys: &[String],
    right_keys: &[String],
    join_type: JoinType,
    cache: &BuildCache,
    probe_rows: usize,
    rows: &SourceRows<'_>,
    out: &mut RuntimeFilters,
) {
    // `Inner`/`Semi` only — see the module note on which sides may be reduced.
    if !matches!(join_type, JoinType::Inner | JoinType::Semi) {
        return;
    }
    // One key column's digest. For a composite key it is a filter on a *projection* of the
    // key: weaker than the key itself, and still sound, because a probe row whose value for one
    // key column is absent from the build side cannot match on all of them. The most selective
    // column is kept (the fewest distinct build values). TPC-H q20 semi-joins `lineitem` on
    // `(l_partkey, l_suppkey)` to 86K `partsupp` rows whose part keys are 21.5K of 2M, so the
    // `l_partkey` digest alone drops ~99% of the scan ahead of its date filter; the single-key
    // rule this replaces placed no filter at all.
    let Some(prepared) = cache.get(&node_key(join)) else {
        return;
    };
    // Gate before digesting: the digest is the cost the gate exists to avoid paying.
    if !rows.force && !worth_filtering(probe_rows, prepared.side.num_rows()) {
        return;
    }
    let mut best: Option<(KeyFilter, &String)> = None;
    for (probe_key, build_key) in left_keys.iter().zip(right_keys) {
        let Ok(build_col) = ops::columns_by_name(&prepared.side, std::slice::from_ref(build_key))
        else {
            continue;
        };
        let Some(filter) = build_col
            .first()
            .and_then(|keys| KeyFilter::build_for_probe(keys, Some(probe_rows)))
        else {
            continue;
        };
        if best
            .as_ref()
            .is_none_or(|(b, _)| filter.distinct_keys() < b.distinct_keys())
        {
            best = Some((filter, probe_key));
        }
    }
    let Some((filter, probe_key)) = best else {
        return;
    };
    let (target, column) = sink_target(probe, probe_key);
    if let Some(meter) = rows.meter {
        for node in path_to(probe, target) {
            meter.mark_runtime_filtered(node);
        }
    }
    out.entry(node_key(target))
        .or_default()
        .push(PendingFilter {
            column,
            filter: Arc::new(filter),
            gauge: Gauge::default(),
            force: rows.force,
        });
}

/// The nodes from `from` down to `target`, excluding `target`: the operators that consume the
/// rows a filter placed on `target`'s output removes, found by address because [`sink_target`]
/// only ever descends to a child. Empty when `target` is `from` or not beneath it.
fn path_to<'a>(from: &'a RelOp, target: &RelOp) -> Vec<&'a RelOp> {
    if std::ptr::eq(from, target) {
        return Vec::new();
    }
    for child in from.children() {
        let mut path = path_to(child, target);
        if !path.is_empty() || std::ptr::eq(child, target) {
            path.insert(0, from);
            return path;
        }
    }
    Vec::new()
}

/// The deepest node in `probe` whose output still carries the join key, and the key's name there.
///
/// Descends only through nodes that neither drop nor recompute the column: a `Filter` (which
/// changes which rows exist, not which columns), a `Project` that passes the column through as a
/// bare column reference, and an **inner `HashJoin`**, into whichever side the column comes from.
/// A `Project` that *computes* the key, or any other node, ends the descent — the filter is then
/// placed on that node's output, where the column demonstrably still exists and still holds join
/// keys.
///
/// ## Why descending through an inner join matters, and why it is sound
///
/// Stopping at a join is what confined this optimization to a star join whose fact table is the
/// *immediate* probe input. Real plans are not shaped that way: TPC-H q5 joins `lineitem` to the
/// date-filtered `orders` first and only then to the 20,037 ASIA suppliers, so the supplier key
/// set — which keeps roughly one `lineitem` row in five — could only be applied to the 9.1M-row
/// join output, long after the 60M-row scan it should have reduced. The same shape recurs in q7,
/// q9 and q10.
///
/// Soundness is the join's own algebra. Every output row of `C ⋈ D` takes its value of a
/// left-sourced column from exactly one row of `C` (and a right-sourced one from `D`) — the join
/// pairs rows, it never invents or alters a value. So a row of `C` whose key the filter refutes
/// can only produce output rows the *outer* join would refute in turn, and removing it earlier
/// removes exactly the same rows from the final answer. The `output` mapping is followed to pick
/// the side and the pre-join name, so a renamed or collided alias tracks the real column.
///
/// Inner only. An outer join *manufactures* NULLs on its null-extended side, so a row that the
/// filter refutes there may still be needed to produce a null-extended output row, and a semi or
/// anti join's output does not carry the right side's columns at all. Restricting to `Inner` is
/// what makes "the value came from one input row" true without further reasoning.
fn sink_target<'a>(probe: &'a RelOp, key: &str) -> (&'a RelOp, String) {
    let mut node = probe;
    let mut name = key.to_string();
    loop {
        match node {
            RelOp::Filter { input, .. } => node = input,
            RelOp::Project { input, exprs } => {
                let Some(item) = exprs.iter().find(|p| p.alias == name) else {
                    return (node, name);
                };
                match &item.expr {
                    bc_expr::Expr::Col { name: source } => {
                        name = source.clone();
                        node = input;
                    }
                    _ => return (node, name),
                }
            }
            RelOp::HashJoin {
                left,
                right,
                join_type: JoinType::Inner,
                output,
                ..
            } => {
                let Some(col) = output.iter().find(|c| c.alias == name) else {
                    return (node, name);
                };
                name = col.name.clone();
                node = match col.side {
                    bc_ir::JoinSide::Left => left,
                    bc_ir::JoinSide::Right => right,
                };
            }
            // An `Aggregate`, on one of its **group keys**. Every output row's key value is one
            // that appeared in the input, so a key the filter refutes describes a group the join
            // above would discard whole — and deleting that group's *input* rows deletes exactly
            // that group and nothing else. Aggregate values are never touched, because a group is
            // either entirely kept or entirely removed.
            //
            // This is the placement a decorrelated correlated subquery needs. It lowers to
            // `Join(outer, Aggregate(inner, group_keys=[k]))`, and the aggregate is computed over
            // the *whole* inner relation even though only the outer's few keys are ever read.
            // Filtering at the join saves nothing (the groups are already built); filtering here
            // means they are never built. Unlike the plan-time semi-join that expresses the same
            // idea (`kyber.rules.joins.agg_semijoin`), this costs no extra pass over the input —
            // the mask rides the scan the aggregate was doing anyway, which is why that rule
            // refuses shapes this can still serve.
            RelOp::Aggregate {
                input, group_keys, ..
            } => {
                let Some(item) = group_keys.iter().find(|k| k.alias == name) else {
                    return (node, name); // an aggregate *value*, not a key: no such guarantee
                };
                match &item.expr {
                    bc_expr::Expr::Col { name: source } => {
                        name = source.clone();
                        node = input;
                    }
                    // A computed key (`GROUP BY lower(x)`) cannot be inverted into a predicate on
                    // an input column, so the filter stops at the aggregate's output.
                    _ => return (node, name),
                }
            }
            _ => return (node, name),
        }
    }
}

/// Apply every enabled filter registered for a node to one of its output morsels.
///
/// Returns `batch` untouched when nothing applies — the overwhelmingly common case, and it must
/// stay free. A filter whose column is missing from the batch is skipped rather than raised on:
/// the mask is an optimization, and refusing to run a correct query because a placement guess
/// did not hold is the wrong trade.
///
/// ## Several filters on one node
///
/// A fact scan under a star join is reduced by every dimension, and how the filters are combined
/// is most of their cost. A mask is one cache-resident lookup per row, so what a mask actually
/// costs is **reading its key column**: an 8-byte value per row streamed from memory, at the
/// widest point of the pipeline. TPC-DS q37 places two filters on an 11.7M-row `inventory` scan,
/// and masking both over every row spent two-thirds of the query's CPU re-reading key columns,
/// bandwidth-bound.
///
/// So the filters run **most selective first** (by the keep-rate each [`Gauge`] has measured),
/// and the morsel is cut down to the survivors as soon as the masks so far remove enough to be
/// worth a copy. Every later filter then reads its key column only at the survivors — after an
/// `item` filter that keeps one row in ten thousand, the `date` filter touches almost none of
/// its column. A mask that removes too little to copy for is ANDed into the next one instead,
/// so a batch is never copied twice for one decision. The order and the cut change only how
/// much work the filters do: each removes exactly the rows its key set refutes, so any order
/// leaves the same relation.
pub(crate) fn apply(
    filters: &[PendingFilter],
    batch: RecordBatch,
) -> Result<RecordBatch, InterpError> {
    if batch.num_rows() == 0 || filters.is_empty() {
        return Ok(batch);
    }
    let mut order: Vec<&PendingFilter> = filters
        .iter()
        .filter(|p| p.force || p.gauge.enabled())
        .collect();
    if order.len() > 1 {
        order.sort_by_key(|p| p.gauge.keep_rate());
    }
    let mut out = batch;
    // Masks over `out` that removed too little, so far, to be worth acting on.
    let mut pending_mask: Option<BooleanBuffer> = None;
    for pending in order {
        let Some(col) = out.column_by_name(&pending.column) else {
            continue;
        };
        let Some(mask) = pending.filter.mask(col) else {
            continue;
        };
        let rows = out.num_rows() as u64;
        // A gauge judges its filter over the rows it was actually handed, which after a more
        // selective filter is the survivors: a filter made redundant by another is then seen to
        // keep nearly everything, and is switched off.
        pending
            .gauge
            .record(rows, mask.values().count_set_bits() as u64);
        let mask = match pending_mask.take() {
            None => mask.values().clone(),
            Some(acc) => &acc & mask.values(),
        };
        let kept = mask.count_set_bits() as u64;
        // Deciding *per morsel* whether to act on the mask is what bounds this optimization's
        // downside. Acting on it means `filter_record_batch`, a copy of every column of every
        // surviving row — and at the scan, where the filter is placed, the batch is at its widest,
        // because projection pushdown narrows it above rather than below. So a mask that keeps
        // most rows would replace a zero-copy morsel slice with a near-full materialization of
        // the relation, to remove almost nothing.
        //
        // Declining is always legal: the filter only ever removes rows that provably cannot
        // match, so *not* removing them leaves the same relation, merely larger. That asymmetry —
        // free to decline, expensive to act — is why the test is here and not only in the
        // [`Gauge`]. The gauge still saw this morsel's outcome, so a filter that keeps declining
        // is switched off for good rather than re-masking forever.
        if !pending.force && kept * 2 > rows {
            pending_mask = Some(mask);
            continue;
        }
        if kept == 0 {
            return Ok(out.slice(0, 0));
        }
        out = arrow::compute::filter_record_batch(&out, &BooleanArray::new(mask, None))?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use bc_expr::Expr;
    use bc_ir::ProjectionItem;

    use super::*;

    fn scan(id: usize) -> RelOp {
        RelOp::Scan { source_id: id }
    }

    fn col(name: &str) -> Expr {
        Expr::Col { name: name.into() }
    }

    fn project(input: RelOp, items: &[(&str, Expr)]) -> RelOp {
        RelOp::Project {
            input: Box::new(input),
            exprs: items
                .iter()
                .map(|(alias, expr)| ProjectionItem {
                    expr: expr.clone(),
                    alias: (*alias).into(),
                })
                .collect(),
        }
    }

    /// The placement that makes the whole optimization worth doing: through a `Filter` and a
    /// pass-through `Project`, all the way to the `Scan`, tracking the rename.
    #[test]
    fn descends_through_filter_and_passthrough_project_to_the_scan() {
        let plan = project(
            RelOp::Filter {
                input: Box::new(scan(0)),
                predicate: col("keep"),
            },
            &[("sk", col("l_suppkey"))],
        );
        let (target, name) = sink_target(&plan, "sk");
        assert!(matches!(target, RelOp::Scan { source_id: 0 }));
        assert_eq!(name, "l_suppkey", "the rename must be followed down");
    }

    fn inner_join(left: RelOp, right: RelOp, out: &[(bc_ir::JoinSide, &str, &str)]) -> RelOp {
        RelOp::HashJoin {
            left: Box::new(left),
            right: Box::new(right),
            left_keys: vec!["l_orderkey".into()],
            right_keys: vec!["o_orderkey".into()],
            join_type: JoinType::Inner,
            output: out
                .iter()
                .map(|(side, name, alias)| bc_ir::JoinOutputCol {
                    side: *side,
                    name: (*name).into(),
                    alias: (*alias).into(),
                })
                .collect(),
            strategy: bc_ir::JoinStrategy::Hash,
        }
    }

    /// The placement TPC-H q5 needs: the supplier key set must reach the `lineitem` scan even
    /// though a `lineitem ⋈ orders` join sits between them. Stopping at the join applied it to
    /// the 9.1M-row join output instead of the 60M-row scan.
    #[test]
    fn descends_through_an_inner_join_into_the_side_that_carries_the_key() {
        let plan = inner_join(
            project(scan(0), &[("l_suppkey", col("l_suppkey"))]),
            scan(1),
            &[(bc_ir::JoinSide::Left, "l_suppkey", "sk")],
        );
        let (target, name) = sink_target(&plan, "sk");
        assert!(
            matches!(target, RelOp::Scan { source_id: 0 }),
            "must reach the probe-side scan, not stop at the join"
        );
        assert_eq!(
            name, "l_suppkey",
            "the join's output mapping renames it back"
        );
    }

    /// The mapping decides the side: a right-sourced column descends into the build side.
    #[test]
    fn descends_into_the_build_side_when_the_key_comes_from_there() {
        let plan = inner_join(
            scan(0),
            project(scan(1), &[("o_custkey", col("o_custkey"))]),
            &[(bc_ir::JoinSide::Right, "o_custkey", "ck")],
        );
        let (target, name) = sink_target(&plan, "ck");
        assert!(matches!(target, RelOp::Scan { source_id: 1 }));
        assert_eq!(name, "o_custkey");
    }

    /// An outer join manufactures NULLs on its null-extended side, so a refuted row may still be
    /// needed to produce an output row. The descent must stop there.
    #[test]
    fn stops_at_a_non_inner_join() {
        let mut plan = inner_join(
            scan(0),
            scan(1),
            &[(bc_ir::JoinSide::Left, "l_suppkey", "sk")],
        );
        if let RelOp::HashJoin { join_type, .. } = &mut plan {
            *join_type = JoinType::Left;
        }
        let (target, name) = sink_target(&plan, "sk");
        assert!(matches!(target, RelOp::HashJoin { .. }));
        assert_eq!(name, "sk");
    }

    /// A column the join's output mapping does not name ends the descent, rather than tracking a
    /// name that means something else on one of the sides.
    #[test]
    fn stops_at_an_inner_join_that_does_not_carry_the_key() {
        let plan = inner_join(
            scan(0),
            scan(1),
            &[(bc_ir::JoinSide::Left, "something_else", "other")],
        );
        let (target, _) = sink_target(&plan, "sk");
        assert!(matches!(target, RelOp::HashJoin { .. }));
    }

    fn aggregate(input: RelOp, keys: &[(&str, Expr)]) -> RelOp {
        RelOp::Aggregate {
            input: Box::new(input),
            group_keys: keys
                .iter()
                .map(|(alias, expr)| ProjectionItem {
                    expr: expr.clone(),
                    alias: (*alias).into(),
                })
                .collect(),
            aggregates: Vec::new(),
        }
    }

    /// A decorrelated correlated subquery is `Join(outer, Aggregate(inner, by k))`, and the
    /// aggregate is built over the whole inner relation for the sake of the outer's few keys.
    /// The filter must reach the aggregate's *input*, where the groups are never built, rather
    /// than its output, where they already have been.
    #[test]
    fn descends_through_an_aggregate_on_its_group_key() {
        let plan = aggregate(
            project(scan(0), &[("l_orderkey", col("l_orderkey"))]),
            &[("k", col("l_orderkey"))],
        );
        let (target, name) = sink_target(&plan, "k");
        assert!(
            matches!(target, RelOp::Scan { source_id: 0 }),
            "must reach the input scan, not stop at the aggregate"
        );
        assert_eq!(name, "l_orderkey");
    }

    /// An aggregate *value* carries no such guarantee — `sum(x)` for a refuted key says nothing
    /// about which input rows produced it — so the descent stops at the aggregate's output.
    #[test]
    fn stops_at_an_aggregate_value_column() {
        let plan = aggregate(scan(0), &[("k", col("l_orderkey"))]);
        let (target, name) = sink_target(&plan, "total");
        assert!(matches!(target, RelOp::Aggregate { .. }));
        assert_eq!(name, "total");
    }

    /// A computed group key cannot be inverted into a predicate on an input column.
    #[test]
    fn stops_at_an_aggregate_with_a_computed_group_key() {
        let plan = aggregate(
            scan(0),
            &[(
                "k",
                Expr::Binary {
                    op: bc_expr::BinaryOp::Add,
                    left: Box::new(col("a")),
                    right: Box::new(col("b")),
                },
            )],
        );
        let (target, _) = sink_target(&plan, "k");
        assert!(matches!(target, RelOp::Aggregate { .. }));
    }

    /// A computed key ends the descent: below the `Project` the column does not exist, and the
    /// values above it are not the scan's.
    #[test]
    fn stops_at_a_project_that_computes_the_key() {
        let plan = project(
            scan(0),
            &[(
                "k",
                Expr::Binary {
                    op: bc_expr::BinaryOp::Add,
                    left: Box::new(col("a")),
                    right: Box::new(col("b")),
                },
            )],
        );
        let (target, name) = sink_target(&plan, "k");
        assert!(matches!(target, RelOp::Project { .. }));
        assert_eq!(name, "k");
    }

    /// A key the projection does not produce at all also ends the descent, rather than
    /// silently tracking a name that means something else below.
    #[test]
    fn stops_at_a_project_that_does_not_carry_the_key() {
        let plan = project(scan(0), &[("other", col("x"))]);
        let (target, _) = sink_target(&plan, "k");
        assert!(matches!(target, RelOp::Project { .. }));
    }

    /// An un-descendable probe side places the filter on itself, which is still sound.
    #[test]
    fn a_breaker_probe_side_keeps_the_filter_at_its_own_output() {
        let plan = RelOp::Distinct {
            input: Box::new(scan(0)),
            keys: Vec::new(),
            order: Vec::new(),
            limit: None,
        };
        let (target, name) = sink_target(&plan, "k");
        assert!(matches!(target, RelOp::Distinct { .. }));
        assert_eq!(name, "k");
    }

    /// The gauge switches a useless filter off, and only after the warmup.
    #[test]
    fn gauge_disables_a_filter_that_keeps_almost_everything() {
        let g = Gauge::default();
        assert!(g.enabled());
        // A pass-everything filter, but still inside the warmup.
        g.record(GAUGE_WARMUP_ROWS / 2, GAUGE_WARMUP_ROWS / 2);
        assert!(g.enabled(), "must not judge before the warmup");
        g.record(GAUGE_WARMUP_ROWS, GAUGE_WARMUP_ROWS);
        assert!(!g.enabled(), "a filter keeping 100% must switch itself off");
    }

    /// A selective filter stays on however long it runs.
    #[test]
    fn gauge_keeps_a_selective_filter_on() {
        let g = Gauge::default();
        for _ in 0..100 {
            g.record(GAUGE_WARMUP_ROWS, GAUGE_WARMUP_ROWS / 10);
        }
        assert!(g.enabled());
    }

    /// A streamed probe relation is a zero-row carrier in `sources`, so only the driving-row
    /// hint can tell placement how large it is. With the hint, a build side too sparse for the
    /// probe limits is digested for a probe side 300x its size; without it, the same plan
    /// places nothing, because the carrier reads as empty. The second half is the control that
    /// makes the first mean something: it is the behaviour before the hint existed.
    #[test]
    fn a_driving_row_hint_places_a_filter_the_carrier_alone_cannot() {
        use std::sync::Arc;

        use arrow::array::Int64Array;
        use arrow::datatypes::{DataType, Field, Schema};

        let probe = Arc::new(Schema::new(vec![Field::new(
            "l_orderkey",
            DataType::Int64,
            true,
        )]));
        let carrier = RecordBatch::new_empty(probe);
        let keys: Vec<i64> = (0..70_000i64).map(|i| i * 100).collect();
        let build = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "o_orderkey",
                DataType::Int64,
                true,
            )])),
            vec![Arc::new(Int64Array::from(keys))],
        )
        .unwrap();
        let sources = vec![vec![carrier], vec![build]];
        let plan = inner_join(
            scan(0),
            scan(1),
            &[(bc_ir::JoinSide::Left, "l_orderkey", "l_orderkey")],
        );
        let RelOp::HashJoin { left, .. } = &plan else {
            unreachable!()
        };
        let placed = |driving| {
            let cache = crate::stream::builds::prebuild_joins_for_chunks(
                &plan, &sources, None, 0, 1, None, driving,
            )
            .unwrap();
            cache
                .filters_for(node_key(left))
                .map(<[PendingFilter]>::len)
        };
        assert_eq!(placed(Some((0, 70_000 * 300))), Some(1));
        assert_eq!(placed(None), None);
    }

    /// A composite key places one filter, on the key column with the fewest distinct build
    /// values — TPC-H q20's `(l_partkey, l_suppkey)` shape. The build side's `a` column holds 10
    /// values and its `b` column 70,000, so the digest must test the probe's `pa`, not `pb`; the
    /// order of the key pairs is reversed as well, so the first pair winning by position would
    /// pick the wrong column.
    #[test]
    fn a_composite_key_filters_on_its_most_selective_column() {
        use std::sync::Arc;

        use arrow::array::Int64Array;
        use arrow::datatypes::{DataType, Field, Schema};

        let int = |n: &str| Field::new(n, DataType::Int64, true);
        let carrier = RecordBatch::new_empty(Arc::new(Schema::new(vec![int("pa"), int("pb")])));
        let build = RecordBatch::try_new(
            Arc::new(Schema::new(vec![int("a"), int("b")])),
            vec![
                Arc::new(Int64Array::from_iter_values(
                    (0..70_000i64).map(|i| i % 10 * 7),
                )),
                Arc::new(Int64Array::from_iter_values((0..70_000i64).map(|i| i * 3))),
            ],
        )
        .unwrap();
        let sources = vec![vec![carrier], vec![build]];
        let plan = RelOp::HashJoin {
            left: Box::new(scan(0)),
            right: Box::new(scan(1)),
            left_keys: vec!["pb".into(), "pa".into()],
            right_keys: vec!["b".into(), "a".into()],
            join_type: JoinType::Semi,
            output: vec![bc_ir::JoinOutputCol {
                side: bc_ir::JoinSide::Left,
                name: "pa".into(),
                alias: "pa".into(),
            }],
            strategy: bc_ir::JoinStrategy::Hash,
        };
        let RelOp::HashJoin { left, .. } = &plan else {
            unreachable!()
        };
        let cache = crate::stream::builds::prebuild_joins_for_chunks(
            &plan,
            &sources,
            None,
            0,
            1,
            None,
            Some((0, 70_000 * 300)),
        )
        .unwrap();
        let placed = cache
            .filters_for(node_key(left))
            .expect("a filter is placed");
        assert_eq!(
            placed.len(),
            1,
            "one filter per join, not one per key column"
        );
        assert_eq!(placed[0].column, "pa");
        assert_eq!(placed[0].filter.distinct_keys(), 10);
    }

    /// The meter lists exactly the operators between a placed filter and its join: here the
    /// `Filter` and `Project` above the filtered scan. The scan's own count is taken before the
    /// filter acts and the join's output is exact, so listing either would withhold a true
    /// count from the learning loop — the control that the list is not simply "everything".
    #[test]
    fn the_meter_lists_the_operators_a_placed_filter_reduces() {
        use std::sync::Arc;

        use arrow::array::Int64Array;
        use arrow::datatypes::{DataType, Field, Schema};

        let carrier = RecordBatch::new_empty(Arc::new(Schema::new(vec![
            Field::new("l_orderkey", DataType::Int64, true),
            Field::new("keep", DataType::Boolean, true),
        ])));
        let build = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "o_orderkey",
                DataType::Int64,
                true,
            )])),
            vec![Arc::new(Int64Array::from_iter_values(
                (0..70_000i64).map(|i| i * 100),
            ))],
        )
        .unwrap();
        let sources = vec![vec![carrier], vec![build.clone()]];
        let probe = project(
            RelOp::Filter {
                input: Box::new(scan(0)),
                predicate: col("keep"),
            },
            &[("l_orderkey", col("l_orderkey"))],
        );
        let plan = inner_join(
            probe,
            scan(1),
            &[(bc_ir::JoinSide::Left, "l_orderkey", "l_orderkey")],
        );
        let meter = crate::stream::meter::Meter::new(&plan, 1);
        crate::stream::builds::prebuild_joins_for_chunks(
            &plan,
            &sources,
            Some(&meter),
            0,
            1,
            None,
            Some((0, 70_000 * 300)),
        )
        .unwrap();
        // Pre-order: join 0, project 1, filter 2, scan 3, build scan 4. Every operator must have
        // run to be reported at all.
        for id in 0..5 {
            meter.morsel(id, 1, &build, 1);
        }
        assert_eq!(meter.finish().runtime_filtered, vec![1, 2]);
    }

    /// The per-join cost gate: a probe side must be big in absolute terms *and* relative to its
    /// build side. Each clause is pinned on both sides of its boundary.
    #[test]
    fn the_gate_weighs_the_probe_against_the_build() {
        // Big probe, tiny build: the star-join shape the gate exists to admit.
        assert!(worth_filtering(11_745_000, 2));
        // Below the absolute floor, however lopsided.
        assert!(!worth_filtering(MIN_PROBE_ROWS - 1, 1));
        assert!(worth_filtering(MIN_PROBE_ROWS, 1));
        // A build side as large as the probe: the digest costs what the mask could save.
        assert!(!worth_filtering(1_500_000, 3_800_000));
        let build = 1_000_000;
        assert!(!worth_filtering(
            build * MIN_PROBE_ROWS_PER_BUILD_ROW - 1,
            build
        ));
        assert!(worth_filtering(build * MIN_PROBE_ROWS_PER_BUILD_ROW, build));
    }

    fn i64_batch(cols: &[(&str, Vec<i64>)]) -> RecordBatch {
        let fields: Vec<arrow::datatypes::Field> = cols
            .iter()
            .map(|(n, _)| arrow::datatypes::Field::new(*n, arrow::datatypes::DataType::Int64, true))
            .collect();
        let arrays: Vec<arrow::array::ArrayRef> = cols
            .iter()
            .map(|(_, v)| {
                Arc::new(arrow::array::Int64Array::from(v.clone())) as arrow::array::ArrayRef
            })
            .collect();
        RecordBatch::try_new(Arc::new(arrow::datatypes::Schema::new(fields)), arrays).unwrap()
    }

    /// The gate sizes a probe side from the sources, never by running it, and reads a streamed
    /// relation's real size rather than its zero-row carrier.
    #[test]
    fn the_probe_size_counts_every_scan_beneath_it_and_honours_a_streamed_source() {
        let sources = vec![
            vec![
                i64_batch(&[("a", vec![1; 10])]),
                i64_batch(&[("a", vec![1; 5])]),
            ],
            vec![i64_batch(&[("b", vec![1; 7])])],
        ];
        let rows = SourceRows {
            sources: &sources,
            driving: None,
            meter: None,
            force: false,
        };
        assert_eq!(rows.scanned(&scan(0)), 15);
        assert_eq!(rows.scanned(&inner_join(scan(1), scan(0), &[])), 22);
        assert_eq!(
            rows.scanned(&scan(9)),
            0,
            "an unknown source sizes to nothing"
        );
        let streamed = SourceRows {
            driving: Some((1, 9_000_000)),
            ..rows
        };
        assert_eq!(streamed.scanned(&scan(1)), 9_000_000);
    }

    fn pending(column: &str, build: Vec<i64>, force: bool) -> PendingFilter {
        let keys: arrow::array::ArrayRef = Arc::new(arrow::array::Int64Array::from(build));
        PendingFilter {
            column: column.into(),
            filter: Arc::new(KeyFilter::build(&keys).unwrap()),
            gauge: Gauge::default(),
            force,
        }
    }

    fn column(b: &RecordBatch, name: &str) -> Vec<i64> {
        b.column_by_name(name)
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .unwrap()
            .values()
            .to_vec()
    }

    /// Two filters on one node keep exactly the rows that pass *both*, with the rest of each
    /// row intact — the AND that replaces one copy per join.
    #[test]
    fn several_filters_on_one_node_keep_the_rows_that_pass_all_of_them() {
        let n = 1_000i64;
        let batch = i64_batch(&[
            ("a", (0..n).collect()),
            ("b", (0..n).map(|i| i % 10).collect()),
            ("payload", (0..n).map(|i| i * 100).collect()),
        ]);
        let filters = [
            pending("a", (0..n).filter(|i| i % 2 == 0).collect(), false),
            pending("b", vec![0, 4], false),
        ];
        let out = apply(&filters, batch).unwrap();
        let expect: Vec<i64> = (0..n)
            .filter(|i| i % 2 == 0 && [0, 4].contains(&(i % 10)))
            .collect();
        assert_eq!(column(&out, "a"), expect);
        assert_eq!(
            column(&out, "payload"),
            expect.iter().map(|i| i * 100).collect::<Vec<_>>()
        );
    }

    /// The filter measured as more selective runs first, and the other then reads only the
    /// survivors — the bandwidth saving the ordering exists for — while the rows kept are the
    /// same as in plan order.
    #[test]
    fn the_more_selective_filter_runs_first_and_the_other_sees_only_its_survivors() {
        let n = 1_000i64;
        let batch = i64_batch(&[("a", (0..n).collect()), ("b", (0..n).collect())]);
        let halves = pending("a", (0..n).filter(|i| i % 2 == 0).collect(), false);
        let few = pending("b", (0..n).filter(|i| i % 100 == 0).collect(), false);
        // History says `few` keeps 1% and `halves` 50%, though `halves` is listed first.
        halves.gauge.record(1_000, 500);
        few.gauge.record(1_000, 10);
        let filters = [halves, few];
        let out = apply(&filters, batch).unwrap();
        assert_eq!(
            column(&out, "a"),
            (0..n).filter(|i| i % 100 == 0).collect::<Vec<_>>()
        );
        assert_eq!(
            filters[0].gauge.seen.load(Ordering::Relaxed),
            1_000 + 10,
            "the less selective filter must have read only the 10 survivors"
        );
        assert_eq!(filters[1].gauge.seen.load(Ordering::Relaxed), 1_000 + 1_000);
    }

    /// A mask that keeps more than half the morsel is not acted on (the copy would cost more
    /// than it removes) — and declining is legal, because the join above still refutes the rows.
    #[test]
    fn a_mask_keeping_most_rows_leaves_the_morsel_whole() {
        let batch = i64_batch(&[("a", (0..100).collect())]);
        let filters = [pending("a", (0..90).collect(), false)];
        assert_eq!(apply(&filters, batch.clone()).unwrap().num_rows(), 100);
        let forced = [pending("a", (0..90).collect(), true)];
        assert_eq!(apply(&forced, batch).unwrap().num_rows(), 90);
    }

    /// A filter whose column is absent, or a morsel with no rows, passes through untouched.
    #[test]
    fn a_missing_column_or_an_empty_morsel_is_passed_through() {
        let batch = i64_batch(&[("a", (0..10).collect())]);
        let filters = [pending("zzz", vec![1], true)];
        assert_eq!(apply(&filters, batch.clone()).unwrap().num_rows(), 10);
        let empty = batch.slice(0, 0);
        let filters = [pending("a", vec![1], true)];
        assert_eq!(apply(&filters, empty).unwrap().num_rows(), 0);
        let none = [pending("a", vec![-5], true)];
        assert_eq!(apply(&none, batch).unwrap().num_rows(), 0);
    }
}
