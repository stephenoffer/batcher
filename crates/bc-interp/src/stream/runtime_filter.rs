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
//!   - [`MIN_SOURCE_ROWS`] keeps the whole mechanism out of small queries, where the CPU it
//!     saves is not what they are waiting for.
//!   - Per morsel, [`apply`] computes the mask but skips the copy unless the mask actually
//!     removes something worth copying for.
//!   - Per filter, a [`Gauge`] watches the keep-rate across morsels and switches a persistently
//!     useless filter off for the rest of the query.
//!
//! These bound the loss; they do not prove a win. The row reductions this achieves are certain
//! (they are counts, see [`MIN_SOURCE_ROWS`]); the wall-clock effect at scale was not measurable
//! on the benchmark box this was developed on, and is the open item.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use arrow::array::RecordBatch;
use bc_ir::{JoinType, RelOp};
use bc_runtime::join::KeyFilter;

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
/// that argument rather than from a measured sweep, which the benchmark box was too noisy to
/// supply (see [`MIN_SOURCE_ROWS`]).
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
}

/// Filters to apply to a node's output, keyed by [`node_key`].
///
/// A node can carry more than one: a probe pipeline feeding two joins on different keys — the
/// `lineitem` scan under TPC-H q21's supplier and orders joins — is reduced by both.
pub(crate) type RuntimeFilters = HashMap<usize, Vec<PendingFilter>>;

/// Rows in the largest input relation below which runtime filtering does not engage at all.
///
/// The gate exists because this optimization trades **CPU work for latency**, and only one of
/// those is what a small query is short of.
///
/// What the filter buys is measured and certain, because it is a row count rather than a
/// timing. At TPC-H sf1, `explain(analyze=True)` shows q21's `lineitem` probe dropping from
/// 3,793,296 rows to 156,739 — a 24x reduction — and its `l_receiptdate > l_commitdate`
/// predicate falling from 122.7 ms of CPU to 5.4 ms; q3's `orders` probe drops 728,486 → 147,126.
///
/// What it costs is a digest of the build side's key column plus a mask per probe morsel, both
/// on the critical path and neither parallelised. On a 96-core box a 6M-row table is already
/// only a couple of milliseconds of *wall* clock however much CPU it burns, so at that size the
/// saving lands somewhere the query was not waiting and the cost lands somewhere it was. The
/// filter needs a probe side big enough that throughput, not per-operator launch latency, is
/// what the query is spending.
///
/// **16M is a deliberately conservative placement, not a measured crossover.** The crossover was
/// not measurable here: the benchmark box runs several concurrent build/benchmark workloads, and
/// an A/B over byte-identical code paths at sf1 returned per-repetition ratios spanning
/// 0.29x–3.91x and a spurious 13% aggregate "win". Anything below ~30% at that scale is noise on
/// this hardware. So the threshold is set where it is defensible without that measurement: above
/// sf1's 6M-row `lineitem`, so the small case takes the identical code path it always did and
/// cannot regress, and below sf10's 60M, where the work removed is unambiguously the dominant
/// term. Re-measuring on a quiet machine should replace this constant with a real crossover.
const MIN_SOURCE_ROWS: usize = 16_000_000;

/// How `BATCHER_RUNTIME_JOIN_FILTER` overrides the default behaviour.
#[derive(PartialEq, Eq)]
enum Switch {
    /// `0` — never filter. The A/B and kill-switch setting.
    Off,
    /// `force` — filter regardless of [`MIN_SOURCE_ROWS`]. **The test hook**, and it is not
    /// optional: the row gate makes this optimization inert on any input small enough to be a
    /// test fixture, so without a way to force it on, every differential and oracle test would
    /// exercise the path that does nothing and the code would ship unverified.
    Force,
    /// Unset or anything else — the shipped behaviour, gated by [`MIN_SOURCE_ROWS`].
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
pub(crate) fn plan_filters(
    plan: &RelOp,
    sources: &[Vec<RecordBatch>],
    cache: &BuildCache,
) -> RuntimeFilters {
    let mut out = RuntimeFilters::new();
    let engage = match switch() {
        Switch::Off => false,
        Switch::Force => true,
        Switch::Default => worth_filtering(sources),
    };
    if engage {
        collect(plan, cache, &mut out);
    }
    out
}

/// Whether this query's inputs are large enough for runtime filtering to pay — see
/// [`MIN_SOURCE_ROWS`]. Reads the *largest* relation, not the total: one big fact table joined
/// against several small dimensions is exactly the shape that benefits, and summing would let a
/// pile of small inputs qualify a query that has no large probe side to reduce.
fn worth_filtering(sources: &[Vec<RecordBatch>]) -> bool {
    sources
        .iter()
        .map(|relation| relation.iter().map(RecordBatch::num_rows).sum::<usize>())
        .max()
        .is_some_and(|rows| rows >= MIN_SOURCE_ROWS)
}

fn collect(plan: &RelOp, cache: &BuildCache, out: &mut RuntimeFilters) {
    if let RelOp::HashJoin {
        left,
        left_keys,
        right_keys,
        join_type,
        ..
    } = plan
    {
        place_for_join(plan, left, left_keys, right_keys, *join_type, cache, out);
    }
    for child in plan.children() {
        collect(child, cache, out);
    }
}

/// Place one join's filter, if it has one to give.
fn place_for_join(
    join: &RelOp,
    probe: &RelOp,
    left_keys: &[String],
    right_keys: &[String],
    join_type: JoinType,
    cache: &BuildCache,
    out: &mut RuntimeFilters,
) {
    // `Inner`/`Semi` only — see the module note on which sides may be reduced.
    if !matches!(join_type, JoinType::Inner | JoinType::Semi) {
        return;
    }
    // A single equi-key: a composite key would need the row encoding the join's own hash table
    // already owns, and digesting one column of it would be a filter on a *projection* of the
    // key — still sound, but far weaker, and not worth a second encoding path here.
    let ([probe_key], [build_key]) = (left_keys, right_keys) else {
        return;
    };
    let Some(prepared) = cache.get(&node_key(join)) else {
        return;
    };
    let Ok(build_col) = ops::columns_by_name(&prepared.side, std::slice::from_ref(build_key))
    else {
        return;
    };
    let Some(filter) = build_col.first().and_then(KeyFilter::build) else {
        return;
    };
    let (target, column) = sink_target(probe, probe_key);
    out.entry(node_key(target))
        .or_default()
        .push(PendingFilter {
            column,
            filter: Arc::new(filter),
            gauge: Gauge::default(),
            force: switch() == Switch::Force,
        });
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
pub(crate) fn apply(
    filters: &[PendingFilter],
    batch: RecordBatch,
) -> Result<RecordBatch, InterpError> {
    let mut out = batch;
    for pending in filters {
        if (!pending.force && !pending.gauge.enabled()) || out.num_rows() == 0 {
            continue;
        }
        let Some(col) = out.column_by_name(&pending.column) else {
            continue;
        };
        let Some(mask) = pending.filter.mask(col) else {
            continue;
        };
        let before = out.num_rows() as u64;
        let kept = mask.values().count_set_bits() as u64;
        pending.gauge.record(before, kept);
        // Deciding *per morsel* whether to act on the mask is what bounds this optimization's
        // downside. Computing the mask is a lookup per row into an L2-resident table and
        // allocates one bit per row; acting on it means `filter_record_batch`, a copy of every
        // column of every surviving row — and at the scan, where the filter is placed, the batch
        // is at its widest, because projection pushdown narrows it above rather than below. So a
        // mask that keeps most rows would replace a zero-copy morsel slice with a near-full
        // materialization of the relation, to remove almost nothing.
        //
        // Skipping the copy is always legal: the filter only ever removes rows that provably
        // cannot match, so *not* removing them leaves the same relation, merely larger. That
        // asymmetry — free to decline, expensive to act — is why the gate is here and not only
        // in the [`Gauge`]. The gauge still sees this morsel's outcome (recorded above), so a
        // filter that keeps declining is switched off for good rather than re-masking forever.
        if !pending.force && kept * 2 > before {
            continue;
        }
        out = arrow::compute::filter_record_batch(&out, &mask)?;
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
}
