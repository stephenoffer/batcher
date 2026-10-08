//! Which side of a join to build on, once both sides' true sizes are known.
//!
//! The planner picks the build side from estimates; by the time these run both relations are
//! materialized, so their sizes are facts. Each rule here is a pure relabeling of the same
//! relation — the sides exchange and the output projection follows them — and each says which
//! join flavors it is a relabeling *for*. Split out of `join_par` along that seam: the grace,
//! broadcast and skew strategies beside it never decide orientation.

use arrow::array::RecordBatch;

use crate::error::InterpError;

/// How many times larger the planner's build side must be than its probe side before the
/// executor overrides the choice.
///
/// A swap is free — both relations are already in hand, so it rebinds two slices and
/// costs nothing to perform — but it is not free to *get wrong*, because the two sides
/// are not interchangeable downstream: the emitted row order follows the probe. So the
/// bar is not "is the left smaller" (which would churn the orientation on every near-tie
/// and on every estimate that was right to within a rounding error) but "is the planner's
/// build side big enough that building it is the thing that hurts".
///
/// 2× is where the asymmetry starts to bite rather than where it becomes dramatic. Below
/// it the two orientations cost within noise of each other and the planner's choice
/// carries information the executor does not have (it saw the *whole* plan, including
/// which side a downstream operator would rather have in order). Above it the build side
/// is paying for a hash table, a chain array and a null mask over rows that would have
/// been a streaming probe.
const SWAP_MIN_RATIO: usize = 2;

/// Build sides below this are never worth correcting.
///
/// One morsel is the granularity at which every other decision in the executor stops
/// caring, and a build table over fewer rows than that is a rounding error against the
/// probe regardless of the ratio: a 3-row probe against a 30-row build is a 10× ratio and
/// a 0 ms difference. Without this floor the rule would fire constantly on the small
/// dimension-table joins where it can only change the output order.
const SWAP_MIN_BUILD_ROWS: usize = bc_arrow::DEFAULT_MORSEL_ROWS;

/// Whether building on the *left* input would be materially cheaper than the planner's
/// choice of the right, judged from the two inputs' true sizes.
///
/// **This is the one join decision the planner cannot make correctly and the executor
/// can.** Kyber picks the build side in `kyber/rules/selection.py` from
/// `estimated rows × estimated row width`, and by the time this runs both relations are
/// materialized, so their sizes are *facts*. Where the estimate was right this changes
/// nothing. Where it was wrong — a `LIKE` predicate, a correlated filter, a join feeding
/// a join, or simply a source with no statistics on its first run — the executor is
/// otherwise obliged to build a hash table over the larger relation, and that single
/// wrong choice decides three things at once:
///
/// * **whether the join spills.** Admission compares the *build* side against the memory
///   envelope. Building the 50 M-row side of a 50 M ⋈ 200 k join spills to disk; building
///   the 200 k-row side does not. That is not a percentage, it is a grace hash join
///   against no disk I/O at all.
/// * **whether the streaming probe is reachable.** `streaming_supported` is gated on
///   `build_rows`, so an oversized build falls through to the shuffle, which copies the
///   probe — the query's largest relation — for no reason.
/// * **the hash table's cache residency**, which is what the radix path exists to
///   protect.
///
/// Restricted to `Inner`, which is the only flavor whose swap is a pure re-labeling.
/// `Semi`/`Anti` emit *left* rows and are not symmetric under a swap at all. `Left`/`Right`
/// are semantically swappable (and [`broadcast_join`] already performs exactly that
/// rewrite in the other direction), but the engine's probe-driven fast paths decline
/// `Right` — `streaming_supported` rejects it outright — so turning a `Left` join into a
/// `Right` one to shrink the build would trade a hash table for a slower path and could
/// lose. `Full` is symmetric but drives from both sides, so there is no build to shrink.
pub(crate) fn build_side_swap_pays(
    join_type: bc_ir::JoinType,
    probe_rows: usize,
    build_rows: usize,
) -> bool {
    matches!(join_type, bc_ir::JoinType::Inner)
        && build_rows >= SWAP_MIN_BUILD_ROWS
        && build_rows >= probe_rows.saturating_mul(SWAP_MIN_RATIO)
}

/// Whether to build on the *left* input of a join whose output order nothing can observe.
///
/// [`build_side_swap_pays`] confines itself to `Inner` because elsewhere a swap is not a pure
/// relabeling *of the row order*: a `Full` join emits probe-major and then the build rows
/// nothing matched, so exchanging the sides emits the same rows in a different sequence.
/// Where the consumer cannot see the sequence — a `COUNT(*)` over the join, a grouped aggregate
/// (`stream::order`) — that objection is gone, and `Full` is exactly as symmetric as `Inner`:
/// every matched pair, plus every unmatched row of *either* side null-extended, whichever side
/// the table is built on. What the build side still decides is everything that costs: the
/// table's size, its cache residency, and the unmatched-row pass, which walks the build.
///
/// TPC-H `orders FULL JOIN lineitem` is the shape — the planner built the 3M-row `lineitem`
/// side and probed it with 500K `orders` rows, and building the smaller side instead measured
/// 72.6 ms against 61.6 ms on 16 cores. `Left`/`Right`/`Semi`/`Anti` stay excluded for the
/// reasons [`build_side_swap_pays`] gives. Same thresholds, so the two rules cannot disagree
/// about what "materially larger" means.
pub(crate) fn order_free_swap_pays(
    join_type: bc_ir::JoinType,
    probe_rows: usize,
    build_rows: usize,
) -> bool {
    matches!(join_type, bc_ir::JoinType::Inner | bc_ir::JoinType::Full)
        && build_rows >= SWAP_MIN_BUILD_ROWS
        && build_rows >= probe_rows.saturating_mul(SWAP_MIN_RATIO)
}

/// The join's output projection with each column's side re-labeled for swapped inputs.
///
/// The output columns name a side and a column *on* that side, so exchanging the two
/// relations is expressible entirely here: nothing about the column, its name, or its
/// alias changes, only which input it is read from. This is what makes a build-side swap
/// a re-labeling rather than a rewrite.
pub(crate) fn flip_output(output: &[bc_ir::JoinOutputCol]) -> Vec<bc_ir::JoinOutputCol> {
    use bc_ir::JoinSide;
    output
        .iter()
        .map(|o| bc_ir::JoinOutputCol {
            side: match o.side {
                JoinSide::Left => JoinSide::Right,
                JoinSide::Right => JoinSide::Left,
            },
            name: o.name.clone(),
            alias: o.alias.clone(),
        })
        .collect()
}

/// Build sides up to this many rows are joined by one shared table and a chunked probe
/// ([`crate::join_par::broadcast_join`]) rather than by hash-partitioning both sides.
///
/// Partitioning copies every column of *both* relations into per-core buckets before a single
/// key is compared, so its cost is the probe side's size, while what it buys -- a table per
/// bucket that fits in cache -- only pays once the build side no longer does. A `Full` join
/// lands here because the per-morsel streaming probe declines it: TPC-H `lineitem` (3M rows,
/// even keys) `FULL JOIN orders` (500K) partitioned for 19 ms of a 54 ms query on 16 cores,
/// and the shared table removed it (op-join-full-outer 53.6 -> 47.4 ms). A million rows of
/// build keeps the table to roughly L3 size on the hosts measured; above it the partitioned
/// join is unchanged.
const ORDER_FREE_BROADCAST_MAX_BUILD_ROWS: usize = 1 << 20;

/// [`crate::par::join_partitioned`] over two materialized sides whose join order nothing above
/// can observe, built on whichever side [`order_free_swap_pays`] picks, as one batch (empty
/// when the join yields no batch at all).
///
/// The output projection is flipped with the sides, so every column is still read from the
/// relation it names; only the row order — which the caller has established nobody sees —
/// can differ from the unswapped join.
#[allow(clippy::too_many_arguments)]
pub(crate) fn join_partitioned_order_free(
    probe: RecordBatch,
    build: RecordBatch,
    probe_keys: &[String],
    build_keys: &[String],
    join_type: bc_ir::JoinType,
    output: &[bc_ir::JoinOutputCol],
    strategy: bc_ir::JoinStrategy,
) -> Result<Vec<RecordBatch>, InterpError> {
    let flipped: Vec<bc_ir::JoinOutputCol>;
    let (probe, build, probe_keys, build_keys, output) =
        if order_free_swap_pays(join_type, probe.num_rows(), build.num_rows()) {
            flipped = flip_output(output);
            (build, probe, build_keys, probe_keys, flipped.as_slice())
        } else {
            (probe, build, probe_keys, build_keys, output)
        };
    if build.num_rows() <= ORDER_FREE_BROADCAST_MAX_BUILD_ROWS
        && !matches!(strategy, bc_ir::JoinStrategy::SortMerge)
    {
        return crate::join_par::broadcast_join(
            &probe, &build, probe_keys, build_keys, join_type, output,
        );
    }
    let (parts, _) = crate::par::join_partitioned(
        std::slice::from_ref(&probe),
        std::slice::from_ref(&build),
        probe_keys,
        build_keys,
        join_type,
        output,
        strategy,
        &crate::ExecOptions::default(),
    )?;
    Ok(parts)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The build-side correction fires only where it can pay: a materially oversized
    /// build, on the one join flavor whose swap is a re-labeling.
    ///
    /// Each `assert!(!...)` here is a case the rule must decline, and each names the gate
    /// that declines it — a rule that fired on all of them would still pass a test that
    /// only checked the positive case.
    #[test]
    fn build_side_correction_fires_only_on_a_materially_oversized_inner_build() {
        use bc_ir::JoinType;
        let big = SWAP_MIN_BUILD_ROWS * 4;

        // The case it exists for: the planner nominated a build 4x the probe.
        assert!(build_side_swap_pays(JoinType::Inner, big / 4, big));
        // Exactly at the ratio still pays; a hair under it does not.
        assert!(build_side_swap_pays(JoinType::Inner, big / 2, big));
        assert!(!build_side_swap_pays(JoinType::Inner, big / 2 + 1, big));
        // A build the planner sized correctly is left alone.
        assert!(!build_side_swap_pays(JoinType::Inner, big, big / 4));

        // The one-morsel floor: a 10x ratio over a trivial build is still a 0 ms
        // difference, and swapping it would only churn the output order.
        assert!(!build_side_swap_pays(
            JoinType::Inner,
            SWAP_MIN_BUILD_ROWS / 10,
            SWAP_MIN_BUILD_ROWS - 1
        ));

        // Every non-`Inner` flavor declines, however lopsided. `Semi`/`Anti` emit left
        // rows and are not symmetric; `Left`/`Right` would trade the probe-driven fast
        // paths for the `Right` ones the engine declines; `Full` drives from both sides.
        for jt in [
            JoinType::Left,
            JoinType::Right,
            JoinType::Full,
            JoinType::Semi,
            JoinType::Anti,
        ] {
            assert!(!build_side_swap_pays(jt, big / 100, big));
        }
    }

    /// The order-free rule admits `Full` beside `Inner`, on the same thresholds, and nothing
    /// whose swap would change which rows are emitted.
    #[test]
    fn the_order_free_swap_admits_full_and_inner_on_the_same_thresholds() {
        use bc_ir::JoinType;
        let big = SWAP_MIN_BUILD_ROWS * 4;
        for jt in [JoinType::Inner, JoinType::Full] {
            assert!(order_free_swap_pays(jt, big / 2, big), "{jt:?}");
            assert!(
                !order_free_swap_pays(jt, big / 2 + 1, big),
                "{jt:?} under the ratio"
            );
            assert!(
                !order_free_swap_pays(jt, big, big / 4),
                "{jt:?} already right"
            );
            assert!(
                !order_free_swap_pays(jt, SWAP_MIN_BUILD_ROWS / 10, SWAP_MIN_BUILD_ROWS - 1),
                "{jt:?} under the floor"
            );
        }
        for jt in [
            JoinType::Left,
            JoinType::Right,
            JoinType::Semi,
            JoinType::Anti,
        ] {
            assert!(!order_free_swap_pays(jt, big / 100, big), "{jt:?}");
        }
    }

    /// Flipping the output twice is the identity, which is the property that makes a
    /// build-side swap a re-labeling: only the side moves, never the column or its alias.
    #[test]
    fn flipping_the_output_sides_twice_is_the_identity() {
        use bc_ir::{JoinOutputCol, JoinSide};
        let output = vec![
            JoinOutputCol {
                side: JoinSide::Left,
                name: "a".into(),
                alias: "x".into(),
            },
            JoinOutputCol {
                side: JoinSide::Right,
                name: "b".into(),
                alias: "y".into(),
            },
        ];
        let once = flip_output(&output);
        assert!(matches!(once[0].side, JoinSide::Right));
        assert!(matches!(once[1].side, JoinSide::Left));

        let twice = flip_output(&once);
        for (before, after) in output.iter().zip(&twice) {
            assert!(matches!(
                (before.side, after.side),
                (JoinSide::Left, JoinSide::Left) | (JoinSide::Right, JoinSide::Right)
            ));
            assert_eq!(before.name, after.name);
            assert_eq!(before.alias, after.alias);
        }
    }
}
