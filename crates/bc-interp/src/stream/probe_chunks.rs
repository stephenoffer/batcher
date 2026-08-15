//! Emitting one probed morsel as however many output morsels its fan-out needs.
//!
//! A join *multiplies* rows, which makes it the one pipeline operator for which "one morsel in,
//! one morsel out" is false. Against a build side holding `f` rows per key, a 16,384-row probe
//! morsel yields `16,384 x f` output rows, and emitting that as a single `RecordBatch` made the
//! streaming executor's peak memory scale with the *product* of the inputs rather than staying
//! the constant its contract promises. Measured before this existed: **13.1 GB RSS** for a
//! cartesian join over two 20,000-row tables, from roughly 500 KB of input.
//!
//! It was never a cartesian-only problem. An ordinary equi-join on a skewed key does the same:
//! a build side holding 100,000 duplicates of one key turns each probe row carrying it into
//! 100,000 output rows in one batch. `ops::remorselize`'s doc comment names this hazard for
//! unnest and unpivot; a join multiplies rows harder than either.
//!
//! The subtlety is the *common* case, which must not pay for the rare one. When a probe morsel's
//! whole result fits in one output morsel — every 1:1 foreign-key join, which is most joins —
//! the gather runs against the **unsliced** indices, because `gather_join_output_with` recognizes
//! an identity permutation and replaces a full column copy with an `Arc::clone`. Slicing would
//! defeat that even when the slice covers everything, so that path stays byte-for-byte what it
//! was and only a genuinely fanned-out morsel is chunked.

use std::sync::Arc;

use arrow::array::RecordBatch;

use crate::ops;
use crate::InterpError;

/// Rows to probe on the very first call, before any fan-out has been observed.
///
/// Small on purpose: it exists to *measure*, and until it has, a fanned-out morsel builds index
/// buffers proportional to its fan-out. 256 rows against a 20,000-row single-key build side is
/// 5.1M index entries (41 MB) instead of the 327M (2.6 GB) a full 16,384-row morsel would build.
/// The cost when fan-out turns out to be 1 — the usual answer — is one extra probe call per
/// operator, after which slices go straight to the full morsel.
const INITIAL_PROBE_ROWS: usize = 256;

/// Floor on the adaptive slice, so an enormous fan-out cannot drive the engine to one probe call
/// per row. At a fan-out of 20,000 this still holds the index buffers to ~1.3M entries.
///
/// This floor is affordable **only because the probe's output is re-morselized afterwards**
/// ([`PendingProbe`]): what it lets overshoot is the `JoinIndices`, two `u32` arrays at 8 bytes
/// per output row, and the gathered batch is bounded separately. A caller whose emitted batch is
/// *final* carries every output column instead, so the same floor would let a 1,000x fan-out emit
/// a 64,000-row wide batch — see [`ProbeSlicer::for_final_output`].
const MIN_PROBE_SLICE: usize = 64;

/// Floor for a caller that emits what the kernel returned, with no re-morselization behind it.
///
/// One row is the true floor there: a single input row's output is indivisible by slicing, so
/// nothing smaller is expressible, and the per-call cost is proportional to the output the call
/// produces rather than fixed.
const MIN_FINAL_SLICE: usize = 1;

// The opening slice must be smaller than a morsel, or the first probe of a fanned-out join is
// the unbounded one this exists to prevent. A `const` assertion so raising the constant fails
// the build rather than a test.
const _: () = assert!(INITIAL_PROBE_ROWS < bc_arrow::DEFAULT_MORSEL_ROWS);
const _: () = assert!(MIN_PROBE_SLICE <= INITIAL_PROBE_ROWS);

/// How many probe rows to take at a time, so the resulting index buffers stay morsel-scale.
///
/// Output morselization (`PendingProbe`) bounds the *gathered batch*, which is the larger term
/// because a gathered row carries every output column. It does not bound the `JoinIndices`
/// themselves: those are two `u32` arrays over the whole probe result, 8 bytes per output row, so
/// a fanned-out morsel still builds them all before the first chunk is emitted. Slicing the probe
/// on the way *in* is what bounds those, and the slice size is a measurement rather than a guess:
/// the previous probe's `output rows / input rows` is the fan-out this one is sized against.
///
/// This is Daft's dynamic-batching idea (`daft-local-execution/src/dynamic_batching`) pointed at a
/// join's fan-out rather than a scan's row width. A stale estimate is harmless — it only makes a
/// slice the wrong size, never the result wrong.
#[derive(Debug)]
pub(crate) struct ProbeSlicer {
    rows: usize,
    /// Smallest slice this caller may be driven to. It is a property of *the caller*, not of the
    /// measurement, because it encodes what the overshoot past it costs — see [`MIN_PROBE_SLICE`]
    /// against [`MIN_FINAL_SLICE`].
    min_rows: usize,
}

impl ProbeSlicer {
    /// For a join: the probe's output is re-morselized by [`PendingProbe`], so the slice may stop
    /// shrinking at [`MIN_PROBE_SLICE`] and let the *index* arrays overshoot by a bounded factor.
    pub(crate) fn new() -> Self {
        Self {
            rows: INITIAL_PROBE_ROWS,
            min_rows: MIN_PROBE_SLICE,
        }
    }

    /// For a caller that emits the kernel's batch as-is (`super::fanout`: unnest, unpivot).
    ///
    /// Nothing re-morselizes behind it, so the emitted batch is exactly `slice x fanout` rows of
    /// every output column and the join's 64-row floor would be the whole bug this slicing exists
    /// to prevent: at a fan-out of 1,000 it emits 64,000 wide rows per call.
    pub(crate) fn for_final_output() -> Self {
        Self {
            rows: INITIAL_PROBE_ROWS,
            min_rows: MIN_FINAL_SLICE,
        }
    }

    /// Rows to take for the next probe.
    pub(crate) fn slice_rows(&self) -> usize {
        self.rows
    }

    /// Record what a probe of `rows_in` rows actually produced, and resize accordingly.
    pub(crate) fn observe(&mut self, rows_in: usize, rows_out: usize) {
        if rows_in == 0 {
            return;
        }
        // Round the fan-out up, so a fractional one (a semi-join, or an inner join that drops
        // rows) never inflates the slice past what the target allows.
        let fanout = rows_out.div_ceil(rows_in).max(1);
        self.rows = (bc_arrow::DEFAULT_MORSEL_ROWS / fanout)
            .clamp(self.min_rows, bc_arrow::DEFAULT_MORSEL_ROWS);
    }
}

/// One probed morsel, emitted as however many output morsels its fan-out needs.
pub(crate) struct PendingProbe {
    morsel: RecordBatch,
    idx: bc_runtime::join::JoinIndices,
    /// Next output row to emit. `None` once the morsel is drained.
    offset: Option<usize>,
    /// Probe-side rows, reported with the first chunk only.
    rows_in: u64,
    /// Probe time, likewise attributed once.
    elapsed_ns: u64,
}

impl PendingProbe {
    pub(crate) fn new(
        morsel: RecordBatch,
        idx: bc_runtime::join::JoinIndices,
        elapsed_ns: u64,
    ) -> Self {
        let rows_in = morsel.num_rows() as u64;
        Self {
            morsel,
            idx,
            offset: Some(0),
            rows_in,
            elapsed_ns,
        }
    }

    pub(crate) fn take_elapsed(&mut self) -> u64 {
        std::mem::take(&mut self.elapsed_ns)
    }

    /// The next output morsel and the probe rows to attribute to it, or `None` when drained.
    pub(crate) fn next_chunk(
        &mut self,
        side: &RecordBatch,
        output: &[bc_ir::JoinOutputCol],
        schema: &Arc<arrow::datatypes::Schema>,
    ) -> Option<Result<(RecordBatch, u64), InterpError>> {
        let start = self.offset?;
        let total = self.idx.left.len();
        // A zero-row probe result still emits one empty morsel, because downstream operators and
        // the metrics contract expect one output per input morsel when nothing matched.
        if start >= total && start > 0 {
            self.offset = None;
            return None;
        }
        let len = (total - start).min(bc_arrow::DEFAULT_MORSEL_ROWS);
        let out = if start == 0 && len == total {
            // The whole result fits one morsel — the overwhelmingly common case, including every
            // 1:1 foreign-key join. Gather from the UNSLICED indices so
            // `gather_join_output_with`'s identity-permutation fast path (which replaces a full
            // column copy with an `Arc::clone`) still recognizes them. Slicing would defeat it
            // even when the slice covers everything, so this path stays byte-for-byte what it was.
            self.offset = None;
            ops::gather_join_output_with(&self.morsel, side, &self.idx, output, Arc::clone(schema))
        } else {
            let chunk = bc_runtime::join::JoinIndices {
                left: self.idx.left.slice(start, len),
                right: self.idx.right.slice(start, len),
            };
            self.offset = Some(start + len);
            ops::gather_join_output_with(&self.morsel, side, &chunk, output, Arc::clone(schema))
        };
        let rows_in = std::mem::take(&mut self.rows_in);
        Some(out.map(|b| (b, rows_in)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Before anything has been probed the slice is deliberately small: it exists to measure,
    /// and a full morsel against an unknown fan-out is exactly the case that blows up.
    #[test]
    fn starts_small_enough_to_measure_safely() {
        assert_eq!(ProbeSlicer::new().slice_rows(), INITIAL_PROBE_ROWS);
    }

    /// The common case: one output row per input row, so the slice opens up to a whole morsel
    /// after the first probe and the mechanism costs one extra probe call per operator, once.
    #[test]
    fn a_one_to_one_join_goes_straight_to_a_full_morsel() {
        let mut s = ProbeSlicer::new();
        s.observe(INITIAL_PROBE_ROWS, INITIAL_PROBE_ROWS);
        assert_eq!(s.slice_rows(), bc_arrow::DEFAULT_MORSEL_ROWS);
    }

    /// A join that drops rows must not be read as licence to take a *bigger* slice than a
    /// morsel — the slice bounds the input, and the output is already bounded elsewhere.
    #[test]
    fn a_row_dropping_join_does_not_exceed_a_morsel() {
        let mut s = ProbeSlicer::new();
        s.observe(1_000, 1); // a semi-join keeping almost nothing
        assert_eq!(s.slice_rows(), bc_arrow::DEFAULT_MORSEL_ROWS);
        s.observe(1_000, 0); // and one keeping nothing at all
        assert_eq!(s.slice_rows(), bc_arrow::DEFAULT_MORSEL_ROWS);
    }

    /// The case this exists for: the slice shrinks in proportion to the fan-out, so the index
    /// buffers stay morsel-scale however much the join multiplies — until the floor, past which
    /// the overshoot is deliberate and bounded (see [`MIN_PROBE_SLICE`]).
    #[test]
    fn the_slice_shrinks_in_proportion_to_fanout() {
        for fanout in [2_usize, 8, 64, 256] {
            let mut s = ProbeSlicer::new();
            s.observe(100, 100 * fanout);
            assert_eq!(
                s.slice_rows(),
                bc_arrow::DEFAULT_MORSEL_ROWS / fanout,
                "fanout {fanout} is above the floor, so the slice is exactly proportional"
            );
            // The point of the whole exercise: rows produced per probe stays near one morsel.
            assert!(s.slice_rows() * fanout <= bc_arrow::DEFAULT_MORSEL_ROWS);
        }
        // Past the floor the slice stops shrinking, so output per probe exceeds one morsel by a
        // bounded factor. That is the trade the floor buys: `PendingProbe` re-morselizes the
        // output anyway, and one probe call per row would cost more than it saves.
        let mut s = ProbeSlicer::new();
        s.observe(100, 100 * 512);
        assert_eq!(s.slice_rows(), MIN_PROBE_SLICE);
        assert!(s.slice_rows() * 512 > bc_arrow::DEFAULT_MORSEL_ROWS);
    }

    /// An enormous fan-out must not drive the slice to one row per probe — the per-call
    /// overhead would then dominate. The floor caps that, at the cost of a bounded overshoot.
    #[test]
    fn an_extreme_fanout_stops_at_the_floor() {
        let mut s = ProbeSlicer::new();
        s.observe(1, 10_000_000);
        assert_eq!(s.slice_rows(), MIN_PROBE_SLICE);
    }

    /// The estimate tracks the data rather than latching, so a join whose fan-out varies across
    /// morsels is sized by what it is doing now, not by what it did first.
    #[test]
    fn the_estimate_follows_the_data_in_both_directions() {
        let mut s = ProbeSlicer::new();
        s.observe(100, 100 * 64);
        let tight = s.slice_rows();
        s.observe(100, 100);
        assert_eq!(s.slice_rows(), bc_arrow::DEFAULT_MORSEL_ROWS);
        assert!(tight < bc_arrow::DEFAULT_MORSEL_ROWS);
    }

    /// An empty probe teaches nothing and must not reset the estimate to the default.
    #[test]
    fn an_empty_probe_leaves_the_estimate_alone() {
        let mut s = ProbeSlicer::new();
        s.observe(100, 100 * 8);
        let before = s.slice_rows();
        s.observe(0, 0);
        assert_eq!(s.slice_rows(), before);
    }
}
