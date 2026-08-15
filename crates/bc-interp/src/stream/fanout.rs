//! Slicing the input of a row-*multiplying* pipeline operator, so its output stays morsel-scale.
//!
//! "One morsel in, one morsel out" is the property that makes a linear run's peak memory a
//! constant, and three operators break it by construction. The join was the first and the worst,
//! and it already has its answer ([`super::probe_chunks`]): size the input slice from the measured
//! fan-out, then emit the result in chunks.
//!
//! `Unnest` and `Unpivot` multiply rows too, and had neither half. An `explode` over a column of
//! thousand-element lists turns a 16,384-row morsel into **16 million rows in one `RecordBatch`** —
//! built whole, before anything downstream sees a row of it. That is not a corner case: it is what
//! a JSON event table, an embedding column, or a per-frame detection list looks like, and
//! `ops::remorselize`'s doc comment has named unnest and unpivot as the hazard all along. It does
//! not fix it here, because it only splits when the morsel target is byte-bounded, and the default
//! target is row-only.
//!
//! So the same answer applies, and this is it — written once, for any kernel of the shape "one
//! batch in, one longer batch out". The slice is a *measurement* (`ProbeSlicer`), not a guess about
//! list lengths: an operator whose fan-out is 1 opens straight back up to a whole morsel and pays
//! one extra call, while a fan-out of a thousand shrinks the slice a thousandfold. Slicing changes
//! only where the boundaries fall — the rows, and their order, are what the unsliced kernel
//! produced, because every kernel here is per-row.

use arrow::array::RecordBatch;

use super::probe_chunks::ProbeSlicer;
use super::Morsels;
use crate::InterpError;

/// Apply `kernel` to each morsel of `child`, taking the input in slices sized so each call's
/// output stays near one morsel.
///
/// `kernel` receives a slice of an input morsel and returns that slice's output. It is called once
/// per slice rather than once per morsel, so a caller recording metrics attributes each emitted
/// batch to the rows that actually produced it.
pub(super) fn fanout_stream<'a, F>(child: Morsels<'a>, mut kernel: F) -> Morsels<'a>
where
    F: FnMut(&RecordBatch) -> Result<RecordBatch, InterpError> + 'a,
{
    // The morsel being consumed and how far into it we have got; a morsel is taken in slices, so
    // it stays alive across several calls.
    let mut current: Option<(RecordBatch, usize)> = None;
    // `for_final_output`, not `new`: what this emits is what the kernel built, with no
    // re-morselization behind it to absorb an overshoot past the slicer's floor.
    let mut slicer = ProbeSlicer::for_final_output();
    let mut source = child;
    Box::new(std::iter::from_fn(move || {
        let (morsel, offset) = match current.take() {
            Some(pair) => pair,
            None => match source.next()? {
                Ok(m) => (m, 0),
                Err(e) => return Some(Err(e)),
            },
        };
        let remaining = morsel.num_rows() - offset;
        let take = slicer.slice_rows().min(remaining);
        // A zero-row morsel has nothing to slice and still has to reach the kernel, which is
        // what carries the schema of an empty relation downstream.
        let slice = if offset == 0 && (take == morsel.num_rows() || morsel.num_rows() == 0) {
            // The slice *is* the morsel — the steady state for any fan-out of 1, which is most
            // of them. Skip the rebuild `slice()` would cost.
            morsel.clone()
        } else {
            morsel.slice(offset, take)
        };
        if offset + take < morsel.num_rows() {
            current = Some((morsel, offset + take));
        }
        let out = match kernel(&slice) {
            Ok(out) => out,
            Err(e) => return Some(Err(e)),
        };
        slicer.observe(slice.num_rows(), out.num_rows());
        Some(Ok(out))
    }))
}

#[cfg(test)]
mod fanout_tests {
    use super::*;
    use arrow::array::{ArrayRef, Int64Array};
    use std::sync::Arc;

    fn batch(vals: Vec<i64>) -> RecordBatch {
        let col: ArrayRef = Arc::new(Int64Array::from(vals));
        RecordBatch::try_from_iter(vec![("v", col)]).unwrap()
    }

    fn values(batches: &[RecordBatch]) -> Vec<i64> {
        batches
            .iter()
            .flat_map(|b| {
                b.column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect()
    }

    /// A kernel that repeats every row `f` times — the shape of an unnest over fixed-length lists.
    fn repeat(b: &RecordBatch, f: usize) -> Result<RecordBatch, InterpError> {
        let col = b
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values()
            .iter()
            .flat_map(|v| std::iter::repeat_n(*v, f))
            .collect::<Vec<_>>();
        Ok(batch(col))
    }

    /// The rows, and their order, are exactly what the unsliced kernel produces. Slicing moves the
    /// batch boundaries and nothing else.
    #[test]
    fn slicing_preserves_the_rows_and_their_order() {
        let input = batch((0..40_000i64).collect());
        let out: Vec<RecordBatch> =
            fanout_stream(Box::new(std::iter::once(Ok(input.clone()))), |b| {
                repeat(b, 8)
            })
            .collect::<Result<_, _>>()
            .unwrap();
        let expected = values(&[repeat(&input, 8).unwrap()]);
        assert_eq!(values(&out), expected);
    }

    /// The point of it: once the fan-out has been measured, no emitted batch is much larger than
    /// a morsel however much the kernel multiplies.
    ///
    /// The **first** batch is the exception, and a deliberate one. The opening slice exists to
    /// measure, and until it has, its output is `INITIAL_PROBE_ROWS x fanout` — 64x smaller than
    /// the unsliced morsel would have produced, and paid once per operator rather than once per
    /// morsel. The join makes the same trade at its `MIN_PROBE_SLICE` floor and for the same
    /// reason: driving the slice small enough to bound the *unmeasured* case would cost every
    /// operator that does not fan out an extra call per morsel forever.
    #[test]
    fn no_emitted_batch_is_much_larger_than_a_morsel_once_the_fanout_is_known() {
        let input = batch((0..40_000i64).collect());
        let out: Vec<RecordBatch> =
            fanout_stream(Box::new(std::iter::once(Ok(input))), |b| repeat(b, 1_000))
                .collect::<Result<_, _>>()
                .unwrap();
        let biggest_after_first = out[1..].iter().map(|b| b.num_rows()).max().unwrap();
        assert!(
            biggest_after_first <= 2 * bc_arrow::DEFAULT_MORSEL_ROWS,
            "a 1000x fan-out emitted a {biggest_after_first}-row batch after measuring; the \
             slice is not tracking it"
        );
        // The measuring batch is bounded too, and by far less than the whole morsel's output
        // (40,000,000 rows) that the unsliced operator produced.
        assert!(out[0].num_rows() <= 256 * 1_000);
        assert_eq!(values(&out).len(), 40_000 * 1_000);
    }

    /// A kernel that does not multiply must not be charged for the mechanism: after one measuring
    /// call the slice is a whole morsel, so a many-morsel stream emits one batch per morsel plus
    /// the one the measurement cost.
    #[test]
    fn a_one_to_one_kernel_pays_a_single_extra_call() {
        let morsels: Vec<Result<RecordBatch, InterpError>> = (0..4)
            .map(|i| {
                Ok(batch(
                    (0..bc_arrow::DEFAULT_MORSEL_ROWS as i64)
                        .map(|v| v + i * 1000)
                        .collect(),
                ))
            })
            .collect();
        let out: Vec<RecordBatch> = fanout_stream(Box::new(morsels.into_iter()), |b| repeat(b, 1))
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(out.len(), 5, "four morsels plus the one measuring slice");
        assert_eq!(values(&out).len(), 4 * bc_arrow::DEFAULT_MORSEL_ROWS);
    }

    /// An empty morsel still reaches the kernel, because its schema is what an empty relation
    /// carries downstream.
    #[test]
    fn an_empty_morsel_still_reaches_the_kernel() {
        let empty = batch(Vec::new());
        let out: Vec<RecordBatch> =
            fanout_stream(Box::new(std::iter::once(Ok(empty))), |b| repeat(b, 4))
                .collect::<Result<_, _>>()
                .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].num_rows(), 0);
    }
}
