//! The partial state of an integer `SUM`: an exact 128-bit total, range-checked only when
//! it is finalized.
//!
//! **Why the state is wider than the result.** An `Int64` `SUM` returns `Int64`, and it used
//! to carry `Int64` between `partial`, `combine` and `finalize` too. That made overflow a
//! property of how the rows were *split*: each partition narrowed its own total to `i64`
//! before anything merged it, so `{i64::MAX, 1}` in one partition and `{-2}` in another --
//! a relation whose true total, `i64::MAX - 1`, fits -- raised `SumOverflow` whenever the
//! first two rows landed together (morsel-parallel, spilled, or distributed) and succeeded
//! single-node. Partition count is a scheduling decision, so the same query's success
//! depended on the machine it ran on.
//!
//! Now no partial is ever narrowed past what it holds. A partial whose own total fits an
//! `i64` -- almost every partial -- keeps the bare `Int64` state it always had; one whose
//! total does not carries the exact `i128` total instead, in the marked form below. `combine`
//! adds bare states with checked `i64` adds and promotes to `i128` at the first overflow
//! rather than raising, widens bare states only when they meet marked ones, and `finalize` is
//! the one place that narrows and checks. Overflow is therefore decided on the **true**
//! total, which is a property of the data alone, and identically on every path.
//! `i128` cannot itself overflow here: `2^64` rows of `i64::MAX` is below `2^127`.
//!
//! **Why a struct, and not a bare `Decimal128(38, 0)`.** `finalize` sees only the state, and
//! a user's `DECIMAL(38, 0)` `SUM` carries a state of exactly that type, which must finalize
//! to itself. Telling the two apart by guessing from the Arrow type would silently turn one
//! into the other. The state is instead a one-field `Struct` whose field name,
//! [`STATE_FIELD`], *is* the marker: no other aggregate produces it, and it survives every
//! place a partial travels -- `concat`, the radix gather, the spill IPC files and the
//! distributed partial batches -- because it is part of the column's `DataType`.
//!
//! **Where the cost lands: only where a total overflows.** The first version widened every
//! partial to the 128-bit state, once per group. That measured +12% on a 20M-row `SUM` into
//! 1k groups and +4% into 2M groups: twice the state bytes through `combine`, and an `i128`
//! scatter where an `i64` one had been. The common case now pays nothing it did not pay
//! before -- the bare state, the `i64` scatter, the identity `finalize` -- and the 128-bit
//! form is built only by a partial or a merge whose own total leaves `i64`.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, Decimal128Array, Int64Array, StructArray};
use arrow::datatypes::{DataType, Decimal128Type, Field, Fields, Int64Type};

use super::accum::{masked_decimal, masked_i64, require, sum_acc};
use super::{accumulate, AggCall, AggFunc};
use crate::error::RuntimeError;

/// The field name that marks a column as an integer `SUM`'s 128-bit partial state.
pub(crate) const STATE_FIELD: &str = "__bc_int64_sum_i128";

/// The exact accumulator type inside the state: `Decimal128(38, 0)` *is* an `i128`, and 38
/// digits holds any `i64` sum (the widest possible is `2^63 * 2^64 < 10^38`).
const INNER: DataType = DataType::Decimal128(38, 0);

fn fields() -> Fields {
    Fields::from(vec![Field::new(STATE_FIELD, INNER, true)])
}

/// Whether `dt` is the integer-`SUM` partial state. Exact: the one-field struct named
/// [`STATE_FIELD`], never a guess from a decimal's precision.
#[must_use]
pub(crate) fn is_state(dt: &DataType) -> bool {
    matches!(dt, DataType::Struct(f) if f.len() == 1 && f[0].name() == STATE_FIELD)
}

/// Wrap an exact per-group total (`Decimal128(38, 0)`, null for an empty or all-null group)
/// in the marked state struct. The struct itself carries no validity: a SQL-NULL sum is a
/// null *child*, so there is exactly one place a reader has to look.
fn wrap(inner: ArrayRef) -> ArrayRef {
    Arc::new(StructArray::new(fields(), vec![inner], None))
}

fn wrap_sums(sums: Vec<i128>, valid: Vec<bool>) -> Result<ArrayRef, RuntimeError> {
    let DataType::Decimal128(p, s) = INNER else {
        unreachable!("INNER is a Decimal128 by construction")
    };
    Ok(wrap(masked_decimal(sums, valid, p, s)?))
}

/// The exact `i128` totals inside a state column.
fn inner(state: &ArrayRef) -> Result<&Decimal128Array, RuntimeError> {
    let s = state
        .as_any()
        .downcast_ref::<StructArray>()
        .filter(|_| is_state(state.data_type()))
        .ok_or_else(|| RuntimeError::UnsupportedAggregate {
            func: "sum".to_string(),
            dtype: state.data_type().to_string(),
        })?;
    Ok(s.column(0).as_primitive::<Decimal128Type>())
}

/// Widen a finished `i64` state (one value per group, null for an empty group) to the
/// marked 128-bit state. One conversion per group, not per row.
pub(crate) fn from_i64(sums: &[i64], valid: Vec<bool>) -> Result<ArrayRef, RuntimeError> {
    wrap_sums(super::promote_wide(sums), valid)
}

/// The state for an `i128` accumulator the fused scan promoted mid-partial: the bare `Int64`
/// when every group's total fits after all (the running total overflowed, the partial did
/// not), else the marked 128-bit form, exactly.
pub(crate) fn from_i128(sums: Vec<i128>, valid: Vec<bool>) -> Result<ArrayRef, RuntimeError> {
    match sums
        .iter()
        .map(|&s| i64::try_from(s))
        .collect::<Result<Vec<i64>, _>>()
    {
        Ok(narrow) => Ok(Arc::new(masked_i64(narrow, valid))),
        Err(_) => wrap_sums(sums, valid),
    }
}

/// Whether `call` is an `Int64` `SUM`, whose partial state this module builds.
///
/// Checked in `accum::accumulate_call`, the per-call kernel `partial` runs, and **not** in
/// `accumulate`: `combine` also uses `accumulate(Sum, ..)` as the plain additive reducer for
/// counts and for `AVG`'s halves, and there it must keep returning what it was given.
pub(crate) fn takes(call: &AggCall) -> bool {
    call.func == AggFunc::Sum
        && matches!(
            call.values.as_ref().map(|v| v.data_type()),
            Some(DataType::Int64)
        )
}

/// [`state`] for a call [`takes`] claimed.
pub(crate) fn call_state(
    call: &AggCall,
    group_ids: &[u32],
    num_groups: usize,
) -> Result<Vec<ArrayRef>, RuntimeError> {
    let values = require(call.values.as_ref(), call.func)?;
    Ok(vec![state(values, group_ids, num_groups)?])
}

/// The integer `SUM` partial state for an `Int64` column.
///
/// Runs the ordinary `i64` kernel (`accum::sum_acc`, with its no-null fast path and its
/// in-kernel wide retry) and returns its bare `Int64` answer when that fits. The kernel
/// reports `SumOverflow` when *this partition's* true total exceeds `i64` -- which is not an
/// error for a partial, only for the finished aggregate. Then, and only then, the column is
/// summed once more straight into `i128` and returned in the marked form. The extra pass is
/// paid only by a partition whose own total does not fit an `i64`.
///
/// `combine` reuses this as the reducer over bare partial totals (`group::merge_state`), which
/// is what makes a merge promote rather than raise when two fitting partials overflow together.
pub(crate) fn state(
    values: &ArrayRef,
    group_ids: &[u32],
    num_groups: usize,
) -> Result<ArrayRef, RuntimeError> {
    match sum_acc(values, group_ids, num_groups, AggFunc::Sum) {
        Ok(narrow) => Ok(narrow),
        Err(RuntimeError::SumOverflow) => {
            let (sums, valid) = exact_sums(values.as_primitive(), group_ids, num_groups);
            wrap_sums(sums, valid)
        }
        Err(e) => Err(e),
    }
}

/// One exact `i128` pass. With a single group the ids are not read (the whole-column
/// convention `accum::global_reduces_whole_column` documents: they may be empty).
fn exact_sums(arr: &Int64Array, group_ids: &[u32], num_groups: usize) -> (Vec<i128>, Vec<bool>) {
    let mut sums = vec![0i128; num_groups];
    let mut valid = vec![false; num_groups];
    let group_of = |i: usize| {
        if num_groups == 1 {
            0
        } else {
            group_ids[i] as usize
        }
    };
    for (i, v) in arr.iter().enumerate() {
        if let Some(v) = v {
            let g = group_of(i);
            sums[g] += i128::from(v);
            valid[g] = true;
        }
    }
    (sums, valid)
}

/// Whether a `SUM`'s state column is the **bare** form: an `Int64` per-group total.
///
/// Two writers produce it, and both write a valid `Int64` total, so widening it is lossless:
/// the current engine, for every partial whose own total fits (the common case), and an
/// engine from before this module, whose persisted partials -- a streaming checkpoint's
/// running state, a changelog delta, a shuffle file -- still reach `combine`. It is
/// unambiguous rather than a guess: a `SUM` state is otherwise `Float64`, a user `Decimal128`,
/// or the marked struct, and a bare `Int64` state only ever comes from an `Int64` input.
fn is_bare(func: AggFunc, state: &[ArrayRef]) -> bool {
    func == AggFunc::Sum
        && state
            .first()
            .is_some_and(|c| c.data_type() == &DataType::Int64)
}

fn widen_state(s: &[ArrayRef]) -> Result<Vec<ArrayRef>, RuntimeError> {
    let a = s[0].as_primitive::<Int64Type>();
    let valid = (0..a.len()).map(|g| a.is_valid(g)).collect();
    Ok(vec![from_i64(a.values(), valid)?])
}

/// Widen every bare `Int64` `SUM` state in `p` to the marked 128-bit state, leaving every
/// other state untouched. `None` when `p` holds none, so the common path copies nothing.
///
/// For every place partials are pooled under one schema and cannot know whether a later one
/// will be marked: the spill path's shared IPC files, and the distributed wire batch
/// (`bc_interp::dist`), which Python pools into shuffle files and checkpoints. It widens
/// unconditionally, once per group, on paths already bound by the disk or the network.
/// In-memory `combine` uses [`upgrade_all`] instead.
pub fn widen_bare(
    p: &super::Partial,
    funcs: &[AggFunc],
) -> Result<Option<super::Partial>, RuntimeError> {
    if !funcs.iter().zip(&p.states).any(|(&f, s)| is_bare(f, s)) {
        return Ok(None);
    }
    let states = funcs
        .iter()
        .zip(&p.states)
        .map(|(&f, s)| {
            if is_bare(f, s) {
                widen_state(s)
            } else {
                Ok(s.clone())
            }
        })
        .collect::<Result<_, RuntimeError>>()?;
    Ok(Some(super::Partial {
        group_columns: p.group_columns.clone(),
        states,
    }))
}

/// Make each `SUM`'s state one type across `parts` so they concatenate: widen the bare
/// states of an aggregate **only when** some other partial holds it marked. `None` -- nothing
/// copied -- in the common case, where every partial of an aggregate is bare (or every one
/// is marked), and `merge_state` then reduces them in their own form.
pub(crate) fn upgrade_all(
    parts: &[super::Partial],
    funcs: &[AggFunc],
) -> Result<Option<Vec<super::Partial>>, RuntimeError> {
    let mixed: Vec<bool> = funcs
        .iter()
        .enumerate()
        .map(|(a, &f)| {
            f == AggFunc::Sum
                && parts.iter().any(|p| is_state(p.states[a][0].data_type()))
                && parts.iter().any(|p| is_bare(f, &p.states[a]))
        })
        .collect();
    if !mixed.contains(&true) {
        return Ok(None);
    }
    parts
        .iter()
        .map(|p| {
            let states = p
                .states
                .iter()
                .enumerate()
                .map(|(a, s)| {
                    if mixed[a] && is_bare(funcs[a], s) {
                        widen_state(s)
                    } else {
                        Ok(s.clone())
                    }
                })
                .collect::<Result<_, RuntimeError>>()?;
            Ok(super::Partial {
                group_columns: p.group_columns.clone(),
                states,
            })
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

/// One state column's pieces made concatenable: when some are marked and others bare `Int64`,
/// widen the bare ones. `None` when they are already one type (the common case).
///
/// Needs no `AggFunc`, because the marker is explicit: only an integer `SUM` is ever marked,
/// so a bare `Int64` beside a marked piece of the *same* state column is that `SUM`'s bare
/// form. Used where merged pieces are concatenated after the merge -- the radix partitions,
/// `concat_disjoint` -- since each piece's merge decides its own form.
pub(crate) fn unify(cols: &[&dyn Array]) -> Result<Option<Vec<ArrayRef>>, RuntimeError> {
    let marked = cols.iter().any(|c| is_state(c.data_type()));
    if !marked || !cols.iter().any(|c| c.data_type() == &DataType::Int64) {
        return Ok(None);
    }
    cols.iter()
        .map(|c| {
            let c = arrow::array::make_array(c.to_data());
            if c.data_type() == &DataType::Int64 {
                Ok(widen_state(&[c])?.remove(0))
            } else {
                Ok(c)
            }
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

/// `combine`'s reducer: add the partials' exact totals per group, keeping the marker.
pub(crate) fn merge(
    state: &ArrayRef,
    group_ids: &[u32],
    num_groups: usize,
) -> Result<ArrayRef, RuntimeError> {
    inner(state)?;
    let totals = state.as_struct().column(0);
    let merged = accumulate(AggFunc::Sum, Some(totals), group_ids, num_groups)?;
    Ok(wrap(merged.into_iter().next().expect("sum has one state")))
}

/// `finalize`: narrow each exact total to `Int64`, raising `SumOverflow` exactly when the
/// **true** total does not fit. A null total (empty or all-null group) stays null.
pub(crate) fn finalize(state: &ArrayRef) -> Result<ArrayRef, RuntimeError> {
    let totals = inner(state)?;
    let mut out = Vec::with_capacity(totals.len());
    let mut valid = Vec::with_capacity(totals.len());
    for g in 0..totals.len() {
        if totals.is_valid(g) {
            out.push(i64::try_from(totals.value(g)).map_err(|_| RuntimeError::SumOverflow)?);
            valid.push(true);
        } else {
            out.push(0);
            valid.push(false);
        }
    }
    Ok(Arc::new(masked_i64(out, valid)))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------------------------
    // The mergeable invariant, end to end: `finalize(combine(partial(p_k)))` over every
    // split, serial and radix `combine`, and the spilled merge (in memory and through the
    // IPC files), must give the single-node answer -- and must raise `SumOverflow` on
    // exactly the inputs whose TRUE total does not fit, on every one of those paths.
    // ---------------------------------------------------------------------------------
    use super::super::spill::{combine_finalize_spilling, DiskSpillStore, MemSpillStore};
    use super::super::{combine_with, finalize as agg_finalize, partial, AggCall, Partial};
    use std::collections::BTreeMap;

    /// Group key -> finished `SUM` (None = SQL NULL).
    type Answer = BTreeMap<Option<i64>, Option<i64>>;

    fn answer(groups: &[ArrayRef], sums: &ArrayRef) -> Answer {
        let s = sums.as_primitive::<Int64Type>();
        assert_eq!(
            sums.data_type(),
            &DataType::Int64,
            "the result type is Int64"
        );
        (0..s.len())
            .map(|r| {
                let k = groups.first().map(|g| {
                    let g = g.as_primitive::<Int64Type>();
                    g.is_valid(r).then(|| g.value(r))
                });
                (k.flatten(), s.is_valid(r).then(|| s.value(r)))
            })
            .collect()
    }

    fn partials_of(
        keys: Option<&ArrayRef>,
        vals: &ArrayRef,
        cuts: &[usize],
    ) -> Result<Vec<Partial>, RuntimeError> {
        let mut bounds = vec![0];
        bounds.extend_from_slice(cuts);
        bounds.push(vals.len());
        bounds
            .windows(2)
            .map(|w| {
                let (off, len) = (w[0], w[1] - w[0]);
                let k: Vec<ArrayRef> = keys.iter().map(|k| k.slice(off, len)).collect();
                let call = AggCall::new(AggFunc::Sum, Some(vals.slice(off, len)));
                partial(&k, std::slice::from_ref(&call), len)
            })
            .collect()
    }

    /// Every path's answer for one split of the input, labelled.
    fn every_path(
        keys: Option<&ArrayRef>,
        vals: &ArrayRef,
        cuts: &[usize],
    ) -> Vec<(String, Result<Answer, RuntimeError>)> {
        let funcs = [AggFunc::Sum];
        let mut out = Vec::new();
        for (name, threshold) in [("serial", usize::MAX), ("radix", 1)] {
            let r = partials_of(keys, vals, cuts).and_then(|ps| {
                let merged = combine_with(&ps, &funcs, threshold)?;
                let cols = agg_finalize(&funcs, &merged)?;
                Ok(answer(&merged.group_columns, &cols[0]))
            });
            out.push((format!("{name} {cuts:?}"), r));
        }
        let mem = partials_of(keys, vals, cuts).and_then(|ps| {
            let mut store = MemSpillStore::new(4);
            let g = combine_finalize_spilling(ps, &funcs, &mut store, 0)?;
            Ok(answer(&g.group_columns, &g.agg_columns[0]))
        });
        out.push((format!("mem-spill {cuts:?}"), mem));
        // A fresh directory per call: the tests run concurrently and share a process id.
        static RUN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let run = RUN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("bc_int_sum_spill_{}_{run}", std::process::id()));
        let disk = partials_of(keys, vals, cuts).and_then(|ps| {
            let mut store = DiskSpillStore::new(dir.clone(), 4)?;
            let g = combine_finalize_spilling(ps, &funcs, &mut store, 0)?;
            Ok(answer(&g.group_columns, &g.agg_columns[0]))
        });
        let _ = std::fs::remove_dir_all(&dir);
        out.push((format!("disk-spill {cuts:?}"), disk));
        out
    }

    /// Every contiguous split of `n` rows into two or three partitions, plus none.
    fn splits(n: usize) -> Vec<Vec<usize>> {
        let mut v = vec![vec![]];
        for a in 1..n {
            v.push(vec![a]);
            for b in a + 1..n {
                v.push(vec![a, b]);
            }
        }
        v
    }

    fn single_node(keys: Option<&ArrayRef>, vals: &ArrayRef) -> Result<Answer, RuntimeError> {
        let k: Vec<ArrayRef> = keys.into_iter().cloned().collect();
        let call = AggCall::new(AggFunc::Sum, Some(vals.clone()));
        let p = partial(&k, std::slice::from_ref(&call), vals.len())?;
        let cols = agg_finalize(&[AggFunc::Sum], &p)?;
        Ok(answer(&p.group_columns, &cols[0]))
    }

    fn i64s(v: &[Option<i64>]) -> ArrayRef {
        Arc::new(Int64Array::from(v.to_vec()))
    }

    /// The F212 case: `{i64::MAX, 1}` and `{-2}` total `i64::MAX - 1`, which fits, but the
    /// first pair's own total does not. It used to raise whenever those two rows shared a
    /// partition and succeed otherwise. Every row order is tried as well as every split,
    /// grouped (beside a group holding nulls and an all-null group) and ungrouped.
    #[test]
    fn a_total_that_fits_succeeds_on_every_split_and_path() {
        let orders: [[i64; 3]; 3] = [[i64::MAX, 1, -2], [-2, i64::MAX, 1], [1, -2, i64::MAX]];
        for order in orders {
            // Ungrouped.
            let vals = i64s(&order.map(Some));
            let want: Answer = [(None, Some(i64::MAX - 1))].into();
            assert_eq!(single_node(None, &vals).unwrap(), want);
            for cuts in splits(3) {
                for (path, got) in every_path(None, &vals, &cuts) {
                    assert_eq!(
                        got.unwrap_or_else(|e| panic!("{path}: {e:?}")),
                        want,
                        "{path}"
                    );
                }
            }
            // Grouped: key 1 holds the three rows interleaved with key 2 (7, NULL, 5) and
            // key 3 (all NULL).
            let keys = i64s(&[1, 2, 3, 1, 2, 3, 1, 2].map(Some));
            let vals = i64s(&[
                Some(order[0]),
                Some(7),
                None,
                Some(order[1]),
                None,
                None,
                Some(order[2]),
                Some(5),
            ]);
            let want: Answer = [
                (Some(1), Some(i64::MAX - 1)),
                (Some(2), Some(12)),
                (Some(3), None),
            ]
            .into();
            assert_eq!(single_node(Some(&keys), &vals).unwrap(), want);
            for cuts in splits(8) {
                for (path, got) in every_path(Some(&keys), &vals, &cuts) {
                    assert_eq!(
                        got.unwrap_or_else(|e| panic!("{path}: {e:?}")),
                        want,
                        "{path}"
                    );
                }
            }
        }
    }

    /// The same cancellation through the *fused* scan, which a lone `SUM` never reaches
    /// (`fused::FUSE_THRESHOLD`): beside a `COUNT(*)`, the partial `{MAX, 1}` is built by
    /// `fused::FusedAcc::SumI64NoNull`/`SumI64` rather than by [`state`].
    #[test]
    fn a_fused_partial_keeps_its_exact_total_too() {
        for with_null in [false, true] {
            let mut v = vec![Some(i64::MAX), Some(1)];
            if with_null {
                v.push(None);
            }
            v.push(Some(-2));
            let n = v.len();
            let vals = i64s(&v);
            let keys = i64s(&vec![Some(0); n]);
            let ps: Vec<Partial> = [(0, n - 1), (n - 1, 1)]
                .iter()
                .map(|&(off, len)| {
                    let calls = [
                        AggCall::new(AggFunc::Sum, Some(vals.slice(off, len))),
                        AggCall::new(AggFunc::CountStar, None),
                    ];
                    partial(&[keys.slice(off, len)], &calls, len).unwrap()
                })
                .collect();
            let funcs = [AggFunc::Sum, AggFunc::CountStar];
            let merged = combine_with(&ps, &funcs, usize::MAX).unwrap();
            let out = agg_finalize(&funcs, &merged).unwrap();
            assert_eq!(out[0].as_primitive::<Int64Type>().value(0), i64::MAX - 1);
            assert_eq!(out[1].as_primitive::<Int64Type>().value(0), n as i64);
        }
    }

    /// A partial in the **legacy** form -- a bare `Int64` total, as an engine before this
    /// module persisted it in a streaming checkpoint or a shuffle file.
    fn legacy(keys: Option<Vec<Option<i64>>>, sums: Vec<Option<i64>>) -> Partial {
        Partial {
            group_columns: keys.map(|k| vec![i64s(&k)]).unwrap_or_default(),
            states: vec![vec![i64s(&sums)]],
        }
    }

    /// Legacy and new partials merge on every path, in either order, grouped and global, and
    /// the true total still decides overflow: `{MAX}` legacy plus a new `{1, -2}` fits, a
    /// legacy `{MAX}` plus a new `{1}` does not.
    #[test]
    fn legacy_int64_partials_merge_with_new_ones() {
        let new_grouped = |v: Vec<Option<i64>>| {
            let k = i64s(&vec![Some(1); v.len()]);
            let call = AggCall::new(AggFunc::Sum, Some(i64s(&v)));
            partial(&[k], std::slice::from_ref(&call), v.len()).unwrap()
        };
        let new_global = |v: Vec<Option<i64>>| {
            let call = AggCall::new(AggFunc::Sum, Some(i64s(&v)));
            partial(&[], std::slice::from_ref(&call), v.len()).unwrap()
        };
        let funcs = [AggFunc::Sum];
        // (legacy, new, expected): grouped with a legacy-only NULL group 2, then global.
        let run = |mut ps: Vec<Partial>, want: Result<Answer, ()>| {
            for flip in [false, true] {
                if flip {
                    ps.reverse();
                }
                for threshold in [usize::MAX, 1] {
                    let r = combine_with(&ps, &funcs, threshold)
                        .and_then(|m| Ok(answer(&m.group_columns, &agg_finalize(&funcs, &m)?[0])));
                    assert_eq!(
                        r.map_err(|e| assert!(matches!(e, RuntimeError::SumOverflow))),
                        want
                    );
                }
                let again: Vec<Partial> = ps
                    .iter()
                    .map(|p| Partial {
                        group_columns: p.group_columns.clone(),
                        states: p.states.clone(),
                    })
                    .collect();
                let mut store = MemSpillStore::new(4);
                let r = combine_finalize_spilling(again, &funcs, &mut store, 0)
                    .map(|g| answer(&g.group_columns, &g.agg_columns[0]));
                assert_eq!(
                    r.map_err(|e| assert!(matches!(e, RuntimeError::SumOverflow))),
                    want
                );
            }
        };
        run(
            vec![
                legacy(Some(vec![Some(1), Some(2)]), vec![Some(i64::MAX), None]),
                new_grouped(vec![Some(1), Some(-2)]),
            ],
            Ok([(Some(1), Some(i64::MAX - 1)), (Some(2), None)].into()),
        );
        run(
            vec![
                legacy(None, vec![Some(i64::MAX)]),
                new_global(vec![Some(1), Some(-2)]),
            ],
            Ok([(None, Some(i64::MAX - 1))].into()),
        );
        run(
            vec![
                legacy(None, vec![Some(i64::MAX)]),
                new_global(vec![Some(1)]),
            ],
            Err(()),
        );
        // Legacy alone -- a restored checkpoint finalized before any new partial arrives.
        run(
            vec![legacy(None, vec![Some(41)]), legacy(None, vec![Some(1)])],
            Ok([(None, Some(42))].into()),
        );
        let only = legacy(Some(vec![Some(7)]), vec![Some(5)]);
        let merged = combine_with(std::slice::from_ref(&only), &funcs, usize::MAX).unwrap();
        assert_eq!(
            merged.states[0][0].data_type(),
            &DataType::Int64,
            "a bare state stays bare"
        );
        assert_eq!(
            answer(
                &merged.group_columns,
                &agg_finalize(&funcs, &merged).unwrap()[0]
            ),
            [(Some(7), Some(5))].into()
        );
    }

    /// The two current-engine forms meet: a partial whose own total overflowed (marked) and
    /// ones that fit (bare). Every combine path -- serial, radix, spilled, a running state
    /// folded one partial at a time (the streaming shape), and `concat_disjoint` over
    /// key-disjoint pieces -- must reach the true total, and the result is `Int64`.
    #[test]
    fn bare_and_marked_partials_merge_on_every_path() {
        use super::super::concat_disjoint;
        let funcs = [AggFunc::Sum];
        let p = |keys: &[i64], v: &[Option<i64>]| {
            let k: Vec<ArrayRef> = if keys.is_empty() {
                vec![]
            } else {
                vec![i64s(&keys.iter().map(|&k| Some(k)).collect::<Vec<_>>())]
            };
            let call = AggCall::new(AggFunc::Sum, Some(i64s(v)));
            partial(&k, std::slice::from_ref(&call), v.len()).unwrap()
        };
        for keys in [vec![], vec![1i64, 1]] {
            let one = |k: &[i64]| if k.is_empty() { vec![] } else { vec![1i64] };
            let marked = p(&keys, &[Some(i64::MAX), Some(1)]);
            assert!(
                is_state(marked.states[0][0].data_type()),
                "its own total overflows"
            );
            let bare = p(&one(&keys), &[Some(-2)]);
            assert_eq!(
                bare.states[0][0].data_type(),
                &DataType::Int64,
                "the common form"
            );
            let key = (!keys.is_empty()).then_some(1);
            let want: Answer = [(key, Some(i64::MAX - 1))].into();
            let dup = |x: &Partial| Partial {
                group_columns: x.group_columns.clone(),
                states: x.states.clone(),
            };
            for threshold in [usize::MAX, 1] {
                for ps in [
                    vec![dup(&marked), dup(&bare)],
                    vec![dup(&bare), dup(&marked)],
                ] {
                    let m = combine_with(&ps, &funcs, threshold).unwrap();
                    assert_eq!(
                        answer(&m.group_columns, &agg_finalize(&funcs, &m).unwrap()[0]),
                        want
                    );
                    let mut store = MemSpillStore::new(4);
                    let g = combine_finalize_spilling(ps, &funcs, &mut store, 0).unwrap();
                    assert_eq!(answer(&g.group_columns, &g.agg_columns[0]), want);
                }
            }
            // A running state folded one partial at a time: two fitting partials overflow
            // *together* (bare + bare -> marked, not an error), and the third brings it back.
            let mut running = p(&one(&keys), &[Some(i64::MAX)]);
            for next in [p(&one(&keys), &[Some(1)]), p(&one(&keys), &[Some(-2)])] {
                running = combine_with(&[running, next], &funcs, usize::MAX).unwrap();
            }
            assert_eq!(
                answer(
                    &running.group_columns,
                    &agg_finalize(&funcs, &running).unwrap()[0]
                ),
                want
            );
        }
        // Key-disjoint pieces in different forms concatenate, each keeping its own total.
        let a = p(&[1, 1], &[Some(i64::MAX), Some(1)]);
        let b = p(&[2], &[Some(5)]);
        let joined = concat_disjoint(&[a, b]).unwrap();
        assert!(is_state(joined.states[0][0].data_type()));
        let t = inner(&joined.states[0][0]).unwrap();
        assert_eq!((t.value(0), t.value(1)), (i128::from(i64::MAX) + 1, 5));
        // And the process-boundary form is always the marked one.
        let wide = widen_bare(&b_again(), &funcs)
            .unwrap()
            .expect("a bare SUM state widens");
        assert!(is_state(wide.states[0][0].data_type()));
        fn b_again() -> Partial {
            let call = AggCall::new(AggFunc::Sum, Some(i64s(&[Some(5)])));
            partial(&[i64s(&[Some(2)])], std::slice::from_ref(&call), 1).unwrap()
        }
    }

    /// A true total past `i64` -- either end -- raises on every path and every split, even
    /// the ones where no single partition overflows (`{MAX}` `{1}` `{1}`).
    #[test]
    fn a_true_overflow_raises_on_every_split_and_path() {
        for rows in [
            [i64::MAX, 1, 1],
            [i64::MIN, -1, 0],
            [i64::MAX, i64::MAX, -1],
        ] {
            let vals = i64s(&rows.map(Some));
            assert!(matches!(
                single_node(None, &vals),
                Err(RuntimeError::SumOverflow)
            ));
            let keys = i64s(&[0, 0, 0].map(Some));
            assert!(matches!(
                single_node(Some(&keys), &vals),
                Err(RuntimeError::SumOverflow)
            ));
            for cuts in splits(3) {
                for k in [None, Some(&keys)] {
                    for (path, got) in every_path(k, &vals, &cuts) {
                        assert!(
                            matches!(got, Err(RuntimeError::SumOverflow)),
                            "{rows:?} {path}: {got:?}"
                        );
                    }
                }
            }
        }
    }

    /// Empty and all-null inputs: a global `SUM` is one NULL row, a grouped one has no rows
    /// or NULL groups -- never 0, and never an overflow.
    #[test]
    fn empty_and_all_null_inputs_are_null_not_zero() {
        let empty = i64s(&[]);
        assert_eq!(single_node(None, &empty).unwrap(), [(None, None)].into());
        assert_eq!(single_node(Some(&empty), &empty).unwrap(), Answer::new());
        let nulls = i64s(&[None, None, None]);
        let keys = i64s(&[Some(1), Some(2), Some(1)]);
        let want: Answer = [(Some(1), None), (Some(2), None)].into();
        for cuts in splits(3) {
            for (path, got) in every_path(Some(&keys), &nulls, &cuts) {
                assert_eq!(got.unwrap(), want, "{path}");
            }
            for (path, got) in every_path(None, &nulls, &cuts) {
                assert_eq!(got.unwrap(), [(None, None)].into(), "{path}");
            }
        }
    }

    /// The marker is what keeps a user's `DECIMAL(38, 0)` `SUM` a decimal: its state is the
    /// bare decimal, it combines as one, and it finalizes to itself -- never narrowed to
    /// `Int64` because it happens to share the accumulator's Arrow type.
    #[test]
    fn a_decimal_38_0_sum_is_not_mistaken_for_an_integer_sum() {
        let big = i128::from(i64::MAX) * 4;
        let d: ArrayRef = Arc::new(
            Decimal128Array::from(vec![big, 1, 2])
                .with_precision_and_scale(38, 0)
                .unwrap(),
        );
        let keys = i64s(&[Some(0), Some(0), Some(0)]);
        let ps: Vec<Partial> = [(0, 2), (2, 1)]
            .iter()
            .map(|&(off, len)| {
                let call = AggCall::new(AggFunc::Sum, Some(d.slice(off, len)));
                partial(&[keys.slice(off, len)], std::slice::from_ref(&call), len).unwrap()
            })
            .collect();
        assert_eq!(ps[0].states[0][0].data_type(), &INNER);
        let merged = combine_with(&ps, &[AggFunc::Sum], usize::MAX).unwrap();
        let out = agg_finalize(&[AggFunc::Sum], &merged).unwrap();
        assert_eq!(out[0].data_type(), &DataType::Decimal128(38, 0));
        assert_eq!(out[0].as_primitive::<Decimal128Type>().value(0), big + 3);
    }

    #[test]
    fn the_marker_is_the_field_name_not_the_decimal_type() {
        let s = from_i64(&[1, 2], vec![true, false]).unwrap();
        assert!(is_state(s.data_type()));
        // A user's DECIMAL(38, 0) SUM state is the bare decimal: never mistaken for ours.
        assert!(!is_state(&DataType::Decimal128(38, 0)));
        // Nor is a one-field struct under any other name.
        let other = DataType::Struct(Fields::from(vec![Field::new("x", INNER, true)]));
        assert!(!is_state(&other));
    }

    #[test]
    fn finalize_narrows_only_the_true_total() {
        let fits = wrap_sums(vec![i128::from(i64::MAX), 0], vec![true, false]).unwrap();
        let out = finalize(&fits).unwrap();
        let out = out.as_primitive::<Int64Type>();
        assert_eq!(out.value(0), i64::MAX);
        assert!(out.is_null(1), "an empty group stays null");

        let over = from_i128(vec![i128::from(i64::MAX) + 1], vec![true]).unwrap();
        assert!(matches!(finalize(&over), Err(RuntimeError::SumOverflow)));
        let under = from_i128(vec![i128::from(i64::MIN) - 1], vec![true]).unwrap();
        assert!(matches!(finalize(&under), Err(RuntimeError::SumOverflow)));
    }

    #[test]
    fn a_partition_whose_own_total_overflows_keeps_it_exactly() {
        let v: ArrayRef = Arc::new(Int64Array::from(vec![i64::MAX, 1, 3]));
        // Grouped: group 0 holds MAX + 1, group 1 holds 3.
        let s = state(&v, &[0, 0, 1], 2).unwrap();
        let t = inner(&s).unwrap();
        assert_eq!(t.value(0), i128::from(i64::MAX) + 1);
        assert_eq!(t.value(1), 3);
        // Global (whole-column, no ids read).
        let s = state(&v, &[], 1).unwrap();
        assert_eq!(inner(&s).unwrap().value(0), i128::from(i64::MAX) + 4);
    }
}
