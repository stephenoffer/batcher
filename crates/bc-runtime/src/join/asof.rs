//! ASOF (nearest-match) join: each left row matched to the right row whose `on` key
//! is nearest in a direction within its `by` group. Left-style (every left row
//! emitted; unmatched → null right). Split out of the join module along the
//! algorithm seam; like the equi-join it carries no single-node assumption —
//! partitioning both sides by `by` makes a global ASOF the union of per-partition
//! ASOFs (the distributed seam).

use arrow::array::{Array, ArrayRef, UInt32Array};
use arrow::row::{OwnedRow, RowConverter, SortField};
use indexmap::IndexMap;

use super::{null_mask, JoinIndices};
use crate::error::RuntimeError;
use crate::measure::NumericKeys;

/// Which side of the left key an ASOF match may come from.
///
/// `Backward` (the default everywhere: pandas, Polars, DuckDB) takes the last known value
/// at or before the left row — the "what was the price when this trade happened" reading.
/// `Forward` takes the first value at or after it. `Nearest` takes whichever of the two is
/// closer, breaking an exact tie toward the backward one, matching pandas'
/// `direction="nearest"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsofDirection {
    Backward,
    Forward,
    Nearest,
}

/// Everything about an ASOF match beyond the keys themselves.
///
/// Bundled rather than passed as three more positional arguments, because the three
/// interact: `Nearest` and a `tolerance` both need a measurable key, and
/// `allow_exact_matches` moves the boundary both directions search from.
#[derive(Debug, Clone, Copy)]
pub struct AsofSpec {
    pub direction: AsofDirection,
    /// Cap on the distance between the matched keys, in the key's own units and in
    /// microseconds for a temporal key. `None` = uncapped.
    pub tolerance: Option<f64>,
    /// Whether a right row whose key *equals* the left row's may be the match.
    ///
    /// `false` is the strict form pandas spells `allow_exact_matches=False`: a backward
    /// join then takes the last row strictly *before* the left key. It is what keeps a
    /// backtest honest — a quote stamped at the same instant as the trade is information
    /// the trade did not have, and matching it is look-ahead bias that inflates every
    /// result downstream without ever looking like a bug.
    pub allow_exact_matches: bool,
}

impl Default for AsofSpec {
    fn default() -> Self {
        AsofSpec {
            direction: AsofDirection::Backward,
            tolerance: None,
            allow_exact_matches: true,
        }
    }
}

/// Compute ASOF (nearest-match) join indices. Every left row is emitted (left-style);
/// it is matched to the right row whose `on` key is nearest *in `direction`* within
/// the same `by` group (exact `by` equality). Unmatched left rows get a null right
/// index (arrow `take` then yields null), exactly like a left outer join.
///
/// [`AsofDirection`] chooses which side of the left key a match may come from. Keys are
/// arrow row-encoded, so `on` (order-preserving) and `by` (equality) work for any type.
/// Rows with a null `on` never match. As with the equi-join primitive, partitioning both
/// sides by `by` makes a global ASOF equal the union of per-partition ASOFs — the seam
/// the distributed path can use.
///
/// `tolerance` caps how far apart the two keys may be, in the key's own units and in
/// **microseconds** for any temporal key. Beyond it the left row is unmatched rather than
/// matched to a stale value, which is the difference between "the quote at the time of the
/// trade" and "some quote from three days earlier". It requires a numeric or temporal `on`
/// key, as does `Nearest`, because both have to subtract two keys; a non-numeric key with
/// either errors rather than silently ignoring the request.
pub fn asof_join_indices(
    left_on: &ArrayRef,
    right_on: &ArrayRef,
    left_by: &[ArrayRef],
    right_by: &[ArrayRef],
    spec: AsofSpec,
) -> Result<JoinIndices, RuntimeError> {
    let AsofSpec {
        direction,
        tolerance,
        allow_exact_matches,
    } = spec;
    let n_left = left_on.len();
    let n_right = right_on.len();

    // Canonicalize signed zero / NaN on the `on` ordering key too. `-0.0` and `0.0` are the
    // *same value* (IEEE equality, and how DuckDB's ASOF inequality treats them), but arrow's
    // row encoding gives them distinct, totally-ordered bytes (`-0.0 < 0.0`). Without folding,
    // a left `on = -0.0` would find no right `on = 0.0` "≤" it (backward), and no right
    // `on = -0.0` "≥" a left `0.0` (forward) — silently missing an exact nearest match that
    // DuckDB emits. Likewise two distinct NaN bit patterns would fail to match, exactly the
    // bug the equi-join fixed by canonicalizing. Folding here (as `canon_f64` does everywhere)
    // is orthogonal to the ordering of *distinct* finite values — it only merges the values
    // that are already equal. An int `on` has no float column, so it is returned unchanged.
    let lon_canon = crate::keys::canonicalize_float_keys(std::slice::from_ref(left_on));
    let ron_canon = crate::keys::canonicalize_float_keys(std::slice::from_ref(right_on));
    let left_on: &ArrayRef = lon_canon.as_deref().map_or(left_on, |c| &c[0]);
    let right_on: &ArrayRef = ron_canon.as_deref().map_or(right_on, |c| &c[0]);

    // A distance is needed only to enforce a tolerance or to choose between the two
    // `Nearest` candidates; a plain backward/forward search reads only the ordering.
    let needs_distance = tolerance.is_some() || direction == AsofDirection::Nearest;
    let (left_num, right_num) = if needs_distance {
        let l = NumericKeys::read(left_on)?;
        let r = NumericKeys::read(right_on)?;
        match (l, r) {
            (Some(l), Some(r)) => (Some(l), Some(r)),
            _ => {
                return Err(RuntimeError::AsofKeyNotMeasurable {
                    dtype: left_on.data_type().to_string(),
                })
            }
        }
    } else {
        (None, None)
    };

    // One shared converter so left/right `on` encodings are mutually order-comparable.
    let on_conv = RowConverter::new(vec![SortField::new(right_on.data_type().clone())])?;
    let left_on_enc = on_conv.convert_columns(std::slice::from_ref(left_on))?;
    let right_on_enc = on_conv.convert_columns(std::slice::from_ref(right_on))?;

    // Canonicalize signed zero on the `by` *equality* keys so `-0.0`/`0.0` group together,
    // matching the hash/sort-merge equi-join paths.
    let lby_canon = crate::keys::canonicalize_float_keys(left_by);
    let rby_canon = crate::keys::canonicalize_float_keys(right_by);
    let left_by: &[ArrayRef] = lby_canon.as_deref().unwrap_or(left_by);
    let right_by: &[ArrayRef] = rby_canon.as_deref().unwrap_or(right_by);

    let by_conv = if left_by.is_empty() {
        None
    } else {
        Some(RowConverter::new(
            right_by
                .iter()
                .map(|a| SortField::new(a.data_type().clone()))
                .collect(),
        )?)
    };
    let left_by_enc = by_conv
        .as_ref()
        .map(|c| c.convert_columns(left_by))
        .transpose()?;
    let right_by_enc = by_conv
        .as_ref()
        .map(|c| c.convert_columns(right_by))
        .transpose()?;

    // A null in any `by` column makes the row match nothing — `by` is an *equality* key and
    // `NULL != NULL` (the same SQL rule the equi-join enforces via `null_mask`). Arrow's row
    // encoding gives a null a concrete byte string, so without this a null-`by` right row
    // would form a group that null-`by` left rows would then "match" — matching every left
    // null to a right null, which neither DuckDB nor the equi-join does. Empty `by` (no
    // grouping columns) has no null to mask.
    let left_by_null = left_by_enc.as_ref().map(|_| null_mask(left_by, n_left));
    let right_by_null = right_by_enc.as_ref().map(|_| null_mask(right_by, n_right));

    // Group right rows by `by` key (byte-encoded; empty key when there are no `by`
    // columns), each group sorted ascending by `on` for binary search.
    let mut groups: IndexMap<Vec<u8>, Vec<(OwnedRow, u32)>> = IndexMap::new();
    for j in 0..n_right {
        if right_on.is_null(j) || right_by_null.as_ref().is_some_and(|m| m[j]) {
            continue;
        }
        let key = right_by_enc
            .as_ref()
            .map_or_else(Vec::new, |e| e.row(j).as_ref().to_vec());
        groups
            .entry(key)
            .or_default()
            .push((right_on_enc.row(j).owned(), j as u32));
    }
    for v in groups.values_mut() {
        v.sort_by(|a, b| a.0.row().cmp(&b.0.row()));
    }

    let mut right_idx: Vec<Option<u32>> = Vec::with_capacity(n_left);
    for i in 0..n_left {
        if left_on.is_null(i) || left_by_null.as_ref().is_some_and(|m| m[i]) {
            right_idx.push(None);
            continue;
        }
        let key = left_by_enc
            .as_ref()
            .map_or_else(Vec::new, |e| e.row(i).as_ref().to_vec());
        let target = left_on_enc.row(i);
        let matched = groups.get(&key).and_then(|g| {
            // The two candidates either side of the left key. `back` is the last row at or
            // before it and `fwd` the first at or after it, so an exact match is *both* —
            // which is why a tie under `Nearest` resolves to the same row either way.
            // `allow_exact_matches` moves the boundary: with it, an equal key is the last
            // row `back` may take and the first `fwd` may take; without it, both must step
            // past the whole run of equal keys.
            let back = {
                let pp = match allow_exact_matches {
                    true => g.partition_point(|(on, _)| on.row() <= target),
                    false => g.partition_point(|(on, _)| on.row() < target),
                };
                (pp > 0).then(|| g[pp - 1].1)
            };
            let fwd = {
                let pp = match allow_exact_matches {
                    true => g.partition_point(|(on, _)| on.row() < target),
                    false => g.partition_point(|(on, _)| on.row() <= target),
                };
                (pp < g.len()).then(|| g[pp].1)
            };
            // `dist` is only ever `None` for a key with no distance, which `needs_distance`
            // has already rejected — so an unwrap-shaped default here would be unreachable
            // rather than lenient. Treating it as "infinitely far" keeps that unreachable.
            let dist = |j: u32| -> f64 {
                match (&left_num, &right_num) {
                    (Some(l), Some(r)) => l.distance(i, r, j as usize).unwrap_or(f64::INFINITY),
                    _ => f64::INFINITY,
                }
            };
            let chosen = match direction {
                AsofDirection::Backward => back,
                AsofDirection::Forward => fwd,
                // Ties go backward, matching pandas' `direction="nearest"`.
                AsofDirection::Nearest => match (back, fwd) {
                    (Some(b), Some(f)) => Some(if dist(f) < dist(b) { f } else { b }),
                    (b, f) => b.or(f),
                },
            }?;
            match tolerance {
                Some(tol) if dist(chosen) > tol => None,
                _ => Some(chosen),
            }
        });
        right_idx.push(matched);
    }

    Ok(JoinIndices {
        left: UInt32Array::from((0..n_left as u32).collect::<Vec<_>>()),
        right: UInt32Array::from(right_idx),
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Int64Array, StringArray};

    use super::*;

    fn i64s(v: Vec<Option<i64>>) -> ArrayRef {
        Arc::new(Int64Array::from(v))
    }
    fn strs(v: Vec<Option<&str>>) -> ArrayRef {
        Arc::new(StringArray::from(v))
    }

    /// A null `by` key matches nothing: `by` is an equality key and `NULL != NULL`. Before the
    /// fix, arrow's row encoder gave a null `by` a concrete byte string, so a null-`by` right
    /// row formed a group and every null-`by` left row "matched" it — disagreeing with DuckDB
    /// and with the equi-join's own `NULL != NULL` rule.
    #[test]
    fn null_by_key_matches_nothing() {
        // left: (sym, ts) — rows 1 and 2 have a null `sym`.
        let left_on = i64s(vec![Some(10), Some(20), Some(30), Some(10)]);
        let left_by = vec![strs(vec![Some("A"), None, None, Some("B")])];
        // right: a null-`sym` quote at ts=5 that must NOT be matched by the null-`sym` left rows.
        let right_on = i64s(vec![Some(5), Some(5), Some(25)]);
        let right_by = vec![strs(vec![None, Some("A"), None])];

        let idx = asof_join_indices(
            &left_on,
            &right_on,
            &left_by,
            &right_by,
            AsofSpec::default(),
        )
        .unwrap();
        let right: Vec<Option<u32>> = (0..idx.right.len())
            .map(|i| idx.right.is_valid(i).then(|| idx.right.value(i)))
            .collect();
        // row0 ("A", 10) -> right#1 ("A", 5); rows 1,2 (null sym) -> None; row3 ("B", ...) -> None.
        assert_eq!(right, vec![Some(1), None, None, None]);
    }

    fn f64s(v: Vec<Option<f64>>) -> ArrayRef {
        Arc::new(arrow::array::Float64Array::from(v))
    }
    fn asof_right(idx: &JoinIndices) -> Vec<Option<u32>> {
        (0..idx.right.len())
            .map(|i| idx.right.is_valid(i).then(|| idx.right.value(i)))
            .collect()
    }

    /// A float `on` key of `-0.0` is the *same value* as `0.0` (IEEE equality; how DuckDB's
    /// ASOF inequality treats it), so a nearest-match search must find it. Before the fix the
    /// `on` column was row-encoded without canonicalizing signed zero, so arrow's total order
    /// (`-0.0 < 0.0`) hid the match: `backward` from `-0.0` found no right `0.0 ≤ -0.0`, and
    /// `forward` from `0.0` found no right `-0.0 ≥ 0.0` — both returned NULL where DuckDB emits
    /// the exact match. Distinct NaN bit patterns had the same defect the equi-join already fixed.
    #[test]
    fn signed_zero_and_nan_on_key_match() {
        // backward: left -0.0 must match right 0.0 (equal → exact nearest).
        let idx = asof_join_indices(
            &f64s(vec![Some(-0.0)]),
            &f64s(vec![Some(0.0)]),
            &[],
            &[],
            AsofSpec::default(),
        )
        .unwrap();
        assert_eq!(
            asof_right(&idx),
            vec![Some(0)],
            "backward: -0.0 must match 0.0"
        );

        // forward: left 0.0 must match right -0.0.
        let idx = asof_join_indices(
            &f64s(vec![Some(0.0)]),
            &f64s(vec![Some(-0.0)]),
            &[],
            &[],
            AsofSpec {
                direction: AsofDirection::Forward,
                ..AsofSpec::default()
            },
        )
        .unwrap();
        assert_eq!(
            asof_right(&idx),
            vec![Some(0)],
            "forward: 0.0 must match -0.0"
        );

        // Two distinct NaN bit patterns are one canonical NaN → an exact match, not a miss.
        let nan2 = f64::from_bits(0x7ff8_0000_0000_0001);
        assert!(nan2.is_nan());
        let idx = asof_join_indices(
            &f64s(vec![Some(f64::NAN)]),
            &f64s(vec![Some(nan2)]),
            &[],
            &[],
            AsofSpec::default(),
        )
        .unwrap();
        assert_eq!(asof_right(&idx), vec![Some(0)], "NaN must match NaN");
    }

    /// Tiny deterministic xorshift RNG.
    struct Rng(u64);
    impl Rng {
        fn below(&mut self, n: u64) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x % n
        }
    }

    /// Independent brute-force ASOF reference. Mirrors the documented rule and tie-break:
    /// backward = the (max on ≤ target, then max original-row) right match within the `by`
    /// group; forward = the (min on ≥ target, then min original-row); nearest = the smallest
    /// |on − target|, preferring the backward candidate on a tie. `tolerance` drops any
    /// candidate further than that from the target. A null `on` or any null `by` (either
    /// side) matches nothing. Returns the chosen right row per left row.
    fn brute_asof(
        left_on: &[Option<i64>],
        right_on: &[Option<i64>],
        left_by: &[Vec<Option<i64>>],
        right_by: &[Vec<Option<i64>>],
        direction: AsofDirection,
        tolerance: Option<u64>,
        allow_exact: bool,
    ) -> Vec<Option<u32>> {
        // Any null in a `by` column means the row has no group at all, so the whole key
        // is `None` rather than a key with a hole in it.
        let by_of = |cols: &[Vec<Option<i64>>], row: usize| -> Option<Vec<i64>> {
            cols.iter().map(|c| c[row]).collect()
        };
        (0..left_on.len())
            .map(|i| {
                let lon = left_on[i]?;
                let lby = by_of(left_by, i)?;
                // Every right row that could match this left row at all.
                let candidates: Vec<(i64, u32)> = (0..right_on.len())
                    .filter_map(|j| {
                        let ron = right_on[j]?;
                        if by_of(right_by, j)? != lby {
                            return None;
                        }
                        if tolerance.is_some_and(|t| (ron - lon).unsigned_abs() > t) {
                            return None;
                        }
                        Some((ron, j as u32))
                    })
                    .collect();
                // The two sides, each under its own documented tie-break: backward takes
                // the largest `on` at or below the target and, among equals, the latest
                // row; forward takes the smallest at or above and, among equals, the
                // earliest. Building them separately keeps the reference a statement of
                // the rule rather than a second copy of the binary search.
                let backward = candidates
                    .iter()
                    .filter(|(on, _)| if allow_exact { *on <= lon } else { *on < lon })
                    .max_by_key(|(on, row)| (*on, *row))
                    .copied();
                let forward = candidates
                    .iter()
                    .filter(|(on, _)| if allow_exact { *on >= lon } else { *on > lon })
                    .min_by_key(|(on, row)| (*on, *row))
                    .copied();
                let chosen = match direction {
                    AsofDirection::Backward => backward,
                    AsofDirection::Forward => forward,
                    // Ties prefer the backward candidate (pandas' `direction="nearest"`).
                    AsofDirection::Nearest => match (backward, forward) {
                        (Some(b), Some(f)) => {
                            Some(if (f.0 - lon).unsigned_abs() < (b.0 - lon).unsigned_abs() {
                                f
                            } else {
                                b
                            })
                        }
                        (b, f) => b.or(f),
                    },
                };
                chosen.map(|(_, j)| j)
            })
            .collect()
    }

    /// Fuzz ASOF against the brute-force reference across random inputs: all three
    /// directions crossed with eight (tolerance, allow-exact) combinations, 0/1/2 `by` columns, nulls in `on` and `by`,
    /// heavy ties on `on`, empty sides, and unsorted input (the impl must sort each group
    /// itself). The reference searches every right row and ranks the candidates, so it
    /// shares no code with the kernel's binary search.
    #[test]
    fn fuzz_asof_matches_brute_force() {
        let mut rng = Rng(0xA50F_1234);
        for _ in 0..2000 {
            let nl = rng.below(8) as usize;
            let nr = rng.below(8) as usize;
            let n_by = rng.below(3) as usize; // 0, 1, or 2 by columns
            let gen_on = |rng: &mut Rng, n: usize| -> Vec<Option<i64>> {
                (0..n)
                    .map(|_| (rng.below(6) != 0).then(|| rng.below(5) as i64 - 2))
                    .collect()
            };
            let gen_by = |rng: &mut Rng, n: usize| -> Vec<Vec<Option<i64>>> {
                (0..n_by)
                    .map(|_| {
                        (0..n)
                            .map(|_| (rng.below(6) != 0).then(|| rng.below(2) as i64))
                            .collect()
                    })
                    .collect()
            };
            let lon = gen_on(&mut rng, nl);
            let ron = gen_on(&mut rng, nr);
            let lby_v = gen_by(&mut rng, nl);
            let rby_v = gen_by(&mut rng, nr);

            let left_on = i64s(lon.clone());
            let right_on = i64s(ron.clone());
            let left_by: Vec<ArrayRef> = lby_v.iter().map(|c| i64s(c.clone())).collect();
            let right_by: Vec<ArrayRef> = rby_v.iter().map(|c| i64s(c.clone())).collect();

            for direction in [
                AsofDirection::Backward,
                AsofDirection::Forward,
                AsofDirection::Nearest,
            ] {
                // `on` values span [-2, 2], so these tolerances cover "nothing matches",
                // the interesting middle, and "the tolerance never binds".
                for (tol, exact) in [
                    (None, true),
                    (Some(0u64), true),
                    (Some(1), true),
                    (Some(2), true),
                    (Some(100), true),
                    (None, false),
                    (Some(1), false),
                    (Some(100), false),
                ] {
                    let idx = asof_join_indices(
                        &left_on,
                        &right_on,
                        &left_by,
                        &right_by,
                        AsofSpec {
                            direction,
                            tolerance: tol.map(|t| t as f64),
                            allow_exact_matches: exact,
                        },
                    )
                    .unwrap();
                    let got: Vec<Option<u32>> = (0..idx.right.len())
                        .map(|i| idx.right.is_valid(i).then(|| idx.right.value(i)))
                        .collect();
                    let want = brute_asof(&lon, &ron, &lby_v, &rby_v, direction, tol, exact);
                    // The chosen right row is unambiguous under our tie-break, so compare exactly.
                    assert_eq!(
                        got, want,
                        "asof mismatch direction={direction:?} tol={tol:?} exact={exact}\n lon={lon:?} ron={ron:?}\n lby={lby_v:?} rby={rby_v:?}"
                    );
                    // Left indices must always be the identity 0..nl (left-style).
                    let lidx: Vec<u32> = (0..idx.left.len()).map(|i| idx.left.value(i)).collect();
                    assert_eq!(lidx, (0..nl as u32).collect::<Vec<_>>());
                }
            }
        }
    }

    /// A multi-column `by` where only one component is null still masks the whole row (any-null
    /// == no match), matching the equi-join's `null_mask` semantics.
    #[test]
    fn partial_null_by_key_matches_nothing() {
        let left_on = i64s(vec![Some(10), Some(10)]);
        let left_by = vec![strs(vec![Some("A"), Some("A")]), i64s(vec![Some(1), None])];
        let right_on = i64s(vec![Some(5), Some(5)]);
        let right_by = vec![strs(vec![Some("A"), Some("A")]), i64s(vec![Some(1), None])];

        let idx = asof_join_indices(
            &left_on,
            &right_on,
            &left_by,
            &right_by,
            AsofSpec::default(),
        )
        .unwrap();
        let right: Vec<Option<u32>> = (0..idx.right.len())
            .map(|i| idx.right.is_valid(i).then(|| idx.right.value(i)))
            .collect();
        // row0 ("A",1) matches right#0; row1 ("A",null) matches nothing.
        assert_eq!(right, vec![Some(0), None]);
    }
}
