//! Stable sort permutation for a `Utf8` / `LargeUtf8` sort key.
//!
//! Arrow's `sort_to_indices` is **not stable** for strings (nor, in general, for any
//! type): rows with equal keys come back in an arbitrary, input-size-dependent order.
//! The fixed-width keys dodge this because a full single-key sort takes the stable LSD
//! radix path (`super::radix_sort`); strings had no such path, so a string `ORDER BY`
//! was the one sort whose tie order was nondeterministic.
//!
//! That nondeterminism is what blocks the parallel sample-sort on a string key: the
//! sample-sort sorts each range independently, and a range's tie order could not be made
//! to agree with the whole-array sort's. This module supplies the missing guarantee —
//! ties resolve to input order — so the serial oracle and the per-range sorts produce the
//! identical relation, and `seq == par` holds for strings as it does for integers.
//!
//! Ties break on the original row index, which is a deterministic total order and is
//! exactly what a stable sort yields, so `sort_unstable_by` is safe (and avoids the
//! merge-sort allocation a `sort_by` would pay).

use arrow::array::{Array, ArrayRef, GenericStringArray, OffsetSizeTrait, UInt32Array};
use arrow::compute::SortOptions;
use arrow::datatypes::DataType;

/// The stable permutation that sorts a string `values` column under `opts`, or `None` if
/// `values` is not a string array (caller falls back to the comparison sort).
///
/// Nulls are grouped first/last per `opts.nulls_first`, in input order; non-null rows sort
/// byte-lexicographically (the ordering arrow itself uses for `Utf8`), descending inverting
/// only the key comparison, never the tie-break — so equal keys always keep input order.
pub(crate) fn stable_sort_indices_str(values: &ArrayRef, opts: SortOptions) -> Option<UInt32Array> {
    match values.data_type() {
        DataType::Utf8 => Some(sort_generic(
            values.as_any().downcast_ref::<GenericStringArray<i32>>()?,
            opts,
        )),
        DataType::LargeUtf8 => Some(sort_generic(
            values.as_any().downcast_ref::<GenericStringArray<i64>>()?,
            opts,
        )),
        _ => None,
    }
}

/// The first eight bytes of `s` as a big-endian `u64`, zero-padded when it is shorter.
///
/// Ordering this integer orders the strings, for every pair whose packs differ. The padding
/// is what makes that true rather than merely usual: a missing byte packs as `0`, no byte is
/// less than `0`, and a string that runs out is a prefix of the one that does not — which is
/// exactly the byte-lexicographic rule. Where the packs are *equal* the answer is unknown
/// (either the strings share a prefix, or one contains a literal NUL where the other ended),
/// and the caller falls through to a full comparison.
#[inline]
fn prefix8(s: &str) -> u64 {
    let b = s.as_bytes();
    let mut buf = [0u8; 8];
    let take = b.len().min(8);
    buf[..take].copy_from_slice(&b[..take]);
    u64::from_be_bytes(buf)
}

/// Rows sampled to decide whether a packed prefix is worth building.
const PREFIX_SAMPLE_ROWS: usize = 512;

/// Whether the first eight bytes discriminate enough rows for a packed key to pay.
///
/// The question is **not** how many distinct packs the sample holds — it is whether the packs
/// tie where the values do not. A URL column is the shape that makes the distinction matter:
/// every value begins `https://`, so one constant pack settles nothing and the pointer chase
/// happens on every comparison anyway (measured: 7% *slower* than comparing the values
/// directly). But a `shipmode` column with seven distinct values also has few distinct packs,
/// and there the pack settles every unequal pair — it is the best key available, not the worst.
/// A distinctness floor cannot tell those apart; comparing the two counts can, because they are
/// equal exactly when the pack is as informative as the value.
///
/// The sample must be drawn with a stride rather than from the head, for the same reason a
/// sorted-looking prefix would mislead — the first 512 rows of a partitioned scan are not a
/// sample of the column.
fn prefix_discriminates<'a>(live: usize, sample: impl Iterator<Item = &'a str>) -> bool {
    if live < PREFIX_SAMPLE_ROWS {
        // Too small for the difference to matter either way; the packed path is no worse.
        return true;
    }
    let (mut packs, mut values): (Vec<u64>, Vec<&str>) = sample
        .take(PREFIX_SAMPLE_ROWS)
        .map(|s| (prefix8(s), s))
        .unzip();
    packs.sort_unstable();
    packs.dedup();
    values.sort_unstable();
    values.dedup();
    // Half the resolution of the values themselves is the floor: the pack is allowed to lose
    // some pairs to a shared prefix, and stops paying once it loses most of them.
    packs.len() * 2 >= values.len()
}

/// [`prefix_discriminates`] over the live rows of a whole array, without an index list to stride.
fn packs_discriminate<O: OffsetSizeTrait>(arr: &GenericStringArray<O>) -> bool {
    let live = arr.len() - arr.null_count();
    let step = (live / PREFIX_SAMPLE_ROWS).max(1);
    let nulls = arr.nulls();
    prefix_discriminates(
        live,
        (0..arr.len())
            .filter(|&i| nulls.is_none_or(|nb| nb.is_valid(i)))
            .step_by(step)
            .map(|i| arr.value(i)),
    )
}

/// Whether `live` is already in the order the sort would put it in, so the permutation is the
/// identity.
///
/// A stable sort leaves an ordered input alone, which makes this an exact answer rather than a
/// heuristic — and it is the difference between `O(n)` and `O(n log n)` on two shapes that are
/// anything but rare. **A range of the parallel sample-sort whose key is constant** is the one
/// that motivated it: `ORDER BY <7-value column>` routes ~850 k identical strings into each
/// range, where every comparison ties on the key, resolves to the row index, and the sort does
/// `n log n` of them to reproduce the order it was handed. Pre-sorted data — a time-ordered scan,
/// a re-sort by the key a file is already clustered on — is the other.
///
/// Costs one comparison per row when it holds and, because `all` short-circuits, about two when
/// it does not.
fn already_ordered<O: OffsetSizeTrait>(
    arr: &GenericStringArray<O>,
    live: &[u32],
    descending: bool,
) -> bool {
    live.windows(2).all(|w| {
        let (x, y) = (arr.value(w[0] as usize), arr.value(w[1] as usize));
        if descending {
            x >= y
        } else {
            x <= y
        }
    })
}

/// Rows below which ranking cannot pay. The hash build is a fixed cost on *every* row, so it
/// only earns back what a comparison sort spends on `log n` of them — and a sort this small is
/// a few cache-resident passes either way.
const RANK_SORT_MIN_ROWS: usize = 1 << 12;

/// Distinct values above which the ranking build gives up and the caller sorts by comparison.
///
/// Two things bound it. The map has to stay cache-resident for a lookup to beat a comparison,
/// and the counting sort's bucket array is scanned in full once — both cheap at 4,096 short
/// strings and neither at a million. It also bounds the *wasted* work when the guess is wrong:
/// a high-cardinality column produces a new distinct value on nearly every row, so it trips
/// this after about `RANK_SORT_MAX_DISTINCT` rows and abandons a few thousand hashes, not a
/// pass over the column. The bad case is a column that is overwhelmingly one value with its
/// long tail arriving late — that pays one hash per row before declining — and it costs about
/// what the packed-prefix build it falls back to costs anyway.
const RANK_SORT_MAX_DISTINCT: usize = 1 << 12;

/// The permutation that sorts `live` by rank rather than by comparison, or `None` when the
/// column has too many distinct values (or too few rows) for that to pay.
///
/// This is the answer to a shape a comparison sort handles badly and a string comparison sort
/// worst of all: `ORDER BY <a column with seven values>`. Every one of the `n log n`
/// comparisons a general sort performs is two offset-buffer reads and a `memcmp`, and nearly
/// all of them compare two rows that hold *the same string* — work spent rediscovering, a few
/// hundred thousand times over, that `AIR` equals `AIR`. Ranking asks the question once per
/// *distinct value* instead: hash each row to a small dense id, order the handful of distinct
/// values among themselves, and place every row by counting. That is one hash and two counting
/// passes per row — `O(n)` with no comparisons at all — against `O(n log n)` `memcmp`s.
///
/// The permutation is identical to the one [`sort_live`]'s comparison paths build, and
/// `rank_and_comparison_sorts_agree` holds it there. Stability comes free and is not incidental:
/// `live` arrives in ascending row order and the scatter walks it in that order, so rows that
/// tie on the key leave in the order they arrived — the guarantee this whole module exists to
/// provide, and the one the parallel sample-sort depends on.
fn rank_sort_live<O: OffsetSizeTrait>(
    arr: &GenericStringArray<O>,
    live: &[u32],
    descending: bool,
) -> Option<Vec<u32>> {
    if live.len() < RANK_SORT_MIN_ROWS {
        return None;
    }
    // `ids` maps a value to a dense id in first-appearance order; `distinct` is its inverse.
    // Both borrow the array's value buffer, so neither copies a string.
    let mut ids: ahash::AHashMap<&str, u32> = ahash::AHashMap::new();
    let mut distinct: Vec<&str> = Vec::new();
    let mut row_ids: Vec<u32> = Vec::with_capacity(live.len());
    for &i in live {
        let value = arr.value(i as usize);
        let id = match ids.get(value) {
            Some(&id) => id,
            None => {
                if distinct.len() == RANK_SORT_MAX_DISTINCT {
                    return None; // too many distinct values for ranking to beat comparing
                }
                let id = distinct.len() as u32;
                distinct.push(value);
                ids.insert(value, id);
                id
            }
        };
        row_ids.push(id);
    }

    // Order the distinct values among themselves — the only byte comparisons this path
    // performs, and there are `d log d` of them for a `d` in the low thousands at worst.
    let mut by_value: Vec<u32> = (0..distinct.len() as u32).collect();
    by_value.sort_unstable_by(|&a, &b| distinct[a as usize].cmp(distinct[b as usize]));
    let d = distinct.len();
    let mut rank_of = vec![0u32; d];
    for (rank, &id) in by_value.iter().enumerate() {
        // Descending reverses the *key* order only. Ties still resolve to input order, which
        // the scatter below gives regardless of how the ranks are numbered.
        rank_of[id as usize] = if descending {
            (d - 1 - rank) as u32
        } else {
            rank as u32
        };
    }

    // Counting sort over the dense ranks: tally, prefix-sum to bucket starts, then scatter.
    let mut starts = vec![0usize; d + 1];
    for &id in &row_ids {
        starts[rank_of[id as usize] as usize + 1] += 1;
    }
    for rank in 0..d {
        starts[rank + 1] += starts[rank];
    }
    let mut out = vec![0u32; live.len()];
    for (position, &id) in row_ids.iter().enumerate() {
        let rank = rank_of[id as usize] as usize;
        out[starts[rank]] = live[position];
        starts[rank] += 1;
    }
    Some(out)
}

/// Sort `live` (the non-null row indices) into key order, stable on the original index.
///
/// Sorting bare indices means every comparison is two random reads of the offset buffer
/// followed by two more of the value bytes — a pointer chase per comparison, `n log n` times,
/// over buffers that leave cache on any sort worth calling large. Carrying the first eight
/// bytes inline turns comparisons that the pack settles into a register compare against a
/// *sequentially* read array.
///
/// That only helps when the pack actually settles them, so [`prefix_discriminates`] decides.
/// Both branches produce the identical permutation — `the_prefix_key_agrees_with_a_full_
/// comparison` holds them to it — so this is a cost choice, not a semantic one.
///
/// Ahead of both sits [`rank_sort_live`], which replaces comparison sorting outright when the
/// column has few enough distinct values to rank them.
fn sort_live<O: OffsetSizeTrait>(
    arr: &GenericStringArray<O>,
    mut live: Vec<u32>,
    opts: SortOptions,
) -> Vec<u32> {
    if already_ordered(arr, &live, opts.descending) {
        return live;
    }
    if let Some(ordered) = rank_sort_live(arr, &live, opts.descending) {
        return ordered;
    }
    let step = (live.len() / PREFIX_SAMPLE_ROWS).max(1);
    let discriminates = prefix_discriminates(
        live.len(),
        live.iter().step_by(step).map(|&i| arr.value(i as usize)),
    );
    if !discriminates {
        live.sort_unstable_by(|&a, &b| {
            let (x, y) = (arr.value(a as usize), arr.value(b as usize));
            let ord = if opts.descending { y.cmp(x) } else { x.cmp(y) };
            ord.then_with(|| a.cmp(&b))
        });
        return live;
    }
    let mut keyed: Vec<(u64, u32)> = live
        .into_iter()
        .map(|i| (prefix8(arr.value(i as usize)), i))
        .collect();
    keyed.sort_unstable_by(|&(px, a), &(py, b)| {
        // Equal packs prove nothing (see `prefix8`), so those fall through to the full
        // comparison; unequal packs settle it without touching the value buffer at all.
        let ord = if px == py {
            let (x, y) = (arr.value(a as usize), arr.value(b as usize));
            if opts.descending {
                y.cmp(x)
            } else {
                x.cmp(y)
            }
        } else if opts.descending {
            py.cmp(&px)
        } else {
            px.cmp(&py)
        };
        // Descending inverts the key comparison only, never the tie-break: equal keys keep
        // input order either way, which is the stability this module exists to provide.
        ord.then_with(|| a.cmp(&b))
    });
    keyed.into_iter().map(|(_, i)| i).collect()
}

/// The `k` best **non-null** row indices of a string key, in sorted order, or `None` for a
/// non-string array or a column whose packed prefix cannot narrow the field.
///
/// Two passes over a `u64` prefix array replace the `O(n log n)` comparison sort [`sort_live`]
/// runs. The first ranks by [`prefix8`] alone and keeps the best `k`; because a strictly smaller
/// pack proves a strictly smaller string, the `k`-th of those packs is a **threshold** every
/// member of the true top-`k` is at or below. The second pass collects exactly the rows at or
/// below it, which the third sorts for real. Only rows the prefix could not separate are ever
/// compared byte-by-byte, and on a discriminating column there are about `k` of them.
pub(super) fn top_k_live(values: &ArrayRef, descending: bool, k: usize) -> Option<Vec<u32>> {
    match values.data_type() {
        DataType::Utf8 => top_k_generic(
            values.as_any().downcast_ref::<GenericStringArray<i32>>()?,
            descending,
            k,
        ),
        DataType::LargeUtf8 => top_k_generic(
            values.as_any().downcast_ref::<GenericStringArray<i64>>()?,
            descending,
            k,
        ),
        _ => None,
    }
}

/// A **weak** order-preserving `u64` per row: the packed prefix, inverted for `descending`.
///
/// Unlike [`super::radix_sort::ranks`] this does not settle every pair — two rows sharing eight
/// leading bytes rank equal whatever their values — so a caller may use it to *narrow* a field but
/// never to order one. `None` for a non-string array, or for a column whose prefix is too
/// undiscriminating to narrow anything (see [`packs_discriminate`]).
pub(super) fn prefix_ranks(values: &ArrayRef, descending: bool) -> Option<Vec<u64>> {
    match values.data_type() {
        DataType::Utf8 => packed(
            values.as_any().downcast_ref::<GenericStringArray<i32>>()?,
            descending,
        ),
        DataType::LargeUtf8 => packed(
            values.as_any().downcast_ref::<GenericStringArray<i64>>()?,
            descending,
        ),
        _ => None,
    }
}

fn packed<O: OffsetSizeTrait>(arr: &GenericStringArray<O>, descending: bool) -> Option<Vec<u64>> {
    // Ask the same question [`sort_live`] asks, and ask it *first*. A column the pack cannot
    // separate — every value starting `https://` — would otherwise build the whole pack array,
    // fill the heap, scan the candidates up to the budget and only then decline, so the caller's
    // full sort would run on top of all of it. Measured on such a column: declining late cost
    // 146 -> 212 ms, and declining on a 512-row sample costs nothing measurable.
    if !packs_discriminate(arr) {
        return None;
    }
    // Packed once, read three times by the callers. Recomputing `prefix8` per pass would chase the
    // offset and value buffers each time; a `u64` per row is 128 KiB at a full morsel and stays in
    // L2. Null slots pack to whatever their (valid) offsets span — never read, since every pass
    // skips them.
    Some(
        (0..arr.len())
            .map(|i| {
                let p = prefix8(arr.value(i));
                if descending {
                    !p
                } else {
                    p
                }
            })
            .collect(),
    )
}

fn top_k_generic<O: OffsetSizeTrait>(
    arr: &GenericStringArray<O>,
    descending: bool,
    k: usize,
) -> Option<Vec<u32>> {
    let n = arr.len();
    let nulls = arr.nulls();
    let packs = packed(arr, descending)?;
    let seeds = super::heap_select_k(n, nulls, k, |i| packs[i]);
    // Fewer live rows than `k`: they are all in the answer, so there is no threshold to find.
    let Some(&last) = seeds.last().filter(|_| seeds.len() == k) else {
        return Some(exact_order(arr, seeds, descending));
    };
    let threshold = packs[last as usize];
    let budget =
        k.saturating_add((k * super::TOP_K_CANDIDATE_SLACK).max(super::TOP_K_CANDIDATE_FLOOR));
    let mut candidates: Vec<u32> = Vec::with_capacity(budget.min(n));
    for (i, &pack) in packs.iter().enumerate() {
        if pack <= threshold && nulls.is_none_or(|nb| nb.is_valid(i)) {
            if candidates.len() == budget {
                return None; // the prefix separates too little to be worth selecting on
            }
            candidates.push(i as u32);
        }
    }
    let mut ordered = exact_order(arr, candidates, descending);
    ordered.truncate(k);
    Some(ordered)
}

/// Sort row indices by their actual string values, ties by input position — the same total order
/// [`sort_live`] produces, applied to the handful of rows the prefix could not settle.
fn exact_order<O: OffsetSizeTrait>(
    arr: &GenericStringArray<O>,
    mut idx: Vec<u32>,
    descending: bool,
) -> Vec<u32> {
    idx.sort_unstable_by(|&a, &b| {
        let (x, y) = (arr.value(a as usize), arr.value(b as usize));
        let ord = if descending { y.cmp(x) } else { x.cmp(y) };
        ord.then_with(|| a.cmp(&b))
    });
    idx
}

fn sort_generic<O: OffsetSizeTrait>(arr: &GenericStringArray<O>, opts: SortOptions) -> UInt32Array {
    let n = arr.len();
    let mut null_idx: Vec<u32> = Vec::new();
    let mut live_idx: Vec<u32> = Vec::with_capacity(n);
    for i in 0..n {
        if arr.is_null(i) {
            null_idx.push(i as u32);
        } else {
            live_idx.push(i as u32);
        }
    }
    let live_idx = sort_live(arr, live_idx, opts);

    let mut out: Vec<u32> = Vec::with_capacity(n);
    if opts.nulls_first {
        out.extend_from_slice(&null_idx);
        out.extend_from_slice(&live_idx);
    } else {
        out.extend_from_slice(&live_idx);
        out.extend_from_slice(&null_idx);
    }
    UInt32Array::from(out)
}

#[cfg(test)]
mod ordered_shortcut_tests {
    use std::sync::Arc;

    use arrow::array::StringArray;

    use super::*;

    /// The two shapes the short-circuit exists for must come back with the *same permutation* the
    /// comparison sort produces, not merely a valid one — the identity is only correct because a
    /// stable sort keeps input order, and asserting against the sort is what says so.
    ///
    /// A constant key is the sample-sort's range on a low-cardinality column (7 distinct values
    /// over 6 M rows put ~850 k identical strings in each range); ascending is a scan already
    /// clustered on the key. The third case is the guard: unordered input must not take it.
    #[test]
    fn an_ordered_input_sorts_to_itself() {
        let constant: Vec<Option<&str>> = vec![Some("AIR"); 2_000];
        let owned: Vec<String> = (0..2_000).map(|i| format!("k{i:06}")).collect();
        let ascending: Vec<Option<&str>> = owned.iter().map(|s| Some(s.as_str())).collect();
        let mut shuffled = ascending.clone();
        shuffled.swap(0, 1_999);

        for vals in [constant, ascending, shuffled] {
            let a: ArrayRef = Arc::new(StringArray::from(vals));
            let arr = a.as_any().downcast_ref::<StringArray>().unwrap();
            for descending in [false, true] {
                let opts = SortOptions {
                    descending,
                    nulls_first: false,
                };
                let live: Vec<u32> = (0..arr.len() as u32).collect();
                let got = sort_live(arr, live.clone(), opts);
                let mut want = live;
                want.sort_by(|&x, &y| {
                    let (p, q) = (arr.value(x as usize), arr.value(y as usize));
                    let ord = if descending { q.cmp(p) } else { p.cmp(q) };
                    ord.then(x.cmp(&y))
                });
                assert_eq!(got, want, "descending={descending}");
            }
        }
    }

    /// The gate must separate "few distinct packs because the column has few distinct values"
    /// from "few distinct packs because they all share a prefix". Those look identical to a
    /// distinctness floor and opposite to the sort: in the first the pack settles every unequal
    /// pair and is the best key available; in the second it settles nothing.
    #[test]
    fn the_gate_reads_resolution_not_distinctness() {
        let low_card: Vec<Option<&str>> = (0..4_000)
            .map(|i| Some(["AIR", "RAIL", "TRUCK", "MAIL", "SHIP", "FOB", "REG AIR"][i % 7]))
            .collect();
        let a: ArrayRef = Arc::new(StringArray::from(low_card));
        assert!(
            packs_discriminate(a.as_any().downcast_ref::<StringArray>().unwrap()),
            "seven distinct values pack to seven distinct keys — the pack settles every pair"
        );

        let owned: Vec<String> = (0..4_000)
            .map(|i| format!("https://example.com/{i:08}"))
            .collect();
        let shared: Vec<Option<&str>> = owned.iter().map(|s| Some(s.as_str())).collect();
        let a: ArrayRef = Arc::new(StringArray::from(shared));
        assert!(
            !packs_discriminate(a.as_any().downcast_ref::<StringArray>().unwrap()),
            "a constant eight-byte prefix settles nothing, whatever the value distinctness"
        );
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::StringArray;

    use super::*;

    fn idx(vals: Vec<Option<&str>>, descending: bool, nulls_first: bool) -> Vec<u32> {
        let a: ArrayRef = Arc::new(StringArray::from(vals));
        stable_sort_indices_str(
            &a,
            SortOptions {
                descending,
                nulls_first,
            },
        )
        .unwrap()
        .values()
        .to_vec()
    }

    #[test]
    fn ties_keep_input_order() {
        // Every key equal -> the permutation must be the identity.
        let v = vec![Some("a"); 64];
        assert_eq!(idx(v, false, false), (0..64u32).collect::<Vec<_>>());
    }

    #[test]
    fn descending_ties_still_keep_input_order() {
        let v = vec![Some("a"); 32];
        assert_eq!(idx(v, true, false), (0..32u32).collect::<Vec<_>>());
    }

    #[test]
    fn orders_bytewise_and_places_nulls() {
        let v = vec![Some("b"), None, Some("a"), Some("c")];
        assert_eq!(idx(v.clone(), false, false), vec![2, 0, 3, 1]);
        assert_eq!(idx(v.clone(), false, true), vec![1, 2, 0, 3]);
        assert_eq!(idx(v, true, false), vec![3, 0, 2, 1]);
    }

    #[test]
    fn interleaved_ties_are_stable() {
        let v: Vec<Option<&str>> = (0..300)
            .map(|i| Some(["aaa", "bbb", "ccc"][i % 3]))
            .collect();
        let got = idx(v, false, false);
        // First 100 are the "aaa" rows: original indices 0,3,6,... ascending.
        assert_eq!(
            got[..100],
            (0..100).map(|i| i as u32 * 3).collect::<Vec<_>>()[..]
        );
    }

    /// The packed-prefix sort must equal a full byte comparison with an index tie-break, on
    /// exactly the inputs where a prefix key can lie.
    ///
    /// The packing carries eight bytes and pads a shorter string with zeros, so the cases
    /// that matter are the ones where those two facts interact: strings that agree for eight
    /// or more bytes and diverge later (the pack ties and the full comparison must decide);
    /// a string that is a *prefix* of another (the pad must read as "ended", and it must not
    /// collide with a real byte); and a string containing a literal NUL where another simply
    /// ended, which packs identically and is the one case padding genuinely cannot resolve.
    #[test]
    fn the_prefix_key_agrees_with_a_full_comparison() {
        let values: Vec<Option<&str>> = vec![
            Some("abcdefghZZZ"), // shares 8 bytes with the next two
            Some("abcdefghAAA"),
            Some("abcdefgh"),   // a prefix of both
            Some("abcdefgh\0"), // packs identically to the line above
            Some("abc"),
            Some("abc\0"), // packs identically to "abc"
            Some(""),
            None,
            Some("\0"),
            Some("abcdefghi"),
            Some("zzzzzzzzzzzz"),
            Some("abcdefgh"), // an exact duplicate, for the tie-break
            None,
            Some("ab"),
        ];
        for descending in [false, true] {
            for nulls_first in [false, true] {
                let got = idx(values.clone(), descending, nulls_first);

                // The reference: nulls grouped per the flag in input order, non-nulls by a
                // full byte comparison with the original index breaking ties.
                let mut nulls: Vec<u32> = Vec::new();
                let mut live: Vec<u32> = Vec::new();
                for (i, v) in values.iter().enumerate() {
                    if v.is_none() {
                        nulls.push(i as u32);
                    } else {
                        live.push(i as u32);
                    }
                }
                live.sort_by(|&a, &b| {
                    let (x, y) = (
                        values[a as usize].unwrap().as_bytes(),
                        values[b as usize].unwrap().as_bytes(),
                    );
                    let ord = if descending { y.cmp(x) } else { x.cmp(y) };
                    ord.then_with(|| a.cmp(&b))
                });
                let mut want = Vec::new();
                if nulls_first {
                    want.extend_from_slice(&nulls);
                    want.extend_from_slice(&live);
                } else {
                    want.extend_from_slice(&live);
                    want.extend_from_slice(&nulls);
                }
                assert_eq!(
                    got, want,
                    "descending={descending} nulls_first={nulls_first}"
                );
            }
        }
    }

    /// Both comparators must produce the identical permutation, on an input long enough that
    /// the gate actually runs and on both sides of it.
    ///
    /// The gate is a *cost* decision, so the one thing that must never depend on it is the
    /// answer. A shared-prefix column (every value starting `https://example.com/`) takes the
    /// index-comparison branch and a distinct one takes the packed branch; the relation is
    /// the same either way.
    #[test]
    fn the_gate_changes_the_cost_and_not_the_answer() {
        for shared in ["", "https://example.com/"] {
            let owned: Vec<String> = (0..PREFIX_SAMPLE_ROWS * 4)
                .map(|i| format!("{shared}{:08x}", (i * 2654435761u64 as usize) % 4096))
                .collect();
            let vals: Vec<Option<&str>> = owned.iter().map(|s| Some(s.as_str())).collect();
            let a: ArrayRef = Arc::new(StringArray::from(vals.clone()));
            let arr = a.as_any().downcast_ref::<StringArray>().unwrap();
            let live: Vec<u32> = (0..owned.len() as u32).collect();

            // The two shapes must land on opposite sides of the gate, or this test would be
            // comparing the packed branch with itself and proving nothing.
            let discriminating = packs_discriminate(arr);
            assert_eq!(
                discriminating,
                shared.is_empty(),
                "gate verdict for shared={shared:?} is not the one this test needs"
            );

            for descending in [false, true] {
                let opts = SortOptions {
                    descending,
                    nulls_first: false,
                };
                let got = sort_live(arr, live.clone(), opts);
                let mut want = live.clone();
                want.sort_by(|&x, &y| {
                    let (p, q) = (arr.value(x as usize), arr.value(y as usize));
                    let ord = if descending { q.cmp(p) } else { p.cmp(q) };
                    ord.then_with(|| x.cmp(&y))
                });
                assert_eq!(got, want, "shared={shared:?} descending={descending}");
            }
        }
    }

    /// A pack that differs must already answer the comparison, or the sort would consult the
    /// value bytes only for genuine ties and get the rest wrong without ever noticing.
    #[test]
    fn an_unequal_pack_orders_the_strings_the_same_way_the_bytes_do() {
        let cases = [
            ("abc", "abd"),
            ("abc", "abcd"),
            ("", "a"),
            ("a", "\u{7f}"),
            ("abcdefgh", "abcdefgi"),
            ("\0", "\0\0"),
            ("z", "zz"),
        ];
        for (x, y) in cases {
            let (px, py) = (prefix8(x), prefix8(y));
            if px != py {
                assert_eq!(
                    px.cmp(&py),
                    x.as_bytes().cmp(y.as_bytes()),
                    "pack order disagreed with byte order for {x:?} vs {y:?}"
                );
            }
        }
        // And the case the padding cannot settle, which must therefore tie rather than lie.
        assert_eq!(prefix8("abc"), prefix8("abc\0"));
        assert_eq!(prefix8("abcdefgh"), prefix8("abcdefghZZZ"));
    }

    #[test]
    fn non_string_returns_none() {
        let a: ArrayRef = Arc::new(arrow::array::Int64Array::from(vec![1i64, 2]));
        assert!(stable_sort_indices_str(
            &a,
            SortOptions {
                descending: false,
                nulls_first: false
            }
        )
        .is_none());
    }

    /// The counting sort over ranks must produce exactly the permutation a full byte
    /// comparison with an index tie-break produces — on a column it accepts, in both
    /// directions.
    ///
    /// The `is_some` assertion is the load-bearing half. Without it this test passes just as
    /// happily when `rank_sort_live` declines every input and the comparison path quietly
    /// answers on its behalf, which is the failure mode that would make the whole optimization
    /// dead code while looking green.
    #[test]
    fn rank_and_comparison_sorts_agree() {
        let shipmodes = ["AIR", "RAIL", "TRUCK", "MAIL", "SHIP", "FOB", "REG AIR"];
        let vals: Vec<Option<&str>> = (0..RANK_SORT_MIN_ROWS * 3)
            .map(|i| Some(shipmodes[(i * 7 + i / 5) % shipmodes.len()]))
            .collect();
        let a: ArrayRef = Arc::new(StringArray::from(vals));
        let arr = a.as_any().downcast_ref::<StringArray>().unwrap();
        let live: Vec<u32> = (0..arr.len() as u32).collect();

        for descending in [false, true] {
            let got = rank_sort_live(arr, &live, descending)
                .expect("seven distinct values are inside the ranking budget");
            let mut want = live.clone();
            want.sort_by(|&x, &y| {
                let (p, q) = (arr.value(x as usize), arr.value(y as usize));
                let ord = if descending { q.cmp(p) } else { p.cmp(q) };
                ord.then_with(|| x.cmp(&y))
            });
            assert_eq!(got, want, "descending={descending}");
        }
    }

    /// Ranking must decline a column with more distinct values than its budget, and the
    /// caller's answer must not change when it does.
    ///
    /// The decline is what bounds the wasted work on a high-cardinality column, so a silent
    /// regression to "accepts everything" would cost a full hash build on exactly the columns
    /// ranking cannot help — invisible except as a slowdown.
    #[test]
    fn ranking_declines_a_high_cardinality_column() {
        let owned: Vec<String> = (0..RANK_SORT_MAX_DISTINCT * 2)
            .map(|i| format!("value-{i:08}"))
            .collect();
        let vals: Vec<Option<&str>> = owned.iter().map(|s| Some(s.as_str())).collect();
        let a: ArrayRef = Arc::new(StringArray::from(vals));
        let arr = a.as_any().downcast_ref::<StringArray>().unwrap();
        let live: Vec<u32> = (0..arr.len() as u32).collect();
        assert!(
            rank_sort_live(arr, &live, false).is_none(),
            "a column of all-distinct values is past the ranking budget"
        );

        // And the sort itself still answers correctly through the path it falls back to.
        let got = sort_live(
            arr,
            live.clone(),
            SortOptions {
                descending: false,
                nulls_first: false,
            },
        );
        let mut want = live;
        want.sort_by(|&x, &y| arr.value(x as usize).cmp(arr.value(y as usize)));
        assert_eq!(got, want);
    }

    /// Nulls, duplicates and both flags together, end to end through the public entry point,
    /// on an input large enough to reach the ranking path.
    ///
    /// `sort_live` never sees a null — `sort_generic` splits them out first — so the risk this
    /// covers is the seam, not the ranking: a permutation built over `live` positions and then
    /// concatenated with the null block has to stay a permutation of *row* indices.
    #[test]
    fn ranking_handles_nulls_and_both_flags() {
        let vals: Vec<Option<&str>> = (0..RANK_SORT_MIN_ROWS * 2)
            .map(|i| match i % 5 {
                0 => None,
                n => Some(["", "b", "aa", "a"][n - 1]),
            })
            .collect();
        for descending in [false, true] {
            for nulls_first in [false, true] {
                let got = idx(vals.clone(), descending, nulls_first);
                let (mut nulls, mut live): (Vec<u32>, Vec<u32>) = (Vec::new(), Vec::new());
                for (i, v) in vals.iter().enumerate() {
                    if v.is_none() {
                        nulls.push(i as u32);
                    } else {
                        live.push(i as u32);
                    }
                }
                live.sort_by(|&x, &y| {
                    let (p, q) = (vals[x as usize].unwrap(), vals[y as usize].unwrap());
                    let ord = if descending { q.cmp(p) } else { p.cmp(q) };
                    ord.then_with(|| x.cmp(&y))
                });
                let mut want = Vec::new();
                if nulls_first {
                    want.extend_from_slice(&nulls);
                    want.extend_from_slice(&live);
                } else {
                    want.extend_from_slice(&live);
                    want.extend_from_slice(&nulls);
                }
                assert_eq!(
                    got, want,
                    "descending={descending} nulls_first={nulls_first}"
                );
            }
        }
    }
}
