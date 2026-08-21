//! Stable sort permutation for a **byte-lexicographic** sort key: `Utf8`, `LargeUtf8`,
//! `Binary`, `LargeBinary` and `FixedSizeBinary`.
//!
//! Arrow's `sort_to_indices` is **not stable** for these types (nor, in general, for any
//! type): rows with equal keys come back in an arbitrary, input-size-dependent order.
//! The fixed-width numeric keys dodge this because a full single-key sort takes the stable
//! LSD radix path (`super::radix_sort`); byte keys had no such path, so a byte-keyed
//! `ORDER BY` was the one sort whose tie order was nondeterministic.
//!
//! That nondeterminism is what blocks the parallel sample-sort on a byte key: the
//! sample-sort sorts each range independently, and a range's tie order could not be made
//! to agree with the whole-array sort's. This module supplies the missing guarantee —
//! ties resolve to input order — so the serial oracle and the per-range sorts produce the
//! identical relation, and `seq == par` holds for byte keys as it does for integers.
//!
//! Ties break on the original row index, which is a deterministic total order and is
//! exactly what a stable sort yields, so `sort_unstable_by` is safe (and avoids the
//! merge-sort allocation a `sort_by` would pay).
//!
//! ## Why one module covers strings *and* binary
//!
//! Every path here compares `&[u8]`, and that is not a convenience — it is the definition of
//! the order for both families. Arrow orders `Utf8` byte-lexicographically (Rust's `str::cmp`
//! *is* `<[u8]>::cmp` on the UTF-8 encoding), and it orders `Binary` the same way. So a string
//! key and a binary key are one sort with two spellings of `value(i)`, and [`ByteKeys`] is
//! that spelling.
//!
//! Saying it the other way round is what makes the point. While this module matched on
//! `Utf8 | LargeUtf8` and returned `None` for the rest, `None` did not mean "a little slower":
//! a `Binary` `ORDER BY` fell past this module, past the numeric radix, past the row encoder
//! (which declines below three keys), and landed on `lexsort_to_indices` with a row-index
//! tie-break column appended. `sample_sort` reads the same type list, so the sort then ran on
//! **one core** whatever the machine, and `dist`'s `range_partitionable` reads it again, so it
//! never distributed either. A short fixed-width key over a wide payload — a hash, a UUID, a
//! checksum, a composite key someone already encoded, and the record layout the CloudSort
//! benchmark defines — was the worst-served sort in the engine.

use std::sync::Arc;

use arrow::array::{ArrayRef, UInt32Array, UInt64Array};
use arrow::compute::SortOptions;
use bc_runtime::byte_key::{ByteKeyColumn, ByteKeys};

/// The widest key the pack covers: one `u64` word.
///
/// Past this a pack stops being the sort key and becomes a prefix, and a prefix cannot order a
/// column on its own. [`sort_live`]'s packed-prefix comparison sort is the answer for wider
/// keys, and it already carries a prefix for exactly that reason — which is also, measured,
/// why this is one word and not two. See [`radix_sort_live`].
const MAX_PACK_BYTES: usize = 8;

/// The stable permutation that sorts a byte-key `values` column under `opts`, or `None` if
/// `values` is not a byte-key array (caller falls back to the comparison sort).
///
/// Nulls are grouped first/last per `opts.nulls_first`, in input order; non-null rows sort
/// byte-lexicographically (the ordering arrow itself uses for `Utf8` and `Binary`), descending
/// inverting only the key comparison, never the tie-break — so equal keys always keep input
/// order.
pub(super) fn stable_sort_indices_bytes(
    values: &ArrayRef,
    opts: SortOptions,
) -> Option<UInt32Array> {
    Some(sort_generic(&ByteKeyColumn::new(values)?, opts))
}

/// Bytes `[from, from + 8)` of `key` as a big-endian `u64`, zero-padded when the key runs out.
///
/// Ordering this integer orders the keys, for every pair whose words differ. The padding is
/// what makes that true rather than merely usual: a missing byte packs as `0`, no byte is less
/// than `0`, and a key that runs out is a prefix of the one that does not — which is exactly
/// the byte-lexicographic rule. Where the words are *equal* the answer is unknown (either the
/// keys share a prefix, or one contains a literal NUL where the other ended) unless
/// [`ByteKeys::exact_pack_width`] has ruled that out, so a caller either checks that first or
/// falls through to a full comparison.
#[inline]
fn pack_word(key: &[u8], from: usize) -> u64 {
    let mut buf = [0u8; 8];
    if from < key.len() {
        let take = (key.len() - from).min(8);
        buf[..take].copy_from_slice(&key[from..from + take]);
    }
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
fn prefix_discriminates<'a>(live: usize, sample: impl Iterator<Item = &'a [u8]>) -> bool {
    if live < PREFIX_SAMPLE_ROWS {
        // Too small for the difference to matter either way; the packed path is no worse.
        return true;
    }
    let (mut packs, mut values): (Vec<u64>, Vec<&[u8]>) = sample
        .take(PREFIX_SAMPLE_ROWS)
        .map(|s| (pack_word(s, 0), s))
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
fn packs_discriminate<A: ByteKeys>(arr: &A) -> bool {
    let nulls = arr.null_buffer();
    let live = arr.len() - nulls.map_or(0, |nb| nb.null_count());
    let step = (live / PREFIX_SAMPLE_ROWS).max(1);
    prefix_discriminates(
        live,
        (0..arr.len())
            .filter(|&i| nulls.is_none_or(|nb| nb.is_valid(i)))
            .step_by(step)
            .map(|i| arr.key(i)),
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
fn already_ordered<A: ByteKeys>(arr: &A, live: &[u32], descending: bool) -> bool {
    live.windows(2).all(|w| {
        let (x, y) = (arr.key(w[0] as usize), arr.key(w[1] as usize));
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
/// This is the answer to a shape a comparison sort handles badly and a byte-key comparison sort
/// worst of all: `ORDER BY <a column with seven values>`. Every one of the `n log n`
/// comparisons a general sort performs is two offset-buffer reads and a `memcmp`, and nearly
/// all of them compare two rows that hold *the same value* — work spent rediscovering, a few
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
fn rank_sort_live<A: ByteKeys>(arr: &A, live: &[u32], descending: bool) -> Option<Vec<u32>> {
    if live.len() < RANK_SORT_MIN_ROWS {
        return None;
    }
    // `ids` maps a value to a dense id in first-appearance order; `distinct` is its inverse.
    // Both borrow the array's value buffer, so neither copies a key.
    let mut ids: ahash::AHashMap<&[u8], u32> = ahash::AHashMap::new();
    let mut distinct: Vec<&[u8]> = Vec::new();
    let mut row_ids: Vec<u32> = Vec::with_capacity(live.len());
    for &i in live {
        let value = arr.key(i as usize);
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

/// Rows below which the byte radix is not worth its counting passes — the comparison sort is a
/// few cache-resident passes at this size and needs no key array at all.
const RADIX_MIN_ROWS: usize = 1 << 12;

/// Rows above which the byte radix declines, because its random-scatter index array no longer
/// fits cache and it loses to the comparison sort.
///
/// The same bound, for the same reason and at the same value, as
/// `super::radix_sort::FLOAT_RADIX_MAX_ROWS`. A whole-relation sort of a large column therefore
/// never radixes — it arrives here only per-range (the parallel sample-sort) or per-run (the
/// external merge sort), and both are below this by construction.
const RADIX_MAX_ROWS: usize = 1 << 18;

/// The permutation that sorts `live` by **LSD radix over the key bytes themselves**, or `None`
/// when the key is too wide, or the slice too small or too large, for that to pay.
///
/// This is the byte-key counterpart of `super::radix_sort`, and it is deliberately not a second
/// radix: the key becomes one order-preserving `u64` word and
/// [`super::radix_sort::radix_sort_indices`] sorts that. One counting-sort implementation serves
/// integers, temporals, floats and now byte keys.
///
/// Exactness is [`ByteKeys::exact_pack_width`]'s decision and not this function's: where the
/// pack could tie two unequal keys, there is no width and this declines rather than resolving
/// the tie by input order and calling it sorted.
///
/// ## Why one word, and why a row cap
///
/// Both bounds are measurements rather than principle, and the measurement is
/// [`report_the_byte_radix_crossover`] — the radix against the packed-prefix comparison sort
/// [`sort_live`] falls to, over `FixedSizeBinary` keys of the widths a fixed-layout record uses
/// (`cargo test --release -p bc-interp --lib -- --ignored --nocapture
/// report_the_byte_radix_crossover`):
///
/// | Rows | 4 bytes | 8 bytes | 10 bytes | 16 bytes |
/// |---|---|---|---|---|
/// | 16,384 | **1.58x** | **1.50x** | 0.92x | 0.93x |
/// | 65,536 | **2.50x** | **1.59x** | 0.90x | 0.86x |
/// | 262,144 | **1.81x** | **1.06x** | 0.97x | 0.72x |
/// | 1,048,576 | **1.15x** | 0.49x | 0.52x | 0.42x |
/// | 4,194,304 | 0.77x | 0.40x | 0.28x | 0.19x |
///
/// Above 1.00x is the radix ahead. Absolute times move several tens of percent run to run on a
/// shared machine; the two crossovers this reads off — one word, and [`RADIX_MAX_ROWS`] rows —
/// do not, and 262,144 is where the eight-byte column crosses.
///
/// The comparison sort is hard to beat here and the reason is worth stating, because it is the
/// same reason this module packs a prefix at all: on a key of eight bytes or fewer the pack
/// *is* the key, so every comparison is a register compare over a sequentially read array and
/// the sort never touches the value buffer. That is already most of what a radix buys, without
/// the radix's eight scatter passes over an index array.
///
/// A **two-word** form was implemented and rejected on those numbers. It sorted the low word,
/// gathered the high word through that permutation and sorted it again — correct LSD, and
/// slower than the comparison sort at every width and size tried, because the second pass adds
/// a gather and eight more scatters to buy separation the eight-byte prefix had very nearly
/// achieved on its own. A 10-byte record key and a 16-byte UUID therefore keep the comparison
/// sort, which is the right answer for them and not a gap.
fn radix_sort_live<A: ByteKeys>(arr: &A, live: &[u32], opts: SortOptions) -> Option<Vec<u32>> {
    if live.len() < RADIX_MIN_ROWS || live.len() > RADIX_MAX_ROWS {
        return None;
    }
    arr.exact_pack_width(MAX_PACK_BYTES)?;
    let packed: ArrayRef = Arc::new(UInt64Array::from(
        live.iter()
            .map(|&i| pack_word(arr.key(i as usize), 0))
            .collect::<Vec<u64>>(),
    ));
    // `packed` is null-free by construction (the caller partitioned the nulls out before this
    // was built), so `opts.nulls_first` cannot reach anything and only `descending` matters.
    Some(
        super::radix_sort::radix_sort_indices(&packed, opts)?
            .values()
            .iter()
            .map(|&p| live[p as usize])
            .collect(),
    )
}

/// Sort `live` (the non-null row indices) into key order, stable on the original index.
///
/// Sorting bare indices means every comparison is two random reads of the offset buffer
/// followed by two more of the value bytes — a pointer chase per comparison, `n log n` times,
/// over buffers that leave cache on any sort worth calling large. The paths below each replace
/// some of that, and are tried cheapest-first:
///
/// 1. [`already_ordered`] — one pass, and the answer outright when it holds.
/// 2. [`rank_sort_live`] — no comparisons at all, when the column has few distinct values.
/// 3. [`radix_sort_live`] — no comparisons at all, when the key is narrow enough to pack whole.
/// 4. The packed-prefix comparison sort — carrying the first eight bytes inline so comparisons
///    the pack settles become a register compare against a *sequentially* read array.
///
/// The prefix only helps when it actually settles comparisons, so [`prefix_discriminates`]
/// decides between the last two spellings. Every branch produces the identical permutation —
/// `the_prefix_key_agrees_with_a_full_comparison` and `a_byte_radix_matches_the_comparison_sort`
/// hold them to it — so this is a cost choice, not a semantic one.
fn sort_live<A: ByteKeys>(arr: &A, mut live: Vec<u32>, opts: SortOptions) -> Vec<u32> {
    if already_ordered(arr, &live, opts.descending) {
        return live;
    }
    if let Some(ordered) = rank_sort_live(arr, &live, opts.descending) {
        return ordered;
    }
    if let Some(ordered) = radix_sort_live(arr, &live, opts) {
        return ordered;
    }
    let step = (live.len() / PREFIX_SAMPLE_ROWS).max(1);
    let discriminates = prefix_discriminates(
        live.len(),
        live.iter().step_by(step).map(|&i| arr.key(i as usize)),
    );
    if !discriminates {
        live.sort_unstable_by(|&a, &b| {
            let (x, y) = (arr.key(a as usize), arr.key(b as usize));
            let ord = if opts.descending { y.cmp(x) } else { x.cmp(y) };
            ord.then_with(|| a.cmp(&b))
        });
        return live;
    }
    let mut keyed: Vec<(u64, u32)> = live
        .into_iter()
        .map(|i| (pack_word(arr.key(i as usize), 0), i))
        .collect();
    keyed.sort_unstable_by(|&(px, a), &(py, b)| {
        // Equal packs prove nothing (see `pack_word`), so those fall through to the full
        // comparison; unequal packs settle it without touching the value buffer at all.
        let ord = if px == py {
            let (x, y) = (arr.key(a as usize), arr.key(b as usize));
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

/// The `k` best **non-null** row indices of a byte key, in sorted order, or `None` for a
/// non-byte-key array or a column whose packed prefix cannot narrow the field.
///
/// Two passes over a `u64` prefix array replace the `O(n log n)` comparison sort [`sort_live`]
/// runs. The first ranks by [`pack_word`] alone and keeps the best `k`; because a strictly
/// smaller pack proves a strictly smaller key, the `k`-th of those packs is a **threshold**
/// every member of the true top-`k` is at or below. The second pass collects exactly the rows at
/// or below it, which the third sorts for real. Only rows the prefix could not separate are ever
/// compared byte-by-byte, and on a discriminating column there are about `k` of them.
pub(super) fn top_k_live(values: &ArrayRef, descending: bool, k: usize) -> Option<Vec<u32>> {
    top_k_generic(&ByteKeyColumn::new(values)?, descending, k)
}

/// A **weak** order-preserving `u64` per row: the packed prefix, inverted for `descending`.
///
/// Unlike [`super::radix_sort::ranks`] this does not settle every pair — two rows sharing eight
/// leading bytes rank equal whatever their values — so a caller may use it to *narrow* a field but
/// never to order one. `None` for a non-byte-key array, or for a column whose prefix is too
/// undiscriminating to narrow anything (see [`packs_discriminate`]).
pub(super) fn prefix_ranks(values: &ArrayRef, descending: bool) -> Option<Vec<u64>> {
    packed(&ByteKeyColumn::new(values)?, descending)
}

fn packed<A: ByteKeys>(arr: &A, descending: bool) -> Option<Vec<u64>> {
    // Ask the same question [`sort_live`] asks, and ask it *first*. A column the pack cannot
    // separate — every value starting `https://` — would otherwise build the whole pack array,
    // fill the heap, scan the candidates up to the budget and only then decline, so the caller's
    // full sort would run on top of all of it. Measured on such a column: declining late cost
    // 146 -> 212 ms, and declining on a 512-row sample costs nothing measurable.
    if !packs_discriminate(arr) {
        return None;
    }
    // Packed once, read three times by the callers. Recomputing the pack per pass would chase the
    // offset and value buffers each time; a `u64` per row is 128 KiB at a full morsel and stays in
    // L2. Null slots pack to whatever their (valid) offsets span — never read, since every pass
    // skips them.
    let nulls = arr.null_buffer();
    Some(
        (0..arr.len())
            .map(|i| {
                if nulls.is_some_and(|nb| nb.is_null(i)) {
                    return 0;
                }
                let p = pack_word(arr.key(i), 0);
                if descending {
                    !p
                } else {
                    p
                }
            })
            .collect(),
    )
}

fn top_k_generic<A: ByteKeys>(arr: &A, descending: bool, k: usize) -> Option<Vec<u32>> {
    let n = arr.len();
    let nulls = arr.null_buffer();
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

/// Sort row indices by their actual key bytes, ties by input position — the same total order
/// [`sort_live`] produces, applied to the handful of rows the prefix could not settle.
fn exact_order<A: ByteKeys>(arr: &A, mut idx: Vec<u32>, descending: bool) -> Vec<u32> {
    idx.sort_unstable_by(|&a, &b| {
        let (x, y) = (arr.key(a as usize), arr.key(b as usize));
        let ord = if descending { y.cmp(x) } else { x.cmp(y) };
        ord.then_with(|| a.cmp(&b))
    });
    idx
}

fn sort_generic<A: ByteKeys>(arr: &A, opts: SortOptions) -> UInt32Array {
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
    use arrow::array::StringArray;

    use super::*;

    fn idx(vals: Vec<Option<&str>>, descending: bool, nulls_first: bool) -> Vec<u32> {
        let a: ArrayRef = Arc::new(StringArray::from(vals));
        stable_sort_indices_bytes(
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
            let (px, py) = (pack_word(x.as_bytes(), 0), pack_word(y.as_bytes(), 0));
            if px != py {
                assert_eq!(
                    px.cmp(&py),
                    x.as_bytes().cmp(y.as_bytes()),
                    "pack order disagreed with byte order for {x:?} vs {y:?}"
                );
            }
        }
        // And the case the padding cannot settle, which must therefore tie rather than lie.
        assert_eq!(pack_word(b"abc", 0), pack_word(b"abc\0", 0));
        assert_eq!(pack_word(b"abcdefgh", 0), pack_word(b"abcdefghZZZ", 0));
    }

    #[test]
    fn a_non_byte_key_returns_none() {
        let a: ArrayRef = Arc::new(arrow::array::Int64Array::from(vec![1i64, 2]));
        assert!(stable_sort_indices_bytes(
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

#[cfg(test)]
mod byte_key_tests {
    use arrow::array::{
        BinaryArray, FixedSizeBinaryArray, LargeBinaryArray, LargeStringArray, StringArray,
    };

    use super::*;

    /// The permutation a full byte comparison with an input-order tie-break produces — the
    /// definition every path in this module is held to.
    fn reference(values: &[Option<Vec<u8>>], descending: bool, nulls_first: bool) -> Vec<u32> {
        let (mut nulls, mut live): (Vec<u32>, Vec<u32>) = (Vec::new(), Vec::new());
        for (i, v) in values.iter().enumerate() {
            if v.is_none() {
                nulls.push(i as u32);
            } else {
                live.push(i as u32);
            }
        }
        live.sort_by(|&a, &b| {
            let (x, y) = (
                values[a as usize].as_deref().unwrap(),
                values[b as usize].as_deref().unwrap(),
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
        want
    }

    /// A deterministic pseudo-random `width`-byte key, with `null_every`-th row null and a
    /// duplicate every seventh so ties, nulls and distinct values all appear.
    fn keys(rows: usize, width: usize, null_every: usize) -> Vec<Option<Vec<u8>>> {
        (0..rows)
            .map(|i| {
                if null_every > 0 && i % null_every == 0 {
                    return None;
                }
                let seed = (i / 7) as u64;
                let mut v = Vec::with_capacity(width);
                let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
                for _ in 0..width {
                    x ^= x >> 33;
                    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
                    v.push((x >> 40) as u8);
                }
                Some(v)
            })
            .collect()
    }

    fn binary(values: &[Option<Vec<u8>>]) -> ArrayRef {
        Arc::new(BinaryArray::from_iter(values.iter().map(|v| v.as_deref())))
    }

    fn fixed(values: &[Option<Vec<u8>>], width: usize) -> ArrayRef {
        Arc::new(
            FixedSizeBinaryArray::try_from_sparse_iter_with_size(
                values.iter().cloned(),
                width as i32,
            )
            .expect("uniform width"),
        )
    }

    /// The radix must produce the *same permutation* the comparison sort does, not merely a
    /// sorted one — the whole claim is that it is a drop-in replacement, and the tie order is
    /// what the parallel sample-sort and the external merge sort depend on agreeing about.
    ///
    /// Random bytes mean roughly half the keys contain a literal `0`, which is exactly the byte
    /// a padded pack could collide — uniform width is what makes it safe, and this is the test
    /// that says so. The widths past one word must **decline**, because a pack that cannot hold
    /// the key cannot order it; that half is asserted here rather than left to
    /// [`report_the_byte_radix_crossover`], which only measures.
    #[test]
    fn a_byte_radix_matches_the_comparison_sort() {
        for width in [1usize, 4, 8] {
            let values = keys(RADIX_MIN_ROWS * 2, width, 0);
            let a = fixed(&values, width);
            let arr = a.as_any().downcast_ref::<FixedSizeBinaryArray>().unwrap();
            let live: Vec<u32> = (0..arr.len() as u32).collect();
            for descending in [false, true] {
                let opts = SortOptions {
                    descending,
                    nulls_first: false,
                };
                let got = radix_sort_live(arr, &live, opts)
                    .unwrap_or_else(|| panic!("width {width} is inside the pack budget"));
                assert_eq!(
                    got,
                    reference(&values, descending, false),
                    "width={width} descending={descending}"
                );
            }
        }
    }

    /// Outside its budget the radix declines, and the sort is still exactly right — the decline
    /// routes to the comparison sort, never to a wrong order.
    ///
    /// Three budgets, each with its own reason: a key wider than one word cannot be packed at
    /// all; a slice below [`RADIX_MIN_ROWS`] cannot earn the key array back; and one above
    /// [`RADIX_MAX_ROWS`] scatters outside cache and loses to comparing.
    #[test]
    fn the_radix_declines_outside_its_budget() {
        let opts = SortOptions {
            descending: false,
            nulls_first: false,
        };
        let wide = keys(RADIX_MIN_ROWS * 2, MAX_PACK_BYTES + 2, 0);
        let small = keys(RADIX_MIN_ROWS - 1, 8, 0);
        for values in [wide, small] {
            let width = values[0].as_ref().unwrap().len();
            let a = fixed(&values, width);
            let arr = a.as_any().downcast_ref::<FixedSizeBinaryArray>().unwrap();
            let live: Vec<u32> = (0..arr.len() as u32).collect();
            assert!(
                radix_sort_live(arr, &live, opts).is_none(),
                "width={width} rows={} is outside the radix budget",
                values.len()
            );
            assert_eq!(
                stable_sort_indices_bytes(&a, opts)
                    .unwrap()
                    .values()
                    .to_vec(),
                reference(&values, false, false),
                "width={width}"
            );
        }
    }

    /// One column under test: its name, the array, and the bytes its order is checked against.
    type Column<'a> = (&'a str, ArrayRef, &'a Vec<Option<Vec<u8>>>);

    /// Every spelling of a byte key must order the same data identically, through the public
    /// entry point, with nulls and both flags.
    ///
    /// This is the contract the generalization exists for: before it, three of these five
    /// types returned `None` here and fell to a different sort with a different tie order.
    #[test]
    fn every_byte_key_type_sorts_alike() {
        let values = keys(RADIX_MIN_ROWS * 2, 8, 11);
        let text: Vec<Option<String>> = values
            .iter()
            .map(|v| {
                v.as_ref()
                    .map(|b| b.iter().map(|c| (b'a' + c % 26) as char).collect())
            })
            .collect();
        let text_bytes: Vec<Option<Vec<u8>>> = text
            .iter()
            .map(|s| s.as_ref().map(|s| s.as_bytes().to_vec()))
            .collect();
        let as_str: Vec<Option<&str>> = text.iter().map(|s| s.as_deref()).collect();

        let columns: Vec<Column<'_>> = vec![
            (
                "utf8",
                Arc::new(StringArray::from(as_str.clone())) as ArrayRef,
                &text_bytes,
            ),
            (
                "large_utf8",
                Arc::new(LargeStringArray::from(as_str.clone())) as ArrayRef,
                &text_bytes,
            ),
            ("binary", binary(&values), &values),
            (
                "large_binary",
                Arc::new(LargeBinaryArray::from_iter(
                    values.iter().map(|v| v.as_deref()),
                )) as ArrayRef,
                &values,
            ),
            ("fixed_size_binary", fixed(&values, 8), &values),
        ];

        for (name, column, expected) in columns {
            for descending in [false, true] {
                for nulls_first in [false, true] {
                    let opts = SortOptions {
                        descending,
                        nulls_first,
                    };
                    let got = stable_sort_indices_bytes(&column, opts)
                        .unwrap_or_else(|| panic!("{name} is a byte key"))
                        .values()
                        .to_vec();
                    assert_eq!(
                        got,
                        reference(expected, descending, nulls_first),
                        "{name} descending={descending} nulls_first={nulls_first}"
                    );
                }
            }
        }
    }

    /// The pack must decline exactly where padding could lie, and the sort must still be right
    /// when it does.
    ///
    /// `b"ab"` and `b"ab\0"` pad to the same eight bytes, so a column holding both cannot be
    /// ordered by the pack at all; a column of mixed lengths with no `0` byte can. Declining is
    /// not a performance detail here — accepting would resolve two unequal keys as a tie and
    /// return them in input order, which is a wrong answer that looks sorted.
    #[test]
    fn a_pad_collision_declines_the_pack() {
        let collides: Vec<Option<Vec<u8>>> = vec![
            Some(b"ab".to_vec()),
            Some(b"ab\0".to_vec()),
            Some(b"a".to_vec()),
        ];
        let a = binary(&collides);
        assert!(
            a.as_any()
                .downcast_ref::<BinaryArray>()
                .unwrap()
                .exact_pack_width(MAX_PACK_BYTES)
                .is_none(),
            "a mixed-length column holding a NUL byte has no exact pack width"
        );

        let clean: Vec<Option<Vec<u8>>> = vec![
            Some(b"ab".to_vec()),
            Some(b"abc".to_vec()),
            Some(b"a".to_vec()),
        ];
        let a = binary(&clean);
        assert_eq!(
            a.as_any()
                .downcast_ref::<BinaryArray>()
                .unwrap()
                .exact_pack_width(MAX_PACK_BYTES),
            Some(3),
            "mixed lengths with no NUL byte pad unambiguously"
        );

        // And the answer is the same either way, because the decline routes to the comparison
        // sort rather than to a wrong order.
        for values in [collides, clean] {
            let column = binary(&values);
            let opts = SortOptions {
                descending: false,
                nulls_first: false,
            };
            let got = stable_sort_indices_bytes(&column, opts)
                .unwrap()
                .values()
                .to_vec();
            assert_eq!(got, reference(&values, false, false));
        }
    }

    /// Report the byte radix against the packed-prefix comparison sort it precedes, at the key
    /// widths a fixed-layout record uses. Not an assertion — the measurement [`radix_sort_live`]
    /// cites for both of its bounds, so that they can be argued from numbers and re-checked when
    /// the machine or the allocator changes.
    ///
    /// Widths past one word are measured through the **two-word LSD** form that was rejected:
    /// sort the low word, gather the high word through that permutation, sort it again. It is
    /// correct — the assertion below holds it to the comparison sort at every width — and it is
    /// the thing the table shows losing, so it lives here rather than nowhere.
    ///
    /// `cargo test --release -p bc-interp --lib -- --ignored --nocapture report_the_byte_radix_crossover`
    #[test]
    #[ignore = "measurement, not an assertion"]
    fn report_the_byte_radix_crossover() {
        use std::time::Instant;

        /// The permutation of `keys` through the shared integer radix.
        fn radix(keys: Vec<u64>, opts: SortOptions) -> Vec<u32> {
            let packed: ArrayRef = Arc::new(UInt64Array::from(keys));
            crate::ops::radix_sort::radix_sort_indices(&packed, opts)
                .expect("a u64 column radixes")
                .values()
                .to_vec()
        }

        println!(
            "{:>9} {:>6} {:>12} {:>12} {:>8}",
            "rows", "width", "radix", "comparison", "ratio"
        );
        for rows in [1usize << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22] {
            for width in [4usize, 8, 10, 16] {
                let values = keys(rows, width, 0);
                let a = fixed(&values, width);
                let arr = a.as_any().downcast_ref::<FixedSizeBinaryArray>().unwrap();
                let live: Vec<u32> = (0..arr.len() as u32).collect();
                let opts = SortOptions {
                    descending: false,
                    nulls_first: false,
                };

                let start = Instant::now();
                let high: Vec<u64> = live
                    .iter()
                    .map(|&i| pack_word(arr.key(i as usize), 0))
                    .collect();
                let permutation = if width <= 8 {
                    radix(high, opts)
                } else {
                    // The rejected two-word form, least-significant word first.
                    let low = radix(
                        live.iter()
                            .map(|&i| pack_word(arr.key(i as usize), 8))
                            .collect(),
                        opts,
                    );
                    let ordered_high: Vec<u64> = low.iter().map(|&p| high[p as usize]).collect();
                    radix(ordered_high, opts)
                        .into_iter()
                        .map(|p| low[p as usize])
                        .collect()
                };
                let by_radix: Vec<u32> =
                    permutation.into_iter().map(|p| live[p as usize]).collect();
                let radix_ms = start.elapsed().as_secs_f64() * 1e3;

                // The same permutation the comparison branch of `sort_live` builds, timed alone.
                let start = Instant::now();
                let mut keyed: Vec<(u64, u32)> = live
                    .iter()
                    .map(|&i| (pack_word(arr.key(i as usize), 0), i))
                    .collect();
                keyed.sort_unstable_by(|&(px, a), &(py, b)| {
                    let ord = if px == py {
                        arr.key(a as usize).cmp(arr.key(b as usize))
                    } else {
                        px.cmp(&py)
                    };
                    ord.then_with(|| a.cmp(&b))
                });
                let by_comparison: Vec<u32> = keyed.into_iter().map(|(_, i)| i).collect();
                let comparison_ms = start.elapsed().as_secs_f64() * 1e3;

                // Both forms must agree, or the table would be comparing a sort with a
                // not-quite-sort and reporting the difference as a speed.
                assert_eq!(by_radix, by_comparison, "rows={rows} width={width}");
                println!(
                    "{rows:>9} {width:>6} {radix_ms:>11.1}ms {comparison_ms:>11.1}ms {:>7.2}x",
                    comparison_ms / radix_ms
                );
            }
        }
    }

    /// Attribute the cost of a byte-key sort across the three things that changed when this
    /// module was generalized: the type-erased column, the pack-width probe, and the radix.
    ///
    /// `cargo test --release -p bc-interp --lib -- --ignored --nocapture report_the_byte_key_sort_costs`
    #[test]
    #[ignore = "measurement, not an assertion"]
    fn report_the_byte_key_sort_costs() {
        use std::time::Instant;

        use arrow::array::StringArray;

        let rows = 1 << 15; // one range of a 2M-row sample-sort
        let opts = SortOptions {
            descending: false,
            nulls_first: false,
        };
        println!(
            "{:>6} {:>12} {:>12} {:>12} {:>12} {:>12}",
            "width", "erased", "monomorph", "probe", "radix", "compare"
        );
        for width in [4usize, 8, 12] {
            let owned: Vec<String> = (0..rows)
                .map(|i| {
                    (0..width)
                        .map(|j| {
                            (b'a' + (((i * 2654435761u64 as usize) >> (j * 5)) % 26) as u8) as char
                        })
                        .collect()
                })
                .collect();
            let vals: Vec<Option<&str>> = owned.iter().map(|s| Some(s.as_str())).collect();
            let a: ArrayRef = Arc::new(StringArray::from(vals));
            let concrete = a.as_any().downcast_ref::<StringArray>().unwrap();
            let erased = ByteKeyColumn::new(&a).unwrap();
            let live: Vec<u32> = (0..rows as u32).collect();

            let time = |f: &dyn Fn()| {
                let start = Instant::now();
                for _ in 0..20 {
                    f();
                }
                start.elapsed().as_secs_f64() * 1e3 / 20.0
            };
            let erased_ms = time(&|| {
                sort_live(&erased, live.clone(), opts);
            });
            let mono_ms = time(&|| {
                sort_live(concrete, live.clone(), opts);
            });
            let probe_ms = time(&|| {
                std::hint::black_box(erased.exact_pack_width(MAX_PACK_BYTES));
            });
            let radix_ms = time(&|| {
                std::hint::black_box(radix_sort_live(&erased, &live, opts));
            });
            // The branch the radix precedes, timed on its own so the two can be compared
            // rather than assumed.
            let compare_ms = time(&|| {
                let mut keyed: Vec<(u64, u32)> = live
                    .iter()
                    .map(|&i| (pack_word(erased.key(i as usize), 0), i))
                    .collect();
                keyed.sort_unstable_by(|&(px, a), &(py, b)| px.cmp(&py).then_with(|| a.cmp(&b)));
                std::hint::black_box(keyed);
            });
            println!(
                "{width:>6} {erased_ms:>11.2}ms {mono_ms:>11.2}ms {probe_ms:>11.2}ms                  {radix_ms:>11.2}ms {compare_ms:>11.2}ms"
            );
        }
    }
}
