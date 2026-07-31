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

/// One distinct pack per this many sampled rows is the floor for using the packed key.
///
/// Measured on 2 M keys (see [`sort_live`] for what the packed key does): at a sampled
/// distinctness of 1.0 the packed key sorts 4.7x faster, at 1/16 it still wins 1.13x, and by
/// 1/500 — every key sharing six or more leading bytes — it *loses* about 7%, because the
/// pack ties on every comparison so the pointer chase happens anyway and the packing is pure
/// overhead. 1/32 sits between the two measured regimes with room on both sides.
const PREFIX_MIN_DISTINCT_RATIO: usize = 32;

/// Whether the first eight bytes discriminate enough rows for a packed key to pay.
///
/// A URL column is the shape that makes this question worth asking: every value begins
/// `https://` , so the pack is constant and settles nothing. Sampling with a stride rather
/// than from the head matters for the same reason a sorted-looking prefix would mislead — the
/// first 512 rows of a partitioned scan are not a sample of the column.
fn prefix_discriminates<O: OffsetSizeTrait>(arr: &GenericStringArray<O>, live: &[u32]) -> bool {
    if live.len() < PREFIX_SAMPLE_ROWS {
        // Too small for the difference to matter either way; the packed path is no worse.
        return true;
    }
    let step = (live.len() / PREFIX_SAMPLE_ROWS).max(1);
    let mut packs: Vec<u64> = live
        .iter()
        .step_by(step)
        .take(PREFIX_SAMPLE_ROWS)
        .map(|&i| prefix8(arr.value(i as usize)))
        .collect();
    packs.sort_unstable();
    packs.dedup();
    packs.len() * PREFIX_MIN_DISTINCT_RATIO >= PREFIX_SAMPLE_ROWS
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
fn sort_live<O: OffsetSizeTrait>(
    arr: &GenericStringArray<O>,
    mut live: Vec<u32>,
    opts: SortOptions,
) -> Vec<u32> {
    if !prefix_discriminates(arr, &live) {
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
            let discriminating = prefix_discriminates(arr, &live);
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
}
