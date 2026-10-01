//! Whole-column string kernels for the case where every byte a column holds is ASCII.
//!
//! The per-row string functions build a `&str` per row and hand it to a closure — which is the
//! right shape for Unicode, and three costs too many for ASCII, the overwhelmingly common data:
//!
//! * `UPPER`/`LOWER` allocated a `String` per row and copied it into a builder. Over TPC-H's
//!   6M `l_shipmode` values that was 28% of `GROUP BY UPPER(l_shipmode)` in `map_str` and
//!   another ~20% in the allocator (`mi_malloc`/`mi_free`/`_mi_page_retire`).
//! * `SUBSTRING` counted characters (`chars().count()`, 15%) and walked `char_indices` per row
//!   to find byte boundaries the offsets already give when a character is a byte.
//! * `LENGTH` collected through an `Option` iterator with a validity test per row.
//!
//! One test over the column's own bytes — a vectorized OR-fold — decides that a character is
//! a byte for every row at once. After it, each function is offset arithmetic plus at most one
//! linear pass over the bytes, and its answer is the one the per-row path gives *by definition*:
//! ASCII has one byte per character, and no ASCII character has a non-ASCII or multi-character
//! case mapping. A column with any non-ASCII byte gets `None` and keeps the per-row path, so
//! nothing about Unicode handling changes here.
//!
//! Every kernel carries the input's validity buffer through unchanged. The value computed under
//! a null slot is whatever its (normally empty) offsets span, and is never observable.

use arrow::array::{Array, Int64Array, StringArray};
use arrow::buffer::{Buffer, OffsetBuffer, ScalarBuffer};

/// This array's own value bytes — not the shared buffer's, which a sliced morsel would re-scan
/// whole — and the offset its first row starts at, when every one of those bytes is ASCII.
pub(super) fn ascii_values(s: &StringArray) -> Option<(&[u8], i32)> {
    let offsets = s.value_offsets();
    let (lo, hi) = (*offsets.first()?, *offsets.last()?);
    let bytes = &s.value_data()[lo as usize..hi as usize];
    is_ascii(bytes).then_some((bytes, lo))
}

/// Whether every byte is below 0x80, as an OR-fold the compiler vectorizes.
///
/// `<[u8]>::is_ascii` stops at the first high byte, which costs it the vector loop; real
/// columns are ASCII throughout, so the fold reads 256 bytes per branch instead.
fn is_ascii(bytes: &[u8]) -> bool {
    let (chunks, rest) = bytes.as_chunks::<256>();
    for chunk in chunks {
        if chunk.iter().fold(0u8, |acc, &b| acc | b) >= 0x80 {
            return false;
        }
    }
    rest.is_ascii()
}

/// `LENGTH` of an ASCII column: each row's byte width, read off the offsets.
pub(super) fn len(s: &StringArray) -> Option<Int64Array> {
    ascii_values(s)?;
    let lens: Vec<i64> = s
        .value_offsets()
        .windows(2)
        .map(|w| i64::from(w[1] - w[0]))
        .collect();
    Some(Int64Array::new(lens.into(), s.nulls().cloned()))
}

/// `UPPER`/`LOWER` of an ASCII column: one pass over its bytes, the offsets kept as they are.
pub(super) fn case_map(s: &StringArray, upper: bool) -> Option<StringArray> {
    let (bytes, base) = ascii_values(s)?;
    let mapped: Vec<u8> = if upper {
        bytes.iter().map(u8::to_ascii_uppercase).collect()
    } else {
        bytes.iter().map(u8::to_ascii_lowercase).collect()
    };
    let offsets = if base == 0 {
        s.offsets().clone()
    } else {
        rebase(s.value_offsets(), base)
    };
    // `new` re-validates the bytes as UTF-8, which for ASCII is the cheap fast path of
    // `from_utf8`; keeping it means this module needs no `unsafe`.
    Some(StringArray::new(
        offsets,
        Buffer::from_vec(mapped),
        s.nulls().cloned(),
    ))
}

/// `SUBSTRING(s, start, length)` of an ASCII column, with `window` the character window the
/// per-row path computes for a row of `n` characters — here `n` bytes — as a half-open range.
pub(super) fn substr(
    s: &StringArray,
    window: impl Fn(i64) -> (usize, usize),
) -> Option<StringArray> {
    let (bytes, base) = ascii_values(s)?;
    let offsets = s.value_offsets();
    let mut out_offsets = Vec::with_capacity(offsets.len());
    out_offsets.push(0i32);
    let mut out = Vec::with_capacity(bytes.len().min(s.len() * 16));
    for w in offsets.windows(2) {
        let (start, end) = ((w[0] - base) as usize, (w[1] - base) as usize);
        let (lo, hi) = window((end - start) as i64);
        out.extend_from_slice(&bytes[start + lo..start + hi]);
        out_offsets.push(i32::try_from(out.len()).ok()?);
    }
    Some(StringArray::new(
        OffsetBuffer::new(ScalarBuffer::from(out_offsets)),
        Buffer::from_vec(out),
        s.nulls().cloned(),
    ))
}

/// `offsets` shifted so the first row starts at zero.
fn rebase(offsets: &[i32], base: i32) -> OffsetBuffer<i32> {
    OffsetBuffer::new(ScalarBuffer::from(
        offsets.iter().map(|&o| o - base).collect::<Vec<_>>(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arr(values: Vec<Option<&str>>) -> StringArray {
        StringArray::from(values)
    }

    #[test]
    fn the_fold_finds_a_high_byte_in_any_position() {
        let mut bytes = vec![b'a'; 1000];
        assert!(is_ascii(&bytes));
        for at in [0, 255, 256, 511, 999] {
            bytes[at] = 0xC3;
            assert!(!is_ascii(&bytes), "missed a high byte at {at}");
            bytes[at] = b'a';
        }
        assert!(is_ascii(&[]));
    }

    #[test]
    fn a_non_ascii_column_is_declined_by_every_kernel() {
        let a = arr(vec![Some("abc"), Some("héllo")]);
        assert!(len(&a).is_none());
        assert!(case_map(&a, true).is_none());
        assert!(substr(&a, |n| (0, n as usize)).is_none());
    }

    /// Every window `substr` can be asked for — negative, zero, past-the-end starts; negative,
    /// zero, absent and huge lengths — held to the per-row path on the same ASCII rows.
    #[test]
    fn substr_matches_the_per_row_path_on_every_window() {
        let rows = vec![
            Some(""),
            Some("a"),
            Some("abcdef"),
            None,
            Some("hello world, longer than sixteen"),
        ];
        let a = arr(rows.clone());
        let starts = [i64::MIN, -40, -7, -6, -2, -1, 0, 1, 2, 6, 7, 40, i64::MAX];
        let lengths = [
            None,
            Some(i64::MIN),
            Some(-3),
            Some(-1),
            Some(0),
            Some(1),
            Some(4),
            Some(100),
            Some(i64::MAX),
        ];
        for start in starts {
            for length in lengths {
                let got = substr(&a, |n| super::super::substr_window(n, start, length)).unwrap();
                let want: Vec<Option<&str>> = rows
                    .iter()
                    .map(|r| r.map(|v| super::super::substr_slice(v, start, length)))
                    .collect();
                assert_eq!(
                    got.iter().collect::<Vec<_>>(),
                    want,
                    "start={start} length={length:?}"
                );
            }
        }
    }

    #[test]
    fn case_maps_match_the_unicode_mapping_on_ascii() {
        let all: String = (0u8..128).map(char::from).collect();
        let rows = vec![Some(all.as_str()), None, Some(""), Some("MiXeD 123 _z@[`{")];
        let a = arr(rows.clone());
        for upper in [true, false] {
            let got = case_map(&a, upper).unwrap();
            let want: Vec<Option<String>> = rows
                .iter()
                .map(|r| {
                    r.map(|v| {
                        if upper {
                            v.to_uppercase()
                        } else {
                            v.to_lowercase()
                        }
                    })
                })
                .collect();
            let got: Vec<Option<String>> = got.iter().map(|o| o.map(str::to_owned)).collect();
            assert_eq!(got, want, "upper={upper}");
        }
    }

    #[test]
    fn a_sliced_column_reads_only_its_own_bytes() {
        // The non-ASCII value sits outside the slice, so the slice is ASCII and served.
        let a = arr(vec![Some("日本"), Some("ab"), None, Some("cde")]);
        let sliced = a.slice(1, 3);
        assert_eq!(
            len(&sliced).unwrap().iter().collect::<Vec<_>>(),
            vec![Some(2), None, Some(3)]
        );
        let up = case_map(&sliced, true).unwrap();
        assert_eq!(
            up.iter().collect::<Vec<_>>(),
            vec![Some("AB"), None, Some("CDE")]
        );
        let sub = substr(&sliced, |n| (1.min(n as usize), n as usize)).unwrap();
        assert_eq!(
            sub.iter().collect::<Vec<_>>(),
            vec![Some("b"), None, Some("de")]
        );
    }
}
