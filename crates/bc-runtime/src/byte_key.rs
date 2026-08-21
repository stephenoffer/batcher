//! The one reading of a **byte-lexicographic** key column: `Utf8`, `LargeUtf8`, `Binary`,
//! `LargeBinary` and `FixedSizeBinary`.
//!
//! Arrow orders all five the same way — `memcmp` on the value bytes, a shorter value that is a
//! prefix of a longer one sorting first — so every part of the engine that *orders* by such a
//! key is doing one thing, and this is where it reads the bytes. `keys.rs` is the sibling for
//! *grouping* key identity; this is its ordering counterpart, and it lives here for the same
//! reason: the range partitioner (`shuffle.rs`) and the sort (`bc_interp::ops::byte_sort`) must
//! agree about what a key is, and the only way to guarantee that is for there to be one answer.
//!
//! The two spellings exist because the two callers need different things from the same trait.
//! [`ByteKeys`] is generic, so a sort that compares a key `n log n` times monomorphizes and
//! pays nothing; [`ByteKeyColumn`] is a type-erased enum over the same trait, so a router that
//! reads each row once can dispatch without a generic parameter threading through its callers.
//! Both are the same five arms and the same `value(i)`.

use arrow::array::{Array, ArrayRef, FixedSizeBinaryArray, GenericByteArray};
use arrow::buffer::NullBuffer;
use arrow::datatypes::{
    ArrowNativeType, BinaryType, ByteArrayType, DataType, LargeBinaryType, LargeUtf8Type, Utf8Type,
};

/// A column of byte-lexicographic sort keys, whatever Arrow type spells it.
///
/// The one thing that varies between `Utf8`, `Binary` and `FixedSizeBinary` is how a row's
/// bytes are reached; everything else a caller needs, `Array` already provides.
pub trait ByteKeys {
    /// Rows in the column, including nulls.
    fn len(&self) -> usize;

    /// Whether row `i` is null.
    fn is_null(&self, i: usize) -> bool;

    /// The column's null buffer, for a caller that wants to test validity in bulk.
    fn null_buffer(&self) -> Option<&NullBuffer>;

    /// Row `i`'s key bytes. Only meaningful for a non-null row.
    fn key(&self, i: usize) -> &[u8];

    /// The width (in bytes, at most `max`) at which a right-zero-padded big-endian integer pack
    /// orders this column **exactly**, or `None` when no such width exists.
    ///
    /// Zero-padding is order-preserving because a value that runs out is a prefix of one that
    /// does not, no byte sorts below `0`, and a prefix sorts first. The one shape that breaks it
    /// is a literal `0` byte in a column of **mixed** lengths — `b"ab"` and `b"ab\0"` pad to the
    /// same eight bytes but are genuinely unequal, and a pack would call them tied.
    ///
    /// So exactness is: uniform length (nothing is padded), or no `0` byte anywhere in the value
    /// buffer (nothing a pad can collide with). `FixedSizeBinary` is uniform by construction,
    /// which is why a *random* fixed-width key — full of `0` bytes — still qualifies.
    fn exact_pack_width(&self, max: usize) -> Option<usize>;

    /// Whether the column holds no rows.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Whether `dt` is a byte-lexicographic key type — the single list every caller reads.
///
/// Stated once because it was previously stated four times and they disagreed: the stable sort
/// accepted two of the five, the parallel sample-sort read that same short list and so declined
/// the rest to a serial sort, and the distributed range partitioner read a third copy. A type
/// added to one and not the others is not a compile error anywhere.
pub fn is_byte_key(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Binary
            | DataType::LargeBinary
            | DataType::FixedSizeBinary(_)
    )
}

impl<T: ByteArrayType> ByteKeys for GenericByteArray<T> {
    fn len(&self) -> usize {
        Array::len(self)
    }

    fn is_null(&self, i: usize) -> bool {
        Array::is_null(self, i)
    }

    fn null_buffer(&self) -> Option<&NullBuffer> {
        Array::nulls(self)
    }

    fn key(&self, i: usize) -> &[u8] {
        self.value(i).as_ref()
    }

    fn exact_pack_width(&self, max: usize) -> Option<usize> {
        let (mut min, mut longest) = (usize::MAX, 0usize);
        let nulls = Array::nulls(self);
        for i in 0..Array::len(self) {
            if nulls.is_some_and(|nb| nb.is_null(i)) {
                continue;
            }
            let len = self.value_length(i).as_usize();
            // Bail on the first over-wide value rather than measuring the whole column and
            // then declining. A caller asks this *before* every sort of a byte key, so the
            // cost of the "no" is paid by every wide column in the engine — and measured, a
            // full offset pass per range was a fifth of the sort on a 12-byte string key.
            if len > max {
                return None;
            }
            min = min.min(len);
            longest = longest.max(len);
        }
        // `min > longest` only when every row is null, and an all-null column is trivially exact
        // at any width — the caller partitions nulls out before it packs anything.
        if min >= longest || !self.value_data().contains(&0) {
            Some(longest)
        } else {
            None
        }
    }
}

impl ByteKeys for FixedSizeBinaryArray {
    fn len(&self) -> usize {
        Array::len(self)
    }

    fn is_null(&self, i: usize) -> bool {
        Array::is_null(self, i)
    }

    fn null_buffer(&self) -> Option<&NullBuffer> {
        Array::nulls(self)
    }

    fn key(&self, i: usize) -> &[u8] {
        self.value(i)
    }

    fn exact_pack_width(&self, max: usize) -> Option<usize> {
        // Uniform width: every value pads identically, so no `0` byte can collide two distinct
        // values. This is the type a fixed-layout record key arrives as.
        let width = self.value_length() as usize;
        (width <= max).then_some(width)
    }
}

/// A byte-key column with its Arrow type resolved once, for a caller that cannot be generic.
///
/// Built by [`ByteKeyColumn::new`], which is also the type test: `None` means the column is not
/// a byte key and the caller must decline. Every method dispatches over the five arms, which is
/// one predictable branch per row — negligible against the binary search or the gather that a
/// per-row caller is doing around it, and the reason the *sort* uses the generic trait instead.
pub enum ByteKeyColumn<'a> {
    /// A `Utf8` column.
    Utf8(&'a GenericByteArray<Utf8Type>),
    /// A `LargeUtf8` column.
    LargeUtf8(&'a GenericByteArray<LargeUtf8Type>),
    /// A `Binary` column.
    Binary(&'a GenericByteArray<BinaryType>),
    /// A `LargeBinary` column.
    LargeBinary(&'a GenericByteArray<LargeBinaryType>),
    /// A `FixedSizeBinary` column of any width.
    FixedSizeBinary(&'a FixedSizeBinaryArray),
}

/// Apply `$method` with `$args` to whichever arm `$self` is.
macro_rules! on_arm {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            ByteKeyColumn::Utf8(a) => ByteKeys::$method(*a $(, $arg)*),
            ByteKeyColumn::LargeUtf8(a) => ByteKeys::$method(*a $(, $arg)*),
            ByteKeyColumn::Binary(a) => ByteKeys::$method(*a $(, $arg)*),
            ByteKeyColumn::LargeBinary(a) => ByteKeys::$method(*a $(, $arg)*),
            ByteKeyColumn::FixedSizeBinary(a) => ByteKeys::$method(*a $(, $arg)*),
        }
    };
}

impl<'a> ByteKeyColumn<'a> {
    /// The byte-key reading of `values`, or `None` if it is not a byte-key column.
    pub fn new(values: &'a ArrayRef) -> Option<Self> {
        let any = values.as_any();
        Some(match values.data_type() {
            DataType::Utf8 => Self::Utf8(any.downcast_ref()?),
            DataType::LargeUtf8 => Self::LargeUtf8(any.downcast_ref()?),
            DataType::Binary => Self::Binary(any.downcast_ref()?),
            DataType::LargeBinary => Self::LargeBinary(any.downcast_ref()?),
            DataType::FixedSizeBinary(_) => Self::FixedSizeBinary(any.downcast_ref()?),
            _ => return None,
        })
    }
}

impl ByteKeys for ByteKeyColumn<'_> {
    fn len(&self) -> usize {
        on_arm!(self, len)
    }

    fn is_null(&self, i: usize) -> bool {
        on_arm!(self, is_null, i)
    }

    fn null_buffer(&self) -> Option<&NullBuffer> {
        on_arm!(self, null_buffer)
    }

    fn key(&self, i: usize) -> &[u8] {
        on_arm!(self, key, i)
    }

    fn exact_pack_width(&self, max: usize) -> Option<usize> {
        on_arm!(self, exact_pack_width, max)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{
        BinaryArray, FixedSizeBinaryArray, LargeBinaryArray, LargeStringArray, StringArray,
    };

    use super::*;

    /// Every spelling of a byte key must read back the same bytes, the same nulls and the same
    /// length. This is the contract the sort and the range partitioner both rest on, and it is
    /// the one that used to be four separate type lists.
    #[test]
    fn every_byte_key_type_reads_the_same_bytes() {
        let values: Vec<Option<&[u8]>> = vec![Some(b"ab"), None, Some(b"cd")];
        let text: Vec<Option<&str>> = vec![Some("ab"), None, Some("cd")];
        let columns: Vec<(&str, ArrayRef)> = vec![
            ("utf8", Arc::new(StringArray::from(text.clone()))),
            ("large_utf8", Arc::new(LargeStringArray::from(text))),
            ("binary", Arc::new(BinaryArray::from(values.clone()))),
            (
                "large_binary",
                Arc::new(LargeBinaryArray::from(values.clone())),
            ),
            (
                "fixed_size_binary",
                Arc::new(
                    FixedSizeBinaryArray::try_from_sparse_iter_with_size(
                        values.iter().map(|v| v.map(|b| b.to_vec())),
                        2,
                    )
                    .expect("uniform width"),
                ),
            ),
        ];
        for (name, column) in columns {
            assert!(is_byte_key(column.data_type()), "{name}");
            let keys = ByteKeyColumn::new(&column).unwrap_or_else(|| panic!("{name}"));
            assert_eq!(keys.len(), 3, "{name}");
            assert_eq!(keys.key(0), b"ab", "{name}");
            assert!(keys.is_null(1), "{name}");
            assert_eq!(keys.key(2), b"cd", "{name}");
            assert_eq!(keys.exact_pack_width(8), Some(2), "{name}");
        }
    }

    /// A type that is not a byte key must be refused rather than misread, because every caller
    /// treats `None` as "decline to my slower path" and a wrong `Some` would sort by the wrong
    /// bytes entirely.
    #[test]
    fn a_non_byte_key_column_is_refused() {
        let column: ArrayRef = Arc::new(arrow::array::Int64Array::from(vec![1i64, 2]));
        assert!(!is_byte_key(column.data_type()));
        assert!(ByteKeyColumn::new(&column).is_none());
    }

    /// The pack width must decline exactly where padding could lie — a mixed-length column
    /// holding a `0` byte — and accept where it cannot.
    #[test]
    fn the_pack_width_declines_a_padding_collision() {
        let collides: ArrayRef = Arc::new(BinaryArray::from(vec![
            Some(b"ab".as_ref()),
            Some(b"ab\0".as_ref()),
        ]));
        assert_eq!(
            ByteKeyColumn::new(&collides).unwrap().exact_pack_width(8),
            None
        );

        let clean: ArrayRef = Arc::new(BinaryArray::from(vec![
            Some(b"ab".as_ref()),
            Some(b"abc".as_ref()),
        ]));
        assert_eq!(
            ByteKeyColumn::new(&clean).unwrap().exact_pack_width(8),
            Some(3)
        );

        let wide: ArrayRef = Arc::new(BinaryArray::from(vec![Some(b"0123456789".as_ref())]));
        assert_eq!(ByteKeyColumn::new(&wide).unwrap().exact_pack_width(8), None);
    }
}
