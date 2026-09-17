//! Engine-compatible digests for `Expr::Hash` — Spark's `hash`, Iceberg's bucket hash,
//! and Daft's default XXH3.
//!
//! Batcher's own digest (the parent module) is the one to reach for: it is cheaper, hashes
//! every type, and is pinned forever. These exist for one reason, which is that a job
//! ported from another engine often *stores* that engine's hash — a bucketed Iceberg table,
//! a surrogate key, a train/test split — and a port that silently recomputes it with a
//! different function moves every row to a different bucket without an error.
//!
//! Each arm therefore reproduces its engine to the bit, including the parts that are
//! accidents of history rather than design. The one worth knowing about is Spark's string
//! hash: `Murmur3_x86_32.hashUnsafeBytes` mixes each trailing byte as a *whole signed
//! word* instead of assembling the standard tail, so it disagrees with every other
//! Murmur3 on any string whose length is not a multiple of four. Iceberg uses the
//! standard tail. They are two functions, and they are kept as two here.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, Int64Array, Int64Builder};
use arrow::compute::cast;
use arrow::datatypes::{
    DataType, Date32Type, Decimal128Type, Float32Type, Float64Type, Int16Type, Int32Type,
    Int64Type, Int8Type, TimeUnit, TimestampMicrosecondType, UInt16Type, UInt32Type, UInt64Type,
    UInt8Type,
};

use crate::ExprError;

const C1: u32 = 0xcc9e_2d51;
const C2: u32 = 0x1b87_3593;

#[inline]
fn mix_k1(k: u32) -> u32 {
    k.wrapping_mul(C1).rotate_left(15).wrapping_mul(C2)
}

#[inline]
fn mix_h1(h: u32, k: u32) -> u32 {
    (h ^ k)
        .rotate_left(13)
        .wrapping_mul(5)
        .wrapping_add(0xe654_6b64)
}

#[inline]
fn fmix(mut h: u32, len: u32) -> u32 {
    h ^= len;
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^ (h >> 16)
}

/// Murmur3_x86_32 of a 4-byte integer (Spark/Guava `hashInt`).
#[inline]
fn hash_int(v: i32, seed: u32) -> u32 {
    fmix(mix_h1(seed, mix_k1(v as u32)), 4)
}

/// Murmur3_x86_32 of an 8-byte integer, low word first (Spark/Guava `hashLong`). This is
/// also the standard hash of the value's 8 little-endian bytes, which is how Iceberg
/// specifies it.
#[inline]
fn hash_long(v: i64, seed: u32) -> u32 {
    let h = mix_h1(seed, mix_k1(v as u32));
    fmix(mix_h1(h, mix_k1((v >> 32) as u32)), 8)
}

/// The aligned 4-byte blocks of `bytes`, folded from `seed`.
#[inline]
fn hash_blocks(bytes: &[u8], seed: u32) -> u32 {
    bytes.chunks_exact(4).fold(seed, |h, w| {
        mix_h1(h, mix_k1(u32::from_le_bytes([w[0], w[1], w[2], w[3]])))
    })
}

/// Spark's `hashUnsafeBytes`: each byte past the last aligned block is sign-extended and
/// mixed as a full block. Not the standard Murmur3 tail — see the module docs.
fn hash_bytes_spark(bytes: &[u8], seed: u32) -> u32 {
    let aligned = bytes.len() - bytes.len() % 4;
    let mut h = hash_blocks(&bytes[..aligned], seed);
    for &b in &bytes[aligned..] {
        h = mix_h1(h, mix_k1(i32::from(b as i8) as u32));
    }
    fmix(h, bytes.len() as u32)
}

/// Standard Murmur3_x86_32 of `bytes` (Guava `hashBytes`, Spark `hashUnsafeBytes2`).
fn hash_bytes_standard(bytes: &[u8], seed: u32) -> u32 {
    let aligned = bytes.len() - bytes.len() % 4;
    let mut h = hash_blocks(&bytes[..aligned], seed);
    let tail = &bytes[aligned..];
    if !tail.is_empty() {
        let k = tail
            .iter()
            .enumerate()
            .fold(0u32, |k, (i, &b)| k ^ (u32::from(b) << (8 * i)));
        h ^= mix_k1(k);
    }
    fmix(h, bytes.len() as u32)
}

/// `BigInteger.toByteArray` of `v`: the shortest big-endian two's-complement encoding.
/// Spark hashes a decimal wider than 18 digits, and Iceberg every decimal, through it.
fn minimal_be_bytes(v: i128) -> Vec<u8> {
    let bytes = v.to_be_bytes();
    let mut start = 0;
    while start < bytes.len() - 1 {
        let (b, next) = (bytes[start], bytes[start + 1]);
        let redundant = (b == 0x00 && next & 0x80 == 0) || (b == 0xff && next & 0x80 != 0);
        if !redundant {
            break;
        }
        start += 1;
    }
    bytes[start..].to_vec()
}

fn unsupported(algorithm: &'static str, dt: &DataType) -> ExprError {
    ExprError::ExpectedType {
        func: format!("hash(algorithm='{algorithm}')"),
        want: "a type that algorithm defines a hash for",
        got: dt.to_string(),
    }
}

/// Spark `hash(e0, e1, …)`: the running hash starts at `seed` and each input's value
/// replaces it with `computeHash(value, running)`; a null leaves it unchanged.
pub(crate) fn spark_murmur3(
    args: &[ArrayRef],
    seed: i64,
    rows: usize,
) -> Result<ArrayRef, ExprError> {
    // Spark's seed is an `Int`; the wider wire value keeps its low 32 bits.
    let mut acc = vec![seed as u32; rows];
    for arr in args {
        spark_fold(&mut acc, arr)?;
    }
    let out: Int64Array = acc.iter().map(|h| Some(i64::from(*h as i32))).collect();
    Ok(Arc::new(out))
}

/// Fold one column into the running Spark hashes, row by row.
fn spark_fold(acc: &mut [u32], arr: &ArrayRef) -> Result<(), ExprError> {
    macro_rules! each {
        ($values:expr, $hash:expr) => {{
            let a = $values;
            for (i, slot) in acc.iter_mut().enumerate() {
                if a.is_valid(i) {
                    *slot = $hash(a.value(i), *slot);
                }
            }
        }};
    }
    match arr.data_type() {
        DataType::Boolean => each!(arr.as_boolean(), |v: bool, h| hash_int(i32::from(v), h)),
        DataType::Int8 => each!(arr.as_primitive::<Int8Type>(), |v: i8, h| hash_int(
            i32::from(v),
            h
        )),
        DataType::Int16 => each!(arr.as_primitive::<Int16Type>(), |v: i16, h| hash_int(
            i32::from(v),
            h
        )),
        DataType::Int32 => each!(arr.as_primitive::<Int32Type>(), hash_int),
        DataType::UInt8 => each!(arr.as_primitive::<UInt8Type>(), |v: u8, h| hash_int(
            i32::from(v),
            h
        )),
        DataType::UInt16 => each!(arr.as_primitive::<UInt16Type>(), |v: u16, h| hash_int(
            i32::from(v),
            h
        )),
        DataType::Date32 => each!(arr.as_primitive::<Date32Type>(), hash_int),
        DataType::Int64 => each!(arr.as_primitive::<Int64Type>(), hash_long),
        DataType::UInt32 => each!(arr.as_primitive::<UInt32Type>(), |v: u32, h| hash_long(
            i64::from(v),
            h
        )),
        DataType::UInt64 => each!(arr.as_primitive::<UInt64Type>(), |v: u64, h| hash_long(
            v as i64, h
        )),
        // Java's `floatToIntBits`/`doubleToLongBits` canonicalize every NaN, and Spark
        // maps `-0.0` to the hash of integer 0 before looking at the bits.
        DataType::Float32 => each!(arr.as_primitive::<Float32Type>(), |v: f32, h| {
            let bits = if v == 0.0 {
                0
            } else if v.is_nan() {
                0x7fc0_0000
            } else {
                v.to_bits() as i32
            };
            hash_int(bits, h)
        }),
        DataType::Float64 => each!(arr.as_primitive::<Float64Type>(), |v: f64, h| {
            let bits = if v == 0.0 {
                0
            } else if v.is_nan() {
                0x7ff8_0000_0000_0000
            } else {
                v.to_bits() as i64
            };
            hash_long(bits, h)
        }),
        DataType::Utf8 | DataType::LargeUtf8 => {
            let s = cast(arr, &DataType::Utf8)?;
            each!(s.as_string::<i32>(), |v: &str, h| hash_bytes_spark(
                v.as_bytes(),
                h
            ))
        }
        DataType::Binary | DataType::LargeBinary => {
            let b = cast(arr, &DataType::Binary)?;
            each!(b.as_binary::<i32>(), hash_bytes_spark)
        }
        DataType::Timestamp(_, _) => {
            let micros = cast(arr, &DataType::Timestamp(TimeUnit::Microsecond, None))?;
            each!(micros.as_primitive::<TimestampMicrosecondType>(), hash_long)
        }
        DataType::Decimal128(precision, _) => {
            let wide = *precision > 18;
            each!(arr.as_primitive::<Decimal128Type>(), |v: i128, h| {
                if wide {
                    hash_bytes_spark(&minimal_be_bytes(v), h)
                } else {
                    hash_long(v as i64, h)
                }
            })
        }
        DataType::List(_) | DataType::LargeList(_) | DataType::FixedSizeList(_, _) => {
            spark_fold_list(acc, arr)?;
        }
        DataType::Struct(_) => {
            // Spark folds a struct's fields in order, skipping a null struct entirely.
            let st = arr.as_struct();
            let mut inner: Vec<u32> = acc.to_vec();
            for field in st.columns() {
                spark_fold(&mut inner, field)?;
            }
            for (i, slot) in acc.iter_mut().enumerate() {
                if st.is_valid(i) {
                    *slot = inner[i];
                }
            }
        }
        other => return Err(unsupported("murmur3", other)),
    }
    Ok(())
}

/// A list folds its elements in order into the running hash (Spark `ArrayData`).
fn spark_fold_list(acc: &mut [u32], arr: &ArrayRef) -> Result<(), ExprError> {
    let list = cast(arr, &DataType::List(list_field(arr)))?;
    let list = list.as_list::<i32>();
    let offsets = list.value_offsets();
    let child = list.values();
    for (i, slot) in acc.iter_mut().enumerate() {
        if list.is_null(i) {
            continue;
        }
        let (start, end) = (offsets[i] as usize, offsets[i + 1] as usize);
        let element = child.slice(start, end - start);
        for k in 0..element.len() {
            let mut one = [*slot];
            spark_fold(&mut one, &element.slice(k, 1))?;
            *slot = one[0];
        }
    }
    Ok(())
}

fn list_field(arr: &ArrayRef) -> arrow::datatypes::FieldRef {
    match arr.data_type() {
        DataType::List(f) | DataType::LargeList(f) | DataType::FixedSizeList(f, _) => Arc::clone(f),
        _ => unreachable!("called on a list type"),
    }
}

fn single_input<'a>(args: &'a [ArrayRef], algorithm: &str) -> Result<&'a ArrayRef, ExprError> {
    match args {
        [one] => Ok(one),
        _ => Err(ExprError::InvalidArgument {
            func: format!("hash(algorithm='{algorithm}')"),
            reason: format!("takes exactly one input, got {}", args.len()),
        }),
    }
}

/// The Iceberg bucket-transform hash of one input, null for a null value.
pub(crate) fn iceberg_murmur3(args: &[ArrayRef]) -> Result<ArrayRef, ExprError> {
    let arr = single_input(args, "iceberg")?;
    let mut out = Int64Builder::with_capacity(arr.len());
    macro_rules! each {
        ($values:expr, $hash:expr) => {{
            let a = $values;
            for i in 0..a.len() {
                out.append_option(a.is_valid(i).then(|| i64::from($hash(a.value(i)) as i32)));
            }
        }};
    }
    match arr.data_type() {
        dt if dt.is_integer() && !matches!(dt, DataType::UInt64) => {
            let wide = cast(arr, &DataType::Int64)?;
            each!(wide.as_primitive::<Int64Type>(), |v| hash_long(v, 0))
        }
        DataType::Date32 => each!(arr.as_primitive::<Date32Type>(), |v: i32| hash_long(
            i64::from(v),
            0
        )),
        DataType::Timestamp(_, _) => {
            let micros = cast(arr, &DataType::Timestamp(TimeUnit::Microsecond, None))?;
            each!(micros.as_primitive::<TimestampMicrosecondType>(), |v| {
                hash_long(v, 0)
            })
        }
        DataType::Utf8 | DataType::LargeUtf8 => {
            let s = cast(arr, &DataType::Utf8)?;
            each!(s.as_string::<i32>(), |v: &str| hash_bytes_standard(
                v.as_bytes(),
                0
            ))
        }
        DataType::Binary | DataType::LargeBinary => {
            let b = cast(arr, &DataType::Binary)?;
            each!(b.as_binary::<i32>(), |v: &[u8]| hash_bytes_standard(v, 0))
        }
        DataType::Decimal128(_, _) => each!(arr.as_primitive::<Decimal128Type>(), |v: i128| {
            hash_bytes_standard(&minimal_be_bytes(v), 0)
        }),
        other => return Err(unsupported("iceberg", other)),
    }
    Ok(Arc::new(out.finish()))
}

/// Daft's default `hash`: XXH3-64 of one input's value bytes with `seed`.
pub(crate) fn daft_xxh3(args: &[ArrayRef], seed: i64) -> Result<ArrayRef, ExprError> {
    let hash64_with_seed =
        |bytes: &[u8], seed: u64| twox_hash_2::XxHash3_64::oneshot_with_seed(seed, bytes);

    let arr = single_input(args, "xxhash3")?;
    let seed = seed as u64;
    // Daft hashes a null as an empty input. A null number or date takes seed 0 whatever the
    // seed; a null string or binary takes the seed, as its empty value would (measured on
    // Daft 0.7.25, `hash(seed=7)`).
    let null_number = hash64_with_seed(&[], 0) as i64;
    let null_bytes = hash64_with_seed(&[], seed) as i64;
    let mut out = Int64Builder::with_capacity(arr.len());
    macro_rules! each {
        ($values:expr, $bytes:expr) => {
            each!($values, $bytes, null_number)
        };
        ($values:expr, $bytes:expr, $null:expr) => {{
            let a = $values;
            for i in 0..a.len() {
                out.append_value(if a.is_valid(i) {
                    hash64_with_seed($bytes(a.value(i)).as_ref(), seed) as i64
                } else {
                    $null
                });
            }
        }};
    }
    match arr.data_type() {
        DataType::Int8 => each!(arr.as_primitive::<Int8Type>(), |v: i8| v.to_le_bytes()),
        DataType::Int16 => each!(arr.as_primitive::<Int16Type>(), |v: i16| v.to_le_bytes()),
        DataType::Int32 => each!(arr.as_primitive::<Int32Type>(), |v: i32| v.to_le_bytes()),
        DataType::Int64 => each!(arr.as_primitive::<Int64Type>(), |v: i64| v.to_le_bytes()),
        DataType::UInt8 => each!(arr.as_primitive::<UInt8Type>(), |v: u8| v.to_le_bytes()),
        DataType::UInt16 => each!(arr.as_primitive::<UInt16Type>(), |v: u16| v.to_le_bytes()),
        DataType::UInt32 => each!(arr.as_primitive::<UInt32Type>(), |v: u32| v.to_le_bytes()),
        DataType::UInt64 => each!(arr.as_primitive::<UInt64Type>(), |v: u64| v.to_le_bytes()),
        DataType::Float32 => each!(arr.as_primitive::<Float32Type>(), |v: f32| v.to_le_bytes()),
        DataType::Float64 => each!(arr.as_primitive::<Float64Type>(), |v: f64| v.to_le_bytes()),
        DataType::Date32 => each!(arr.as_primitive::<Date32Type>(), |v: i32| v.to_le_bytes()),
        DataType::Utf8 | DataType::LargeUtf8 => {
            let s = cast(arr, &DataType::Utf8)?;
            each!(
                s.as_string::<i32>(),
                |v: &str| v.as_bytes().to_vec(),
                null_bytes
            )
        }
        DataType::Binary | DataType::LargeBinary => {
            let b = cast(arr, &DataType::Binary)?;
            each!(b.as_binary::<i32>(), |v: &[u8]| v.to_vec(), null_bytes)
        }
        other => return Err(unsupported("xxhash3", other)),
    }
    Ok(Arc::new(out.finish()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{BooleanArray, Float64Array, Int32Array, StringArray};

    fn ints(a: &ArrayRef) -> Vec<Option<i64>> {
        let a = a.as_primitive::<Int64Type>();
        (0..a.len())
            .map(|i| a.is_valid(i).then(|| a.value(i)))
            .collect()
    }

    fn strs(values: Vec<Option<&str>>) -> ArrayRef {
        Arc::new(StringArray::from(values))
    }

    /// `pyspark/sql/functions/builtin.py`, the `hash` docstring: `hash('ABC')` is
    /// -757602832 and `hash('ABC', 'DEF')` is 599895104, both under Spark's seed 42.
    #[test]
    fn spark_documented_examples() {
        let abc = strs(vec![Some("ABC")]);
        let def = strs(vec![Some("DEF")]);
        let one = spark_murmur3(&[Arc::clone(&abc)], 42, 1).unwrap();
        assert_eq!(ints(&one), vec![Some(-757_602_832)]);
        let two = spark_murmur3(&[abc, def], 42, 1).unwrap();
        assert_eq!(ints(&two), vec![Some(599_895_104)]);
    }

    /// A null input leaves the running hash where it was: `hash(NULL)` is the seed.
    #[test]
    fn spark_null_is_the_seed() {
        let n = strs(vec![None]);
        assert_eq!(ints(&spark_murmur3(&[n], 42, 1).unwrap()), vec![Some(42)]);
    }

    /// Spark's `-0.0` and `0.0` hash alike, as do all NaNs, and an int and a long of the
    /// same value do not (4 bytes against 8).
    #[test]
    fn spark_float_and_width_rules() {
        let f: ArrayRef = Arc::new(Float64Array::from(vec![0.0, -0.0, f64::NAN, -f64::NAN]));
        let h = ints(&spark_murmur3(&[f], 42, 4).unwrap());
        assert_eq!(h[0], h[1]);
        assert_eq!(h[2], h[3]);
        let i32s: ArrayRef = Arc::new(Int32Array::from(vec![1]));
        let i64s: ArrayRef = Arc::new(Int64Array::from(vec![1]));
        assert_ne!(
            ints(&spark_murmur3(&[i32s], 42, 1).unwrap()),
            ints(&spark_murmur3(&[i64s], 42, 1).unwrap())
        );
    }

    /// The Iceberg spec's reference values (Appendix B, "32-bit Murmur3 hash"):
    /// int 34 and long 34 → 2017239379, string "iceberg" → 1210000089,
    /// date 2017-11-16 (17486) → -653330422, decimal 14.20 → -500754589.
    #[test]
    fn iceberg_spec_reference_values() {
        let i: ArrayRef = Arc::new(Int32Array::from(vec![34]));
        assert_eq!(
            ints(&iceberg_murmur3(&[i]).unwrap()),
            vec![Some(2_017_239_379)]
        );
        let s = strs(vec![Some("iceberg"), None]);
        assert_eq!(
            ints(&iceberg_murmur3(&[s]).unwrap()),
            vec![Some(1_210_000_089), None]
        );
        let d: ArrayRef = Arc::new(arrow::array::Date32Array::from(vec![17486]));
        assert_eq!(
            ints(&iceberg_murmur3(&[d]).unwrap()),
            vec![Some(-653_330_422)]
        );
        let dec: ArrayRef = Arc::new(
            arrow::array::Decimal128Array::from(vec![1420])
                .with_precision_and_scale(9, 2)
                .unwrap(),
        );
        assert_eq!(
            ints(&iceberg_murmur3(&[dec]).unwrap()),
            vec![Some(-500_754_589)]
        );
    }

    #[test]
    fn iceberg_declines_floats_and_several_inputs() {
        let f: ArrayRef = Arc::new(Float64Array::from(vec![1.0]));
        assert!(iceberg_murmur3(&[Arc::clone(&f)]).is_err());
        let i: ArrayRef = Arc::new(Int64Array::from(vec![1]));
        assert!(iceberg_murmur3(&[Arc::clone(&i), i]).is_err());
    }

    /// Daft 0.7.25 `col.hash()` (XXH3-64, seed 0), read back as the Int64 bits of its
    /// UInt64 result: 1 → 3439722301264460078, "ABC" → 2615927343983396622, 1.5 →
    /// 2693614958850487384, a null integer → 3244421341483603138 whatever the seed, a null
    /// string under `seed=7` → -7981820518085198152, and `hash(seed=7)` of 1 →
    /// 12012694768356207812. Daft's boolean encoding is not one
    /// these bytes reproduce, so a boolean is declined rather than guessed.
    #[test]
    fn daft_xxh3_reference_values() {
        let x: ArrayRef = Arc::new(Int64Array::from(vec![Some(1), None]));
        assert_eq!(
            ints(&daft_xxh3(&[Arc::clone(&x)], 0).unwrap()),
            vec![
                Some(3_439_722_301_264_460_078),
                Some(3_244_421_341_483_603_138)
            ]
        );
        assert_eq!(
            ints(&daft_xxh3(&[x], 7).unwrap()),
            vec![
                Some(12_012_694_768_356_207_812_u64 as i64),
                Some(3_244_421_341_483_603_138)
            ]
        );
        let s = strs(vec![Some("ABC")]);
        assert_eq!(
            ints(&daft_xxh3(&[s], 0).unwrap()),
            vec![Some(2_615_927_343_983_396_622)]
        );
        let n = strs(vec![None]);
        assert_eq!(
            ints(&daft_xxh3(&[n], 7).unwrap()),
            vec![Some(-7_981_820_518_085_198_152)]
        );
        let f: ArrayRef = Arc::new(Float64Array::from(vec![1.5]));
        assert_eq!(
            ints(&daft_xxh3(&[f], 0).unwrap()),
            vec![Some(2_693_614_958_850_487_384)]
        );
        let b: ArrayRef = Arc::new(BooleanArray::from(vec![true]));
        assert!(daft_xxh3(&[b], 0).is_err());
    }

    /// The wire shapes Python emits (`tests/unit/data/ir_snapshot_golden.json::hash_algorithm`
    /// and `::hash_default_seed`): a named digest deserializes to its arm, and an absent
    /// `algorithm` is Batcher's own, so every plan written before the field existed reads
    /// back unchanged.
    #[test]
    fn algorithm_wire_field_round_trips() {
        use crate::{Expr, HashAlgorithm};
        let named: Expr = serde_json::from_str(
            r#"{"e":"hash","inputs":[{"e":"col","name":"x"},{"e":"col","name":"y"}],"seed":42,"algorithm":"murmur3"}"#,
        )
        .unwrap();
        assert!(matches!(
            named,
            Expr::Hash {
                seed: 42,
                algorithm: HashAlgorithm::Murmur3,
                ..
            }
        ));
        let plain: Expr =
            serde_json::from_str(r#"{"e":"hash","inputs":[{"e":"col","name":"x"}]}"#).unwrap();
        assert!(matches!(
            plain,
            Expr::Hash {
                seed: 0,
                algorithm: HashAlgorithm::Batcher,
                ..
            }
        ));
        for tag in ["iceberg", "xxhash3"] {
            let json = format!(r#"{{"e":"hash","inputs":[],"algorithm":"{tag}"}}"#);
            assert!(serde_json::from_str::<Expr>(&json).is_ok(), "{tag}");
        }
    }

    /// The other tags this change put on the wire deserialize too
    /// (`ir_snapshot_golden.json::math2_round_even`, `::list_func_nulls_first`,
    /// `::list_func_with_nulls`, `::list_binary_jaccard_nonzero`).
    #[test]
    fn new_function_tags_deserialize() {
        use crate::Expr;
        let col = r#"{"e":"col","name":"a"}"#;
        let two = r#"{"e":"lit","value":{"int":2}}"#;
        let shapes = [
            format!(r#"{{"e":"math2","fn":"round_even","left":{col},"right":{two}}}"#),
            format!(r#"{{"e":"list","fn":"sort_nulls_first","input":{col}}}"#),
            format!(r#"{{"e":"list","fn":"sort_desc_nulls_first","input":{col}}}"#),
            format!(r#"{{"e":"list","fn":"unique_with_nulls","input":{col}}}"#),
            format!(r#"{{"e":"list","fn":"n_unique_with_nulls","input":{col}}}"#),
            format!(r#"{{"e":"list_binary","fn":"jaccard_nonzero","left":{col},"right":{col}}}"#),
        ];
        for json in &shapes {
            assert!(serde_json::from_str::<Expr>(json).is_ok(), "{json}");
        }
    }

    #[test]
    fn minimal_big_endian_encoding() {
        assert_eq!(minimal_be_bytes(0), vec![0]);
        assert_eq!(minimal_be_bytes(1420), vec![0x05, 0x8c]);
        assert_eq!(minimal_be_bytes(127), vec![0x7f]);
        assert_eq!(minimal_be_bytes(128), vec![0x00, 0x80]);
        assert_eq!(minimal_be_bytes(-1), vec![0xff]);
        assert_eq!(minimal_be_bytes(-129), vec![0xff, 0x7f]);
    }
}
