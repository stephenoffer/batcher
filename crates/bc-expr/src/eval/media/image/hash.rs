//! Perceptual hashes: the fingerprints that make image near-duplicate detection a join.
//!
//! A scraped corpus is full of the same picture at three resolutions, two codecs and one
//! watermark. None of that is findable by content hash — every byte differs — and finding
//! it with a model costs an embedding per image. A perceptual hash costs a 32x32 decode
//! and reduces the whole question to `bit_count(a ^ b) < k`, which the engine already
//! evaluates as an ordinary predicate over an `Int64` column.
//!
//! Three of them exist here because they trade the same way every fingerprint family does.
//! [`super::dhash`] compares horizontally adjacent pixels, [`ahash`] thresholds at the
//! mean, and [`phash`] keeps the low-frequency DCT coefficients. `ahash` is the cheapest
//! and the least discriminating, `phash` the most robust to rescaling and re-encoding, and
//! `dhash` sits between them. A dedup pass usually spends `ahash` or `dhash` as a blocking
//! key and `phash` to confirm.
//!
//! All three return **`Int64`, reinterpreted rather than clamped**, for the reason spelled
//! out on `dhash`: the FFI boundary rejects a `u64` above `i64::MAX`, and XOR/popcount read
//! the same bits either way.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, GenericBinaryArray, Int64Array, OffsetSizeTrait};

use super::{decode_rgb_resized, map_rows, rec601};
use crate::ExprError;

/// The side of the reduced image whose bits become the hash. 8x8 = 64 bits, which is what
/// makes the digest an `Int64` and the distance one `bit_count`.
const HASH_SIDE: u32 = 8;

/// The side `phash` reduces to before the DCT. 32 so that the 8x8 block kept afterwards is
/// genuinely the low quarter of each frequency axis; reducing straight to 8x8 would make
/// the DCT a no-op on a signal that has already lost the frequencies it was meant to rank.
const DCT_SIDE: u32 = 32;

/// Decode, resize to `(side, side)`, and reduce to one luma plane of `side * side` values.
fn luma_square(data: &[u8], side: u32) -> Option<Vec<f64>> {
    let rgb = decode_rgb_resized(data, side, side)?;
    let want = (side * side * 3) as usize;
    if rgb.len() != want {
        return None;
    }
    Some(
        rgb.chunks_exact(3)
            .map(|p| f64::from(rec601([p[0], p[1], p[2]])))
            .collect(),
    )
}

/// Pack 64 comparisons against `threshold` into one `Int64`, most-significant bit first.
fn pack(values: &[f64], threshold: f64) -> i64 {
    let mut bits: u64 = 0;
    for &v in values {
        bits = (bits << 1) | u64::from(v > threshold);
    }
    bits as i64
}

fn build(hashes: Vec<Option<i64>>) -> ArrayRef {
    Arc::new(Int64Array::from(hashes))
}

/// `ahash()` → the 64-bit average hash: an 8x8 luma reduction thresholded at its own mean.
pub(super) fn ahash<O: OffsetSizeTrait>(
    bytes: &GenericBinaryArray<O>,
) -> Result<ArrayRef, ExprError> {
    let hashes: Vec<Option<i64>> = map_rows(bytes.len(), |i| {
        if bytes.is_null(i) {
            return None;
        }
        let luma = luma_square(bytes.value(i), HASH_SIDE)?;
        let mean = luma.iter().sum::<f64>() / luma.len() as f64;
        Some(pack(&luma, mean))
    });
    Ok(build(hashes))
}

/// `phash()` → the 64-bit DCT perceptual hash (the standard pHash).
///
/// Reduce to 32x32 luma, take a 2-D DCT-II, keep the top-left 8x8 block of coefficients
/// (the lowest frequencies, which is where a picture's identity lives and where rescaling
/// and re-encoding do not reach), and threshold each against the block's **median**.
///
/// The DC coefficient is excluded from the median. It carries the image's mean brightness,
/// which is an order of magnitude larger than every other coefficient, so leaving it in
/// drags the median far above the rest and makes nearly all 64 bits zero — a hash that
/// collides on everything. This is the classic implementation detail of pHash and the one
/// a naive restatement gets wrong.
pub(super) fn phash<O: OffsetSizeTrait>(
    bytes: &GenericBinaryArray<O>,
) -> Result<ArrayRef, ExprError> {
    let cos = dct_basis(DCT_SIDE as usize);
    let hashes: Vec<Option<i64>> = map_rows(bytes.len(), |i| {
        if bytes.is_null(i) {
            return None;
        }
        let luma = luma_square(bytes.value(i), DCT_SIDE)?;
        let coeffs = dct_2d(&luma, DCT_SIDE as usize, &cos);
        let side = HASH_SIDE as usize;
        let n = DCT_SIDE as usize;
        let block: Vec<f64> = (0..side)
            .flat_map(|r| (0..side).map(move |c| (r, c)))
            .map(|(r, c)| coeffs[r * n + c])
            .collect();
        let mut rest: Vec<f64> = block[1..].to_vec();
        rest.sort_by(f64::total_cmp);
        let median = rest[rest.len() / 2];
        Some(pack(&block, median))
    });
    Ok(build(hashes))
}

/// The `cos((2x+1) k pi / 2n)` table a DCT-II of size `n` reads, indexed `k * n + x`.
///
/// Precomputed once per batch rather than per image: the table is 1,024 doubles for the
/// 32-wide transform and every row of the column reads exactly the same one, so building
/// it per row would spend more time in `cos` than in the transform itself.
fn dct_basis(n: usize) -> Vec<f64> {
    let mut table = vec![0.0; n * n];
    for k in 0..n {
        for x in 0..n {
            table[k * n + x] =
                (((2 * x + 1) as f64) * (k as f64) * std::f64::consts::PI / (2.0 * n as f64)).cos();
        }
    }
    table
}

/// A separable 2-D DCT-II over an `n * n` plane, rows then columns.
///
/// Separable rather than a direct 4-D sum: `O(n^3)` against `O(n^4)`, which at n=32 is 32x
/// less arithmetic per image. Unnormalized — only the *ordering* of the coefficients
/// against their median matters to the hash, and a constant scale cannot change it.
fn dct_2d(plane: &[f64], n: usize, cos: &[f64]) -> Vec<f64> {
    let mut rows = vec![0.0; n * n];
    for r in 0..n {
        for k in 0..n {
            let mut acc = 0.0;
            for x in 0..n {
                acc += plane[r * n + x] * cos[k * n + x];
            }
            rows[r * n + k] = acc;
        }
    }
    let mut out = vec![0.0; n * n];
    for c in 0..n {
        for k in 0..n {
            let mut acc = 0.0;
            for y in 0..n {
                acc += rows[y * n + c] * cos[k * n + y];
            }
            out[k * n + c] = acc;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{AsArray, BinaryArray};
    use arrow::datatypes::Int64Type;

    fn png(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 3]) -> Vec<u8> {
        let img = image::RgbImage::from_fn(w, h, |x, y| image::Rgb(f(x, y)));
        let mut out = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut out, image::ImageFormat::Png)
            .unwrap();
        out.into_inner()
    }

    fn hashes(f: fn(&BinaryArray) -> Result<ArrayRef, ExprError>, imgs: &[Vec<u8>]) -> Vec<i64> {
        let arr = BinaryArray::from_iter(imgs.iter().map(|b| Some(b.as_slice())));
        let out = f(&arr).unwrap();
        let a = out.as_primitive::<Int64Type>();
        (0..a.len()).map(|i| a.value(i)).collect()
    }

    fn distance(a: i64, b: i64) -> u32 {
        (a ^ b).count_ones()
    }

    /// The property both hashes exist for: rescaling must not move the fingerprint.
    #[test]
    fn a_rescaled_image_keeps_almost_every_bit() {
        let scene = |scale: u32| {
            png(64 * scale, 64 * scale, |x, y| {
                let v = ((x / scale) * 4 % 256) as u8;
                [v, (((y / scale) * 4) % 256) as u8, 128]
            })
        };
        for f in [phash as fn(&BinaryArray) -> _, ahash] {
            let got = hashes(f, &[scene(1), scene(4)]);
            assert!(
                distance(got[0], got[1]) <= 6,
                "rescaling moved {} bits",
                distance(got[0], got[1])
            );
        }
    }

    /// And the property that makes it useful: different pictures must be far apart.
    #[test]
    fn different_images_are_far_apart() {
        let stripes = png(64, 64, |x, _| {
            let v = if (x / 4) % 2 == 0 { 0 } else { 255 };
            [v, v, v]
        });
        let gradient = png(64, 64, |_, y| [(y * 4) as u8, 0, 0]);
        for f in [phash as fn(&BinaryArray) -> _, ahash] {
            let got = hashes(f, &[stripes.clone(), gradient.clone()]);
            assert!(
                distance(got[0], got[1]) >= 10,
                "distinct images only {} bits apart",
                distance(got[0], got[1])
            );
        }
    }

    /// The DC-coefficient trap: a hash that thresholds against a median including DC
    /// comes out nearly all-zero, and every image collides.
    #[test]
    fn phash_uses_most_of_its_bits() {
        let img = png(64, 64, |x, y| {
            [(x * 4) as u8, (y * 4) as u8, ((x + y) * 2) as u8]
        });
        let bits = hashes(phash, &[img])[0].count_ones();
        assert!((8..=56).contains(&bits), "degenerate hash: {bits} bits set");
    }

    #[test]
    fn a_null_or_undecodable_row_is_null() {
        let arr = BinaryArray::from_iter(vec![None, Some(b"not an image".as_slice())]);
        for f in [phash as fn(&BinaryArray) -> _, ahash] {
            let out = f(&arr).unwrap();
            assert_eq!(out.null_count(), 2);
        }
    }
}
