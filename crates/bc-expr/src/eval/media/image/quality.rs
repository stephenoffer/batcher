//! Image-curation measures: how bright an image is, and how sharp.
//!
//! A scraped image corpus is full of rows that decode perfectly and teach a model nothing —
//! blank placeholder tiles, all-white scans, out-of-focus photographs, and the grey box a CDN
//! serves when an asset is missing. None of them fail a decode, so nothing upstream catches
//! them, and a vision model trained on them learns the placeholder.
//!
//! Both measures reduce an image to one number, so filtering a corpus is an ordinary predicate
//! over a column rather than a per-file Python loop with an image library in it. Both work on
//! the luma (Rec. 601 grey) channel, so a colour cast does not read as detail.
//!
//! They are downsampled before measuring. That is deliberate for `sharpness`: full-resolution
//! sensor noise reads as high-frequency detail and makes a blurry 12-megapixel photograph score
//! like a sharp one, which is the failure mode a naive Laplacian variance has.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Builder, GenericBinaryArray, OffsetSizeTrait};

use super::super::map_rows;
use crate::ExprError;

/// The edge the two measures share: decode, downsample, and flatten to luma.
///
/// 128 on the long side keeps enough structure to tell focus from blur while bounding the cost
/// per image — a corpus mixing thumbnails and 50-megapixel scans otherwise spends all its time
/// on the scans. An image already smaller than this is left alone: enlarging it would invent no
/// detail and would give a one-pixel image an interior it does not have.
const MEASURE_SIDE: u32 = 128;

fn luma_plane(data: &[u8]) -> Option<(Vec<f64>, usize, usize)> {
    let img = image::load_from_memory(data).ok()?;
    // Only ever *down*scale. `resize` fits within the box in both directions, so it happily
    // enlarges a thumbnail to 128 on a side — which invents no detail, costs work, and turned
    // a 1x1 image into a 128x128 uniform field that reported a sharpness of 0 where it has no
    // interior pixel to measure at all.
    let small = if img.width() > MEASURE_SIDE || img.height() > MEASURE_SIDE {
        img.resize(
            MEASURE_SIDE,
            MEASURE_SIDE,
            image::imageops::FilterType::Triangle,
        )
    } else {
        img
    };
    let grey = small.into_luma8();
    let (w, h) = (grey.width() as usize, grey.height() as usize);
    if w == 0 || h == 0 {
        return None;
    }
    Some((grey.pixels().map(|p| p.0[0] as f64).collect(), w, h))
}

fn build(rows: Vec<Option<f64>>) -> ArrayRef {
    let mut builder = Float64Builder::with_capacity(rows.len());
    for row in rows {
        match row {
            Some(v) => builder.append_value(v),
            None => builder.append_null(),
        }
    }
    Arc::new(builder.finish())
}

/// `brightness()` → the mean luma of the image, normalized to `[0, 1]`.
///
/// The blank-image detector. A placeholder tile, a blown-out scan, and the grey box a CDN
/// serves for a missing asset all sit at an extreme, while a photograph of anything lands in
/// the middle. Filtering the two ends removes a class of row that decodes fine and carries no
/// signal.
pub(crate) fn brightness<O: OffsetSizeTrait>(
    bytes: &GenericBinaryArray<O>,
) -> Result<ArrayRef, ExprError> {
    let rows: Vec<Option<f64>> = map_rows(bytes.len(), |i| {
        if bytes.is_null(i) {
            return None;
        }
        let (luma, _, _) = luma_plane(bytes.value(i))?;
        Some(luma.iter().sum::<f64>() / luma.len() as f64 / 255.0)
    });
    Ok(build(rows))
}

/// `sharpness()` → the variance of the Laplacian of the luma plane, normalized to `[0, 1]`.
///
/// The standard focus measure. A sharp image has strong second derivatives at its edges and so
/// a high variance; a blurred or empty one has almost none. Values are small in absolute terms
/// — a well-focused photograph lands around 0.01 to 0.05 after normalization — so pick the
/// threshold from a histogram of your own corpus rather than from a remembered number.
///
/// It measures *detail*, not quality: a photograph of a brick wall scores higher than a
/// portrait, and a noisy image scores higher than a clean one. Use it to find the blurred tail
/// of a corpus, not to rank images against each other.
pub(crate) fn sharpness<O: OffsetSizeTrait>(
    bytes: &GenericBinaryArray<O>,
) -> Result<ArrayRef, ExprError> {
    let rows: Vec<Option<f64>> = map_rows(bytes.len(), |i| {
        if bytes.is_null(i) {
            return None;
        }
        let (luma, w, h) = luma_plane(bytes.value(i))?;
        // The 4-neighbour Laplacian needs an interior pixel, so anything smaller has no
        // second derivative to measure.
        if w < 3 || h < 3 {
            return None;
        }
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let mut n = 0u64;
        for row in 1..h - 1 {
            for col in 1..w - 1 {
                let at = |r: usize, c: usize| luma[r * w + c];
                let lap = at(row - 1, col) + at(row + 1, col) + at(row, col - 1) + at(row, col + 1)
                    - 4.0 * at(row, col);
                sum += lap;
                sum_sq += lap * lap;
                n += 1;
            }
        }
        let count = n as f64;
        let mean = sum / count;
        let variance = (sum_sq / count) - mean * mean;
        // The Laplacian of an 8-bit plane spans ±1020, so its variance is bounded by 1020².
        // Dividing by that puts the measure on the same [0, 1] scale as `brightness`, which
        // is what makes the two comparable in one filter.
        Some((variance.max(0.0) / (1020.0 * 1020.0)).min(1.0))
    });
    Ok(build(rows))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::BinaryArray;

    /// A PNG of a callable `(x, y) -> grey`.
    fn png_of(size: u32, f: impl Fn(u32, u32) -> u8) -> Vec<u8> {
        let img = image::GrayImage::from_fn(size, size, |x, y| image::Luma([f(x, y)]));
        let mut out = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageLuma8(img)
            .write_to(&mut out, image::ImageFormat::Png)
            .unwrap();
        out.into_inner()
    }

    fn values(out: &ArrayRef) -> Vec<Option<f64>> {
        use arrow::array::AsArray;
        use arrow::datatypes::Float64Type;
        let a = out.as_primitive::<Float64Type>();
        (0..a.len())
            .map(|i| (!a.is_null(i)).then(|| a.value(i)))
            .collect()
    }

    fn column(images: Vec<Option<Vec<u8>>>) -> GenericBinaryArray<i32> {
        BinaryArray::from_iter(images.iter().map(|o| o.as_deref()))
    }

    #[test]
    fn brightness_spans_black_to_white() {
        let arr = column(vec![
            Some(png_of(64, |_, _| 0)),
            Some(png_of(64, |_, _| 128)),
            Some(png_of(64, |_, _| 255)),
        ]);
        let got = values(&brightness(&arr).unwrap());
        assert!(got[0].unwrap() < 0.01);
        assert!((got[1].unwrap() - 0.5).abs() < 0.02);
        assert!(got[2].unwrap() > 0.99);
    }

    /// The reason the measure exists: a blank tile and a photograph must be separable.
    #[test]
    fn a_flat_image_has_no_sharpness_and_a_detailed_one_does() {
        let flat = column(vec![Some(png_of(64, |_, _| 128))]);
        let checker = column(vec![Some(png_of(64, |x, y| {
            if (x / 2 + y / 2) % 2 == 0 {
                0
            } else {
                255
            }
        }))]);
        let flat_score = values(&sharpness(&flat).unwrap())[0].unwrap();
        let detail_score = values(&sharpness(&checker).unwrap())[0].unwrap();
        assert!(flat_score < 1e-6);
        assert!(detail_score > flat_score * 100.0);
    }

    /// A gradient has a first derivative but almost no second one, which is what separates
    /// "smoothly varying" from "in focus".
    #[test]
    fn a_smooth_gradient_scores_far_below_an_edge() {
        let gradient = column(vec![Some(png_of(64, |x, _| (x * 4) as u8))]);
        let edge = column(vec![Some(png_of(64, |x, _| if x < 32 { 0 } else { 255 }))]);
        let smooth = values(&sharpness(&gradient).unwrap())[0].unwrap();
        let sharp = values(&sharpness(&edge).unwrap())[0].unwrap();
        assert!(sharp > smooth);
    }

    #[test]
    fn both_measures_stay_inside_the_unit_interval() {
        let arr = column(vec![Some(png_of(64, |x, y| {
            if (x + y) % 2 == 0 {
                0
            } else {
                255
            }
        }))]);
        for out in [brightness(&arr).unwrap(), sharpness(&arr).unwrap()] {
            let v = values(&out)[0].unwrap();
            assert!((0.0..=1.0).contains(&v), "out of range: {v}");
        }
    }

    #[test]
    fn a_null_or_undecodable_row_is_null_rather_than_an_error() {
        let arr = column(vec![None, Some(b"not an image".to_vec())]);
        for out in [brightness(&arr).unwrap(), sharpness(&arr).unwrap()] {
            assert_eq!(values(&out), vec![None, None]);
        }
    }
}
