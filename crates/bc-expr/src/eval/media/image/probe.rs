//! Header-only facts: what an image is, without decoding a pixel of it.
//!
//! Everything here costs one header read — bytes the decoder was going to touch anyway —
//! where the answer would otherwise cost a full decode or, more often, not be asked at
//! all. That is the whole point. A curation query wants to know which files are really
//! PNGs wearing a `.jpg` extension, which carry an alpha channel a 3-channel model will
//! silently mangle, and which are portrait; paying a megapixel decode per row to learn any
//! of them is what makes people skip the check and find out later.
//!
//! `decode()` already returns width/height/channels/mode from one such read. These are the
//! facts it does not carry, kept as separate ops rather than widened into that struct
//! because each is a scalar a predicate reads directly.

use std::io::Cursor;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanArray, Float64Array, GenericBinaryArray, OffsetSizeTrait, StringArray,
};

use super::map_rows;
use crate::ExprError;

/// The decoder for one image's header, or `None` when the bytes match no known container.
fn decoder(data: &[u8]) -> Option<impl image::ImageDecoder + '_> {
    image::ImageReader::new(Cursor::new(data))
        .with_guessed_format()
        .ok()?
        .into_decoder()
        .ok()
}

/// `aspect_ratio()` → width / height as Float64, from the header alone.
///
/// A zero-height image yields null rather than an infinity: the ratio of an image with no
/// rows is not a number a predicate should be asked to compare, and `inf` would silently
/// pass every `> threshold` filter written to find panoramas.
pub(super) fn aspect_ratio<O: OffsetSizeTrait>(
    bytes: &GenericBinaryArray<O>,
) -> Result<ArrayRef, ExprError> {
    let rows: Vec<Option<f64>> = map_rows(bytes.len(), |i| {
        if bytes.is_null(i) {
            return None;
        }
        let (w, h) = image::ImageDecoder::dimensions(&decoder(bytes.value(i))?);
        (h > 0).then(|| f64::from(w) / f64::from(h))
    });
    Ok(Arc::new(Float64Array::from(rows)))
}

/// `has_alpha()` → whether the colour type carries an alpha channel, from the header.
pub(super) fn has_alpha<O: OffsetSizeTrait>(
    bytes: &GenericBinaryArray<O>,
) -> Result<ArrayRef, ExprError> {
    let rows: Vec<Option<bool>> = map_rows(bytes.len(), |i| {
        if bytes.is_null(i) {
            return None;
        }
        Some(image::ImageDecoder::color_type(&decoder(bytes.value(i))?).has_alpha())
    });
    Ok(Arc::new(BooleanArray::from(rows)))
}

/// `format()` → the container's lowercase name, sniffed from the magic bytes.
///
/// From the bytes, never from the path: a corpus downloaded by content type is full of
/// files whose extension and container disagree, and every one of them is a row that
/// decodes fine and breaks whatever downstream step branched on the name.
///
/// The name is the **container's** name, not its file extension: `jpeg`, not `jpg`. Three
/// things in the engine answer this question about the same bytes — this expression, the
/// `format` column an image listing emits, and the `mime` subtype — and they have to agree
/// or a pipeline that switches between them silently selects different rows. It is also
/// the vocabulary `.image.encode(format)` accepts, which is what makes "read the container
/// off a column and hand it back to the encoder" work rather than fail on a name the
/// encoder has never heard of.
///
/// `image`'s own `extensions_str()` leads with `jpg`, so it is deliberately not used here.
pub(super) fn format<O: OffsetSizeTrait>(
    bytes: &GenericBinaryArray<O>,
) -> Result<ArrayRef, ExprError> {
    let rows: Vec<Option<&'static str>> = map_rows(bytes.len(), |i| {
        if bytes.is_null(i) {
            return None;
        }
        image::ImageReader::new(Cursor::new(bytes.value(i)))
            .with_guessed_format()
            .ok()?
            .format()
            .map(container_name)
    });
    Ok(Arc::new(StringArray::from(rows)))
}

/// The canonical container name for a decoded format.
///
/// Named explicitly rather than derived, so the four names this namespace can *write*
/// (`ENCODE_FORMATS`) are spelled here exactly as the encoder accepts them. Anything else
/// falls back to its leading extension, which is the best available name for a container
/// that can only be read.
fn container_name(fmt: image::ImageFormat) -> &'static str {
    match fmt {
        image::ImageFormat::Png => "png",
        image::ImageFormat::Jpeg => "jpeg",
        image::ImageFormat::Gif => "gif",
        image::ImageFormat::Bmp => "bmp",
        other => other.extensions_str().first().copied().unwrap_or("unknown"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{AsArray, BinaryArray};
    use arrow::datatypes::Float64Type;

    fn encoded(w: u32, h: u32, fmt: image::ImageFormat, alpha: bool) -> Vec<u8> {
        let img = if alpha {
            image::DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
                w,
                h,
                image::Rgba([1, 2, 3, 4]),
            ))
        } else {
            image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(w, h, image::Rgb([1, 2, 3])))
        };
        let mut out = std::io::Cursor::new(Vec::new());
        img.write_to(&mut out, fmt).unwrap();
        out.into_inner()
    }

    fn column(imgs: Vec<Vec<u8>>) -> BinaryArray {
        BinaryArray::from_iter(imgs.iter().map(|b| Some(b.as_slice())))
    }

    #[test]
    fn aspect_ratio_reads_the_header() {
        let arr = column(vec![
            encoded(40, 20, image::ImageFormat::Png, false),
            encoded(10, 40, image::ImageFormat::Png, false),
        ]);
        let out = aspect_ratio(&arr).unwrap();
        let a = out.as_primitive::<Float64Type>();
        assert_eq!(a.value(0), 2.0);
        assert_eq!(a.value(1), 0.25);
    }

    #[test]
    fn has_alpha_distinguishes_rgb_from_rgba() {
        let arr = column(vec![
            encoded(4, 4, image::ImageFormat::Png, true),
            encoded(4, 4, image::ImageFormat::Png, false),
        ]);
        let out = has_alpha(&arr).unwrap();
        let a = out.as_boolean();
        assert!(a.value(0));
        assert!(!a.value(1));
    }

    /// The reason it sniffs bytes rather than paths: the two disagree constantly.
    #[test]
    fn format_names_the_real_container() {
        let arr = column(vec![
            encoded(4, 4, image::ImageFormat::Png, false),
            encoded(4, 4, image::ImageFormat::Jpeg, false),
            encoded(4, 4, image::ImageFormat::Gif, false),
        ]);
        let out = format(&arr).unwrap();
        let a = out.as_string::<i32>();
        assert_eq!(a.value(0), "png");
        // `jpeg`, not `jpg`: the name has to be one `.image.encode(format)` accepts, and
        // the one the image listing's own `format` column reports for the same bytes.
        assert_eq!(a.value(1), "jpeg");
        assert_eq!(a.value(2), "gif");
    }

    #[test]
    fn a_null_or_unrecognized_row_is_null() {
        let arr = BinaryArray::from_iter(vec![None, Some(b"not an image".as_slice())]);
        assert_eq!(aspect_ratio(&arr).unwrap().null_count(), 2);
        assert_eq!(has_alpha(&arr).unwrap().null_count(), 2);
        assert_eq!(format(&arr).unwrap().null_count(), 2);
    }
}
