//! Image-decode evaluation for `Expr::Image` (the `.image` namespace).
//!
//! This is the interpreter *oracle* for image decoding. The JIT cannot compile
//! library-backed decode, so `bc-codegen` marks `Expr::Image` unsupported and
//! falls back here — the two never diverge because there is only this one
//! implementation. Decode runs per row over the whole batch; a row whose bytes
//! are null or fail to decode yields a null result (corrupt inputs don't fail
//! the batch), matching the multimodal source's header-metadata convention.

use std::io::Cursor;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BinaryArray, FixedSizeListArray, Int32Array, StructArray, UInt8Array,
};
use arrow::buffer::NullBuffer;
use arrow::datatypes::{DataType, Field};

use crate::{ExprError, ImageFunc};

/// Evaluate an image function over a Binary array of encoded image bytes.
pub(crate) fn eval_image(
    func: ImageFunc,
    arr: &ArrayRef,
    width: Option<i64>,
    height: Option<i64>,
) -> Result<ArrayRef, ExprError> {
    let bytes =
        arr.as_any()
            .downcast_ref::<BinaryArray>()
            .ok_or_else(|| ExprError::ExpectedBinary {
                func: format!("{func:?}"),
                got: arr.data_type().to_string(),
            })?;
    match func {
        ImageFunc::Decode => decode_dims(bytes),
        ImageFunc::ToTensor => to_tensor(bytes, width, height),
        ImageFunc::Resize => resize(bytes, width, height),
    }
}

/// `resize(w, h)` → re-encoded PNG bytes at the new size. Null/undecodable → null.
fn resize(
    bytes: &BinaryArray,
    width: Option<i64>,
    height: Option<i64>,
) -> Result<ArrayRef, ExprError> {
    let w = width.ok_or(ExprError::MissingImageArg {
        func: "resize".into(),
        arg: "width",
    })? as u32;
    let h = height.ok_or(ExprError::MissingImageArg {
        func: "resize".into(),
        arg: "height",
    })? as u32;
    let mut out: Vec<Option<Vec<u8>>> = Vec::with_capacity(bytes.len());
    for i in 0..bytes.len() {
        if bytes.is_null(i) {
            out.push(None);
        } else {
            out.push(resize_png(bytes.value(i), w, h));
        }
    }
    Ok(Arc::new(BinaryArray::from_iter(out)))
}

/// Decode, resize to `(w, h)`, and re-encode as PNG; `None` on any failure.
fn resize_png(data: &[u8], w: u32, h: u32) -> Option<Vec<u8>> {
    let img = image::load_from_memory(data).ok()?;
    let resized = img.resize_exact(w, h, image::imageops::FilterType::Triangle);
    let mut buf = Cursor::new(Vec::new());
    resized.write_to(&mut buf, image::ImageFormat::Png).ok()?;
    Some(buf.into_inner())
}

/// `decode` → struct `{width: Int32, height: Int32}` (header read only).
fn decode_dims(bytes: &BinaryArray) -> Result<ArrayRef, ExprError> {
    let mut widths: Vec<i32> = Vec::with_capacity(bytes.len());
    let mut heights: Vec<i32> = Vec::with_capacity(bytes.len());
    let mut valid: Vec<bool> = Vec::with_capacity(bytes.len());
    for i in 0..bytes.len() {
        if bytes.is_null(i) {
            widths.push(0);
            heights.push(0);
            valid.push(false);
            continue;
        }
        match image_dimensions(bytes.value(i)) {
            Some((w, h)) => {
                widths.push(w as i32);
                heights.push(h as i32);
                valid.push(true);
            }
            None => {
                widths.push(0);
                heights.push(0);
                valid.push(false);
            }
        }
    }
    let nulls = NullBuffer::from(valid);
    let fields = vec![
        Arc::new(Field::new("width", DataType::Int32, false)),
        Arc::new(Field::new("height", DataType::Int32, false)),
    ];
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from(widths)),
        Arc::new(Int32Array::from(heights)),
    ];
    let struct_arr = StructArray::new(fields.into(), columns, Some(nulls));
    Ok(Arc::new(struct_arr))
}

/// `to_tensor(w, h)` → `FixedSizeList<UInt8>` of length `w*h*3` (RGB8, resized).
fn to_tensor(
    bytes: &BinaryArray,
    width: Option<i64>,
    height: Option<i64>,
) -> Result<ArrayRef, ExprError> {
    let w = width.ok_or(ExprError::MissingImageArg {
        func: "to_tensor".into(),
        arg: "width",
    })? as u32;
    let h = height.ok_or(ExprError::MissingImageArg {
        func: "to_tensor".into(),
        arg: "height",
    })? as u32;
    let per_row = (w as usize) * (h as usize) * 3;
    let mut values: Vec<u8> = Vec::with_capacity(bytes.len() * per_row);
    let mut valid: Vec<bool> = Vec::with_capacity(bytes.len());
    for i in 0..bytes.len() {
        let pixels: Option<Vec<u8>> = if bytes.is_null(i) {
            None
        } else {
            decode_rgb_resized(bytes.value(i), w, h)
        };
        match pixels {
            Some(buf) if buf.len() == per_row => {
                values.extend_from_slice(&buf);
                valid.push(true);
            }
            _ => {
                values.resize(values.len() + per_row, 0);
                valid.push(false);
            }
        }
    }
    let field = Arc::new(Field::new("item", DataType::UInt8, false));
    let arr = FixedSizeListArray::new(
        field,
        per_row as i32,
        Arc::new(UInt8Array::from(values)),
        Some(NullBuffer::from(valid)),
    );
    Ok(Arc::new(arr))
}

/// Read just the image header to get `(width, height)`; `None` on any failure.
fn image_dimensions(data: &[u8]) -> Option<(u32, u32)> {
    image::ImageReader::new(Cursor::new(data))
        .with_guessed_format()
        .ok()?
        .into_dimensions()
        .ok()
}

/// Decode, resize to `(w, h)`, and flatten to RGB8; `None` on any failure.
fn decode_rgb_resized(data: &[u8], w: u32, h: u32) -> Option<Vec<u8>> {
    let img = image::load_from_memory(data).ok()?;
    let resized = img.resize_exact(w, h, image::imageops::FilterType::Triangle);
    Some(resized.to_rgb8().into_raw())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 2×3 red PNG, encoded once so the test has no I/O.
    fn red_png(width: u32, height: u32) -> Vec<u8> {
        let buf = image::RgbImage::from_pixel(width, height, image::Rgb([255, 0, 0]));
        let mut out = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(buf)
            .write_to(&mut out, image::ImageFormat::Png)
            .unwrap();
        out.into_inner()
    }

    #[test]
    fn decode_reads_dimensions() {
        let arr: ArrayRef = Arc::new(BinaryArray::from(vec![
            Some(red_png(2, 3).as_slice()),
            None,
            Some(b"not an image".as_slice()),
        ]));
        let out = eval_image(ImageFunc::Decode, &arr, None, None).unwrap();
        let s = out.as_any().downcast_ref::<StructArray>().unwrap();
        let w = s.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let h = s.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
        assert!(s.is_valid(0) && w.value(0) == 2 && h.value(0) == 3);
        assert!(s.is_null(1)); // null bytes → null struct
        assert!(s.is_null(2)); // undecodable bytes → null struct
    }

    #[test]
    fn to_tensor_decodes_and_resizes() {
        let arr: ArrayRef = Arc::new(BinaryArray::from(vec![
            Some(red_png(8, 8).as_slice()),
            None,
        ]));
        let out = eval_image(ImageFunc::ToTensor, &arr, Some(4), Some(4)).unwrap();
        let fsl = out.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
        assert_eq!(fsl.value_length(), 4 * 4 * 3);
        assert!(fsl.is_valid(0));
        assert!(fsl.is_null(1));
        let row0 = fsl.value(0);
        let px = row0.as_any().downcast_ref::<UInt8Array>().unwrap();
        // Resized solid-red image stays red: first pixel ~ (255, 0, 0).
        assert_eq!(px.value(0), 255);
        assert_eq!(px.value(1), 0);
        assert_eq!(px.value(2), 0);
    }

    #[test]
    fn resize_reencodes_at_new_size() {
        let arr: ArrayRef = Arc::new(BinaryArray::from(vec![
            Some(red_png(8, 8).as_slice()),
            None,
            Some(b"not an image".as_slice()),
        ]));
        let out = eval_image(ImageFunc::Resize, &arr, Some(4), Some(2)).unwrap();
        let b = out.as_any().downcast_ref::<BinaryArray>().unwrap();
        assert!(b.is_valid(0));
        // The re-encoded PNG decodes back to the requested 4×2 dimensions.
        assert_eq!(image_dimensions(b.value(0)), Some((4, 2)));
        assert!(b.is_null(1)); // null input → null
        assert!(b.is_null(2)); // undecodable input → null
    }
}
